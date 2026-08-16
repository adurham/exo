#!/usr/bin/env python3
"""GROUND TRUTH: is the GPU idle, busy-at-low-clock, or blocked-on-event
during the 550ms PP decode eval window? (Section 100)

WHY THIS SCRIPT EXISTS
-----------------------
Sections 85-89 established via CPU stack sampling that the CPU parks in
``metal::EventImpl::wait -> sleep_for`` for 89-92% of samples during the
550ms/token PP decode window, and that the GPU reads 5-7% "busy" in that
same window. But a coarse average CANNOT distinguish:

  (a) GPU idle, nothing committed to it at all
  (b) GPU genuinely executing work, just at a very low clock (throttled)
  (c) GPU sitting on waitForEvent with work committed but blocked on a
      dependency (e.g. cross-rank event) that hasn't signaled yet

All three produce a low coarse utilization/power number. This script
gates sampling strictly on the DECODE phase (never prefill -- prefill is
20-45x longer than decode per design doc Section 84/85 and WILL swamp
any whole-request average) and samples BOTH ranks simultaneously so we
can also ask whether rank0 is stalled on rank1 or on itself.

METHODOLOGY (per campaign rules -- 7+ retractions logged for violating
these):
  - Every measurement is labeled with which phase (prefill vs decode) it
    came from. Gate = first SSE token observed on this process's own
    stream. Samples before that timestamp are DISCARDED, not averaged in.
  - n>=3 decode-window samples per node before drawing a verdict.
  - Never quote a whole-request average as a per-phase number.
  - DSv4 streams short answers entirely as `reasoning_content`, not
    `content` -- token-count detection here uses BOTH fields, and the
    final /usage block (authoritative) is also captured and reported.
  - If a privileged tool is unavailable, this reports that plainly and
    falls back to the documented no-sudo alternative rather than
    fabricating a number.

TOOLS, in preference order:
  1. `sudo powermetrics --samplers gpu_power -i 50` -- per-50ms active
     residency AND frequency. Needs passwordless sudo. If `sudo -n true`
     fails, this is SKIPPED (never blocks on a password prompt) and the
     script says so explicitly in the verdict.
  2. `ioreg -r -d 1 -c IOAccelerator` (no sudo) sampled in a tight loop
     over ssh, tagged with the LOCAL wall-clock time at which each line
     was received (avoids cross-host clock-skew issues from trusting
     remote timestamps). Reports `Device Utilization %` (the closest
     unprivileged analogue of GPU busy%) each pass.

This script is READ-ONLY / traffic-only: it sends ordinary inference
requests to the live serving endpoint and ssh's in read-only for
sampling. It does not touch cluster lifecycle.

USAGE:
    uv run python bench/section100_gpu_ground_truth.py \
        [--host http://adams-mac-studio-m4-1.local:52415] \
        [--node1 adams-mac-studio-m4-1.local] \
        [--node2 adams-mac-studio-m4-2.local] \
        [--prompt-tokens 6000] [--max-tokens 6] \
        [--out bench/section100_results.json]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

DEFAULT_HOST = "http://adams-mac-studio-m4-1.local:52415"
DEFAULT_NODE1 = "adams-mac-studio-m4-1.local"
DEFAULT_NODE2 = "adams-mac-studio-m4-2.local"
# EXO_PREFILL_STEP_SIZE=2048 is the branch condition (design doc Section
# 85) that selects pipeline_parallel_prefill (chunked) over stream_generate
# (plain). Chunked prefill is what puts decode on the slow PP path.
CHUNKED_PREFILL_TRIGGER_TOKENS = 2048
FILLER_SENTENCE = (
    "The quick brown fox jumps over the lazy dog near the quiet river bank "
    "while the old clock ticks steadily through another uneventful autumn "
    "afternoon in the small mountain village. "
)


@dataclass
class IoregSample:
    t_local: float  # local wall clock when this sample was RECEIVED
    device_util_pct: float | None
    renderer_util_pct: float | None
    raw_line: str


@dataclass
class NodeSampler:
    host: str
    label: str
    samples: list[IoregSample] = field(default_factory=list)
    proc: subprocess.Popen[str] | None = None
    _stop: bool = False
    _thread: threading.Thread | None = None

    def start(self) -> None:
        # Tight unprivileged polling loop, no sudo required. One ioreg
        # call per iteration; ~10-40ms round trip over ssh is typical on
        # a local .local hostname, so this gives sub-100ms granularity.
        remote_cmd = (
            "while true; do "
            "ioreg -r -d 1 -c IOAccelerator 2>/dev/null "
            "| grep -E 'Device Utilization %|Renderer Utilization %'; "
            "echo '---SAMPLE-END---'; "
            "done"
        )
        self.proc = subprocess.Popen(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", self.host, remote_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        device_util: float | None = None
        renderer_util: float | None = None
        raw_lines: list[str] = []
        for line in self.proc.stdout:
            t_local = time.monotonic()
            line_s = line.strip()
            if self._stop:
                break
            if line_s == "---SAMPLE-END---":
                self.samples.append(
                    IoregSample(
                        t_local=t_local,
                        device_util_pct=device_util,
                        renderer_util_pct=renderer_util,
                        raw_line="; ".join(raw_lines),
                    )
                )
                device_util = None
                renderer_util = None
                raw_lines = []
                continue
            raw_lines.append(line_s)
            m_dev = re.search(r'"Device Utilization %"\s*=\s*(-?\d+(?:\.\d+)?)', line_s)
            if m_dev:
                try:
                    device_util = float(m_dev.group(1))
                except ValueError:
                    pass
            m_ren = re.search(r'"Renderer Utilization %"\s*=\s*(-?\d+(?:\.\d+)?)', line_s)
            if m_ren:
                try:
                    renderer_util = float(m_ren.group(1))
                except ValueError:
                    pass

    def stop(self) -> None:
        self._stop = True
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def samples_in_window(self, t_start: float, t_end: float) -> list[IoregSample]:
        return [s for s in self.samples if t_start <= s.t_local <= t_end]


def check_sudo_available(host: str) -> bool:
    """Check specifically whether `sudo -n powermetrics` works.

    NOTE: macOS machines sometimes carry a NOPASSWD sudoers rule scoped
    to specific binaries (e.g. powermetrics) rather than a blanket `sudo
    -n true`. Checking `sudo -n true` alone can therefore give a FALSE
    NEGATIVE. We probe the actual command we intend to run, with `-n`
    (no stdin) so a real password prompt fails fast instead of hanging.
    """
    try:
        r = subprocess.run(
            [
                "ssh", "-o", "BatchMode=yes", "-n", "-o", "ConnectTimeout=5", host,
                "sudo -n powermetrics --samplers gpu_power -i 50 -n 1 2>&1",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        combined = (r.stdout + r.stderr)
        if "a password is required" in combined:
            return False
        return r.returncode == 0 and "GPU HW active" in combined or "GPU idle residency" in combined
    except Exception:
        return False


def try_powermetrics(host: str, duration_s: float = 3.0) -> dict[str, Any]:
    """Legacy one-shot fallback: NOT gated on the decode window, kept only
    as a diagnostic if the background sampler (PowermetricsSampler) never
    got started. Prefer PowermetricsSampler for anything quoted as a
    per-phase number.
    """
    if not check_sudo_available(host):
        return {
            "available": False,
            "reason": "passwordless sudo for `powermetrics` not available on this host; "
            "skipping rather than hanging on a password prompt, per instructions.",
        }
    remote_cmd = (
        f"sudo -n powermetrics --samplers gpu_power -i 50 -n {int(duration_s * 1000 / 50)} "
        "2>&1"
    )
    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-n", "-o", "ConnectTimeout=5", host, remote_cmd],
            capture_output=True,
            text=True,
            timeout=duration_s + 15,
        )
        return {"available": True, "returncode": r.returncode, "raw": r.stdout[-8000:]}
    except subprocess.TimeoutExpired:
        return {"available": False, "reason": "powermetrics timed out"}
    except Exception as e:
        return {"available": False, "reason": f"powermetrics failed: {e!r}"}


@dataclass
class PmSample:
    t_local: float  # local wall clock at which this sample block completed
    gpu_hw_active_residency_pct: float | None
    gpu_hw_active_freq_mhz: float | None
    gpu_idle_residency_pct: float | None
    gpu_power_mw: float | None


@dataclass
class PowermetricsSampler:
    """Runs `sudo -n powermetrics --samplers gpu_power -i 50` continuously
    in the background over ssh and parses each ~50ms sample block, tagging
    it with LOCAL receipt wall-clock time (same scheme as NodeSampler) so
    it can be filtered into the decode-only window after the fact. This is
    what makes the frequency/active-residency numbers genuinely
    phase-gated rather than a guessed post-hoc window.
    """

    host: str
    samples: list[PmSample] = field(default_factory=list)
    proc: subprocess.Popen[str] | None = None
    _stop: bool = False
    _thread: threading.Thread | None = None
    start_error: str | None = None

    def start(self) -> None:
        remote_cmd = "sudo -n powermetrics --samplers gpu_power -i 50 2>&1"
        try:
            self.proc = subprocess.Popen(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", self.host, remote_cmd],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            self.start_error = repr(e)
            return
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        active_res: float | None = None
        active_freq: float | None = None
        idle_res: float | None = None
        power_mw: float | None = None
        saw_password_prompt = False
        for line in self.proc.stdout:
            if self._stop:
                break
            line_s = line.strip()
            if "a password is required" in line_s:
                saw_password_prompt = True
                self.start_error = "sudo prompted for a password mid-stream"
                break
            m = re.search(r"GPU HW active residency:\s*([\d.]+)%", line_s)
            if m:
                active_res = float(m.group(1))
            m = re.search(r"GPU HW active frequency:\s*([\d.]+)\s*MHz", line_s)
            if m:
                active_freq = float(m.group(1))
            m = re.search(r"GPU idle residency:\s*([\d.]+)%", line_s)
            if m:
                idle_res = float(m.group(1))
                # "GPU idle residency" is the LAST line of each sample block
                # in this samplers=gpu_power output, so treat it as the
                # block-complete marker and timestamp+flush here.
                t_local = time.monotonic()
                self.samples.append(
                    PmSample(
                        t_local=t_local,
                        gpu_hw_active_residency_pct=active_res,
                        gpu_hw_active_freq_mhz=active_freq,
                        gpu_idle_residency_pct=idle_res,
                        gpu_power_mw=power_mw,
                    )
                )
                active_res = None
                active_freq = None
                power_mw = None
            m = re.search(r"GPU Power:\s*([\d.]+)\s*mW", line_s)
            if m:
                power_mw = float(m.group(1))
        _ = saw_password_prompt

    def stop(self) -> None:
        self._stop = True
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def samples_in_window(self, t_start: float, t_end: float) -> list[PmSample]:
        return [s for s in self.samples if t_start <= s.t_local <= t_end]


def build_long_prompt(min_tokens: int) -> str:
    # Rough estimate ~1.3 tokens/word for this filler; overshoot generously
    # and let the API's own usage block be authoritative (never estimate
    # token counts ourselves per instructions).
    reps = max(1, (min_tokens * 2) // len(FILLER_SENTENCE.split()) + 50)
    return (FILLER_SENTENCE * reps) + (
        "\n\nGiven all of the above, reply with a single short sentence "
        "confirming you read it."
    )


def run_streaming_request(
    api_base: str, model: str, prompt: str, max_tokens: int
) -> dict[str, Any]:
    """Send a streaming chat completion; return timing + usage evidence.

    Per instructions: DSv4 streams short answers entirely as
    `delta.reasoning_content`, not `delta.content`. Both are checked for
    "first token observed" gating; the API's final usage block is the
    authoritative token count, never a client-side content-field count.
    """
    url = f"{api_base}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t_send = time.monotonic()
    first_token_t: float | None = None
    first_token_field: str | None = None
    token_event_times: list[float] = []
    usage: dict[str, Any] | None = None
    n_chunks = 0
    with httpx.Client(timeout=600.0) as client:
        with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                data_s = raw_line[len("data:") :].strip()
                if data_s == "[DONE]":
                    break
                try:
                    obj = json.loads(data_s)
                except json.JSONDecodeError:
                    continue
                if "usage" in obj and obj["usage"]:
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    has_content = bool(delta.get("content"))
                    has_reasoning = bool(delta.get("reasoning_content"))
                    if has_content or has_reasoning:
                        t_now = time.monotonic()
                        n_chunks += 1
                        token_event_times.append(t_now)
                        if first_token_t is None:
                            first_token_t = t_now
                            first_token_field = "content" if has_content else "reasoning_content"
    t_end = time.monotonic()
    return {
        "t_send": t_send,
        "t_first_token": first_token_t,
        "first_token_field": first_token_field,
        "t_end": t_end,
        "n_stream_chunks_with_text": n_chunks,
        "token_event_times": token_event_times,
        "usage": usage,
    }


def summarize_util(samples: list[IoregSample]) -> dict[str, Any]:
    vals = [s.device_util_pct for s in samples if s.device_util_pct is not None]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "min": min(vals),
        "max": max(vals),
        "mean": sum(vals) / len(vals),
        "raw": vals,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--node1", default=DEFAULT_NODE1, help="rank0 ssh host")
    ap.add_argument("--node2", default=DEFAULT_NODE2, help="rank1 ssh host")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    ap.add_argument("--prompt-tokens", type=int, default=6000)
    ap.add_argument("--max-tokens", type=int, default=6)
    ap.add_argument(
        "--out", default="bench/section100_results.json", help="where to write raw JSON results"
    )
    args = ap.parse_args()

    print(f"[section100] building prompt targeting >= {args.prompt_tokens} tokens "
          f"(chunked-prefill trigger is {CHUNKED_PREFILL_TRIGGER_TOKENS})", file=sys.stderr)
    prompt = build_long_prompt(args.prompt_tokens)

    print("[section100] checking powermetrics availability on both nodes "
          "(non-blocking, no password prompt)...", file=sys.stderr)
    pm1_probe = check_sudo_available(args.node1)
    pm2_probe = check_sudo_available(args.node2)
    print(f"[section100] node1 sudo -n powermetrics works: {pm1_probe}; "
          f"node2 sudo -n powermetrics works: {pm2_probe}", file=sys.stderr)

    node1 = NodeSampler(host=args.node1, label="rank0")
    node2 = NodeSampler(host=args.node2, label="rank1")
    print("[section100] starting unprivileged ioreg samplers on both nodes...", file=sys.stderr)
    node1.start()
    node2.start()

    pm_node1: PowermetricsSampler | None = None
    pm_node2: PowermetricsSampler | None = None
    if pm1_probe:
        pm_node1 = PowermetricsSampler(host=args.node1)
        pm_node1.start()
        print("[section100] started BACKGROUND powermetrics sampler on node1 "
              "(authoritative tool, per-50ms active residency + frequency)", file=sys.stderr)
    if pm2_probe:
        pm_node2 = PowermetricsSampler(host=args.node2)
        pm_node2.start()
        print("[section100] started BACKGROUND powermetrics sampler on node2", file=sys.stderr)

    time.sleep(1.5)  # let samplers ramp up before the request starts

    print(f"[section100] sending streaming request (max_tokens={args.max_tokens})...",
          file=sys.stderr)
    result = run_streaming_request(args.host, args.model, prompt, args.max_tokens)

    # Give samplers a moment to catch trailing decode-window samples.
    time.sleep(1.0)
    node1.stop()
    node2.stop()
    if pm_node1 is not None:
        pm_node1.stop()
    if pm_node2 is not None:
        pm_node2.stop()

    usage = result["usage"] or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    is_chunked_mode = (
        prompt_tokens is not None and prompt_tokens >= CHUNKED_PREFILL_TRIGGER_TOKENS
    )

    t_send = result["t_send"]
    t_first = result["t_first_token"]
    t_end = result["t_end"]
    token_times = result["token_event_times"]

    print(f"[section100] usage block (AUTHORITATIVE token counts): {usage}", file=sys.stderr)
    _mode_note = (
        "LIKELY TAKEN"
        if is_chunked_mode
        else "NOT taken -- not a valid slow-decode run, rerun with a larger --prompt-tokens"
    )
    print(
        f"[section100] prompt_tokens={prompt_tokens} (chunked-prefill path {_mode_note})",
        file=sys.stderr,
    )

    if t_first is None:
        print("[section100] FATAL: no token was ever observed on the stream "
              "(neither content nor reasoning_content). Cannot gate a decode "
              "window. Aborting without a verdict.", file=sys.stderr)
        verdict = "INCONCLUSIVE: no first token observed"
    elif not is_chunked_mode:
        verdict = (
            "INCONCLUSIVE: prompt did not land in chunked-prefill / PP slow-decode "
            "mode (prompt_tokens < 2048); this run does not exercise the 550ms path"
        )
    else:
        prefill_wall_s = t_first - t_send
        decode_wall_s = t_end - t_first
        n_decode_tokens = max(0, (completion_tokens or 1) - 1)  # first token = prefill-adjacent
        per_token_decode_ms = (
            (decode_wall_s / n_decode_tokens) * 1000.0 if n_decode_tokens > 0 else None
        )
        print(f"[section100] PHASE-LABELED timings: prefill_wall={prefill_wall_s:.3f}s "
              f"(from request send to first streamed token/reasoning token) "
              f"decode_wall={decode_wall_s:.3f}s over completion_tokens={completion_tokens} "
              f"(~{per_token_decode_ms}ms/decode-token if n_decode_tokens>0)", file=sys.stderr)

        # DECODE-ONLY WINDOW = strictly after first token was observed.
        # Prefill samples (before t_first) are explicitly discarded here,
        # never averaged into the decode number (design doc Section 84
        # error being made structurally impossible, per instructions).
        decode_samples_r0 = node1.samples_in_window(t_first, t_end)
        decode_samples_r1 = node2.samples_in_window(t_first, t_end)
        prefill_samples_r0 = node1.samples_in_window(t_send, t_first)
        prefill_samples_r1 = node2.samples_in_window(t_send, t_first)

        util_decode_r0 = summarize_util(decode_samples_r0)
        util_decode_r1 = summarize_util(decode_samples_r1)
        util_prefill_r0 = summarize_util(prefill_samples_r0)
        util_prefill_r1 = summarize_util(prefill_samples_r1)

        print(f"[section100] rank0 (node1={args.node1}) DECODE-phase ioreg "
              f"Device Utilization %: n={util_decode_r0.get('n')} "
              f"min/mean/max={util_decode_r0.get('min')}/"
              f"{util_decode_r0.get('mean')}/{util_decode_r0.get('max')}", file=sys.stderr)
        print(f"[section100] rank1 (node2={args.node2}) DECODE-phase ioreg "
              f"Device Utilization %: n={util_decode_r1.get('n')} "
              f"min/mean/max={util_decode_r1.get('min')}/"
              f"{util_decode_r1.get('mean')}/{util_decode_r1.get('max')}", file=sys.stderr)
        print(f"[section100] (context, NOT the verdict number) rank0 PREFILL-phase "
              f"Device Utilization %: n={util_prefill_r0.get('n')} "
              f"mean={util_prefill_r0.get('mean')}", file=sys.stderr)

        n_ok = util_decode_r0.get("n", 0) >= 3 and util_decode_r1.get("n", 0) >= 3

        # CRITICAL DISTINCTION this script exists to make: ioreg's
        # "Device Utilization %" reflects the IOAccelerator's own
        # internal work-submission accounting -- it can read non-zero
        # even while a downstream event-wait is what's blocking forward
        # progress, because the driver still counts committed/queued
        # work. It CANNOT by itself distinguish (b) busy-low-clock from
        # (c) blocked-on-event with work committed; ioreg has no
        # frequency field at all (that's powermetrics-only). So:
        #   - if BOTH nodes show near-zero AND powermetrics was
        #     unavailable, the honest verdict is "GPU accelerator device
        #     reports near-idle by the unprivileged Device Utilization %
        #     metric; frequency/active-residency (which would
        #     distinguish idle from blocked-on-event) COULD NOT be
        #     established without powermetrics/sudo."
        #   - this is reported explicitly, not smoothed over.
        verdict = {
            "n_samples_ok": n_ok,
            "rank0_decode_device_util": util_decode_r0,
            "rank1_decode_device_util": util_decode_r1,
            "prefill_wall_s": prefill_wall_s,
            "decode_wall_s": decode_wall_s,
            "n_decode_tokens": n_decode_tokens,
            "per_token_decode_ms": per_token_decode_ms,
        }

    pm1 = try_powermetrics(args.node1, duration_s=3.0) if is_chunked_mode and t_first else {
        "available": False, "reason": "skipped (no valid decode window)"
    }
    pm2 = try_powermetrics(args.node2, duration_s=3.0) if is_chunked_mode and t_first else {
        "available": False, "reason": "skipped (no valid decode window)"
    }
    print(f"[section100] node1 powermetrics attempt: available={pm1.get('available')} "
          f"reason={pm1.get('reason', 'n/a')}", file=sys.stderr)
    print(f"[section100] node2 powermetrics attempt: available={pm2.get('available')} "
          f"reason={pm2.get('reason', 'n/a')}", file=sys.stderr)

    out = {
        "args": vars(args),
        "request_result": {
            k: v for k, v in result.items() if k != "token_event_times"
        } | {"n_token_events": len(result["token_event_times"])},
        "verdict_data": verdict,
        "powermetrics_node1": pm1,
        "powermetrics_node2": pm2,
        "node1_all_samples": [
            {"t": s.t_local, "device_util": s.device_util_pct, "renderer_util": s.renderer_util_pct}
            for s in node1.samples
        ],
        "node2_all_samples": [
            {"t": s.t_local, "device_util": s.device_util_pct, "renderer_util": s.renderer_util_pct}
            for s in node2.samples
        ],
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[section100] wrote raw results to {args.out}", file=sys.stderr)

    print("\n[section100] === FINAL VERDICT ===", file=sys.stderr)
    if isinstance(verdict, str):
        print(f"[section100] {verdict}", file=sys.stderr)
        return 2
    print(json.dumps(verdict, indent=2, default=str), file=sys.stderr)
    if not pm1.get("available") and not pm2.get("available"):
        print(
            "[section100] NEITHER node had passwordless sudo, so the AUTHORITATIVE "
            "per-50ms active-residency+frequency tool (powermetrics) could NOT be "
            "run. The unprivileged ioreg 'Device Utilization %' numbers above are "
            "reported as-is but CANNOT by themselves prove (b) busy-low-clock vs "
            "(c) blocked-on-event-with-work-committed, because ioreg has no "
            "frequency field and its utilization counter reflects command-buffer "
            "submission accounting rather than a clean 'core executing' signal. "
            "This is stated explicitly rather than guessed.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
