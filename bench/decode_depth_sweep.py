#!/usr/bin/env python3
"""Decode-cost-vs-context-depth sweep for the PP DSv4 cluster.

WHY THIS EXISTS (design doc Section 85):

Sections 50-84 treated decode throughput as BIMODAL -- ~0.6 tok/s
"slow mode" vs ~22 tok/s "fast mode" -- and hunted for a per-session
state that selects between them. Reading the live log showed the two
modes were simply two different CONTEXT DEPTHS measured in the same
runner process with no restart between them:

    14,273-token prompt -> last_layer_eval 549-554ms
        47-token prompt -> last_layer_eval  15.3-16.5ms

This script exists to settle that properly rather than by anecdote: it
sweeps context depth on ONE live build and reports per-token decode
cost at each depth, so the question "is this bimodal, or is it a
monotonic function of depth?" is answered by a curve instead of by two
runs that happened to sit at opposite ends of it.

METHODOLOGY GUARDRAILS (this campaign has logged 7 retractions from
violating exactly these -- they are structural here, not reminders):

  * ONE BUILD. Every point comes from the same live cluster process.
    This script cannot and must not be used to compare two builds.
  * Every measurement is DEPTH-LABELED. There is no unlabeled number
    in the output.
  * n >= 3 repetitions per depth, and the report shows every sample
    plus the median -- never a lone number.
  * Prompt token counts are GROUND TRUTH read back from the runner's
    own log, never estimated from the text this script generated.
  * Decode cost is read from the per-token [LAYER_PHASE] lines, which
    are emitted per decode step, so the measurement window is gated to
    DECODE by construction. A whole-request average cannot support a
    per-phase claim (the Section 84 error).
  * Each prompt carries a unique NONCE AT THE FRONT so the KV prefix
    cache cannot serve it. A cache hit would silently turn a deep-
    context measurement into a shallow one.

It also optionally captures native stacks from BOTH ranks during the
decode phase (macOS /usr/bin/sample, no install required), which is
what actually names the mechanism behind a GPU-idle wall-clock wait.
Sampling is triggered off the first streamed token so it can never
land in prefill.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass, field

NODE1 = "adams-mac-studio-m4-1.local"
NODE2 = "adams-mac-studio-m4-2.local"
BASE_URL = f"http://{NODE1}:52415"
MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
REMOTE_LOG = "~/.exo/exo_log/exo.log"

# A needle the model must reproduce, so every depth is quality-gated and
# a "fast" number from a stream that never really ran is caught.
NEEDLE_KEY = "AURORA-VECTOR"


def _filler(paragraphs: int) -> str:
    """Deterministic, non-repetitive-enough filler.

    Deliberately NOT a single repeated sentence: a degenerate prompt can
    exercise different attention/routing behaviour than real text, and
    this sweep's whole point is to characterise the real decode path.
    """
    topics = [
        "distributed inference schedulers allocate pipeline stages across nodes",
        "thunderbolt fabrics expose queue pairs with distinct completion semantics",
        "mixture of experts routing selects a sparse subset of feedforward blocks",
        "key value caches grow linearly with sequence length during autoregressive decode",
        "quantised weight formats trade numerical precision for memory bandwidth",
        "speculative decoding drafts several tokens before a single verification pass",
        "unified memory architectures remove explicit host to device staging copies",
        "attention indexers rank historical positions before gathering a top subset",
    ]
    out: list[str] = []
    for i in range(paragraphs):
        t = topics[i % len(topics)]
        out.append(
            f"Section {i}. In practice {t}; the resulting behaviour depends on "
            f"configuration {i * 7 % 97} and on the observed interaction between "
            f"stage {i % 11} and stage {(i * 3) % 13} of the surrounding system."
        )
    return " ".join(out)


def build_prompt(target_tokens: int) -> str:
    """Build a prompt of roughly ``target_tokens`` tokens.

    The count is APPROXIMATE by construction -- the authoritative number
    is parsed back out of the runner log afterwards. ~0.75 words/token is
    a deliberate under-estimate so we overshoot slightly rather than
    landing short of the depth we meant to test.
    """
    nonce = uuid.uuid4().hex
    header = (
        f"Reference identifier {nonce}. "
        f"Remember this key exactly: {NEEDLE_KEY}. "
    )
    # ~13 words per filler sentence, ~1.35 tokens/word.
    approx_tokens_per_paragraph = 30
    paragraphs = max(1, target_tokens // approx_tokens_per_paragraph)
    body = _filler(paragraphs)
    tail = f" Question: repeat the key exactly as given. Answer with the key only."
    return header + body + tail


@dataclass
class RunResult:
    depth_label: int
    prompt_tokens: int | None = None
    prefill_path: str | None = None
    prefill_ms: float | None = None
    ttft_s: float = 0.0
    decode_tokens: int = 0
    decode_wall_s: float = 0.0
    needle_ok: bool = False
    text: str = ""
    eval_ms: list[float] = field(default_factory=list)
    usage_prompt_tokens: int | None = None
    usage_completion_tokens: int | None = None
    error: str | None = None

    @property
    def decode_tok_s(self) -> float:
        return self.decode_tokens / self.decode_wall_s if self.decode_wall_s > 0 else 0.0

    @property
    def eval_median(self) -> float | None:
        return statistics.median(self.eval_ms) if self.eval_ms else None


def remote_log_line_count() -> int:
    out = subprocess.run(
        ["ssh", NODE1, f"wc -l < {REMOTE_LOG}"],
        capture_output=True, text=True, timeout=60,
    )
    return int(out.stdout.strip() or 0)


def remote_log_since(start_line: int) -> str:
    out = subprocess.run(
        ["ssh", NODE1, f"tail -n +{start_line} {REMOTE_LOG}"],
        capture_output=True, text=True, timeout=180,
    )
    return out.stdout


_EVAL_RE = re.compile(r"last_layer_eval=([\d.]+)ms")
_SEND_RE = re.compile(r"last_layer_send=([\d.]+)ms")
_CHUNKED_RE = re.compile(r"Chunked prefill complete: (\d+) tokens in ([\d.]+)s")
_PLAIN_RE = re.compile(r"Prefill complete: (\d+) tokens in ([\d.]+)s")


def parse_window(text: str) -> tuple[int | None, str | None, float | None, list[float], list[float]]:
    prompt_tokens: int | None = None
    prefill_path: str | None = None
    prefill_ms: float | None = None
    m = _CHUNKED_RE.search(text)
    if m:
        prompt_tokens = int(m.group(1))
        prefill_path = "chunked"
        prefill_ms = float(m.group(2)) * 1000.0
    else:
        m = _PLAIN_RE.search(text)
        if m:
            prompt_tokens = int(m.group(1))
            prefill_path = "plain"
            prefill_ms = float(m.group(2)) * 1000.0
    evals = [float(x) for x in _EVAL_RE.findall(text)]
    sends = [float(x) for x in _SEND_RE.findall(text)]
    return prompt_tokens, prefill_path, prefill_ms, evals, sends


def capture_stacks(tag: str, duration_s: int) -> None:
    """Sample native stacks on BOTH ranks concurrently.

    Uses macOS's built-in /usr/bin/sample -- nothing is installed on the
    user's machines. The runner is identified by largest-RSS python under
    the exo venv, because the supervisor and the resource tracker share
    the same executable path.
    """
    pick_pid = (
        "ps aux | grep 'repos/exo/.venv/bin/python' | grep -v grep | "
        "sort -k6 -n -r | head -1 | awk '{print $2}'"
    )
    def go(host: str) -> None:
        cmd = (
            f"PID=$({pick_pid}); "
            f"echo \"host={host} pid=$PID\"; "
            f"/usr/bin/sample $PID {duration_s} -f /tmp/stack_{tag}_{host}.txt >/dev/null 2>&1; "
            f"echo done"
        )
        subprocess.run(["ssh", host, cmd], capture_output=True, text=True,
                       timeout=duration_s + 120)
    threads = [threading.Thread(target=go, args=(h,)) for h in (NODE1, NODE2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def run_one(depth: int, max_tokens: int, sample_stacks: bool,
            sample_seconds: int, tag: str) -> RunResult:
    res = RunResult(depth_label=depth)
    prompt = build_prompt(depth)
    start_line = remote_log_line_count() + 1

    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.time()
    first_tok_t: float | None = None
    last_tok_t: float | None = None
    ntok = 0
    chunks: list[str] = []
    sampler: threading.Thread | None = None

    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                # The API's own usage block is the authoritative token
                # accounting -- prefer it over anything this script can
                # infer from the text it generated.
                usage = obj.get("usage")
                if isinstance(usage, dict):
                    res.usage_prompt_tokens = usage.get("prompt_tokens")
                    res.usage_completion_tokens = usage.get("completion_tokens")
                delta = obj.get("choices", [{}])[0].get("delta", {})
                # DSv4-Flash is a thinking model: with a short max_tokens
                # the whole response arrives as `reasoning_content` and
                # `content` is never populated. Counting only `content`
                # silently reports zero decode tokens for a stream that
                # genuinely decoded -- count both.
                piece = (delta.get("content") or "") + (
                    delta.get("reasoning_content") or ""
                )
                if not piece:
                    continue
                now = time.time()
                if first_tok_t is None:
                    first_tok_t = now
                    # Gate sampling on the FIRST STREAMED TOKEN so the
                    # window can never include prefill (the Section 84
                    # mistake, made structurally impossible here).
                    if sample_stacks:
                        sampler = threading.Thread(
                            target=capture_stacks, args=(tag, sample_seconds)
                        )
                        sampler.start()
                last_tok_t = now
                ntok += 1
                chunks.append(piece)
    except Exception as exc:  # noqa: BLE001 - surfaced in the report
        res.error = f"{type(exc).__name__}: {exc}"

    if sampler is not None:
        sampler.join()

    res.text = "".join(chunks)
    res.needle_ok = NEEDLE_KEY in res.text
    res.decode_tokens = max(0, ntok - 1)
    if first_tok_t is not None:
        res.ttft_s = first_tok_t - t0
        if last_tok_t is not None:
            res.decode_wall_s = last_tok_t - first_tok_t

    # Let the supervisor flush the runner's stderr into exo.log.
    time.sleep(4)
    window = remote_log_since(start_line)
    pt, pp, pms, evals, sends = parse_window(window)
    res.prompt_tokens, res.prefill_path, res.prefill_ms = pt, pp, pms
    res.eval_ms = evals
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=str, default="50,1000,4000,14000")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=10)
    ap.add_argument("--sample-at", type=int, default=0,
                    help="depth label at which to capture native stacks (0=never)")
    ap.add_argument("--sample-seconds", type=int, default=6)
    ap.add_argument("--out", type=str, default="/tmp/decode_depth_sweep.json")
    args = ap.parse_args()

    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    results: list[RunResult] = []

    for depth in depths:
        for rep in range(args.reps):
            do_sample = (args.sample_at != 0 and depth == args.sample_at and rep == 0)
            tag = f"d{depth}"
            print(f"\n=== depth~{depth} rep {rep + 1}/{args.reps}"
                  f"{' [STACK SAMPLE]' if do_sample else ''} ===", flush=True)
            r = run_one(depth, args.max_tokens, do_sample, args.sample_seconds, tag)
            results.append(r)
            print(
                f"  prompt_tokens={r.prompt_tokens} path={r.prefill_path} "
                f"prefill={r.prefill_ms:.0f}ms " if r.prefill_ms else
                f"  prompt_tokens={r.prompt_tokens} path={r.prefill_path} ",
                flush=True,
            )
            em = r.eval_median
            print(
                f"  usage_prompt_tokens={r.usage_prompt_tokens} "
                f"completion={r.usage_completion_tokens}", flush=True,
            )
            print(
                f"  ttft={r.ttft_s:.1f}s decode={r.decode_tokens}tok "
                f"@{r.decode_tok_s:.2f} tok/s | last_layer_eval "
                f"median={em if em is None else round(em, 1)}ms n={len(r.eval_ms)} "
                f"| needle={'YES' if r.needle_ok else 'NO'}"
                f"{' | ERROR ' + r.error if r.error else ''}",
                flush=True,
            )

    print("\n\n================ DEPTH SWEEP SUMMARY ================")
    print(f"{'depth':>8} {'prompt_tok':>11} {'path':>8} {'eval_med_ms':>12} "
          f"{'eval_samples':>40} {'tok/s':>7} {'needle':>7}")
    for r in results:
        samples = ",".join(f"{v:.0f}" for v in r.eval_ms[:12])
        em = r.eval_median
        print(f"{r.depth_label:>8} {str(r.prompt_tokens):>11} "
              f"{str(r.prefill_path):>8} "
              f"{('%.1f' % em) if em is not None else 'n/a':>12} "
              f"{samples:>40} {r.decode_tok_s:>7.2f} "
              f"{'YES' if r.needle_ok else 'NO':>7}")

    print("\n---- per-depth aggregate (median of per-run medians) ----")
    by_depth: dict[int, list[float]] = {}
    for r in results:
        if r.eval_median is not None:
            by_depth.setdefault(r.prompt_tokens or r.depth_label, []).append(r.eval_median)
    for d in sorted(by_depth):
        vals = by_depth[d]
        print(f"  prompt_tokens={d:>7}  eval_median={statistics.median(vals):>8.1f}ms  "
              f"runs={len(vals)}  per-run={[round(v, 1) for v in vals]}")

    with open(args.out, "w") as f:
        json.dump([r.__dict__ for r in results], f, indent=1, default=str)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
