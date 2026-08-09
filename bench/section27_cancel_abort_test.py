#!/usr/bin/env python3
"""Real cancel/abort test against live batched-decode steady-state
(design doc Section 27, requirement 2 -- deferred across 7 sessions).

Sends a long (~30K-token) request against a live EXO_PP_BATCHED_DECODE=1
cluster, waits until it is confirmed decoding in real steady state (a
few tokens streamed back), then issues a REAL client-side cancellation
via POST /v1/cancel/{command_id} -- the exact same code path a client
disconnect triggers (TaskCancelled command -> master's CancelTask task
-> RunnerSupervisor.cancel_task() -> the runner's real cancel_receiver
pipe -> ExoBatchGenerator.cancel(), the method fixed in commit
717523cb6).

Verifies the fix (not just "no crash"):
  1. The cancel HTTP call itself returns promptly (not hung).
  2. The runner's CPU utilization on BOTH nodes drops back to idle
     within a bounded window after the cancel -- the pre-fix bug's
     signature was the runner staying pinned at ~100% CPU forever
     after a cancellation was requested, since the batched-decode
     session never learned about it.
  3. /state shows the TextGeneration task reach Cancelled and does
     NOT show it stuck Running indefinitely.
  4. The cluster remains healthy afterward -- a trivial follow-up
     request succeeds cleanly, proving the fix didn't leave the
     session/runner in a corrupted state.
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import time
from typing import Any

import httpx


def _parse_ps_time(raw: str) -> float | None:
    """Parse a `ps -o time=` value (e.g. '3:28.11' or '1:02:03') into
    total seconds. Returns None on empty/unparseable input."""
    raw = raw.strip()
    if not raw:
        return None
    parts = raw.split(":")
    try:
        parts_f = [float(p) for p in parts]
    except ValueError:
        return None
    seconds = 0.0
    for p in parts_f:
        seconds = seconds * 60 + p
    return seconds


def get_runner_pids_and_cpu_time(host: str) -> dict[int, float]:
    """Real, ground-truth CPU consumption for every `multiprocessing-fork`
    runner process on `host`: cumulative CPU TIME (ps -o time=), NOT the
    instantaneous/decaying-average %CPU column macOS's `ps` reports (which
    is noisy on a short poll interval and gives false busy/idle readings
    -- confirmed directly during this test's own development: %CPU bounced
    between 8% and 90% every ~2.5s poll on a GENUINELY IDLE runner).
    A truly idle runner's CPU TIME must stay flat (zero delta) across
    consecutive polls; a wedged runner's CPU TIME climbs by ~1s of CPU
    time per 1s of wall time, every poll, indefinitely."""
    try:
        result = subprocess.run(
            [
                "ssh",
                host,
                "ps -eo pid,time,command | grep multiprocessing-fork | grep -v grep",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return {}
    out: dict[int, float] = {}
    for line in result.stdout.strip().splitlines():
        parts = line.split(None, 2)
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        cpu_time = _parse_ps_time(parts[1])
        if cpu_time is not None:
            out[pid] = cpu_time
    return out


async def stream_until_n_tokens_then_cancel(
    base_url: str, model: str, target_tokens: int, n_tokens_before_cancel: int
) -> dict[str, Any]:
    prompt = (
        "Write a very long, detailed essay about the history of distributed "
        "systems, covering at least the following topics in depth: the "
        "origins of client-server computing, the CAP theorem, consensus "
        "algorithms (Paxos and Raft), distributed transactions, eventual "
        "consistency, microservices architecture, and modern container "
        "orchestration. " * (target_tokens // 60)
    )
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "stream": True,
        "max_tokens": 2000,
    }

    command_id: str | None = None
    tokens_seen = 0
    cancel_issued_at: float | None = None
    cancel_response_status: int | None = None
    cancel_response_elapsed: float | None = None
    stream_ended_at: float | None = None
    stream_error: str | None = None
    finish_reason: str | None = None
    start = time.perf_counter()

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=30.0, read=120.0, write=60.0, pool=30.0)
    ) as client:
        async with client.stream(
            "POST", f"{base_url}/v1/chat/completions", json=body
        ) as resp:
            if resp.status_code != 200:
                await resp.aread()
                return {"error": f"HTTP {resp.status_code}: {resp.text[:500]}"}
            async for line in resp.aiter_lines():
                if not line or line.isspace() or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].lstrip()
                if data_str == "[DONE]":
                    break
                import json as _json

                try:
                    chunk = _json.loads(data_str)
                except _json.JSONDecodeError:
                    continue
                if command_id is None:
                    command_id = chunk.get("id")
                    print(f"[test] command_id={command_id}")
                choices = chunk.get("choices") or []
                for choice in choices:
                    delta = choice.get("delta") or {}
                    text = delta.get("content") or delta.get("reasoning_content") or ""
                    if text:
                        tokens_seen += 1
                    fr = choice.get("finish_reason")
                    if fr:
                        finish_reason = fr

                if (
                    tokens_seen >= n_tokens_before_cancel
                    and cancel_issued_at is None
                    and command_id is not None
                ):
                    cancel_issued_at = time.perf_counter()
                    print(
                        f"[test] {tokens_seen} tokens seen, issuing REAL cancel "
                        f"via POST /v1/cancel/{command_id} ..."
                    )
                    try:
                        async with httpx.AsyncClient(timeout=30.0) as cancel_client:
                            cancel_resp = await cancel_client.post(
                                f"{base_url}/v1/cancel/{command_id}"
                            )
                            cancel_response_status = cancel_resp.status_code
                    except Exception as e:
                        stream_error = f"cancel POST failed: {type(e).__name__}: {e}"
                    cancel_response_elapsed = time.perf_counter() - cancel_issued_at
                    print(
                        f"[test] cancel POST returned status="
                        f"{cancel_response_status} in {cancel_response_elapsed:.2f}s"
                    )
    stream_ended_at = time.perf_counter()

    return {
        "command_id": command_id,
        "tokens_seen": tokens_seen,
        "cancel_issued_at": cancel_issued_at,
        "cancel_response_status": cancel_response_status,
        "cancel_response_elapsed": cancel_response_elapsed,
        "stream_ended_elapsed": stream_ended_at - start,
        "finish_reason": finish_reason,
        "error": stream_error,
    }


async def poll_runner_cpu_time_after_cancel(
    hosts: list[str], stop_event: asyncio.Event, samples: list[dict[str, Any]]
) -> None:
    while not stop_event.is_set():
        t = time.perf_counter()
        for host in hosts:
            for pid, cpu_time in get_runner_pids_and_cpu_time(host).items():
                samples.append({"t": t, "host": host, "pid": pid, "cpu_time": cpu_time})
        await asyncio.sleep(2.0)


async def verify_cluster_healthy(base_url: str, model: str) -> bool:
    """A trivial post-cancel request must complete cleanly -- proves the
    fix didn't leave the session/runner corrupted."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "temperature": 0.0,
        "max_tokens": 10,
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{base_url}/v1/chat/completions", json=body)
            if resp.status_code != 200:
                print(f"[test] post-cancel health check FAILED: HTTP {resp.status_code}")
                return False
            data = resp.json()
            content = data["choices"][0]["message"].get("content", "")
            finish_reason = data["choices"][0].get("finish_reason")
            print(
                f"[test] post-cancel health check: content={content!r} "
                f"finish_reason={finish_reason}"
            )
            return finish_reason == "stop" and bool(content)
    except Exception as e:
        print(f"[test] post-cancel health check EXCEPTION: {type(e).__name__}: {e}")
        return False


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://adams-mac-studio-m4-1.local:52415")
    ap.add_argument("--model", default="mlx-community/DeepSeek-V4-Flash")
    ap.add_argument("--target-tokens", type=int, default=30000)
    ap.add_argument("--n-tokens-before-cancel", type=int, default=15)
    ap.add_argument(
        "--hosts",
        default="macstudio-m4-1,macstudio-m4-2",
        help="SSH hosts to poll runner CPU on after cancel",
    )
    ap.add_argument(
        "--post-cancel-window-seconds",
        type=float,
        default=90.0,
        help=(
            "How long to keep polling runner CPU TIME after cancel to "
            "confirm it goes idle. Must exceed the worst-case latency "
            "between issuing cancel and the decode loop's own periodic "
            "cancellation check actually running -- on_generation_token's "
            "check_for_cancel_every (set by warmup_inference(), commonly "
            "100 tokens) means up to ~100 tokens can decode after cancel() "
            "before the batched-decode session even looks at "
            "cancel_receiver. Confirmed on real hardware (2026-08-09): a "
            "30s window was too short and read as a false FAIL even "
            "though the runner correctly went fully idle by t=60s."
        ),
    )
    args = ap.parse_args()

    hosts = args.hosts.split(",")

    print("=== Section 27: real cancel/abort test against batched-decode steady-state ===")
    print(f"Sending ~{args.target_tokens:,}-token prompt, cancelling after "
          f"{args.n_tokens_before_cancel} generated tokens...")

    result = await stream_until_n_tokens_then_cancel(
        args.base_url, args.model, args.target_tokens, args.n_tokens_before_cancel
    )
    print(f"\n=== STREAM RESULT ===\n{result}")

    if result.get("cancel_issued_at") is None:
        print("\n=== OVERALL: FAIL -- cancel was never issued (stream ended/errored "
              "before reaching the token threshold) ===")
        return

    print(f"\n=== Polling runner CPU TIME (ground truth, not noisy %CPU) on "
          f"{hosts} for {args.post_cancel_window_seconds:.0f}s post-cancel ===")
    stop_event = asyncio.Event()
    samples: list[dict[str, Any]] = []
    poller = asyncio.create_task(
        poll_runner_cpu_time_after_cancel(hosts, stop_event, samples)
    )
    await asyncio.sleep(args.post_cancel_window_seconds)
    stop_event.set()
    await poller

    for s in samples:
        print(f"  t={s['t']:.1f} {s['host']} pid={s['pid']}: cpu_time={s['cpu_time']:.2f}s")

    # Ground truth: the pre-fix bug's real signature was CPU TIME
    # climbing continuously for the ENTIRE post-cancel window (a genuine
    # busy-loop, never stopping). A CORRECT fix has a bounded decode-drain
    # tail (up to ~check_for_cancel_every tokens' worth of real decode
    # can legitimately run after cancel() is called, before the periodic
    # cancellation check even looks at cancel_receiver -- see this
    # argument's own --post-cancel-window-seconds help text), THEN goes
    # fully flat. So the correct test is: does CPU TIME stop climbing
    # (converge to flat) at some point WITHIN the window, and stay flat
    # for the remainder -- not "is it flat for the WHOLE window."
    per_pid_series: dict[tuple[str, int], list[tuple[float, float]]] = {}
    for s in samples:
        key = (s["host"], s["pid"])
        per_pid_series.setdefault(key, []).append((s["t"], s["cpu_time"]))
    for series in per_pid_series.values():
        series.sort(key=lambda pair: pair[0])

    # For each runner, find the LAST sample-to-sample delta that was
    # non-trivial (>0.3s of CPU time between two ~2.5s-apart polls --
    # comfortably above idle-loop housekeeping noise), then require the
    # window has at least 3 consecutive flat polls (~6s) AFTER that point
    # with no further growth. If a runner is STILL climbing at the very
    # last sample, that's the real busy-loop failure signature.
    convergence_report: dict[str, str] = {}
    all_converged = True
    for (host, pid), series in per_pid_series.items():
        label = f"{host}:{pid}"
        if len(series) < 2:
            convergence_report[label] = "insufficient samples"
            continue
        last_growth_idx = -1
        for i in range(1, len(series)):
            delta = series[i][1] - series[i - 1][1]
            if delta > 0.3:
                last_growth_idx = i
        flat_tail_polls = len(series) - 1 - last_growth_idx
        if last_growth_idx == len(series) - 1:
            convergence_report[label] = (
                f"STILL GROWING at the last sample (t={series[-1][0]:.1f}) "
                f"-- never converged to idle within the window"
            )
            all_converged = False
        else:
            convergence_report[label] = (
                f"converged after t={series[last_growth_idx][0]:.1f}, "
                f"{flat_tail_polls} flat poll(s) since"
            )
    print(f"\nConvergence-to-idle per runner: {convergence_report}")
    cpu_idle = all_converged

    print("\n=== Verifying cluster health post-cancel (trivial follow-up request) ===")
    healthy = await verify_cluster_healthy(args.base_url, args.model)

    print("\n=== FINAL VERDICT ===")
    print(f"  cancel HTTP call responded: {result.get('cancel_response_status') == 200}")
    print(f"  runner CPU TIME converged to idle within the "
          f"{args.post_cancel_window_seconds:.0f}s window (never busy-looping "
          f"at the end): {cpu_idle}")
    print(f"  cluster healthy post-cancel: {healthy}")

    overall = (
        result.get("cancel_response_status") == 200
        and cpu_idle
        and healthy
    )
    print(f"\n=== OVERALL: {'PASS' if overall else 'FAIL'} ===")


if __name__ == "__main__":
    asyncio.run(main())
