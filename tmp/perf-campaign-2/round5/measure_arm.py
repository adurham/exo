#!/usr/bin/env python3
"""Measure ONE speculative-decode gamma arm against the LIVE cluster.

Sends `--reps` streaming chat completions at a fixed prompt depth (built via
bench/quality_probe_dsv4.py's build_prompt, imported the way
tmp/perf-campaign-2/round3/r3_needle_capture.py does), records client-side
timing (ttft/total/decode_tps), then SSHes to both cluster nodes to scrape
the cumulative `[MTP] cycles=... mean_accept=X/G hist=...` counter and any
`[MTP-PROF]` phase-timing dumps, taking a BEFORE snapshot at script start so
only the delta accrued during this arm is attributed to it.

CRITICAL: this script NEVER fabricates a number. Any input it cannot find
(the [MTP] line absent because EXO_DSV4_MTP_LOG_INTERVAL was unset at
runner-launch time; MTP-PROF absent because EXO_DSV4_MTP_PROFILE was unset;
ssh failures; etc.) results in a null field plus an entry in the top-level
`warnings` list — never a silently substituted default.

Usage:
  measure_arm.py --gamma 4 --reps 3 --depth 89000 \\
      --out results/arm_g4_boot1.json --tag boot1
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[3]
assert (REPO_ROOT / "bench" / "quality_probe_dsv4.py").exists(), (
    f"expected repo root at {REPO_ROOT}, but bench/quality_probe_dsv4.py "
    "not found there -- adjust parents[N] if this script moved"
)

_spec = importlib.util.spec_from_file_location(
    "qp", str(REPO_ROOT / "bench" / "quality_probe_dsv4.py")
)
qp = importlib.util.module_from_spec(_spec)
sys.modules["qp"] = qp
_spec.loader.exec_module(qp)  # type: ignore[union-attr]

# NOTE: ab_probe_tier1.py hardcodes MODEL="mlx-community/DeepSeek-V4-Flash",
# but a live /state query during this dispatch's verification (2026-09-03)
# showed the ACTUALLY PLACED model id is "deepseek-ai/DeepSeek-V4-Flash-0731"
# -- the harness-map's assumed model id is stale. Use --model to override if
# placement changes again; this default matches the live cluster as of now.
API = "http://192.168.86.201:52415"
MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
NODES = ["macstudio-m4-1", "macstudio-m4-2"]
LOG_PATH = "~/exo.log"
SSH_TIMEOUT_S = 30

MTP_LINE_RE = re.compile(
    r"\[MTP\] cycles=(?P<cycles>\d+) "
    r"mean_accept=(?P<mean_accept>[\d.]+)/(?P<gamma_denominator>\d+) "
    r"hist=(?P<hist>[\d:,]+)"
)
# One phase-series row, e.g.:
#   [MTP-PROF]   B=4 draft      mean=  1.23ms min=  0.98ms max=  2.10ms n=512
#   [MTP-PROF]   B=4 rb_pool_restores mean=  0.00 min=  0.00 max=  0.00 n=512
MTP_PROF_ROW_RE = re.compile(
    r"\[MTP-PROF\]\s+B=(?P<bucket>\d+)\s+(?P<series>\S+)\s+"
    r"mean=\s*(?P<mean>[\d.]+)(?P<mean_ms>ms)?\s+"
    r"min=\s*(?P<min>[\d.]+)(?P<min_ms>ms)?\s+"
    r"max=\s*(?P<max>[\d.]+)(?P<max_ms>ms)?\s+"
    r"n=(?P<n>\d+)"
)


def parse_hist(hist_str: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in hist_str.split(","):
        k, _, v = part.partition(":")
        if k and v:
            out[k] = int(v)
    return out


def parse_mtp_line(line: str | None) -> dict[str, Any] | None:
    if not line:
        return None
    m = MTP_LINE_RE.search(line)
    if not m:
        return None
    return {
        "cycles": int(m.group("cycles")),
        "mean_accept": float(m.group("mean_accept")),
        "gamma_denominator": int(m.group("gamma_denominator")),
        "hist": parse_hist(m.group("hist")),
        "raw": line.strip(),
    }


def ssh_run(node: str, remote_cmd: str, warnings: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", node, remote_cmd],
            capture_output=True,
            text=True,
            timeout=SSH_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        warnings.append(f"ssh to {node} timed out running: {remote_cmd!r}")
        return None
    except Exception as e:  # noqa: BLE001
        warnings.append(f"ssh to {node} failed ({e!r}) running: {remote_cmd!r}")
        return None
    if proc.returncode != 0 and not proc.stdout.strip():
        # grep returns 1 when no lines match -- not a hard failure, just
        # means the line hasn't been emitted (e.g. LOG_INTERVAL unset).
        return None
    return proc.stdout


def get_last_mtp_line(node: str, warnings: list[str]) -> dict[str, Any] | None:
    cmd = (
        r"grep -oE '\[MTP\] cycles=[0-9]+ mean_accept=[0-9.]+/[0-9]+ "
        r"hist=[0-9:,]+' " + LOG_PATH + " | tail -1"
    )
    out = ssh_run(node, cmd, warnings)
    parsed = parse_mtp_line(out.strip() if out else None)
    if out is not None and out.strip() and parsed is None:
        warnings.append(f"[MTP] line found on {node} but failed to parse: {out!r}")
    return parsed


def get_mtp_prof_table(node: str, warnings: list[str]) -> dict[str, Any] | None:
    cmd = r"grep '\[MTP-PROF\]' " + LOG_PATH + " | tail -2000"
    out = ssh_run(node, cmd, warnings)
    if not out or not out.strip():
        return None
    table: dict[str, dict[str, dict[str, Any]]] = {}
    n_rows = 0
    for line in out.splitlines():
        m = MTP_PROF_ROW_RE.search(line)
        if not m:
            continue
        n_rows += 1
        bucket = f"B={m.group('bucket')}"
        series = m.group("series")
        unit = "ms" if m.group("mean_ms") else "count"
        table.setdefault(bucket, {})[series] = {
            "mean": float(m.group("mean")),
            "min": float(m.group("min")),
            "max": float(m.group("max")),
            "n": int(m.group("n")),
            "unit": unit,
        }
    if n_rows == 0:
        warnings.append(
            f"[MTP-PROF] lines present on {node} but none matched the row regex"
        )
        return None
    return table


def send_rep(
    client: httpx.Client, api: str, prompt: str, max_tokens: int, model: str
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    t_first = None
    usage = None
    finish_reason = None
    text_parts: list[str] = []
    # Wall-clock arrival time (seconds since t0) of EVERY chunk that carried
    # at least one token of content. This is the raw evidence needed to
    # detect burst/buffered delivery -- do NOT collapse it to a single
    # ttft/total pair before recording it, that is exactly the bug that
    # made arm_g3_boot1/arm_g4_boot1 report 300+ tok/s decode.
    chunk_times: list[float] = []
    # NOTE on streaming correctness: client.stream(...) + r.iter_lines() is
    # true incremental SSE consumption -- httpx does not read the whole
    # response body before this loop starts (that would require r.text,
    # r.read(), or list(r.iter_lines()), none of which are used here).
    # iter_lines() yields a line as soon as a '\n' is seen in the decoded
    # byte stream; it does not wait for the connection to close. So if
    # tokens still arrive in a visible end-of-window burst below, that is
    # evidence of server-side (or intermediary) buffering, not a client bug.
    with client.stream("POST", f"{api}/v1/chat/completions", json=body) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload.strip() == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices", []):
                if ch.get("finish_reason"):
                    finish_reason = ch["finish_reason"]
                delta = ch.get("delta", {}).get("content") or ""
                if delta:
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    chunk_times.append(now - t0)
                    text_parts.append(delta)
    t_end = time.perf_counter()

    ttft_s = (t_first - t0) if t_first is not None else None
    total_s = t_end - t0
    prompt_tokens = usage.get("prompt_tokens") if usage else None
    completion_tokens = usage.get("completion_tokens") if usage else None

    n_chunks = len(chunk_times)
    t_first_tok: float | None = None
    t_last_tok: float | None = None
    decode_window_s: float | None = None
    decode_tps: float | None = None
    interchunk_ms: dict[str, float | None] = {"p50": None, "max": None}
    stream_looks_buffered: bool | None = None

    if n_chunks >= 2:
        t_first_tok = chunk_times[0]
        t_last_tok = chunk_times[-1]
        decode_window_s = t_last_tok - t_first_tok
        if decode_window_s > 0:
            # n-1: the window spans BETWEEN the first and last token, so it
            # covers n_chunks - 1 inter-token gaps, not n_chunks tokens.
            decode_tps = (n_chunks - 1) / decode_window_s

        gaps_ms = [
            (chunk_times[i] - chunk_times[i - 1]) * 1000.0
            for i in range(1, n_chunks)
        ]
        sorted_gaps = sorted(gaps_ms)
        mid = len(sorted_gaps) // 2
        if len(sorted_gaps) % 2 == 1:
            p50 = sorted_gaps[mid]
        else:
            p50 = (sorted_gaps[mid - 1] + sorted_gaps[mid]) / 2.0
        interchunk_ms = {"p50": p50, "max": max(sorted_gaps)}

        # Smoking-gun check for burst/buffered delivery: if more than half
        # of all token-bearing chunks land within the final 10% of the
        # decode window, the stream was not really arriving incrementally.
        threshold = t_first_tok + 0.9 * decode_window_s
        n_in_tail = sum(1 for t in chunk_times if t >= threshold)
        stream_looks_buffered = n_in_tail > (n_chunks / 2.0)
    elif n_chunks == 1:
        t_first_tok = chunk_times[0]
        t_last_tok = chunk_times[0]

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": ttft_s,
        "total_s": total_s,
        "finish_reason": finish_reason,
        "n_chunks": n_chunks,
        "t_first_tok": t_first_tok,
        "t_last_tok": t_last_tok,
        "decode_window_s": decode_window_s,
        "decode_tps": decode_tps,
        "interchunk_ms": interchunk_ms,
        "stream_looks_buffered": stream_looks_buffered,
        "response_chars": sum(len(p) for p in text_parts),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=int, required=True,
                     help="the EXO_SPECULATIVE_GAMMA the currently-running "
                          "cluster was launched with (recorded, not set by "
                          "this script -- it does not relaunch anything)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--depth", type=int, default=89000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="untagged")
    ap.add_argument("--max-tokens", type=int, default=300,
                     help="must be >=200 to get a solid decode window")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--api", default=API)
    args = ap.parse_args()

    if args.max_tokens < 200:
        print(
            f"WARNING: --max-tokens={args.max_tokens} < 200; decode window "
            "may be too short for a solid TPS estimate",
            file=sys.stderr,
        )

    warnings: list[str] = []

    print(f"[{args.tag}] taking BEFORE [MTP] snapshot on both nodes...", file=sys.stderr)
    mtp_before = {node: get_last_mtp_line(node, warnings) for node in NODES}
    for node in NODES:
        if mtp_before[node] is None:
            warnings.append(
                f"no [MTP] line found on {node} at BEFORE snapshot "
                "(expected if EXO_DSV4_MTP_LOG_INTERVAL is unset)"
            )

    reps: list[dict[str, Any]] = []
    with httpx.Client(timeout=httpx.Timeout(3600.0, connect=30.0)) as client:
        for i in range(args.reps):
            prompt, expected_needle = qp.build_prompt(args.depth, seed=7749 + i)
            print(f"[{args.tag}] rep {i}: sending ({len(prompt):,} chars)...",
                  file=sys.stderr)
            rec = send_rep(client, args.api, prompt, args.max_tokens, args.model)
            rec["rep"] = i
            rec["warmup"] = (i == 0)
            rec["expected_needle"] = expected_needle
            reps.append(rec)
            print(
                f"[{args.tag}] rep {i} done: ttft={rec['ttft_s']} "
                f"total={rec['total_s']:.2f}s decode_tps={rec['decode_tps']}",
                file=sys.stderr,
            )

    print(f"[{args.tag}] taking AFTER [MTP] snapshot + MTP-PROF dump on both nodes...",
          file=sys.stderr)
    mtp_after = {node: get_last_mtp_line(node, warnings) for node in NODES}
    mtp_prof = {node: get_mtp_prof_table(node, warnings) for node in NODES}
    for node in NODES:
        if mtp_after[node] is None:
            warnings.append(
                f"no [MTP] line found on {node} at AFTER snapshot "
                "(expected if EXO_DSV4_MTP_LOG_INTERVAL is unset)"
            )
        if mtp_prof[node] is None:
            warnings.append(
                f"no [MTP-PROF] data found on {node} "
                "(expected if EXO_DSV4_MTP_PROFILE is unset)"
            )

    # ---- MTP counter delta computation (per node) ----
    # mean_accept in the log line is a running/cumulative mean since process
    # start, not a per-arm value. To get the acceptance rate attributable to
    # THIS arm we back out the cumulative accepted-token sums from
    # (mean * cycles) at each snapshot and diff those, then divide by the
    # cycle delta. hist bins are cumulative counts, so they can be diffed
    # directly bin-by-bin.
    mtp_counters: dict[str, Any] = {}
    for node in NODES:
        before = mtp_before[node]
        after = mtp_after[node]
        entry: dict[str, Any] = {
            "before": before,
            "after": after,
            "cycles_delta": None,
            "mean_accept_delta": None,
            "hist_delta": None,
        }
        if before is not None and after is not None:
            cycles_delta = after["cycles"] - before["cycles"]
            entry["cycles_delta"] = cycles_delta
            if cycles_delta > 0:
                total_accepted_before = before["mean_accept"] * before["cycles"]
                total_accepted_after = after["mean_accept"] * after["cycles"]
                entry["mean_accept_delta"] = (
                    total_accepted_after - total_accepted_before
                ) / cycles_delta
            else:
                warnings.append(
                    f"{node}: cycles_delta <= 0 ({cycles_delta}) between "
                    "BEFORE/AFTER snapshots -- no new cycles accrued during "
                    "this arm (or the log rotated); mean_accept_delta is null"
                )
            if set(before["hist"]) | set(after["hist"]):
                hist_delta = {}
                for k in set(before["hist"]) | set(after["hist"]):
                    hist_delta[k] = after["hist"].get(k, 0) - before["hist"].get(k, 0)
                entry["hist_delta"] = hist_delta
        mtp_counters[node] = entry

    # ---- derived_tps ----
    # derived_tps = (1 + mean_accept_delta) / cycle_ms * 1000
    # cycle_ms comes from the MTP-PROF "total" series mean, for whichever
    # node/bucket has usable data. Preference: node macstudio-m4-1, the
    # bucket with the largest sample count `n` (most representative of the
    # steady-state decode regime this arm actually ran).
    derived_tps = None
    derived_tps_source: dict[str, Any] | None = None
    mean_accept_delta_for_derive = None
    for node in NODES:
        cand = mtp_counters[node]["mean_accept_delta"]
        if cand is not None:
            mean_accept_delta_for_derive = cand
            break

    cycle_ms = None
    best_n = -1
    best_node = best_bucket = None
    for node in NODES:
        table = mtp_prof[node]
        if not table:
            continue
        for bucket, series_map in table.items():
            total = series_map.get("total")
            if total is None or total.get("unit") != "ms":
                continue
            if total["n"] > best_n:
                best_n = total["n"]
                cycle_ms = total["mean"]
                best_node = node
                best_bucket = bucket

    if mean_accept_delta_for_derive is None:
        warnings.append(
            "derived_tps is null: no node had a usable mean_accept_delta "
            "(no [MTP] BEFORE+AFTER pair with cycles_delta>0 on any node)"
        )
    if cycle_ms is None:
        warnings.append(
            "derived_tps is null: no node had a usable MTP-PROF 'total' "
            "series mean (in ms) to use as cycle_ms"
        )
    if mean_accept_delta_for_derive is not None and cycle_ms is not None and cycle_ms > 0:
        derived_tps = (1 + mean_accept_delta_for_derive) / cycle_ms * 1000
        derived_tps_source = {
            "node": best_node,
            "bucket": best_bucket,
            "cycle_ms": cycle_ms,
            "mean_accept_delta": mean_accept_delta_for_derive,
        }

    measured_reps = [r for r in reps if not r["warmup"]]
    decode_tps_vals = [r["decode_tps"] for r in measured_reps if r["decode_tps"] is not None]
    if not decode_tps_vals:
        warnings.append(
            "no measured (non-warmup) rep produced a usable decode_tps"
        )
    aggregate_decode_tps = {
        "min": min(decode_tps_vals) if decode_tps_vals else None,
        "median": statistics.median(decode_tps_vals) if decode_tps_vals else None,
        "max": max(decode_tps_vals) if decode_tps_vals else None,
        "n": len(decode_tps_vals),
    }

    result = {
        "arm": {"gamma": args.gamma, "tag": args.tag, "depth": args.depth,
                 "reps": args.reps, "max_tokens": args.max_tokens,
                 "model": args.model, "api": args.api},
        "reps": reps,
        "aggregate_decode_tps": aggregate_decode_tps,
        "mtp_counters": mtp_counters,
        "mtp_prof": mtp_prof,
        "derived_tps": derived_tps,
        "derived_tps_source": derived_tps_source,
        "warnings": warnings,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{args.tag}] wrote {out_path}", file=sys.stderr)
    if warnings:
        print(f"[{args.tag}] {len(warnings)} WARNING(S):", file=sys.stderr)
        for w in warnings:
            print(f"  - {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
