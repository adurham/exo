#!/usr/bin/env /usr/bin/python3
"""Round-11 analysis: read replay.jsonl + capture.jsonl, extract phase-marks
dicts, and report per-phase median/min/max (RANGES, never means -- standing
campaign rule), the derived dispatch_and_ipc_gap, and the closure check.

PM RUNS THIS, NOT THE IMPLEMENTING SUBAGENT.

Usage:
    /usr/bin/python3 tmp/perf-campaign-2/round11/analyze_marks.py

Reads:
    tmp/perf-campaign-2/round11/results/replay.jsonl
    tmp/real-usage-capture-20260902/phase2/capture.jsonl

The replay driver's requests carry marks two ways depending on what the SSE
stream exposed to a stdlib client: `generation_stats_raw` (the raw
`: generation_stats {...}` SSE comment lines captured verbatim by
replay_c1.py) is the primary source here, since it is guaranteed present in
replay.jsonl regardless of what the passive-capture proxy chose to persist.
capture.jsonl is cross-referenced only to report the request count sanity
check and is not required for the marks extraction itself.
"""

import json
import statistics
import sys
from pathlib import Path

RESULTS_PATH = (
    Path(__file__).resolve().parent / "results" / "replay.jsonl"
)
CAPTURE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "real-usage-capture-20260902"
    / "phase2"
    / "capture.jsonl"
)

CLOSURE_TOLERANCE_MS = 10.0

# The API-side ordered mark sequence (a1-a7), used to walk "adjacent deltas"
# for the closure check. Field names as emitted by exo.api.phase_marks /
# generate_chat_stream.
API_MARK_ORDER = [
    "messages_serialized_ms",
    "command_published_ms",
    "first_chunk_received_ms",
    "first_sse_written_ms",
    "last_sse_written_ms",
    "stream_closed_ms",
]

RUNNER_MARK_ORDER = [
    "template_rendered_ms",
    "tokenized_ms",
    "trie_matched_ms",
    "kv_restored_lazy_no_eval_ms",
    "prefill_start_ms",
    "prefill_done_ms",
    "cache_commit_pre_first_token_ms",
    "first_token_emitted_ms",
    "last_token_ms",
    "stop_detected_ms",
]


def extract_generation_stats(raw_lines: list[str]) -> dict | None:
    """Parse the LAST `: generation_stats {...}` SSE comment line captured
    for a request (the terminal one carries the full stats block)."""
    if not raw_lines:
        return None
    last = raw_lines[-1]
    prefix = ": generation_stats "
    if not last.startswith(prefix):
        return None
    try:
        return json.loads(last[len(prefix) :])
    except json.JSONDecodeError:
        return None


def load_replay_records() -> list[dict]:
    if not RESULTS_PATH.exists():
        print(f"ERROR: {RESULTS_PATH} does not exist", file=sys.stderr)
        return []
    records = []
    with RESULTS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_capture_count() -> int | None:
    if not CAPTURE_PATH.exists():
        return None
    n = 0
    with CAPTURE_PATH.open() as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def range_stats(values: list[float]) -> str:
    if not values:
        return "n=0"
    return (
        f"n={len(values)} median={statistics.median(values):.2f}ms "
        f"min={min(values):.2f}ms max={max(values):.2f}ms"
    )


def main() -> int:
    records = load_replay_records()
    print(f"Loaded {len(records)} replay records from {RESULTS_PATH}")

    capture_count = load_capture_count()
    if capture_count is not None:
        print(f"capture.jsonl has {capture_count} total lines (all-time, not "
              f"scoped to this replay -- cross-reference only)")

    per_phase_api: dict[str, list[float]] = {k: [] for k in API_MARK_ORDER}
    per_phase_runner: dict[str, list[float]] = {k: [] for k in RUNNER_MARK_ORDER}
    dispatch_and_ipc_gaps: list[float] = []
    closure_gaps: list[float] = []
    no_marks_count = 0

    for rec in records:
        stats = extract_generation_stats(rec.get("generation_stats_raw") or [])
        if stats is None:
            no_marks_count += 1
            continue

        api_marks = stats.get("api_phase_marks_ms")
        runner_marks = stats.get("phase_marks_ms")

        if not api_marks and not runner_marks:
            no_marks_count += 1
            continue

        if api_marks:
            for k in API_MARK_ORDER:
                if k in api_marks:
                    per_phase_api[k].append(api_marks[k])
        if runner_marks:
            for k in RUNNER_MARK_ORDER:
                if k in runner_marks:
                    per_phase_runner[k].append(runner_marks[k])

        # Derived dispatch_and_ipc_gap:
        #   (API: first_chunk_received - command_published, i.e. the
        #    first_chunk_received_ms delta itself, since marks are already
        #    cumulative-from-previous-mark deltas)
        # - (RUNNER: first_token_emitted - task_received, i.e. the SUM of
        #    every runner delta from template_rendered_ms through
        #    first_token_emitted_ms)
        if api_marks and runner_marks and "first_chunk_received_ms" in api_marks:
            runner_to_first_token = 0.0
            have_all = True
            for k in RUNNER_MARK_ORDER:
                if k == "first_token_emitted_ms":
                    if k in runner_marks:
                        runner_to_first_token += runner_marks[k]
                    else:
                        have_all = False
                    break
                if k in runner_marks:
                    runner_to_first_token += runner_marks[k]
                else:
                    have_all = False
                    break
            if have_all:
                gap = api_marks["first_chunk_received_ms"] - runner_to_first_token
                dispatch_and_ipc_gaps.append(gap)

        # Closure check: (stream_closed - handler_entered) - sum(adjacent
        # API deltas). "stream_closed - handler_entered" is the
        # INDEPENDENTLY-measured total span attached under
        # exo.api.phase_marks.TOTAL_SPAN_KEY (a single perf_counter
        # subtraction taken at register()-time vs the last mark-time),
        # never itself derived from the delta chain -- this is what makes
        # the check non-tautological. |median| <= 10ms proves the delta
        # chain and the independent total agree, i.e. no mark was skipped,
        # double-counted, or measured against the wrong clock.
        if api_marks and all(k in api_marks for k in API_MARK_ORDER):
            total_span_key = "_handler_to_last_mark_span_ms"
            if total_span_key in api_marks:
                summed_deltas = sum(api_marks[k] for k in API_MARK_ORDER)
                closure_gaps.append(api_marks[total_span_key] - summed_deltas)

    print()
    print("=== API-process phase marks (ranges, never means) ===")
    for k in API_MARK_ORDER:
        print(f"  {k}: {range_stats(per_phase_api[k])}")

    print()
    print("=== Runner-process phase marks (ranges, never means) ===")
    for k in RUNNER_MARK_ORDER:
        print(f"  {k}: {range_stats(per_phase_runner[k])}")

    print()
    print("=== Derived: dispatch_and_ipc_gap ===")
    print(
        "  formula: (API first_chunk_received_ms) - "
        "(RUNNER sum template_rendered..first_token_emitted)"
    )
    print(f"  {range_stats(dispatch_and_ipc_gaps)}")

    print()
    print("=== Closure check ===")
    print(
        "  closure_gap = (stream_closed - handler_entered) - "
        "sum(adjacent API deltas)"
    )
    if closure_gaps:
        med = statistics.median(closure_gaps)
        passed = abs(med) <= CLOSURE_TOLERANCE_MS
        print(f"  median closure_gap = {med:.3f}ms (tolerance "
              f"+/-{CLOSURE_TOLERANCE_MS}ms)")
        print(f"  CLOSURE CHECK: {'PASS' if passed else 'FAIL'}")
    else:
        print("  no complete API mark sequences available -- CLOSURE CHECK: FAIL "
              "(no data)")

    print()
    print(f"Requests with NO marks at all: {no_marks_count} "
          f"(out of {len(records)} total)")
    if no_marks_count > 0:
        print("  NONZERO -- this means G4 (env var not reaching the runner) "
              "or G6 (a finish path missing instrumentation) FAILED. "
              "Investigate before trusting any of the above numbers.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
