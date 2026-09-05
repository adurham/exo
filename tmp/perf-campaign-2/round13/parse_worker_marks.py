#!/usr/bin/env python3
"""Campaign-2 Round-13 Gate-A parser/analyzer for worker-side PHASE_MARK lines.

Consumes the raw combined output of `cluster-diag.sh marks <node> [N]` (i.e.
`tail -n N ~/exo.log | grep PHASE_MARK` on a worker node) and computes the
Gate-A statistics for the `state-update-applied -> plan_step-observed` delta.

STDLIB ONLY. Python 3.9 compatible. Run as:
    /usr/bin/python3 parse_worker_marks.py <marks_file> [<marks_file2> ...]
    (or pipe a single file's content via stdin with no argv)

Each argv file is analyzed SEPARATELY (never pooled across nodes -- hard gate
G2: no cross-node clock arithmetic). A JSON summary is written next to each
input file as `<inputbasename>.analysis.json`.

============================================================================
EXACT EMITTER FORMAT (quoted verbatim from src/exo/worker/phase_marks.py,
this parser MUST match it byte-for-byte -- do not guess):

    mark_state_applied(event_idx):
        f"{_MARK_PREFIX} state_applied event_idx={event_idx} "
        f"t={time.perf_counter():.6f}"
    -> "PHASE_MARK state_applied event_idx=<int> t=<float %.6f>"

    mark_plan_step_observed(event_idx, wake_kind, task):
        f"{_MARK_PREFIX} plan_step_observed event_idx={event_idx} "
        f"t={time.perf_counter():.6f} wake_kind={wake_kind} "
        f"task={task.__class__.__name__}"
    -> "PHASE_MARK plan_step_observed event_idx=<int> t=<float %.6f> wake_kind=<event|event_raced_timeout|timeout> task=<ClassName>"

The `task=` field is NEW as of this round (closes the A2 instrumentation
gap). It carries the dispatched task's concrete class name VERBATIM --
e.g. `task=CreateRunner`, `task=DownloadModel`, `task=TextGeneration`. The
emitter (src/exo/worker/phase_marks.py) does NOT classify it as
backoff-gated vs request-path; that policy lives HERE, in this parser, so
it stays auditable against PREDICTION.md (see A2 below). Older captures
predating this change have NO `task=` field at all -- this parser treats
that absence as UNDETERMINED per mark (see A2), never as a silent
assumption in either direction.

Both lines are logged via `logger.info(...)` (loguru), so in the real
~/exo.log they will be prefixed by loguru's own timestamp/level/location
preamble before the literal "PHASE_MARK ..." text begins. This parser does
NOT assume anything about that preamble's format -- it locates the
"PHASE_MARK ..." substring within each line and parses from there, ignoring
whatever precedes it (loguru formatting, ANSI codes, etc.) and silently
ignoring any line that does not contain a well-formed PHASE_MARK payload.

_MARK_PREFIX = "PHASE_MARK". event_idx is the pairing key (IndexedEvent.idx,
globally monotonic per-worker, NOT necessarily contiguous). Timestamps `t=`
are `time.perf_counter()` values -- an ARBITRARY per-process epoch, valid
only for computing deltas WITHIN one worker process's own mark stream
(never compared across files/nodes).
============================================================================

BINDING AMENDMENTS IMPLEMENTED (round13/PREDICTION.md, DATED AMENDMENT):

A2 (timeout scoping): Gate A's "ZERO timeout-driven wakes on the request
   path" excludes dispatches gated by KeyedBackoff.should_proceed
   (CreateRunner / DownloadModel retries). The emitted mark line now carries
   a `task=<ClassName>` field recording the dispatched task's concrete class
   name VERBATIM (closed this round -- src/exo/worker/phase_marks.py /
   src/exo/worker/main.py). This parser -- not the emitter -- implements the
   A2 classification:
     - `task` in {CreateRunner, DownloadModel} -> BACKOFF-GATED, excluded
       from Gate A's zero.
     - any other task class name -> REQUEST-PATH, a true `timeout` wake on
       one of these counts against the zero. An unanticipated task type
       shows up as ITSELF (its real class name) in the breakdown, never
       silently folded into either bucket.
     - NO `task=` field at all (pre-this-round captures) -> UNDETERMINED
       per mark, named explicitly (`_TASK_TYPE_FIELD_MISSING`), never
       silently assumed request-path or backoff-gated. Any UNDETERMINED
       true-timeout mark blocks a clean Gate-A pass on the timeout
       sub-condition -- old-format data can never mechanically produce a
       clean verdict.
   Both the total true-timeout count and the request-path-scoped count are
   reported explicitly, plus a full per-task-type breakdown of true
   timeouts so the exclusion is auditable rather than asserted.

A3 (wake classification): wake_kind is 3-way ("event", "event_raced_timeout",
   "timeout"), read directly off the `wake_kind=` field. Only "timeout"
   counts as a true timeout-driven wake; "event_raced_timeout" is event-driven
   and does NOT count against Gate A's zero.

A4 (pairing): each plan_step_observed pairs to the EARLIEST unpaired
   state_applied mark seen so far (in file order / event_idx order of
   arrival), not by counting, not by nearest, not by latest. Implemented
   with a FIFO queue of unclaimed state_applied event_idxs: every
   state_applied line seen (in the order it appears in the log) is pushed;
   every plan_step_observed line pops the OLDEST unclaimed entry. This
   directly encodes "since the prior planner wake" because a wake always
   drains every state_applied that arrived before it, and coalescing
   (several state_applied between two wakes) leaves the extra entries in
   the queue for later/no pairing -- it is never assumed that the count of
   state_applied lines between wakes equals 1, and event_idx values are
   never used for arithmetic/counting, only carried through as identifiers.
"""

import json
import re
import statistics
import sys
from os.path import basename, dirname, join
from typing import Dict, List, Optional, Tuple

_MARK_PREFIX = "PHASE_MARK"

_STATE_APPLIED_RE = re.compile(
    r"PHASE_MARK\s+state_applied\s+event_idx=(?P<event_idx>-?\d+)\s+"
    r"t=(?P<t>-?\d+\.\d+)"
)
_PLAN_STEP_OBSERVED_RE = re.compile(
    r"PHASE_MARK\s+plan_step_observed\s+event_idx=(?P<event_idx>-?\d+)\s+"
    r"t=(?P<t>-?\d+\.\d+)\s+wake_kind=(?P<wake_kind>event_raced_timeout|event|timeout)"
    r"(?:\s+task=(?P<task>\S+))?"
)

_VALID_WAKE_KINDS = ("event", "event_raced_timeout", "timeout")

# A2 classification: task class names gated by KeyedBackoff.should_proceed
# (CreateRunner / DownloadModel retries) are excluded from Gate A's zero per
# the round-13 pre-registration (PREDICTION.md). Any OTHER task class name
# is request-path. This is the ONLY place this classification is encoded --
# it is deliberately absent from the emitter (src/exo/worker/phase_marks.py),
# so it stays auditable here against PREDICTION.md rather than baked in
# silently upstream.
_BACKOFF_GATED_TASK_TYPES = frozenset(("CreateRunner", "DownloadModel"))

# A2 finding (pre-this-round data only): older captures predate the `task=`
# field entirely. Name the gap precisely so the UNDETERMINED warning is
# specific, not vague.
_TASK_TYPE_FIELD_MISSING = (
    "This plan_step_observed mark has no `task=` field -- it predates the "
    "round-13 instrumentation fix to src/exo/worker/phase_marks.py / "
    "src/exo/worker/main.py that records the dispatched task's concrete "
    "class name. It cannot be mechanically classified as a CreateRunner/"
    "DownloadModel backoff-gated retry dispatch vs a request-path dispatch."
)

# Gate A pre-registered PASS bands (round12/PREDICTION.md, inherited verbatim
# by round13 -- DO NOT ADJUST).
_GATE_A_MEDIAN_MAX_MS = 10.0
_GATE_A_P99_MAX_MS = 20.0

# Baseline (flag-OFF) fingerprint bands, informational only.
_BASELINE_MEDIAN_LO_MS, _BASELINE_MEDIAN_HI_MS = 35.0, 65.0
_BASELINE_P95_LO_MS, _BASELINE_P95_HI_MS = 85.0, 110.0


class ParsedMark(object):
    __slots__ = ("kind", "event_idx", "t", "wake_kind", "task", "line_no")

    def __init__(self, kind, event_idx, t, wake_kind, task, line_no):
        self.kind = kind
        self.event_idx = event_idx
        self.t = t
        self.wake_kind = wake_kind
        self.task = task  # None for state_applied marks or pre-fix data.
        self.line_no = line_no


def parse_lines(lines):
    """Parse raw log lines into a list of ParsedMark, ignoring non-matching
    lines silently (they may be unrelated log noise interleaved by grep's
    context or by concurrent loggers writing to the same file)."""
    marks = []
    for i, raw_line in enumerate(lines):
        m = _STATE_APPLIED_RE.search(raw_line)
        if m:
            marks.append(
                ParsedMark(
                    kind="state_applied",
                    event_idx=int(m.group("event_idx")),
                    t=float(m.group("t")),
                    wake_kind=None,
                    task=None,
                    line_no=i + 1,
                )
            )
            continue
        m = _PLAN_STEP_OBSERVED_RE.search(raw_line)
        if m:
            marks.append(
                ParsedMark(
                    kind="plan_step_observed",
                    event_idx=int(m.group("event_idx")),
                    t=float(m.group("t")),
                    wake_kind=m.group("wake_kind"),
                    task=m.group("task"),  # None if the line predates task=
                    line_no=i + 1,
                )
            )
            continue
        # else: unrelated noise line, ignore silently.
    return marks


class PairResult(object):
    def __init__(self):
        self.pairs = []  # list of (state_applied ParsedMark, plan_step_observed ParsedMark, delta_ms)
        self.unpaired_observed = []  # plan_step_observed marks with no unclaimed state_applied available
        self.unclaimed_state_applied = []  # state_applied marks never claimed by any observed mark


def pair_marks(marks):
    """Amendment A4: each plan_step_observed pairs to the EARLIEST unpaired
    state_applied mark. FIFO queue of unclaimed state_applied event_idx/t,
    in file arrival order (NOT sorted/counted by event_idx value -- event_idx
    is monotonic but not contiguous, so pairing must not assume any relation
    between consecutive event_idx values or between the *count* of
    state_applied marks and wakes; it is purely "oldest still-unclaimed
    entry in arrival order")."""
    result = PairResult()
    queue = []  # list of ParsedMark(kind="state_applied"), FIFO

    for mark in marks:
        if mark.kind == "state_applied":
            queue.append(mark)
        elif mark.kind == "plan_step_observed":
            if queue:
                sa = queue.pop(0)
                delta_ms = (mark.t - sa.t) * 1000.0
                result.pairs.append((sa, mark, delta_ms))
            else:
                result.unpaired_observed.append(mark)

    # Whatever is left in the queue was never claimed by any observed mark.
    result.unclaimed_state_applied = list(queue)
    return result


def percentile(sorted_values, pct):
    """Nearest-rank percentile, stdlib only, pct in [0, 100]."""
    if not sorted_values:
        raise ValueError("percentile of empty sequence")
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    k = (pct / 100.0) * (n - 1)
    f = int(k)
    c = min(f + 1, n - 1)
    if f == c:
        return sorted_values[f]
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1


def compute_stats(deltas_ms):
    if not deltas_ms:
        return None
    s = sorted(deltas_ms)
    return {
        "n": len(s),
        "min_ms": s[0],
        "max_ms": s[-1],
        "median_ms": statistics.median(s),
        "p95_ms": percentile(s, 95),
        "p99_ms": percentile(s, 99),
    }


def classify_wakes(pairs):
    """Amendment A3: 3-way wake classification off the plan_step_observed
    mark's wake_kind field. Returns counts dict."""
    counts = {"event": 0, "event_raced_timeout": 0, "timeout": 0}
    for _sa, observed, _delta in pairs:
        wk = observed.wake_kind
        if wk in counts:
            counts[wk] += 1
        # else: unreachable given the regex only matches valid wake_kinds.
    return counts


class TimeoutScoping(object):
    """Amendment A2: classification of every TRUE `timeout` wake into
    backoff-gated (excluded from Gate A's zero) vs request-path (counted)
    vs undetermined (no `task=` field -- pre-this-round data)."""

    def __init__(self):
        self.total_true_timeouts = 0
        self.request_path_true_timeouts = 0
        self.backoff_gated_true_timeouts = 0
        self.undetermined_true_timeouts = 0
        # task_class_name -> count, for the auditable breakdown. The
        # UNDETERMINED bucket (no task= field) is tracked separately under
        # the key None.
        self.by_task_type = {}


def classify_timeout_scoping(pairs):
    """Amendment A2: for every TRUE `timeout` wake (wake_kind == "timeout"),
    classify its `task=` field as backoff-gated, request-path, or
    undetermined (missing field). Non-timeout wakes are irrelevant to A2
    scoping and are not touched here."""
    scoping = TimeoutScoping()
    for _sa, observed, _delta in pairs:
        if observed.wake_kind != "timeout":
            continue
        scoping.total_true_timeouts += 1
        task = observed.task
        scoping.by_task_type[task] = scoping.by_task_type.get(task, 0) + 1
        if task is None:
            scoping.undetermined_true_timeouts += 1
        elif task in _BACKOFF_GATED_TASK_TYPES:
            scoping.backoff_gated_true_timeouts += 1
        else:
            scoping.request_path_true_timeouts += 1
    return scoping


def gate_a_verdict(stats, request_path_timeouts):
    median_ok = stats["median_ms"] <= _GATE_A_MEDIAN_MAX_MS
    p99_ok = stats["p99_ms"] <= _GATE_A_P99_MAX_MS
    timeouts_ok = request_path_timeouts == 0
    overall = median_ok and p99_ok and timeouts_ok
    return {
        "median_ms": stats["median_ms"],
        "median_bound_ms": _GATE_A_MEDIAN_MAX_MS,
        "median_pass": median_ok,
        "p99_ms": stats["p99_ms"],
        "p99_bound_ms": _GATE_A_P99_MAX_MS,
        "p99_pass": p99_ok,
        "request_path_timeout_wakes": request_path_timeouts,
        "request_path_timeout_pass": timeouts_ok,
        "overall_pass": overall,
    }


def baseline_fingerprint_check(stats):
    median_match = _BASELINE_MEDIAN_LO_MS <= stats["median_ms"] <= _BASELINE_MEDIAN_HI_MS
    p95_match = _BASELINE_P95_LO_MS <= stats["p95_ms"] <= _BASELINE_P95_HI_MS
    return {
        "median_ms": stats["median_ms"],
        "median_expected_lo_ms": _BASELINE_MEDIAN_LO_MS,
        "median_expected_hi_ms": _BASELINE_MEDIAN_HI_MS,
        "median_match": median_match,
        "p95_ms": stats["p95_ms"],
        "p95_expected_lo_ms": _BASELINE_P95_LO_MS,
        "p95_expected_hi_ms": _BASELINE_P95_HI_MS,
        "p95_match": p95_match,
        "fingerprint_matches_baseline": median_match and p95_match,
    }


def analyze_source(source_name, lines):
    """Analyze one node's mark stream in isolation. Returns (report_text,
    json_summary_dict). Raises ValueError if zero parseable marks found --
    caller must treat that as a hard failure (non-zero exit), never a
    silent empty pass."""
    marks = parse_lines(lines)
    if not marks:
        raise ValueError(
            "0 parseable PHASE_MARK lines found in source '%s' -- refusing to "
            "compute a Gate A verdict from an empty set." % source_name
        )

    n_state_applied = sum(1 for m in marks if m.kind == "state_applied")
    n_plan_observed = sum(1 for m in marks if m.kind == "plan_step_observed")

    pair_result = pair_marks(marks)
    deltas_ms = [d for (_sa, _obs, d) in pair_result.pairs]
    stats = compute_stats(deltas_ms)
    if stats is None:
        # No plan_step_observed mark could be paired to any state_applied
        # mark at all -- e.g. a stream of only unmatched state_applied
        # lines, or only unpaired plan_step_observed lines. Still a valid
        # (if degenerate) parse of marks, but there is no Gate-A delta
        # distribution to report a verdict against.
        raise ValueError(
            "0 paired (state_applied, plan_step_observed) deltas in source "
            "'%s' -- %d state_applied and %d plan_step_observed marks were "
            "parsed but none could be paired; refusing to compute a Gate A "
            "verdict from an empty paired set."
            % (source_name, n_state_applied, n_plan_observed)
        )

    wake_counts = classify_wakes(pair_result.pairs)

    # A2: classify every TRUE timeout wake by task type using the `task=`
    # field. An UNDETERMINED true timeout (no task= field) is NOT silently
    # treated as request-path OR backoff-gated -- it blocks a clean Gate-A
    # pass on the timeout sub-condition, same as a real request-path
    # timeout would, because the true classification is unknown and Gate
    # A's zero cannot be certified clean while it is.
    scoping = classify_timeout_scoping(pair_result.pairs)
    a2_undetermined = scoping.undetermined_true_timeouts > 0
    # The number that actually gates Gate A's third sub-condition: known
    # request-path timeouts PLUS any undetermined ones (since an
    # undetermined timeout might be request-path -- it cannot be presumed
    # clean).
    gate_a_relevant_timeout_count = (
        scoping.request_path_true_timeouts + scoping.undetermined_true_timeouts
    )

    lines_out = []
    lines_out.append("=" * 78)
    lines_out.append("Gate A Analysis: %s" % source_name)
    lines_out.append("=" * 78)
    lines_out.append("")
    lines_out.append("-- Mark counts --")
    lines_out.append("state_applied marks parsed:      %d" % n_state_applied)
    lines_out.append("plan_step_observed marks parsed: %d" % n_plan_observed)
    lines_out.append("")
    lines_out.append("-- Pairing (amendment A4: earliest-unpaired state_applied) --")
    lines_out.append("paired deltas (n):                              %d" % stats["n"])
    lines_out.append(
        "plan_step_observed marks NOT paired (no unclaimed state_applied): %d"
        % len(pair_result.unpaired_observed)
    )
    lines_out.append(
        "state_applied marks NEVER observed (unclaimed at end of stream):  %d"
        % len(pair_result.unclaimed_state_applied)
    )
    lines_out.append("")
    lines_out.append("-- Gate-A paired delta statistics (ms) --")
    lines_out.append("n:      %d" % stats["n"])
    lines_out.append("min:    %.3f ms" % stats["min_ms"])
    lines_out.append("median: %.3f ms" % stats["median_ms"])
    lines_out.append("p95:    %.3f ms" % stats["p95_ms"])
    lines_out.append("p99:    %.3f ms" % stats["p99_ms"])
    lines_out.append("max:    %.3f ms" % stats["max_ms"])
    lines_out.append("")
    lines_out.append("-- Wake classification (amendment A3, 3-way) --")
    lines_out.append("event:                %d" % wake_counts["event"])
    lines_out.append("event_raced_timeout:  %d  (event-driven; does NOT count against Gate A zero)" % wake_counts["event_raced_timeout"])
    lines_out.append("timeout (true):       %d  (only these count as timeout-driven)" % wake_counts["timeout"])
    lines_out.append("")
    lines_out.append("-- Timeout scoping (amendment A2: CreateRunner/DownloadModel excluded) --")
    lines_out.append("total true-timeout wakes:                     %d" % scoping.total_true_timeouts)
    lines_out.append("  backoff-gated (excluded from Gate A zero):  %d" % scoping.backoff_gated_true_timeouts)
    lines_out.append("  request-path (counts against Gate A zero):  %d" % scoping.request_path_true_timeouts)
    lines_out.append("  UNDETERMINED (no task= field):               %d" % scoping.undetermined_true_timeouts)
    lines_out.append("")
    lines_out.append("  -- per-task-type true-timeout breakdown (auditable against PREDICTION.md) --")
    for task_name in sorted(scoping.by_task_type.keys(), key=lambda k: (k is None, k)):
        count = scoping.by_task_type[task_name]
        if task_name is None:
            label = "<UNDETERMINED: no task= field>"
        elif task_name in _BACKOFF_GATED_TASK_TYPES:
            label = "%s (BACKOFF-GATED, excluded)" % task_name
        else:
            label = "%s (REQUEST-PATH, counted)" % task_name
        lines_out.append("    %-45s %d" % (label, count))
    lines_out.append("")
    if a2_undetermined:
        lines_out.append("!" * 78)
        lines_out.append(
            "UNDETERMINED: %d true-timeout mark(s) have no `task=` field and "
            "cannot be scoped." % scoping.undetermined_true_timeouts
        )
        lines_out.append("Reason: %s" % _TASK_TYPE_FIELD_MISSING)
        lines_out.append(
            "These are NOT presumed request-path or backoff-gated; they are "
            "conservatively counted against Gate A's zero (see "
            "gate-A-relevant count below) so old-format data can never "
            "silently produce a clean Gate-A verdict."
        )
        lines_out.append("!" * 78)
    lines_out.append(
        "Gate-A-relevant request-path true-timeout count (request-path + "
        "undetermined): %d" % gate_a_relevant_timeout_count
    )
    lines_out.append("")

    gate_a = gate_a_verdict(stats, gate_a_relevant_timeout_count)
    lines_out.append("-- Gate A verdict (pre-registered bands, computed mechanically) --")
    lines_out.append(
        "median <= %.1f ms:  measured %.3f ms  -> %s"
        % (_GATE_A_MEDIAN_MAX_MS, gate_a["median_ms"], "PASS" if gate_a["median_pass"] else "FAIL")
    )
    lines_out.append(
        "p99 <= %.1f ms:     measured %.3f ms  -> %s"
        % (_GATE_A_P99_MAX_MS, gate_a["p99_ms"], "PASS" if gate_a["p99_pass"] else "FAIL")
    )
    lines_out.append(
        "request-path timeout-driven wakes == 0: measured %d%s  -> %s"
        % (
            gate_a["request_path_timeout_wakes"],
            " (includes UNDETERMINED marks, conservatively counted)" if a2_undetermined else "",
            "PASS" if gate_a["request_path_timeout_pass"] else "FAIL",
        )
    )
    lines_out.append("")
    lines_out.append(
        "GATE A OVERALL: %s%s"
        % (
            "PASS" if gate_a["overall_pass"] else "FAIL",
            " (** blocked by UNDETERMINED-scoped true-timeout mark(s); "
            "resolve by re-capturing with the current instrumentation **)"
            if a2_undetermined and not gate_a["overall_pass"]
            else "",
        )
    )
    lines_out.append("")

    baseline = baseline_fingerprint_check(stats)
    lines_out.append("-- Baseline (flag-OFF) fingerprint check (informational) --")
    lines_out.append(
        "median in [%.0f, %.0f] ms: measured %.3f ms -> %s"
        % (
            _BASELINE_MEDIAN_LO_MS,
            _BASELINE_MEDIAN_HI_MS,
            baseline["median_ms"],
            "MATCH" if baseline["median_match"] else "NO MATCH",
        )
    )
    lines_out.append(
        "p95 in [%.0f, %.0f] ms:    measured %.3f ms -> %s"
        % (
            _BASELINE_P95_LO_MS,
            _BASELINE_P95_HI_MS,
            baseline["p95_ms"],
            "MATCH" if baseline["p95_match"] else "NO MATCH",
        )
    )
    lines_out.append(
        "baseline fingerprint overall: %s"
        % ("MATCH" if baseline["fingerprint_matches_baseline"] else "NO MATCH")
    )
    lines_out.append("")

    report_text = "\n".join(lines_out)

    json_summary = {
        "source": source_name,
        "counts": {
            "state_applied_marks_parsed": n_state_applied,
            "plan_step_observed_marks_parsed": n_plan_observed,
            "paired_n": stats["n"],
            "plan_step_observed_unpaired": len(pair_result.unpaired_observed),
            "state_applied_never_observed": len(pair_result.unclaimed_state_applied),
        },
        "stats_ms": stats,
        "wake_classification": wake_counts,
        "timeout_scoping": {
            "total_true_timeout_wakes": scoping.total_true_timeouts,
            "backoff_gated_true_timeout_wakes": scoping.backoff_gated_true_timeouts,
            "request_path_true_timeout_wakes": scoping.request_path_true_timeouts,
            "undetermined_true_timeout_wakes": scoping.undetermined_true_timeouts,
            "gate_a_relevant_true_timeout_wakes": gate_a_relevant_timeout_count,
            "by_task_type": dict(
                ("<UNDETERMINED>" if k is None else k, v)
                for k, v in scoping.by_task_type.items()
            ),
            "undetermined": a2_undetermined,
            "undetermined_reason": _TASK_TYPE_FIELD_MISSING if a2_undetermined else None,
        },
        "gate_a": gate_a,
        "baseline_fingerprint": baseline,
    }
    return report_text, json_summary


def _write_json_summary(input_path, json_summary):
    out_dir = dirname(input_path) or "."
    out_name = basename(input_path) + ".analysis.json"
    out_path = join(out_dir, out_name)
    with open(out_path, "w") as f:
        json.dump(json_summary, f, indent=2, sort_keys=True)
    return out_path


def main(argv):
    if len(argv) <= 1:
        # stdin mode: single anonymous source.
        raw = sys.stdin.read()
        lines = raw.splitlines()
        try:
            report_text, json_summary = analyze_source("<stdin>", lines)
        except ValueError as e:
            sys.stderr.write("FATAL: %s\n" % e)
            return 1
        print(report_text)
        # No file path to write JSON alongside for stdin; write to cwd.
        out_path = "stdin.analysis.json"
        with open(out_path, "w") as f:
            json.dump(json_summary, f, indent=2, sort_keys=True)
        print("JSON summary written to: %s" % out_path)
        return 0

    exit_code = 0
    for input_path in argv[1:]:
        try:
            with open(input_path, "r") as f:
                lines = f.read().splitlines()
        except OSError as e:
            sys.stderr.write("FATAL: could not read '%s': %s\n" % (input_path, e))
            exit_code = 1
            continue

        try:
            report_text, json_summary = analyze_source(input_path, lines)
        except ValueError as e:
            sys.stderr.write("FATAL: %s\n" % e)
            exit_code = 1
            continue

        print(report_text)
        out_path = _write_json_summary(input_path, json_summary)
        print("JSON summary written to: %s" % out_path)
        print("")

    return exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
