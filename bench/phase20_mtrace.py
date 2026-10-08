#!/usr/bin/env python3
"""bench/phase20_mtrace.py - Metal System Trace capture / export / analysis.

Phase-20, BRIEF M.  Gives the GPU's OWN kernel / command-buffer timeline of a
decode round from OUTSIDE the process: no code change, no relaunch.  This is
the plan's Fallback-B tool for gate 0d (when powermetrics residency is within
5 points of idle -> wrong GPU/process) and the independent cross-check of the
Phase-1 host timer.

Subcommands
-----------
  record  --node N --pid P --secs S --out /tmp/x.trace [--dry-run]
          Build (and unless --dry-run, run over ssh) the ``xcrun xctrace
          record --attach`` command.  A HARD ``timeout`` wraps the remote
          command.  NEVER point this at the exo runner yourself unless the PM
          says so -- see MTRACE-NOTES.md.
  export  --trace PATH [--node N] --out DIR [--toc] [--schemas ...]
          Export the GPU table schemas to XML (locally, or via ssh + scp when
          --node is given).  ``--toc`` dumps the table of contents only.
  analyze --dir DIR [--window-start-ns A] [--window-end-ns B]
          [--round-gap-ms 3] [--json OUT.json] [--md OUT.md]
          Stream-parse the exported XML and print JSON + markdown:
          GPU-busy fraction over the window, command-buffer count/rate,
          interval stats, idle-gap histogram (>50 us), a heuristic
          segmentation into "rounds", and the top-10 longest GPU intervals.
  selftest
          Parse the bundled fixtures + synthetic edge cases, assert known
          numbers.  Pure-parse: no xctrace, no cluster.

Data model (learned from ``xcrun xctrace export --toc`` on xctrace 27.0,
Metal System Trace template; see MTRACE-NOTES.md for the verbatim excerpt)
-------------------------------------------------------------------------
* A table element is ``<table schema="...">`` with ordered ``<col><mnemonic>``
  children.  Every ``<row>`` has children in the SAME order as the cols.
* A row child either carries inline text, or an ``id`` (first sighting) / a
  ``ref`` pointing at a previously declared ``id``.  Resolve refs positionally.
* ``start-time`` and ``duration`` engineering-types are integer NANOSECONDS
  (start fmt 00:00.782.830 <-> value 782830041 ns).
* GPU busy timeline = ``metal-gpu-state-intervals`` rows with state Active;
  Idle rows partition the complement.  Both may be split per channel -> merge
  (union) before summing.
* Per-encoder GPU execution intervals = ``metal-gpu-intervals``
  (channel-name Compute/Render, event-label, cmdbuffer-id, encoder-id,
  gpu-submission-id, start-latency = CPU->GPU latency in ns).

Everything is stdlib; XML is parsed with ``xml.etree.ElementTree.iterparse``
so a 100s-of-MB export does not have to fit in memory.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: schemas we export / know how to read, in preference order.
DEFAULT_SCHEMAS = [
    "metal-gpu-state-intervals",
    "metal-gpu-intervals",
    "metal-application-command-buffer-submissions",
    "gpu-performance-state-intervals",
    "metal-command-buffer-completed",
]

#: schemas whose rows are (start, duration, state/Active|Idle) GPU state.
GPU_STATE_SCHEMA = "metal-gpu-state-intervals"
#: schemas whose rows are per-encoder GPU execution intervals.
GPU_INTERVAL_SCHEMA = "metal-gpu-intervals"
#: command-buffer submission rows.
CMD_BUFFER_SCHEMA = "metal-application-command-buffer-submissions"
#: GPU performance-state intervals.
PERF_STATE_SCHEMA = "gpu-performance-state-intervals"

XPATH_TMPL = (
    '/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]'
)

#: documented expected trace size per second of Metal System Trace, for the
#: record subcommand's "expected size" note (measured on a 200-batch toy).
BYTES_PER_SEC = (30 * 1024 * 1024, 90 * 1024 * 1024)  # ~30-90 MB/s on M4 Max


# --------------------------------------------------------------------------
# XML streaming parser
# --------------------------------------------------------------------------

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def stream_rows(path: str):
    """Yield ``(schema, cols, row_dict)`` for every row in an exported XML.

    Streaming: ``iterparse`` with ``element.clear()`` after each row, so peak
    memory is one row plus the col list.  ``cols`` is the ordered mnemonic
    list of the enclosing schema; a dict is rebuilt for every row (the schema
    is repeated per row in the source, so this is exact, not global state).
    """
    cur_cols: list[str] = []
    cur_schema = ""
    ids: dict[str, str] = {}
    # We need cols before rows, but a table can repeat <schema> per row.  Keep
    # the most recent col list and reset it whenever a new <col> run starts.
    for event, el in ET.iterparse(path, events=("start", "end")):
        tag = _local(el.tag)
        if event == "start":
            if tag == "schema":
                cur_schema = el.get("name", "")
                cur_cols = []
            elif tag == "col":
                m = el.find("mnemonic")
                cur_cols.append(m.text if m is not None and m.text else f"col{len(cur_cols) + 1}")
        else:  # end
            if tag == "row":
                row: dict[str, str | None] = {}
                for idx, c in enumerate(el):
                    if idx >= len(cur_cols):
                        break
                    key = cur_cols[idx]
                    ref = c.get("ref")
                    if ref is not None:
                        row[key] = ids.get(ref)
                    else:
                        val = c.text if c.text is not None else c.get("fmt")
                        row[key] = val
                        cid = c.get("id")
                        if cid is not None and val is not None:
                            ids[cid] = val
                yield cur_schema, list(cur_cols), row
                el.clear()


def _int(v) -> int:
    if v is None:
        return 0
    if isinstance(v, int):
        return v
    s = str(v).strip()
    if not s:
        return 0
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except ValueError:
            return 0


# --------------------------------------------------------------------------
# Interval helpers
# --------------------------------------------------------------------------

def merge_intervals(iv: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Union overlapping/adjacent [start, end) intervals -> sorted disjoint."""
    if not iv:
        return []
    iv = sorted(iv)
    out = [list(iv[0])]
    for s, e in iv[1:]:
        if s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def clip_intervals(iv: list[tuple[int, int]], a: int, b: int) -> list[tuple[int, int]]:
    """Clip intervals to the closed window [a, b)."""
    return [(max(s, a), min(e, b)) for s, e in iv if min(e, b) > max(s, a)]


def total_len(iv: list[tuple[int, int]]) -> int:
    return sum(e - s for s, e in iv)


def gaps_between(iv: list[tuple[int, int]]) -> list[int]:
    """Gaps between consecutive (already merged, sorted) intervals."""
    return [iv[i + 1][0] - iv[i][1] for i in range(len(iv) - 1)]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

class Trace:
    """Parsed, compacted contents of an export directory (or one file)."""

    def __init__(self) -> None:
        self.active: list[tuple[int, int]] = []      # merged Active union
        self.idle: list[tuple[int, int]] = []        # merged Idle union
        self.gpu_intervals: list[dict] = []          # per-encoder rows (compact)
        self.cmd_submissions: list[dict] = []        # submission rows (compact)
        self.perf_states: list[dict] = []            # perf-state rows
        self.schemas_seen: set[str] = set()
        self.source_files: list[str] = []

    # -- window ------------------------------------------------------------
    def active_span(self) -> tuple[int, int] | None:
        if not self.active:
            return None
        return self.active[0][0], self.active[-1][1]

    def busy_over(self, a: int, b: int) -> int:
        return total_len(clip_intervals(self.active, a, b))


def _iter_xml_files(path: str) -> list[str]:
    if os.path.isdir(path):
        out = []
        for name in sorted(os.listdir(path)):
            if name.endswith(".xml"):
                out.append(os.path.join(path, name))
        return out
    return [path]


def load_trace(path: str) -> Trace:
    tr = Trace()
    for f in _iter_xml_files(path):
        tr.source_files.append(f)
        act: list[tuple[int, int]] = []
        idl: list[tuple[int, int]] = []
        for schema, cols, row in stream_rows(f):
            tr.schemas_seen.add(schema)
            if schema == GPU_STATE_SCHEMA:
                s = _int(row.get("start"))
                d = _int(row.get("duration"))
                state = (row.get("state") or "").strip()
                if state.lower() == "active":
                    act.append((s, s + d))
                elif state.lower() == "idle":
                    idl.append((s, s + d))
            elif schema == GPU_INTERVAL_SCHEMA:
                d = _int(row.get("duration"))
                tr.gpu_intervals.append({
                    "start": _int(row.get("start")),
                    "duration": d,
                    "channel": row.get("channel-name"),
                    "label": row.get("event-label"),
                    "cmdbuffer_id": row.get("cmdbuffer-id"),
                    "encoder_id": row.get("encoder-id"),
                    "submission_id": row.get("gpu-submission-id"),
                    "start_latency": _int(row.get("start-latency")),
                })
            elif schema == CMD_BUFFER_SCHEMA:
                tr.cmd_submissions.append({
                    "start": _int(row.get("start")),
                    "duration": _int(row.get("duration")),
                    "event_type": row.get("event-type"),
                    "num_encoders": _int(row.get("num-encoders")),
                    "encoder_time": _int(row.get("encoder-time")),
                    "label": row.get("event-label"),
                })
            elif schema == PERF_STATE_SCHEMA:
                tr.perf_states.append({
                    "start": _int(row.get("start")),
                    "duration": _int(row.get("duration")),
                    "state": row.get("gpu-performance-state"),
                    "induced": row.get("is-induced"),
                })
        tr.active = merge_intervals(tr.active + act)
        tr.idle = merge_intervals(tr.idle + idl)
    return tr


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

def _gap_histogram(gaps_ns: list[int]) -> list[dict]:
    """Buckets for gaps strictly greater than 50 us (the >50us histogram)."""
    edges = [100_000, 500_000, 1_000_000, 3_000_000, 10_000_000]
    labels = ["50-100us", "100-500us", "0.5-1ms", "1-3ms", "3-10ms", ">10ms"]
    counts = [0] * len(labels)
    for g in gaps_ns:
        if g <= 50_000:
            continue
        idx = len(labels) - 1
        for i, hi in enumerate(edges):
            if g <= hi:
                idx = i
                break
        counts[idx] += 1
    return [{"bucket": lbl, "count": c} for lbl, c in zip(labels, counts)]


def _rounds(active: list[tuple[int, int]], round_gap_ns: int) -> list[dict]:
    """Segment the merged active timeline into rounds split by gaps >=
    round_gap_ns.  Per round: busy_ms, gap_ms (internal + trailing), start."""
    if not active:
        return []
    rounds = []
    cur_start = active[0][0]
    cur_busy = active[0][1] - active[0][0]
    cur_gap = 0
    n = 1
    for i in range(1, len(active)):
        g = active[i][0] - active[i - 1][1]
        if g >= round_gap_ns:
            rounds.append({
                "start_ns": cur_start,
                "n_bursts": n,
                "busy_ms": cur_busy / 1e6,
                "gap_ms": cur_gap / 1e6,
            })
            cur_start = active[i][0]
            cur_busy = active[i][1] - active[i][0]
            cur_gap = 0
            n = 1
        else:
            cur_busy += active[i][1] - active[i][0]
            cur_gap += g
            n += 1
    rounds.append({
        "start_ns": cur_start,
        "n_bursts": n,
        "busy_ms": cur_busy / 1e6,
        "gap_ms": cur_gap / 1e6,
    })
    return rounds


def analyze(
    tr: Trace,
    window_start: int | None = None,
    window_end: int | None = None,
    round_gap_ms: float = 3.0,
) -> dict:
    span = tr.active_span()
    if span is None:
        return {
            "ok": False,
            "reason": "no Active GPU-state intervals found",
            "schemas_seen": sorted(tr.schemas_seen),
            "source_files": tr.source_files,
        }
    a = span[0] if window_start is None else window_start
    b = span[1] if window_end is None else window_end
    if b <= a:
        raise ValueError(f"empty analysis window: start={a} end={b}")

    active_win = clip_intervals(tr.active, a, b)
    busy_ns = total_len(active_win)
    window_ns = b - a
    gaps = gaps_between(active_win)

    # per-encoder interval stats (clip to window)
    ivs = [x for x in tr.gpu_intervals if x["duration"] > 0]
    ivs_win = [
        x for x in ivs
        if min(x["start"] + x["duration"], b) > max(x["start"], a)
    ]
    durs = sorted(x["duration"] for x in ivs_win)
    dur_stats = None
    if durs:
        dur_stats = {
            "count": len(durs),
            "total_ms": sum(durs) / 1e6,
            "mean_ms": statistics.mean(durs) / 1e6,
            "median_ms": statistics.median(durs) / 1e6,
            "p90_ms": durs[int(0.9 * (len(durs) - 1))] / 1e6,
            "max_ms": durs[-1] / 1e6,
        }

    # command-buffer rate: distinct cmdbuffer ids over the window
    cb_ids = {x["cmdbuffer_id"] for x in ivs_win if x["cmdbuffer_id"] is not None}
    cb_rate = (len(cb_ids) / (window_ns / 1e9)) if window_ns else None

    subs_win = [s for s in tr.cmd_submissions if a <= s["start"] < b]
    sub_rate = (len(subs_win) / (window_ns / 1e9)) if window_ns else None

    top = sorted(ivs_win, key=lambda x: x["duration"], reverse=True)[:10]
    top10 = [
        {
            "label": t["label"],
            "channel": t["channel"],
            "duration_ms": t["duration"] / 1e6,
            "cmdbuffer_id": t["cmdbuffer_id"],
        }
        for t in top
    ]

    rounds = _rounds(active_win, int(round_gap_ms * 1e6))

    # perf-state distribution weighted by duration within window
    perf = Counter()
    for p in tr.perf_states:
        s = max(p["start"], a)
        e = min(p["start"] + p["duration"], b)
        if e > s:
            perf[p["state"] or "?"] += e - s

    out = {
        "ok": True,
        "source_files": tr.source_files,
        "schemas_seen": sorted(tr.schemas_seen),
        "window": {
            "start_ns": a,
            "end_ns": b,
            "span_ms": window_ns / 1e6,
            "auto": window_start is None and window_end is None,
        },
        "gpu_busy": {
            "busy_ms": busy_ns / 1e6,
            "idle_ms": (window_ns - busy_ns) / 1e6,
            "busy_fraction": busy_ns / window_ns,
        },
        "active_intervals": len(active_win),
        "idle_gaps": {
            "count": len(gaps),
            "gt_50us_count": sum(1 for g in gaps if g > 50_000),
            "median_ms": (statistics.median(gaps) / 1e6) if gaps else None,
            "max_ms": (max(gaps) / 1e6) if gaps else None,
            "histogram_gt_50us": _gap_histogram(gaps),
        },
        "encoder_intervals": dur_stats,
        "command_buffers": {
            "distinct_ids": len(cb_ids),
            "rate_per_s": cb_rate,
            "submissions_in_window": len(subs_win),
            "submission_rate_per_s": sub_rate,
        },
        "perf_state_ms": {k: v / 1e6 for k, v in perf.most_common()},
        "rounds": {
            "round_gap_ms": round_gap_ms,
            "count": len(rounds),
            "busy_ms_median": (statistics.median(r["busy_ms"] for r in rounds) if rounds else None),
            "gap_ms_median": (statistics.median(r["gap_ms"] for r in rounds) if rounds else None),
            "detail": rounds[:50],
            "detail_truncated": len(rounds) > 50,
        },
        "top10_intervals": top10,
    }
    return out


# --------------------------------------------------------------------------
# Markdown renderer
# --------------------------------------------------------------------------

def render_md(res: dict) -> str:
    if not res.get("ok"):
        return f"# Metal System Trace analysis\n\n**FAILED**: {res.get('reason')}\n"
    L = ["# Metal System Trace analysis", ""]
    w = res["window"]
    L.append(f"- Window: {w['start_ns']} .. {w['end_ns']} ns "
             f"({w['span_ms']:.3f} ms{', auto' if w['auto'] else ''})")
    g = res["gpu_busy"]
    L.append(f"- **GPU-busy fraction: {g['busy_fraction']*100:.2f}%** "
             f"(busy {g['busy_ms']:.3f} ms / idle {g['idle_ms']:.3f} ms)")
    L.append(f"- Active intervals: {res['active_intervals']}")
    L.append("")
    L.append("## Idle gaps (>50 us)")
    gp = res["idle_gaps"]
    L.append(f"- gaps: {gp['count']}, >50us: {gp['gt_50us_count']}, "
             f"median {gp['median_ms']} ms, max {gp['max_ms']} ms")
    for h in gp["histogram_gt_50us"]:
        L.append(f"  - {h['bucket']}: {h['count']}")
    L.append("")
    e = res["encoder_intervals"]
    if e:
        L.append("## Encoder intervals")
        L.append(f"- n={e['count']} total {e['total_ms']:.3f} ms; "
                 f"mean {e['mean_ms']:.4f} / median {e['median_ms']:.4f} / "
                 f"p90 {e['p90_ms']:.4f} / max {e['max_ms']:.4f} ms")
        L.append("")
    cb = res["command_buffers"]
    L.append("## Command buffers")
    L.append(f"- distinct ids {cb['distinct_ids']}, rate {cb['rate_per_s']} /s; "
             f"submissions {cb['submissions_in_window']} "
             f"({cb['submission_rate_per_s']} /s)")
    L.append("")
    r = res["rounds"]
    L.append(f"## Rounds (gap>={r['round_gap_ms']} ms)")
    L.append(f"- count {r['count']}, busy median {r['busy_ms_median']} ms, "
             f"gap median {r['gap_ms_median']} ms")
    for i, rd in enumerate(r["detail"][:10]):
        L.append(f"  - round {i}: busy {rd['busy_ms']:.3f} ms, "
                 f"gap {rd['gap_ms']:.3f} ms, {rd['n_bursts']} bursts")
    L.append("")
    L.append("## Top-10 longest GPU intervals")
    for t in res["top10_intervals"]:
        L.append(f"- {t['duration_ms']:.4f} ms [{t['channel']}] {t['label']}")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------
# record / export
# --------------------------------------------------------------------------

def build_record_cmd(node: str, pid: str, secs: float, out: str) -> list[str]:
    inner = (
        f"/usr/bin/xcrun xctrace record --template 'Metal System Trace' "
        f"--attach {pid} --time-limit {secs:g}s --no-prompt --output {shlex.quote(out)}"
    )
    if node in ("local", "", "laptop", None):
        return ["bash", "-lc", inner]
    # macOS has NO coreutils `timeout` (verified on studio2: `zsh: command not
    # found: timeout`).  Use /usr/bin/perl's alarm as a portable hard cap --
    # `--time-limit` already bounds the recording, this is belt-and-braces for
    # a wedged attach.  perl ships with macOS.
    hard = int(secs) + 45
    wrapped = (
        f"/usr/bin/perl -e 'alarm shift; exec @ARGV' {hard} "
        f"/usr/bin/xcrun xctrace record --template 'Metal System Trace' "
        f"--attach {pid} --time-limit {secs:g}s --no-prompt --output {shlex.quote(out)}"
    )
    return ["ssh", node, wrapped]


def cmd_record(a) -> int:
    secs = a.secs
    lo, hi = BYTES_PER_SEC
    print(f"# record: node={a.node} pid={a.pid} secs={secs:g} out={a.out}")
    print(f"# expected trace size ~{lo//(1024*1024)}-{hi//(1024*1024)} MB/s "
          f"-> ~{int(lo*secs)//(1024*1024)}-{int(hi*secs)//(1024*1024)} MB for {secs:g}s")
    print(f"# overhead: tracing perturbs the target; measure tok/s with and "
          f"without (see MTRACE-NOTES.md)")
    cmd = build_record_cmd(a.node, a.pid, secs, a.out)
    print("# command:")
    print("  " + " ".join(cmd))
    print("# (argv: %r)" % (cmd,))
    if a.dry_run:
        print("# --dry-run: not executing")
        return 0
    hard = int(secs) + 90
    print(f"# hard local timeout {hard}s")
    try:
        r = subprocess.run(cmd, timeout=hard)
        return r.returncode
    except subprocess.TimeoutExpired:
        print(f"# TIMEOUT after {hard}s", file=sys.stderr)
        return 124


def cmd_export(a) -> int:
    schemas = a.schemas or DEFAULT_SCHEMAS
    os.makedirs(a.out, exist_ok=True)
    results = []
    for s in schemas:
        out = os.path.join(a.out, f"{s}.xml")
        xp = XPATH_TMPL.format(schema=s)
        inner = (
            f"/usr/bin/xcrun xctrace export --input {shlex.quote(a.trace)} "
            f"--xpath {shlex.quote(xp)} --output {shlex.quote(out)}"
        )
        if a.node:
            tmp = f"/tmp/p20_{s}.xml"
            inner = (
                f"/usr/bin/xcrun xctrace export --input {shlex.quote(a.trace)} "
                f"--xpath {shlex.quote(xp)} --output {shlex.quote(tmp)}"
            )
            print(f"# exporting {s} on {a.node} -> {tmp}")
            subprocess.run(["ssh", a.node, inner], timeout=1800)
            print(f"# scp {a.node}:{tmp} -> {out}")
            subprocess.run(["scp", f"{a.node}:{tmp}", out], timeout=1800)
        else:
            print(f"# exporting {s} -> {out}")
            try:
                subprocess.run(["bash", "-lc", inner], timeout=1800)
            except subprocess.TimeoutExpired:
                print(f"# TIMEOUT exporting {s}", file=sys.stderr)
        results.append((s, out, os.path.getsize(out) if os.path.exists(out) else None))
    print("# export results (schema, path, bytes):")
    for s, o, n in results:
        print(f"  {s}\t{o}\t{n}")
    return 0


def cmd_toc(a) -> int:
    inner = (f"/usr/bin/xcrun xctrace export --input {shlex.quote(a.trace)} "
             f"--toc --output {shlex.quote(a.out)}")
    if a.node:
        tmp = "/tmp/p20_toc.xml"
        subprocess.run(["ssh", a.node,
                        inner.replace(shlex.quote(a.out), shlex.quote(tmp))], timeout=1800)
        subprocess.run(["scp", f"{a.node}:{tmp}", a.out], timeout=1800)
    else:
        subprocess.run(["bash", "-lc", inner], timeout=1800)
    print(f"# toc written to {a.out}")
    return 0


# --------------------------------------------------------------------------
# analyze CLI
# --------------------------------------------------------------------------

def cmd_analyze(a) -> int:
    tr = load_trace(a.dir)
    res = analyze(tr, a.window_start_ns, a.window_end_ns, a.round_gap_ms)
    text = json.dumps(res, indent=2)
    if a.json:
        with open(a.json, "w") as f:
            f.write(text)
    print(text)
    if a.md:
        with open(a.md, "w") as f:
            f.write(render_md(res))
        print(f"# markdown written to {a.md}")
    else:
        print("\n" + render_md(res))
    return 0 if res.get("ok") else 2


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------

FIXDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "phase20_tests", "fixtures", "mtrace")


def _synth_write(path: str, cols: list[tuple[str, str]], rows: list[list],
                 schema: str = GPU_STATE_SCHEMA) -> None:
    """Write a minimal xctrace-shaped XML for tests."""
    parts = ['<?xml version="1.0"?>', "<trace-query-result>", "<node>",
             f'<schema name="{schema}">']
    for mnem, etype in cols:
        parts.append(f"<col><mnemonic>{mnem}</mnemonic>"
                     f"<engineering-type>{etype}</engineering-type></col>")
    parts.append("</schema>")
    for r in rows:
        cells = []
        for v in r:
            if v is None:
                cells.append("<sentinel/>")
            elif isinstance(v, int):
                cells.append(f"<start-time fmt=\"{v}\">{v}</start-time>")
            else:
                cells.append(f"<string fmt=\"{v}\">{v}</string>")
        parts.append("<row>" + "".join(cells) + "</row>")
    parts.append("</node></trace-query-result>")
    with open(path, "w") as f:
        f.write("\n".join(parts))


def _selftest_checks() -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []

    def ck(name, cond, detail=""):
        checks.append((name, bool(cond), detail))

    # --- fixture parse -------------------------------------------------
    fx = os.path.join(FIXDIR, "gpu_state_intervals_trim.xml")
    if os.path.exists(fx):
        tr = load_trace(fx)
        res = analyze(tr)
        ck("fixture state interval parse", tr.active_span() is not None,
           f"active_span={tr.active_span()}")
        ck("fixture busy fraction in (0,1)",
           0 < res["gpu_busy"]["busy_fraction"] < 1,
           f"{res['gpu_busy']['busy_fraction']}")
    else:
        ck("fixture gpu_state_intervals_trim.xml present", False, fx)

    fi = os.path.join(FIXDIR, "gpu_intervals_trim.xml")
    if os.path.exists(fi):
        tr = load_trace(fi)
        ck("fixture gpu-intervals parse", len(tr.gpu_intervals) >= 20,
           f"n={len(tr.gpu_intervals)}")
    else:
        ck("fixture gpu_intervals_trim.xml present", False, fi)

    # --- synthetic edge cases -----------------------------------------
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        # empty table
        p = os.path.join(td, "empty.xml")
        _synth_write(p, [("start", "start-time"), ("duration", "duration"),
                         ("state", "gpu-state")], [])
        tr = load_trace(p)
        r = analyze(tr)
        ck("empty table -> ok False", r["ok"] is False, str(r.get("reason")))

        # overlapping intervals must merge before summing
        p = os.path.join(td, "overlap.xml")
        _synth_write(p, [("start", "start-time"), ("duration", "duration"),
                         ("state", "gpu-state")],
                     [[0, 1000, "Active"], [500, 1000, "Active"]])
        tr = load_trace(p)
        ck("overlap merged before summing",
           total_len(tr.active) == 1500, f"{tr.active}")

        # window clipping: an interval crossing an edge contributes only its
        # in-window part
        p = os.path.join(td, "clip.xml")
        _synth_write(p, [("start", "start-time"), ("duration", "duration"),
                         ("state", "gpu-state")],
                     [[0, 100, "Active"], [1000, 100, "Active"]])
        tr = load_trace(p)
        r = analyze(tr, window_start=50, window_end=500)
        ck("window clips crossing intervals",
           abs(r["gpu_busy"]["busy_ms"] - 0.00005) < 1e-9,
           f"busy_ms={r['gpu_busy']['busy_ms']}")

        # multiple queues: two channels Active at the same time -> union
        p = os.path.join(td, "multi.xml")
        _synth_write(p, [("start", "start-time"), ("duration", "duration"),
                         ("state", "gpu-state")],
                     [[0, 1000, "Active"], [0, 1000, "Active"],
                      [0, 1000, "Idle"]])
        tr = load_trace(p)
        ck("multiple queues unioned not summed",
           total_len(tr.active) == 1000, f"{tr.active}")

        # gap histogram + rounds
        p = os.path.join(td, "rounds.xml")
        rows = []
        for k in range(5):                     # 5 bursts of 1 ms, 1 ms gaps
            rows.append([k * 2_000_000, 1_000_000, "Active"])
        _synth_write(p, [("start", "start-time"), ("duration", "duration"),
                         ("state", "gpu-state")], rows)
        tr = load_trace(p)
        r = analyze(tr, round_gap_ms=0.5)
        ck("rounds split at big gaps", r["rounds"]["count"] == 5,
           f"count={r['rounds']['count']}")
        r2 = analyze(tr, round_gap_ms=3.0)
        ck("rounds merge when gap<threshold", r2["rounds"]["count"] == 1,
           f"count={r2['rounds']['count']}")

    return checks


def cmd_selftest(a) -> int:
    checks = _selftest_checks()
    bad = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}   {detail}")
    print(f"# selftest: {len(checks) - len(bad)}/{len(checks)} passed")
    return 1 if bad else 0


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phase20_mtrace", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="build/run the xctrace record --attach cmd")
    r.add_argument("--node", default="local")
    r.add_argument("--pid", required=True)
    r.add_argument("--secs", type=float, default=30.0)
    r.add_argument("--out", default="/tmp/p20.trace")
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(func=cmd_record)

    e = sub.add_parser("export", help="export GPU tables to XML")
    e.add_argument("--trace", required=True)
    e.add_argument("--node", default=None)
    e.add_argument("--out", required=True)
    e.add_argument("--schemas", nargs="*", default=None)
    e.add_argument("--toc", action="store_true")
    e.set_defaults(func=cmd_export)

    an = sub.add_parser("analyze", help="analyze exported XML")
    an.add_argument("--dir", required=True)
    an.add_argument("--window-start-ns", type=int, default=None)
    an.add_argument("--window-end-ns", type=int, default=None)
    an.add_argument("--round-gap-ms", type=float, default=3.0)
    an.add_argument("--json", default=None)
    an.add_argument("--md", default=None)
    an.set_defaults(func=cmd_analyze)

    st = sub.add_parser("selftest", help="pure-parse self test")
    st.set_defaults(func=cmd_selftest)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if getattr(a, "toc", False) and a.cmd == "export":
        return cmd_toc(a)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
