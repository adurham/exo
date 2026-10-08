#!/usr/bin/env python3
"""Phase 0d — GPU-busy fraction of the decode round + host-wait attribution.

Two independent measurements of the SAME decode window on BOTH nodes:

  * ``powermetrics --samplers gpu_power`` -> mean ``GPU HW active residency``
    (weighted by the per-block ``elapsed`` ms) minus the idle baseline =
    **GPU-busy fraction**.  Also the GPU frequency-bin distribution (low
    P-states during decode == latency-bound signature).
  * ``/usr/bin/sample <runner-pid>`` -> a text call graph; for every leaf stack
    path we assign the DEEPEST matching category among {comm, gpu_wait,
    python_busy} and report the fraction of samples in each, per thread
    (Python main thread + all threads).

Decision gate (PREREG 0d, frozen):
  * GPU-busy >= 90% both nodes AND host-Python < 5%  => GPU-serialized
    (Mode-2 matters most).
  * GPU-busy < 85% either node                     => host/comm bound
    (Phase-1 timer mandatory).
  * node gap > 10 points                          => one rank waiting on the
    other (jaccl / load imbalance).
  * falsifier: residency within 5 points of idle  => wrong GPU/process; stop,
    use Fallback B (``xcrun xctrace`` Metal System Trace attach 30 s).

READ-ONLY tool.  It never POSTs a generation request, never starts/stops a
node process.  The live ``decode`` subcommand is intended to be run by the
campaign PM; this worker built and unit-tested everything OFFLINE against the
captured idle fixtures.

Prompt builders / SSE parsing are COPY-ADAPTED (not imported) from
``/private/tmp/levers-wt/bench/phase19_round_measure.py`` (``build_prompt``,
``stream_once``) and ``phase19_agentic_measure.py`` (agentic arm builder).

Usage:
  python bench/phase20_gpu_busy.py parse-pm  FILE [--idle FILE] [--md FILE]
  python bench/phase20_gpu_busy.py parse-sample FILE [--classify-config F] [--md FILE]
  python bench/phase20_gpu_busy.py idle [--secs 50] [--out FILE]
  python bench/phase20_gpu_busy.py decode --workload {benign,agentic} [--dry-run] ...
  python bench/phase20_gpu_busy.py report [--idle F] [--benign F] [--agentic F]
  python bench/phase20_gpu_busy.py fallback-b [--pid N] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
NODES: dict[str, str] = {"studio1": "m4-1", "studio2": "m4-2"}
RANK: dict[str, int] = {"studio2": 0, "studio1": 1}   # PREREG D7: m4-2 = rank 0
API_BASE = os.environ.get("PHASE20_API_BASE", "http://192.168.86.48:52415")
API = API_BASE + "/v1/chat/completions"
MODEL = "dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
STATE_DB = "file:/Users/adam.durham/.hermes/state.db?mode=ro"
SESSION_ID = "20261007_092009_9a2ed7"
SYS_HASH = "13466837d495ef2fda6d113fbc7656ab13431a876c7efe9e40a2d706a99a708a"
RUNNER_PGREP = "multiprocessing.spawn import spawn_main"
RAW_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs/benchmarks/phase20-throughput/raw")

# Persistent own-request registry.  Every phase20 tool runs in its OWN process
# and, immediately before each generation POST, appends one JSONL line
# ``{"label": <str>, "t": <float epoch>}`` here (via ChunkGuard(registry_path=)
# / register_own_request()).  The guard's ENTRY idle-check refuses to start if
# any generation POST on a node is newer than min_idle_s (600 s) unless it
# matches a registered own request -- so a later run must LOAD this file and
# hand the epochs to ChunkGuard, or it stalls ~10 min on its own (or a sibling
# tool's) traffic.  Same pattern as bench/phase20_delta_ladder.py.
DEFAULT_REGISTRY_NAME = "own_requests.jsonl"


def load_own_requests(path: str | None) -> list[float]:
    """Read the guard's persistent own-request registry and return its ``t`` epochs.

    The registry is the JSONL file ``ChunkGuard(registry_path=...)`` appends to:
    one object per line, ``{"label": <str>, "t": <float epoch>}``.  A missing
    file (first run) or a malformed/blank line is tolerated and skipped -- this
    reader must never fail a run.  Registrations are made immediately BEFORE
    each POST, so an epoch read back here can only correspond to a request
    already on a node.
    """
    out: list[float] = []
    if not path or not os.path.exists(path):
        return out
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line)["t"]
                    out.append(float(t))
                except (json.JSONDecodeError, TypeError, KeyError, ValueError):
                    continue
    except OSError:
        return out
    return out

# PREREG 0d thresholds
GATE_GPU_SERIALIZED = 90.0     # >= both nodes
GATE_HOST_BOUND = 85.0         # < either node
GATE_PYTHON_SERIALIZED = 5.0   # host-python < 5%
GATE_NODE_GAP = 10.0           # > 10 points
FALSIFIER_IDLE_DELTA = 5.0     # within 5 points of idle

# --------------------------------------------------------------------------- #
# powermetrics parsing
# --------------------------------------------------------------------------- #
_BLOCK_RE = re.compile(r"^\*\*\* Sampled system activity .*?\(([0-9.]+)ms elapsed\)")
_FREQ_RE = re.compile(r"GPU HW active frequency:\s*([0-9.]+)\s*MHz")
_RES_RE = re.compile(r"GPU HW active residency:\s*([0-9.]+)%\s*\((.*)\)")
_BIN_RE = re.compile(r"([0-9.]+)\s*MHz:\s*([0-9.]+)%")
_SW_REQ_RE = re.compile(r"GPU SW requested state:\s*\((.*)\)")
_SW_STATE_RE = re.compile(r"GPU SW state:\s*\((.*)\)")
_SW_BIN_RE = re.compile(r"(SW_)?P([0-9]+)\s*:\s*([0-9.]+)%")
_IDLE_RE = re.compile(r"GPU idle residency:\s*([0-9.]+)%")
_POWER_RE = re.compile(r"GPU Power:\s*([0-9.]+)\s*(mW|W)")
# block header date e.g. (Wed Oct  7 21:02:58 2026 -0500)
_BLOCK_DATE_RE = re.compile(r"^\*\*\* Sampled system activity \(([^)]*)\)")


def _bins(text: str) -> list[dict]:
    """Parse an ordered list of (label/freq, pct) bins.  ORDER preserved: some
    fixtures repeat an MHz label (1182 MHz twice) so this is NOT a dict."""
    out = []
    for lab, pct in _BIN_RE.findall(text):
        out.append({"freq_mhz": int(float(lab)), "pct": float(pct)})
    return out


def _sw_bins(text: str) -> list[dict]:
    out = []
    for sw, num, pct in _SW_BIN_RE.findall(text):
        out.append({"state": ("SW_P" if sw else "P") + num, "pct": float(pct)})
    return out


def parse_powermetrics(path: str) -> dict:
    """Parse a ``powermetrics`` text capture into an ordered list of blocks."""
    txt = open(path, "r", errors="replace").read()
    lines = txt.splitlines()
    header = {"machine": None, "os": None, "boot_time": None}
    for ln in lines[:8]:
        if ln.startswith("Machine model:"):
            header["machine"] = ln.split(":", 1)[1].strip()
        elif ln.startswith("OS version:"):
            header["os"] = ln.split(":", 1)[1].strip()
        elif ln.startswith("Boot time:"):
            header["boot_time"] = ln.split(":", 1)[1].strip()

    blocks = []
    cur = None
    for ln in lines:
        m = _BLOCK_RE.match(ln)
        if m:
            dm = _BLOCK_DATE_RE.match(ln)
            cur = {"elapsed_ms": float(m.group(1)),
                   "timestamp": dm.group(1) if dm else None,
                   "hw_active_freq_mhz": None, "hw_active_residency_pct": None,
                   "hw_bins": [], "sw_requested": [], "sw_state": [],
                   "idle_residency_pct": None, "power_mw": None}
            blocks.append(cur)
            continue
        if cur is None:
            continue
        m = _FREQ_RE.search(ln)
        if m:
            cur["hw_active_freq_mhz"] = int(float(m.group(1)))
            continue
        m = _RES_RE.search(ln)
        if m:
            cur["hw_active_residency_pct"] = float(m.group(1))
            cur["hw_bins"] = _bins(m.group(2))
            continue
        m = _SW_REQ_RE.search(ln)
        if m:
            cur["sw_requested"] = _sw_bins(m.group(1))
            continue
        m = _SW_STATE_RE.search(ln)
        if m:
            cur["sw_state"] = _sw_bins(m.group(1))
            continue
        m = _IDLE_RE.search(ln)
        if m:
            cur["idle_residency_pct"] = float(m.group(1))
            continue
        m = _POWER_RE.search(ln)
        if m:
            val = float(m.group(1))
            cur["power_mw"] = val * (1000.0 if m.group(2) == "W" else 1.0)
            continue
    return {"path": os.path.abspath(path), "header": header, "blocks": blocks}


def _weighted_mean(pairs: list[tuple[float, float]]) -> float | None:
    """pairs = [(weight, value)]; weight-weighted mean, None if empty/0wt."""
    tot = sum(w for w, _ in pairs)
    if not tot:
        return None
    return sum(w * v for w, v in pairs) / tot


def pm_summary(parsed: dict) -> dict:
    """Aggregate the blocks: elapsed-ms-weighted mean active residency + bin
    distributions; the FIRST block is also reported separately (it can be
    unrepresentative)."""
    blocks = parsed["blocks"]
    if not blocks:
        return {"n_blocks": 0, "note": "no sampled blocks parsed"}

    def _acc(sel):
        return [(b["elapsed_ms"], b[sel]) for b in blocks
                if b.get(sel) is not None]

    res = _acc("hw_active_residency_pct")
    mean_res = _weighted_mean(res)

    # frequency bins: align by position (order matters, duplicate labels exist)
    n_bins = max((len(b["hw_bins"]) for b in blocks), default=0)
    freq_list: list[int | None] = []
    bin_pairs: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    for b in blocks:
        for i, cell in enumerate(b["hw_bins"]):
            if i >= n_bins:
                continue
            if i >= len(freq_list):
                freq_list.append(cell["freq_mhz"])
            bin_pairs[i].append((b["elapsed_ms"], cell["pct"]))
    freq_dist = [{"freq_mhz": freq_list[i] if i < len(freq_list) else None,
                  "pct": _weighted_mean(bin_pairs[i])}
                 for i in range(n_bins)]

    # P-state distributions (requested / sw state)
    def _state_dist(key):
        idx = {}
        for b in blocks:
            for cell in b[key]:
                idx.setdefault(cell["state"], []).append((b["elapsed_ms"], cell["pct"]))
        return [{"state": s, "pct": _weighted_mean(v)} for s, v in idx.items()]

    idle_res = [{"elapsed_ms": b["elapsed_ms"], "pct": b["idle_residency_pct"]}
                for b in blocks if b.get("idle_residency_pct") is not None]
    power = [{"elapsed_ms": b["elapsed_ms"], "mw": b["power_mw"]}
             for b in blocks if b.get("power_mw") is not None]

    first = blocks[0]
    return {
        "n_blocks": len(blocks),
        "total_elapsed_ms": sum(b["elapsed_ms"] for b in blocks),
        "mean_hw_active_residency_pct": mean_res,
        "first_block_hw_active_residency_pct": first.get("hw_active_residency_pct"),
        "mean_hw_active_freq_mhz": _weighted_mean(
            [(b["elapsed_ms"], b["hw_active_freq_mhz"]) for b in blocks
             if b.get("hw_active_freq_mhz") is not None]),
        "hw_active_freq_mhz_by_block": [b["hw_active_freq_mhz"] for b in blocks],
        "hw_bin_distribution": freq_dist,
        "sw_requested_distribution": _state_dist("sw_requested"),
        "sw_state_distribution": _state_dist("sw_state"),
        "mean_idle_residency_pct": _weighted_mean([(d["elapsed_ms"], d["pct"]) for d in idle_res]),
        "mean_power_mw": _weighted_mean([(d["elapsed_ms"], d["mw"]) for d in power]),
        "per_block_residency_pct": [b.get("hw_active_residency_pct") for b in blocks],
    }


# --------------------------------------------------------------------------- #
# `sample` call-graph parsing
# --------------------------------------------------------------------------- #
_ENTRY_RE = re.compile(r"^(?P<indent>[\s+!]*?)(?P<count>\d+)\s+(?P<sym>\S.*?)\s*$")
_THREAD_RE = re.compile(r"Thread_(\d+)")
_ADDR_RE = re.compile(r"\s*\[0x[0-9a-fA-Fx, ]+\]")
_OFF_RE = re.compile(r"\s*\+\s*\d+(?:\s*,\s*\d+)?\s*$")
_INLIB_RE = re.compile(r"\s*\(in [^)]*\)")


def norm_frame(sym: str) -> str:
    """Strip address, ``(in lib...)`` and byte-offset from a frame symbol."""
    s = _ADDR_RE.sub("", sym)
    s = _INLIB_RE.sub("", s)
    s = _OFF_RE.sub("", s)
    return s.strip()


def _sections(lines: list[str]) -> dict:
    def find(pred, start=0):
        for i in range(start, len(lines)):
            if pred(lines[i]):
                return i
        return None

    cg = find(lambda l: l.strip() == "Call graph:")
    if cg is None:
        cg = find(lambda l: l.startswith("Call graph"))
    total = find(lambda l: l.startswith("Total number in stack"), (cg or 0) + 1)
    sort = find(lambda l: l.startswith("Sort by top of stack"), (total or cg or 0) + 1)
    binimg = find(lambda l: l.startswith("Binary Images:"), (sort or total or cg or 0) + 1)
    end = len(lines)
    return {
        "call_graph": lines[(cg + 1):(total if total is not None else end)] if cg is not None else [],
        "total_in_stack": lines[(total + 1):(sort if sort is not None else end)] if total is not None else [],
        "sort_by_top_of_stack": lines[(sort + 1):(binimg if binimg is not None else end)] if sort is not None else [],
        "binary_images": lines[(binimg + 1):end] if binimg is not None else [],
        "has_call_graph": cg is not None,
    }


def _parse_call_graph(seg: list[str]) -> list[dict]:
    """Return a flat list of entries with their relative depth within the file.

    Root lines (no ``+``/``!`` marker, symbol begins with ``Thread_``) define a
    base indent; a child's relative depth is (indent - base)//2.  Both ``+``
    and ``!`` are indent characters (the ``!`` marks a diverging sub-path).
    """
    entries: list[dict] = []
    base = 0
    for ln in seg:
        m = _ENTRY_RE.match(ln)
        if not m:
            continue
        indent = m.group("indent")
        sym = m.group("sym")
        count = int(m.group("count"))
        is_root = ("+" not in indent and "!" not in indent
                   and sym.startswith("Thread_"))
        if is_root:
            base = len(indent.rstrip()) if indent.strip() == "" else len(indent)
            rel = 0
        else:
            rel = max(0, (len(indent) - base) // 2)
        tid = None
        if is_root:
            tm = _THREAD_RE.search(sym)
            tid = int(tm.group(1)) if tm else None
        entries.append({"indent_len": len(indent), "rel": rel, "count": count,
                        "sym": norm_frame(sym), "raw": sym,
                        "is_root": is_root, "thread_id": tid})
    # assign each entry to its owning root thread
    cur_tid = None
    for e in entries:
        if e["is_root"]:
            cur_tid = e["thread_id"]
        e["owner"] = cur_tid
    return entries


# (leaf enumeration lives in _leaf_paths below — single-pass, avoids rescan)


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
DEFAULT_CLASSIFY_CONFIG: dict = {
    "precedence": ["comm", "gpu_wait", "python_busy"],
    "categories": {
        "comm": {
            "patterns": [
                r"jaccl", r"\bibv_", r"\brdma", r"RDMA", r"librdma", r"libmlx5",
                r"librxe", r"libthunderboltrdma", r"librdmacm", r"libibverbs",
                r"libibmad", r"libibumad", r"libccan", r"\bverbs\b",
                r"mach_msg.*(rdma|jaccl)", r"AllReduce", r"all_sum", r"allsum",
                r"collective", r"send_recv", r"ib_qp", r"poll_cq", r"\bWC_",
            ],
        },
        "gpu_wait": {
            "patterns": [
                r"mlx::core::eval", r"eval_impl", r"Scheduler::wait",
                r"metal::", r"\bMTL", r"AGX", r"IOGPU", r"StreamThread",
                r"Scheduler::synchronize", r"stream::synchronize", r"mach_msg",
                r"metal::compute", r"command_buffer", r"waitUntilCompleted",
            ],
            "weak_patterns": [
                r"__psynch_cvwait", r"_pthread_cond_wait", r"\bsemaphore",
                r"__semwait_signal", r"std::condition_variable",
            ],
            "require_anchor": True,
            "anchors": [
                r"mlx::core::eval", r"eval_impl", r"Scheduler::wait",
                r"metal::", r"\bMTL", r"AGX", r"IOGPU", r"StreamThread",
                r"Scheduler::synchronize", r"stream::synchronize", r"mach_msg",
            ],
        },
        "python_busy": {
            # Only interpreter-ACTIVE frames.  Wait primitives
            # (_PySemaphore_Wait, _PyParkingLot_Park, take_gil,
            # PyEval_RestoreThread, lock_PyThread_*) are deliberately NOT here;
            # a path ending on a blocking primitive is short-circuited to
            # "other" (or an anchor category) by classify_path.
            "patterns": [
                r"_PyEval_EvalFrameDefault", r"PyEval_EvalCode",
                r"PyObject_", r"_PyObject", r"PyDict_", r"PyList_", r"PyLong_",
                r"PyUnicode_", r"PyNumber_", r"PySequence_", r"PyTuple_",
                r"PyMethod_", r"PyFunction_", r"PyIter_", r"PyFrame_",
                r"Py_TYPE", r"pymalloc", r"gc_collect", r"PyBytes_", r"PySlice_",
                r"_PyEval_", r"PyUnicode", r"PyFloat_",
            ],
        },
    },
    # A leaf stack ending on one of these means the thread was BLOCKED (not
    # running).  Such a path is attributed to comm/gpu_wait only when a
    # matching anchor frame is present anywhere on the path; otherwise "other".
    "blocking_patterns": [
        r"^__psynch_cvwait$", r"^__workq_kernreturn$", r"^__semwait_signal$",
        r"^nanosleep$", r"^read$", r"^kevent", r"^select$", r"^poll$",
        r"^__select$", r"^mach_msg", r"^semaphore_wait",
        r"^_pthread_cond_wait$", r"^__ulock_wait", r"^psynch_cvwait$",
        r"^swtch_pri$", r"^thread_switch$", r"^readv$", r"^__read_nocancel$",
    ],
}


def _load_classify_config(path: str | None) -> dict:
    if not path:
        return DEFAULT_CLASSIFY_CONFIG
    cfg = json.load(open(path))
    return cfg


def _leaf_paths(entries: list[dict]) -> list[tuple[dict, list[str]]]:
    """Yield ``(leaf_entry, root->leaf path)`` for every thread tree.

    One pass per thread: maintain a stack of ancestors keyed by relative depth,
    emit (leaf, path) whenever an entry has no deeper descendant in its tree.
    """
    by_thread: dict[int | None, list[int]] = {}
    order: list[int | None] = []
    for i, e in enumerate(entries):
        if e["owner"] not in by_thread:
            order.append(e["owner"])
        by_thread.setdefault(e["owner"], []).append(i)
    out = []
    for tid in order:
        idxs = by_thread[tid]
        stack: list[dict] = []
        for pos, i in enumerate(idxs):
            e = entries[i]
            while stack and stack[-1]["rel"] >= e["rel"]:
                stack.pop()
            stack.append(e)
            nxt = entries[idxs[pos + 1]] if pos + 1 < len(idxs) else None
            if nxt is None or nxt["rel"] <= e["rel"]:
                out.append((e, [s["sym"] for s in stack]))
    return out


def _compile_config(cfg: dict) -> dict:
    """Precompile every regex in a classify-config once."""
    comp = {"precedence": cfg.get("precedence", ["comm", "gpu_wait", "python_busy"]),
            "categories": {}, "blocking": [re.compile(p) for p in cfg.get("blocking_patterns", [])]}
    for cat, spec in cfg["categories"].items():
        comp["categories"][cat] = {
            "patterns": [re.compile(p) for p in spec.get("patterns", [])],
            "weak": [re.compile(p) for p in spec.get("weak_patterns", [])],
            "require_anchor": spec.get("require_anchor", False),
            "anchors": [re.compile(p) for p in spec.get("anchors", [])],
        }
    return comp


def _frame_categories(frame: str, comp: dict, path_has_anchor: bool) -> list[str]:
    hits = []
    for cat, spec in comp["categories"].items():
        if any(p.search(frame) for p in spec["patterns"]):
            hits.append(cat)
            continue
        if spec["weak"] and any(p.search(frame) for p in spec["weak"]):
            if not spec["require_anchor"] or path_has_anchor:
                hits.append(cat)
    return hits


def classify_path(path: list[str], comp: dict) -> str:
    """Category of a root->leaf stack path.

    Walk the path deepest-frame first.  The first frame with a STRONG match
    (comm pattern, or a gpu_wait pattern / anchored weak pattern) decides.  If a
    frame instead matches a *blocking* primitive the thread was blocked (parked,
    cond-wait, sleep, read, mach_msg...) and it is NOT python-busy: attribute to
    "other" unless a strong comm/gpu_wait anchor sat deeper.  Anything left ->
    "other".  This prevents an ancestor ``_PyEval_EvalFrameDefault`` frame from
    stealing a blocked thread's samples.
    """
    prec = comp["precedence"]
    path_has_anchor = any(
        any(p.search(f) for p in comp["categories"].get("gpu_wait", {}).get("anchors", []))
        for f in path)
    for frame in reversed(path):
        hits = _frame_categories(frame, comp, path_has_anchor)
        if hits:
            for c in prec:
                if c in hits:
                    return c
            return hits[0]
        if any(p.search(frame) for p in comp["blocking"]):
            return "other"
    return "other"


def classify_sample(parsed: dict, cfg: dict) -> dict:
    entries = parsed["entries"]
    comp = _compile_config(cfg)

    # per-thread table
    threads: dict[int | None, dict] = {}
    for e in entries:
        if e["is_root"]:
            threads[e["thread_id"]] = {
                "thread_id": e["thread_id"], "header": e["raw"],
                "total_samples": e["count"], "categories": {},
                "top_frames": []}
    grand = {"total": 0, "categories": {}}
    unclassified: dict[str, int] = {}
    top_by_thread: dict[int | None, dict[str, int]] = {}
    for leaf, path in _leaf_paths(entries):
        cat = classify_path(path, comp)
        tid = leaf["owner"]
        t = threads.setdefault(tid, {"thread_id": tid, "header": "(unknown)",
                                     "total_samples": 0, "categories": {},
                                     "top_frames": []})
        t["categories"][cat] = t["categories"].get(cat, 0) + leaf["count"]
        grand["total"] += leaf["count"]
        grand["categories"][cat] = grand["categories"].get(cat, 0) + leaf["count"]
        leaf_sym = path[-1] if path else leaf["sym"]
        top_by_thread.setdefault(tid, {})
        top_by_thread[tid][leaf_sym] = top_by_thread[tid].get(leaf_sym, 0) + leaf["count"]
        if cat == "other":
            unclassified[leaf_sym] = unclassified.get(leaf_sym, 0) + leaf["count"]

    for tid, t in threads.items():
        tot = sum(t["categories"].values()) or t["total_samples"]
        t["categories_pct"] = {c: round(100.0 * n / tot, 2)
                               for c, n in t["categories"].items()} if tot else {}
        tf = sorted(top_by_thread.get(tid, {}).items(), key=lambda x: -x[1])[:5]
        t["top_frames"] = [{"frame": f, "samples": n} for f, n in tf]
        t["is_main"] = "Py_RunMain" in t["header"] or any(
            e["owner"] == tid and "Py_RunMain" in e["sym"] for e in entries)

    grand_pct = {c: round(100.0 * n / grand["total"], 2)
                 for c, n in grand["categories"].items()} if grand["total"] else {}

    main_tid = next((tid for tid, t in threads.items() if t.get("is_main")), None)
    main = threads.get(main_tid) if main_tid is not None else None

    return {
        "n_threads": len(threads),
        "total_samples": grand["total"],
        "all_threads_pct": grand_pct,
        "all_threads_counts": grand["categories"],
        "main_thread_id": main_tid,
        "main_thread_pct": (main or {}).get("categories_pct", {}),
        "threads": [threads[t] for t in threads],
        "unclassified_top_frames": [
            {"frame": f, "samples": n}
            for f, n in sorted(unclassified.items(), key=lambda x: -x[1])[:15]],
        "top_of_stack_section": parsed.get("top_of_stack", []),
    }


def parse_sample(path: str, cfg: dict | None = None) -> dict:
    txt = open(path, "r", errors="replace").read()
    lines = txt.splitlines()
    header = {"pid": None, "process": None, "date": None, "tool": None}
    for ln in lines[:25]:
        if ln.startswith("Analysis of sampling"):
            m = re.search(r"sampling (\S+) \(pid (\d+)\)", ln)
            if m:
                header["process"], header["pid"] = m.group(1), int(m.group(2))
        elif ln.startswith("Date/Time:"):
            header["date"] = ln.split(":", 1)[1].strip()
        elif ln.startswith("Analysis Tool:"):
            header["tool"] = ln.split(":", 1)[1].strip()
    sec = _sections(lines)
    entries = _parse_call_graph(sec["call_graph"])

    # Sort-by-top-of-stack section -> {frame: collapsed_count}
    top = []
    for ln in sec["sort_by_top_of_stack"]:
        m = re.match(r"^(?P<sym>.+?)\s{2,}(?P<count>\d+)\s*$", ln)
        if m:
            top.append({"frame": norm_frame(m.group("sym")),
                        "count": int(m.group("count"))})
    parsed = {"path": os.path.abspath(path), "header": header, "entries": entries,
              "sections_present": {k: bool(v) for k, v in sec.items()},
              "total_in_stack": sec["total_in_stack"],
              "top_of_stack": top}
    parsed["classification"] = classify_sample(parsed, cfg or DEFAULT_CLASSIFY_CONFIG)
    return parsed


# --------------------------------------------------------------------------- #
# window validity + report / verdict
# --------------------------------------------------------------------------- #
def window_inside_stream(win_start: float, win_end: float,
                         first_tok: float | None, last_tok: float | None,
                         tol_s: float = 0.5) -> tuple[bool, str]:
    """True iff the sampling window lay entirely inside [first,last] token.

    A window must start AFTER the first content token and end BEFORE the last
    token (within ``tol_s``); otherwise it overlapped prefill or the tail
    flush and is INVALID.
    """
    if first_tok is None or last_tok is None:
        return False, "no token timestamps captured"
    if win_start < first_tok - tol_s:
        return False, (f"window starts {-1 * (first_tok - win_start):.3f}s before "
                       "the first token (prefill contamination)")
    if win_end > last_tok + tol_s:
        return False, (f"window ends {win_end - last_tok:.3f}s after the last "
                       "token (post-stream flush)")
    return True, "window fully inside [first token, last token]"


def _mean_active(pm: dict) -> float | None:
    return pm.get("mean_hw_active_residency_pct") if pm else None


def decide(idle: dict | None, benign: dict | None, agentic: dict | None,
           *, window_valid: dict | None = None) -> dict:
    """Compute GPU-busy per node and the PREREG-0d verdict.

    ``idle``/``benign``/``agentic`` are per-node dicts:
        {node: {"mean_hw_active_residency_pct": x, "sample": <classify dict>}}
    """
    idle = idle or {}
    results: dict[str, dict] = {}
    gpu_busy: dict[str, float] = {}
    for node in NODES:
        i = _mean_active(idle.get(node) or {})
        decode_means = []
        for arm in (benign, agentic):
            a = _mean_active((arm or {}).get(node) or {})
            if a is not None:
                decode_means.append(a)
        dec = statistics.fmean(decode_means) if decode_means else None
        py = None
        for arm in (agentic, benign):
            cls = ((arm or {}).get(node) or {}).get("sample")
            if cls:
                pct = cls.get("main_thread_pct") or cls.get("all_threads_pct") or {}
                if pct:
                    py = pct.get("python_busy")
                    break
        results[node] = {"idle_mean_active_pct": i, "decode_mean_active_pct": dec,
                         "gpu_busy_pct": (None if (i is None or dec is None) else dec - i),
                         "host_python_pct": py}
        if i is not None and dec is not None:
            gpu_busy[node] = dec - i

    flags: dict[str, bool] = {"window_invalid": False}
    reasons: list[str] = []

    # falsifier: residency within 5 of idle on EITHER node
    falsified = any(abs(gpu_busy[n]) <= FALSIFIER_IDLE_DELTA for n in gpu_busy)
    flags["falsifier_within_5_of_idle"] = falsified
    if falsified:
        reasons.append("residency within 5 points of idle -> wrong GPU/process; "
                       "stop and use Fallback B (xcrun xctrace Metal System Trace, 30 s)")

    both = set(NODES)
    have_both = both <= set(gpu_busy)
    gpu_serialized = (have_both and all(gpu_busy[n] >= GATE_GPU_SERIALIZED for n in NODES))
    pys = [r["host_python_pct"] for r in results.values()
           if r["host_python_pct"] is not None]
    py_low = bool(pys) and all(p < GATE_PYTHON_SERIALIZED for p in pys)
    flags["gpu_serialized"] = bool(gpu_serialized and py_low)
    if flags["gpu_serialized"]:
        reasons.append("GPU-busy >= 90% both nodes AND host-Python < 5% -> "
                       "GPU-serialized (PROF Mode-2 matters most)")

    host_bound = any(gpu_busy[n] < GATE_HOST_BOUND for n in gpu_busy)
    flags["host_comm_bound"] = host_bound
    if host_bound:
        reasons.append("GPU-busy < 85% on at least one node -> host/comm bound "
                       "(Phase-1 timer mandatory)")

    gap = (max(gpu_busy.values()) - min(gpu_busy.values())) if have_both else None
    flags["rank_imbalance"] = bool(gap is not None and gap > GATE_NODE_GAP)
    if flags["rank_imbalance"]:
        reasons.append(f"node gap {gap:.2f} points > 10 -> one rank waiting on "
                       "the other (jaccl / load imbalance)")

    if window_valid is not None:
        flags["window_invalid"] = not all(window_valid.values())
        if flags["window_invalid"]:
            reasons.append("at least one sampling window did not lie entirely "
                           "inside the decode stream interval -> INVALID run")

    if flags["falsifier_within_5_of_idle"]:
        primary = "FALSIFIER_STOP_FALLBACK_B"
    elif flags["window_invalid"]:
        primary = "INVALID_WINDOW"
    elif flags["gpu_serialized"]:
        primary = "GPU_SERIALIZED"
    elif flags["host_comm_bound"]:
        primary = "HOST_COMM_BOUND"
    elif flags["rank_imbalance"]:
        primary = "RANK_IMBALANCE"
    else:
        primary = "INCONCLUSIVE"

    return {"per_node": results, "gpu_busy_pct": gpu_busy,
            "node_gap_points": gap, "flags": flags, "primary": primary,
            "reasons": reasons}


# --------------------------------------------------------------------------- #
# markdown rendering
# --------------------------------------------------------------------------- #
def pm_markdown(name: str, pm: dict) -> str:
    s = pm_summary(pm)
    if not s.get("n_blocks"):
        return f"## {name} — no blocks\n"
    out = [f"## {name}",
           f"- blocks: {s['n_blocks']}  total elapsed: {s['total_elapsed_ms']:.1f} ms",
           f"- **mean hw-active residency: {s['mean_hw_active_residency_pct']:.2f}%** "
           f"(first block {s['first_block_hw_active_residency_pct']}%)",
           f"- mean hw-active freq: {s['mean_hw_active_freq_mhz']} MHz",
           f"- hw freq bins (ordered): " +
           " ".join(f"{c['freq_mhz']}MHz:{c['pct']:.2f}%" for c in s["hw_bin_distribution"]),
           f"- idle residency mean: {s['mean_idle_residency_pct']}%  "
           f"power mean: {s['mean_power_mw']} mW"]
    return "\n".join(out) + "\n"


def sample_markdown(name: str, sm: dict) -> str:
    c = sm["classification"]
    out = [f"## {name} — sample ({c['n_threads']} threads, {c['total_samples']} samples)",
           f"- all-threads: " + " ".join(f"{k}={v}%" for k, v in c["all_threads_pct"].items()),
           f"- main thread (id {c['main_thread_id']}): " +
           " ".join(f"{k}={v}%" for k, v in c["main_thread_pct"].items()),
           "- top threads:"]
    for t in sorted(c["threads"], key=lambda x: -x["total_samples"])[:5]:
        tops = ", ".join(f"{f['frame']}({f['samples']})" for f in t["top_frames"][:3])
        out.append(f"  - Thread_{t['thread_id']} n={t['total_samples']} "
                   f"cat={t['categories_pct']} top=[{tops}]")
    if c["unclassified_top_frames"]:
        out.append("- unclassified top frames: " +
                   ", ".join(f"{f['frame']}({f['samples']})"
                             for f in c["unclassified_top_frames"][:8]))
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# SSH plumbers
# --------------------------------------------------------------------------- #
def _runner_pid(node: str, timeout: float = 30) -> int | None:
    try:
        r = subprocess.run(["ssh", node, f"pgrep -f '{RUNNER_PGREP}'"],
                           capture_output=True, text=True, timeout=timeout)
        for tok in r.stdout.split():
            if tok.isdigit():
                return int(tok)
    except Exception:
        pass
    return None


def _ssh_capture(node: str, remote_cmd: str, out_path: str,
                 timeout: float) -> dict:
    """Run a read-only command on a node; stream stdout to out_path."""
    t0 = time.time()
    with open(out_path, "w") as fh:
        p = subprocess.run(["ssh", node, remote_cmd], stdout=fh,
                           stderr=subprocess.PIPE, text=True, timeout=timeout)
    return {"node": node, "cmd": remote_cmd, "out": out_path,
            "start_epoch": t0, "end_epoch": time.time(),
            "returncode": p.returncode, "stderr": (p.stderr or "")[-2000:]}


def cmd_idle(args) -> int:
    """Both nodes concurrently, no requests.  READ-ONLY."""
    secs = args.secs
    n = max(1, int(round(secs * 1000 / 500)))
    os.makedirs(args.outdir, exist_ok=True)
    results: dict[str, dict] = {}
    lock = threading.Lock()

    def run(node: str, tag: str):
        cmd = f"sudo -n powermetrics --samplers gpu_power -i 500 -n {n}"
        out = os.path.join(args.outdir, f"idle.{tag}.powermetrics.txt")
        try:
            info = _ssh_capture(node, cmd, out, timeout=secs + 60)
        except Exception as exc:  # pragma: no cover - network
            info = {"node": node, "error": repr(exc)}
        with lock:
            results[node] = info

    threads = [threading.Thread(target=run, args=(n, t))
               for n, t in NODES.items()]
    if args.dry_run:
        print(json.dumps({"dry_run": True, "plan": "idle",
                          "nodes": NODES, "per_node_cmd":
                          f"sudo -n powermetrics --samplers gpu_power -i 500 -n {n}",
                          "outdir": args.outdir}, indent=1))
        return 0
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    summary = {}
    for node, info in results.items():
        if info.get("error") or not os.path.exists(info.get("out", "")):
            summary[node] = {"error": info.get("error") or "no output"}
            continue
        pm = parse_powermetrics(info["out"])
        summary[node] = {"file": info["out"], "start_epoch": info["start_epoch"],
                         "end_epoch": info["end_epoch"],
                         "summary": pm_summary(pm)}
    blob = {"kind": "idle", "secs": secs, "nodes": summary}
    if args.out:
        json.dump(blob, open(args.out, "w"), indent=1)
    print(json.dumps(blob, indent=1))
    return 0


# --------------------------------------------------------------------------- #
# prompt builders (copy-adapted from phase19_round_measure.py / agentic)
# --------------------------------------------------------------------------- #
_SENTENCES = [
    "The observer pattern defines a one-to-many dependency so that when one object changes state all its dependents are notified automatically.",
    "A binary search tree keeps the key of every internal node greater than all keys in its left subtree and less than those in its right subtree.",
    "Garbage collection reclaims memory that was allocated by a program but is no longer reachable from any live root reference.",
    "MapReduce expresses a computation as a map phase that emits intermediate key value pairs and a reduce phase that aggregates them.",
    "The CAP theorem states a distributed data store can simultaneously guarantee only two of consistency availability and partition tolerance.",
    "Vector clocks track causal ordering between events in a distributed system without relying on a global wall clock.",
    "A bloom filter answers set membership queries with no false negatives and a tunable rate of false positives.",
    "Consistent hashing places both keys and nodes on a ring so that adding or removing a node moves only a small fraction of keys.",
    "The two-phase commit protocol coordinates an atomic transaction across participants at the cost of blocking on coordinator failure.",
    "A skip list provides expected logarithmic search time using a hierarchy of linked lists with geometrically decreasing density.",
    "Content addressable storage names an object by a hash of its bytes so identical content is stored only once.",
    "A write ahead log records intent before mutating state so a crash can be replayed to a consistent point.",
]
TASKS = {
    "count": "\n\nTask: List the integers from 1 to 450 in order, each on its own line, and nothing else.",
    "done": "\n\nTask: Reply with ONLY the single word DONE and nothing else.",
}
AGENTIC_INSTRUCTION = (
    "\n\n=== CONTINUATION REQUEST ===\n"
    "You have just been resumed. Do NOT run anything yet.\n"
    "First, inside your reasoning, audit the investigation transcript above: list "
    "every concrete finding with the command/evidence that produced it, then identify "
    "the single largest unexplained gap in the evidence, and work out the exact next "
    "shell command that would close it and why that command rather than any other.\n"
    "Then answer ONLY with a JSON object of the form "
    '{"findings":[{"claim":"...","evidence":"...","command":"..."}],'
    '"biggest_gap":"...","next_command":"...","why_this_command":"..."}.\n'
    "Be exhaustive and cite specific numbers from the transcript.")


def build_prompt(target_tokens: int, salt: str, task: str) -> str:
    """Benign synthetic filler prompt.  Copy-adapted from
    phase19_round_measure.py:build_prompt (attribution: levers-wt phase19)."""
    rng = random.Random(20261007)
    target_chars = int(target_tokens * 5.59)
    head = f"[SESSION-SALT {salt}]\n"
    parts = [head]
    n = len(head)
    while n < target_chars:
        s = _SENTENCES[rng.randrange(len(_SENTENCES))]
        parts.append(s + " ")
        n += len(s) + 1
    parts.append(TASKS[task])
    return "".join(parts)


def build_agentic_prompt(salt: str, char_budget: int | None,
                         db: str = STATE_DB, session_id: str = SESSION_ID,
                         sys_hash: str = SYS_HASH) -> tuple[str, dict]:
    """preamble + real session messages + instruction.  Copy-adapted from
    phase19_agentic_measure.py (attribution: levers-wt phase19)."""
    import sqlite3
    con = sqlite3.connect(db, uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT prompt FROM system_prompts WHERE hash=?", (sys_hash,))
    row = cur.fetchone()
    preamble = row[0] if row else ""
    cur.execute("SELECT id,role,content,reasoning_content,tool_calls,tool_name,"
                "timestamp FROM messages WHERE session_id=? ORDER BY timestamp,id",
                (session_id,))
    msgs = [dict(r) for r in cur.fetchall()]
    con.close()

    def render(m):
        role, c = m["role"], (m["content"] or "")
        rc = m["reasoning_content"] or ""
        tc = m["tool_calls"] or ""
        if role == "assistant":
            parts = []
            if rc:
                parts.append("<reasoning>\n" + rc + "\n</reasoning>")
            if c:
                parts.append(c)
            if tc:
                parts.append("<tool_calls>\n" + tc + "\n</tool_calls>")
            return "\n\n[assistant]\n" + "\n".join(parts)
        if role == "tool":
            return f"\n\n[tool_result name={m['tool_name']}]\n{c}"
        if role == "system":
            return f"\n\n[system]\n{c}"
        return f"\n\n[{role}]\n{c}"

    head = f"[SESSION-SALT {salt}]\n"
    body = head + preamble
    used = 0
    for m in msgs:
        piece = render(m)
        if char_budget is not None and len(body) + len(piece) > char_budget:
            break
        body += piece
        used += 1
    body += AGENTIC_INSTRUCTION
    return body, {"n_messages_total": len(msgs), "n_messages_used": used,
                  "preamble_chars": len(preamble), "prompt_chars": len(body)}


def stream_once(prompt: str, max_tokens: int, effort: str | None = None,
                on_first_token=None, stop_event=None) -> dict:
    """Stream a completion; record first/last token epochs.

    Copy-adapted from phase19_agentic_measure.py:stream_once (attribution:
    levers-wt phase19).  ``on_first_token(epoch, resp)`` is invoked exactly
    once, on the first token, so the samplers can be launched.

    This model is a REASONING model: it streams its output in
    ``choices[].delta.reasoning_content`` (the live capture showed
    ``content=''`` and ``reasoning_content='We'``).  We therefore accumulate
    BOTH ``content`` and ``reasoning_content`` as output text, and fire the
    first-token time (and ``on_first_token``) on the first NON-EMPTY delta of
    EITHER kind.  Accumulating only ``content`` left ttft permanently None and
    the window-validity logic with no token timestamps (INVALID window), and
    the samplers never launched -- so every decode run measured nothing.
    Mirrors bench/phase20_delta_ladder.py / phase20_common.py:stream_once.
    """
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True}
    if effort:
        body["reasoning_effort"] = effort
    req = urllib.request.Request(API, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = first = last = None
    cchars = 0
    usage = None
    stats = None
    fired = False
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            if stop_event is not None and stop_event.is_set():
                break
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if line.startswith(": generation_stats"):
                try:
                    stats = json.loads(line.split(" ", 2)[2])
                except Exception:
                    pass
                continue
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            # FIX 3: epoch timestamps must be WALL CLOCK (time.time()), the same
            # clock _ssh_capture stamps its start_epoch/end_epoch with.  Pre-fix
            # this was time.perf_counter() (monotonic, ~seconds since boot),
            # which window_inside_stream compared against ~1.79e9 wall epochs --
            # an unmatched-unit comparison that could never be inside.
            now_epoch = time.time()
            now = time.perf_counter()          # monotonic: for durations only
            if usage is None and obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                d = ch.get("delta") or {}
                # FIX 2: a reasoning model streams delta.reasoning_content, not
                # delta.content.  Accumulate BOTH; first-token fires on the
                # first non-empty delta of EITHER kind.
                ctxt = d.get("content") or d.get("reasoning_content") or ""
                if ctxt:
                    if first is None:
                        first, ttft = now_epoch, now - t0
                        if on_first_token and not fired:
                            fired = True
                            on_first_token(now_epoch, resp)
                    cchars += len(ctxt)
                    last = now_epoch
    wall = time.perf_counter() - t0
    return {"wall_s": round(wall, 3), "ttft_s": round(ttft, 4) if ttft else None,
            "first_token_epoch": first, "last_token_epoch": last,
            "decode_s": round(last - first, 4) if (first and last) else None,
            "content_chars": cchars, "usage": usage, "stats": stats}


# --------------------------------------------------------------------------- #
# decode (live; PM only) + dry-run plan
# --------------------------------------------------------------------------- #
def decode_plan(args) -> dict:
    secs = args.secs
    pm_n = max(1, int(round(secs * 1000 / 500)))
    return {
        "sub": "decode", "workload": args.workload, "depth": args.depth,
        "secs": secs, "max_tokens": args.max_tokens,
        "steps": [
            "canary ok (R4) via phase20_guard.canary()",
            "ChunkGuard(label, own_requests=<loaded registry>, registry_path=<file>) "
            "+ register_own_request() around EVERY request (R1/R2/R3)",
            "feed prompt COLD (max_tokens=small) -- prefix build, NOT measured",
            "MEASURED request = EXACT SAME prompt (expect prefill=0 via prompt-end checkpoint)",
            "verify exo.log 'turn reuse: ... prefill=0' before trusting the window",
            f"on FIRST SSE content token: start powermetrics (-i 500 -n {pm_n}) AND "
            f"sample <runner-pid> {secs + 5} 1 -mayDie -file /tmp/p20_sample.<arm>.<node>.txt "
            "on BOTH nodes",
            "keep streaming until the window elapses; close the stream; check window "
            "inside [first token, last token]",
            "scp sample files back; parse; write raw/gpu_busy.<workload>.json + .md",
        ],
        "runner_pid_cmd": f"pgrep -f '{RUNNER_PGREP}'",
        "nodes": NODES, "note": "LIVE: only the PM runs this.",
    }


def cmd_decode(args) -> int:
    if args.dry_run:
        print(json.dumps(decode_plan(args), indent=1))
        return 0

    # LIVE path — imports the guard lazily so the offline build/tests never
    # need it.  The PM owns the cluster; this path only orchestrates.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import phase20_guard as guard  # type: ignore
    except Exception as exc:  # pragma: no cover - live only
        print(json.dumps({"error": f"phase20_guard unavailable: {exc!r}"},
                         indent=1))
        return 2

    salt = os.urandom(4).hex()
    if args.workload == "agentic":
        prompt, meta = build_agentic_prompt(salt, int(args.depth * 5.59))
    else:
        prompt, meta = build_prompt(args.depth, salt, "count"), {}
    pm_n = max(1, int(round(args.secs * 1000 / 500)))
    os.makedirs(args.outdir, exist_ok=True)
    arm = args.workload
    rec: dict = {"kind": "decode", "workload": arm, "depth": args.depth,
                 "secs": args.secs, "max_tokens": args.max_tokens,
                 "salt": salt, "meta": meta, "prompt_chars": len(prompt)}
    window_valid: dict[str, bool] = {}
    node_out: dict[str, dict] = {}

    # FIX 1: persistent own-request registry.  Every phase20 tool runs in its
    # own process; without loading the shared registry the guard's entry
    # idle-check treats this tool's own (or a sibling tool's) recent POST as
    # foreign traffic and refuses to start for min_idle_s (600 s).  Default
    # lives beside the run outputs in <outdir>/raw/; --own-registry overrides.
    raw_dir = os.path.join(args.outdir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    registry_path = (getattr(args, "own_registry", None)
                     or os.path.join(raw_dir, DEFAULT_REGISTRY_NAME))
    own_requests = load_own_requests(registry_path)

    with guard.ChunkGuard(f"phase20-gpu-busy-{arm}",
                          max_wall_s=args.max_wall,
                          log_dir=args.outdir,
                          own_requests=own_requests,
                          registry_path=registry_path) as g:
        g.register_own_request()
        warm = stream_once(prompt, args.warm_tokens, stop_event=getattr(g, "cancel_event", None))
        rec["warmup"] = {"usage": warm.get("usage")}

        # FIX 2: the samplers must NOT block the SSE read loop.  The inline
        # first-token callback therefore does two cheap things only -- record
        # the first-token epoch into ``node_out`` and SPAWN the per-node
        # capture thread(s) -- then returns immediately so ``stream_once`` can
        # keep reading tokens at full rate.  The blocking ssh captures
        # (powermetrics + sample) run in the background; we join them (bounded)
        # AFTER the stream ends and only then parse.
        sampler_threads: list[threading.Thread] = []

        def _sampler_one(node: str, tag: str, first_epoch):
            pid = _runner_pid(node)
            pm_out = os.path.join(args.outdir, f"{arm}.{tag}.powermetrics.txt")
            sm_out = os.path.join(args.outdir, f"{arm}.{tag}.sample.txt")
            pm_cmd = f"sudo -n powermetrics --samplers gpu_power -i 500 -n {pm_n}"
            sm_cmd = (f"/usr/bin/sample {pid or 0} {args.secs + 5} 1 -mayDie "
                      f"-file /tmp/{arm}.{tag}.sample.txt; "
                      f"cat /tmp/{arm}.{tag}.sample.txt")
            try:
                ri = _ssh_capture(node, pm_cmd, pm_out, timeout=args.secs + 60)
                ri2 = _ssh_capture(node, sm_cmd, sm_out, timeout=args.secs + 60)
                node_out[tag].update({"pid": pid, "powermetrics": ri,
                                      "sample": ri2, "_pending": False})
            except Exception as exc:  # pragma: no cover - network
                node_out[tag].update({"_pending": False, "error": repr(exc)})

        def start_samplers(first_epoch, resp=None):
            """``stream_once`` callback: ``on_first_token(first_epoch, resp)``.

            Invoked INLINE from inside the SSE read loop, so it must return
            promptly -- record the epoch for every node, then spawn the
            background capture thread(s).  It never joins/blocks.
            """
            # record the first-token epoch for both nodes up front, so a
            # capture that later errors still leaves a labelled entry.
            for tag in NODES.values():
                node_out[tag] = {"first_token_epoch": first_epoch,
                                 "_pending": True}
            for node, tag in NODES.items():
                th = threading.Thread(target=_sampler_one,
                                      args=(node, tag, first_epoch),
                                      name=f"p20-sampler-{tag}", daemon=True)
                sampler_threads.append(th)
                th.start()

        g.register_own_request()
        dec = stream_once(prompt, args.max_tokens, on_first_token=start_samplers,
                          stop_event=getattr(g, "cancel_event", None))

        # the stream has ended -> collect the background captures (bounded)
        # BEFORE parsing.  A capture that overran its bound is left pending;
        # the parse step below simply finds no file for it.
        sampler_join_s = args.secs * 3 + 120
        for th in sampler_threads:
            th.join(timeout=sampler_join_s)
        rec["decode"] = {k: dec.get(k) for k in
                         ("wall_s", "ttft_s", "first_token_epoch", "last_token_epoch",
                          "decode_s", "content_chars", "usage", "stats")}

    # validity: window elapsed must lie inside [first, last] token
    first, last = dec.get("first_token_epoch"), dec.get("last_token_epoch")
    for tag, info in node_out.items():
        if info.get("powermetrics"):
            ok, why = window_inside_stream(info["powermetrics"]["start_epoch"],
                                           info["powermetrics"]["end_epoch"], first, last)
            window_valid[tag] = ok
            info["window_check"] = why
    rec["window_valid"] = window_valid

    # parse back
    parsed = {}
    for tag, info in node_out.items():
        entry = {}
        pmf = (info.get("powermetrics") or {}).get("out")
        smf = (info.get("sample") or {}).get("out")
        if pmf and os.path.exists(pmf):
            entry["powermetrics"] = pm_summary(parse_powermetrics(pmf))
        if smf and os.path.exists(smf):
            sm = parse_sample(smf)
            entry["sample"] = sm["classification"]
        parsed[tag] = entry
    rec["parsed"] = parsed

    os.makedirs(RAW_DIR, exist_ok=True)
    jpath = os.path.join(RAW_DIR, f"gpu_busy.{arm}.json")
    json.dump(rec, open(jpath, "w"), indent=1)
    print(json.dumps({"wrote": jpath, "window_valid": window_valid}, indent=1))
    return 0


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _load_json(path: str | None) -> dict | None:
    return json.load(open(path)) if path and os.path.exists(path) else None


def cmd_report(args) -> int:
    idle = _load_json(args.idle)
    benign = _load_json(args.benign)
    agentic = _load_json(args.agentic)

    def nodes_of(blob):
        if not blob:
            return {}
        out = {}
        for node, info in (blob.get("nodes") or {}).items():
            if "summary" in info:
                out[node] = {"mean_hw_active_residency_pct":
                             info["summary"].get("mean_hw_active_residency_pct")}
            else:
                out[node] = info
        for tag, entry in (blob.get("parsed") or {}).items():
            out.setdefault(tag, {})
            if "powermetrics" in entry:
                out[tag]["mean_hw_active_residency_pct"] = \
                    entry["powermetrics"].get("mean_hw_active_residency_pct")
            if "sample" in entry:
                out[tag]["sample"] = entry["sample"]
        return out

    window_valid = None
    if agentic and agentic.get("window_valid"):
        window_valid = agentic["window_valid"]
    decision = decide(nodes_of(idle), nodes_of(benign), nodes_of(agentic),
                      window_valid=window_valid)
    print(json.dumps(decision, indent=1))
    return 0


def fallback_b_commands(pid: int | None, out: str) -> list[str]:
    """Construct (never run) the Fallback-B capture commands: a 30 s Metal
    System Trace attached to the runner, per node."""
    pid = pid or 0
    cmds = []
    for node in NODES:
        cmds.append(
            f"ssh {node} \"xcrun xctrace record --template 'Metal System Trace' "
            f"--attach {pid} --time-limit 30s --output /tmp/{out}.{NODES[node]}.trace "
            f"--no-prompt && xcrun xctrace export --input "
            f"/tmp/{out}.{NODES[node]}.trace --toc\"")
    return cmds


def cmd_fallback_b(args) -> int:
    print(json.dumps({"fallback_b": fallback_b_commands(args.pid, args.out),
                      "note": "documented, NOT run"}, indent=1))
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cmd_parse_pm(args) -> int:
    pm = parse_powermetrics(args.file)
    s = pm_summary(pm)
    dec = idle = None
    if args.idle:
        idle = pm_summary(parse_powermetrics(args.idle))
        dec = {"decode_mean_active_pct": s["mean_hw_active_residency_pct"],
               "idle_mean_active_pct": idle["mean_hw_active_residency_pct"],
               "gpu_busy_pct": (None if (idle["mean_hw_active_residency_pct"] is None
                                         or s["mean_hw_active_residency_pct"] is None)
                                else s["mean_hw_active_residency_pct"]
                                - idle["mean_hw_active_residency_pct"])}
    out = {"path": pm["path"], "header": pm["header"], "summary": s,
           "decode_vs_idle": dec}
    print(json.dumps(out, indent=1))
    if args.md:
        md = pm_markdown(os.path.basename(args.file), pm)
        if idle is not None:
            md += pm_markdown(os.path.basename(args.idle) + " (idle)", parse_powermetrics(args.idle))
        open(args.md, "w").write(md)
    return 0


def _cmd_parse_sample(args) -> int:
    cfg = _load_classify_config(args.classify_config)
    sm = parse_sample(args.file, cfg)
    print(json.dumps({"path": sm["path"], "header": sm["header"],
                      "classification": sm["classification"],
                      "sections_present": sm["sections_present"]}, indent=1))
    if args.md:
        open(args.md, "w").write(sample_markdown(os.path.basename(args.file), sm))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 0d GPU-busy tool")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("parse-pm")
    p.add_argument("file")
    p.add_argument("--idle", default=None)
    p.add_argument("--md", default=None)
    p.set_defaults(func=_cmd_parse_pm)

    p = sub.add_parser("parse-sample")
    p.add_argument("file")
    p.add_argument("--classify-config", default=None)
    p.add_argument("--md", default=None)
    p.set_defaults(func=_cmd_parse_sample)

    p = sub.add_parser("idle")
    p.add_argument("--secs", type=float, default=50)
    p.add_argument("--out", default=None)
    p.add_argument("--outdir", default="/Users/adam.durham/.hermes/cache/scratch/phase20/gpubusy")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_idle)

    p = sub.add_parser("decode")
    p.add_argument("--workload", choices=["benign", "agentic"], required=True)
    p.add_argument("--depth", type=int, default=100000)
    p.add_argument("--secs", type=float, default=50)
    p.add_argument("--max-tokens", type=int, default=2600)
    p.add_argument("--warm-tokens", type=int, default=8)
    p.add_argument("--max-wall", type=float, default=900)
    p.add_argument("--outdir", default="/Users/adam.durham/.hermes/cache/scratch/phase20/gpubusy")
    p.add_argument("--own-registry", default=None, dest="own_registry",
                   help="persistent own-request registry JSONL "
                        "(default: <outdir>/raw/" + DEFAULT_REGISTRY_NAME + ")")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_decode)

    p = sub.add_parser("report")
    p.add_argument("--idle", default=None)
    p.add_argument("--benign", default=None)
    p.add_argument("--agentic", default=None)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("fallback-b")
    p.add_argument("--pid", type=int, default=None)
    p.add_argument("--out", default="p20_fallback_b")
    p.set_defaults(func=cmd_fallback_b)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
