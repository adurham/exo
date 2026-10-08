#!/usr/bin/env python3
"""Phase-20 shared helpers for the 0c delta-prefill ladder (BRIEF L).

Pure stdlib.  Importable as ``phase20_common`` when ``bench/`` is on sys.path.

What lives here (used by bench/phase20_delta_ladder.py and its tests):
  * request/stream client for the OpenAI-compatible exo endpoint (reads the
    ``: generation_stats`` SSE COMMENT frame, which does NOT start with ``data:``)
  * prompt / conversation builders (unique salt ONCE at the head of a base;
    multi-turn branching delta rep shape)
  * engine-log extraction -- the ROW TRUTH source.  All regexes are pinned to
    REAL lines captured from this boot's exo.log (see tests).
  * log transport seam (LocalFileLog / SSHNodeLog / LogReader) so the ladder can
    read a mock's temp log instead of a node
  * rows/s math + wall-time prediction
  * summarize(): the two tables (ctx ladder + delta-size sweep) + the three
    pre-registered 0c decision rules + the fresh-feed DEGRADED_REFERENCE falsifier

Nothing in this module sends a generation request on import, starts/stops a
process on a node, or writes to state.db.  The HTTP client only fires when a
caller invokes stream_once().
"""
from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import time
import urllib.request

# --------------------------------------------------------------------------- const
MODEL = "dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
DEFAULT_API = os.environ.get(
    "PHASE20_API", "http://macstudio-m4-1.tail19c543.ts.net:52415"
)
NODE_LOG_PATH = "~/.exo/exo_log/exo.log"

# model constants used ONLY for prediction / calibration defaults (never for row
# truth -- row truth always comes from the engine log or usage).
DEFAULT_CHARS_PER_TOKEN = 5.111          # battery filler measured 5.111 c/t
FILLER_CHARS_PER_TOKEN_FALLBACK = 5.59   # phase19 sentences measured 5.59 c/t

# wall prediction (PREREG 0c chunk discipline):  rows / 200 * 1.3 + 15 s
PRED_ROWS_PER_S = 200.0
PRED_FACTOR = 1.3
PRED_OVERHEAD_S = 15.0
CHUNK_MAX_WALL_S = 900.0                 # R3: chunks <= 15 min

# 0c decision thresholds (PREREG)
CTX_DEPTH_DEGRADE = 0.15                 # >15% 20K->110K  => ctx-depth cost
FIXED_OVERHEAD_RATIO = 0.50             # d256 < 50% of d4096 => fixed per-call overhead
FLAT_SPREAD = 0.10                       # <10% spread across delta sizes => Phase 4 skipped
FRESH_REF_MIN_ROWS_PER_S = 255.0         # fresh-feed reference falsifier

LADDER_CTXS = (("20k", 20000), ("50k", 50000), ("110k", 110000))
DELTA_LADDER_ROWS = 2048                 # the fixed delta used on the ctx ladder
DELTA_SWEEP_ROWS = (256, 1024, 4096, 8192)
FRESH_REF_TOKENS = 100000

# ------------------------------------------------------------------------ regex
# Pinned to REAL lines (verbatim shapes) from m41_0901_1400.log.zst:
#   [ 2026-10-07 09:22:00.257 | INFO  | ...engine_prefill:333 ] [DSV41] prefill controls: fence_every=2 transient_budget_mb=2048 score_row_bytes=1 fence_hook=on (rows=245, base=2048)
#   [ 2026-10-07 09:22:01.933 | INFO  | ...engine:_start_turn:957 ] [DSV41] turn reuse: prompt=24340 prefill=245 reuse=24095 cache=24340 rewind=24270 UNCOMMITTED
#   [ 2026-10-07 09:22:56.053 | WARNING | ...engine:_start_turn:964 ] [DSV41] reuse undershoot: refed=869 rows (reused=24431)
#   [ 2026-10-07 09:23:07.067 | INFO  | ...session:get:1004 ] [DSV41] session reuse: this 28743-token prompt matches a resident conversation on 25397 rows
_TS = r"\[\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*\|"
RE_LOG_TS = re.compile(_TS)
RE_PREFILL = re.compile(
    r"\[DSV41\] prefill controls:.*?\(rows=(\d+),\s*base=(\d+)\)"
)
RE_TURNREUSE = re.compile(
    r"\[DSV41\] turn reuse:\s*prompt=(\d+)\s+prefill=(\d+)\s+reuse=(\d+)"
    r"\s+cache=(\d+)(?:\s+rewind=(\d+))?\s+UNCOMMITTED"
)
RE_UNDERSHOOT = re.compile(
    r"\[DSV41\] reuse undershoot:\s*refed=(\d+)\s+rows\s+\(reused=(\d+)\)"
)
RE_SESSIONREUSE = re.compile(
    r"\[DSV41\] session reuse: this (\d+)-token prompt matches a resident "
    r"conversation on (\d+) rows"
)
RE_POST = re.compile(r"API request:\s*POST\s+(\S+)")
RE_TASK_START = re.compile(r"Starting task TextGeneration\(")


def parse_log_ts(line: str) -> float | None:
    """Node-local CDT timestamp (no tz marker) -> POSIX seconds.

    Only *deltas* between two lines in the same window are ever used, so the
    missing timezone is harmless.
    """
    m = RE_LOG_TS.search(line)
    if not m:
        return None
    stamp = m.group(1)
    try:
        base = time.mktime(time.strptime(stamp[:19], "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        return None
    frac = 0.0
    if "." in stamp:
        frac = float("0." + stamp.split(".", 1)[1])
    return base + frac


# ------------------------------------------------------------------ log analysis
def analyze_window(text: str) -> dict:
    """Extract this request's prefill shape from a raw log window.

    ROW TRUTH: ``delta_rows = turn_reuse.prefill`` when a ``turn reuse:`` line is
    present (emitted only when reuse>0); otherwise the feed is cold and the
    caller falls back to ``usage.prompt_tokens / ttft``.

    Returns a dict with key names used verbatim in the JSONL record:
      has_turn_reuse, log_prefill, reuse, rewind, cache, prompt_log,
      prefill_controls_rows, prefill_controls_base, prefill_start_s,
      turn_reuse_s, prefill_s_log, undershoot_refed, undershoot_reused,
      session_reuse, post_s, posts, task_starts
    """
    out: dict = {
        "has_turn_reuse": False,
        "log_prefill": None, "reuse": None, "rewind": None, "cache": None,
        "prompt_log": None,
        "prefill_controls_rows": None, "prefill_controls_base": None,
        "prefill_start_s": None, "turn_reuse_s": None, "prefill_s_log": None,
        "undershoot_refed": None, "undershoot_reused": None,
        "session_reuse": None, "post_s": None, "posts": [], "task_starts": 0,
    }
    pc_s = tu_s = None
    for line in text.splitlines():
        m = RE_PREFILL.search(line)
        if m:
            out["prefill_controls_rows"] = int(m.group(1))
            out["prefill_controls_base"] = int(m.group(2))
            pc_s = parse_log_ts(line)
            out["prefill_start_s"] = pc_s
            continue
        m = RE_TURNREUSE.search(line)
        if m:
            out["has_turn_reuse"] = True
            out["prompt_log"] = int(m.group(1))
            out["log_prefill"] = int(m.group(2))
            out["reuse"] = int(m.group(3))
            out["cache"] = int(m.group(4))
            out["rewind"] = int(m.group(5)) if m.group(5) else None
            tu_s = parse_log_ts(line)
            out["turn_reuse_s"] = tu_s
            continue
        m = RE_UNDERSHOOT.search(line)
        if m:
            out["undershoot_refed"] = int(m.group(1))
            out["undershoot_reused"] = int(m.group(2))
            continue
        m = RE_SESSIONREUSE.search(line)
        if m:
            out["session_reuse"] = int(m.group(2))
            continue
        m = RE_POST.search(line)
        if m:
            ts = parse_log_ts(line)
            out["posts"].append(ts)
            out["post_s"] = ts
            continue
        if RE_TASK_START.search(line):
            out["task_starts"] += 1

    # prefill wall = t(turn reuse) - t(prefill controls); fall back to the POST
    # line when the prefill-controls line was not emitted for a small delta.
    if tu_s is not None:
        start = pc_s if pc_s is not None else out["post_s"]
        if start is not None:
            out["prefill_s_log"] = round(tu_s - start, 6)
    return out


def rows_per_s_log(record: dict) -> float | None:
    """Delta rows/s from the engine log:  log_prefill / prefill_s_log."""
    m = record.get("log_prefill")
    d = record.get("prefill_s_log")
    if m is None or d in (None, 0) or d <= 0:
        return None
    return m / d


def rows_per_s_ttft(record: dict) -> float | None:
    """Client-side cross-check.  For a delta: log_prefill / ttft.  For a cold
    feed: usage.prompt_tokens / ttft (prior-campaign method: 100000/373.9)."""
    ttft = record.get("ttft_s")
    if ttft in (None, 0) or ttft <= 0:
        return None
    rows = record.get("log_prefill")
    if rows is None:
        rows = record.get("prompt_tokens")
    if rows is None:
        return None
    return rows / ttft


# ------------------------------------------------------------------ log transport
class LocalFileLog:
    """A node log that is really a local file (used by the mock/tests)."""

    def __init__(self, path: str):
        self.path = path

    def size(self) -> int:
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0

    def read_from(self, offset: int) -> str:
        if not os.path.exists(self.path):
            return ""
        with open(self.path, "rb") as fh:
            fh.seek(offset)
            return fh.read().decode("utf-8", "replace")


class SSHNodeLog:
    """Read-only tail of a node's exo.log over ssh (the production seam)."""

    def __init__(self, node: str, log_path: str = NODE_LOG_PATH, timeout_s: float = 30.0):
        self.node = node
        self.log_path = log_path
        self.timeout_s = timeout_s

    def size(self) -> int:
        cmd = f"stat -f %z {self.log_path} 2>/dev/null || echo 0"
        proc = subprocess.run(
            ["ssh", self.node, cmd], capture_output=True, text=True,
            timeout=self.timeout_s,
        )
        try:
            return int(proc.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return 0

    def read_from(self, offset: int) -> str:
        # tail -c +N is 1-indexed -> off+1
        cmd = f"tail -c +{offset + 1} {self.log_path} 2>/dev/null"
        proc = subprocess.run(
            ["ssh", self.node, cmd], capture_output=True, text=True,
            timeout=self.timeout_s,
        )
        return proc.stdout


class LogReader:
    """Snapshot-before / read-after seam over one or more node logs.

    ``sources`` values need only expose ``size() -> int`` and
    ``read_from(offset) -> str`` (LocalFileLog / SSHNodeLog / a test fake).
    """

    def __init__(self, sources: dict):
        self.sources = dict(sources)

    def snapshot(self) -> dict[str, int]:
        return {n: s.size() for n, s in self.sources.items()}

    def collect(self, snap: dict[str, int]) -> dict[str, str]:
        return {n: self.sources[n].read_from(off) for n, off in snap.items()}

    @classmethod
    def for_nodes(cls, nodes=("studio1", "studio2")) -> "LogReader":
        return cls({n: SSHNodeLog(n) for n in nodes})


def pick_window(windows: dict[str, str], prefer=("studio1", "studio2")) -> tuple[str, str]:
    """Prefer a node window that actually carries a ``turn reuse:`` line."""
    for node in prefer:
        txt = windows.get(node)
        if txt and RE_TURNREUSE.search(txt):
            return node, txt
    for node in prefer:
        if windows.get(node):
            return node, windows[node]
    any_node = next(iter(windows), None)
    return (any_node, windows.get(any_node, "")) if any_node else ("", "")


# -------------------------------------------------------------------- prompt build
_FILLER_SENTENCES = [
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

# long, deterministic, ~1000-token target, low EOS risk (same as phase19)
TASK_COUNT = "\n\nTask: List the integers from 1 to 450 in order, each on its own line, and nothing else."


def make_salt() -> str:
    return "SALT-" + os.urandom(8).hex()


def build_filler(target_tokens: int, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
                 seed: int = 20261007, task: str | None = None) -> str:
    """Deterministic filler sized by chars/token (NO salt inside -- a salt
    re-tokenizes the whole prompt; the salt goes on once, at the head)."""
    target_chars = int(target_tokens * chars_per_token)
    rng = __import__("random").Random(seed)
    parts: list[str] = []
    n = 0
    while n < target_chars:
        s = _FILLER_SENTENCES[rng.randrange(len(_FILLER_SENTENCES))]
        parts.append(s + " ")
        n += len(s) + 1
    body = "".join(parts)
    if task:
        body += task
    return body


def build_base(ctx_tokens: int, salt: str, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
               seed: int = 20261007, task: str | None = TASK_COUNT) -> str:
    """A base conversation head: SALT ONCE + filler (no delta text)."""
    return f"[SESSION-SALT {salt}]\n" + build_filler(ctx_tokens, chars_per_token, seed, task)


def build_delta(delta_tokens: int, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
                seed: int = 20261007) -> str:
    """A NEW unique delta text of ~D tokens.  Salt-free and seed-shifted so two
    reps of the same nominal size are NOT byte-identical (a reused filler would
    otherwise match a restored session)."""
    return "\n\n" + build_filler(delta_tokens, chars_per_token, seed + delta_tokens)


def base_messages(base: str) -> list[dict]:
    return [{"role": "user", "content": base}]


def delta_messages(base: str, reply: str, delta_text: str) -> list[dict]:
    """The real agentic branching shape: prefix contains the previous prompt."""
    return [
        {"role": "user", "content": base},
        {"role": "assistant", "content": reply},
        {"role": "user", "content": delta_text},
    ]


# ----------------------------------------------------------------------- wall math
def predict_wall_s(rows: int, rows_per_s: float = PRED_ROWS_PER_S,
                   factor: float = PRED_FACTOR, overhead_s: float = PRED_OVERHEAD_S) -> float:
    """PREREG 0c chunk discipline: rows / 200 rows/s * 1.3 + 15 s."""
    return rows / rows_per_s * factor + overhead_s


# --------------------------------------------------------------------- HTTP client
def stream_once(api_base: str, messages: list[dict], *, max_tokens: int = 1,
                temperature: float = 0, timeout: float = 3600.0,
                model: str = MODEL) -> dict:
    """One streamed chat completion.  Reads the ``: generation_stats`` SSE COMMENT
    frame (does NOT start with ``data:``).  reasoning_effort intentionally omitted."""
    url = api_base.rstrip("/") + "/v1/chat/completions"
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "stream": True}
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = first = last = None
    usage = stats = created = finish = None
    content_chars = 0
    content_parts: list[str] = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
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
            now = time.perf_counter()
            if created is None:
                created = obj.get("created")
            if usage is None and obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                txt = (ch.get("delta") or {}).get("content")
                if txt:
                    if ttft is None:
                        ttft, first = now - t0, now
                    content_chars += len(txt)
                    content_parts.append(txt)
                    last = now
    wall = time.perf_counter() - t0
    decode_s = (last - first) if (first is not None and last is not None and last > first) else None
    return {
        "wall_s": round(wall, 3),
        "ttft_s": round(ttft, 4) if ttft is not None else None,
        "decode_s": round(decode_s, 4) if decode_s else None,
        "content": "".join(content_parts),
        "content_chars": content_chars,
        "finish_reason": finish,
        "created": created,
        "usage": usage,
        "stats": stats,
    }


# ---------------------------------------------------------------- summarize utils
def stat_block(values) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "median": None, "min": None, "max": None}
    return {"n": len(vals), "median": round(statistics.median(vals), 3),
            "min": round(min(vals), 3), "max": round(max(vals), 3)}


def load_jsonl(paths: list[str]) -> list[dict]:
    out: list[dict] = []
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def _delta_cells(records: list[dict], ctx: str) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for r in records:
        if r.get("kind") != "delta" or r.get("ctx_label") != ctx:
            continue
        if r.get("collapsed") or r.get("invalid"):
            continue
        out.setdefault(r.get("delta_nominal"), []).append(r)
    return out


def _med(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def summarize(records: list[dict]) -> dict:
    """Build Table A + Table B and evaluate the three 0c decision rules.

    Records with ``collapsed`` (reuse < 0.9*base) or ``invalid`` (DEGRADED
    reference) are excluded from the tables and listed separately.
    """
    excluded_collapsed = [r for r in records if r.get("collapsed")]
    excluded_invalid = [r for r in records if r.get("invalid")]
    fresh_ref = [r for r in records if r.get("cell") == "fresh100k"]

    # ---- Table A: ctx ladder at the fixed 2048 delta + fresh reference
    table_a: list[dict] = []
    ctx_med: dict[str, float | None] = {}
    for label, _ in LADDER_CTXS:
        cells = _delta_cells(records, label)
        recs = cells.get(DELTA_LADDER_ROWS, [])
        rates = [rows_per_s_log(r) for r in recs]
        blk = stat_block(rates)
        ctx_med[label] = blk["median"]
        table_a.append({"ctx": label, "delta_rows": DELTA_LADDER_ROWS,
                        "actual_rows_median": _med([r.get("log_prefill") for r in recs]),
                        **blk})
    frates = [rows_per_s_ttft(r) for r in fresh_ref]
    table_a.append({"ctx": "fresh100k", "delta_rows": None,
                    "actual_rows_median": _med([r.get("prompt_tokens") for r in fresh_ref]),
                    **stat_block(frates)})

    # ---- Table B: delta-size sweep at 50K
    table_b: list[dict] = []
    sweep_med: dict[int, float | None] = {}
    by50 = _delta_cells(records, "50k")
    ref4096 = None
    tmp_b = []
    for d in DELTA_SWEEP_ROWS:
        recs = by50.get(d, [])
        rates = [rows_per_s_log(r) for r in recs]
        blk = stat_block(rates)
        sweep_med[d] = blk["median"]
        tmp_b.append({"delta_label": d,
                      "actual_rows_median": _med([r.get("log_prefill") for r in recs]),
                      **blk})
    ref4096 = sweep_med.get(4096)
    for row in tmp_b:
        r = row["median"]
        row["vs_4096"] = round(r / ref4096, 3) if (r is not None and ref4096) else None
        table_b.append(row)

    # ---- decision rules
    verdicts: list[str] = []
    r20, r110 = ctx_med.get("20k"), ctx_med.get("110k")
    if r20 and r110 is not None:
        degrade = (r20 - r110) / r20
        if degrade > CTX_DEPTH_DEGRADE:
            verdicts.append(f"CTX-DEPTH COST: rows/s degrades {degrade*100:.1f}% "
                            f"20K->110K (>{CTX_DEPTH_DEGRADE*100:.0f}%) at the "
                            f"{DELTA_LADDER_ROWS}-row delta")
        else:
            verdicts.append(f"ctx-depth benign: {degrade*100:.1f}% 20K->110K "
                            f"(<={CTX_DEPTH_DEGRADE*100:.0f}%)")
    else:
        verdicts.append("ctx-depth: INSUFFICIENT DATA (need 20k + 110k at 2048)")

    d256, d4096 = sweep_med.get(256), sweep_med.get(4096)
    if d256 and d4096:
        if d256 < FIXED_OVERHEAD_RATIO * d4096:
            verdicts.append(f"FIXED PER-CALL OVERHEAD: d256={d256:.1f} rows/s < "
                            f"{FIXED_OVERHEAD_RATIO*100:.0f}% of d4096={d4096:.1f} "
                            f"=> Phase 4 lever")
        else:
            verdicts.append(f"no fixed-overhead cliff: d256={d256:.1f} vs "
                            f"d4096={d4096:.1f} rows/s "
                            f"({d256/d4096*100:.0f}% of d4096)")
    else:
        verdicts.append("fixed-overhead: INSUFFICIENT DATA (need 50k d256 + d4096)")

    sm = [sweep_med[d] for d in DELTA_SWEEP_ROWS if sweep_med.get(d)]
    if len(sm) >= 2:
        spread = (max(sm) - min(sm)) / statistics.median(sm)
        if spread < FLAT_SPREAD:
            verdicts.append(f"FLAT across delta sizes: spread {spread*100:.1f}% "
                            f"<{FLAT_SPREAD*100:.0f}% => Phase 4 skipped, slope recorded")
        else:
            verdicts.append(f"delta-size slope present: spread {spread*100:.1f}% "
                            f"(>={FLAT_SPREAD*100:.0f}%)")
    else:
        verdicts.append("flat-vs-slope: INSUFFICIENT DATA (need >=2 sweep sizes)")

    if frates:
        med = _med(frates)
        if med is not None and med < FRESH_REF_MIN_ROWS_PER_S:
            verdicts.append(f"DEGRADED_REFERENCE: fresh-feed 100K = {med:.1f} rows/s "
                            f"< {FRESH_REF_MIN_ROWS_PER_S:.0f} => cluster degraded, "
                            f"every cell of that chunk INVALID")
        else:
            verdicts.append(f"fresh reference OK: {med:.1f} rows/s "
                            f">= {FRESH_REF_MIN_ROWS_PER_S:.0f}")
    else:
        verdicts.append("fresh reference: NOT RUN (no fresh100k record)")

    return {
        "table_a": table_a, "table_b": table_b, "verdicts": verdicts,
        "collapsed": excluded_collapsed, "invalid": excluded_invalid,
        "n_records": len(records),
    }
