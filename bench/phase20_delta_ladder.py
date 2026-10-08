#!/usr/bin/env python3
"""Phase-20 0c: delta-prefill ladder + fresh-feed reference (BRIEF L).

The PM runs the live chunks; this tool is built and proven OFFLINE against
``bench/phase20_tests/mock_exo_server.py`` (which replays real exo.log line
shapes).  It NEVER sends a generation request unless a caller runs a subcommand
against a live ``--api``; the cluster is READ-ONLY to this worker.

What it measures (plan 0c):
  (i)  prefill to ctx in {20K, 50K, 110K}, then a 2048-row delta; rows/s from
       the engine log (``[DSV41] turn reuse: ... prefill=M``).
  (ii) at 50K, deltas of {256, 1024, 4096, 8192} rows x reps.
  plus a fresh-feed 100K reference (falsifier: < 255 rows/s => DEGRADED).

Chunk discipline (PREREG 0c): predict each step's wall (rows/200*1.3 + 15 s);
if it would exceed the chunk's remaining wall, STOP before starting it, mark the
cell NOT_RUN_WALL_CAP.  ``register_own_request()`` immediately before every HTTP
request; ``guard.check()`` between requests; on ChunkAborted stop cleanly, write
what you have, exit 75.  A canary runs before each chunk.

Subcommands:
  pilot      4K base + {256,1024} deltas x1 rep  (live shape validation, ~1 min)
  chunk1     20K + 50K bases, 2048-row deltas x reps
  chunk2     110K base, 2048-row deltas x reps
  chunk3     50K base, delta-size sweep {256,1024,4096,8192} x reps
  fresh100k  one fresh cold 100K feed (the reference)
  summarize  rebuild Table A + Table B (markdown + CSV) + decision-rule verdicts

Options: --api URL --reps N --chars-per-token X --out-dir DIR --dry-run
         --max-wall S --calibrate --base-salt HEX --log-source FILE(s)
         --own-registry PATH (persistent own-request registry)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import phase20_common as C  # noqa: E402

DEFAULT_OUT_DIR = os.path.join("docs", "benchmarks", "phase20-throughput")
EXIT_OK = 0
EXIT_CHUNK_ABORTED = 75
EXIT_GUARD_FAILURE = 70
EXIT_DEGRADED = 3

# ---------------------------------------------------------------------- guard glue
GUARD_OVERRIDE = None      # tests inject a stub module here
STUB_GUARD_PATH = "/private/tmp/p20-guard/bench/phase20_guard.py"


class _StubGuardFailure(RuntimeError):
    pass


class _StubChunkAborted(_StubGuardFailure):
    pass


class _StubChunkGuard:
    """Minimal implementation of the GUARD-CONTRACT when the real guard has not
    landed.  Records own requests; check() raises if the caller set the abort
    flag (tests drive it).  Never touches the network or the cluster."""

    def __init__(self, label, *, max_wall_s=900.0, **_kw):
        self.label = label
        self.max_wall_s = max_wall_s
        self.aborted = False
        self.reason = None
        self.own_requests: list[float] = []
        self.t_start = None

    def __enter__(self):
        self.t_start = time.time()
        return self

    def __exit__(self, *exc):
        pass

    def register_own_request(self, t_start=None):
        self.own_requests.append(time.time() if t_start is None else t_start)

    def check(self):
        if self.aborted:
            raise _StubChunkAborted(self.reason or "ABORTED")

    def remaining_s(self) -> float:
        if self.t_start is None:
            return self.max_wall_s
        return self.max_wall_s - (time.time() - self.t_start)


class _StubGuardModule:
    GuardFailure = _StubGuardFailure
    ChunkAborted = _StubChunkAborted
    ChunkGuard = _StubChunkGuard

    @staticmethod
    def canary(nodes=("studio1", "studio2"), timeout_s=90.0):
        return {"ok": True, "state": "stub", "per_node": {}, "median": {}}

    @staticmethod
    def idle_check(*a, **k):
        return {"ok": True, "reasons": [], "detail": {"stub": True}}


def resolve_guard():
    """Real guard if importable / landed, else the internal stub."""
    if GUARD_OVERRIDE is not None:
        return GUARD_OVERRIDE
    for name in ("phase20_guard",):
        if name in sys.modules:
            return sys.modules[name]
    try:
        import phase20_guard as g                     # noqa: E402
        return g
    except ImportError:
        pass
    if os.path.exists(STUB_GUARD_PATH):
        spec = importlib.util.spec_from_file_location("phase20_guard", STUB_GUARD_PATH)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            sys.modules["phase20_guard"] = mod
            return mod
        except Exception:
            pass
    return _StubGuardModule


# ---------------------------------------------------------- own-request registry
DEFAULT_REGISTRY_NAME = "own_requests.jsonl"


def load_own_requests(path: str) -> list[float]:
    """Read the guard's persistent own-request registry and return its ``t`` epochs.

    The registry is the JSONL file ``ChunkGuard(registry_path=...)`` appends to:
    one object per line, ``{"label": <str>, "t": <float epoch>}``.  A missing file
    (first run) or a malformed/blank line is tolerated and skipped -- this reader
    must never fail a chunk.  Registrations are made immediately BEFORE each POST,
    so an epoch read back here can only correspond to a POST already on a node.
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
                    obj = json.loads(line)
                    t = obj["t"]
                except (json.JSONDecodeError, TypeError, KeyError):
                    continue
                try:
                    out.append(float(t))
                except (TypeError, ValueError):
                    continue
    except OSError:
        return out
    return out


# ------------------------------------------------------------------------- plan
def _chunk_specs(chunk: str, reps: int, chars_per_token: float,
                 base_salt: str | None, fresh_tokens: int = C.FRESH_REF_TOKENS) -> list[dict]:
    """Steps for a chunk.  A step: {kind(base|delta|fresh), ctx_label,
    ctx_nominal, delta_nominal, cell, salt}.

    Every step carries the salt of its ctx's base so that a delta's 3-message
    head is byte-identical to the base that produced it (the branching shape)."""
    steps: list[dict] = []
    run_salt = base_salt or C.make_salt()

    def salt_for(label: str) -> str:
        # base_salt given verbatim for the single-ctx chunks; chunk1's two ctxs
        # get distinct deterministic salts derived from the run salt.
        return f"{run_salt}-{label}"

    def base(label, ctx):
        steps.append({"kind": "base", "ctx_label": label, "ctx_nominal": ctx,
                      "delta_nominal": None, "salt": salt_for(label),
                      "cell": f"ctx{label}_base"})

    def deltas(label, ctx, rows, reps_n):
        for i in range(reps_n):
            steps.append({"kind": "delta", "ctx_label": label, "ctx_nominal": ctx,
                          "delta_nominal": rows, "rep": i,
                          "salt": salt_for(label), "cell": f"ctx{label}_d{rows}"})

    if chunk == "pilot":
        base("4k", 4000)
        deltas("4k", 4000, 256, 1)
        deltas("4k", 4000, 1024, 1)
    elif chunk == "chunk1":
        base("20k", 20000)
        deltas("20k", 20000, C.DELTA_LADDER_ROWS, reps)
        base("50k", 50000)
        deltas("50k", 50000, C.DELTA_LADDER_ROWS, reps)
    elif chunk == "chunk2":
        base("110k", 110000)
        deltas("110k", 110000, C.DELTA_LADDER_ROWS, reps)
    elif chunk == "chunk2a":
        # split of chunk2: the 110K cold feed alone (~730 s + canary) is the
        # bulk; run it alone so the chunk fits the 15-min cap.
        base("110k", 110000)
    elif chunk == "chunk2b":
        # the 3 x 2048 delta reps against the (already resident) 110K base.
        deltas("110k", 110000, C.DELTA_LADDER_ROWS, reps)
    elif chunk == "chunk3":
        base("50k", 50000)
        for rows in C.DELTA_SWEEP_ROWS:
            deltas("50k", 50000, rows, reps)
    elif chunk == "fresh100k":
        steps.append({"kind": "fresh", "ctx_label": "fresh100k",
                      "ctx_nominal": fresh_tokens, "delta_nominal": None,
                      "salt": run_salt, "cell": "fresh100k"})
    else:
        raise SystemExit(f"unknown chunk {chunk!r}")
    return steps


def _step_rows(step: dict) -> int:
    if step["kind"] in ("base", "fresh"):
        return step["ctx_nominal"]
    return step["delta_nominal"]


def plan_chunk(chunk: str, *, reps: int = 3, chars_per_token: float = C.DEFAULT_CHARS_PER_TOKEN,
               max_wall_s: float = C.CHUNK_MAX_WALL_S, base_salt: str | None = None,
               fresh_tokens: int = C.FRESH_REF_TOKENS) -> dict:
    steps = _chunk_specs(chunk, reps, chars_per_token, base_salt, fresh_tokens)
    rows_out = []
    total = 0.0
    breached = None
    for st in steps:
        rows = _step_rows(st)
        wall = C.predict_wall_s(rows)
        total += wall
        over = total > max_wall_s
        rows_out.append({**{k: st[k] for k in ("kind", "cell", "ctx_label",
                                               "delta_nominal")},
                         "rows": rows, "pred_wall_s": round(wall, 1),
                         "cum_wall_s": round(total, 1),
                         "wall_cap": "NOT_RUN_WALL_CAP" if over else "ok"})
        if over and breached is None:
            breached = st["cell"]
    return {"chunk": chunk, "max_wall_s": max_wall_s, "steps": rows_out,
            "pred_total_s": round(total, 1),
            "fits": total <= max_wall_s, "first_breach": breached,
            "canary_estimate_s": 90.0}


# --------------------------------------------------------------------- log helper
def _read_window(log_reader, snap: dict) -> tuple[str, dict]:
    windows = log_reader.collect(snap)
    node, text = C.pick_window(windows)
    return node, C.analyze_window(text)


# ------------------------------------------------------------------------ runner
def _make_log_reader(args) -> C.LogReader:
    if args.log_source:
        srcs = {}
        for spec in args.log_source:
            name, _, path = spec.partition("=")
            srcs[name] = C.LocalFileLog(path)
        return C.LogReader(srcs)
    return C.LogReader.for_nodes(("studio1", "studio2"))


def run_chunk(chunk: str, args) -> int:
    guard_mod = resolve_guard()
    out_dir = args.out_dir
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    out_path = os.path.join(raw_dir, f"delta_ladder.{chunk}.jsonl")
    ref_path = os.path.join(raw_dir, "fresh_ref.json")
    # FIX 4: the per-ctx base ACTUAL rows sidecar (the base's usage.prompt_tokens).
    # A delta chunk that is a SEPARATE process (chunk2b) loads this so its classifier
    # compares reuse against the base's real depth, not the nominal ctx.
    base_actual_path = os.path.join(raw_dir, C.BASE_ACTUAL_NAME)

    steps = _chunk_specs(chunk, args.reps, args.chars_per_token, args.base_salt,
                         getattr(args, "fresh_tokens", C.FRESH_REF_TOKENS))

    # -- persistent own-request registry: an aborted/killed chunk's registrations
    #    must not block a later chunk's ChunkGuard entry idle-check for min_idle_s.
    #    Default lives beside the chunk JSONL + guard.json in <out>/raw/.
    registry_path = (getattr(args, "own_registry", None)
                     or os.path.join(raw_dir, DEFAULT_REGISTRY_NAME))
    own_requests = load_own_requests(registry_path)

    # -- dry run: print plan + predicted walls, send nothing
    if args.dry_run:
        plan = plan_chunk(chunk, reps=args.reps, chars_per_token=args.chars_per_token,
                          max_wall_s=args.max_wall, base_salt=args.base_salt,
                          fresh_tokens=getattr(args, "fresh_tokens", C.FRESH_REF_TOKENS))
        _print_plan(plan)
        return EXIT_OK

    # -- DEGRADED_REFERENCE gate: a recorded fresh reference < 255 => do not record
    if chunk != "fresh100k" and os.path.exists(ref_path):
        try:
            ref = json.load(open(ref_path))
        except Exception:
            ref = {}
        # FIX 3: prefer the engine-derived fresh rate (prompt_tps), fall back to the
        # legacy ttft-derived key; guard against None before formatting.
        rate = C.rows_per_s_fresh(ref)
        if rate is None:
            rate = ref.get("rows_per_s_ttft")
        if rate is not None and rate < C.FRESH_REF_MIN_ROWS_PER_S:
            print(f"DEGRADED_REFERENCE: recorded fresh reference {rate:.1f} < "
                  f"{C.FRESH_REF_MIN_ROWS_PER_S:.0f} rows/s -> refusing to record "
                  f"chunk {chunk}", file=sys.stderr)
            return EXIT_DEGRADED

    # -- canary before the chunk
    try:
        can = guard_mod.canary()
        state = can.get("state") if isinstance(can, dict) else getattr(can, "state", None)
        print(f"canary: {state}", flush=True)
        if state == "degraded":
            print("canary degraded -> aborting chunk", file=sys.stderr)
            return EXIT_GUARD_FAILURE
    except Exception as exc:                       # canary is best-effort offline
        print(f"canary skipped: {exc}", file=sys.stderr)

    log_reader = _make_log_reader(args)
    salt = args.base_salt or C.make_salt()
    records: list[dict] = []
    base_reply: dict[str, str] = {}
    # FIX 4: seed the base's ACTUAL depth from the sidecar.  A delta chunk (chunk2b)
    # runs in a SEPARATE process from its base (chunk2a), so without this the
    # classifier would fall back to the NOMINAL ctx (110000) and wrongly flag the
    # legitimate 110k deltas (reuse=100352, one rung below the base's real 100503).
    base_actual: dict[str, int] = C.load_base_actual(base_actual_path)
    base_actual_before = dict(base_actual)
    chunk_start = time.monotonic()
    pred_cum = 0.0

    with open(out_path, "a") as out_fh:
        try:
            with guard_mod.ChunkGuard(chunk, max_wall_s=args.max_wall,
                                      log_dir=raw_dir,
                                      own_requests=own_requests,
                                      registry_path=registry_path) as guard:
                for st in steps:
                    guard.check()
                    rows = _step_rows(st)
                    # FIX 2(b): wall pre-check on TRUE elapsed wall only.  The
                    # previous check added ``pred_cum`` (a sum of predicted walls)
                    # on top of the real elapsed wall, double-counting time already
                    # spent and blocking steps the chunk could actually afford.
                    pred = C.predict_wall_s(rows)
                    elapsed = time.monotonic() - chunk_start
                    remaining = args.max_wall - elapsed
                    if elapsed + pred > args.max_wall:
                        rec = _blocked_record(chunk, st, rows, pred, remaining)
                        records.append(rec)
                        out_fh.write(json.dumps(rec) + "\n")
                        out_fh.flush()
                        print(f"NOT_RUN_WALL_CAP: {st['cell']} pred {pred:.0f}s + "
                              f"used {elapsed:.0f}s > cap {args.max_wall:.0f}s",
                              file=sys.stderr)
                        break

                    msgs, cell_salt = _build_messages(st, salt, base_reply,
                                                      args.chars_per_token)
                    snap = log_reader.snapshot()
                    t_start = time.time()
                    guard.register_own_request(t_start)
                    try:
                        r = C.stream_once(args.api, msgs, max_tokens=1,
                                          temperature=0, timeout=args.timeout)
                    except Exception as exc:
                        rec = _error_record(chunk, st, rows, exc)
                        records.append(rec)
                        out_fh.write(json.dumps(rec) + "\n")
                        out_fh.flush()
                        print(f"request error on {st['cell']}: {exc}", file=sys.stderr)
                        break

                    node, logf = _read_window(log_reader, snap)
                    rec = _record(chunk, st, rows, r, logf, node, t_start,
                                  args.chars_per_token, base_actual)
                    records.append(rec)
                    out_fh.write(json.dumps(rec) + "\n")
                    out_fh.flush()
                    # FIX 2(b): ``pred_cum`` is retained ONLY as a predicted-elapsed
                    # report (never a gate) -- the true wall is the elapsed clock.
                    pred_cum += max(pred, r.get("wall_s") or 0.0)

                    if st["kind"] in ("base",) and r.get("content") is not None:
                        base_reply[st["ctx_label"]] = r["content"]
                        # FIX 2(c): remember the base's ACTUAL depth (usage
                        # prompt_tokens) -- the delta classifier measures reuse
                        # against it, not the nominal ctx.  A cold base emits no
                        # ``turn reuse:`` line, so gate on prompt_tokens only.
                        if rec.get("prompt_tokens") is not None:
                            base_actual[st["ctx_label"]] = rec["prompt_tokens"]
                        elif rows:
                            base_actual[st["ctx_label"]] = rows
                        # FIX 4: persist the base's ACTUAL depth so a LATER delta
                        # chunk in a separate process can read it (merge-safe).
                        if base_actual != base_actual_before:
                            C.save_base_actual(base_actual_path, base_actual)
                            base_actual_before = dict(base_actual)
                    if st["kind"] == "fresh":
                        with open(ref_path, "w") as rf:
                            json.dump(rec, rf)
                        # FIX 3: the fresh rate is the engine's own prompt_tps when
                        # present (a direct prefill rate that does NOT depend on the
                        # client seeing a content delta), else the ttft cross-check.
                        # Guard None before formatting: the pre-fix code crashed with
                        # ``TypeError: unsupported format string passed to
                        # NoneType.__format__`` here whenever the fresh rate was None.
                        rate = C.rows_per_s_fresh(rec)
                        if rate is None:
                            rate = rec.get("rows_per_s_ttft")
                        if rate is not None and rate < C.FRESH_REF_MIN_ROWS_PER_S:
                            print(f"DEGRADED_REFERENCE: fresh100k "
                                  f"{rate:.1f} rows/s < "
                                  f"{C.FRESH_REF_MIN_ROWS_PER_S:.0f}", file=sys.stderr)
                            return EXIT_DEGRADED
                        print(f"fresh100k rows/s = "
                              f"{rate:.1f}" if rate is not None
                              else "fresh100k rows/s = None (no prompt_tps, no ttft)",
                              flush=True)
        except guard_mod.ChunkAborted as exc:
            print(f"ChunkAborted: {exc}", file=sys.stderr)
            return EXIT_CHUNK_ABORTED
        except guard_mod.GuardFailure as exc:
            print(f"GuardFailure: {exc}", file=sys.stderr)
            return EXIT_GUARD_FAILURE

    print(f"wrote {len(records)} records -> {out_path}", flush=True)
    return EXIT_OK


def _build_messages(st, salt, base_reply, chars_per_token):
    step_salt = st.get("salt", salt)
    if st["kind"] in ("base", "fresh"):
        base = C.build_base(st["ctx_nominal"], step_salt, chars_per_token,
                            seed=20261007 + st["ctx_nominal"])
        return C.base_messages(base), base
    # delta: branch from the base with a NEW unique delta text.  FIX 2(a): the rep
    # index folds into the delta's seed so two reps of the same nominal size build
    # DISTINCT text -- otherwise rep 2..N is byte-identical to rep 1 and the engine
    # serves it from cache (prefill=0), measuring nothing.
    base = C.build_base(st["ctx_nominal"], step_salt, chars_per_token,
                        seed=20261007 + st["ctx_nominal"])
    reply = base_reply.get(st["ctx_label"], "ok")
    dtext = C.build_delta(st["delta_nominal"], chars_per_token,
                          seed=30370000 + st["ctx_nominal"] + st["delta_nominal"],
                          nonce=st.get("rep", 0) or 0)
    return C.delta_messages(base, reply, dtext), dtext


def _record(chunk, st, rows, r, logf, node, t_start, chars_per_token, base_actual):
    usage = r.get("usage") or {}
    # FIX 3: prefer usage.prompt_tokens; fall back to the generation_stats frame's
    # prompt_tokens (that frame is the only source when the response carries no
    # usage chunk -- the live reasoning stream returned content='').
    prompt_tokens = usage.get("prompt_tokens")
    if prompt_tokens is None:
        prompt_tokens = r.get("prompt_tokens")
    out = {
        "chunk": chunk,
        "cell": st["cell"],
        "rep": st.get("rep"),
        "kind": "fresh" if st["kind"] in ("base", "fresh") else "delta",
        "ctx_nominal": st["ctx_nominal"],
        "ctx_label": st["ctx_label"],
        "delta_nominal": st["delta_nominal"],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": usage.get("completion_tokens"),
        "ttft_s": r.get("ttft_s"),
        "wall_s": r.get("wall_s"),
        "decode_s": r.get("decode_s"),
        "content_chars": r.get("content_chars"),
        # FIX 3: engine's own direct prefill rate + token/cache fields, surfaced
        # verbatim from the ``: generation_stats`` SSE comment frame.
        "prompt_tps": r.get("prompt_tps"),
        "generation_tokens": r.get("generation_tokens"),
        "prefix_cache_hit": r.get("prefix_cache_hit"),
        "log_prefill": logf.get("log_prefill"),
        "reuse": logf.get("reuse"),
        "rewind": logf.get("rewind"),
        "prefill_controls_rows": logf.get("prefill_controls_rows"),
        "prefill_s_log": logf.get("prefill_s_log"),
        "has_turn_reuse": logf.get("has_turn_reuse"),
        "log_node": node,
        "t_start": round(t_start, 3),
        "chars_per_token": chars_per_token,
        "invalid": False,
        "cache_hit": False,
        "notes": "",
    }
    out["rows_per_s_log"] = C.rows_per_s_log({**out, "log_prefill": out["log_prefill"],
                                              "prefill_s_log": out["prefill_s_log"]})
    out["rows_per_s_ttft"] = C.rows_per_s_ttft(
        {"log_prefill": out["log_prefill"], "prompt_tokens": prompt_tokens,
         "ttft_s": out["ttft_s"]})
    # FIX 3: the fresh-feed rate -- the engine's own prompt_tps when present, else
    # the TTFT cross-check.  Non-null for a fresh feed even though content is ''.
    out["rows_per_s_fresh"] = C.rows_per_s_fresh(
        {"prompt_tps": out["prompt_tps"], "log_prefill": out["log_prefill"],
         "prompt_tokens": prompt_tokens, "ttft_s": out["ttft_s"]})
    if out["kind"] == "delta":
        # FIX 2(a): prefill=0 (reuse == prompt) means the engine served this exact
        # prompt from cache -- the rep measured nothing and must be excluded, not
        # recorded as a 0-row delta.
        if C.is_cache_hit(out):
            out["cache_hit"] = True
            out["collapsed"] = False
            out["notes"] = (f"cache hit: prefill=0 reuse={out['reuse']} == "
                            f"prompt={prompt_tokens} (full-cache hit, no rows fed)")
            # FIX 3: corroborate the log-derived rule with the engine's own flag.
            flag = out["prefix_cache_hit"]
            if flag is not None and not C.is_full_cache_hit_flag(flag):
                out["notes"] += f" | prefix_cache_hit={flag!r} (log says full hit)"
        else:
            # FIX 2(c): collapse is measured against the base's OWN actual depth
            # (ladder rungs are legitimate), not the nominal ctx.
            base_rows = base_actual.get(st["ctx_label"], st["ctx_nominal"])
            reuse = out["reuse"]
            if C.is_collapsed(reuse, base_rows):
                out["collapsed"] = True
                out["notes"] = (f"collapsed: reuse={reuse} < base_actual-{C.LADDER_RUNG}"
                                f"={base_rows - C.LADDER_RUNG}"
                                if reuse is not None else "collapsed: no turn-reuse line")
            else:
                out["collapsed"] = False
            # FIX 3 corroboration: the engine flag claiming a full hit while the log
            # shows rows fed is a discrepancy -- surfaced, never used to reclassify.
            flag = out["prefix_cache_hit"]
            if C.is_full_cache_hit_flag(flag):
                out["notes"] = (out["notes"] + " | " if out["notes"] else "") + \
                    f"prefix_cache_hit={flag!r} but log prefill={out['log_prefill']}"
    else:
        out["collapsed"] = False
        if out["has_turn_reuse"] and (out["reuse"] or 0) > 0:
            out["notes"] = "base feed reused (resident conversation matched)"
    return out


def _blocked_record(chunk, st, rows, pred, remaining):
    return {
        "chunk": chunk, "cell": st["cell"], "rep": st.get("rep"),
        "kind": "fresh" if st["kind"] in ("base", "fresh") else "delta",
        "ctx_nominal": st["ctx_nominal"], "ctx_label": st["ctx_label"],
        "delta_nominal": st["delta_nominal"], "invalid": True,
        "not_run": "NOT_RUN_WALL_CAP",
        "notes": f"pred_wall {pred:.0f}s > remaining {remaining:.0f}s",
    }


def _error_record(chunk, st, rows, exc):
    rec = {
        "chunk": chunk, "cell": st["cell"], "rep": st.get("rep"),
        "kind": "fresh" if st["kind"] in ("base", "fresh") else "delta",
        "ctx_nominal": st["ctx_nominal"], "ctx_label": st["ctx_label"],
        "delta_nominal": st["delta_nominal"], "invalid": True,
        "notes": f"request error: {exc}",
    }
    return rec


# ------------------------------------------------------------------------- print
def _print_plan(plan: dict) -> None:
    print(f"# {plan['chunk']}  (dry-run: nothing sent)")
    print(f"# max_wall={plan['max_wall_s']:.0f}s  canary_est={plan['canary_estimate_s']:.0f}s")
    hdr = f"{'cell':<18}{'kind':<7}{'rows':>8}{'pred_wall':>11}{'cum':>9}  status"
    print(hdr)
    print("-" * len(hdr))
    for s in plan["steps"]:
        print(f"{s['cell']:<18}{s['kind']:<7}{s['rows']:>8}{s['pred_wall_s']:>10.1f}s"
              f"{s['cum_wall_s']:>8.1f}s  {s['wall_cap']}")
    total_incl_canary = plan["pred_total_s"] + plan["canary_estimate_s"]
    print("-" * len(hdr))
    print(f"pred_total (steps)      = {plan['pred_total_s']:.1f}s "
          f"({plan['pred_total_s']/60:.1f} min)")
    print(f"+ canary estimate       = {total_incl_canary:.1f}s "
          f"({total_incl_canary/60:.1f} min)")
    print(f"fits max_wall({plan['max_wall_s']:.0f}s) incl canary: "
          f"{total_incl_canary <= plan['max_wall_s']}")
    if plan["first_breach"]:
        print(f"first_breach = {plan['first_breach']}  "
              f"=> split the chunk at this step (earlier steps fit)")


# -------------------------------------------------------------------- summarize
def summarize_files(args) -> int:
    raw_dir = os.path.join(args.out_dir, "raw")
    paths = sorted(
        os.path.join(raw_dir, f) for f in os.listdir(raw_dir)
        if f.startswith("delta_ladder.") and f.endswith(".jsonl")
    ) if os.path.isdir(raw_dir) else []
    records = C.load_jsonl(paths)
    # FIX 4: prefer the persisted per-ctx base ACTUAL rows sidecar (the base ran in a
    # separate process); summarize() falls back to the sibling base/fresh record in
    # the raw JSONL when a ctx is absent here.
    base_actual = C.load_base_actual(os.path.join(raw_dir, C.BASE_ACTUAL_NAME))
    summary = C.summarize(records, base_actual=base_actual)
    _emit_summary(summary, records, args.out_dir, paths)
    return EXIT_OK


def _emit_summary(summary, records, out_dir, paths) -> None:
    os.makedirs(out_dir, exist_ok=True)
    md = ["# Phase-20 0c delta ladder - summary", ""]
    md.append(f"records: {summary['n_records']} from {len(paths)} JSONL file(s)")
    md.append("")
    md.append("## Table A - ctx ladder (2048-row delta) + fresh reference")
    md.append("| ctx | delta_rows | actual_rows(med) | n | rows/s med | min | max |")
    md.append("|---|---|---|---|---|---|---|")
    for r in summary["table_a"]:
        md.append(f"| {r['ctx']} | {r['delta_rows']} | {r['actual_rows_median']} | "
                  f"{r['n']} | {r['median']} | {r['min']} | {r['max']} |")
    md.append("")
    md.append("## Table B - delta-size sweep at 50K")
    md.append("| delta_label | actual_rows(med) | n | rows/s med | min | max | vs_4096 |")
    md.append("|---|---|---|---|---|---|---|")
    for r in summary["table_b"]:
        md.append(f"| {r['delta_label']} | {r['actual_rows_median']} | {r['n']} | "
                  f"{r['median']} | {r['min']} | {r['max']} | {r['vs_4096']} |")
    md.append("")
    md.append("## Decision-rule verdicts")
    for v in summary["verdicts"]:
        md.append(f"- {v}")
    if summary["collapsed"]:
        md.append("")
        md.append("## Collapsed reps (excluded)")
        for r in summary["collapsed"]:
            md.append(f"- {r.get('cell')} rep{r.get('rep')}: reuse={r.get('reuse')} "
                      f"log_prefill={r.get('log_prefill')} ({r.get('notes')})")
    if summary["invalid"]:
        md.append("")
        md.append("## Invalid / not-run cells")
        for r in summary["invalid"]:
            md.append(f"- {r.get('cell')}: {r.get('notes')}")
    if summary.get("cache_hits"):
        md.append("")
        md.append("## Cache-hit reps (excluded -- measured nothing)")
        for r in summary["cache_hits"]:
            md.append(f"- {r.get('cell')} rep{r.get('rep')}: prefill=0 "
                      f"reuse={r.get('reuse')} prompt={r.get('prompt_tokens')} "
                      f"({r.get('notes')})")
    text = "\n".join(md) + "\n"
    md_path = os.path.join(out_dir, "delta_ladder.summary.md")
    with open(md_path, "w") as fh:
        fh.write(text)

    csv_path = os.path.join(out_dir, "delta_ladder.summary.csv")
    with open(csv_path, "w") as fh:
        fh.write("table,key,delta_rows,actual_rows_median,n,median,min,max,vs_4096\n")
        for r in summary["table_a"]:
            fh.write(f"A,{r['ctx']},{r['delta_rows']},{r['actual_rows_median']},"
                     f"{r['n']},{r['median']},{r['min']},{r['max']},\n")
        for r in summary["table_b"]:
            fh.write(f"B,{r['delta_label']},{r['delta_label']},{r['actual_rows_median']},"
                     f"{r['n']},{r['median']},{r['min']},{r['max']},{r['vs_4096']}\n")
    print(text, flush=True)
    print(f"wrote {md_path}\nwrote {csv_path}", flush=True)


# ------------------------------------------------------------------------ CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["pilot", "chunk1", "chunk2", "chunk2a",
                                        "chunk2b", "chunk3", "fresh100k", "summarize"])
    ap.add_argument("--api", default=C.DEFAULT_API)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--chars-per-token", type=float, default=C.DEFAULT_CHARS_PER_TOKEN,
                    dest="chars_per_token")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, dest="out_dir")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run")
    ap.add_argument("--max-wall", type=float, default=C.CHUNK_MAX_WALL_S, dest="max_wall")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--base-salt", default=None, dest="base_salt")
    ap.add_argument("--fresh-tokens", type=int, default=C.FRESH_REF_TOKENS,
                    dest="fresh_tokens", help="fresh100k feed size (offline tests)")
    ap.add_argument("--log-source", action="append", default=None, dest="log_source",
                    help="node=path (repeat); the offline seam instead of ssh")
    ap.add_argument("--own-registry", default=None, dest="own_registry",
                    help="persistent own-request registry JSONL "
                         "(default: <out-dir>/raw/" + DEFAULT_REGISTRY_NAME + ")")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "summarize":
        return summarize_files(args)
    return run_chunk(args.command, args)


if __name__ == "__main__":
    raise SystemExit(main())
