#!/usr/bin/env python3
"""Phase-19 LIVE interleaved gamma matrix (per-request spec_gamma sweep).

WHAT IT MEASURES
----------------
deploy/next14-gamma adds a per-request speculative draft length to the dsv41
engine: ``ChatCompletionRequest.spec_gamma`` (int, clamped to [1,6]); when the
field is absent the engine uses its default gamma 3 (byte-identical to prod).
The streamed ``: generation_stats`` SSE COMMENT frame (a line beginning
``: generation_stats {`` -- NOT a ``data:`` line) carries PROCESS-CUMULATIVE
counters, of which this driver reads:

  mtp_cycles_cumulative              sum over decode rounds of gamma-per-round
  mtp_accepted_drafts_cumulative     sum over rounds of accepted draft rows
  mtp_accepted_histogram_cumulative  7 ints; index k = number of rounds that
                                     accepted EXACTLY k drafts (k = 0..6)
  prompt_tps                         prefill tok/s for this request

Because the counters accumulate since process start, the delta between two
successive requests is that request's per-round behaviour.  A single running
(prev_cycles, prev_accepted, prev_hist) is therefore carried across EVERY
request in the run (all arms, all workloads) and each rep is derived as a delta
of it -- never reset per arm.

THE MATRIX (why it is interleaved)
----------------------------------
Each workload (``benign`` = synthetic filler @ --depth tokens; ``agentic`` = the
real Hermes session 20261007_092009_9a2ed7 replayed from state.db) is measured
over several gamma arms.  Arms are interleaved round-robin
(``for r in range(reps): for g in gammas: one rep``) instead of run in blocks, so
slow thermal / contention drift on the shared cluster spreads across arms rather
than loading onto whichever arm ran last.  Each workload opens with ONE cold rep (the workload's fixed-salt prompt, uncached
the first time) to bring the engine/caches up; every later rep reuses that
workload's FIXED salt and is a prefix-cache hit.  Cold reps are valid data but are
excluded from arm statistics.

HOW TO READ A REP RECORD (checkpoint JSONL, one object per line)
----------------------------------------------------------------
  key           "<workload>:<gamma>:<seq>"; seq 0 = cold, 1..reps = warm
  is_cold       cold rep (excluded from arm stats) / warm arm rep
  rounds        d_cycles / gamma                 (decode rounds this request)
  total_rounds  sum(hist_delta)                  (independent round count; ~= rounds)
  mean_accepted gamma * d_accepted / d_cycles    (drafts accepted per round)
  decode_tps    (completion_tokens - 1) / decode_s
  ms_per_round  decode_s * 1000 / rounds         (wall cost of one MTP round)
  gamma_implied d_cycles / rounds_est, where
                rounds_est = completion_tokens / (1 + d_acc / d_cycles)
                -> the effective draft length the engine actually used
  hist_delta    7 ints: this request's rounds grouped by accepted-k
  per_position  p1..pg (g = gamma): p_k = sum(hist_delta[k:]) / total_rounds,
                the survival curve P(a round accepts >= k drafts)
  drift_probe   true on the first benign rep (the young-process 100 K round-wall
                probe used to classify process-state vs host/thermal drift)
  guard_ok      idle-guard verdict captured immediately before the request

MODES
-----
  plan        build both prompts OFFLINE from state.db, print the exact rep
              schedule + token estimates, hit NOTHING (no api, no ssh)
  fieldcheck  cheap LIVE proof the field is honoured: one short request WITH
              spec_gamma=4 and one WITHOUT, compare their histogram deltas
  warmup      sleep, send one throwaway request, prime the cumulative deltas, exit
  full        the interleaved matrix (default)

IDLE-GUARD
----------
Before EVERY rep this driver reuses ``bench/run_campaign.py``'s ``guard()``: the
state.db MAX(ended_at) for provider='custom' must be > 600 s old AND the newest
FOREIGN ``POST /v1/chat/completions`` line in the cluster log must be > 600 s old
(our own posts are matched by the response ``created`` epoch vs the local send
epoch, tolerance 5 s).  We never bench through foreign traffic: a guard failure
flushes the checkpoint and exits 4.  >= 3 consecutive request errors exit 5.

stdlib only (urllib/json/os/sqlite3/subprocess/statistics/time/argparse); no numpy.

Usage:
  python3 bench/phase19_gamma_matrix.py --mode plan --workload both
  python3 bench/phase19_gamma_matrix.py --mode warmup
  python3 bench/phase19_gamma_matrix.py --mode fieldcheck --no-warmup
  python3 bench/phase19_gamma_matrix.py --mode full --gammas 3,4,5 --reps 3
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

API_BASE = "http://macstudio-m4-1.tail19c543.ts.net:52415"
DEFAULT_OUT_DIR = os.path.join("docs", "benchmarks", "phase19-latency", "raw")
DEFAULT_GAMMAS = "3,4,5"
GUARD_STALE_S = 600.0
MAX_CONSEC_ERRORS = 3
WARMUP_SLEEP_S = 60.0
BENIGN_TASK = "count"
# fixed per-workload salts so resume / re-runs hit the SAME server prefix
SALTS = {"benign": "b1a2c3d4", "agentic": "e5f6a7b8"}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# sibling harness: prompt builders + stream_once + derive (do NOT duplicate them)
M = _load("phase19_agentic_measure", os.path.join(HERE, "phase19_agentic_measure.py"))
# sibling campaign driver: reuse its exact idle-guard
CAMP = _load("run_campaign", os.path.join(HERE, "run_campaign.py"))


# ------------------------------------------------------------------ stats utils
def pct(values, q):
    """Linear-interpolation percentile (numpy default), stdlib only."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return round(vals[0], 4)
    pos = (q / 100.0) * (len(vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    return round(vals[lo] + (vals[hi] - vals[lo]) * (pos - lo), 4)


def iqr(values):
    return [pct(values, 25), pct(values, 75)]


def median(values):
    vals = [v for v in values if v is not None]
    return round(statistics.median(vals), 4) if vals else None


# ------------------------------------------------------------ prompts / schedule
def chat_url(api):
    api = api.rstrip("/")
    return api if api.endswith("/v1/chat/completions") else api + "/v1/chat/completions"


def build_one(workload, salt, depth, char_budget):
    """Build one prompt (workload) for a given salt. Reuses M's builders."""
    if workload == "benign":
        prompt = M.build_benign_prompt(depth, salt, BENIGN_TASK)
        meta = {"workload": workload, "salt": salt, "prompt_chars": len(prompt),
                "depth_target": depth, "pred_tokens_3p48": int(len(prompt) / 3.479)}
        return prompt, meta
    prompt, meta = M.build_agentic_prompt(salt, char_budget)
    meta = dict(meta)
    meta["workload"] = workload
    meta["salt"] = salt
    return prompt, meta


def build_schedule(workloads, gammas, reps):
    """Exact rep order: one cold rep per workload, then round-robin warm arms."""
    sched = []
    for wl in workloads:
        sched.append({"workload": wl, "gamma": gammas[0], "seq": 0, "is_cold": True})
        for r in range(reps):
            for g in gammas:
                sched.append({"workload": wl, "gamma": g, "seq": 1 + r, "is_cold": False})
    return sched


def first_benign_index(schedule):
    """The benign cold rep is the first benign entry: the young-process 100 K
    round-wall measurement used for drift classification (process vs host)."""
    return next((i for i, s in enumerate(schedule) if s["workload"] == "benign"), None)


# ------------------------------------------------------------------ derivation
def derive_rep(rec, prev_cyc, prev_acc, prev_hist, gamma):
    """Per-rep record from M.derive() + histogram deltas + survival curve."""
    d = M.derive(rec, prev_cyc, prev_acc, gamma)
    hist = d.get("hist_cum")
    hist_delta = None
    total_rounds = None
    per_position = None
    if hist is not None and prev_hist is not None and len(hist) == len(prev_hist):
        hist_delta = [int(h) - int(p) for h, p in zip(hist, prev_hist, strict=False)]
        total_rounds = sum(hist_delta)
        if total_rounds > 0:
            per_position = {f"p{k}": round(sum(hist_delta[k:]) / total_rounds, 5)
                            for k in range(1, gamma + 1)}
    return {
        "prompt_tokens": d.get("prompt_tokens"),
        "completion_tokens": d.get("completion_tokens"),
        "reasoning_tokens": d.get("reasoning_tokens"),
        "ttft_s": rec.get("ttft_s"),
        "decode_s": rec.get("decode_s"),
        "wall_s": rec.get("wall_s"),
        "decode_tps": d.get("decode_tps"),
        "rounds": d.get("rounds"),
        "mean_accepted": d.get("mean_accepted"),
        "ms_per_round": d.get("ms_per_round"),
        "gamma_implied": d.get("gamma_implied"),
        "prompt_tps": d.get("prompt_tps"),
        "prefix_hit": d.get("prefix_hit"),
        "finish_reason": rec.get("finish_reason"),
        "hist_cum": hist,
        "hist_delta": hist_delta,
        "total_rounds": total_rounds,
        "per_position": per_position,
    }


def advance_prev(rec, prev_cyc, prev_acc, prev_hist):
    """Update the GLOBAL cumulative baseline from a successful request."""
    st = rec.get("stats") or {}
    cyc = st.get("mtp_cycles_cumulative")
    if cyc is not None:
        prev_cyc = cyc
        prev_acc = st.get("mtp_accepted_drafts_cumulative")
        hist = st.get("mtp_accepted_histogram_cumulative")
        if hist is not None:
            prev_hist = hist
    return prev_cyc, prev_acc, prev_hist


# --------------------------------------------------------------- checkpoint I/O
class Checkpoint:
    """Append-only JSONL; each write is flushed so an abort never loses reps."""

    def __init__(self, path):
        self.path = path
        self.seen = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                with contextlib.suppress(Exception):
                    obj = json.loads(line)
                    if obj.get("key"):
                        self.seen[obj["key"]] = obj

    def has(self, key):
        return key in self.seen

    def add(self, rec):
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if rec.get("key"):
            self.seen[rec["key"]] = rec


def _mine_path(checkpoint_path):
    return checkpoint_path + ".own_requests.json"


def load_mine(checkpoint_path):
    path = _mine_path(checkpoint_path)
    if os.path.exists(path):
        with contextlib.suppress(Exception), open(path) as f:
            return {float(x) for x in json.load(f)}
    return set()


def persist_mine(mine, checkpoint_path):
    with open(_mine_path(checkpoint_path), "w") as f:
        json.dump(sorted(mine), f)


def _track_mine(rec, mine, checkpoint_path):
    if rec and rec.get("request_send_epoch") is not None:
        mine.add(float(rec["request_send_epoch"]))
        persist_mine(mine, checkpoint_path)


# --------------------------------------------------------------------- guard
def guard(tag, mine, guards):
    """Reuse run_campaign's exact guard; collect + return the verdict."""
    ok, rec = CAMP.guard(tag, mine)
    guards.append(rec)
    return ok


# ------------------------------------------------------------------- warmup
def prime(mine, guards, checkpoint_path, do_sleep):
    """(optional) sleep 60 s, one throwaway benign request, prime the baseline."""
    if do_sleep:
        time.sleep(WARMUP_SLEEP_S)
    if not guard("warmup", mine, guards):
        print("ABORT_GUARD warmup", flush=True)
        return None
    salt = os.urandom(4).hex()
    prompt = M.build_benign_prompt(1000, salt, "done")
    rec = None
    try:
        rec = M.stream_once(prompt, 8, None, spec_gamma=None)
    except Exception as e:  # noqa: BLE001 - report and continue
        print(f"WARMUP_ERROR {type(e).__name__}: {e}", flush=True)
    _track_mine(rec, mine, checkpoint_path)
    prev_cyc = prev_acc = prev_hist = None
    if rec is not None:
        prev_cyc, prev_acc, prev_hist = advance_prev(rec, None, None, None)
    print("WARMUP " + json.dumps({"salt": salt, "cycles_cum": prev_cyc,
                                  "accepted_cum": prev_acc, "hist_cum": prev_hist,
                                  "prompt_tokens": (rec or {}).get("usage"),
                                  "decode_s": (rec or {}).get("decode_s")}), flush=True)
    return (prev_cyc, prev_acc, prev_hist)


# --------------------------------------------------------------- fieldcheck
def run_fieldcheck(args, checkpoint_path, guards, mine, prev):
    """One short request WITH spec_gamma=4, one WITHOUT; compare histograms."""
    with4_salt = os.urandom(4).hex()
    absent_salt = os.urandom(4).hex()
    p_with4 = M.build_benign_prompt(2000, with4_salt, BENIGN_TASK)
    p_absent = M.build_benign_prompt(2000, absent_salt, BENIGN_TASK)
    prev_cyc, prev_acc, prev_hist = prev

    if not guard("fieldcheck_with4", mine, guards):
        print("ABORT_GUARD fieldcheck_with4", flush=True)
        return 4
    rec_with4 = None
    try:
        rec_with4 = M.stream_once(p_with4, 200, None, spec_gamma=4)
    except Exception as e:  # noqa: BLE001
        print(f"FIELDCHECK_ERROR with4 {type(e).__name__}: {e}", flush=True)
    _track_mine(rec_with4, mine, checkpoint_path)
    d_with4 = derive_rep(rec_with4 or {}, prev_cyc, prev_acc, prev_hist, 4)
    if rec_with4 is not None:
        prev_cyc, prev_acc, prev_hist = advance_prev(rec_with4, prev_cyc, prev_acc, prev_hist)

    if not guard("fieldcheck_absent", mine, guards):
        print("ABORT_GUARD fieldcheck_absent", flush=True)
        return 4
    rec_absent = None
    try:
        # field ABSENT -> spec_gamma stays None -> engine default 3
        rec_absent = M.stream_once(p_absent, 200, None)
    except Exception as e:  # noqa: BLE001
        print(f"FIELDCHECK_ERROR absent {type(e).__name__}: {e}", flush=True)
    _track_mine(rec_absent, mine, checkpoint_path)
    d_absent = derive_rep(rec_absent or {}, prev_cyc, prev_acc, prev_hist, 3)

    def hist_gamma(hist_delta):
        """Highest populated histogram bin = the gamma actually used.

        A round can accept exactly k drafts only when gamma >= k, so the top
        non-empty bin of a request's histogram delta IS its gamma.  This is the
        robust discriminator; gamma_implied is unreliable at short generations
        (an incomplete trailing round inflates rounds_est) and is a note only.
        """
        if not hist_delta:
            return None
        nz = [i for i, v in enumerate(hist_delta) if v and v > 0]
        return max(nz) if nz else None

    with4 = {"hist_delta": d_with4["hist_delta"],
             "hist_gamma": hist_gamma(d_with4["hist_delta"]),
             "gamma_implied": d_with4["gamma_implied"], "rounds": d_with4["rounds"]}
    absent = {"hist_delta": d_absent["hist_delta"],
              "hist_gamma": hist_gamma(d_absent["hist_delta"]),
              "gamma_implied": d_absent["gamma_implied"], "rounds": d_absent["rounds"]}
    honored = (with4["hist_gamma"] == 4 and absent["hist_gamma"] == 3)
    print("FIELDCHECK " + json.dumps({"with4": with4, "absent": absent,
                                      "field_honored": honored, "guards": guards}), flush=True)
    return 0 if honored else 1


# --------------------------------------------------------------------- full
def run_matrix(args, gammas, workloads, warm_prompts, warm_metas, checkpoint_path,
               guards, mine, prev):
    # deploy/next14-gamma: the cold rep uses the SAME fixed-salt prompt as the
    # warm reps, so it warms exactly the prefix they reuse.  (A fresh cold salt
    # would waste a second cold prefill and make the cold rep non-comparable to
    # the warm ones; is_cold still marks it as the first, uncached rep.)
    schedule = build_schedule(workloads, gammas, args.reps)
    drift_idx = first_benign_index(schedule)
    print("SCHEDULE " + json.dumps(schedule), flush=True)

    cp = Checkpoint(checkpoint_path)
    prev_cyc, prev_acc, prev_hist = prev
    reps_out = []
    consec_err = 0
    for idx, s in enumerate(schedule):
        wl, g, seq = s["workload"], s["gamma"], s["seq"]
        key = f"{wl}:{g}:{seq}"
        drift = (idx == drift_idx)
        if args.resume and cp.has(key):
            print(f"RESUME skip {key}", flush=True)
            continue
        if not guard(key, mine, guards):
            print(f"ABORT_GUARD {json.dumps({'key': key, 'guard': guards[-1]})}", flush=True)
            return 4, reps_out

        salt = SALTS[wl]
        prompt = warm_prompts[wl]
        rec = None
        try:
            rec = M.stream_once(prompt, args.max_tokens, None, spec_gamma=g)
        except Exception as e:  # noqa: BLE001 - record + continue
            rec = None
            err = f"{type(e).__name__}: {e}"
        else:
            err = None
        _track_mine(rec, mine, checkpoint_path)

        if err is not None or rec is None:
            consec_err += 1
            bad = {"key": key, "workload": wl, "gamma": g, "seq": seq,
                   "is_cold": s["is_cold"], "salt": salt, "drift_probe": drift,
                   "guard_ok": guards[-1].get("ok"),
                   "error": err or "no response"}
            cp.add(bad)
            reps_out.append(bad)
            print("REP " + json.dumps(bad), flush=True)
            if consec_err >= MAX_CONSEC_ERRORS:
                print(f"ABORT_ERRORS {consec_err} consecutive", flush=True)
                return 5, reps_out
            continue

        consec_err = 0
        derived = derive_rep(rec, prev_cyc, prev_acc, prev_hist, g)
        row = {"key": key, "workload": wl, "gamma": g, "seq": seq,
               "is_cold": s["is_cold"], "salt": salt, "drift_probe": drift,
               "guard_ok": guards[-1].get("ok"), **derived}
        cp.add(row)
        reps_out.append(row)
        prev_cyc, prev_acc, prev_hist = advance_prev(rec, prev_cyc, prev_acc, prev_hist)
        print("REP " + json.dumps({k: row.get(k) for k in (
            "key", "is_cold", "prompt_tokens", "completion_tokens", "reasoning_tokens",
            "ttft_s", "decode_s", "decode_tps", "rounds", "total_rounds", "mean_accepted",
            "ms_per_round", "gamma_implied", "prompt_tps", "prefix_hit", "finish_reason",
            "hist_delta", "per_position", "drift_probe")}), flush=True)
        time.sleep(2)

    arms = summarize(reps_out, gammas, workloads)
    drift_rep = next((r for r in reps_out if r.get("drift_probe")), None)
    final = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "mode": "full", "params": _params(args, gammas, workloads, warm_metas),
             "schedule": schedule, "by_arm": arms, "drift_probe": drift_rep,
             "guards": guards, "checkpoint": checkpoint_path, "reps": reps_out}
    _write_out(args, final)
    print("SUMMARY " + json.dumps({"by_arm": arms, "drift_probe": drift_rep,
                                   "n_reps": len(reps_out), "guard_failures":
                                   sum(0 if g.get("ok") else 1 for g in guards)}), flush=True)
    return 0, reps_out


def summarize(reps, gammas, workloads):
    arms = {}
    for wl in workloads:
        for g in gammas:
            rs = [r for r in reps
                  if r.get("workload") == wl and r.get("gamma") == g
                  and not r.get("is_cold") and "error" not in r
                  and r.get("ms_per_round") is not None]
            if not rs:
                continue
            arms[f"{wl}:{g}"] = {
                "workload": wl, "gamma": g, "n": len(rs),
                "decode_tps_median": median([r.get("decode_tps") for r in rs]),
                "decode_tps_iqr": iqr([r.get("decode_tps") for r in rs]),
                "ms_per_round_median": median([r.get("ms_per_round") for r in rs]),
                "ms_per_round_iqr": iqr([r.get("ms_per_round") for r in rs]),
                "mean_accepted_median": median([r.get("mean_accepted") for r in rs]),
                "gamma_implied_median": median([r.get("gamma_implied") for r in rs]),
                "total_rounds_median": median([r.get("total_rounds") for r in rs]),
                "total_rounds_sum": sum(r.get("total_rounds") or 0 for r in rs),
                "per_position_median": {f"p{k}": median(
                    [(r.get("per_position") or {}).get(f"p{k}") for r in rs])
                    for k in range(1, g + 1)},
            }
    return arms


def _params(args, gammas, workloads, metas):
    return {"workload": args.workload, "gammas": gammas, "reps": args.reps,
            "max_tokens": args.max_tokens, "depth": args.depth,
            "char_budget": args.char_budget, "api": chat_url(args.api),
            "model": M.MODEL, "session_id": M.SESSION_ID, "salts": SALTS,
            "temperature": 0, "effort": None, "prompts": metas}


def _write_out(args, final):
    out = args.out
    if not out:
        ts = time.strftime("%Y%m%d-%H%M%S")
        out = os.path.join(DEFAULT_OUT_DIR, f"gamma-matrix-{ts}.json")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(final, f, indent=1)
    print(f"WROTE {out}", flush=True)
    return out


# --------------------------------------------------------------------- plan
def run_plan(args, gammas, workloads):
    warm = {wl: build_one(wl, SALTS[wl], args.depth, args.char_budget) for wl in workloads}
    schedule = build_schedule(workloads, gammas, args.reps)
    print("OFFLINE PLAN (no network; no api, no ssh)")
    print("MODEL " + M.MODEL + "   SESSION " + M.SESSION_ID)
    print("PROMPTS " + json.dumps({wl: warm[wl][1] for wl in workloads}))
    print("COLD_SALT: each workload's cold rep uses its fixed salt (warms the "
          "prefix the warm arms reuse); arms reuse salts " + json.dumps(SALTS))
    print("SCHEDULE " + json.dumps(schedule))
    n_cold = sum(1 for s in schedule if s["is_cold"])
    per_wl = {wl: sum(1 for s in schedule if s["workload"] == wl) for wl in workloads}
    est = {}
    for wl in workloads:
        chars = warm[wl][1]["prompt_chars"]
        est[wl] = {"prompt_chars": chars, "pred_tokens_3p48": int(chars / 3.479)}
    print("SCHEDULE_SUMMARY " + json.dumps({
        "n_reps": len(schedule), "n_cold": n_cold, "per_workload": per_wl,
        "arms": [f"{wl}:{g}" for wl in workloads for g in gammas],
        "est_prompt_tokens": est, "max_tokens": args.max_tokens}))
    return 0


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Phase-19 live interleaved gamma matrix")
    ap.add_argument("--workload", default="both", choices=["benign", "agentic", "both"])
    ap.add_argument("--gammas", default=DEFAULT_GAMMAS, help='comma list of arms, e.g. "3,4,5"')
    ap.add_argument("--reps", type=int, default=3, help="timed warm reps per arm")
    ap.add_argument("--max-tokens", type=int, default=1000)
    ap.add_argument("--depth", type=int, default=100000, help="benign depth (tokens)")
    ap.add_argument("--char-budget", type=int, default=None,
                    help="agentic char budget (default None = full real session)")
    ap.add_argument("--api", default=API_BASE)
    ap.add_argument("--out", default=None,
                    help="final JSON path (default " + DEFAULT_OUT_DIR + "/gamma-matrix-<ts>.json)")
    ap.add_argument("--checkpoint", default=None, help="JSONL path (default <out>.jsonl)")
    ap.add_argument("--resume", action="store_true",
                    help="skip any rep whose key already appears in the checkpoint")
    ap.add_argument("--mode", default="full",
                    choices=["full", "fieldcheck", "plan", "warmup"])
    ap.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True,
                    help="sleep 60 s + one throwaway request before the run (default on)")
    a = ap.parse_args()

    gammas = [int(x) for x in a.gammas.split(",") if x.strip()]
    if not gammas or any(g < 1 or g > 6 for g in gammas):
        print("bad --gammas (each must be 1..6)", flush=True)
        return 2
    workloads = ["benign", "agentic"] if a.workload == "both" else [a.workload]
    M.API = chat_url(a.api)

    if a.mode == "plan":
        return run_plan(a, gammas, workloads)

    ts = time.strftime("%Y%m%d-%H%M%S")
    if not a.out:
        a.out = os.path.join(DEFAULT_OUT_DIR, f"gamma-matrix-{ts}.json")
    checkpoint_path = a.checkpoint or (a.out + ".jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)) or ".", exist_ok=True)

    mine = load_mine(checkpoint_path)
    guards = []

    if a.mode == "warmup":
        print("READY " + json.dumps({"mode": "warmup", "sleep_s": WARMUP_SLEEP_S}), flush=True)
        if prime(mine, guards, checkpoint_path, do_sleep=True) is None:
            return 4
        return 0

    if a.mode == "fieldcheck":
        prev = prime(mine, guards, checkpoint_path, do_sleep=a.warmup)
        if prev is None:
            return 4
        return run_fieldcheck(a, checkpoint_path, guards, mine, prev)

    warm_prompts = {}
    warm_metas = {}
    for wl in workloads:
        warm_prompts[wl], warm_metas[wl] = build_one(wl, SALTS[wl], a.depth, a.char_budget)
    prev = (None, None, None)
    if a.warmup:
        primed = prime(mine, guards, checkpoint_path, do_sleep=True)
        if primed is None:
            return 4
        prev = primed
    rc, _ = run_matrix(a, gammas, workloads, warm_prompts, warm_metas,
                       checkpoint_path, guards, mine, prev)
    return rc


if __name__ == "__main__":
    sys.exit(main())
