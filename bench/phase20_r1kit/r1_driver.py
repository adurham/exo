#!/usr/bin/env python3
"""R1 FIXED-REPLAY arm runner (Phase-5 R1 kit) -- PREPARED, DO NOT RUN until GO.

Thin extension of the shipped ship-day driver
`/Users/adam.durham/.hermes/cache/scratch/p3b/p3b_driver.py`. It adds exactly the
two things the R1 baseline-vs-lever A/B needs and the base driver lacks:

  1. --salt  : a FIXED base salt. The base driver hardcodes ``os.urandom(4).hex()``
               per rep, so its prompt CONTENT differs every rep and every arm —
               fatal for a timing A/B (skill: FIXED replay so content divergence
               cannot pollute timing). Here the per-rep salt is ``f"{BASE}-{rep}"``,
               fully deterministic => the SAME bytes are replayed on BOTH arms.

  2. own-request registry PERSISTENCE + RELOAD. The base driver passes no
     ``registry_path`` and never reloads, so a chunk killed mid-run leaves its own
     POSTs unregistered and the NEXT chunk's idle gate aborts on them. Here the
     registry is written by ChunkGuard and RELOADED into every chunk's own list.

Everything else (harness modules, stream_once shape, summary keys, chunking) is
inherited verbatim from p3b_driver so the numbers are comparable to the frozen
same-harness baseline.

Usage (one arm per invocation; run the SAME --salt on both arms):
  r1_driver.py --arm benign  --total-reps 4 --reps-per-chunk 4 \
               --depth 20000 --max-tokens 800 --salt r1fix-a \
               --label r1_benign --out <dir>/r1_benign.json
  r1_driver.py --arm agentic --total-reps 6 --reps-per-chunk 2 \
               --salt r1fix-a --label r1_agentic --out <dir>/r1_agentic.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

SCRATCH = "/Users/adam.durham/.hermes/cache/scratch"
P3B = os.path.join(SCRATCH, "p3b")
GUARD_DIR = "/private/tmp/phase20-campaign/bench"
HARNESS_DIR = "/private/tmp/next16-instr/bench"
sys.path.insert(0, GUARD_DIR)
sys.path.insert(0, HARNESS_DIR)
sys.path.insert(0, P3B)

import phase20_guard as G            # noqa: E402  (re-exported)
import p3b_driver as P3B             # noqa: E402  reuse stream_once + RM/AM
RM = P3B.RM
AM = P3B.AM
stream_once = P3B.stream_once

DEFAULT_REGISTRY = os.path.join(SCRATCH, "p5", "r1kit", "own_requests.jsonl")
LOG_DIR = os.environ.get("P3B_LOG_DIR", os.path.join(SCRATCH, "p5", "r1kit", "guard"))


def load_own(registry_path: str) -> list[float]:
    """Reload the persisted own-request epochs (skill: re-read after any restart)."""
    out: list[float] = []
    if not registry_path or not os.path.exists(registry_path):
        return out
    with open(registry_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(float(json.loads(line)["t"]))
            except Exception:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["benign", "agentic"])
    ap.add_argument("--total-reps", type=int, default=6)
    ap.add_argument("--reps-per-chunk", type=int, default=0)
    ap.add_argument("--depth", type=int, default=20000)
    ap.add_argument("--task", default="count", choices=list(RM.TASKS))
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--gamma", type=int, default=3)
    ap.add_argument("--salt", default="r1fix-a", help="FIXED base salt (same on both arms)")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-wall", type=float, default=900)
    ap.add_argument("--registry", default=DEFAULT_REGISTRY)
    a = ap.parse_args()

    per = a.reps_per_chunk if a.reps_per_chunk > 0 else a.total_reps
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        open(a.out + ".jsonl", "w").close()

    own = load_own(a.registry)  # reload persisted own epochs ONCE at start
    recs: list[dict] = []
    prev_cyc = prev_acc = None
    aborted_reason = None
    n = 0
    while n < a.total_reps:
        k = min(per, a.total_reps - n)
        if k <= 0:
            break
        label = f"{a.label}_c{n}"
        print(f"[{time.strftime('%T')}] wait-idle before {label} ({k} reps)...", flush=True)
        G.wait_for_idle(max_wait_s=1500, own_requests=own)
        with G.ChunkGuard(label, max_wall_s=a.max_wall, log_dir=LOG_DIR,
                          registry_path=a.registry, own_requests=own) as guard:
            for j in range(k):
                if guard.cancel_event.is_set():
                    break
                salt = f"{a.salt}-{n}"           # FIXED per rep index (identical across arms)
                if a.arm == "agentic":
                    prompt, meta = AM.build_agentic_prompt(salt, None)
                else:
                    prompt = RM.build_prompt(a.depth, salt, a.task)
                    meta = {"prompt_chars": len(prompt)}
                guard.register_own_request(time.time())
                r = stream_once(prompt, a.max_tokens)
                r.update(RM.derive(r, prev_cyc, prev_acc, a.gamma))
                r.update({"rep": n, "arm": a.arm, "chunk": label, "salt": salt, "meta": meta})
                if r.get("cycles_cum") is not None:
                    prev_cyc, prev_acc = r["cycles_cum"], r["accepted_cum"]
                recs.append(r)
                n += 1
                if a.out:
                    with open(a.out + ".jsonl", "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(r) + "\n")
                print(json.dumps({kk: r.get(kk) for kk in (
                    "rep", "arm", "salt", "prompt_tokens", "completion_tokens",
                    "reasoning_tokens", "ttft_s", "decode_s", "decode_tps", "rounds",
                    "mean_accepted", "ms_per_round", "gamma_implied", "finish_reason")}),
                    flush=True)
                time.sleep(2)
        if guard.aborted:
            print(f"[{time.strftime('%T')}] chunk {label} ABORTED: {guard.reason}", flush=True)
            if guard.reason == "ABORTED_USER_ARRIVED":
                aborted_reason = guard.reason
                break
        own = load_own(a.registry)  # pick up this chunk's own registrations
        time.sleep(15)

    dec = [r["decode_tps"] for r in recs if r.get("decode_tps")]
    msr = [r["ms_per_round"] for r in recs if r.get("ms_per_round")]
    mar = [r["mean_accepted"] for r in recs if r.get("mean_accepted") is not None]
    summary = {"label": a.label, "arm": a.arm, "salt_base": a.salt, "round_prof": None,
               "reps": len(recs), "depth": a.depth, "max_tokens": a.max_tokens,
               "decode_tps_median": round(statistics.median(dec), 3) if dec else None,
               "decode_tps_all": dec,
               "ms_per_round_median": round(statistics.median(msr), 2) if msr else None,
               "ms_per_round_all": msr,
               "mean_accepted_median": round(statistics.median(mar), 4) if mar else None,
               "aborted": aborted_reason is not None, "abort_reason": aborted_reason}
    print("SUMMARY " + json.dumps(summary), flush=True)
    if a.out:
        json.dump({"summary": summary, "recs": recs}, open(a.out, "w"), indent=1)
    return 0 if aborted_reason is None else 1


if __name__ == "__main__":
    sys.exit(main())
