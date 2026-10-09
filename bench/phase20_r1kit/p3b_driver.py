#!/usr/bin/env python3
"""Phase-3B ship-validation arm runner (chunk-aware).

Runs one arm (benign or agentic) as a sequence of guard-wrapped chunks, each
<=15 min (PREREG R3), carrying the MTP cycle delta (prev_cyc/prev_acc) ACROSS
chunks so every rep after the first yields a valid ms/round / mean_accepted.

round_prof is NOT passed: neither production f4bb14746 nor deploy/next17-levers
carries the per-request round_prof field (only next16-instr does), so both arms
are measured symmetrically by client end-to-end ms/round + decode t/s.

Usage:
  p3b_driver.py --arm benign --total-reps 6 --reps-per-chunk 6 --depth 20000 \
                --max-tokens 800 --label p3b_prod_benign --out /tmp/p3b/prod_benign.json
  p3b_driver.py --arm agentic --total-reps 6 --reps-per-chunk 2 \
                --label p3b_prod_agentic --out /tmp/p3b/prod_agentic.json
"""
from __future__ import annotations
import argparse, contextlib, json, os, statistics, sys, time, urllib.request

HARNESS_DIR = "/private/tmp/next16-instr/bench"
GUARD_DIR = "/private/tmp/phase20-campaign/bench"
sys.path.insert(0, GUARD_DIR)
sys.path.insert(0, HARNESS_DIR)
import phase20_guard as G                      # noqa: E402
import phase19_round_measure as RM            # noqa: E402
import phase19_agentic_measure as AM          # noqa: E402

MODEL = RM.MODEL
API = os.environ.get("PHASE19_API",
                     "http://192.168.86.48:52415/v1/chat/completions")
LOG_DIR = os.environ.get("P3B_LOG_DIR",
                         "/Users/adam.durham/.hermes/cache/scratch/p3b/guard")


def stream_once(prompt, max_tokens):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True}
    req = urllib.request.Request(API, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = first = last = None
    chars = 0
    usage = stats = finish = None
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if line.startswith(": generation_stats"):
                with contextlib.suppress(Exception):
                    stats = json.loads(line.split(" ", 2)[2])
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
            if usage is None and obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                d = ch.get("delta") or {}
                t = d.get("content") or d.get("reasoning_content") or ""
                if t:
                    if ttft is None:
                        ttft, first = now - t0, now
                    chars += len(t)
                    last = now
    wall = time.perf_counter() - t0
    return {"wall_s": round(wall, 3), "ttft_s": round(ttft, 4) if ttft else None,
            "decode_s": round(last - first, 4) if (first and last and last > first) else None,
            "content_chars": chars, "usage": usage, "stats": stats, "finish_reason": finish}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["benign", "agentic"])
    ap.add_argument("--total-reps", type=int, default=6)
    ap.add_argument("--reps-per-chunk", type=int, default=0)  # 0 => one chunk
    ap.add_argument("--depth", type=int, default=20000)
    ap.add_argument("--task", default="count", choices=list(RM.TASKS))
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--gamma", type=int, default=3)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-wall", type=float, default=900)
    a = ap.parse_args()

    per = a.reps_per_chunk if a.reps_per_chunk > 0 else a.total_reps
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        open(a.out + ".jsonl", "w").close()

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
        G.wait_for_idle(max_wait_s=1500)
        with G.ChunkGuard(label, max_wall_s=a.max_wall, log_dir=LOG_DIR) as guard:
            for j in range(k):
                if guard.cancel_event.is_set():
                    break
                salt = os.urandom(4).hex()
                if a.arm == "agentic":
                    prompt, meta = AM.build_agentic_prompt(salt, None)
                else:
                    prompt = RM.build_prompt(a.depth, salt, a.task)
                    meta = {"prompt_chars": len(prompt)}
                guard.register_own_request(time.time())
                r = stream_once(prompt, a.max_tokens)
                r.update(RM.derive(r, prev_cyc, prev_acc, a.gamma))
                r["rep"] = n
                r["arm"] = a.arm
                r["chunk"] = label
                r["meta"] = meta
                if r.get("cycles_cum") is not None:
                    prev_cyc, prev_acc = r["cycles_cum"], r["accepted_cum"]
                recs.append(r)
                n += 1
                if a.out:
                    with open(a.out + ".jsonl", "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(r) + "\n")
                print(json.dumps({kk: r.get(kk) for kk in (
                    "rep", "arm", "chunk", "prompt_tokens", "completion_tokens",
                    "reasoning_tokens", "ttft_s", "decode_s", "decode_tps", "rounds",
                    "mean_accepted", "ms_per_round", "gamma_implied", "finish_reason")}),
                    flush=True)
                time.sleep(2)
        if guard.aborted:
            print(f"[{time.strftime('%T')}] chunk {label} ABORTED: {guard.reason}", flush=True)
            if guard.reason == "ABORTED_USER_ARRIVED":
                aborted_reason = guard.reason
                break
        time.sleep(15)

    dec = [r["decode_tps"] for r in recs if r.get("decode_tps")]
    msr = [r["ms_per_round"] for r in recs if r.get("ms_per_round")]
    mar = [r["mean_accepted"] for r in recs if r.get("mean_accepted") is not None]
    summary = {"label": a.label, "arm": a.arm, "round_prof": None,
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
