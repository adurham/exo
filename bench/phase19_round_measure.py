#!/usr/bin/env python3
"""Phase-19 live round-wall / acceptance measurement for the DSv4.1 dsv41 engine.

NO RELAUNCH.  Public API only.

Mechanism (source-verified):
  * dsv41 pins gamma at 3 (rounds.py `_one_round` never calls
    GammaPolicy.update(), so `policy.next()` returns its start value 3).
  * The engine accumulates `_spec_drafted` (sum of gamma) and `_spec_accepted`
    (sum of accepted drafts) into the response's `: generation_stats` SSE frame
    as `mtp_cycles_cumulative` / `mtp_accepted_drafts_cumulative`.
  * => per request: rounds = d_cycles/3 ; mean_accepted = 3*d_accepted/d_cycles.
  * Cross-check gamma from an independent round estimate:
      rounds_est = completion_tokens / mean_committed, mean_committed = 1+mean_acc
      gamma_implied = d_cycles / rounds_est   (should be ~3)

NOTE: the stats frame is a comment line `: generation_stats {json}` -- it does
NOT start with `data:`, parse it BEFORE the data-line check.

Usage:
  python3 phase19_round_measure.py --depth 100000 --reps 3 --task count
"""
from __future__ import annotations

import argparse, json, os, random, statistics, sys, time, urllib.request

API = os.environ.get("PHASE19_API", "http://macstudio-m4-1.tail19c543.ts.net:52415/v1/chat/completions")
MODEL = "dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"

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
    # long, deterministic, ~1000 tokens, low EOS risk
    "count": "\n\nTask: List the integers from 1 to 450 in order, each on its own line, and nothing else.",
    "done": "\n\nTask: Reply with ONLY the single word DONE and nothing else.",
}


def build_prompt(target_tokens: int, salt: str, task: str) -> str:
    rng = random.Random(20261007)
    target_chars = int(target_tokens * 5.59)   # measured chars/token this session
    head = f"[SESSION-SALT {salt}]\n"
    parts = [head]; n = len(head)
    while n < target_chars:
        s = _SENTENCES[rng.randrange(len(_SENTENCES))]
        parts.append(s + " "); n += len(s) + 1
    parts.append(TASKS[task])
    return "".join(parts)


def stream_once(prompt: str, max_tokens: int, effort: str | None) -> dict:
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True}
    if effort:
        body["reasoning_effort"] = effort
    req = urllib.request.Request(API, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = first = last = None
    chars = 0; usage = None; stats = None
    with urllib.request.urlopen(req, timeout=3600) as resp:
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
            if usage is None and obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                txt = (ch.get("delta") or {}).get("content")
                if txt:
                    if ttft is None:
                        ttft, first = now - t0, now
                    chars += len(txt); last = now
    wall = time.perf_counter() - t0
    return {"wall_s": round(wall, 3),
            "ttft_s": round(ttft, 4) if ttft else None,
            "decode_s": round(last - first, 4) if (first and last) else None,
            "content_chars": chars, "usage": usage, "stats": stats}


def derive(rec, prev_cyc, prev_acc, gamma):
    st = rec.get("stats") or {}
    u = rec.get("usage") or {}
    cyc = st.get("mtp_cycles_cumulative"); acc = st.get("mtp_accepted_drafts_cumulative")
    gen = u.get("completion_tokens")
    out = {"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": gen,
           "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
           "cycles_cum": cyc, "accepted_cum": acc,
           "prefix_hit": st.get("prefix_cache_hit"),
           "prompt_tps": round(st.get("prompt_tps") or 0, 1),
           "decode_tps": None, "rounds": None, "mean_accepted": None,
           "ms_per_round": None, "gamma_implied": None}
    if rec.get("decode_s") and gen and gen > 1 and rec["decode_s"] > 0:
        out["decode_tps"] = round((gen - 1) / rec["decode_s"], 3)
    if cyc is not None and prev_cyc is not None and cyc > prev_cyc:
        d_cyc = cyc - prev_cyc
        d_acc = (acc - (prev_acc or 0)) if acc is not None else None
        rounds = d_cyc / gamma
        out["rounds"] = round(rounds, 1)
        if d_acc is not None:
            out["mean_accepted"] = round(gamma * d_acc / d_cyc, 4)
            mc = 1.0 + d_acc / d_cyc               # mean committed tokens / round
            if gen and mc > 0:
                out["rounds_est"] = round(gen / mc, 1)
                if out["rounds_est"]:
                    out["gamma_implied"] = round(d_cyc / out["rounds_est"], 3)
        if rec.get("decode_s"):
            out["ms_per_round"] = round(rec["decode_s"] * 1000.0 / rounds, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=1400)
    ap.add_argument("--gamma", type=int, default=3)
    ap.add_argument("--label", default="run")
    ap.add_argument("--effort", default=None)
    ap.add_argument("--task", default="count", choices=list(TASKS))
    ap.add_argument("--fresh", action="store_true", help="new salt each rep (forces cold prefill)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    salt = os.urandom(4).hex()
    prompt = build_prompt(a.depth, salt, a.task)
    print(f"[{a.label}] filler chars={len(prompt)} salt={salt} task={a.task}", flush=True)
    recs = []; prev_cyc = prev_acc = None
    for i in range(a.reps):
        p = prompt
        if a.fresh and i > 0:
            p = build_prompt(a.depth, os.urandom(4).hex(), a.task)
        r = stream_once(p, a.max_tokens, a.effort)
        r.update(derive(r, prev_cyc, prev_acc, a.gamma)); r["rep"] = i
        recs.append(r)
        if r.get("cycles_cum") is not None:
            prev_cyc, prev_acc = r["cycles_cum"], r["accepted_cum"]
        print(json.dumps({k: r.get(k) for k in ("rep","prompt_tokens","completion_tokens",
              "reasoning_tokens","ttft_s","decode_s","decode_tps","rounds","mean_accepted",
              "ms_per_round","gamma_implied","prefix_hit")}), flush=True)
        time.sleep(2)
    dec = [r["decode_tps"] for r in recs if r.get("decode_tps")]
    mar = [r["mean_accepted"] for r in recs if r.get("mean_accepted") is not None]
    msr = [r["ms_per_round"] for r in recs if r.get("ms_per_round")]
    gimp = [r["gamma_implied"] for r in recs if r.get("gamma_implied")]
    summary = {"label": a.label, "depth_target": a.depth, "reps": len(recs),
               "prompt_tokens": recs[0].get("prompt_tokens"),
               "decode_tps_median": round(statistics.median(dec), 3) if dec else None,
               "decode_tps_all": dec,
               "mean_accepted_median": round(statistics.median(mar), 4) if mar else None,
               "ms_per_round_median": round(statistics.median(msr), 2) if msr else None,
               "gamma_implied_median": round(statistics.median(gimp), 3) if gimp else None}
    print("SUMMARY", json.dumps(summary), flush=True)
    if a.out:
        recs.append({"__summary__": summary})
        json.dump(recs, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
