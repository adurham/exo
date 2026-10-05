#!/usr/bin/env python3
"""Ladder exactness gate — run AFTER the ladder fix deploys to the cluster.

Purpose (Fable review item 6): a ladder refeed (rewind to a margin rung, refeed
the tail) must produce the SAME generation as a straight full feed. This script
runs the within-build test at cluster scale:

  T1: [user X]                          -> A1        (build; may be full feed)
  T2: [user X, assistant A1, user Q]    -> A2        (LADDER refeed path exercised)
  T3: identical to T2                   -> A3        (second pass, different cache history)

Gates:
  * A2 == A3  (bitwise text equality at temp 0; same inputs, two cache histories)
  * optional: A2 == A2_prev where A2_prev was captured on the PRE-ladder build
    (cross-build exactness; pass --a2-prev path to a JSON with {"content": ...}).
  * T2/T3 prefill cost must be small (log shows a margin-rung rewind, not a full
    feed) — report the delta from the response usage + wall time.

Usage:
  python3 ladder_exactness.py --api http://macstudio-m4-1.tail19c543.ts.net:52415 \
      --label laddercheck --prompt-file <any long prompt text> [--a2-prev prev.json]
"""
import argparse, json, os, sys, time, urllib.request

def chat(api, model, messages, max_tokens, timeout=7200):
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens,
                       "temperature": 0.0, "reasoning_effort": "low", "stream": False}).encode()
    req = urllib.request.Request(api + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        ch = (d.get("choices") or [{}])[0]
        m = ch.get("message") or {}
        return {"ok": True, "elapsed": time.time() - t0, "usage": d.get("usage"),
                "finish": ch.get("finish_reason"),
                "content": m.get("content") or "",
                "reasoning": m.get("reasoning_content") or "",
                "error": None, "raw": d}
    except Exception as e:
        return {"ok": False, "elapsed": time.time() - t0, "usage": None, "finish": None,
                "content": "", "reasoning": "", "error": str(e)[:300], "raw": None}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://macstudio-m4-1.tail19c543.ts.net:52415")
    ap.add_argument("--model", default="dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw")
    ap.add_argument("--label", default="laddercheck")
    ap.add_argument("--depth", type=int, default=350000)
    ap.add_argument("--prompt-file", default=None, help="reuse an existing long prompt")
    ap.add_argument("--a2-prev", default=None, help="prior-build A2 JSON for cross-check")
    ap.add_argument("--replay-text", default="done",
                    help="middle-turn assistant text; must match what A2_prev used "
                         "(default 'done') for the cross-build gate to be valid")
    ap.add_argument("--outdir", default=os.path.expanduser("~/.hermes/cache/scratch/laddercheck"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rec = os.path.join(args.outdir, "record.txt")
    def log(s):
        print(s, flush=True)
        with open(rec, "a") as f:
            f.write(s + "\n")

    if args.prompt_file:
        X = open(args.prompt_file).read()
    else:
        s = "The quick brown fox jumps over the lazy dog while the cluster serves tokens at steady pace. "
        body = s * int(args.depth * 5.111 / len(s))
        mid = len(body) // 2
        X = body[:mid] + " The vault code is 8492. " + body[mid:] + "\n\nNow reply with exactly: done"

    Q = "Now answer this question: What is the vault code? Reply with just the 4-digit number."
    log(f"=== ladder exactness {time.strftime('%F %T')} depth~{args.depth} ===")

    log("T1: build [user X]...")
    r1 = chat(args.api, args.model, [{"role": "user", "content": X}], 128)
    log(f"T1: {r1['elapsed']:.1f}s ok={r1['ok']} finish={r1['finish']} content={r1['content'][:60]!r} usage={r1['usage']}")
    if not r1["ok"]:
        log("T1 FAILED — abort"); return 1
    A1 = args.replay_text
    json.dump({"content": r1["content"], "usage": r1["usage"], "elapsed": r1["elapsed"]},
              open(os.path.join(args.outdir, "T1.json"), "w"), indent=2)
    if (r1["content"] or "").strip() not in (A1, ""):
        log(f"NOTE: T1 content {r1['content'][:40]!r} != replay text {A1!r} — "
            "using replay text for the middle turn (cross-build parity)")

    msgs = [{"role": "user", "content": X},
            {"role": "assistant", "content": A1},
            {"role": "user", "content": Q}]
    log("T2: ladder refeed path [user X, assistant A1, user Q]...")
    r2 = chat(args.api, args.model, msgs, 512)
    log(f"T2: {r2['elapsed']:.1f}s ok={r2['ok']} finish={r2['finish']} content={r2['content'][:120]!r} usage={r2['usage']}")
    json.dump({"content": r2["content"], "usage": r2["usage"], "elapsed": r2["elapsed"]},
              open(os.path.join(args.outdir, "T2.json"), "w"), indent=2)

    log("T3: identical repeat...")
    r3 = chat(args.api, args.model, msgs, 512)
    log(f"T3: {r3['elapsed']:.1f}s ok={r3['ok']} finish={r3['finish']} content={r3['content'][:120]!r} usage={r3['usage']}")
    json.dump({"content": r3["content"], "usage": r3["usage"], "elapsed": r3["elapsed"]},
              open(os.path.join(args.outdir, "T3.json"), "w"), indent=2)

    log("")
    log(f"GATE-1 same-build exactness: A2==A3 -> {r2['content'] == r3['content']}")
    if args.a2_prev and os.path.exists(args.a2_prev):
        prev = json.load(open(args.a2_prev))
        pc = prev.get("content") or ""
        log(f"GATE-2 cross-build exactness: A2==A2_prev -> {r2['content'] == pc}")
    log(f"GATE-3 refeed cost: T2 wall {r2['elapsed']:.1f}s, T3 wall {r3['elapsed']:.1f}s "
        f"(both must be far below a ~2500s full feed; check the engine log for the rewind line)")
    log(f"contents: A2={r2['content'][:120]!r} A3={r3['content'][:120]!r}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
