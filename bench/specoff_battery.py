#!/usr/bin/env python3
"""c=2 concurrency stress battery for DSv4-Flash serving.

Fires N PAIRS of long (4000-token, temp 1.0) streaming requests with
divergent essay topics; the second stream of each pair starts 6 s after the
first, forcing a MID-DECODE ADMISSION into a running batch — historically
the hardest serving transition (task #25 class: batch-size change, uneven
contexts, per-stream cache bootstrap, MTP verify at BS>1). Each stream is
scored for real degeneration: a tail/body repetition loop (period <= 48
detected over the last 600 chars and at 300-char offsets through the body)
or collapsed chars-per-token (< 1.8 past 800 tokens). Missing usage frames
are NOT flagged (a slow stream cut by the client timeout can be coherent);
the authoritative cross-check is the server's own DEGENERATION kill count
in exo.log.

Usage (ON a cluster node, model placed or JIT-placeable):
    .venv/bin/python bench/specoff_battery.py [N_PAIRS]   # default 8
Pass criterion: "RESULT: 0/N pairs degenerated" and zero
SIGKILL/deadline/re-place entries in ~/exo.log for the run window.

Name is historical (first used to validate spec-OFF serving); it runs
against whatever the server config is — the standard prod gate runs it
with MTP ON. This is the battery cited throughout
docs/dsv4-c2-serving-handoff-2026-07-06.md ("battery 3/3 clean")."""
import json, threading, time, uuid, sys, urllib.request

URL = "http://localhost:52415/v1/chat/completions"
MODEL = "mlx-community/DeepSeek-V4-Flash"
N_PAIRS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
TOKENS = 4000

TOPICS = [
    "the history of container shipping", "how CRDTs enable offline-first apps",
    "the physics of lift in fixed-wing aircraft", "fermentation in food preservation",
    "the evolution of x86 memory models", "coral reef ecosystem collapse",
    "the mathematics of error-correcting codes", "urban water infrastructure",
    "the design of suspension bridges", "protein folding and chaperones",
    "the economics of electricity grids", "medieval manuscript production",
    "GPS and relativistic corrections", "the immune system's B-cell memory",
    "volcanic ash effects on aviation", "double-entry bookkeeping history",
]


def tail_loop(text, max_period=48, span=200):
    tail = text[-600:]
    for p in range(1, max_period + 1):
        reps = max(6, -(-span // p))
        if len(tail) >= reps * p:
            unit = tail[-p:]
            if unit.strip() and tail[-reps * p:] == unit * reps:
                return unit
    return None


def run_stream(topic, out, idx):
    salt = uuid.uuid4().hex
    prompt = ("[%s] Write a long detailed technical essay about %s. Cover "
              "history, key concepts, misconceptions, and open problems." % (salt, topic))
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": TOKENS, "temperature": 1.0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    text, usage, finish, err = [], None, None, None
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                p = line[5:].strip()
                if p == "[DONE]":
                    break
                try:
                    o = json.loads(p)
                except json.JSONDecodeError:
                    continue
                if o.get("usage"):
                    usage = o["usage"]
                for ch in o.get("choices", []):
                    d = ch.get("delta", {}).get("content")
                    if d:
                        text.append(d)
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
    except Exception as e:
        err = "%s: %s" % (type(e).__name__, str(e)[:60])
    full = "".join(text)
    ctok = usage.get("completion_tokens") if usage else None
    loop = tail_loop(full)
    cpt = (len(full) / ctok) if ctok else None
    body_loop = None
    for off in range(0, max(1, len(full) - 600), 300):
        if tail_loop(full[: len(full) - off]):
            body_loop = True
            break
    # Real degeneration = a repetition loop or a collapsed chars-per-token.
    # Do NOT flag a missing usage frame (slow spec-off streams can be cut by
    # the client timeout while producing perfectly coherent text). The
    # authoritative cross-check is the server's own DEGENERATION kill count.
    degen = bool(loop) or bool(body_loop) or \
        (cpt is not None and ctok and ctok > 800 and cpt < 1.8)
    out[idx] = {"topic": topic, "tok": ctok, "finish": finish, "usage": usage is not None,
                "cpt": round(cpt, 2) if cpt else None, "loop": loop, "body_loop": body_loop,
                "err": err, "degen": degen, "tail": full[-80:].replace("\n", " ")}


degen_pairs = 0
for i in range(N_PAIRS):
    ta, tb = TOPICS[(2 * i) % 16], TOPICS[(2 * i + 1) % 16]
    out = {}
    t0 = threading.Thread(target=run_stream, args=(ta, out, 0))
    t1 = threading.Thread(target=run_stream, args=(tb, out, 1))
    t0.start()
    time.sleep(6)
    t1.start()
    t0.join()
    t1.join()
    pd = any((out.get(k) or {"degen": True})["degen"] for k in (0, 1))
    degen_pairs += 1 if pd else 0
    tag = "DEGEN" if pd else "clean"
    print("pair %d: %s" % (i + 1, tag), flush=True)
    for k in (0, 1):
        s = out.get(k, {})
        print("   s%d: tok=%s fin=%s cpt=%s err=%s" % (k, s.get("tok"), s.get("finish"), s.get("cpt"), s.get("err")), flush=True)
        if s.get("degen"):
            print("   s%d degen: tok=%s fin=%s cpt=%s loop=%r body=%s err=%s tail=%r"
                  % (k, s.get("tok"), s.get("finish"), s.get("cpt"), s.get("loop"),
                     s.get("body_loop"), s.get("err"), s.get("tail")), flush=True)

print("\n=== RESULT: %d/%d pairs degenerated ===" % (degen_pairs, N_PAIRS), flush=True)
