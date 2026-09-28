#!/usr/bin/env python3
"""p56 -- DSv4.1 DSpark acceptance + speculative decode on both nodes (TP=2).

Phase A (teacher-forced, no rollback): plain greedy decode; at every step the
draft head also drafts from the current anchor (width 3 and 5). Acceptance for
gamma g = leading drafted tokens equal to the tokens greedy actually produced.
Also times plain steps and draft rounds.

Phase B (real loop): chunk verify of [anchor, d1..dg] in one forward, accept,
rollback, feed taps. Sweeps gamma, reports tok/s and agreement with Phase A's
greedy text.
"""
import json, os, sys, time
import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, HOME + "/dsv41-test")
import mlx.core as mx

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
N = int(os.environ.get("P56_TOKENS", "128"))
GAMMAS = [int(g) for g in os.environ.get("P56_GAMMAS", "2,3,4,5").split(",")]

group = mx.distributed.init(backend="jaccl", strict=True)
RANK = group.rank()


def log(*a):
    if RANK == 0:
        print("[p56]", *a, flush=True)


def bcast(x):
    return mx.distributed.all_sum(mx.where(RANK == 0, x, mx.zeros_like(x)), group=group)


from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.deepseek_v41 import spec as SP  # noqa: E402
from mlx_lm.models.exl3.loader import Exl3Checkpoint  # noqa: E402

t0 = time.time()
model, _ = eb.build_model(MODEL, native_dir=NATIVE, rank=RANK, world=2, group=group)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
ck = Exl3Checkpoint(MODEL)
head = eb.build_mtp(ck, model.args, rank=RANK, world=2, group=group)
mx.eval(mx.distributed.all_sum(mx.ones(1), group=group))
log(f"built {time.time()-t0:.0f}s active={mx.get_active_memory()/1e9:.1f}GB peak={mx.get_peak_memory()/1e9:.1f}GB")
TAPS = list(model.args.dspark_target_layer_ids)

from tokenizers import Tokenizer  # noqa: E402
import jinja2  # noqa: E402
tok = Tokenizer.from_file(MODEL + "/tokenizer.json")
tmpl = jinja2.Environment().from_string(open(MODEL + "/chat_template.jinja").read())
PROMPTS = [
    "Explain in a few paragraphs why the sky is blue, then give one surprising fact about light.",
    "Write a Python function that parses an ISO-8601 date string without using datetime, with tests.",
    "List ten practical tips for keeping a sourdough starter healthy, one sentence each.",
]


def enc(q):
    return tok.encode(tmpl.render(messages=[{"role": "user", "content": q}],
                                  add_generation_prompt=True, enable_thinking=False),
                      add_special_tokens=False).ids


def tapcat(taps):
    return mx.concatenate([taps[L] for L in TAPS], axis=-1)


def prefill(ids, extra):
    cache = model.make_cache(1, max_seq_len=len(ids) + extra)
    logits, taps = model(mx.array([ids]), cache, last_logit_only=True, return_taps=True)
    dsc = head.make_cache(1)
    head.append_ctx(tapcat(taps), dsc)
    DUMP["last_prefill_taps"] = tapcat(taps)
    nxt = bcast(mx.argmax(logits[:, -1], axis=-1).astype(mx.int32))
    mx.eval(nxt)
    return cache, dsc, nxt


BC_CHECK = [0]


def draft(anchor, dsc, width):
    d, _ = head.draft(anchor, model.embed, model.head, dsc, width=width)
    raw = d.astype(mx.int32)
    d = bcast(raw)
    mx.eval(d, raw)
    if BC_CHECK[0] < 3:
        BC_CHECK[0] += 1
        print(f"[p56 r{RANK}] draft raw={np.array(raw).tolist()} bcast={np.array(d).tolist()}", flush=True)
    return d


DUMP = {}


def phase_a(ids, dump=False):
    cache, dsc, nxt = prefill(ids, N + 16)
    if dump:
        DUMP["prefill_taps"] = []
    out = [int(nxt.item())]
    drafts = {3: [], 5: []}
    t_step, t_draft = [], []
    for i in range(N):
        s = time.perf_counter()
        for w in (3, 5):
            drafts[w].append([int(v) for v in np.array(draft(nxt, dsc, w))[0]])
        t_draft.append((time.perf_counter() - s) / 2)
        s = time.perf_counter()
        logits, taps = model(nxt[None], cache, last_logit_only=True, return_taps=True)
        head.append_ctx(tapcat(taps), dsc)
        if dump:
            DUMP.setdefault("step_taps", []).append(tapcat(taps))
        nxt = bcast(mx.argmax(logits[:, -1], axis=-1).astype(mx.int32))
        mx.eval(nxt)
        t_step.append(time.perf_counter() - s)
        out.append(int(nxt.item()))
        if out[-1] == 1:
            break
    acc = {}
    for w in (3, 5):
        for g in range(1, w + 1):
            hits = []
            for i, d in enumerate(drafts[w]):
                fut = out[i + 1:i + 1 + g]
                if len(fut) < g:
                    break
                k = 0
                while k < g and d[k] == fut[k]:
                    k += 1
                hits.append(k)
            acc[(w, g)] = float(np.mean(hits)) if hits else float("nan")
    return out, acc, np.median(t_step[3:]), np.median(t_draft[3:])


def phase_b(ids, gamma):
    cache, dsc, nxt = prefill(ids, N + 32)
    pos = cache.offset
    out = [int(nxt.item())]
    hist = []
    t0 = time.perf_counter()
    while len(out) < N + 1:
        d = draft(nxt, dsc, gamma)
        vin = mx.concatenate([nxt.reshape(1, 1), d], axis=1)
        sn = SP.snap(cache, pos)
        logits, taps = model(vin, cache, return_taps=True)
        targ = bcast(mx.argmax(logits[0], axis=-1).astype(mx.int32))
        mx.eval(targ)
        st = SP.stashes(cache)
        tg, dd = np.array(targ), np.array(d)[0]
        n_acc = 0
        while n_acc < gamma and tg[n_acc] == dd[n_acc]:
            n_acc += 1
        hist.append(n_acc)
        new = [int(v) for v in dd[:n_acc]] + [int(tg[n_acc])]
        target = pos + n_acc + 1
        SP.rollback(cache, sn, target, st)
        head.append_ctx(tapcat(taps)[:, :n_acc + 1], dsc)
        pos = target
        nxt = mx.array([new[-1]], dtype=mx.int32)
        stop = False
        for t in new:
            out.append(t)
            if t == 1:
                stop = True
                break
        if stop:
            break
    dt = time.perf_counter() - t0
    return out, hist, (len(out) - 1) / dt


def vcost(ids):
    res = {}
    for R in (1, 2, 3, 4, 5, 6):
        cache, dsc, nxt = prefill(ids, 64)
        pos = cache.offset
        ts = []
        for i in range(12):
            vin = mx.array([[int(nxt.item())] * R], dtype=mx.int32)
            sn = SP.snap(cache, pos)
            s0 = time.perf_counter()
            lg = model(vin, cache)
            mx.eval(lg)
            ts.append(time.perf_counter() - s0)
            SP.rollback(cache, sn, pos + 1, SP.stashes(cache))
            pos += 1
        res[R] = float(np.median(ts[3:]) * 1e3)
    return res


summary = []
for pi, q in enumerate(PROMPTS):
    ids = enc(q)
    greedy, acc, ts, td = phase_a(ids, dump=(pi == 0))
    if pi == 0 and RANK == 0:
        allt = mx.concatenate([DUMP["last_prefill_taps"]] + DUMP.get("step_taps", []), axis=1)
        mx.save(HOME + "/p56_taps.npy", allt.astype(mx.float32))
        json.dump({"ids": ids, "greedy": greedy}, open(HOME + "/p56_greedy.json", "w"))
        log(f"dumped taps {allt.shape}")
    log(f"prompt {pi}: {len(ids)} tok | plain step {ts*1e3:.1f} ms ({1/ts:.2f} tok/s) | draft round {td*1e3:.1f} ms")
    log("  accepted/round (teacher-forced): " + "  ".join(
        f"w{w}g{g}={acc[(w, g)]:.2f}" for (w, g) in sorted(acc)))
    if pi == 0:
        log("  greedy text: " + tok.decode(greedy[:90]).replace("\n", " "))
    if acc[(5, 1)] < 0.3:
        log("  acceptance too low -- skipping the spec loop for this prompt")
        continue
    if pi == 0:
        vc = vcost(ids)
        log("  verify forward ms by rows: " + "  ".join(f"R{r}={v:.1f}" for r, v in vc.items()))
    for g in GAMMAS:
        sp, hist, tps = phase_b(ids, g)
        n = min(len(sp), len(greedy))
        agree = next((i for i in range(n) if sp[i] != greedy[i]), n)
        log(f"  spec gamma={g}: {tps:.2f} tok/s  mean_acc={np.mean(hist):.2f} "
            f"(tokens/round {np.mean(hist)+1:.2f}) rounds={len(hist)} "
            f"identical_prefix={agree}/{n}")
        summary.append((pi, g, tps, float(np.mean(hist))))
        if pi == 0 and g == 3:
            log("  spec text:   " + tok.decode(sp[:90]).replace("\n", " "))
log(f"peak={mx.get_peak_memory()/1e9:.1f}GB")
for g in GAMMAS:
    r = [s for s in summary if s[1] == g]
    log(f"SUMMARY gamma={g}: mean {np.mean([s[2] for s in r]):.2f} tok/s, mean_acc {np.mean([s[3] for s in r]):.2f}")
log("P56_DONE")
