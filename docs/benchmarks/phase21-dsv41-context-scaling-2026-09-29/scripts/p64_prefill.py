#!/usr/bin/env python3
"""p64 -- DSv4.1 long-prompt prefill speed on both nodes (TP=2).

Real text tokens (repeated prompt corpus), chunked prefill, lengths 2K..32K.
Reports prefill tok/s per length and the first decoded token after prefill.
"""
import json, os, sys, time
HOME = os.path.expanduser("~")
sys.path.insert(0, HOME + "/dsv41-test")
import mlx.core as mx

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LENS = [int(x) for x in os.environ.get("P64_LENS", "2048,8192,16384,32768").split(",")]
USE_DRIVER = os.environ.get("P64_DRIVER", "1") == "1"
CHUNK = int(os.environ.get("P64_CHUNK", "512"))

group = mx.distributed.init(backend="jaccl", strict=True)
RANK = group.rank()


def log(*a):
    if RANK == 0:
        print("[p64]", *a, flush=True)


from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402

model, _ = eb.build_model(MODEL, native_dir=NATIVE, rank=RANK, world=2, group=group)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
mx.eval(mx.distributed.all_sum(mx.ones(1), group=group))
log(f"built active={mx.get_active_memory()/1e9:.1f}GB driver={USE_DRIVER}")
if USE_DRIVER:
    from mlx_lm.models.deepseek_v41 import prefill as PF
    t0 = time.perf_counter(); PF.warmup(model); log(f"warmup {time.perf_counter()-t0:.1f}s")
base = json.load(open(HOME + "/p30_prompt_ids.json"))
for L in LENS:
    ids = (base * (L // len(base) + 1))[:L]
    cache = model.make_cache(1, max_seq_len=L + 64)
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    if USE_DRIVER:
        am = PF.prefill(model, ids, cache, last_logit_only=True, argmax=True, long_threshold=int(os.environ.get('P64_LONG', '8192')))
        mx.eval(am)
    else:
        for a in range(0, L, CHUNK):
            am = model(mx.array([ids[a:a + CHUNK]]), cache, last_logit_only=True, argmax=True)
            mx.eval(am)
    dt = time.perf_counter() - t0
    nxt = am.reshape(-1)[-1:].astype(mx.int32)
    toks = [int(nxt.item())]
    steps = []
    for _ in range(32):
        t1 = time.perf_counter()
        nxt = model(nxt[None], cache, last_logit_only=True, argmax=True).reshape(-1)[-1:].astype(mx.int32)
        mx.eval(nxt); toks.append(int(nxt.item()))
        steps.append((time.perf_counter() - t1) * 1e3)
    dd = sorted(steps)[16] / 1e3
    log("  decode ms per step: " + " ".join(f"{v:.0f}" for v in steps))
    log(f"prefill {L:6d} tok: {dt:7.1f} s = {L/dt:6.1f} tok/s  peak={mx.get_peak_memory()/1e9:.1f}GB  then decode median {1/dd:.1f} tok/s  first toks {toks[:8]}")
    del cache
    mx.clear_cache()
log("P64_DONE")
