# P05 Phase A: studio-hardware lm_head mxfp8 microbench (standalone process,
# runs BESIDE the live production runner — p01/p03 recipe, ZERO relaunches).
# Question: does mxfp8 lm_head actually beat BF16 at the PRODUCTION call
# shapes (M=1 decode/sampler, M=3 draft, M=4 verify) on the real M4 Max
# silicon? The laptop showed M=1 1.93x faster but M=4 0.83x (qmv_wide
# byte-inefficient at N=129280). This decides whether the live A/B is
# worth relaunches BEFORE spending them.
# Also: markov_w2 (M=1, 3x/draft cycle) and the sampler unit
# (logsumexp+argmax) end-to-end.
import os

for v in ("MLX_GEMV_BATCH_INVARIANT", "MLX_STEEL_BATCH_INVARIANT"):
    assert os.environ.get(v) == "1", f"{v}=1 required (production parity)"

import json
import struct
import time
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn

CKPT = Path.home() / ".exo/models/deepseek-ai--DeepSeek-V4-Flash-0731"
OUT = Path.home() / "repos/exo/tmp/p05-lmhead-mxfp8-20260830"
OUT.mkdir(parents=True, exist_ok=True)
V, D, R = 129280, 4096, 256


def load_bf16_from_shard(shard, key):
    with open(CKPT / shard, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
        info = hdr[key]
        o0, o1 = info["data_offsets"]
        fh.seek(n + 8 + o0)
        raw = fh.read(o1 - o0)
    u16 = np.frombuffer(raw, dtype=np.uint16)
    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    return f32.reshape(info["shape"])


def bench(fn, n=20, warmup=5):
    for _ in range(warmup):
        mx.eval(fn())
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1e6


def main():
    print("loading real head.weight + markov_w2 from checkpoint shards...")
    head_np = load_bf16_from_shard("model-00045-of-00048.safetensors", "head.weight")
    markov_np = load_bf16_from_shard(
        "model-00048-of-00048.safetensors", "mtp.2.markov_head.markov_w2.weight"
    )
    assert head_np.shape == (V, D) and markov_np.shape == (V, R)

    lm_bf16 = nn.Linear(D, V, bias=False)
    lm_bf16.weight = mx.array(head_np).astype(mx.bfloat16)
    lm_bf16.eval()
    mx.eval(lm_bf16.parameters())

    lm_q = nn.Linear(D, V, bias=False)
    lm_q.weight = mx.array(head_np).astype(mx.bfloat16)
    qmod = lm_q.to_quantized(group_size=32, bits=8, mode="mxfp8")
    qmod.eval()
    mx.eval(qmod.parameters())

    mk_bf16 = nn.Linear(R, V, bias=False)
    mk_bf16.weight = mx.array(markov_np).astype(mx.bfloat16)
    mk_bf16.eval()
    mx.eval(mk_bf16.parameters())
    mk_q = nn.Linear(R, V, bias=False)
    mk_q.weight = mx.array(markov_np).astype(mx.bfloat16)
    mkq = mk_q.to_quantized(group_size=32, bits=8, mode="mxfp8")
    mkq.eval()
    mx.eval(mkq.parameters())

    w_bytes = {
        "lm_bf16": V * D * 2,
        "lm_mxfp8": qmod.weight.nbytes + qmod.scales.nbytes,
        "mk_bf16": V * R * 2,
        "mk_mxfp8": mkq.weight.nbytes + mkq.scales.nbytes,
    }
    print("weight bytes:", {k: f"{v/1e6:.1f}MB" for k, v in w_bytes.items()})

    rng = np.random.default_rng(20260830)
    Hs = rng.standard_normal((16, D)).astype(np.float32)
    Hn = Hs / (np.sqrt((Hs ** 2).mean(-1, keepdims=True)) + 1e-6)
    x = mx.array(Hn).astype(mx.bfloat16)

    results = {"host": os.uname().nodename, "gpu": mx.device_info()["architecture"],
               "weights": w_bytes, "lm_head": {}, "markov_w2": {}, "sampler": {}}

    print(f"\n{'op':<22} {'M':>2} {'BF16 us':>10} {'mxfp8 us':>10} {'speedup':>8} "
          f"{'BF16 GB/s':>10} {'mxfp8 GB/s':>10}")
    for M in (1, 3, 4):
        xm = x[:M]
        t_b = bench(lambda: lm_bf16(xm))
        t_q = bench(lambda: qmod(xm))
        # bytes: weight read once per call (batched) for both
        g_b = w_bytes["lm_bf16"] / (t_b * 1e-6) / 1e9
        g_q = w_bytes["lm_mxfp8"] / (t_q * 1e-6) / 1e9
        print(f"{'lm_head':<22} {M:>2} {t_b:>10.1f} {t_q:>10.1f} "
              f"{t_b/t_q:>7.2f}x {g_b:>10.0f} {g_q:>10.0f}")
        results["lm_head"][M] = {"bf16_us": t_b, "mxfp8_us": t_q,
                                 "bf16_gbps": g_b, "mxfp8_gbps": g_q}

    # markov_w2 at M=1 (draft loop: called 3x per draft, M=1 each)
    mr = rng.standard_normal((4, R)).astype(np.float32)
    mr = mr / (np.sqrt((mr ** 2).mean(-1, keepdims=True)) + 1e-6)
    xr = mx.array(mr).astype(mx.bfloat16)[:1]
    t_b = bench(lambda: mk_bf16(xr))
    t_q = bench(lambda: mkq(xr))
    print(f"{'markov_w2':<22} {1:>2} {t_b:>10.1f} {t_q:>10.1f} {t_b/t_q:>7.2f}x "
          f"{w_bytes['mk_bf16']/(t_b*1e-6)/1e9:>10.0f} "
          f"{w_bytes['mk_mxfp8']/(t_q*1e-6)/1e9:>10.0f}")
    results["markov_w2"] = {"bf16_us": t_b, "mxfp8_us": t_q}

    # sampler unit (production form: lp - logsumexp(lp) then argmax)
    x4 = x[:4]

    def sampler_bf16():
        lp = lm_bf16(x4).astype(mx.float32)
        logprobs = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        return mx.argmax(logprobs, axis=-1)

    def sampler_q():
        lp = qmod(x4).astype(mx.float32)
        logprobs = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        return mx.argmax(logprobs, axis=-1)

    t_b = bench(sampler_bf16)
    t_q = bench(sampler_q)
    print(f"{'sampler(L=4)':<22} {4:>2} {t_b:>10.1f} {t_q:>10.1f} {t_b/t_q:>7.2f}x")
    results["sampler"] = {"bf16_us": t_b, "mxfp8_us": t_q}

    # ---- production-cycle projection (per P03 spec-ON cycle table) ----
    # verify lm_head L=4 x1 + draft lm_head L=3 x1 + markov x3 + sampler L=4
    # (sampler rides the verify logits; draft L=3 lm_head inside draft tail)
    cyc_b = (results["lm_head"][4]["bf16_us"]
             + results["lm_head"][3]["bf16_us"]
             + 3 * results["markov_w2"]["bf16_us"]
             + results["sampler"]["bf16_us"])
    cyc_q = (results["lm_head"][4]["mxfp8_us"]
             + results["lm_head"][3]["mxfp8_us"]
             + 3 * results["markov_w2"]["mxfp8_us"]
             + results["sampler"]["mxfp8_us"])
    tok_per_cycle = 3.2
    print(f"\nlm_head-family per cycle: BF16 {cyc_b/1e3:.2f} ms vs mxfp8 {cyc_q/1e3:.2f} ms"
          f" -> delta {(cyc_b-cyc_q)/1e3:+.2f} ms/cycle = {(cyc_b-cyc_q)/1e3/tok_per_cycle:+.2f} ms/token")
    print(f"(P03 projection was -0.7 ms/token, from -2.3 ms/cycle)")
    results["cycle_projection"] = {
        "bf16_ms": cyc_b / 1e3, "mxfp8_ms": cyc_q / 1e3,
        "delta_ms_per_cycle": (cyc_b - cyc_q) / 1e3,
        "delta_ms_per_token": (cyc_b - cyc_q) / 1e3 / tok_per_cycle,
    }

    out = OUT / "studio_lmhead_microbench.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()