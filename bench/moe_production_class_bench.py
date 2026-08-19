"""Production-class MoE expert-GEMM microbenchmark.

Instantiates the REAL mlx-lm SwitchGLU (quantized via nn.quantize, exactly the
way exo's model loader does) at DeepSeek-V4-Flash TP-rank shapes, drives it with
a realistic *ragged* top-6-of-256 routing assignment, and reports achieved
TFLOPS using ACTIVATED-only FLOP counting.

Then measures a matched-shape dense fp16 GEMM ceiling in the same process so the
comparison is thermally/clock fair.

Deliberately does NOT call mx.gather_qmm with hand-picked tile params -- MLX
picks its own kernel exactly as production does.

Run:  .venv/bin/python bench/moe_production_class_bench.py
"""

import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mlx-lm"))
from mlx_lm.models.switch_layers import SwitchGLU  # noqa: E402

# --- DeepSeek-V4-Flash production config (per TP rank, worldSize=2) ---
HIDDEN = 4096
INTER_PER_RANK = 1024  # moe_intermediate_size 2048 // 2 TP ranks
N_EXPERTS = 256
TOP_K = 6
GROUP_SIZE = 32
BITS = 4
QMODE = "mxfp4"
DTYPE = mx.bfloat16

WARMUP = 5
ITERS = 20


def make_routing(n_tokens: int, seed: int = 0) -> mx.array:
    """Realistic ragged top-k routing: argsort of random gated logits.

    Uses a Zipf-ish expert bias so the per-expert group sizes are skewed the way
    real routers are, not a uniform n_tokens*TOP_K/N_EXPERTS everywhere.
    """
    mx.random.seed(seed)
    # skewed expert prior (some experts genuinely more popular)
    prior = mx.random.normal((N_EXPERTS,)) * 1.0
    logits = mx.random.normal((n_tokens, N_EXPERTS)) + prior
    idx = mx.argpartition(-logits, TOP_K, axis=-1)[:, :TOP_K]
    return idx.astype(mx.uint32)


def bench_call(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        y = fn()
        mx.eval(y)
    mx.synchronize()
    samples = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        y = fn()
        mx.eval(y)
        mx.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), samples


def main():
    print("MLX:", mx.__version__)
    print(
        f"config: hidden={HIDDEN} inter/rank={INTER_PER_RANK} experts={N_EXPERTS} "
        f"top_k={TOP_K} quant={QMODE} g={GROUP_SIZE} b={BITS} dtype={DTYPE}"
    )

    glu = SwitchGLU(HIDDEN, INTER_PER_RANK, N_EXPERTS, bias=False)
    glu.set_dtype(DTYPE)
    nn.quantize(glu, group_size=GROUP_SIZE, bits=BITS, mode=QMODE)
    mx.eval(glu.parameters())
    print("quantized classes:", type(glu.gate_proj).__name__, type(glu.down_proj).__name__)

    results = []
    for n_tokens in (512, 2048, 8192):
        idx = make_routing(n_tokens)
        x = mx.random.normal((1, n_tokens, HIDDEN)).astype(DTYPE)
        mx.eval(x, idx)

        # group-size stats (raggedness evidence)
        counts = mx.zeros((N_EXPERTS,), dtype=mx.uint32)
        counts = counts.at[idx.flatten()].add(mx.ones((idx.size,), dtype=mx.uint32))
        mx.eval(counts)
        c = counts.tolist()
        nz = [v for v in c if v > 0]

        def fn():
            return glu(x, idx)

        t, samples = bench_call(fn)
        assign = n_tokens * TOP_K
        # activated-only FLOPs: gate + up (K=HIDDEN,N=INTER) + down (K=INTER,N=HIDDEN)
        flops = 2 * assign * (2 * HIDDEN * INTER_PER_RANK + INTER_PER_RANK * HIDDEN)
        tflops = flops / t / 1e12
        # weight bytes touched (4-bit) for experts actually used, worst case all
        used = len(nz)
        wbytes = used * 3 * HIDDEN * INTER_PER_RANK * BITS / 8
        gbs = wbytes / t / 1e9
        print(
            f"\n[MoE SwitchGLU] L={n_tokens:5d} assignments={assign:6d} "
            f"experts_used={used}/{N_EXPERTS} M/expert min={min(nz)} "
            f"median={statistics.median(nz):.0f} max={max(nz)}"
        )
        print(
            f"  median {t*1e3:8.3f} ms  p10 {min(samples)*1e3:7.3f}  "
            f"TFLOPS {tflops:6.2f}   weight-read {gbs:7.1f} GB/s"
        )
        results.append((n_tokens, assign, t, tflops))

    print("\n=== Matched-shape DENSE ceiling (same total M, K, N), same session ===")
    for n_tokens, assign, t_moe, tf_moe in results:
        M = assign
        # gate+up: (M,HIDDEN)x(HIDDEN,INTER); down: (M,INTER)x(INTER,HIDDEN)
        a1 = mx.random.normal((M, HIDDEN)).astype(mx.float16)
        b1 = mx.random.normal((HIDDEN, INTER_PER_RANK)).astype(mx.float16)
        b1b = mx.random.normal((HIDDEN, INTER_PER_RANK)).astype(mx.float16)
        a2 = mx.random.normal((M, INTER_PER_RANK)).astype(mx.float16)
        b2 = mx.random.normal((INTER_PER_RANK, HIDDEN)).astype(mx.float16)
        mx.eval(a1, b1, b1b, a2, b2)

        def dfn():
            return (a1 @ b1, a1 @ b1b, a2 @ b2)

        td, _ = bench_call(dfn)
        flops = 2 * M * (2 * HIDDEN * INTER_PER_RANK + INTER_PER_RANK * HIDDEN)
        tfd = flops / td / 1e12
        print(
            f"L={n_tokens:5d} M={M:6d}  dense {td*1e3:8.3f} ms {tfd:6.2f} TFLOPS | "
            f"MoE {t_moe*1e3:8.3f} ms {tf_moe:6.2f} TFLOPS | ratio {tf_moe/tfd*100:5.1f}%"
        )


if __name__ == "__main__":
    main()
