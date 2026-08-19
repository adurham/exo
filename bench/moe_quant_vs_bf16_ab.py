"""MoE quantized-vs-bf16 A/B: is the 62.6%-of-ceiling gap dequant or intrinsic?

`bench/moe_production_class_bench.py` measured the real mlx-lm SwitchGLU at
DeepSeek-V4-Flash TP-rank prefill shapes running at 62.6% of a matched-shape
dense-GEMM ceiling, and `docs/moe-per-stage-gpu-breakdown-2026-08-18.md` proved
the missing time is INSIDE the three expert GEMMs (gather/scatter are ~4-5%).

Two candidate explanations for the in-kernel gap:
  (A) DEQUANTIZATION overhead -- the quantized kernel must unpack 4-bit/8-bit
      weights + scales to bf16 before the MAC, costing ALU/registers the dense
      fp16 GEMM never pays.
  (B) INTRINSIC to the ragged gathered matmul -- per-expert run lengths are
      small and uneven, so tile occupancy / weight-stream reuse is poor
      regardless of the weight dtype.

These are separable with a 2x2 at IDENTICAL shape and IDENTICAL routing:

                      | bf16 weights      | quantized weights
    ------------------+-------------------+---------------------------
    gathered (ragged) | gather_mm         | gather_qmm   <- production
    dense (M rows)    | plain matmul      | quantized_matmul

  - dense_bf16 is the ceiling (what the 62.6% was measured against).
  - dense_quant  / dense_bf16   isolates PURE dequant cost (no raggedness).
  - gather_bf16  / dense_bf16   isolates PURE raggedness cost (no dequant).
  - gather_quant / dense_bf16   is the production 62.6% number, reproduced.

If dequant were the cause, dense_quant/dense_bf16 would be ~= the production
ratio. If raggedness were the cause, gather_bf16/dense_bf16 would be.

All arms run in ONE process, interleaved, so clocks/thermals are fair. Every
arm uses the SAME routing indices, so per-expert run lengths are identical
across quantized and bf16 gathered arms.

Quant modes covered:
  mxfp4 g32   -- what make_quantization_config() in deepseek_v4.py specifies
                 for routed experts (the conversion recipe).
  affine 8b g64 -- what the DEPLOYED checkpoint's config.json actually says
                 (see MOE_KERNEL_HANDOFF.md "THE PREMISE CORRECTION"). Both
                 are measured so the answer is not recipe-vs-checkpoint
                 dependent.

Run:  .venv/bin/python bench/moe_quant_vs_bf16_ab.py
"""

from __future__ import annotations

import argparse
import json
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
DTYPE = mx.bfloat16

# 1024 = real per-rank L at EXO_PREFILL_STEP_SIZE=2048 (SEQ_SPLIT halves the
# nominal chunk across the 2 TP ranks); 2048 = real per-rank L at 4096.
DEFAULT_LENGTHS = (1024, 2048)

QUANT_ARMS = (
    ("mxfp4_g32", dict(group_size=32, bits=4, mode="mxfp4")),
    ("affine8_g64", dict(group_size=64, bits=8, mode="affine")),
)

WARMUP = 5
ITERS = 20


def make_routing(n_tokens: int, seed: int = 0) -> mx.array:
    """Realistic ragged top-k routing with a skewed per-expert prior.

    Identical generator to bench/moe_production_class_bench.py so the two
    benchmarks' numbers are directly comparable.
    """
    mx.random.seed(seed)
    prior = mx.random.normal((N_EXPERTS,)) * 1.0
    logits = mx.random.normal((n_tokens, N_EXPERTS)) + prior
    idx = mx.argpartition(-logits, TOP_K, axis=-1)[:, :TOP_K]
    return idx.astype(mx.uint32)


def bench_call(fn, warmup: int = WARMUP, iters: int = ITERS):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    samples = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), samples


def moe_flops(assignments: int) -> float:
    """Activated-only FLOPs: gate + up (K=HIDDEN,N=INTER) + down (K=INTER,N=HIDDEN)."""
    return 2.0 * assignments * (2 * HIDDEN * INTER_PER_RANK + INTER_PER_RANK * HIDDEN)


def build_switchglu(quant: dict | None) -> SwitchGLU:
    glu = SwitchGLU(HIDDEN, INTER_PER_RANK, N_EXPERTS, bias=False)
    glu.set_dtype(DTYPE)
    if quant is not None:
        nn.quantize(glu, **quant)
    mx.eval(glu.parameters())
    return glu


def dense_arms(M: int, quant: dict | None):
    """Matched-shape dense arm: 2x (M,H)@(H,I) + 1x (M,I)@(I,H), no gather."""
    a1 = mx.random.normal((M, HIDDEN)).astype(DTYPE)
    a2 = mx.random.normal((M, INTER_PER_RANK)).astype(DTYPE)
    w1 = mx.random.normal((INTER_PER_RANK, HIDDEN)).astype(DTYPE)
    w1b = mx.random.normal((INTER_PER_RANK, HIDDEN)).astype(DTYPE)
    w2 = mx.random.normal((HIDDEN, INTER_PER_RANK)).astype(DTYPE)
    mx.eval(a1, a2, w1, w1b, w2)

    if quant is None:
        # transpose=True convention: y = x @ W.T, W is (out, in)
        b1, b1b, b2 = w1.T, w1b.T, w2.T
        mx.eval(b1, b1b, b2)

        def fn():
            return (a1 @ b1, a1 @ b1b, a2 @ b2)

        return fn

    packed = []
    for w in (w1, w1b, w2):
        q = mx.quantize(w, **quant)
        mx.eval(*q)
        packed.append(q)
    gs, bits, mode = quant["group_size"], quant["bits"], quant["mode"]

    def fn():
        out = []
        for a, q in zip((a1, a1, a2), packed):
            wq, sc, *bi = q
            out.append(
                mx.quantized_matmul(
                    a, wq, sc, bi[0] if bi else None,
                    transpose=True, group_size=gs, bits=bits, mode=mode,
                )
            )
        return tuple(out)

    return fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="*", default=list(DEFAULT_LENGTHS))
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--json-out", type=str, default=None)
    args = ap.parse_args()

    print("MLX:", mx.__version__, "|", mx.default_device())
    print(
        f"config: hidden={HIDDEN} inter/rank={INTER_PER_RANK} experts={N_EXPERTS} "
        f"top_k={TOP_K} act_dtype={DTYPE}"
    )

    # Build every module ONCE (weight init is expensive) and reuse across L.
    modules = {"bf16": build_switchglu(None)}
    for name, q in QUANT_ARMS:
        modules[name] = build_switchglu(q)
    print("arms:", ", ".join(modules))

    records = []
    for n_tokens in args.lengths:
        idx = make_routing(n_tokens)
        x = mx.random.normal((1, n_tokens, HIDDEN)).astype(DTYPE)
        mx.eval(x, idx)

        counts = mx.zeros((N_EXPERTS,), dtype=mx.uint32)
        counts = counts.at[idx.flatten()].add(mx.ones((idx.size,), dtype=mx.uint32))
        mx.eval(counts)
        nz = [v for v in counts.tolist() if v > 0]
        M = n_tokens * TOP_K
        flops = moe_flops(M)

        print(
            f"\n===== L={n_tokens} (per-rank)  assignments M={M}  "
            f"experts_used={len(nz)}/{N_EXPERTS}  rows/expert "
            f"min={min(nz)} median={statistics.median(nz):.0f} max={max(nz)} ====="
        )

        row = {"L": n_tokens, "M": M, "experts_used": len(nz)}

        # --- gathered arms (production topology) ---
        for name, glu in modules.items():
            t, s = bench_call(
                lambda g=glu, _x=x, _i=idx: g(_x, _i), iters=args.iters
            )
            row[f"gather_{name}_ms"] = t * 1e3
            row[f"gather_{name}_tflops"] = flops / t / 1e12
            print(
                f"  gather  {name:12s} {t*1e3:8.3f} ms  "
                f"{flops/t/1e12:6.2f} TFLOPS  (best {min(s)*1e3:7.3f} ms)"
            )

        # --- dense arms (same M,K,N; no gather, no raggedness) ---
        for name, q in (("bf16", None), *QUANT_ARMS):
            fn = dense_arms(M, q)
            t, s = bench_call(fn, iters=args.iters)
            row[f"dense_{name}_ms"] = t * 1e3
            row[f"dense_{name}_tflops"] = flops / t / 1e12
            print(
                f"  dense   {name:12s} {t*1e3:8.3f} ms  "
                f"{flops/t/1e12:6.2f} TFLOPS  (best {min(s)*1e3:7.3f} ms)"
            )

        # --- attribution ---
        ceil_tf = row["dense_bf16_tflops"]
        print(f"\n  -- % of dense-bf16 ceiling ({ceil_tf:.2f} TFLOPS) --")
        for key in sorted(k for k in row if k.endswith("_tflops")):
            pct = row[key] / ceil_tf * 100.0
            row[key.replace("_tflops", "_pct_of_ceiling")] = pct
            print(f"     {key[:-7]:24s} {pct:6.1f}%")

        for qname, _ in QUANT_ARMS:
            deq = row[f"dense_{qname}_pct_of_ceiling"]
            rag = row["gather_bf16_pct_of_ceiling"]
            prod = row[f"gather_{qname}_pct_of_ceiling"]
            print(
                f"\n  [{qname}] dequant-only={deq:.1f}%  raggedness-only={rag:.1f}%  "
                f"production(both)={prod:.1f}%  "
                f"multiplicative-predict={deq*rag/100:.1f}%"
            )
        records.append(row)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(records, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
