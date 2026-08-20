#!/Users/adam.durham/repos/exo/.venv/bin/python
"""Offline repro harness for the quantized moe.all_sum failure (2026-08-19).

Isolates whether the bug is in OUR quantize/all_gather/dequant logic or in the
transport's all_gather contract, WITHOUT touching the live cluster.

Modes:
  --mode local   single process, world=1 simulated: pure math + shape/dtype audit
  --mode dist    real multi-rank; run under `mlx.launch -n 2 --backend ring ...`

The distributed mode is deliberately staged so the FIRST failing stage names
itself: each stage is a separate, barrier-separated collective.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import mlx.core as mx


def log(rank: int, msg: str) -> None:
    sys.stderr.write(f"[rank{rank}] {msg}\n")
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# The code under test, lifted verbatim from
# mlx-lm@feat/moe-allsum-quant-2026-08-19 : mlx_lm/models/deepseek_v4.py
# ---------------------------------------------------------------------------
def dequant_sum_shards(
    wq_all, scales_all, biases_all, *, n_shards, out_shape, out_dtype, bits, group_size
):
    per_rank_rows = wq_all.shape[0] // n_shards
    total = mx.zeros(out_shape, dtype=mx.float32).reshape(per_rank_rows, -1)
    for _r in range(n_shards):
        _lo = _r * per_rank_rows
        _hi = _lo + per_rank_rows
        _deq = mx.dequantize(
            wq_all[_lo:_hi],
            scales_all[_lo:_hi],
            biases_all[_lo:_hi],
            group_size=group_size,
            bits=bits,
        )
        total = total + _deq.astype(mx.float32)
    return total.reshape(out_shape).astype(out_dtype)


def quantized_moe_all_sum(y, group, *, bits=8, group_size=64, verbose=False, rank=0):
    orig_dtype = y.dtype
    orig_shape = y.shape
    world_size = group.size()
    y2 = y.reshape(-1, orig_shape[-1]).astype(mx.float32)
    wq, scales, biases = mx.quantize(y2, group_size=group_size, bits=bits)
    if verbose:
        log(rank, f"quantize: y2={y2.shape}/{y2.dtype} -> wq={wq.shape}/{wq.dtype} "
                  f"scales={scales.shape}/{scales.dtype} biases={biases.shape}/{biases.dtype}")
        log(rank, f"nbytes: y={y.nbytes} wq={wq.nbytes} scales={scales.nbytes} biases={biases.nbytes}")

    if verbose:
        log(rank, "STAGE A: all_gather(wq) ...")
    wq_all = mx.distributed.all_gather(wq, group=group)
    mx.eval(wq_all)
    if verbose:
        log(rank, f"STAGE A ok -> {wq_all.shape}/{wq_all.dtype}")
        log(rank, "STAGE B: all_gather(scales) ...")
    scales_all = mx.distributed.all_gather(scales, group=group)
    mx.eval(scales_all)
    if verbose:
        log(rank, f"STAGE B ok -> {scales_all.shape}/{scales_all.dtype}")
        log(rank, "STAGE C: all_gather(biases) ...")
    biases_all = mx.distributed.all_gather(biases, group=group)
    mx.eval(biases_all)
    if verbose:
        log(rank, f"STAGE C ok -> {biases_all.shape}/{biases_all.dtype}")

    return dequant_sum_shards(
        wq_all, scales_all, biases_all,
        n_shards=world_size, out_shape=orig_shape, out_dtype=orig_dtype,
        bits=bits, group_size=group_size,
    )


# ---------------------------------------------------------------------------
def run_local(args) -> int:
    """Single process: audit shapes/dtypes and the local math, no collectives."""
    bits, gs = args.bits, args.group
    failures = []
    print("=" * 72)
    print("LOCAL MODE: shape/dtype contract audit + math (no collectives)")
    print("=" * 72)

    y = mx.random.normal((args.batch, args.hidden)).astype(mx.bfloat16)
    y2 = y.reshape(-1, y.shape[-1]).astype(mx.float32)
    wq, scales, biases = mx.quantize(y2, group_size=gs, bits=bits)
    mx.eval(wq, scales, biases)
    print(f"y      {tuple(y.shape)} {y.dtype} nbytes={y.nbytes}")
    print(f"wq     {tuple(wq.shape)} {wq.dtype} nbytes={wq.nbytes}")
    print(f"scales {tuple(scales.shape)} {scales.dtype} nbytes={scales.nbytes}")
    print(f"biases {tuple(biases.shape)} {biases.dtype} nbytes={biases.nbytes}")
    total_q = wq.nbytes + scales.nbytes + biases.nbytes
    print(f"payload/rank: bf16 all_sum={y.nbytes}  quant all_gather={total_q} "
          f"({100.0*total_q/y.nbytes:.1f}% of baseline, before x(world) gather blowup)")

    # --- FINDING 1: is the fp32 collective wrapper in scope for scales/biases?
    if scales.dtype == mx.float32:
        print("\n!! scales/biases are float32 -> deepseek_v4's _collective_fp32_safe "
              "wrapper WILL silently downcast them to bfloat16 on the wire.")

    # --- FINDING 2: does all_gather concatenate along axis 0 as assumed?
    print("\nSimulating a 2-shard gather locally (mx.concatenate on axis 0):")
    n = 2
    shards = [mx.random.normal((args.batch, args.hidden)).astype(mx.float32) for _ in range(n)]
    qs = [mx.quantize(s, group_size=gs, bits=bits) for s in shards]
    wq_all = mx.concatenate([q[0] for q in qs], axis=0)
    sc_all = mx.concatenate([q[1] for q in qs], axis=0)
    bi_all = mx.concatenate([q[2] for q in qs], axis=0)
    got = dequant_sum_shards(
        wq_all, sc_all, bi_all, n_shards=n,
        out_shape=(args.batch, args.hidden), out_dtype=mx.bfloat16,
        bits=bits, group_size=gs,
    )
    exact = sum(shards)
    mx.eval(got, exact)
    err = float(mx.max(mx.abs(got.astype(mx.float32) - exact)))
    rel = err / float(mx.max(mx.abs(exact)))
    print(f"  max abs err={err:.5f}  rel={rel:.5f}  (int{bits} g{gs})")
    if rel > 0.05:
        failures.append(f"local dequant-sum rel error {rel:.4f} too high")
    else:
        print("  -> LOCAL MATH IS CORRECT (matches the 6 passing unit tests)")

    # --- FINDING 3: the axis-0 assumption when y is 3-D (B, L, H) -----------
    print("\n3-D reshape check (real MoE call site passes (B, L, H)):")
    y3 = mx.random.normal((2, args.batch // 2, args.hidden)).astype(mx.bfloat16)
    y2b = y3.reshape(-1, y3.shape[-1])
    print(f"  y3={tuple(y3.shape)} -> y2={tuple(y2b.shape)}; "
          f"out_shape passed to dequant_sum_shards = {tuple(y3.shape)}")
    per_rank_rows = y2b.shape[0]
    print(f"  dequant_sum_shards allocates mx.zeros(out_shape).reshape({per_rank_rows}, -1)")
    ok3 = (y3.size == per_rank_rows * (y3.size // per_rank_rows))
    print(f"  reshape compatible: {ok3}")

    print("\nLOCAL VERDICT:", "FAIL: " + "; ".join(failures) if failures else
          "our quantize/dequant/sum math is SOUND in-process. "
          "Any live failure must therefore be in the all_gather transport path.")
    return 1 if failures else 0


def run_dist(args) -> int:
    world = mx.distributed.init()
    rank, size = world.rank(), world.size()
    log(rank, f"init ok: rank={rank}/{size} backend-visible")

    # Stage 0: prove a plain bf16 all_sum of the SAME payload works (control).
    y = (mx.ones((args.batch, args.hidden), dtype=mx.bfloat16) * (rank + 1))
    mx.eval(y)
    log(rank, f"STAGE 0 (control): all_sum bf16 {tuple(y.shape)} ...")
    t = time.time()
    ctrl = mx.distributed.all_sum(y, group=world)
    mx.eval(ctrl)
    log(rank, f"STAGE 0 ok in {time.time()-t:.3f}s, ctrl[0,0]={float(ctrl[0,0])} "
              f"(expect {sum(range(1, size+1))})")

    # Stage 0b: a plain bf16 all_gather (is all_gather itself healthy at all?)
    log(rank, "STAGE 0b (control): all_gather bf16 ...")
    t = time.time()
    g = mx.distributed.all_gather(y, group=world)
    mx.eval(g)
    log(rank, f"STAGE 0b ok in {time.time()-t:.3f}s -> {tuple(g.shape)}")

    # Stage 0c: all_gather of a uint32 array (the wq dtype) -- the real suspect.
    u = mx.full((args.batch, max(1, args.hidden // 4)), rank + 1, dtype=mx.uint32)
    mx.eval(u)
    log(rank, f"STAGE 0c: all_gather uint32 {tuple(u.shape)} ...")
    t = time.time()
    gu = mx.distributed.all_gather(u, group=world)
    mx.eval(gu)
    log(rank, f"STAGE 0c ok in {time.time()-t:.3f}s -> {tuple(gu.shape)}/{gu.dtype}")

    # Stage 1: the real thing.
    yq = mx.random.normal((args.batch, args.hidden)).astype(mx.bfloat16) * (rank + 1)
    mx.eval(yq)
    log(rank, "STAGE 1: quantized_moe_all_sum ...")
    t = time.time()
    out = quantized_moe_all_sum(
        yq, world, bits=args.bits, group_size=args.group, verbose=True, rank=rank
    )
    mx.eval(out)
    log(rank, f"STAGE 1 ok in {time.time()-t:.3f}s -> {tuple(out.shape)}/{out.dtype}")

    # Correctness vs exact all_sum of the same input.
    exact = mx.distributed.all_sum(yq.astype(mx.bfloat16), group=world)
    mx.eval(exact)
    err = float(mx.max(mx.abs(out.astype(mx.float32) - exact.astype(mx.float32))))
    denom = float(mx.max(mx.abs(exact.astype(mx.float32))))
    log(rank, f"RESULT: max abs err={err:.5f} rel={err/denom:.5f}")
    log(rank, "ALL STAGES PASSED")
    return 0


def run_dist_lazy(args) -> int:
    """Production-faithful: NO intermediate mx.eval, fp32-safe wrapper active,
    called in a loop like the real per-layer MoE combine point.

    This is the shape that actually ran on the cluster. It tests the
    'the 3 all_gathers are not ordered/paired across ranks' hypothesis, which
    is an MLX-graph-scheduling property and therefore transport-independent.
    """
    world = mx.distributed.init()
    rank = world.rank()

    # Reproduce deepseek_v4.py's _collective_fp32_safe wrapper exactly.
    orig_gather = mx.distributed.all_gather
    downcast_count = [0]

    def fp32_safe(x, *a, **kw):
        if isinstance(x, mx.array) and x.dtype == mx.float32:
            downcast_count[0] += 1
            return orig_gather(x.astype(mx.bfloat16), *a, **kw).astype(mx.float32)
        return orig_gather(x, *a, **kw)

    mx.distributed.all_gather = fp32_safe
    log(rank, "installed _collective_fp32_safe wrapper (as production does)")

    n_layers = args.layers
    log(rank, f"LAZY MODE: {n_layers} sequential quantized all_sums, no intermediate eval")
    t = time.time()
    h = mx.random.normal((args.batch, args.hidden)).astype(mx.bfloat16) * (rank + 1)
    for _i in range(n_layers):
        h = quantized_moe_all_sum(
            h, world, bits=args.bits, group_size=args.group, verbose=False, rank=rank
        )
        h = h * mx.array(0.5, dtype=mx.bfloat16)  # stand-in for the rest of the layer
    mx.eval(h)
    log(rank, f"LAZY MODE ok: {n_layers} layers in {time.time()-t:.3f}s, "
              f"fp32 downcasts={downcast_count[0]} (2 per layer: scales+biases), "
              f"h[0,0]={float(h[0,0]):.5f}")
    mx.distributed.all_gather = orig_gather
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["local", "dist", "dist-lazy"], default="local")
    p.add_argument("--layers", type=int, default=32)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--bits", type=int, default=int(os.environ.get("EXO_DSV4_MOE_ALLSUM_QUANT_BITS", "8")))
    p.add_argument("--group", type=int, default=int(os.environ.get("EXO_DSV4_MOE_ALLSUM_QUANT_GROUP", "64")))
    args = p.parse_args()
    if args.mode == "local":
        return run_local(args)
    if args.mode == "dist":
        return run_dist(args)
    return run_dist_lazy(args)


if __name__ == "__main__":
    sys.exit(main())
