# Copyright © 2026 Adam Durham (hermes-gw)
"""Production-shape kernel bench: oQ (native-MLX mixed precision) vs affine-3.

STATUS: scaffold, not a verdict. The dense arm is valid. The expert arm is NOT
apples-to-apples yet -- the reference side quantizes only 32 of 384 experts, so
its gather touches a much smaller stack, and neither arm sorts the indices the
way the engine does before `gather_qmm`. Do not quote the expert ratios from
this script until both are fixed and an EXL3-kernel arm exists. The format-speed
question it was written to answer is already answered for the dense slice by the
2026-10-09 pricing round (native affine q4/q5-g64 = 2.47x/2.29x the EXL3 fused
kernel at m=4); what oQ2.5e actually needs measured is quality, not this.

The question this was meant to answer: the oQ2.5e pack's advantage is supposed
to be that its tensors run on stock MLX kernels instead of EXL3's trellis
decode. This measures the packed tensors the pack actually ships, at the row
counts the engine sees (1 = decode, 6 = spec verify, 512/2048 = prefill chunk),
against the Q1d reference config (affine 3-bit g64) quantized from the same
weights.

Arms
  oq    the pack's own packed weights + its per-module bits/group/mode
  a3    the same weights at affine 3-bit, group 64 (the Q1d reference)
  bf16  float baseline (dense only, for scale)

Everything is measured on layer 0 of the mini pack (~9.7 GiB real bytes).

Usage:
    PYTHONPATH=$PWD/src:$PWD/mlx-lm ~/repos/exo/.venv/bin/python \
        tests/dsv41_oq/kernel_bench.py --oq-pack ~/.hermes/cache/scratch/oqmini
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(ROOT, "mlx-lm"), os.path.join(ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from mlx_lm.models.deepseek_v41 import oq_build  # noqa: E402

DENSE = ["attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_b",
         "ffn.shared_experts.w1", "ffn.shared_experts.w2",
         "ffn.shared_experts.w3"]
EXPERTS = ["ffn.experts.w1", "ffn.experts.w2", "ffn.experts.w3"]


def timed(fn, reps=5, warmup=2):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize() if hasattr(mx, "synchronize") else None
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        samples.append((time.perf_counter() - t0) * 1e3)
    del out
    return statistics.median(samples)


def dense_arms(ck, module, rows):
    """-> list of (arm, ms, out_shape) for one dense module at `rows` rows."""
    w, s, b, qi = ck._parts(module)                      # noqa: SLF001
    bits, group, mode = oq_build._qparams(qi)            # noqa: SLF001
    out_dim, packed_in = w.shape
    in_dim = ((packed_in * 32) // bits) // 1
    x = mx.random.normal((rows, in_dim)).astype(mx.bfloat16)
    res = []
    res.append(("oq", timed(lambda: mx.quantized_matmul(
        x, w, s, b, transpose=True, group_size=group, bits=bits, mode=mode)), (rows, out_dim)))
    # reference: dequantize once, requantize at affine 3-bit g64
    deq = mx.dequantize(w, s, b, group_size=group, bits=bits, mode=mode).astype(mx.bfloat16)
    mx.eval(deq)
    w3, s3, b3 = mx.quantize(deq, group_size=64, bits=3)
    mx.eval(w3, s3, b3)
    res.append(("a3", timed(lambda: mx.quantized_matmul(
        x, w3, s3, b3, transpose=True, group_size=64, bits=3, mode="affine")), (rows, out_dim)))
    res.append(("bf16", timed(lambda: x @ deq.T), (rows, out_dim)))
    del w3, s3, b3, deq
    mx.clear_cache()
    return res, (bits, group, mode, in_dim, out_dim)


def expert_arms(ck, module, rows, k=6):
    w_full, s_full, b_full, qi = ck._parts(module)       # noqa: SLF001
    bits, group, mode = oq_build._qparams(qi)            # noqa: SLF001
    n_exp, out_dim, packed_in = w_full.shape
    in_dim = ((packed_in * 32) // bits)
    x = mx.random.normal((rows, in_dim)).astype(mx.bfloat16)
    idx = mx.array(np.random.randint(0, n_exp, size=(rows, k)).astype(np.int32))
    res = []
    res.append(("oq", timed(lambda: mx.gather_qmm(
        x, w_full, s_full, b_full, rhs_indices=idx, transpose=True,
        group_size=group, bits=bits, mode=mode)), (rows, k, out_dim)))
    # reference arm: same experts at affine 3-bit g64 (per-expert stacks kept)
    e_sel = mx.array(np.linspace(0, n_exp - 1, 32).astype(np.int32))
    deq = mx.dequantize(mx.take(w_full, e_sel, axis=0),
                        mx.take(s_full, e_sel, axis=0),
                        mx.take(b_full, e_sel, axis=0) if b_full is not None else None,
                        group_size=group, bits=bits, mode=mode).astype(mx.bfloat16)
    mx.eval(deq)
    w3, s3, b3 = mx.quantize(deq, group_size=64, bits=3)
    mx.eval(w3, s3, b3)
    idx32 = mx.array(np.random.randint(0, 32, size=(rows, k)).astype(np.int32))
    res.append(("a3(32exp)", timed(lambda: mx.gather_qmm(
        x, w3, s3, b3, rhs_indices=idx32, transpose=True,
        group_size=64, bits=3, mode="affine")), (rows, k, out_dim)))
    del w3, s3, b3, deq
    mx.clear_cache()
    return res, (bits, group, mode, n_exp, in_dim, out_dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oq-pack", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--dense-rows", default="1,6,2048")
    ap.add_argument("--expert-rows", default="1,6,256")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    ck = oq_build.OqCheckpoint(os.path.expanduser(args.oq_pack))
    pre = f"layers.{args.layer}."
    report: dict = {"layer": args.layer, "dense": {}, "experts": {}}

    for rows in [int(r) for r in args.dense_rows.split(",")]:
        print(f"\n## dense, rows={rows}  (ms, median of 5)\n")
        print("| module | bits/group/mode | in->out | oq | a3 | bf16 | oq/a3 |")
        print("|---|---|---|---|---|---|---|")
        for tail in DENSE:
            arm, meta = dense_arms(ck, pre + tail, rows)
            d = {k: v for k, v, _ in arm}
            report["dense"].setdefault(tail, {})[rows] = {
                k: round(v, 4) for k, v in d.items()}
            print(f"| {tail} | {meta[0]}/{meta[1]}/{meta[2]} | {meta[3]}->{meta[4]} | "
                  f"{d['oq']:.3f} | {d['a3']:.3f} | {d['bf16']:.3f} | {d['oq']/d['a3']:.2f} |")

    for rows in [int(r) for r in args.expert_rows.split(",")]:
        print(f"\n## routed experts, rows={rows}, top-6  (ms, median of 5)\n")
        print("| stack | bits/group/mode | oq | a3 (32 of 384) | oq/a3 |")
        print("|---|---|---|---|---|")
        for tail in EXPERTS:
            arm, meta = expert_arms(ck, pre + tail, rows)
            d = {k: v for k, v, _ in arm}
            report["experts"].setdefault(tail, {})[rows] = {
                k: round(v, 4) for k, v in d.items()}
            print(f"| {tail} | {meta[0]}/{meta[1]}/{meta[2]} | {d['oq']:.3f} | "
                  f"{d[f'a3(32exp)']:.3f} | {d['oq']/d['a3(32exp)']:.2f} |")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
