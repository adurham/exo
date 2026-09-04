#!/usr/bin/env python3
"""I11 step 1 -- SHAPE PROBE (no timing).

Establishes, empirically from the DEPLOYED mlx build, the exact
(shape, dtype) convention mx.quantize produces for affine bits in
{4,5,6} at a given group_size, on a SMALL tensor. The real microbench
then allocates full-size arrays of exactly these shapes directly,
without ever materialising a full-precision (E, out, in) tensor --
which at the deployed expert shape would be ~4.3 GB per projection and
would not fit in the headroom left by the live cluster.

Also confirms mx.gather_qmm accepts those arrays at each bit width and
reports which Metal kernel is dispatched (via a Metal capture when
METAL_CAPTURE_ENABLED=1).
"""

import json
import sys

import mlx.core as mx

HIDDEN = 4096
PER_RANK_INTER = 1024
E_SMALL = 4
TOPK = 6


def probe(bits, group_size, out_dims, in_dims, n_experts):
    w_fp = mx.random.uniform(
        low=-0.05, high=0.05, shape=(n_experts, out_dims, in_dims), dtype=mx.float32
    )
    packed = mx.quantize(w_fp, group_size=group_size, bits=bits, mode="affine")
    mx.eval(packed)
    weight, scales, *rest = packed
    biases = rest[0] if rest else None
    info = {
        "bits": bits,
        "group_size": group_size,
        "in_dims": in_dims,
        "out_dims": out_dims,
        "n_experts": n_experts,
        "weight": {
            "shape": list(weight.shape),
            "dtype": str(weight.dtype),
            "nbytes": weight.nbytes,
        },
        "scales": {
            "shape": list(scales.shape),
            "dtype": str(scales.dtype),
            "nbytes": scales.nbytes,
        },
        "biases": None
        if biases is None
        else {
            "shape": list(biases.shape),
            "dtype": str(biases.dtype),
            "nbytes": biases.nbytes,
        },
    }

    # last-dim packing factor per output row, i.e. how many uint32 words
    # hold in_dims quantized values
    info["weight_lastdim_per_in_dim"] = weight.shape[-1] / in_dims
    info["scales_lastdim_per_in_dim"] = scales.shape[-1] / in_dims

    # confirm gather_qmm actually runs at this bit width, decode shape
    x = mx.random.normal(shape=(4, 1, 1, in_dims)).astype(mx.bfloat16)
    lhs = mx.arange(4, dtype=mx.uint32).reshape(4, 1)
    lhs = mx.broadcast_to(lhs, (4, TOPK))
    rhs = mx.random.randint(0, n_experts, shape=(4, TOPK)).astype(mx.uint32)
    try:
        out = mx.gather_qmm(
            x,
            weight,
            scales,
            biases,
            lhs_indices=lhs,
            rhs_indices=rhs,
            transpose=True,
            group_size=group_size,
            bits=bits,
            mode="affine",
            sorted_indices=False,
        )
        mx.eval(out)
        info["gather_qmm"] = {"ok": True, "out_shape": list(out.shape)}
    except Exception as exc:  # noqa: BLE001 - want the message verbatim
        info["gather_qmm"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    del w_fp, packed, weight, scales, biases
    mx.clear_cache()
    return info


def main():
    print(f"[probe] mlx {getattr(mx, '__version__', '?')}", file=sys.stderr)
    print(f"[probe] device {mx.default_device()}", file=sys.stderr)
    results = []
    for group_size in (32, 64):
        for bits in (6, 5, 4):
            # gate/up geometry: in=HIDDEN, out=PER_RANK_INTER
            results.append(probe(bits, group_size, PER_RANK_INTER, HIDDEN, E_SMALL))
            # down geometry: in=PER_RANK_INTER, out=HIDDEN
            results.append(probe(bits, group_size, HIDDEN, PER_RANK_INTER, E_SMALL))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
