#!/usr/bin/env python3
"""q1e A3 -- actual 3-bit packing: mx.quantize effective bpw on the production rank expert shapes.

Runs OFFLINE on studio2 (idle-guarded), in the node venv. NO engine touch, no writes under
~/repos/exo, no generation request. Memory-lean: tensors are production-rank-shaped (tens of MB).

Question: what does mx.quantize ACTUALLY emit for bits=3, group_size in {128,64}? Its true
effective bpw discriminates 3.125 (dense 3-bit scale-only) vs 3.25 (scale+bias) vs 3.325
(older 10-vals-per-uint32 packing), and decides B_arm = 33.974 * eff_bpw GB/rank precisely.

Method:
  * For each production rank expert shape and each (bits, group_size): build a correctly-shaped
    tensor, mx.quantize it, and report q / scales / biases dtype+shape+nbytes.
    nbytes depends ONLY on shape+dtype+bits+group_size, not on the data (documented caveat:
    the tensor DATA here is synthetic random, the SHAPE is the real production rank shape).
  * eff_bpw = total_bytes*8/numel;  B_arm = 33.974 * eff_bpw (GB/rank, all 40 MoE layers).
  * q4g64 anchor: the Round-1 doc measured q4g64 = 4.25 bpw; if this method reproduces it the
    method is validated.
  * Also: MLX peak/active/cache semantics micro-test (does get_peak_memory count cached
    buffers? does reset zero it?) -- resolves the A2 non-weight-peak definition.
  * Also: best-effort read of a REAL expert's shapes from the node's EXL3 checkpoint
    (validation of the production geometry; data-independent for the bpw result).
"""
from __future__ import annotations

import json
import os
import sys
import time

import mlx.core as mx

B_ARM_BASIS = 33.974  # GB/rank per bpw (= N_rank/8/1e9, N_rank=271,790,899,200)

# Production rank expert shapes (native MLX [out,in] layout, per TP rank).
#   full H=2304 -> per-rank H=1152; D=5120.
SHAPES = {
    "gu_fused_rank": (2304, 5120),   # [2H_rank, D] fused gate+up
    "gate_rank":     (1152, 5120),   # [H_rank, D]
    "dn_rank":       (5120, 1152),   # [D, H_rank]
}
CONFIGS = [(3, 128), (3, 64), (4, 64)]  # (bits, group_size); q4g64 is the anchor


def prod(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def describe(a) -> dict:
    return {
        "dtype": str(a.dtype),
        "shape": [int(x) for x in a.shape],
        "nbytes": int(a.nbytes),
    }


def run_cfg(shape: tuple[int, ...], bits: int, gs: int, mode: str | None = None, W=None) -> dict:
    if W is None:
        W = mx.random.uniform(shape=shape, dtype=mx.float16)
    mx.eval(W)
    t0 = time.time()
    if mode:
        out = mx.quantize(W, group_size=gs, bits=bits, mode=mode)
    else:
        out = mx.quantize(W, group_size=gs, bits=bits)
    arrs = list(out) if isinstance(out, (tuple, list)) else [out]
    for a in arrs:
        mx.eval(a)
    names = ["q", "scales", "biases"]
    rec: dict = {
        "shape": list(shape),
        "bits": bits,
        "group_size": gs,
        "mode": mode,
        "n_arrays_returned": len(arrs),
        "numel": prod(shape),
        "W_dtype": str(W.dtype),
        "W_nbytes": int(W.nbytes),
    }
    total = 0
    parts = []
    for i, a in enumerate(arrs):
        nm = names[i] if i < len(names) else f"arr{i}"
        d = describe(a)
        rec[nm] = d
        total += d["nbytes"]
        parts.append(f"{nm}.nbytes={d['nbytes']}")
    rec["total_bytes"] = total
    rec["arithmetic"] = " + ".join(parts) + f" = {total} bytes"
    rec["eff_bpw"] = total * 8.0 / rec["numel"]
    rec["B_arm_gb"] = B_ARM_BASIS * rec["eff_bpw"]
    rec["seconds"] = round(time.time() - t0, 4)
    return rec


def mlx_mem_semantics() -> dict:
    """Resolve whether get_peak_memory counts cached (freed) buffers, and what reset does."""
    mx.eval(mx.zeros(1))
    r: dict = {}
    mx.reset_peak_memory()
    r["peak_after_reset_at_idle"] = int(mx.get_peak_memory())
    r["active_at_idle"] = int(mx.get_active_memory())
    r["cache_at_idle"] = int(mx.get_cache_memory())
    n = 256 * 1024 * 1024  # 256 MB uint8
    a = mx.zeros((n,), dtype=mx.uint8)
    mx.eval(a)
    r["after_alloc_256mb"] = {
        "active": int(mx.get_active_memory()),
        "peak": int(mx.get_peak_memory()),
        "cache": int(mx.get_cache_memory()),
    }
    del a
    mx.eval(mx.zeros(1))
    r["after_free"] = {
        "active": int(mx.get_active_memory()),
        "peak": int(mx.get_peak_memory()),
        "cache": int(mx.get_cache_memory()),
    }
    b = mx.zeros((n,), dtype=mx.uint8)
    mx.eval(b)
    r["after_realloc_256mb"] = {
        "active": int(mx.get_active_memory()),
        "peak": int(mx.get_peak_memory()),
        "cache": int(mx.get_cache_memory()),
    }
    del b
    mx.eval(mx.zeros(1))
    mx.reset_peak_memory()
    r["peak_after_2nd_reset"] = int(mx.get_peak_memory())
    r["active_after_2nd_reset"] = int(mx.get_active_memory())
    r["cache_after_2nd_reset"] = int(mx.get_cache_memory())
    return r


def real_expert() -> dict:
    """Best-effort: read one REAL expert's shapes from the node's EXL3 checkpoint."""
    r: dict = {"note": "validates production geometry; data-independent for eff_bpw"}
    try:
        sys.path.insert(0, os.path.join(os.path.expanduser("~/repos/exo"), "mlx-lm"))
        from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_dense_layer
        from mlx_lm.models.exl3.reconstruct import reconstruct_public_mlx

        model = os.path.expanduser(
            "~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw")
        ckpt = Exl3Checkpoint(model)
        pre = "layers.20.ffn.experts.0"
        for k in ("w1", "w2", "w3"):
            try:
                raw = load_dense_layer(ckpt, f"{pre}.{k}")
                r[k + "_stored_type"] = type(raw).__name__
                del raw
            except Exception as e:
                r[k + "_stored_err"] = repr(e)
        w1 = reconstruct_public_mlx(load_dense_layer(ckpt, f"{pre}.w1"))
        mx.eval(w1)
        shp = tuple(int(x) for x in w1.shape)
        r["w1_reconstructed"] = {"dtype": str(w1.dtype), "shape": list(shp)}
        # quantize the REAL reconstructed weight (nbytes == shape-determined)
        r["w1_recon_q3g128"] = run_cfg(shp, 3, 128, W=w1)
        r["w1_recon_q3g64"] = run_cfg(shp, 3, 64, W=w1)
        r["w1_recon_q4g64"] = run_cfg(shp, 4, 64, W=w1)
    except Exception as e:  # non-fatal: synthetic-shape result stands on its own
        r["error"] = repr(e)
    return r


def main() -> None:
    import platform
    out = {
        "bench": "q1e_a3_packing",
        "host": platform.node(),
        "mlx_version": mx.__version__,
        "B_arm_basis_gb_per_bpw": B_ARM_BASIS,
        "shapes": {k: list(v) for k, v in SHAPES.items()},
        "results": {},
    }
    for sname, shape in SHAPES.items():
        out["results"][sname] = {}
        for bits, gs in CONFIGS:
            key = f"q{bits}g{gs}"
            out["results"][sname][key] = run_cfg(shape, bits, gs)
    # q3 with mxfp-style mode probe (if the mlx build exposes it) -- informational only
    for mode in ("mxfp4",):
        try:
            out.setdefault("mode_probe", {})[f"q3g128_{mode}"] = run_cfg(
                SHAPES["gu_fused_rank"], 3, 128, mode=mode)
        except Exception as e:
            out.setdefault("mode_probe", {})[f"q3g128_{mode}"] = {"error": repr(e)}
    out["mlx_mem_semantics"] = mlx_mem_semantics()
    out["real_expert_read"] = real_expert()
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
