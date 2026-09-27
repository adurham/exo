#!/usr/bin/env python3
"""Phase 1C: localize the full-layer fp16 delta, and tick the plan's named API.

Two pieces of evidence for the phase-1 report:

  1. Root-cause localization of the only non-bit-exact step in the
     reconstruction path. Phase 1 showed inner decode (trellis codec) is
     bit-exact on every class while reconstruct_public_weights (inner +
     Hadamard + sign folding) differs by <= 4.9e-4 = exactly 1 fp16 ULP at
     the affected magnitudes. This script isolates WHERE that comes from:
       - signs unpack (ref vs mlx): expect bit-exact
       - left/right Hadamard (ref vs mlx) on identical fp32 input: fp32-level
         summation-order difference expected
       - pre-cast fp32 full reconstruction vs post-cast fp16: how many
         elements cross a rounding boundary, and by how many ULPs
  2. The plan's literal API path: "Load one routed-expert tensor group from a
     dealignai shard with ponyexl3.mlx.weights.load_safetensors, build an
     EXL3Layer". Runs that end to end on layers.3.ffn.experts.0.w1 and
     compares tensor bytes against safe_open.

Run inside ~/phase1-exl3/.venv on macstudio-m4-1.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

import numpy as np

MODEL_DIR = os.path.expanduser(
    "~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
)
OUT: dict = {}


def jd(o):
    """json-safe floats (drop numpy scalar types)."""
    if isinstance(o, dict):
        return {k: jd(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jd(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float):
        return float(o)
    return o


def part_load_safetensors() -> None:
    import mlx.core as mx
    from safetensors import safe_open

    from ponyexl3.mlx.weights import load_safetensors
    from ponyexl3.ref.layer import EXL3Layer
    from ponyexl3.ref.decode import decode_packed_trellis
    from ponyexl3.mlx.reconstruct import reconstruct_inner_mlx

    idx = json.load(
        open(os.path.join(MODEL_DIR, "model.safetensors.index.json"), encoding="utf-8")
    )["weight_map"]
    module = "layers.3.ffn.experts.0.w1"
    shard = idx[module + ".trellis"]
    shard_path = os.path.join(MODEL_DIR, shard)

    print(f"--- load_safetensors on {shard} (module {module})", flush=True)
    d = load_safetensors(shard_path)
    print(f"    tensors loaded: {len(d)}", flush=True)

    with open(shard_path, "rb") as f:
        import struct

        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr_keys = [k for k in hdr if k != "__metadata__"]
    names_match = sorted(hdr_keys) == sorted(d.keys())

    tk = module + ".trellis"
    got = np.array(d[tk])
    with safe_open(shard_path, framework="np") as st:
        ref = np.array(st.get_tensor(tk))
    trellis_equal = got.shape == ref.shape and bool(np.array_equal(got, ref))

    trellis = got.astype(np.uint16)
    suh = np.array(d[module + ".suh"])
    svh = np.array(d[module + ".svh"])
    mul1 = bool(int(np.array(d[module + ".mul1"])))
    in_tiles, out_tiles, packed_size = trellis.shape
    k = packed_size * 16 // 256
    layer = EXL3Layer(
        key=module,
        in_features=in_tiles * 16,
        out_features=out_tiles * 16,
        k=k,
        trellis=trellis,
        suh=suh,
        svh=svh,
        mcg=False,
        mul1=mul1,
    )
    layer.validate()
    ref_i = np.asarray(decode_packed_trellis(trellis, k, layer.codebook_mode), dtype=np.float32)
    got_i = np.asarray(
        np.array(reconstruct_inner_mlx(mx.array(trellis), k, mcg=False, mul1=mul1)),
        dtype=np.float32,
    )
    inner_equal = bool(ref_i.shape == got_i.shape and np.array_equal(ref_i, got_i))

    OUT["load_safetensors"] = {
        "shard": shard,
        "tensors_loaded": len(d),
        "header_names_match": bool(names_match),
        "trellis_bytes_equal_vs_safe_open": bool(trellis_equal),
        "rebuilt_layer_k": k,
        "inner_decode_bit_exact": inner_equal,
    }
    print(
        f"    header_names_match={names_match} trellis_bytes_equal={trellis_equal} "
        f"inner_decode_bit_exact={inner_equal}",
        flush=True,
    )


def part_localize() -> None:
    import mlx.core as mx

    from ponyexl3.ref.loader import load_exl3_layer
    from ponyexl3.ref.decode import decode_packed_trellis
    from ponyexl3.ref.hadamard import preapply_had_left, preapply_had_right
    from ponyexl3.mlx.hadamard import preapply_had_left_mlx, preapply_had_right_mlx
    from ponyexl3.ref.signs import unpack_signs_or_pass
    from ponyexl3.mlx.signs import unpack_signs_or_pass_mlx
    from ponyexl3.ref.reconstruct import reconstruct_public_weights
    from ponyexl3.mlx.reconstruct import reconstruct_public_mlx

    for module in ["layers.0.attn.wq_a", "layers.11.ffn.shared_experts.w1"]:
        print(f"\n--- localizing {module}", flush=True)
        layer = load_exl3_layer(MODEL_DIR, module)
        layer.validate()
        rec: dict = {}

        # (a) signs unpack: ref vs mlx
        s_ref = unpack_signs_or_pass(layer.suh)
        s_mlx = np.array(unpack_signs_or_pass_mlx(mx.array(layer.suh)))
        rec["signs_unpack_bit_exact"] = bool(
            s_ref is not None and np.array_equal(np.asarray(s_ref), s_mlx)
        )
        rec["signs_size"] = int(np.asarray(s_ref).size) if s_ref is not None else 0

        # (b) Hadamard isolation on identical fp32 input
        w32 = decode_packed_trellis(layer.trellis, layer.k, layer.codebook_mode).astype(np.float32)
        scale = float(np.abs(w32).mean())
        hl_ref = preapply_had_left(w32)
        hl_mlx = np.array(preapply_had_left_mlx(mx.array(w32)))
        dl = np.abs(hl_ref - hl_mlx)
        rec["hadamard_left"] = {
            "max_abs_diff": float(dl.max()),
            "n_differing": int((dl > 0).sum()),
            "size": int(dl.size),
            "max_rel_to_input_mean": float(dl.max() / scale),
        }
        if layer.svh is not None:
            hr_ref = preapply_had_right(hl_ref)
            hr_mlx = np.array(preapply_had_right_mlx(mx.array(hl_ref)))
            dr = np.abs(hr_ref - hr_mlx)
            rec["hadamard_right"] = {
                "max_abs_diff": float(dr.max()),
                "n_differing": int((dr > 0).sum()),
                "size": int(dr.size),
                "max_rel_to_input_mean": float(dr.max() / scale),
            }

        # (c) pre-cast fp32 vs post-cast fp16 on the full public path
        #     library functions first (the numbers phase 1 quoted)
        ref16 = reconstruct_public_weights(
            layer.trellis, layer.suh, layer.svh, layer.k, mcg=layer.mcg, mul1=layer.mul1
        )
        mlx16 = np.array(reconstruct_public_mlx(layer))
        n_diff = int((ref16 != mlx16).sum())
        d16 = np.abs(ref16.astype(np.float32) - mlx16.astype(np.float32))
        ulp = np.spacing(np.abs(ref16))
        mask = ref16 != mlx16
        ratios = (d16[mask] / ulp[mask]) if mask.any() else np.array([0.0])
        rec["public_path"] = {
            "size": int(ref16.size),
            "n_differing_fp16": n_diff,
            "frac_differing": float(n_diff / ref16.size),
            "max_abs_diff_fp16": float(d16.max()),
            "max_diff_in_fp16_ulp": float(np.max(ratios)),
            "all_diffs_within_1_ulp": bool(np.max(ratios) <= 1.0 + 1e-9) if mask.any() else True,
        }

        # inline fp32 replicas, same order as the library, to see the
        # pre-cast fp32 delta that the fp16 cast then surfaces
        suh32 = np.asarray(unpack_signs_or_pass(layer.suh), dtype=np.float32)
        svh32 = np.asarray(unpack_signs_or_pass(layer.svh), dtype=np.float32)
        a = preapply_had_left(w32)
        a = a * suh32.reshape(-1, 1)
        a = preapply_had_right(a)
        a = a * svh32.reshape(1, -1)
        b = mx.array(w32)
        b = preapply_had_left_mlx(b)
        b = b * mx.array(suh32).reshape(-1, 1)
        b = preapply_had_right_mlx(b)
        b = b * mx.array(svh32).reshape(1, -1)
        b32 = np.array(b)
        d32 = np.abs(a - b32)
        rec["pre_cast_fp32"] = {
            "max_abs_diff": float(d32.max()),
            "n_differing": int((d32 > 0).sum()),
            "size": int(d32.size),
            "max_rel_to_output_mean": float(d32.max() / np.abs(a).mean()),
        }
        # how many of the fp32 differences actually cross an fp16 boundary?
        rec["fp32_to_fp16_crossing"] = {
            "fp32_differ_but_fp16_equal": int(((d32 > 0) & (a.astype(np.float16) == b32.astype(np.float16))).sum()),
            "fp32_differ_and_fp16_differ": int(((d32 > 0) & (a.astype(np.float16) != b32.astype(np.float16))).sum()),
        }

        OUT[module] = rec
        print(json.dumps(jd(rec), indent=1), flush=True)


def main() -> int:
    print(f"model: {MODEL_DIR}\n", flush=True)
    try:
        part_load_safetensors()
    except Exception as exc:  # noqa: BLE001
        OUT["load_safetensors_error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    try:
        part_localize()
    except Exception as exc:  # noqa: BLE001
        OUT["localize_error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    path = os.path.expanduser("~/phase1c-results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jd(OUT), f, indent=1)
    print(f"\nwrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
