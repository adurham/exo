#!/usr/bin/env python3
"""Phase 1 gate: EXL3 format validation for the dealignai DSv4.1-Flash 2.9bpw repo.

The plan (docs/deepseek-v41-exl3-plan.md, phase 1) answers ONE question before
any porting: can PonyExl3 read this model's EXL3 containers at all, and does its
implementation agree bit-exactly with its own CPU reference on this data?

This script does that end-to-end on real model tensors, not synthetic ones:

  1. format acceptance   -- quantization_config.json version 1.4.2 parses, and
                            each tensor group resolves to an EXL3Layer whose
                            trellis shape matches its declared tile geometry
  2. mlx-vs-ref agreement -- decode the same real trellis with the MLX (Metal)
                            kernel and the numpy CPU reference; compare bit-exact
  3. full-layer agreement -- reconstruct_inner / reconstruct_public_weights
                            (inner + su/sv outer) on both paths, compare
  4. coverage             -- run the four distinct quant classes in this model:
                            3-bit routed experts, 5-bit attention projections,
                            6-bit head, 4-bit MTP/shared-expert groups

Pass condition (the plan's own words): bit-exact is the pass condition.

Run inside ~/phase1-exl3/.venv on macstudio-m4-1.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import dataclass, field

import numpy as np

MODEL_DIR = os.path.expanduser(
    "~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
)

RESULTS: list[dict] = []


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


# --------------------------------------------------------------------------
# 1. format acceptance
# --------------------------------------------------------------------------
def check_format() -> dict:
    qpath = os.path.join(MODEL_DIR, "quantization_config.json")
    with open(qpath, encoding="utf-8") as f:
        qcfg = json.load(f)

    checks = {
        "quant_method": qcfg.get("quant_method"),
        "version": qcfg.get("version"),
        "bits": qcfg.get("bits"),
        "head_bits": qcfg.get("head_bits"),
        "mtp_bits": qcfg.get("mtp_bits"),
        "codebook": qcfg.get("codebook"),
    }
    print("quantization_config:", json.dumps(checks), flush=True)

    ok = (
        checks["quant_method"] == "exl3"
        and checks["version"] == "1.4.2"
        and checks["codebook"] == "mul1"
    )
    record(
        "format/quantization_config v1.4.2 accepted",
        ok,
        f"quant_method={checks['quant_method']} version={checks['version']} "
        f"codebook={checks['codebook']} bits={checks['bits']} "
        f"head_bits={checks['head_bits']} mtp_bits={checks['mtp_bits']}",
    )
    return qcfg


# --------------------------------------------------------------------------
# helpers to pick representative real groups
# --------------------------------------------------------------------------
def pick_groups() -> list[tuple[str, str]]:
    """(label, module_key) for one real module per quant class."""
    out: list[tuple[str, str]] = []

    # 3-bit routed expert (the bulk of the model)
    out.append(("routed-expert-3bit", "layers.3.ffn.experts.0.w1"))
    # 5-bit attention projection
    out.append(("attention-5bit", "layers.0.attn.wq_b"))
    # 6-bit attention / head
    out.append(("attention-6bit", "layers.0.attn.wq_a"))
    out.append(("head-6bit", "head"))
    # 4-bit shared expert / MTP
    out.append(("shared-expert-4bit", "layers.11.ffn.shared_experts.w1"))
    return out


# --------------------------------------------------------------------------
# 2 + 3. decode agreement, inner and full-layer
# --------------------------------------------------------------------------
def check_group(label: str, module_key: str) -> None:
    import mlx.core as mx

    from ponyexl3.ref.loader import load_exl3_layer, layer_meta_from_config
    from ponyexl3.ref.decode import decode_packed_trellis
    from ponyexl3.ref.codebook import codebook_mode_from_flags
    from ponyexl3.ref.reconstruct import reconstruct_public_weights
    from ponyexl3.mlx.decode import decode_packed_trellis_mlx
    from ponyexl3.mlx.reconstruct import reconstruct_inner_mlx

    # ---- metadata
    meta = layer_meta_from_config(MODEL_DIR, module_key)
    # ---- real tensors from the model
    layer = load_exl3_layer(MODEL_DIR, module_key)
    layer.validate()  # shape-vs-declared-geometry check

    k = int(meta["k"])
    cb = int(codebook_mode_from_flags(mcg=layer.mcg, mul1=layer.mul1))

    print(
        f"\n--- {label}: {module_key}\n"
        f"    in={meta['in_features']} out={meta['out_features']} k={k} "
        f"bits={meta['bits_per_weight']} mul1={layer.mul1} mcg={layer.mcg} "
        f"trellis={layer.trellis.shape}",
        flush=True,
    )

    # ---- inner decode: numpy ref vs mlx metal
    ref = np.asarray(
        decode_packed_trellis(layer.trellis, k, cb), dtype=np.float32
    )
    got = np.asarray(
        np.array(reconstruct_inner_mlx(layer.trellis, k, mcg=layer.mcg, mul1=layer.mul1)),
        dtype=np.float32,
    )
    inner_ok = ref.shape == got.shape and np.array_equal(ref, got)
    record(
        f"{label}/inner_decode ref-vs-mlx",
        inner_ok,
        f"shapes ref{ref.shape} mlx{got.shape} "
        f"equal={np.array_equal(ref, got) if ref.shape == got.shape else 'n/a'} "
        f"max_abs_diff={float(np.max(np.abs(ref - got))) if ref.shape == got.shape else float('nan')}",
    )

    # ---- full layer (inner + hadamard/signs) vs the mlx full-layer path
    ref_w = np.asarray(
        reconstruct_public_weights(layer.trellis, layer.suh, layer.svh, k, mcg=layer.mcg, mul1=layer.mul1),
        dtype=np.float16,
    )
    from ponyexl3.mlx.reconstruct import reconstruct_public_mlx

    got_w = np.asarray(np.array(reconstruct_public_mlx(layer)), dtype=np.float16)
    full_ok = ref_w.shape == got_w.shape and np.array_equal(ref_w, got_w)
    record(
        f"{label}/full_layer ref-vs-mlx",
        full_ok,
        f"shapes ref{ref_w.shape} mlx{got_w.shape} "
        f"equal={np.array_equal(ref_w, got_w) if ref_w.shape == got_w.shape else 'n/a'} "
        f"max_abs_diff={float(np.max(np.abs(ref_w.astype(np.float32) - got_w.astype(np.float32)))) if ref_w.shape == got_w.shape else float('nan')}",
    )


def main() -> int:
    print(f"model: {MODEL_DIR}", flush=True)
    qcfg = check_format()

    groups = pick_groups()
    print("\ngroups under test:", flush=True)
    for label, key in groups:
        print(f"  {label:22s} {key}", flush=True)

    for label, key in groups:
        try:
            check_group(label, key)
        except Exception as exc:  # noqa: BLE001
            record(f"{label}/ERROR", False, f"{type(exc).__name__}: {exc}")
            traceback.print_exc()

    # ---- verdict
    total = len(RESULTS)
    passed = sum(1 for r in RESULTS if r["ok"])
    failed = total - passed
    print("\n" + "=" * 72)
    print(f"PHASE 1 VERDICT: {passed}/{total} checks passed, {failed} failed")
    bad = [r for r in RESULTS if not r["ok"]]
    for r in bad:
        print(f"  FAILED: {r['check']} -- {r['detail']}")
    print("=" * 72)
    print(json.dumps({"checks": RESULTS, "passed": passed, "total": total}, indent=1)[:200])
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
