# Copyright © 2026 Adam Durham (hermes-gw)
"""PackedEngramTable: 4-bit affine engram rows read out of the oQ pack itself.

The abliterated 1017 oQ pack carries its engram tables quantized (U32 packed
rows + BF16 scales & biases, group 32) instead of the native release's E4M3 +
E8M0 layout, and declares that in ``omlx_deepseek_v41.engram_tables``. This
checks the reader against ``mx.quantize``/``mx.dequantize`` on the same bytes,
including the unique-row scatter the forward path uses.
"""
from __future__ import annotations

import json
import os
import sys

import mlx.core as mx
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(ROOT, "mlx-lm"), os.path.join(ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mlx_lm.models.deepseek_v41 import oq_build  # noqa: E402

N_ROWS, DIM, BITS, GROUP = 64, 256, 4, 32


def _make_pack(root: str):
    os.makedirs(root, exist_ok=True)
    w = mx.random.normal((N_ROWS, DIM)).astype(mx.bfloat16)
    wq, scales, biases = mx.quantize(w, group_size=GROUP, bits=BITS)
    mx.eval(wq, scales, biases)
    mod = "language_model.layers.1.engram.embed"
    shard = "model-00001-of-00001.safetensors"
    mx.save_safetensors(
        os.path.join(root, shard),
        {f"{mod}.weight": wq, f"{mod}.scales": scales, f"{mod}.biases": biases},
    )
    cfg = {
        "model_type": "deepseek_v41",
        "text_config": {"engram_layer_ids": [1], "num_hidden_layers": 1,
                        "hidden_size": DIM},
        "omlx_deepseek_v41": {
            "version": 1,
            "quantized_modules": {},
            "engram_in_index": True,
            "engram_tables": {
                mod: {"weight_file": shard, "weight_key": f"{mod}.weight",
                      "scale_file": shard, "scale_key": f"{mod}.scales",
                      "bias_key": f"{mod}.biases", "bits": BITS,
                      "group_size": GROUP, "mode": "affine"},
            },
        },
    }
    with open(os.path.join(root, "config.json"), "w") as fh:
        json.dump(cfg, fh)
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as fh:
        json.dump({"weight_map": {f"{mod}.weight": shard,
                                  f"{mod}.scales": shard,
                                  f"{mod}.biases": shard}}, fh)
    ref = mx.dequantize(wq, scales, biases, group_size=GROUP, bits=BITS,
                        mode="affine")
    mx.eval(ref)
    return mod, np.asarray(ref.astype(mx.float32))


def test_packed_engram_rows_match_dequantize(tmp_path):
    root = str(tmp_path / "pack")
    mod, ref = _make_pack(root)
    ck = oq_build.OqCheckpoint(root)
    spec = ck.engram_tables[mod]
    tab = oq_build.PackedEngramTable(ck, 1, spec)

    picks = np.array([0, 7, 7, 63], dtype=np.int64)          # duplicates + ends
    got = np.asarray(tab(picks).astype(mx.float32))
    assert got.shape == (4, DIM)
    for i, r in enumerate(picks):
        assert np.allclose(got[i], ref[r], atol=1e-3), f"row {r} mismatch"


def test_packed_engram_prefetch_uses_same_rows(tmp_path):
    root = str(tmp_path / "pack2")
    mod, ref = _make_pack(root)
    ck = oq_build.OqCheckpoint(root)
    tab = oq_build.PackedEngramTable(ck, 1, ck.engram_tables[mod])

    idx = np.array([[5, 1, 5]], dtype=np.int64)
    tab.prefetch(idx)
    got = np.asarray(tab(idx).astype(mx.float32))
    assert got.shape == (1, 3, DIM)
    for j, r in enumerate(idx[0]):
        assert np.allclose(got[0, j], ref[r], atol=1e-3)


def test_dim_inferred_from_packed_width(tmp_path):
    root = str(tmp_path / "pack3")
    mod, _ = _make_pack(root)
    ck = oq_build.OqCheckpoint(root)
    tab = oq_build.PackedEngramTable(ck, 1, ck.engram_tables[mod])
    assert tab._dim == DIM                                        # noqa: SLF001
