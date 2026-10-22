# Copyright © 2026 Adam Durham (hermes-gw)
"""Fixtures shared by the dsv41 oQ-loader tests.

Three things live here:

* :func:`write_synth_pack` -- a *tiny* native-MLX (oQ) checkpoint written in
  the published pack's exact layout (``language_model.*`` names, uint32 packed
  weights, bf16 scales/biases, per-module modes), so TP invariance can be
  tested mechanically without any download;
* :class:`InProcGroup` -- a real 2-rank ``all_sum`` rendezvous (threading
  barrier + shared slots) so a ``world=2`` build can be run against a
  ``world=1`` build in one process;
* :func:`real_pack` -- the path of a downloaded subset of the published pack,
  or ``None`` when it is not on this machine.
"""

from __future__ import annotations

import json
import os
import threading

import numpy as np
import mlx.core as mx

#: Tiny model geometry. Every dimension that a ``world=2`` slice touches is a
#: multiple of 128 so the 128-wide block geometry (and group_size 64) divides.
SYNTH = dict(
    vocab_size=512, dim=512, n_layers=2, moe_inter_dim=256,
    n_heads=4, head_dim=512, rope_head_dim=64, q_lora_rank=256,
    o_lora_rank=128, o_groups=2, window_size=128,
    n_routed_experts=8, n_activated_experts=2, hc_mult=4,
    engram_max_ngram_size=4, engram_n_heads=8, engram_head_dim=256,
)

#: Per-layer quant spec: tail -> (bits, group_size, mode). Deliberately mixed
#: so nothing in the loader may assume affine.
LAYER_QUANT = {
    "attn.wq_a": (8, 64, "affine"),
    "attn.wq_b": (4, 64, "affine"),
    "attn.wkv": (8, 64, "affine"),
    "attn.wo_b": (8, 64, "affine"),
    "ffn.shared_experts.w1": (8, 64, "affine"),
    "ffn.shared_experts.w2": (8, 64, "affine"),
    "ffn.shared_experts.w3": (8, 64, "affine"),
    "ffn.experts.w1": (2, 64, "affine"),
    "ffn.experts.w2": (3, 64, "affine"),
    "ffn.experts.w3": (2, 64, "affine"),
}


def _shapes():
    s = SYNTH
    hc_mix = (2 + s["hc_mult"]) * s["hc_mult"]
    hc_dim = s["hc_mult"] * s["dim"]
    per_group_in = s["n_heads"] * s["head_dim"] // s["o_groups"]
    return {
        "attn.wq_a": (s["q_lora_rank"], s["dim"]),
        "attn.wq_b": (s["n_heads"] * s["head_dim"], s["q_lora_rank"]),
        "attn.wkv": (s["head_dim"], s["dim"]),
        "attn.wo_a": (s["o_groups"] * s["o_lora_rank"], per_group_in),
        "attn.wo_b": (s["dim"], s["o_groups"] * s["o_lora_rank"]),
        "ffn.shared_experts.w1": (s["moe_inter_dim"], s["dim"]),
        "ffn.shared_experts.w2": (s["dim"], s["moe_inter_dim"]),
        "ffn.shared_experts.w3": (s["moe_inter_dim"], s["dim"]),
        "ffn.gate": (s["n_routed_experts"], s["dim"]),
        "hc_attn_fn": (hc_mix, hc_dim),
        "hc_ffn_fn": (hc_mix, hc_dim),
    }


def _expert_shapes():
    s = SYNTH
    return {
        "ffn.experts.w1": (s["n_routed_experts"], s["moe_inter_dim"], s["dim"]),
        "ffn.experts.w2": (s["n_routed_experts"], s["dim"], s["moe_inter_dim"]),
        "ffn.experts.w3": (s["n_routed_experts"], s["moe_inter_dim"], s["dim"]),
    }


def _vec_shapes():
    s = SYNTH
    hc_mix = (2 + s["hc_mult"]) * s["hc_mult"]
    return {
        "attn.attn_sink": ((s["n_heads"],), "F32"),
        "attn.q_norm.weight": ((s["q_lora_rank"],), "BF16"),
        "attn.kv_norm.weight": ((s["head_dim"],), "BF16"),
        "attn_norm.weight": ((s["dim"],), "BF16"),
        "ffn_norm.weight": ((s["dim"],), "BF16"),
        "ffn.gate.bias": ((s["n_routed_experts"],), "F32"),
        "ffn.gate.bias_vl": ((s["n_routed_experts"],), "F32"),
        "hc_attn_base": ((hc_mix,), "F32"),
        "hc_attn_scale": ((3,), "F32"),
        "hc_ffn_base": ((hc_mix,), "F32"),
        "hc_ffn_scale": ((3,), "F32"),
    }


def np_of(arr: mx.array) -> np.ndarray:
    """numpy view of an mlx array (this numpy cannot buffer bf16 directly).

    Integer dtypes are returned as-is (packed weights must stay exact); float
    dtypes are widened to fp32, which is lossless for bf16/fp16 comparisons.
    """
    if arr.dtype in (mx.uint32, mx.uint8, mx.int32):
        return np.array(arr)
    return np.array(arr.astype(mx.float32))


def bf16_bits(arr: mx.array) -> np.ndarray:
    """The raw 16-bit payload of a bf16/fp16 array (bit-exact comparison)."""
    return np.array(arr.view(mx.uint16))


def _np(arr: mx.array) -> tuple[np.ndarray, str]:
    """(numpy payload, safetensors dtype string) for an mlx array."""
    if arr.dtype == mx.bfloat16:
        return np.array(arr.view(mx.uint16)), "BF16"
    if arr.dtype == mx.uint32:
        return np.array(arr), "U32"
    if arr.dtype == mx.float32:
        return np.array(arr), "F32"
    raise TypeError(f"unsupported dtype {arr.dtype}")


def write_safetensors(path: str, tensors: dict[str, mx.array]) -> None:
    """Minimal safetensors writer (uint32/bf16/f32), header padded to 8 bytes."""
    header, blobs, pos = {}, [], 0
    order = sorted(tensors)
    for name in order:
        payload, dt = _np(tensors[name])
        raw = payload.tobytes()
        header[name] = {"dtype": dt, "shape": list(payload.shape),
                        "data_offsets": [pos, pos + len(raw)]}
        pos += len(raw)
        blobs.append(raw)
    hdr = json.dumps(header, separators=(",", ":")).encode()
    hdr += b" " * ((8 - len(hdr) % 8) % 8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(len(hdr).to_bytes(8, "little"))
        fh.write(hdr)
        for raw in blobs:
            fh.write(raw)


def synth_config() -> dict:
    s = SYNTH
    text = {
        "vocab_size": s["vocab_size"],
        "hidden_size": s["dim"],
        "num_hidden_layers": s["n_layers"],
        "moe_intermediate_size": s["moe_inter_dim"],
        "num_attention_heads": s["n_heads"],
        "head_dim": s["head_dim"],
        "qk_rope_head_dim": s["rope_head_dim"],
        "q_lora_rank": s["q_lora_rank"],
        "o_lora_rank": s["o_lora_rank"],
        "o_groups": s["o_groups"],
        "sliding_window": s["window_size"],
        "rms_norm_eps": 1e-20,
        "n_routed_experts": s["n_routed_experts"],
        "num_experts_per_tok": s["n_activated_experts"],
        "scoring_func": "sqrtsoftplus",
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.5,
        "swiglu_limit": 10.0,
        "compress_ratios": [],
        "kv_source_layer_ids": [],
        "index_source_layer_ids": [],
        "rope_theta": 10000.0,
        "max_position_embeddings": 4096,
        "hc_mult": s["hc_mult"],
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-6,
        "engram_layer_ids": [],
        "num_nextn_predict_layers": 0,
        "dspark_target_layer_ids": [],
    }
    quant = {}
    for layer in range(s["n_layers"]):
        for tail, (bits, group, mode) in LAYER_QUANT.items():
            quant[f"language_model.layers.{layer}.{tail}"] = {
                "bits": bits, "group_size": group, "mode": mode,
                "quantize_input": True,
            }
    return {
        "model_type": "deepseek_v41",
        "text_config": text,
        "omlx_deepseek_v41": {
            "version": 1,
            "quantized_modules": quant,
            "engram_tables": {},
            "engram_in_index": False,
            "preserve_mtp": False,
        },
    }


def _quant(w: mx.array, bits: int, group: int, mode: str, name: str):
    q = mx.quantize(w, group_size=group, bits=bits, mode=mode)
    mx.eval(*q)
    out = {f"{name}.weight": q[0], f"{name}.scales": q[1]}
    if len(q) > 2:
        out[f"{name}.biases"] = q[2]
    return out


def write_synth_pack(root: str, seed: int = 0) -> str:
    """Write a tiny oQ pack at ``root`` and return the directory."""
    s = SYNTH
    cfg = synth_config()
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=1)
    rng = mx.random
    mx.random.seed(seed)
    tensors: dict[str, mx.array] = {}
    shapes = _shapes()
    exp = _expert_shapes()
    vecs = _vec_shapes()
    for layer in range(s["n_layers"]):
        pre = f"language_model.layers.{layer}."
        for tail, (bits, group, mode) in LAYER_QUANT.items():
            if tail in exp:
                shape = exp[tail]
            else:
                shape = shapes[tail]
            w = (rng.normal(shape) * 0.02).astype(mx.bfloat16)
            tensors.update(_quant(w, bits, group, mode, pre + tail))
        tensors[pre + "attn.wo_a.weight"] = (
            rng.normal(shapes["attn.wo_a"]) * 0.02).astype(mx.bfloat16)
        tensors[pre + "ffn.gate.weight"] = (
            rng.normal(shapes["ffn.gate"]) * 0.02).astype(mx.bfloat16)
        for tail in ("hc_attn_fn", "hc_ffn_fn"):
            tensors[pre + tail] = rng.normal(shapes[tail]).astype(mx.float32)
        for tail, (shape, dt) in vecs.items():
            arr = rng.normal(shape).astype(mx.float32)
            if dt == "BF16":
                arr = arr.astype(mx.bfloat16)
            tensors[pre + tail] = arr
    tensors["language_model.embed.weight"] = (
        rng.normal((s["vocab_size"], s["dim"])) * 0.02).astype(mx.bfloat16)
    tensors["language_model.head.weight"] = (
        rng.normal((s["vocab_size"], s["dim"])) * 0.02).astype(mx.bfloat16)
    tensors["language_model.norm.weight"] = (
        rng.normal((s["dim"],))).astype(mx.float32)
    shard = "model-00001-of-00001.safetensors"
    write_safetensors(os.path.join(root, shard), tensors)
    index = {"metadata": {"format": "pt"},
             "weight_map": {k: shard for k in tensors}}
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as fh:
        json.dump(index, fh)
    return root


# --------------------------------------------------------------------------
# in-process multi-rank collective (no threads)
# --------------------------------------------------------------------------
class ReplayCollective:
    """Deterministic stand-in for ``all_sum`` over ``world`` ranks in one process.

    Two MLX graphs cannot be evaluated from two Python threads safely (MLX
    segfaults), so the ranks run *sequentially* and the collective is replayed:
    every payload each rank feeds a collective is recorded by call index, and
    the ``k``-th call returns ``local + sum(peers' k-th payloads)`` as soon as
    those payloads are known. A payload at call depth ``d`` depends only on
    collectives of depth ``< d``, so re-running the forward ``depth + 1`` times
    converges exactly -- after which the last pass is numerically identical to
    a real two-rank all_sum.
    """

    def __init__(self, rank: int, world: int, store: dict, known: dict):
        self.rank, self.world = rank, world
        self.store, self.known = store, known
        self.calls = 0
        self.complete = True
        self.pass_idx = 0

    def reset(self, pass_idx: int = 0) -> None:
        self.calls = 0
        self.complete = True
        self.pass_idx = pass_idx

    def __call__(self, x, group=None) -> mx.array:
        k = self.calls
        self.calls += 1
        mx.eval(x)
        self.store[(self.rank, k)] = (x, self.pass_idx)
        total = x
        for r in range(self.world):
            if r == self.rank:
                continue
            peer = self.known.get((r, k))
            if peer is None:
                self.complete = False
            else:
                total = total + peer
        return total


def install_replay() -> None:
    """Point the model's ``all_sum`` at the replay group it was handed.

    ``build_block(..., group=g)`` passes ``g`` into every collective, so the
    stub only has to route the call back into that object -- no thread-local
    state, no global rank.
    """
    from mlx_lm.models.deepseek_v41 import collective

    def all_sum(x, group=None):
        if isinstance(group, ReplayCollective):
            return group(x, group)
        return x

    collective.all_sum = all_sum


def run_replayed(fn, world: int, max_passes: int = 12):
    """Run ``fn(rank)`` for every rank until the collective store is complete.

    Returns ``(outputs, passes)``. ``fn(rank)`` must run one rank's forward
    against the group it built that rank with.
    """
    store: dict = {}
    known: dict = {}
    groups = [ReplayCollective(r, world, store, known) for r in range(world)]
    install_replay()
    outs = None
    for passes in range(1, max_passes + 1):
        for g in groups:
            g.reset(passes - 1)
        outs = [fn(r, groups[r]) for r in range(world)]
        # Trust only payloads whose upstream collectives were already correct:
        # pass p is the first in which depth p-1 is exact, so accept k < p-1
        # *and* k == p-1 from this pass. Keeping a stale deep payload would
        # silently converge to the wrong answer.
        known.update({key: val for key, (val, p) in store.items()
                      if key[1] <= passes - 1})
        if all(g.complete for g in groups):
            return outs, passes
    raise AssertionError(
        f"replayed collectives did not converge in {max_passes} passes")


# --------------------------------------------------------------------------
# the published pack, when it has been fetched
# --------------------------------------------------------------------------
#: Environment override for the downloaded subset of the published pack.
PACK_ENV = "EXO_DSV41_OQ_PACK"
DEFAULT_PACK = os.path.expanduser(
    "~/.hermes/cache/scratch/oqmini")


def real_pack() -> str | None:
    """The published pack's local subset, or None when it is not present."""
    path = os.environ.get(PACK_ENV) or DEFAULT_PACK
    idx = os.path.join(path, "model.safetensors.index.json")
    return path if os.path.exists(idx) else None
