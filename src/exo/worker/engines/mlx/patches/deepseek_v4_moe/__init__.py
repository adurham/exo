"""DSv4 MoE: promote routed-expert SwitchGLU → BatchedSwitchGLU (fuse gate+up).

Mirrors the Qwen3.5 MoE patch (qwen3_5_moe/common.py:_patch_switch_mlp) but
minimal: DSv4 doesn't use the Qwen custom fused Metal kernels, so we only
do the SwitchGLU → BatchedSwitchGLU class swap + fuse_weights() — which
cuts the routed-expert path from 3 gather_qmm dispatches (gate, up, down)
to 2 (fused gate+up, down) using MLX's built-in gather_qmm.

Bit-equivalent: same math, fused dispatch. No quality loss.

DSv4's routed experts are mxfp4-quantized (group_size=32) at load, which
satisfies BatchedSwitchGLU's requirement (quantized SwitchLinear with
matching group_size/bits/mode on gate_proj + up_proj).
"""

from __future__ import annotations

import mlx.core as mx
from loguru import logger


def _patch_switch_mlp(moe) -> bool:
    """Promote ``moe.switch_mlp`` to BatchedSwitchGLU and fuse gate+up.

    Returns True if the promotion ran, False if skipped (already promoted,
    not a SwitchGLU, or gate/up not quantized / mismatched).
    """
    from mlx_lm.models.switch_layers import BatchedSwitchGLU, SwitchGLU

    swl = getattr(moe, "switch_mlp", None)
    if swl is None:
        return False
    if isinstance(swl, BatchedSwitchGLU):
        return False  # already promoted
    if not isinstance(swl, SwitchGLU):
        return False  # not a SwitchGLU (unexpected)
    gp = getattr(swl, "gate_proj", None)
    up = getattr(swl, "up_proj", None)
    if gp is None or up is None:
        return False
    # fuse_weights requires quantized projections with matching
    # group_size/bits/mode. Skip cleanly if not quantized (vanilla path).
    for attr in ("scales", "biases", "group_size", "bits", "mode"):
        if not hasattr(gp, attr) or not hasattr(up, attr):
            return False
    if gp.group_size != up.group_size or gp.bits != up.bits or gp.mode != up.mode:
        return False
    # In-place class swap: SwitchGLU and BatchedSwitchGLU share __init__
    # signature and parameters, so reassigning __class__ is safe and avoids
    # reallocating the projection weights.
    swl.__class__ = BatchedSwitchGLU
    swl.fuse_weights()
    return True


def apply_dsv4_moe_patches(model) -> None:
    """Promote every DeepseekV4MoE switch_mlp to BatchedSwitchGLU.

    Walks both the main decoder blocks (layers[i].ffn.switch_mlp) and the
    MTP module's body blocks (mtp.layers[j].ffn.switch_mlp) so the fused
    path is used consistently in decode and in the MTP draft/verify path.

    Robust to the model being either the full wrapper (has
    ``.language_model.model.layers``) or the inner model (has ``.layers``)
    — the distributed shard_and_load path returns the inner model after
    tensor_auto_parallel, while the single-device load path passes the
    full wrapper.
    """
    layers = None
    lm = getattr(model, "language_model", None)
    if lm is not None:
        body = getattr(lm, "model", None)
        if body is not None:
            layers = getattr(body, "layers", None)
    if layers is None:
        # Inner-model form (post-shard): model.layers directly.
        layers = getattr(model, "layers", None)
    if layers is None:
        logger.warning("DSv4 MoE patch: no .layers found; skipping")
        return

    promoted = 0
    skipped = 0
    for layer in layers:
        ffn = getattr(layer, "ffn", None)
        if ffn is None:
            continue
        if _patch_switch_mlp(ffn):
            promoted += 1
        else:
            skipped += 1

    # MTP module body blocks (same DeepseekV4MoE structure). The MTP
    # module hangs off the inner model (body.mtp) or the language model.
    mtp = None
    if lm is not None:
        mtp = getattr(lm, "mtp", None) or getattr(getattr(lm, "model", None), "mtp", None)
    if mtp is None:
        mtp = getattr(model, "mtp", None)
    if mtp is not None:
        mtp_layers = getattr(mtp, "layers", None)
        if mtp_layers is not None:
            for layer in mtp_layers:
                ffn = getattr(layer, "ffn", None)
                if ffn is None:
                    continue
                if _patch_switch_mlp(ffn):
                    promoted += 1
                else:
                    skipped += 1

    logger.info(
        f"DSv4 MoE patch: promoted {promoted} switch_mlp to BatchedSwitchGLU "
        f"(fused gate+up), skipped {skipped}"
    )