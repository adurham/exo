import json
import os
import re
import sys
import tempfile
import time
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from exo.worker.engines.mlx.vision import VisionProcessor

# Monkey-patch for transformers 5.x compatibility
# Kimi's tokenization_kimi.py imports bytes_to_unicode from the old location
# which was moved in transformers 5.0.0rc2
try:
    import transformers.models.gpt2.tokenization_gpt2 as gpt2_tokenization
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    if not hasattr(gpt2_tokenization, "bytes_to_unicode"):
        gpt2_tokenization.bytes_to_unicode = bytes_to_unicode  # type: ignore[attr-defined]
except ImportError:
    pass  # transformers < 5.0 or bytes_to_unicode not available

from mlx_lm.models.cache import KVCache
from mlx_lm.models.deepseek_v3 import DeepseekV3Model
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.models.model_cards import ModelId
from exo.worker.engines.mlx.constants import TRUST_REMOTE_CODE

try:
    from mlx_lm.tokenizer_utils import load_tokenizer
except ImportError:
    from mlx_lm.tokenizer_utils import load as load_tokenizer
import contextlib

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load_model
from pydantic import RootModel

from exo.download.download_utils import build_model_path
from exo.shared.types.common import Host
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import TaskId, TextGeneration
from exo.shared.types.text_generation import ChatTemplateValue, TextGenerationTaskParams
from exo.shared.types.worker.instances import (
    BoundInstance,
    MlxJacclInstance,
    MlxRingInstance,
)
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.shared.types.worker.shards import (
    CfgShardMetadata,
    PipelineShardMetadata,
    ShardMetadata,
    TensorShardMetadata,
)
from exo.worker.engines.mlx.auto_parallel import (
    get_inner_model,
    get_layers,
    pipeline_auto_parallel,
    tensor_auto_parallel,
)
from exo.worker.engines.mlx.types import Model
from exo.worker.runner.bootstrap import logger


def get_weights_size(model_shard_meta: ShardMetadata) -> Memory:
    return Memory.from_float_kb(
        (model_shard_meta.end_layer - model_shard_meta.start_layer)
        / model_shard_meta.n_layers
        * model_shard_meta.model_card.storage_size.in_kb
        / (
            1
            if isinstance(model_shard_meta, PipelineShardMetadata)
            else model_shard_meta.world_size
        )
    )


class HostList(RootModel[list[str]]):
    @classmethod
    def from_hosts(cls, hosts: list[Host]) -> "HostList":
        return cls(root=[str(host) for host in hosts])


# jaccl init retry-with-backoff. A runner SIGKILLed mid-RDMA (the c>=2 peer
# wedge path) leaks its QPs; the Thunderbolt RDMA driver reaps them
# ASYNCHRONOUSLY, so the very next runner's `queue_pair_rtr` hits errno 16
# (EBUSY) and crashes. Without backoff the master re-places instantly with a
# FRESH instance_id (placement.py mints a new one each time), resetting the
# per-instance retry budget — so the loop hammers <1s apart forever, keeping the
# device saturated and never letting the leaked QPs drain (instance stuck
# PREPARING until a full redeploy). Retrying init IN-PROCESS with backoff fixes
# it at the source: a failed init RAII-destroys its own QPs (Connection::
# ~Connection -> ibv_destroy_qp), so retries don't accumulate, and the wait lets
# the prior runner's leaked QPs get reaped -> a later attempt succeeds, usually
# WITHOUT the runner ever crashing (no re-place cycle at all). Total worst-case
# wait stays under the supervisor's 180s bring-up watchdog.
_JACCL_INIT_MAX_ATTEMPTS = int(os.environ.get("EXO_JACCL_INIT_MAX_ATTEMPTS", "6"))
_JACCL_INIT_BACKOFF_BASE = float(os.environ.get("EXO_JACCL_INIT_BACKOFF_BASE", "3.0"))
_JACCL_INIT_BACKOFF_CAP = float(os.environ.get("EXO_JACCL_INIT_BACKOFF_CAP", "30.0"))


def _init_jaccl_with_backoff(rank: int) -> mx.distributed.Group:
    """mx.distributed.init(jaccl) with backoff over transient RDMA-busy init
    failures (leaked-QP EBUSY from a prior runner's ungraceful teardown)."""
    for attempt in range(_JACCL_INIT_MAX_ATTEMPTS):
        try:
            group = mx.distributed.init(backend="jaccl", strict=True)
            if attempt > 0:
                logger.info(
                    f"rank {rank} jaccl init succeeded on attempt {attempt + 1}"
                )
            return group
        except Exception as e:
            if attempt + 1 >= _JACCL_INIT_MAX_ATTEMPTS:
                raise
            delay = min(
                _JACCL_INIT_BACKOFF_CAP, _JACCL_INIT_BACKOFF_BASE * (2.0**attempt)
            )
            logger.warning(
                f"rank {rank} jaccl init attempt {attempt + 1}/"
                f"{_JACCL_INIT_MAX_ATTEMPTS} failed: {e}. Likely leaked-QP EBUSY "
                f"from a prior runner's RDMA teardown; backing off {delay:.0f}s to "
                f"let the device drain, then retrying."
            )
            time.sleep(delay)
    # Unreachable: the final attempt either returns or raises above.
    raise RuntimeError("jaccl init retry loop exited unexpectedly")


def mlx_distributed_init(
    bound_instance: BoundInstance,
) -> mx.distributed.Group:
    """
    Initialize MLX distributed.
    """
    rank = bound_instance.bound_shard.device_rank
    logger.info(f"Starting initialization for rank {rank}")

    with tempfile.TemporaryDirectory() as tmpdir:
        coordination_file = str(
            Path(tmpdir) / f"hosts_{bound_instance.instance.instance_id}_{rank}.json"
        )
        # TODO: singleton instances
        match bound_instance.instance:
            case MlxRingInstance(hosts_by_node=hosts_by_node, ephemeral_port=_):
                hosts_for_node = hosts_by_node[bound_instance.bound_node_id]
                hosts_json = HostList.from_hosts(hosts_for_node).model_dump_json()

                with open(coordination_file, "w") as f:
                    _ = f.write(hosts_json)

                logger.info(
                    f"rank {rank} hostfile: {coordination_file} hosts: {hosts_json}"
                )

                os.environ["MLX_HOSTFILE"] = coordination_file
                os.environ["MLX_RANK"] = str(rank)
                # os.environ["MLX_RING_VERBOSE"] = "1"  # NOTE: we don't use it enough to care (turn on again if need to)

                group = mx.distributed.init(backend="ring", strict=True)

            case MlxJacclInstance(
                jaccl_devices=jaccl_devices, jaccl_coordinators=jaccl_coordinators
            ):
                assert all(
                    jaccl_devices[i][i] is None for i in range(len(jaccl_devices))
                )
                # Use RDMA connectivity matrix
                jaccl_devices_json = json.dumps(jaccl_devices)

                with open(coordination_file, "w") as f:
                    _ = f.write(jaccl_devices_json)

                jaccl_coordinator = jaccl_coordinators[bound_instance.bound_node_id]

                logger.info(
                    f"rank {rank} MLX_IBV_DEVICES: {coordination_file} with devices: {jaccl_devices_json}"
                )
                logger.info(f"rank {rank} MLX_JACCL_COORDINATOR: {jaccl_coordinator}")
                os.environ["MLX_IBV_DEVICES"] = coordination_file
                os.environ["MLX_RANK"] = str(rank)
                os.environ["MLX_JACCL_COORDINATOR"] = jaccl_coordinator
                group = _init_jaccl_with_backoff(rank)

        logger.info(f"Rank {rank} mlx distributed initialization complete")

        return group


def initialize_mlx(
    bound_instance: BoundInstance,
) -> mx.distributed.Group:
    # should we unseed it?
    # TODO: pass in seed from params
    mx.random.seed(42)

    assert len(bound_instance.instance.shard_assignments.node_to_runner) > 1, (
        "Tried to initialize mlx for a single node instance"
    )
    return mlx_distributed_init(bound_instance)


def load_mlx_items(
    bound_instance: BoundInstance,
    group: mx.distributed.Group | None,
) -> Generator[
    ModelLoadingResponse, None, tuple[Model, TokenizerWrapper, "VisionProcessor | None"]
]:
    set_wired_limit_for_model(get_weights_size(bound_instance.bound_shard))

    if group is None:
        logger.info(f"Single device used for {bound_instance.instance}")
        model_path = build_model_path(bound_instance.bound_shard.model_card.model_id)
        start_time = time.perf_counter()
        model, _ = load_model(model_path, lazy=True, strict=False)
        # Eval layers one by one for progress reporting
        try:
            inner = get_inner_model(model)
            layers = get_layers(inner)
            total = len(layers)
            for i, layer in enumerate(layers):
                mx.eval(layer)  # type: ignore
                yield ModelLoadingResponse(layers_loaded=i, total=total)
        except ValueError as e:
            logger.opt(exception=e).debug(
                "Model architecture doesn't support layer-by-layer progress tracking",
            )
        mx.eval(model)
        end_time = time.perf_counter()
        logger.info(f"Time taken to load model: {(end_time - start_time):.2f}s")
        from exo.worker.engines.mlx.patches import maybe_apply_patches

        maybe_apply_patches(model, model_path)
        tokenizer = get_tokenizer(model_path, bound_instance.bound_shard)

    else:
        logger.info("Starting distributed init")
        start_time = time.perf_counter()
        model, tokenizer = yield from shard_and_load(
            bound_instance.bound_shard,
            group=group,
        )
        end_time = time.perf_counter()
        logger.info(
            f"Time taken to shard and load model: {(end_time - start_time):.2f}s"
        )

        # --- Non-rank-0 nodes: pull MTP q4 from rank 0 after barrier ---
        shard_meta = bound_instance.bound_shard
        _is_pp_r0 = (
            isinstance(shard_meta, PipelineShardMetadata)
            and shard_meta.device_rank == 0
        )
        if not _is_pp_r0 and os.environ.get("EXO_SPECULATIVE", "0") == "1":
            try:
                import hashlib
                from .generator.batch_generate import ExoBatchGenerator

                _resolver = ExoBatchGenerator.__new__(ExoBatchGenerator)
                _resolver.model = model
                _resolver.model_id = shard_meta.model_card.model_id
                _mtp_repo = _resolver._detect_mtp_repo()
                if _mtp_repo:
                    _cache_dir = Path.home() / ".cache" / "exo" / "mtp_weights"
                    _cache_dir.mkdir(parents=True, exist_ok=True)
                    _cache_key = hashlib.md5(_mtp_repo.encode()).hexdigest()[:12]
                    _q4_path = _cache_dir / f"mtp_{_cache_key}_q4.safetensors"

                    if not _q4_path.exists():
                        _q4_filename = _q4_path.name
                        from exo.shared.constants import EXO_FILE_SERVER_PORT

                        _peer_ips = _find_peer_ips()
                        _downloaded = False
                        for _ip in _peer_ips:
                            _url = f"http://{_ip}:{EXO_FILE_SERVER_PORT}/mtp_cache/{_q4_filename}"
                            try:
                                import urllib.request

                                logger.info(f"Downloading MTP q4 from peer: {_url}")
                                urllib.request.urlretrieve(_url, str(_q4_path))
                                logger.info(f"MTP q4 downloaded via P2P: {_q4_path}")
                                _downloaded = True
                                break
                            except Exception as e:
                                logger.warning(
                                    f"P2P MTP download from {_ip} failed: {e}"
                                )

                        if not _downloaded:
                            _resolver._resolve_mtp_weights()
                    else:
                        logger.info(f"MTP weights ready (cached): {_q4_path}")
            except Exception as e:
                logger.warning(f"MTP preparation failed on non-rank-0: {e}")

    set_wired_limit_for_model(get_weights_size(bound_instance.bound_shard))

    mx.clear_cache()

    vision_config = bound_instance.bound_shard.model_card.vision

    if vision_config is not None:
        from exo.worker.engines.mlx.vision import VisionProcessor

        vision_start_time = time.perf_counter()
        try:
            vision_processor: VisionProcessor | None = VisionProcessor(
                vision_config, bound_instance.bound_shard.model_card.model_id
            )
            vision_processor.load()
            logger.info(
                f"Time taken to load vision weights: {(time.perf_counter() - vision_start_time):.2f}s"
            )
        except Exception as e:
            logger.opt(exception=e).error(
                "Failed to load vision weights — disabling vision for this runner"
            )
            vision_processor = None
    else:
        vision_processor = None

    return cast(Model, model), tokenizer, vision_processor


def shard_and_load(
    shard_metadata: ShardMetadata,
    group: mx.distributed.Group,
) -> Generator[ModelLoadingResponse, None, tuple[nn.Module, TokenizerWrapper]]:
    model_path = build_model_path(shard_metadata.model_card.model_id)

    # --- Prepare MTP weights BEFORE model load (rank 0 only, no OOM risk) ---
    _is_pp_for_mtp = (
        isinstance(shard_metadata, PipelineShardMetadata)
        and shard_metadata.device_rank == 0
    )
    if _is_pp_for_mtp and os.environ.get("EXO_SPECULATIVE", "0") == "1":
        try:
            _prepare_mtp_weights(shard_metadata.model_card.model_id, model_path)
        except Exception as e:
            logger.warning(f"MTP weight preparation failed: {e}")

    model, _ = load_model(model_path, lazy=True, strict=False)
    logger.debug(model)

    # Optionally overlay the DEDICATED mlx-community DSv4 MTP head onto the
    # native (checkpoint-bundled) mtp[0], BEFORE tensor sharding. The dedicated
    # head (mlx-community/DeepSeek-V4-Flash-MTP-bf16) is the same trained MTP
    # weights re-packaged (fused switch_mlp, decoder.* prefix, affine-8bit).
    # We overlay here while the module is still unsharded so the subsequent
    # tensor_auto_parallel shards it identically to the native head. Gated by
    # EXO_DSV4_MTP_DEDICATED=1 so the proven native path stays the default.
    if (
        os.environ.get("EXO_DSV4_MTP", "0") == "1"
        and os.environ.get("EXO_DSV4_MTP_DEDICATED", "0") == "1"
    ):
        try:
            _overlay_dsv4_dedicated_mtp(model, model_path)
        except Exception as e:
            logger.warning(
                f"DSv4 dedicated MTP overlay failed ({e}); keeping native MTP head."
            )

    # DSpark 3-stage draft head (task #19, arXiv:2607.05147). Attached BEFORE
    # tensor sharding so its DeepseekV4MoE ffns shard exactly like the native
    # mtp head's. Draft-phase swap in dsv4_mtp is separately gated on the
    # module being present.
    if os.environ.get("EXO_DSV4_DSPARK", "0") == "1":
        _dspark_ok = 1
        try:
            _overlay_dsv4_dspark(model)
        except Exception as e:
            _dspark_ok = 0
            logger.warning(
                f"DSv4 DSpark overlay failed ({e}); falling back to MTP-1 draft."
            )
        # RANK-CONSISTENCY GUARD (2026-07-12, field incident): a rank with a
        # missing/partial head dir falls back to MTP-1 while its peer attaches
        # DSpark — the two ranks then run DIFFERENT draft paths and desync the
        # TP collectives on the first spec cycle. Agree across the group: if
        # ANY rank failed, ALL ranks detach and fall back together.
        if group is not None and group.size() > 1:
            _agree = mx.distributed.all_sum(
                mx.array([_dspark_ok], dtype=mx.int32), group=group
            )
            mx.eval(_agree)
            if int(_agree.item()) < group.size() and _dspark_ok:
                inner_m = getattr(model, "model", None)
                if inner_m is not None and hasattr(inner_m, "dspark"):
                    del inner_m.dspark
                from mlx_lm.models.deepseek_v4 import set_dspark_taps

                set_dspark_taps([])
                logger.warning(
                    "DSpark overlay succeeded locally but failed on a peer "
                    "rank; detaching for rank consistency (MTP-1 fallback)."
                )

    if hasattr(model, "model") and isinstance(model.model, DeepseekV3Model):  # type: ignore
        pass
        # TODO: See if we should quantize the model.
        # def is_attention_layer(path: str) -> bool:
        #     path = path.lower()

        #     return "self_attn" in path and "layernorm" not in path

        # def quant_predicate(path: str, module: nn.Module):
        #     if not isinstance(module, nn.Linear):
        #         return False

        #     return is_attention_layer(path)
        # model, config = quantize_model(
        #        model, config, group_size=KV_GROUP_SIZE, bits=ATTENTION_KV_BITS, quant_predicate=quant_predicate, mode=QUANTIZE_MODEL_MODE
        #    )

    assert isinstance(model, nn.Module)

    tokenizer = get_tokenizer(model_path, shard_metadata)

    logger.info(f"Group size: {group.size()}, group rank: {group.rank()}")

    match shard_metadata:
        case TensorShardMetadata():
            logger.info(f"loading model from {model_path} with tensor parallelism")
            model = yield from tensor_auto_parallel(model, group)
        case PipelineShardMetadata():
            logger.info(f"loading model from {model_path} with pipeline parallelism")
            model = yield from pipeline_auto_parallel(model, group, shard_metadata)
            mx.eval(model.parameters())
        case CfgShardMetadata():
            raise ValueError(
                "CfgShardMetadata is not supported for text model loading - "
                "this metadata type is only for image generation models"
            )

    # TODO: Do we need this?
    mx.eval(model)

    # Apply model-type kernel-fusion patches (MoE switch_mlp gate+up fusion
    # for DSv4, batched fused kernels for Qwen3.5 MoE, etc.) AFTER sharding
    # so any fused-weight buffers the patches build (e.g. BatchedSwitchGLU's
    # concatenated gate+up) are constructed from the sharded weights. The
    # single-device load path calls maybe_apply_patches in load_mlx_items;
    # the distributed path previously did NOT, so no patches applied on TP
    # clusters (a pre-existing gap). Apply to the inner model form returned
    # by tensor_auto_parallel — maybe_apply_patches / apply_dsv4_moe_patches
    # handle both wrapper and inner-model layouts.
    try:
        from exo.worker.engines.mlx.patches import maybe_apply_patches

        maybe_apply_patches(model, model_path)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"maybe_apply_patches failed on distributed path: {e}")

    # Convert bf16 → fp16 for Qwen3.5 MoE if COMPUTE_DTYPE is fp16
    config_path = model_path / "config.json"
    if config_path.exists():
        import json

        with open(config_path) as f:
            _cfg = json.load(f)
        if _cfg.get("model_type") == "qwen3_5_moe":
            from exo.worker.engines.mlx.patches.qwen3_5_moe.common import (
                convert_model_to_compute_dtype,
            )

            convert_model_to_compute_dtype(model)

    logger.debug("SHARDED")
    logger.debug(model)

    # --- Load draft model for PP speculation (device_rank 0 only, before barrier) ---
    # Use device_rank (stable, assigned by master) not group.rank() (JACCL, non-deterministic)
    # Skip draft model when MTP is available — MTP replaces it and saves ~1-2 GB.
    _draft_path = os.environ.get("EXO_PP_DRAFT_MODEL", "")
    _is_pp_rank0 = (
        isinstance(shard_metadata, PipelineShardMetadata)
        and shard_metadata.device_rank == 0
    )
    _has_mtp = (
        os.environ.get("EXO_SPECULATIVE", "0") == "1"
        and any(
            (Path.home() / ".cache" / "exo" / "mtp_weights").glob(
                "mtp_*_q4.safetensors"
            )
        )
        if (Path.home() / ".cache" / "exo" / "mtp_weights").exists()
        else False
    )
    if not _has_mtp and _draft_path and _is_pp_rank0:
        try:
            from .pp_speculation import load_draft_model

            _draft_result = load_draft_model(_draft_path)
            if _draft_result is not None:
                model._pp_draft_model, model._pp_draft_cache = _draft_result  # type: ignore
                logger.info(f"PP draft model loaded on rank 0: {_draft_path}")
        except Exception as e:
            logger.warning(f"PP draft model load error: {e}")
    elif _has_mtp and _draft_path and _is_pp_rank0:
        logger.info("Skipping draft model load — MTP is available (saves ~1-2 GB)")

    # Synchronize processes before generation to avoid timeout
    # (rank 1 waits here while rank 0 finishes loading draft model + MTP quantization)
    mx_barrier(group)

    return model, tokenizer


def _overlay_dsv4_dedicated_mtp(model: Any, model_path: Path) -> None:
    """Overlay mlx-community/DeepSeek-V4-Flash-MTP-bf16 onto model.model.mtp[0].

    Must run BEFORE tensor sharding (module still has full, unsharded weights).
    The dedicated head ships the SAME trained MTP weights as the checkpoint-
    bundled native head, re-packaged with a ``decoder.`` prefix and affine-8bit
    quantization (fused switch_mlp). We:
      1. resolve/download the head's single model.safetensors,
      2. strip the ``decoder.`` prefix,
      3. quantize mtp[0] to affine-8bit only where the head provides ``.scales``,
      4. load_weights(strict=False) — the head omits zero affine biases, which
         is fine.
    """
    import json

    import mlx.core as mx
    import mlx.nn as nn
    from huggingface_hub import hf_hub_download

    inner = getattr(model, "model", None)
    mtp_list = getattr(inner, "mtp", None) if inner is not None else None
    if not mtp_list:
        raise RuntimeError("model has no model.mtp[] to overlay")
    mtp0 = mtp_list[0]

    repo = "mlx-community/DeepSeek-V4-Flash-MTP-bf16"
    sf = hf_hub_download(repo, "model.safetensors")
    raw = mx.load(sf)
    # Strip the decoder. prefix → matches DeepseekV4MTPModule's own param tree.
    remap = {
        (k[len("decoder."):] if k.startswith("decoder.") else k): v
        for k, v in raw.items()
    }

    # Quantize mtp0 only on submodules the head ships quantized (i.e. has a
    # matching ``<path>.scales``), and quantize EACH layer to match the head's
    # ACTUAL on-disk packing rather than a single hardcoded scheme.
    #
    # The canonical mlx-community/DeepSeek-V4-Flash-MTP-bf16 head is NOT uniform
    # affine: every quantized layer ships uint8 scales with NO biases (true mxfp
    # packing), where the routed experts (ffn.switch_mlp.*) are mxfp4 and all
    # other projections (attn.*, e_proj, h_proj, ffn.shared_experts.*) are
    # mxfp8 — all group_size=32. Previously this overlay force-quantized mtp0 to
    # affine group_size=64, which synthesizes bf16 biases and a 64-wide scale
    # tensor. Loading the head's mxfp8 weights (uint8 scales, 128-wide,
    # group_size=32, no biases) onto that affine layer then crashed at MTP
    # predict with "[quantized_matmul] Scales and biases should have the same
    # shape. Received scales (4096,128) and biases (4096,64)" (e_proj/h_proj).
    #
    # Infer the per-layer scheme directly from the on-disk tensors:
    #   - uint8 scales + no biases  -> mxfp; bits from weight/scale width ratio
    #       ratio = packed_weight.shape[-1]*(32//bits_guess)... computed below
    #       as 4 -> mxfp4, 8 -> mxfp8; group_size = in_features / scale_groups.
    #   - non-uint8 scales (bf16/fp16/fp32) -> affine; group_size from scales.
    # This mirrors mlx_lm's own DSv4 packing conventions per layer.
    import mlx.core as _mx

    def _infer_quant_params(path: str) -> dict[str, Any] | None:
        w = remap.get(f"{path}.weight")
        s = remap.get(f"{path}.scales")
        if w is None or s is None:
            return None
        b = remap.get(f"{path}.biases")
        in_packed = int(w.shape[-1])
        scale_groups = int(s.shape[-1])
        if s.dtype == _mx.uint8 and b is None:
            # mxfp: uint32-packed weights hold (32 // bits) values per word.
            # in_features = in_packed * (32 // bits); group_size =
            # in_features / scale_groups. mxfp group_size is always 32, so
            # bits = 32 // (in_packed * 32 / scale_groups / 32)
            #      = scale_groups * 32 // in_packed ... resolve via the two
            # supported mxfp widths (4, 8) by matching group_size == 32.
            for cand_bits in (4, 8):
                in_features = in_packed * (32 // cand_bits)
                if in_features % scale_groups:
                    continue
                if in_features // scale_groups == 32:
                    return {
                        "group_size": 32,
                        "bits": cand_bits,
                        "mode": "mxfp4" if cand_bits == 4 else "mxfp8",
                    }
            return None
        # affine: group_size = in_features / scale_groups, bf16/fp16/fp32 scales.
        # in_features for affine uint32 packing = in_packed * (32 // bits); the
        # head's affine layers (if any) follow the model config's bits.
        cfg_bits = int(bits)
        in_features = in_packed * (32 // cfg_bits)
        grp = in_features // scale_groups if scale_groups else int(gs)
        return {"group_size": grp, "bits": cfg_bits, "mode": "affine"}

    cfg = json.loads((model_path / "config.json").read_text())
    q = cfg.get("quantization", {}) or {}
    gs = int(q.get("group_size", 64))
    bits = int(q.get("bits", 8))
    scale_paths = {k[: -len(".scales")] for k in remap if k.endswith(".scales")}
    _qparams: dict[str, dict[str, Any]] = {}
    for _p in scale_paths:
        _ip = _infer_quant_params(_p)
        if _ip is not None:
            _qparams[_p] = _ip

    def _qpred(path: str, m: Any) -> Any:
        if not (hasattr(m, "to_quantized") and path in scale_paths):
            return False
        # Return a per-layer params dict so each layer is packed to match the
        # head's on-disk format (mxfp4 experts / mxfp8 rest / affine fallback).
        return _qparams.get(path, {"group_size": gs, "bits": bits, "mode": "affine"})

    nn.quantize(mtp0, group_size=gs, bits=bits, class_predicate=_qpred)

    # strict=False: the head omits all-zero affine biases (.biases) — harmless.
    mtp0.load_weights(list(remap.items()), strict=False)
    mx.eval(mtp0.parameters())
    logger.info(
        f"Overlaid dedicated DSv4 MTP head from {repo} "
        f"({len(remap)} tensors) onto mtp[0]."
    )


def _overlay_dsv4_dspark(model: Any) -> None:
    """Attach + load the DSpark 3-stage draft head as ``model.model.dspark``.

    Loads the locally converted dedicated head (built by the DSpark
    converter from deepseek-ai/DeepSeek-V4-Flash-DSpark's mtp.* shards;
    recipe-matched to the serving checkpoint: mxfp4 experts / mxfp8
    projections at group 32, bf16 markov heads, f32 confidence/hc). Also
    arms the ctx-capture taps at ``dspark_target_layer_ids``.
    """
    import json as _json
    import re as _re

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.deepseek_v4 import (
        DeepseekV4DSparkModule,
        set_dspark_taps,
    )

    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "args"):
        raise RuntimeError("model has no inner DSv4 model/args")

    head_dir = Path(
        os.environ.get(
            "EXO_DSV4_DSPARK_DIR",
            str(Path.home() / ".exo/models/local--DeepSeek-V4-Flash-DSpark-MTP"),
        )
    )
    head_cfg = _json.loads((head_dir / "config.json").read_text())
    for _k in (
        "dspark_block_size",
        "dspark_markov_rank",
        "dspark_noise_token_id",
        "dspark_target_layer_ids",
        "n_mtp_layers",
    ):
        if _k in head_cfg:
            setattr(inner.args, _k, head_cfg[_k])

    raw = mx.load(str(head_dir / "model.safetensors"))

    _specials = {
        "main_proj": "main_proj",
        "main_norm": "main_norm",
        "norm": "norm",
        "hc_head": "hc_head",
        "markov_head.markov_w1": "markov_w1",
        "markov_head.markov_w2": "markov_w2",
        "confidence_head.proj": "confidence_proj",
    }

    def _remap(k: str) -> str:
        m = _re.match(r"decoder\.(\d+)\.(.*)", k)
        st, rest = m.group(1), m.group(2)
        for pre, dst in _specials.items():
            if rest == pre or rest.startswith(pre + "."):
                return dst + rest[len(pre):]
        return f"stages.{st}.{rest}"

    weights = {_remap(k): v for k, v in raw.items()}

    mod = DeepseekV4DSparkModule(inner.args)

    # Per-layer scheme inference (same convention as the dedicated-MTP
    # overlay): uint8 scales with no biases -> mxfp; bits from the u32
    # packing ratio at group 32.
    schemes: dict[str, dict[str, Any]] = {}
    for k, v in weights.items():
        if not k.endswith(".scales"):
            continue
        base = k[: -len(".scales")]
        w = weights[base + ".weight"]
        in_packed = int(w.shape[-1])
        n_groups = int(v.shape[-1])
        for cand_bits in (4, 8):
            in_features = in_packed * (32 // cand_bits)
            if n_groups and in_features % n_groups == 0 and in_features // n_groups == 32:
                schemes[base] = {
                    "group_size": 32,
                    "bits": cand_bits,
                    "mode": "mxfp4" if cand_bits == 4 else "mxfp8",
                }
                break
        else:
            raise RuntimeError(f"cannot infer quant scheme for {base}")

    def _qpred(path: str, m: Any) -> Any:
        if not (hasattr(m, "to_quantized") and path in schemes):
            return False
        return schemes[path]

    nn.quantize(mod, group_size=32, bits=8, class_predicate=_qpred)
    mod.load_weights(list(weights.items()), strict=True)
    mx.eval(mod.parameters())

    inner.dspark = mod
    set_dspark_taps(mod.target_layer_ids)
    logger.info(
        f"DSpark draft head attached from {head_dir} "
        f"({len(weights)} tensors, {len(mod.stages)} stages, "
        f"block_size={mod.block_size}, taps={mod.target_layer_ids})."
    )


def _prepare_mtp_weights(model_id: str, model_path: Path) -> None:
    """Download + quantize MTP weights before model is loaded (no OOM risk).

    Detects MTP support from config.json, resolves weights, auto-quantizes.
    Runs on rank 0 only, before the barrier.
    """
    import hashlib
    import json

    # Detect model_type from config.json (no model load needed)
    config_path = model_path / "config.json"
    if not config_path.exists():
        return
    config = json.loads(config_path.read_text())
    model_type = config.get("model_type", "") or config.get("text_config", {}).get(
        "model_type", ""
    )
    if "qwen3_5" not in model_type:
        return  # Only Qwen3.5 models have MTP

    from .generator.batch_generate import ExoBatchGenerator

    # Map model_id to HF repo
    mtp_repo = ExoBatchGenerator._model_id_to_hf_repo(model_id)
    if not mtp_repo:
        return

    cache_dir = Path.home() / ".cache" / "exo" / "mtp_weights"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.md5(mtp_repo.encode()).hexdigest()[:12]
    q4_path = cache_dir / f"mtp_{cache_key}_q4.safetensors"

    if q4_path.exists():
        logger.info(f"MTP weights ready (rank 0): {q4_path}")
        return

    # Resolve bf16 weights (download from HF if needed)
    bf16_path = cache_dir / f"mtp_{cache_key}.safetensors"
    if not bf16_path.exists():
        # Check local model for MTP weights
        _resolver = ExoBatchGenerator.__new__(ExoBatchGenerator)
        _resolver.model_id = model_id

        # Dummy model attribute for _extract_mtp_from_local
        local_path = _resolver._extract_mtp_from_local(model_path, cache_dir, cache_key)
        if not local_path:
            try:
                local_path = _resolver._extract_mtp_from_hf(mtp_repo)
            except Exception as e:
                logger.warning(f"MTP download failed: {e}")
                return
        if local_path:
            bf16_path = Path(local_path)

    if bf16_path.exists():
        # Auto-quantize (model not loaded yet, plenty of memory)
        q4 = ExoBatchGenerator._auto_quantize_mtp(bf16_path, q4_path)
        if q4:
            logger.info(f"MTP weights ready (rank 0): {q4}")
        else:
            logger.info(f"MTP weights ready (rank 0, bf16): {bf16_path}")


def _find_peer_ips() -> list[str]:
    """Find peer IPs from EXO_DISCOVERY_PEERS for P2P file transfer."""
    import re

    peers = os.environ.get("EXO_DISCOVERY_PEERS", "")
    # Format: /ip4/192.168.x.x/tcp/52415/p2p/...
    return re.findall(r"/ip4/([\d.]+)/", peers)


def get_tokenizer(model_path: Path, shard_metadata: ShardMetadata) -> TokenizerWrapper:
    """Load tokenizer for a model shard. Delegates to load_tokenizer_for_model_id."""
    return load_tokenizer_for_model_id(
        shard_metadata.model_card.model_id,
        model_path,
        trust_remote_code=shard_metadata.model_card.trust_remote_code,
    )


def get_eos_token_ids_for_model(model_id: ModelId) -> list[int] | None:
    """
    Get the EOS token IDs for a model based on its ID.

    Some models require explicit EOS token configuration that isn't in their
    tokenizer config. This function returns the known EOS token IDs for such models.

    Args:
        model_id: The HuggingFace model ID

    Returns:
        List of EOS token IDs, or None if the model uses standard tokenizer config
    """
    model_id_lower = model_id.lower()
    if "kimi-k2" in model_id_lower:
        return [163586]
    elif "glm-5" in model_id_lower:
        # 154820: <|endoftext|>, 154827: <|user|>, 154829: <|observation|>
        return [154820, 154827, 154829]
    elif "glm" in model_id_lower:
        # For GLM-4.7 and older
        return [151336, 151329, 151338]
    elif "gpt-oss" in model_id_lower:
        return [200002, 200012]
    elif (
        "qwen3.5" in model_id_lower
        or "qwen-3.5" in model_id_lower
        or "qwen3.6" in model_id_lower
        or "qwen-3.6" in model_id_lower
    ):
        # For Qwen3.5 / Qwen3.6: 248046 (<|im_end|>), 248044 (<|endoftext|>)
        return [248046, 248044]
    elif "gemma-4" in model_id_lower or "gemma-3" in model_id_lower:
        return [1, 106, 50]
    return None


def load_tokenizer_for_model_id(
    model_id: ModelId, model_path: Path, *, trust_remote_code: bool = TRUST_REMOTE_CODE
) -> TokenizerWrapper:
    """
    Load tokenizer for a model given its ID and local path.

    This is the core tokenizer loading logic, handling special cases for different
    model families (Kimi, GLM, etc.) and transformers 5.x compatibility.

    Args:
        model_id: The HuggingFace model ID (e.g., "moonshotai/Kimi-K2-Instruct")
        model_path: Local path where the model/tokenizer files are stored

    Returns:
        TokenizerWrapper instance configured for the model
    """
    model_id_lower = model_id.lower()
    eos_token_ids = get_eos_token_ids_for_model(model_id)

    # Kimi uses a custom TikTokenTokenizer that transformers 5.x can't load via AutoTokenizer
    if "kimi-k2" in model_id_lower:
        import importlib.util
        import types

        sys.path.insert(0, str(model_path))

        # Load tool_declaration_ts first (tokenization_kimi imports it with relative import)
        tool_decl_path = model_path / "tool_declaration_ts.py"
        if tool_decl_path.exists():
            spec = importlib.util.spec_from_file_location(
                "tool_declaration_ts", tool_decl_path
            )
            if spec and spec.loader:
                tool_decl_module = importlib.util.module_from_spec(spec)
                sys.modules["tool_declaration_ts"] = tool_decl_module
                spec.loader.exec_module(tool_decl_module)

        # Load tokenization_kimi with patched source (convert relative to absolute import)
        tok_path = model_path / "tokenization_kimi.py"
        source = tok_path.read_text()
        source = source.replace("from .tool_declaration_ts", "from tool_declaration_ts")
        spec = importlib.util.spec_from_file_location("tokenization_kimi", tok_path)
        if spec:
            tok_module = types.ModuleType("tokenization_kimi")
            tok_module.__file__ = str(tok_path)
            sys.modules["tokenization_kimi"] = tok_module
            exec(compile(source, tok_path, "exec"), tok_module.__dict__)  # noqa: S102
            TikTokenTokenizer = tok_module.TikTokenTokenizer  # type: ignore[attr-defined]  # noqa: N806
        else:
            from tokenization_kimi import TikTokenTokenizer  # type: ignore[import-not-found]  # noqa: I001

        hf_tokenizer: Any = TikTokenTokenizer.from_pretrained(model_path)  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]

        # Patch encode to use internal tiktoken model directly
        # transformers 5.x has a bug in the encode->pad path for slow tokenizers
        def _patched_encode(text: str, **_kwargs: object) -> list[int]:
            # Pass allowed_special="all" to handle special tokens like <|im_user|>
            return list(hf_tokenizer.model.encode(text, allowed_special="all"))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]

        hf_tokenizer.encode = _patched_encode
        return TokenizerWrapper(
            hf_tokenizer,
            eos_token_ids=eos_token_ids,
            tool_call_start="<|tool_calls_section_begin|>",
            tool_call_end="<|tool_calls_section_end|>",
            tool_parser=_parse_kimi_tool_calls,
        )

    # We should really consider going back to mlx lm load to get tokenizer
    tokenizer = load_tokenizer(
        model_path,
        tokenizer_config_extra={"trust_remote_code": trust_remote_code},
        eos_token_ids=eos_token_ids,
    )

    return tokenizer


def _normalize_tool_calls(msg_dict: dict[str, Any]) -> None:
    """Normalize tool_calls in a message dict.

    OpenAI format has tool_calls[].function.arguments as a JSON string,
    but some chat templates (e.g., GLM) expect it as a dict.
    """
    tool_calls = msg_dict.get("tool_calls")
    if not tool_calls or not isinstance(tool_calls, list):
        return

    for tc in tool_calls:  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(tc, dict):
            continue
        func = tc.get("function")  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        if not isinstance(func, dict):
            continue
        args = func.get("arguments")  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        if isinstance(args, str):
            with contextlib.suppress(json.JSONDecodeError):
                func["arguments"] = json.loads(args)


def _collect_nested_property_names(schema: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    properties: dict[str, Any] = schema.get("properties", {})  # type: ignore[reportAny]
    for prop_spec in properties.values():  # pyright: ignore[reportAny]
        if not isinstance(prop_spec, dict):
            continue
        if prop_spec.get("type") == "array":  # type: ignore[reportAny]
            items: dict[str, Any] | None = prop_spec.get("items")  # type: ignore[reportAny]
            if isinstance(items, dict) and items.get("type") == "object":  # type: ignore[reportAny]
                inner_props: dict[str, Any] = items.get("properties", {})  # type: ignore[reportAny]
                for k in inner_props:  # pyright: ignore[reportUnknownVariableType]
                    names.add(str(k))  # pyright: ignore[reportUnknownArgumentType]
                names.update(_collect_nested_property_names(items))  # pyright: ignore[reportUnknownArgumentType]
    return names


def _schemas_lost_in_prompt(prompt: str, tools: list[dict[str, Any]]) -> bool:
    """Return True if nested property names from any tool schema are absent."""
    for tool in tools:
        fn: dict[str, Any] = tool.get("function", {})  # type: ignore
        params: dict[str, Any] = fn.get("parameters", {})  # type: ignore
        nested = _collect_nested_property_names(params)
        if nested and not all(name in prompt for name in nested):
            return True
    return False


_LOSSY_TEMPLATE_PATTERN = re.compile(
    r"""inner_type\s*==\s*["']object \| object["']\s*or\s*inner_type\|length\s*>\s*\d+""",
)


def _patch_lossy_chat_template(template: str) -> str | None:
    """Patch chat templates that collapse nested object schemas to ``any[]``.

    Some templates (e.g., GPT-OSS) have a guard like::

        inner_type == "object | object" or inner_type|length > 50

    The length check silently drops complex array-of-object schemas.
    We remove the length guard, keeping only the object-union check.
    Returns the patched template, or *None* if no patch was needed.
    """
    patched, n = _LOSSY_TEMPLATE_PATTERN.subn(
        lambda m: m.group(0).split(" or ")[0],  # keep only the object-union check
        template,
    )
    return patched if n > 0 else None


def _needs_dsml_encoding(task_params: TextGenerationTaskParams) -> bool:
    return "deepseek-v3.2" in task_params.model.lower()


def _needs_v4_encoding(task_params: TextGenerationTaskParams) -> bool:
    return "deepseek-v4" in task_params.model.lower()


def _v4_reasoning_effort(task_params: TextGenerationTaskParams) -> str | None:
    effort = task_params.reasoning_effort
    if effort == "xhigh":
        return "max"
    if effort == "high":
        return "high"
    return None


def _strip_v4_thinking_markers(content: str) -> str:
    """Remove `<think>…</think>` blocks and any stray `<think>`/`</think>` tags
    from prior-turn assistant content.

    The V4 encoder drops `reasoning_content` for older turns when
    `drop_thinking=True`"""
    block = re.compile(r"<think>.*?</think>", re.DOTALL)
    if not content:
        return content
    cleaned = block.sub("", content)
    return cleaned.replace("<think>", "").replace("</think>", "")


def consolidate_system_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    System messages almost exclusively must go at the start of a message
    and there must only be a single one.

    Also, Codex sends "developer" messages which are just system prompts.
    """
    system_parts: list[str] = []
    non_system: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") in ("system", "developer"):
            content = cast(str, msg.get("content", ""))
            if content:
                system_parts.append(content)
        else:
            non_system.append(msg)
    formatted_messages = non_system
    if system_parts:
        formatted_messages.insert(
            0, {"role": "system", "content": "\n".join(system_parts)}
        )
    return formatted_messages


def render_chat_template(
    tokenizer: TokenizerWrapper,
    messages: list[dict[str, Any]],
    task_params: TextGenerationTaskParams,
) -> str:
    """
    Convert TextGenerationTaskParams to a chat template prompt.

    Converts the internal format (input + instructions) to a messages list
    that can be processed by the tokenizer's chat template.

    When chat_template_messages is available (from Chat Completions API),
    uses those directly to preserve tool_calls, thinking, and other fields.
    """
    formatted_messages = consolidate_system_messages(messages)

    # For assistant prefilling, append content after templating to avoid a closing turn token.
    partial_assistant_content: str | None = None
    if formatted_messages and formatted_messages[-1].get("role") == "assistant":
        partial_assistant_content = cast(str, formatted_messages[-1].get("content", ""))
        formatted_messages = formatted_messages[:-1]

    if _needs_dsml_encoding(task_params):
        from exo.worker.engines.mlx.vendor.dsml_encoding import encode_messages

        prompt = encode_messages(
            messages=formatted_messages,
            # Only use chat mode if enable thinking is explicitly Fakse.
            thinking_mode="chat"
            if task_params.enable_thinking is False
            else "thinking",
            tools=task_params.tools,
        )
        if partial_assistant_content:
            prompt += partial_assistant_content
        return prompt

    if _needs_v4_encoding(task_params):
        from exo.worker.engines.mlx.vendor.deepseek_v4_encoding import (
            encode_messages as encode_messages_v4,
        )

        v4_messages = [dict(m) for m in formatted_messages]
        for msg in v4_messages:
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, str):
                    msg["content"] = _strip_v4_thinking_markers(content)
        if task_params.tools:
            for msg in v4_messages:
                if msg.get("role") in ("system", "developer"):
                    msg["tools"] = task_params.tools
                    break
            else:
                v4_messages.insert(
                    0, {"role": "system", "content": "", "tools": task_params.tools}
                )

        prompt = encode_messages_v4(
            messages=v4_messages,
            thinking_mode="chat"
            if task_params.enable_thinking is False
            else "thinking",
            reasoning_effort=_v4_reasoning_effort(task_params),
        )
        if partial_assistant_content:
            prompt += partial_assistant_content
        return prompt

    for msg in formatted_messages:
        _normalize_tool_calls(msg)

    # Put reasoning content in thinking block for GPT OSS
    if "gpt-oss" in task_params.model.lower():
        for msg in formatted_messages:
            if msg.get("role") == "assistant" and "thinking" not in msg:
                rc = msg.get("reasoning_content")
                if isinstance(rc, str) and rc:
                    msg["thinking"] = rc

    extra_kwargs: dict[str, Any] = {}
    if task_params.enable_thinking is not None:
        # Qwen3 and GLM use "enable_thinking"; DeepSeek uses "thinking".
        # Jinja ignores unknown variables, so passing both is safe.
        extra_kwargs["enable_thinking"] = task_params.enable_thinking
        extra_kwargs["thinking"] = task_params.enable_thinking
    if task_params.reasoning_effort is not None:
        extra_kwargs["reasoning_effort"] = task_params.reasoning_effort

    patched_template: str | None = None
    if task_params.tools:
        original_template: str | None = getattr(tokenizer, "chat_template", None)
        if isinstance(original_template, str):
            patched_template = _patch_lossy_chat_template(original_template)
            if patched_template is not None:
                logger.info(
                    "Patched lossy chat template (removed inner_type length guard)"
                )

    prompt: str = tokenizer.apply_chat_template(
        formatted_messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=task_params.tools,
        **({"chat_template": patched_template} if patched_template is not None else {}),
        **extra_kwargs,
    )

    if task_params.tools and _schemas_lost_in_prompt(prompt, task_params.tools):
        logger.warning("Chat template lost nested tool schemas even after patching")

    if partial_assistant_content:
        prompt += partial_assistant_content

    return prompt


def apply_chat_template(
    tokenizer: TokenizerWrapper,
    task_params: TextGenerationTaskParams,
) -> str:
    messages: list[dict[str, ChatTemplateValue]] = []
    
    def _flatten_content(content: Any) -> Any:
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
            return "\n\n".join(text_parts) if text_parts else ""
        return content

    if task_params.chat_template_messages is not None:
        # Use pre-formatted messages that preserve tool_calls, thinking, etc.
        for msg in task_params.chat_template_messages:
            flattened_msg = msg.copy()
            if "content" in flattened_msg and isinstance(flattened_msg["content"], list):
                flattened_msg["content"] = _flatten_content(flattened_msg["content"])
            messages.append(flattened_msg)
    else:
        # Add system message (instructions) if present
        if task_params.instructions:
            messages.append({"role": "system", "content": task_params.instructions})

        # Convert input to messages
        for msg in task_params.input:
            if not msg.content:
                logger.warning("Received message with empty content, skipping")
                continue
            messages.append({"role": msg.role, "content": _flatten_content(msg.content)})

    prompt = render_chat_template(tokenizer, messages, task_params)
    logger.debug(prompt)

    return prompt


def system_prompt_token_count(
    task_params: TextGenerationTaskParams,
    tokenizer: TokenizerWrapper,
) -> int:
    """Approximate token count of the system prompt portion of the input."""
    parts: list[str] = []
    if task_params.chat_template_messages is not None:
        for msg in task_params.chat_template_messages:
            if msg.get("role") in ("system", "developer"):
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
    else:
        if task_params.instructions:
            parts.append(task_params.instructions)
        for msg in task_params.input:
            if msg.role in ("system", "developer"):
                parts.append(msg.content)
    if len(parts) == 0:
        return 0
    return len(tokenizer.encode(" ".join(parts), add_special_tokens=False))


def detect_thinking_prompt_suffix(prompt: str, tokenizer: TokenizerWrapper) -> bool:
    """
    Detect if prompt ends with a thinking opening tag that should be
    prepended to the output stream.
    """
    think_token = tokenizer.think_start

    return think_token is not None and prompt.rstrip().endswith(think_token)


def fix_unmatched_think_end_tokens(
    tokens: mx.array, tokenizer: TokenizerWrapper
) -> mx.array:
    if not tokenizer.has_thinking:
        return tokens
    # Newer mlx-lm (>= #1114) exposes plural think_*_tokens sequences for
    # multi-token thinking markers; the version pinned in uv.lock only has
    # the singular think_*_id ints. Fall back to wrapping the single ids.
    raw_start = getattr(tokenizer, "think_start_tokens", None)
    raw_end = getattr(tokenizer, "think_end_tokens", None)
    if raw_start is None or raw_end is None:
        raw_start = (tokenizer.think_start_id,)
        raw_end = (tokenizer.think_end_id,)
    think_start_tokens: list[int] = list(raw_start)
    think_end_tokens: list[int] = list(raw_end)
    token_list: list[int] = cast(list[int], tokens.tolist())
    result: list[int] = []

    depth = 0
    accumulated_think_start_length = 0
    accumulated_think_end_length = 0

    for token in token_list:
        if token == think_start_tokens[accumulated_think_start_length]:
            accumulated_think_start_length += 1
            if accumulated_think_start_length == len(think_start_tokens):
                depth += 1
                accumulated_think_start_length = 0

        elif token == think_end_tokens[accumulated_think_end_length]:
            accumulated_think_end_length += 1
            if accumulated_think_end_length == len(think_end_tokens):
                if depth == 0:
                    result.extend(think_start_tokens)
                else:
                    depth -= 1
                accumulated_think_end_length = 0

        else:
            accumulated_think_start_length = 0
            accumulated_think_end_length = 0

        result.append(token)
    return mx.array(result)


class NullKVCache(KVCache):
    """
    A KVCache that pretends to exist but holds zero tokens.
    It satisfies .state/.meta_state and never allocates real keys/values.
    """

    def __init__(self, dtype: mx.Dtype = mx.float16):
        super().__init__()
        # zero-length K/V so shapes/dtypes are defined but empty
        self.keys = mx.zeros((1, 1, 0, 1), dtype=dtype)
        self.values = mx.zeros((1, 1, 0, 1), dtype=dtype)
        self.offset = 0

    @property
    def state(self) -> tuple[mx.array, mx.array]:
        # matches what mx.save_safetensors / mx.eval expect
        assert self.keys is not None and self.values is not None
        return self.keys, self.values

    @state.setter
    def state(self, v: tuple[mx.array, mx.array]) -> None:
        raise NotImplementedError("We should not be setting a NullKVCache.")


def mlx_force_oom(size: int = 200000) -> None:
    """
    Force an Out-Of-Memory (OOM) error in MLX by performing large tensor operations.
    """
    mx.set_default_device(mx.gpu)
    a = mx.random.uniform(shape=(size, size), dtype=mx.float32)
    b = mx.random.uniform(shape=(size, size), dtype=mx.float32)
    mx.eval(a, b)
    c = mx.matmul(a, b)
    d = mx.matmul(a, c)
    e = mx.matmul(b, c)
    f = mx.sigmoid(d + e)
    mx.eval(f)


def set_wired_limit_for_model(model_size: Memory):
    """
    A context manager to temporarily change the wired limit.

    Note, the wired limit should not be changed during an async eval.  If an
    async eval could be running pass in the streams to synchronize with prior
    to exiting the context manager.
    """
    if not mx.metal.is_available():
        return

    max_rec_size = Memory.from_bytes(
        int(mx.device_info()["max_recommended_working_set_size"])
    )
    if model_size > 0.9 * max_rec_size:
        logger.warning(
            f"Generating with a model that requires {model_size.in_float_mb:.1f} MB "
            f"which is close to the maximum recommended size of {max_rec_size.in_float_mb:.1f} "
            "MB. This can be slow. See the documentation for possible work-arounds: "
            "https://github.com/ml-explore/mlx-lm/tree/main#large-models"
        )
    mx.set_wired_limit(max_rec_size.in_bytes)
    logger.info(f"Wired limit set to {max_rec_size}.")


def mlx_cleanup(
    model: Model | None,
    tokenizer: TokenizerWrapper | None,
    group: mx.distributed.Group | None,
) -> None:
    del model, tokenizer, group
    mx.clear_cache()
    import gc

    gc.collect()


def mx_any(bool_: bool, group: mx.distributed.Group | None) -> bool:
    if group is None:
        return bool_
    num_true = mx.distributed.all_sum(
        mx.array(bool_), group=group, stream=mx.default_stream(mx.Device(mx.cpu))
    )
    mx.eval(num_true)
    return num_true.item() > 0


def mx_min_int(value: int, group: mx.distributed.Group | None) -> int:
    """Return the minimum of ``value`` across all ranks in ``group``.

    Used to make per-rank branch decisions collective: when a control-flow
    gate depends on a local integer (e.g. ``len(self._queue)``) that can
    legally differ across ranks for a window, branching on the raw local
    value lets ranks take divergent paths — the rank that enters a batched
    collective issues a TP all_reduce the other rank never matches, and the
    cluster wedges on JACCL forever. Agreeing on the MIN first guarantees
    every rank makes the identical branch decision.

    Routes through the CPU stream and uses ``mx.eval`` exactly like
    ``mx_any`` so the small control-plane reduce never aliases a bf16 decode
    buffer on the GPU stream. Call on the coord subgroup (``get_coord_group``)
    so it doesn't share the model TP group's ``next_call_id_`` counter.
    """
    if group is None:
        return value
    reduced = mx.distributed.all_min(
        mx.array(value), group=group, stream=mx.default_stream(mx.Device(mx.cpu))
    )
    mx.eval(reduced)
    return int(reduced.item())


# Per-process cache of coordination subgroups. Split off the model TP
# group exactly once per parent so all non-model-forward collectives
# (agree_on_tasks, agree_on_cancellations, has_work gate, upstream uid
# sync, MTP draft broadcast, etc.) share an isolated `next_call_id_`
# counter / QPs / buffer pool from the model TP group.
#
# Why: at c=2 with frequent step()s, the runner's small CPU-side mx_any
# collectives interleave with the model forward's TP all_sum collectives
# in the encoder dispatch queue. They share a single atomic call_id
# counter on the parent group; tiny scheduling differences across ranks
# cause call_id N to encode op A on rank 0 (e.g. 1-byte mx_any) and op
# B on rank 1 (e.g. 16384-byte bf16 all_sum). UC FIFO matching surfaces
# that as IBV_WC_LOC_LEN_ERR / silent buffer corruption (max_tasks
# bit-flipped to ~1B → 152 GB metal::malloc OOM at c=2). Diagnosed via
# JACCL_TRACE_HASH=1 2026-05-07.
#
# Splitting into a sibling subgroup gives the coord traffic its own
# next_call_id_ counter, ibv_context, PD/CQ/QPs, and buffer pool
# (mlx@97741a86 + 73b08d67). Cross-subgroup traffic can't share UC
# FIFOs, so model TP and coord traffic stop interfering.
#
# Keyed by id(parent_group). Color = 0xC00D ("coord") is just a stable
# arbitrary value; both ranks must call split with the same color.
_COORD_GROUP_CACHE: dict[int, mx.distributed.Group | None] = {}
_COORD_GROUP_COLOR: int = 0xC00D


def get_coord_group(
    group: mx.distributed.Group | None,
) -> mx.distributed.Group | None:
    """Return the sibling coord subgroup for ``group``, splitting once on
    first use. Returns ``group`` itself when there's no need to split
    (single-rank or non-TP). Both ranks must call this in matching
    order — same code path on both ranks at the same point.

    PP mode (EXO_PP_NO_COORD_COLLECTIVE=1): returns None so that mx_any /
    agree_on_* calls become local-only no-ops. Under MlxRing (TCP backend),
    group.split() throws ("[ring] Group split not supported"), so coord
    collectives would share the full PP group's TCP socket with the p2p
    send/recv. When a p2p recv blocks (rank 0 waiting for rank 1's model
    forward at 500K context), the coord all_sum can't be sent until the p2p
    completes → Event::wait timeout → runner crash. Both PP ranks serve the
    same request so the collective gate is unnecessary — the p2p handoff in
    PipelineFirstLayer/PipelineLastLayer already synchronizes the ranks.
    """
    if group is None or group.size() <= 1:
        return group
    if os.environ.get("EXO_PP_NO_COORD_COLLECTIVE") == "1":
        return None
    cached = _COORD_GROUP_CACHE.get(id(group))
    if cached is not None:
        return cached
    try:
        sub = group.split(_COORD_GROUP_COLOR)
    except RuntimeError:
        # Ring backend doesn't support group.split(). TCP is already reliable,
        # so the coordinator subgroup isn't needed. Use the full group.
        sub = group
    _COORD_GROUP_CACHE[id(group)] = sub
    return sub


def pipeline_agree_prefix_hit_length(
    local_hit_length: int,
    group: mx.distributed.Group | None,
    request_tag: int,
) -> int:
    """Agree on a single KV-prefix-cache hit-length across all PP ranks.

    PP mode (EXO_PP_NO_COORD_COLLECTIVE=1) has no coord-collective channel
    (see ``get_coord_group``'s docstring) — a coord all_sum queued behind a
    blocked p2p recv on the shared transport can deadlock. This function
    does NOT use mx_any/mx_min_int/get_coord_group; instead it does a plain
    linear reduce (leaves → rank 0) + broadcast (rank 0 → leaves) over the
    SAME raw ``group`` object PipelineFirstLayer/PipelineLastLayer already
    use for the per-layer hidden-state handoff, using the identical
    send/recv_like + mx.eval discipline already proven in production for
    PP+MTP's token/tag exchange (pp_speculation.py) — not a new transport
    pattern.

    MUST be called as a discrete pre-step strictly before any prefill chunk
    sends begin for the request identified by ``request_tag``, so there is
    no in-flight p2p traffic this could queue behind or be paired with by
    mistake. Every rank must call this exactly once per request, in the
    same order, with the same ``request_tag`` — a per-process monotonic
    counter incremented once per ``submit()`` call works because PP mode
    processes exactly one request at a time in lockstep across ranks
    (EXO_MAX_CONCURRENT_REQUESTS=1 is enforced under Pipeline sharding).

    Each rank's independently-maintained ``KVPrefixCache`` SHOULD reach an
    identical local hit-length for the same request in the common case:
    every rank processes the same sequence of add/update calls with
    identical token boundaries every turn, so their tries/snapshots stay
    naturally in lockstep. This function verifies that with a cheap
    min+max reduce rather than assuming it, and returns 0 (forcing a
    uniform cold prefill on every rank) whenever the ranks disagree —
    e.g. one rank evicted a leaf the other didn't, or a crash/reconnect
    left one rank's cache cold. Reconciling a MISMATCH by trimming a
    rank's already-materialized non-sliceable-layer state (RotatingKVCache/
    ArraysCache — sliding-window/SSM-style caches) down to some smaller
    agreed depth isn't generally safe: those layers can only be restored
    to a depth where a snapshot actually exists, and forcing an arbitrary
    target can silently fall back further than the OTHER rank's, exactly
    the class of asymmetric-hit-length bug this exists to prevent (see
    references/jaccl-reconnect-crash-loop-and-git-reset-trap.md bug #3).
    "Unanimous or cold" is the only safe-by-construction rule here — a rare
    disagreement costs one extra cold prefill, never a desync.

    Wire format: fixed-shape ``(3,)`` int32 array
    ``[request_tag, running_min, running_max]`` in both directions — int32,
    NOT bf16 (bf16's 8-bit mantissa silently rounds integers above 256, and
    hit-lengths here run into the tens of thousands). A tag mismatch on
    receipt raises immediately (protocol invariant violated — p2p channel
    desync between ranks) rather than silently pairing with a stale or
    foreign message; the caller should NOT treat that as a locally
    recoverable condition, same as any other p2p fault.

    Returns the agreed hit-length: the unanimous local value if every rank
    reported the same one, else 0.
    """
    if group is None or group.size() <= 1:
        return local_hit_length

    rank = group.rank()
    world_size = group.size()

    def _send(min_v: int, max_v: int, dst: int) -> None:
        wire = mx.array([request_tag, min_v, max_v], dtype=mx.int32)
        mx.eval(wire)
        sent = mx.distributed.send(wire, dst, group=group)
        mx.eval(sent)

    def _recv(src: int) -> tuple[int, int]:
        wire = mx.distributed.recv_like(
            mx.zeros(3, dtype=mx.int32), src, group=group
        )
        mx.eval(wire)
        raw = cast(list[int], wire.tolist())
        tag, min_v, max_v = (int(v) for v in raw)
        if tag != request_tag:
            raise RuntimeError(
                f"pipeline_agree_prefix_hit_length: tag mismatch on rank "
                f"{rank} (expected {request_tag}, got {tag}) — p2p channel "
                "desync between ranks; treating as a hard fault rather "
                "than silently proceeding with a possibly-mispaired value."
            )
        return min_v, max_v

    # Phase 1: linear reduce, leaves -> rank 0.
    running_min = local_hit_length
    running_max = local_hit_length
    if rank != world_size - 1:
        peer_min, peer_max = _recv(rank + 1)
        running_min = min(running_min, peer_min)
        running_max = max(running_max, peer_max)
    if rank != 0:
        _send(running_min, running_max, rank - 1)

    # Phase 2: broadcast the agreed (min, max) back down, rank 0 -> leaves.
    if rank == 0:
        agreed_min, agreed_max = running_min, running_max
        if world_size > 1:
            _send(agreed_min, agreed_max, 1)
    else:
        agreed_min, agreed_max = _recv(rank - 1)
        if rank != world_size - 1:
            _send(agreed_min, agreed_max, rank + 1)

    return agreed_min if agreed_min == agreed_max else 0


def prewarm_coord_group(group: mx.distributed.Group | None) -> None:
    """Eagerly create the coord subgroup at a known lockstep sync point
    (typically end of runner warmup) so both ranks call split() in
    matching order before any coord-routed traffic. Then run a tiny
    mx_any to verify the subgroup actually transfers — surfaces any
    QP-exchange wedge as a fast warmup failure rather than a silent
    decode-time deadlock.
    """
    if group is None or group.size() <= 1:
        return
    coord = get_coord_group(group)
    # Verification probe — single-byte all_sum on the coord subgroup.
    _ = mx_any(False, coord)


def mx_barrier(group: mx.distributed.Group | None):
    if group is None:
        return
    mx.eval(
        mx.distributed.all_sum(
            mx.array(1.0), group=group, stream=mx.default_stream(mx.Device(mx.cpu))
        )
    )


def _parse_kimi_tool_calls(text: str):
    import regex as re

    # kimi has a fixed function naming scheme, with a json formatted arg
    #   functions.multiply:0<|tool_call_argument_begin|>{"a": 2, "b": 3}
    _func_name_regex = re.compile(
        r"^\s*((?:functions\.)?(.+?):\d+)\s*<\|tool_call_argument_begin\|>", re.DOTALL
    )
    _func_arg_regex = re.compile(r"<\|tool_call_argument_begin\|>\s*(.*)\s*", re.DOTALL)
    _tool_call_split_regex = re.compile(
        r"<\|tool_call_begin\|>(.*?)<\|tool_call_end\|>", re.DOTALL
    )

    def _parse_single_tool(text: str) -> dict[str, Any]:
        func_name_match = _func_name_regex.search(text)
        if func_name_match is None:
            raise ValueError("No tool call found.")
        tool_call_id = func_name_match.group(1)  # e.g. "functions.get_weather:0"
        func_name = func_name_match.group(2)  # e.g. "get_weather"

        func_args_match = _func_arg_regex.search(text)
        if func_args_match is None:
            raise ValueError("No tool call arguments found.")
        func_args = func_args_match.group(1)
        arg_dct = json.loads(func_args)  # pyright: ignore[reportAny]

        return dict(id=tool_call_id, name=func_name, arguments=arg_dct)  # pyright: ignore[reportAny]

    tool_matches = _tool_call_split_regex.findall(text)
    if tool_matches:
        return [_parse_single_tool(match) for match in tool_matches]  # pyright: ignore[reportAny]
    else:
        return [_parse_single_tool(text)]


def mx_all_gather_tasks(
    tasks: list[TextGeneration],
    group: mx.distributed.Group | None,
) -> tuple[list[TextGeneration], list[TextGeneration]]:
    def encode_task_id(task_id: TaskId) -> list[int]:
        utf8_task_id = task_id.encode()
        return [
            int.from_bytes(utf8_task_id[i : i + 1]) for i in range(len(utf8_task_id))
        ]

    def decode_task_id(encoded_task_id: list[int]) -> TaskId:
        return TaskId(
            bytes.decode(b"".join((x).to_bytes(length=1) for x in encoded_task_id))
        )

    uuid_byte_length = 36

    n_tasks = len(tasks)
    # group=None (PP mode with EXO_PP_NO_COORD_COLLECTIVE): skip the collective
    # entirely. mx.distributed.all_gather with group=None uses the DEFAULT
    # group (the full PP group), not "no group" — so calling it would still
    # do a 2-rank collective, defeating the purpose. Return local-only.
    if group is None:
        agreed = list(tasks)
        different: list[TextGeneration] = []
        return agreed, different
    my_rank = group.rank()
    # Route through the CPU stream (same as mx_any). On the default GPU
    # stream this all_gather aliases its output buffer with whatever bf16
    # decode op happened to land there last, and we read back float-bit
    # patterns as garbage int32s in the billions (e.g. 0x3E983B5F).
    # mx_any does this correctly — match it.
    _cpu = mx.default_stream(mx.Device(mx.cpu))
    all_counts = cast(
        list[int],
        mx.distributed.all_gather(
            mx.array([n_tasks]), group=group, stream=_cpu
        ).tolist(),
    )
    max_tasks = max(all_counts)
    world_size: int = 1 if group is None else group.size()

    if max_tasks == 0:
        return [], []

    # Sanity guard against JACCL transport corruption: if the all_gather
    # return buffer was bit-flipped, max_tasks can come back as a garbage
    # int in the billions. Without this guard, the next mx.array(padded)
    # silently asks Metal for ~152 GB and crashes the runner mid-collective,
    # taking the peer rank down with it via the unfinished RDMA call.
    # No realistic workload submits more than a handful of pending tasks at
    # once, so any value over 1024 is almost certainly garbage.
    #
    # We also log local n_tasks alongside all_counts[my_rank] so we can
    # localize the corruption: if they differ, JACCL clobbered our own
    # slot in the gather output (transport bug); if they match, the
    # corruption is in our local Python state (rare).
    if max_tasks > 1024:
        local_vs_received = all_counts[my_rank] if my_rank < len(all_counts) else "<oob>"
        raise RuntimeError(
            f"mx_all_gather_tasks: implausible max_tasks={max_tasks} from "
            f"all_counts={all_counts!r} (rank={my_rank}, local n_tasks={n_tasks}, "
            f"received self={local_vs_received}) — JACCL all_gather likely "
            f"returned a corrupted buffer. Crashing the runner cleanly so "
            f"the supervisor can restart it (better than a 152 GB metal::malloc)."
        )

    padded = [encode_task_id(task.task_id) for task in tasks] + [
        [0] * uuid_byte_length
    ] * (max_tasks - n_tasks)

    assert all(len(encoded_task_id) == uuid_byte_length for encoded_task_id in padded)

    gathered = cast(
        list[list[list[int]]],
        mx.distributed.all_gather(mx.array(padded), group=group, stream=_cpu)
        .reshape(world_size, max_tasks, -1)
        .tolist(),
    )
    all_task_ids: list[list[TaskId]] = [
        [decode_task_id(encoded_task_id) for encoded_task_id in rank_tasks[:count]]
        for rank_tasks, count in zip(gathered, all_counts, strict=True)
    ]

    agreed_ids = set[TaskId].intersection(*(set(tids) for tids in all_task_ids))

    local_tasks = {task.task_id: task for task in tasks}
    agreed = [local_tasks[tid] for tid in sorted(agreed_ids)]
    different = [task for task in tasks if task.task_id not in agreed_ids]
    return agreed, different
