"""Model loading for the DSv4.1 (EXL3) engine.

WHY THIS IS NOT ``load_mlx_items``. The DSv4.1 checkpoint is an exllamav3-style
EXL3 quant (2.9 bpw trellis groups) whose weights are consumed directly by
hand-written Metal kernels. ``mlx_lm.utils.load_model`` cannot load it, and
exo's generic post-load machinery (``tensor_auto_parallel``,
``maybe_apply_patches``, KVPrefixCache wiring) does not apply: the model has its
own cache object, its own cross-layer shared state, and its tensor parallelism
is built INTO ``exl3_build.build_model`` (routed experts take one rank's
intermediate-width slice and are ``all_sum``ed; dense projections are sharded on
128-wide Hadamard blocks; the head is vocab-sharded).

What IS reused from exo: the shard metadata's ``device_rank`` / ``world_size``
(the TP geometry), the model card's ``storage_size`` for the wired-limit
decision, exo's tokenizer loader, and the distributed group exo already
established for this instance.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.download.download_utils import build_model_path
from exo.shared.models.model_cards import ModelCard
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.shared.types.worker.shards import ShardMetadata, TensorShardMetadata
from exo.worker.engines.mlx.dsv41.engram import ensure_engram_token_map
from exo.worker.engines.mlx.dsv41.errors import Dsv41UnsupportedPlacement
from exo.worker.engines.mlx.utils_mlx import get_tokenizer, set_wired_limit_for_model
from exo.worker.runner.bootstrap import logger

#: Build only these layers (comma-separated) instead of the full stack. The
#: harness path for layer-subset validation on one node; unset in serving.
_LAYERS_ENV = "EXO_DSV41_LAYERS"
#: Native (unquantized) DSv4.1 release holding the engram tables of layers 1/14;
#: the EXL3 checkpoint does not carry them. Default: the sibling model dir.
_ENGRAM_DIR_ENV = "EXO_DSV41_ENGRAM_DIR"
_ENGRAM_DIR_NAME = "deepseek-ai--DeepSeek-V4.1-Flash-engram"


@dataclass
class Dsv41Loaded:
    """Everything the engine needs after a successful load."""

    model: Any  # mlx_lm.models.deepseek_v41.Model
    tokenizer: TokenizerWrapper
    args: Any  # mlx_lm.models.deepseek_v41.ModelArgs
    model_path: Path
    built_layers: list[int]
    full_stack: bool
    rank: int
    world: int
    load_seconds: float
    engram_token_map: Path | None = None
    head: Any | None = None
    notes: list[str] = field(default_factory=list)


def tp_geometry(shard_metadata: ShardMetadata) -> tuple[int, int]:
    """Rank/world for the in-loader tensor parallelism, from exo's shard.

    DSv4.1's TP is engine-owned (see the module docstring), so the ONLY thing
    this engine needs from exo's placement is the geometry: ``device_rank`` and
    ``world_size`` off a ``TensorShardMetadata``. Anything else -- in particular
    a multi-rank Pipeline shard -- is refused loudly, because the model's
    cross-layer caches cannot be split on a layer boundary (see
    ``Dsv41UnsupportedPlacement``).
    """
    world = shard_metadata.world_size
    if isinstance(shard_metadata, TensorShardMetadata):
        if world not in (1, 2):
            raise Dsv41UnsupportedPlacement(
                f"DSv4.1 supports world_size 1 or 2 (in-loader TP), got {world}. "
                "The EXL3 expert split and the vocab-sharded head are built for "
                "exactly 2 ranks."
            )
        return shard_metadata.device_rank, world
    if world == 1:
        # Single-node cycle: exo forces Pipeline/Ring placement, and a one-rank
        # "pipeline" is simply the whole model on this node.
        return 0, 1
    raise Dsv41UnsupportedPlacement(
        "DSv4.1 does not support multi-rank Pipeline sharding: its layers share "
        "per-forward state and cross-layer caches (compressed KV, index top-k, "
        "engram ids) that cannot cross a layer-range boundary. Place it with "
        "Tensor sharding (the model card sets supports_tensor=true)."
    )


def _requested_layers(n_layers: int) -> list[int] | None:
    raw = os.environ.get(_LAYERS_ENV, "").strip()
    if not raw:
        return None
    layers = sorted({int(x) for x in raw.split(",") if x.strip()})
    for lid in layers:
        if not 0 <= lid < n_layers:
            raise ValueError(f"{_LAYERS_ENV}: layer {lid} outside 0..{n_layers - 1}")
    logger.warning(
        f"[DSV41] {_LAYERS_ENV}={raw}: building a LAYER SUBSET. Structurally "
        "incomplete by construction -- wiring/regression tests only, never "
        "serving (every layer outside the subset is skipped)."
    )
    return layers


def read_args(model_path: Path) -> Any:
    """Read ``ModelArgs`` for a checkpoint without building any weights."""
    from mlx_lm.models.deepseek_v41.config import ModelArgs

    with open(model_path / "config.json") as f:
        config: dict[str, Any] = json.load(f)
    return ModelArgs.from_dict(config)


def load_dsv41(
    shard_metadata: ShardMetadata,
    model_card: ModelCard,
    *,
    group: mx.distributed.Group | None,
    model_path: Path | None = None,
) -> Generator[ModelLoadingResponse, None, Dsv41Loaded]:
    """Load the DSv4.1 body, reporting per-layer progress to the supervisor."""
    from mlx_lm.models.deepseek_v41 import exl3_build

    rank, world = tp_geometry(shard_metadata)
    path = model_path or build_model_path(model_card.model_id)
    args = read_args(path)
    requested = _requested_layers(int(getattr(args, "n_layers", 0)))
    full_stack = requested is None

    if world > 1 and group is None:
        logger.warning(
            "[DSV41] world_size > 1 but no distributed group: building this "
            "rank's expert slice WITHOUT the all_sum reduction. Numerically "
            "wrong (this is the single-node test geometry) — serving must pass "
            "the instance's group."
        )

    set_wired_limit_for_model(model_card.storage_size)
    started = time.perf_counter()
    logger.info(
        f"[DSV41] loading EXL3 checkpoint {path} "
        f"(rank {rank}/{world}, wired limit raised for "
        f"{model_card.storage_size.in_gb:.1f} GB)"
    )
    native_dir = _resolve_engram_dir(path, args, requested)
    model, report = exl3_build.build_model(
        str(path),
        native_dir=None if native_dir is None else str(native_dir),
        layers=requested,
        rank=rank,
        world=world,
        group=group,
    )
    built = sorted(int(k) for k in (report.get("layers") or {}))
    total = len(getattr(model, "layers", []))
    # The EXL3 build is eager Python work with no per-layer hook; report the
    # layers as a completed sweep so the supervisor's load progress advances
    # (RunnerLoading carries layers_loaded/total_layers) without pretending to
    # stream a build that has already finished.
    for i in range(total):
        yield ModelLoadingResponse(layers_loaded=i + 1, total=total)

    tokenizer = get_tokenizer(path, shard_metadata)
    engram_token_map = _resolve_engram_token_map(path, args, tokenizer)
    if engram_token_map is not None:
        with open(engram_token_map) as f:
            model.set_token_map(json.load(f))

    load_seconds = time.perf_counter() - started
    notes = [
        f"built={len(built)}/{int(getattr(args, 'n_layers', total))} layers",
        f"rank={rank}/{world}",
        f"mtp_layers_skipped={report.get('n_mtp_layers_skipped')}",
        # image rows route with gate.bias_vl (checkpoint or sidecar); False means
        # an image span would be routed with the text bias
        f"vl_bias_loaded={report.get('vl_bias_loaded')}",
    ]
    logger.info(
        f"[DSV41] body loaded in {load_seconds:.1f}s: {', '.join(notes)}; "
        f"active={mx.get_active_memory() / 1e9:.1f} GB"
    )
    return Dsv41Loaded(
        model=model,
        tokenizer=tokenizer,
        args=args,
        model_path=path,
        built_layers=built,
        full_stack=full_stack,
        rank=rank,
        world=world,
        load_seconds=load_seconds,
        engram_token_map=engram_token_map,
        notes=notes,
    )


def build_draft_head(
    loaded: Dsv41Loaded, group: mx.distributed.Group | None
) -> Any | None:
    """Attach the DSpark draft head (``mtp.{0,1,2}``) for speculative decode.

    Returns the head, or ``None`` when the build is a layer subset / the
    checkpoint carries no ``mtp.*`` groups. The head is deliberately optional:
    greedy decode works without it, so a failed draft build degrades to plain
    decoding rather than failing the load.
    """
    from mlx_lm.models.deepseek_v41 import exl3_build
    from mlx_lm.models.exl3.loader import Exl3Checkpoint

    if not loaded.full_stack:
        logger.info("[DSV41] layer-subset build: skipping the DSpark draft head")
        return None
    try:
        ck = Exl3Checkpoint(str(loaded.model_path))
        if not any(k.startswith("mtp.") for k in ck.index):
            logger.info("[DSV41] checkpoint has no mtp.* groups: no draft head")
            return None
        head = exl3_build.build_mtp(
            ck, loaded.args, rank=loaded.rank, world=loaded.world, group=group
        )
    except Exception as e:  # noqa: BLE001 -- optional acceleration path
        logger.warning(
            f"[DSV41] DSpark draft head build failed ({type(e).__name__}: {e}); "
            "continuing with greedy decode only."
        )
        return None
    loaded.head = head
    logger.info(
        f"[DSV41] DSpark draft head attached "
        f"({len(head.stages)} stages, block={head.block_size}, "
        f"markov_rank={head.markov_rank}, taps={list(head.args.dspark_target_layer_ids)})"
    )
    return head


def _resolve_engram_dir(
    path: Path, args: Any, requested: list[int] | None
) -> Path | None:
    """Native release dir for the engram tables, when the build has engram layers.

    The EXL3 checkpoint has no engram tables; ``build_block`` reads them
    row-on-demand from the native release and raises without it.
    """
    engram_layers = set(getattr(args, "engram_layer_ids", ()) or ())
    if requested is not None:
        engram_layers &= set(requested)
    if not engram_layers:
        return None
    override = os.environ.get(_ENGRAM_DIR_ENV)
    native = Path(override).expanduser() if override else path.parent / _ENGRAM_DIR_NAME
    if not native.is_dir():
        raise FileNotFoundError(
            f"DSv4.1 layers {sorted(engram_layers)} need the native engram release "
            f"(not in the EXL3 checkpoint); expected it at {native}. Download "
            f"deepseek-ai/DeepSeek-V4.1-Flash engram tables there or set {_ENGRAM_DIR_ENV}."
        )
    logger.info(f"[DSV41] engram tables: {native}")
    return native


def _resolve_engram_token_map(
    path: Path, args: Any, tokenizer: TokenizerWrapper
) -> Path | None:
    """Resolve the engram token map when this build contains engram layers.

    A layer-subset build that excludes layers 1/14 needs neither the native
    engram release nor the token map; the model raises on a forward that needs
    one, so the engine resolves this at load time rather than at request time.
    """
    engram_layers = tuple(getattr(args, "engram_layer_ids", ()) or ())
    if not engram_layers:
        return None
    hf_tokenizer = getattr(tokenizer, "_tokenizer", tokenizer)
    return ensure_engram_token_map(path, hf_tokenizer)
