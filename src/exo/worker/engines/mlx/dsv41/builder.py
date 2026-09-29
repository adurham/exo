"""The exo ``Builder`` for DeepSeek-V4.1 (EXL3).

Subclasses :class:`~exo.worker.engines.mlx.builder.MlxBuilder` on purpose: the
MLX bring-up it inherits is exactly what this engine needs and nothing it
does not -- ``initialize_mlx`` establishes the instance's distributed group,
and the jaccl data-path probe runs BEFORE a multi-minute load (that probe
exists because a freshly established QP pair can come up with a dead UC data
path; paying for it before the load rather than inside warmup is not specific
to any model).

What is overridden is only what is genuinely different:

* ``load``  -- the EXL3 checkpoint is built by the mlx-lm fork's
  ``exl3_build.build_model`` (see ``dsv41.load``); it cannot go through
  ``mlx_lm.utils.load_model`` or exo's generic post-load machinery, and the
  DSpark draft head is attached here too.
* ``build`` -- the engine is :class:`~exo.worker.engines.mlx.dsv41.engine.Dsv41Engine`,
  which serves one request at a time and owns its own cache; the batched
  generators (``BatchGenerator``/``SequentialGenerator``) and
  ``KVPrefixCache`` are deliberately not involved.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.worker.engines.base import Engine
from exo.worker.engines.mlx.builder import MlxBuilder
from exo.worker.engines.mlx.dsv41.engine import Dsv41Engine
from exo.worker.engines.mlx.dsv41.load import Dsv41Loaded, build_draft_head, load_dsv41
from exo.worker.runner.bootstrap import logger

#: Set to ``0`` to serve plain greedy decode even when the checkpoint carries a
#: DSpark draft head (useful for A/B runs; the head is an accelerator, never a
#: correctness requirement).
_SPEC_ENV = "EXO_DSV41_SPECULATIVE"


def speculation_enabled() -> bool:
    return os.environ.get(_SPEC_ENV, "1") != "0"


@dataclass
class Dsv41Builder(MlxBuilder):
    """Loads the DSv4.1 EXL3 checkpoint and builds the DSv4.1 engine."""

    loaded_dsv41: Dsv41Loaded | None = None
    #: Instance-level caps captured by ``connect``; the engine takes them as
    #: constructor arguments (its own dataclass fields) rather than reading
    #: ``bound_instance`` at request time.
    engine_kwargs: dict[str, Any] = field(default_factory=dict)

    def load(self, bound_instance: BoundInstance) -> Generator[ModelLoadingResponse]:
        self.bound_instance = bound_instance
        shard = bound_instance.bound_shard
        loaded: Dsv41Loaded = yield from load_dsv41(
            shard, shard.model_card, group=self.group
        )
        if speculation_enabled():
            build_draft_head(loaded, self.group)
        else:
            logger.info(
                f"[DSV41] {_SPEC_ENV}=0: skipping the DSpark draft head "
                "(plain greedy decode)."
            )
        self.loaded_dsv41 = loaded
        # ``MlxBuilder``'s fields are unused for this engine: there is no
        # mlx-lm model object and no BatchGenerator. Set the tokenizer anyway
        # so ``close()`` (inherited) has a consistent object to drop.
        self.tokenizer = loaded.tokenizer
        self.engine_kwargs = _engine_kwargs_from_instance(bound_instance)

    def build(self) -> Engine:
        loaded = self.loaded_dsv41
        assert loaded is not None, "Dsv41Builder.build() before load()"
        engine = Dsv41Engine(
            loaded=loaded,
            model_id=self.model_id,
            group=self.group,
            cancel_receiver=self.cancel_receiver,
            event_sender=self.event_sender,
            device_rank=0 if self.group is None else self.group.rank(),
            speculative=speculation_enabled(),
            **self.engine_kwargs,
        )
        logger.info(
            f"[DSV41] engine built: {len(loaded.built_layers)}/{len(loaded.model.layers)} "
            f"layers, rank {loaded.rank}/{loaded.world}, "
            f"speculative={engine.speculative} (gamma={engine.gamma}), "
            f"prefill chunk {engine._chunk} tokens."
        )
        return engine


def _engine_kwargs_from_instance(bound_instance: BoundInstance) -> dict[str, Any]:
    """Instance-level caps for the engine's constructor.

    Same source as the MLX path's prefix-cache caps: ``BoundInstance.instance``
    carries ``max_kv_tokens`` and ``prefill_step_size`` when placement (or the
    operator) set them. Anything absent stays unset so the engine's own
    defaults (checkpoint ``max_seq_len``, 512-token prefill chunks) apply.
    """
    instance = bound_instance.instance
    kwargs: dict[str, Any] = {}
    max_kv_tokens = getattr(instance, "max_kv_tokens", None)
    if max_kv_tokens is not None:
        kwargs["max_kv_tokens"] = int(max_kv_tokens)
    prefill_step_size = getattr(instance, "prefill_step_size", None)
    if prefill_step_size is None:
        env_step = os.environ.get("EXO_PREFILL_STEP_SIZE")
        prefill_step_size = int(env_step) if env_step else None
    if prefill_step_size is not None:
        kwargs["prefill_chunk_size"] = int(prefill_step_size)
    return kwargs
