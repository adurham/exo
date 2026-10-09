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

import inspect
import os
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from exo.shared.types.events import RunnerStatusUpdated
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.shared.types.worker.runners import RunnerLoading
from exo.worker.engines.base import Engine
from exo.worker.engines.mlx.builder import MlxBuilder
from exo.worker.engines.mlx.dsv41.engine import Dsv41Engine
from exo.worker.engines.mlx.dsv41.load import Dsv41Loaded, build_draft_head, load_dsv41
from exo.worker.runner.bootstrap import logger

#: Set to ``0`` to serve plain greedy decode even when the checkpoint carries a
#: DSpark draft head (useful for A/B runs; the head is an accelerator, never a
#: correctness requirement).
_SPEC_ENV = "EXO_DSV41_SPECULATIVE"

#: Minimum spacing of the load-warmup liveness beats (same 15 s throttle as
#: ``Engine.prefill_heartbeat``; well inside the 45 s supervisor window).
_LOAD_HEARTBEAT_SECONDS = 15.0


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
    #: The DSv4.1 vision tower (~1 GB/rank), or None when disabled/absent.
    vision: Any | None = None

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
        # Compile every serving kernel shape now, with host-synced collectives,
        # before any real forward: without it the first forward's ~47 s compile
        # storm skews the ranks and trips the Metal watchdog (p114/p115).
        # The warmup emits fence-backed liveness beats (ROUND-Q1B): it is the
        # longest event-free stretch of the load and the supervisor's silence
        # watchdog otherwise kills it once it outlasts ~66 s.
        load_warmup(loaded, heartbeat=self._load_heartbeat(bound_instance, loaded))
        self.loaded_dsv41 = loaded
        self.vision = _load_vision(loaded)
        # ``MlxBuilder``'s fields are unused for this engine: there is no
        # mlx-lm model object and no BatchGenerator. Set the tokenizer anyway
        # so ``close()`` (inherited) has a consistent object to drop.
        self.tokenizer = loaded.tokenizer
        self.engine_kwargs = _engine_kwargs_from_instance(bound_instance)
        if self.vision is not None:
            self.engine_kwargs["vision_processor"] = self.vision

    def _load_heartbeat(
        self, bound_instance: BoundInstance, loaded: Dsv41Loaded
    ) -> Callable[[], None]:
        """Throttled liveness beat for the load-time warmup.

        Re-sends the runner's current status (``RunnerLoading``, all layers
        loaded: the load generator's last yield) so the supervisor's
        silence clock resets. Any event resets it (supervisor
        ``_forward_events``). Called from the warmup's per-``fence_every``-layer
        fence, i.e. only after layers of committed compute; a wedged
        collective blocks that fence and therefore still goes silent and is
        still killed. Throttled like ``Engine.prefill_heartbeat``.
        """
        total = len(getattr(loaded.model, "layers", []))
        status = RunnerLoading(layers_loaded=total, total_layers=total)
        runner_id = bound_instance.bound_runner_id
        # last=0.0: the FIRST fence beats unthrottled, so the clock is reset
        # as soon as the warmup has committed its first layers.
        state = {"last": 0.0, "beats": 0, "t0": time.monotonic()}

        def beat() -> None:
            state["beats"] += 1
            now = time.monotonic()
            if now - state["last"] < _LOAD_HEARTBEAT_SECONDS:
                return
            state["last"] = now
            logger.info(
                f"[DSV41] load warmup liveness: fence {state['beats']} at "
                f"{now - state['t0']:.1f}s (rank {loaded.rank}/{loaded.world})"
            )
            self.event_sender.send(
                RunnerStatusUpdated(runner_id=runner_id, runner_status=status)
            )

        return beat

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


def _load_vision(loaded: Dsv41Loaded) -> Any | None:
    """Load the checkpoint's vision tower so image requests are served.

    ``EXO_DSV41_VISION=0`` skips it (text-only; image requests are refused per
    request). A checkpoint without a tower also serves text only.
    """
    if os.environ.get("EXO_DSV41_VISION", "1") == "0" or loaded.model_path is None:
        logger.info("[DSV41] vision tower not loaded (EXO_DSV41_VISION=0)")
        return None
    from exo.worker.engines.mlx.dsv41.vision import load_dsv41_vision

    try:
        vision = load_dsv41_vision(loaded.model_path)
    except ValueError as e:
        logger.warning(f"[DSV41] no vision tower: {e}")
        return None
    mx.eval(vision.tower.parameters())
    logger.info(f"[DSV41] vision tower loaded: {vision}")
    return vision


def load_warmup(
    loaded: Dsv41Loaded, heartbeat: Callable[[], None] | None = None
) -> None:
    """Compile every serving kernel shape (body + draft head) before serving.

    ``heartbeat`` is forwarded as the mlx-lm warmup's ``fence_hook`` when the
    installed mlx-lm accepts it. An older mlx-lm without the parameter gets
    the old call (no beats) and a loud warning, since its warmup then has no
    liveness signal against the supervisor's silence watchdog.
    """
    from mlx_lm.models.deepseek_v41 import prefill as _prefill

    kwargs: dict[str, Any] = {}
    if heartbeat is not None:
        if "fence_hook" in inspect.signature(_prefill.load_warmup).parameters:
            kwargs["fence_hook"] = heartbeat
        else:
            logger.warning(
                "[DSV41] installed mlx-lm load_warmup has no fence_hook: the "
                "load warmup runs WITHOUT liveness beats (a warmup longer "
                "than the supervisor's silence window will be SIGKILLed)"
            )
    times = _prefill.load_warmup(loaded.model, loaded.head, **kwargs)
    logger.info(f"[DSV41] load warmup (kernel compile): {times}")


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
    # Per-chunk prefill transient budget (MB). Not an instance field, so it only
    # comes from the environment; absent => the engine's own default (2048 MB).
    # A malformed value falls back to the default rather than crashing a worker
    # at engine-construction time.
    transient_budget_mb = _env_int("EXO_PREFILL_TRANSIENT_BUDGET_MB")
    if transient_budget_mb is not None:
        kwargs["prefill_transient_budget_mb"] = transient_budget_mb
    return kwargs


def _env_int(name: str) -> int | None:
    """Parse an integer env var; ``None`` when unset, ``None`` when malformed."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"[DSV41] ignoring non-integer {name}={raw!r}")
        return None
