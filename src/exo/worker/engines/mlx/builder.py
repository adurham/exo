import contextlib
import os
from collections.abc import Generator
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.common import ModelId
from exo.shared.types.events import Event
from exo.shared.types.tasks import TaskId
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines.base import Builder, Engine
from exo.worker.runner.bootstrap import logger
from exo.worker.runner.llm_inference.batch_generator import (
    BatchGenerator,
    SequentialGenerator,
)
from exo.worker.runner.llm_inference.tool_parsers import make_mlx_parser

from .cache import KVPrefixCache
from .types import Model
from .utils_mlx import (
    initialize_mlx,
    load_mlx_items,
)
from .vision import VisionProcessor


@dataclass
class MlxBuilder(Builder):
    model_id: ModelId
    event_sender: MpSender[Event]
    cancel_receiver: MpReceiver[TaskId]
    inference_model: Model | None = None
    tokenizer: TokenizerWrapper | None = None
    group: mx.distributed.Group | None = None
    vision_processor: VisionProcessor | None = None
    # Captured during connect/load so build() can read instance-level
    # caps (max_prefix_sessions, max_prefix_bytes, max_kv_tokens,
    # kv_cache_bits) when constructing the prefix cache.
    bound_instance: BoundInstance | None = None

    def connect(self, bound_instance: BoundInstance) -> None:
        self.group = initialize_mlx(bound_instance)
        self.bound_instance = bound_instance
        self._probe_data_path()

    def _probe_data_path(self) -> None:
        """Exercise the RDMA data path end-to-end BEFORE the model load.

        On this hardware a freshly established jaccl QP pair intermittently
        comes up with the UC data path silently dead (all_recv=0 on both
        ranks for the full reliable-deadline window; TCP side-channel fine;
        observed repeatedly on 2026-07-06, including on a freshly rebooted
        OS). When that happens, the failure previously surfaced only in
        WARMUP — after a multi-minute model load — so every dice roll on QP
        setup cost a full load cycle. Probe here with two collectives that
        mirror warmup's failing shapes (1-chunk tiny and ~128 KB multi-chunk
        at sz=2); on a reliable-deadline throw, group.reconnect() re-creates
        the QPs and the probe retries — each roll costs at most ~2×15 s and
        no load. Both ranks run connect() together, and observed failures
        are symmetric (both deadline), so both reach reconnect()'s
        coordinator barrier together.
        """
        group = self.group
        if group is None or group.size() < 2:
            return
        attempts = int(os.environ.get("EXO_JACCL_CONNECT_PROBE_ATTEMPTS", "5"))
        for attempt in range(attempts + 1):
            try:
                small = mx.distributed.all_sum(mx.array([1.0]), group=group)
                mx.eval(small)
                large = mx.distributed.all_sum(
                    mx.zeros((32768,), dtype=mx.float32), group=group
                )
                mx.eval(large)
                if attempt > 0:
                    logger.info(
                        f"jaccl data-path probe passed after {attempt} reconnect(s)"
                    )
                return
            except Exception as probe_err:
                if attempt >= attempts or not hasattr(group, "reconnect"):
                    raise
                logger.warning(
                    f"jaccl data-path probe failed "
                    f"(attempt {attempt + 1}/{attempts + 1}): {probe_err}. "
                    "Reconnecting QPs and retrying."
                )
                group.reconnect()  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]

    def load(self, bound_instance: BoundInstance) -> Generator[ModelLoadingResponse]:
        self.bound_instance = bound_instance
        (
            self.inference_model,
            self.tokenizer,
            self.vision_processor,
        ) = yield from load_mlx_items(bound_instance, self.group)

    def close(self) -> None:
        with contextlib.suppress(NameError, AttributeError):
            del self.inference_model
        with contextlib.suppress(NameError, AttributeError):
            del self.tokenizer
        with contextlib.suppress(NameError, AttributeError):
            del self.group

    def build(
        self,
    ) -> Engine:
        assert self.inference_model
        assert self.tokenizer

        vision_processor = self.vision_processor

        tool_parser = None
        logger.info(
            f"model has_tool_calling={self.tokenizer.has_tool_calling} using tokens {self.tokenizer.tool_call_start}, {self.tokenizer.tool_call_end}"
        )
        if (
            self.tokenizer.tool_call_start
            and self.tokenizer.tool_call_end
            and self.tokenizer.tool_parser  # type: ignore
        ):
            tool_parser = make_mlx_parser(
                self.tokenizer.tool_call_start,
                self.tokenizer.tool_call_end,
                self.tokenizer.tool_parser,  # type: ignore
            )

        # Plumb instance-level caps to the prefix cache. Without this
        # max_sessions stayed None and the trie grew unbounded — the
        # eviction code at cache.py:_evict_if_needed only fires on a
        # cap, so a missing cap meant 100% no-op. Was the root cause
        # of repeated OOM wedges as Hermes accumulated leaves across
        # turns.
        instance = self.bound_instance.instance if self.bound_instance else None
        kv_prefix_cache = KVPrefixCache(
            self.group,
            max_sessions=getattr(instance, "max_prefix_sessions", None),
            max_bytes=getattr(instance, "max_prefix_bytes", None),
            max_kv_tokens=getattr(instance, "max_kv_tokens", None),
            kv_cache_bits=getattr(instance, "kv_cache_bits", None),
        )

        device_rank = 0 if self.group is None else self.group.rank()
        if os.environ.get("EXO_NO_BATCH"):
            logger.info("using SequentialGenerator (batching disabled)")
            return SequentialGenerator(
                model=self.inference_model,
                tokenizer=self.tokenizer,
                group=self.group,
                tool_parser=tool_parser,
                kv_prefix_cache=kv_prefix_cache,
                model_id=self.model_id,
                device_rank=device_rank,
                cancel_receiver=self.cancel_receiver,
                event_sender=self.event_sender,
                vision_processor=vision_processor,
            )
        else:
            logger.info("using BatchGenerator")
            return BatchGenerator(
                model=self.inference_model,
                tokenizer=self.tokenizer,
                group=self.group,
                tool_parser=tool_parser,
                kv_prefix_cache=kv_prefix_cache,
                model_id=self.model_id,
                device_rank=device_rank,
                cancel_receiver=self.cancel_receiver,
                event_sender=self.event_sender,
                vision_processor=vision_processor,
            )
