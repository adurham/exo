# pyright: reportPrivateUsage=false
"""Adapter translating ``BatchedDecodeSession``'s raw per-step output
into the same ``GenerationBatch.Response`` shape ``ExoBatchGenerator``
already consumes from mlx-lm's own ``GenerationBatch``/from
``_step_pp_spec``'s hand-built responses.

Per a `consult` review (2026-08-05) before this integration step: this
module is a DELIBERATELY THIN translation layer -- it does NO
distributed communication and holds no model/cache state of its own.
Critically, it does NOT reimplement `finish_reason` semantics from
scratch; it mirrors `_step_pp_spec`'s own EOS/max-tokens decision
logic exactly (real EOS-id set membership test, real
`StopIteration`-equivalent max-tokens check via an explicit
``max_tokens`` counter this module tracks per request) so the two
paths can never silently drift apart on what counts as "done" and
why. Stop-STRING matching, incremental detokenization, and tool-call
parsing are explicitly NOT this module's concern -- ``ExoBatchGenerator``
already has `apply_all_parsers`/`map_responses_to_chunks` downstream of
`GenerationBatch.Response` construction handling exactly that, for
BOTH the existing serial and pp_spec paths; this module only needs to
produce a Response with a correct `finish_reason` for that existing
downstream pipeline to keep working unmodified.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from exo.worker.engines.mlx.pp_batched_decode_runtime import BatchedDecodeSession


@dataclass
class BatchedDecodeResponseAdapter:
    """Wraps ONE ``BatchedDecodeSession`` and tracks the minimal
    per-request bookkeeping (EOS id set, max_tokens counters) needed
    to translate its raw ``{request_id: (token, is_done)}`` step
    output into real ``GenerationBatch.Response`` objects -- mirrors
    ``_step_pp_spec``'s own EOS-membership-test /
    tokens-generated-vs-max_tokens decision logic exactly, not a
    reimplementation.

    Deliberately does NOT wrap ``RankOneMirrorSession`` -- rank 1
    never produces a ``GenerationBatch.Response`` (it has nothing to
    report to the client; matches ``RankOneMirrorSession``'s own
    zero-decision-logic design and ``ExoBatchGenerator``'s existing
    "results only flow from rank 0" convention for the pp_spec path
    too).
    """

    session: BatchedDecodeSession
    eos_ids: frozenset[int]
    _tokens_generated: dict[int, int] = field(default_factory=dict)
    _max_tokens: dict[int, int] = field(default_factory=dict)

    def admit(
        self,
        request_id: int,
        cache_slot: int,
        prefilled_cache: object,
        initial_token: int,
        sampler: object,
        max_tokens: int,
    ) -> tuple[object, "AdmitResponse"]:
        """Admit a new request, returning (a) the real
        ``StepMessage`` the caller must send to rank 1 (identical
        contract to ``BatchedDecodeSession.admit_request`` -- this
        adapter does not change that part at all) and (b) the FIRST
        response for this request (the token its prefill already
        produced), already correctly classified against EOS/
        max_tokens=1 edge cases exactly like ``_step_pp_spec``'s own
        first-token branch does.
        """
        from typing import cast

        from exo.worker.engines.mlx.pp_batched_decode_runtime import Sampler
        from exo.worker.engines.mlx.types import KVCacheType

        message = self.session.admit_request(
            request_id=request_id,
            cache_slot=cache_slot,
            prefilled_cache=cast(KVCacheType, prefilled_cache),
            initial_token=initial_token,
            sampler=cast(Sampler, sampler),
        )
        self._tokens_generated[request_id] = 1
        self._max_tokens[request_id] = max_tokens

        is_eos = initial_token in self.eos_ids
        is_length = self._tokens_generated[request_id] >= max_tokens
        finish_reason: str | None = None
        if is_eos:
            finish_reason = "stop"
        elif is_length:
            finish_reason = "length"

        return message, AdmitResponse(token=initial_token, finish_reason=finish_reason)

    def classify_step_results(
        self, step_results: dict[int, tuple[int, bool]]
    ) -> dict[int, "StepResponse"]:
        """Given ``BatchedDecodeSession.finish_step``'s own raw
        ``{request_id: (new_token, is_done)}`` output (the SAME dict,
        never re-derived), classify each request's real
        ``finish_reason`` -- mirrors ``_step_pp_spec``'s steady-state
        branch's own EOS-membership test and
        tokens-vs-max_tokens check exactly. The ``is_done`` bool
        ``finish_step`` returns is currently always ``False`` (that
        module's own docstring: stop-condition policy is a caller
        concern) -- THIS adapter is that caller, and is the thing
        that actually implements the policy, exactly matching
        `_step_pp_spec`'s.
        """
        results: dict[int, StepResponse] = {}
        for request_id, (token, _is_done_stub) in step_results.items():
            if request_id not in self._tokens_generated:
                raise BatchedDecodeAdapterError(
                    f"classify_step_results: request_id={request_id} was "
                    f"never admitted via this adapter's admit() -- cannot "
                    f"classify a request this adapter has no bookkeeping "
                    f"for"
                )
            self._tokens_generated[request_id] += 1
            is_eos = token in self.eos_ids
            is_length = (
                self._tokens_generated[request_id] >= self._max_tokens[request_id]
            )
            finish_reason: str | None = None
            if is_eos:
                finish_reason = "stop"
            elif is_length:
                finish_reason = "length"
            results[request_id] = StepResponse(token=token, finish_reason=finish_reason)
        return results

    def forget(self, request_id: int) -> None:
        """Drop this adapter's own per-request bookkeeping for
        ``request_id`` -- callers invoke this AFTER the request's
        final response has been consumed (whether via eviction or
        natural completion), matching
        ``BatchedDecodeSession.on_evict_ack``'s own
        remove-per-request-state timing."""
        self._tokens_generated.pop(request_id, None)
        self._max_tokens.pop(request_id, None)


class BatchedDecodeAdapterError(RuntimeError):
    """Raised when this adapter is asked to classify a request it has
    no bookkeeping for -- fail-stop, matching this whole session's
    established discipline (see ``pp_scheduler_protocol.py``'s module
    docstring point 3) rather than silently guessing a finish_reason
    for an unknown request."""


@dataclass(frozen=True)
class AdmitResponse:
    """First response for a newly-admitted request (the token its
    prefill pass already produced), classified exactly like
    ``StepResponse`` below."""

    token: int
    finish_reason: str | None


@dataclass(frozen=True)
class StepResponse:
    """One request's classified result for a single batched decode
    step -- ``finish_reason`` is ``None`` (still generating), "stop"
    (real EOS token id, membership-tested against the SAME
    ``eos_ids_from_tokenizer(self.tokenizer)`` set every other path in
    ``ExoBatchGenerator`` uses), or "length" (this request has now
    generated >= its own ``max_tokens``)."""

    token: int
    finish_reason: str | None
