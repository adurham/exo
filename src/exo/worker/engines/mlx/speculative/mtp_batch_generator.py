#!/usr/bin/env python3
"""MTP Speculative Decoding integrated with mlx_lm's BatchGenerator.

Subclasses ``BatchGenerator`` against the post-rewrite mlx-lm API
(``PromptProcessingBatch`` / ``GenerationBatch`` / ``SequenceStateMachine``)
to add MTP drafting + S>1 verification with correct GDN state rollback
via :class:`SpeculativeArraysCache`.

At BS=1 with no pending prompts: drafts γ tokens with MTP, verifies at
S=γ+1 in a single forward pass, buffers accepted tokens, and yields one
per ``next()`` call. At BS>1 or while prompts are still being prefilled,
falls through to the parent generator (no speculative).
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

import mlx.core as mx
from mlx_lm.generate import BatchGenerator, GenerationBatch, PromptProcessingBatch

from .mtp_module import MTPPredictor, draft_tokens, speculative_forward

if TYPE_CHECKING:
    from collections.abc import Sequence


class MTPBatchGenerator(BatchGenerator):
    """``BatchGenerator`` with MTP speculative decoding for BS=1."""

    def __init__(
        self,
        model: Any,
        mtp_predictor: MTPPredictor,
        gamma: int = 2,
        temp: float = 0.0,
        alpha: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        self.mtp: MTPPredictor = mtp_predictor
        self.gamma: int = gamma
        self.temp: float = temp
        self.alpha: float = alpha

        self._token_buffer: dict[int, deque[GenerationBatch.Response]] = {}
        self._captured: dict[str, mx.array] = {}
        self._mtp_pre_norm: dict[int, mx.array] = {}
        self._mtp_prefilled: set[int] = set()
        self._request_temp: dict[int, float] = {}

        self._setup_hidden_capture()

    # ── hidden-state capture ───────────────────────────────────────────

    def _setup_hidden_capture(self) -> None:
        """Wrap the model's final RMSNorm to stash the pre-norm hidden state.

        Captures the **last** hidden states fed into the final norm during
        every model forward; the prefill case (S>1) is also stashed under
        ``prompt_pre_norm`` so the caller's ``submit()`` path can build the
        MTP cache from it before generation starts.
        """
        inner = getattr(self.model, "model", None) or self.model.language_model.model
        original_norm = inner.norm
        captured = self._captured

        class _CapturingNorm:
            def __init__(self, orig: Any) -> None:
                self._orig = orig
                self.weight = orig.weight

            def __call__(self, x: mx.array) -> mx.array:
                captured["pre_norm"] = x
                if x.shape[1] > 1:
                    captured["prompt_pre_norm"] = x
                return self._orig(x)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._orig, name)

        inner.norm = _CapturingNorm(original_norm)

    # ── per-uid cleanup ────────────────────────────────────────────────

    def _cleanup_uid(self, uid: int) -> None:
        self._mtp_pre_norm.pop(uid, None)
        self._mtp_prefilled.discard(uid)
        self._token_buffer.pop(uid, None)
        self._request_temp.pop(uid, None)

    def remove(
        self, uids: "Sequence[int]", return_prompt_caches: bool = False
    ) -> dict[int, Any]:
        result = super().remove(uids, return_prompt_caches=return_prompt_caches)
        for uid in uids:
            self._cleanup_uid(uid)
        return result

    # ── _next dispatch ─────────────────────────────────────────────────

    def _next(
        self,
    ) -> tuple[list[PromptProcessingBatch.Response], list[GenerationBatch.Response]]:
        gen_batch = self._generation_batch

        # Buffered tokens from a previous spec cycle have priority — drain
        # them one per call, mirroring the API contract that ``next()`` only
        # produces a single token per sequence per invocation.
        if len(gen_batch) == 1:
            uid = gen_batch.uids[0]
            if uid in self._token_buffer and self._token_buffer[uid]:
                return [], self._yield_buffered(uid)

        spec_eligible = (
            self.gamma > 0
            and len(gen_batch) == 1
            and len(self._prompt_batch) == 0
            and len(self._unprocessed_sequences) == 0
        )
        if spec_eligible:
            uid = gen_batch.uids[0]
            if uid not in self._mtp_prefilled:
                return self._first_step_and_capture(uid)
            return [], self._speculative_next(uid)

        # Standard path. Any uid that survives a non-spec step had its
        # ``pre_norm`` capture polluted (BS>1 forward, prompt processing,
        # etc.) — drop the prefilled flag so the next BS=1 idle window
        # re-captures from a clean BS=1 forward before drafting.
        result = super()._next()
        for uid in self._generation_batch.uids:
            self._mtp_prefilled.discard(uid)
        return result

    # ── single decode step that also captures pre_norm ────────────────

    def _first_step_and_capture(
        self, uid: int
    ) -> tuple[list[PromptProcessingBatch.Response], list[GenerationBatch.Response]]:
        prompt_responses, generation_responses = super()._next()
        if not generation_responses:
            return prompt_responses, generation_responses

        decode_pre_norm = self._captured.get("pre_norm")
        if decode_pre_norm is not None and decode_pre_norm.shape[0] == 1:
            mx.eval(decode_pre_norm)
            self._mtp_pre_norm[uid] = decode_pre_norm[:, -1:, :]
            self._mtp_prefilled.add(uid)
        return prompt_responses, generation_responses

    # ── buffer drain ───────────────────────────────────────────────────

    def _yield_buffered(self, uid: int) -> list[GenerationBatch.Response]:
        buf = self._token_buffer[uid]
        response = buf.popleft()
        if not buf:
            del self._token_buffer[uid]

        if response.finish_reason is not None:
            self._filter_finished_uid(uid)
            self._cleanup_uid(uid)

        return [response]

    def _filter_finished_uid(self, uid: int) -> None:
        gen_batch = self._generation_batch
        if uid not in gen_batch.uids:
            return
        idx = gen_batch.uids.index(uid)
        keep = [i for i in range(len(gen_batch)) if i != idx]
        gen_batch.filter(keep)

    # ── speculative cycle ──────────────────────────────────────────────

    def _speculative_next(self, uid: int) -> list[GenerationBatch.Response]:
        """One verify/accept cycle. Returns the first yielded response;
        the rest land in ``self._token_buffer[uid]`` for subsequent calls.
        """
        gen_batch = self._generation_batch
        idx = gen_batch.uids.index(uid)

        # ``_next_tokens`` holds the token sampled during the previous
        # ``_step()`` call — i.e. the next token to yield. Without it we
        # have no candidate for the verify-cycle prefix; defer to the
        # parent's generation step.
        y = gen_batch._next_tokens
        if y is None or not gen_batch._next_logprobs:
            return gen_batch.next()

        pre_norm = self._mtp_pre_norm.get(uid)
        if pre_norm is None:
            return gen_batch.next()

        y_val = int(y[0].item())
        y_logprobs = gen_batch._next_logprobs[0]

        gamma = self.gamma
        temp = self._request_temp.get(uid, self.temp)
        alpha = self.alpha

        # 1. Draft γ tokens via MTP (lazy chain — no eval here)
        next_token_arr = y.reshape(1, 1)
        draft_ids, draft_probs = draft_tokens(
            self.mtp, pre_norm, next_token_arr, gamma, temp
        )

        # 2. Verify forward over [y, draft_0, ..., draft_{γ-1}]
        draft_concat = mx.concatenate(
            [d.reshape(1, 1) for d in draft_ids], axis=1
        )  # (1, γ)
        verify_input = mx.concatenate(
            [next_token_arr, draft_concat], axis=1
        )  # (1, γ+1)
        verify_pre_norm, verify_logits = speculative_forward(
            self.model, verify_input, gen_batch.prompt_cache, speculative=True
        )

        # 3. Build acceptance check (lazy — single async_eval below)
        target_tokens = mx.argmax(verify_logits[:, :gamma, :], axis=-1)

        accept_ratios: list[mx.array] = []
        uniforms: mx.array | None = None
        corrections: list[mx.array] = []
        bonus_token: mx.array | None = None
        matches: mx.array | None = None
        all_next: mx.array | None = None

        if temp == 0:
            matches = mx.equal(target_tokens, draft_concat).squeeze(0)
            all_next = mx.argmax(verify_logits[0], axis=-1)
            logprobs_all = verify_logits[0] - mx.logsumexp(
                verify_logits[0], axis=-1, keepdims=True
            )
            mx.async_eval(matches, all_next, logprobs_all, verify_pre_norm)
        else:
            for i in range(gamma):
                p = mx.softmax(verify_logits[0, i] / temp, axis=-1)
                q = draft_probs[i]
                p_di = p[draft_ids[i].squeeze()]
                q_di = q[0, draft_ids[i].squeeze()]
                ratio = p_di / mx.maximum(q_di, 1e-10)
                accept_ratios.append(mx.minimum(ratio**alpha, 1.0))
            uniforms = mx.random.uniform(shape=(gamma,))
            for i in range(gamma):
                p = mx.softmax(verify_logits[0, i] / temp, axis=-1)
                q = draft_probs[i][0]
                residual = mx.maximum(p - q, 0.0)
                corrections.append(mx.random.categorical(mx.log(residual + 1e-10)))
            bonus_token = mx.random.categorical(verify_logits[0, gamma] * (1.0 / temp))
            logprobs_all = verify_logits[0] - mx.logsumexp(
                verify_logits[0], axis=-1, keepdims=True
            )
            mx.async_eval(
                accept_ratios,
                uniforms,
                corrections,
                bonus_token,
                logprobs_all,
                verify_pre_norm,
                draft_concat,
            )

        # 4. Determine acceptance
        n_accepted = 0
        for i in range(gamma):
            if temp == 0:
                assert matches is not None
                if matches[i].item():
                    n_accepted += 1
                else:
                    break
            else:
                assert uniforms is not None
                if uniforms[i].item() < accept_ratios[i].item():
                    n_accepted += 1
                else:
                    break

        # 5. Roll back GDN cache for rejected drafts
        rollback = gamma - n_accepted
        if rollback > 0:
            for c in gen_batch.prompt_cache:
                if hasattr(c, "offset"):
                    c.offset -= rollback
                elif hasattr(c, "rollback"):
                    c.rollback(n_accepted)

        # Unwrap SpeculativeArraysCache wrappers so the next step sees the
        # plain ArraysCache again.
        for i, c in enumerate(gen_batch.prompt_cache):
            if hasattr(c, "base"):
                gen_batch.prompt_cache[i] = c.base

        # 6. Compute bonus token + logprobs
        if n_accepted == gamma:
            if temp == 0:
                assert all_next is not None
                bonus_val = int(all_next[gamma].item())
            else:
                assert bonus_token is not None
                bonus_val = int(bonus_token.item())
            bonus_lp = logprobs_all[gamma]
        else:
            if temp == 0:
                assert all_next is not None
                bonus_val = int(all_next[n_accepted].item())
            else:
                bonus_val = int(corrections[n_accepted].item())
            bonus_lp = logprobs_all[n_accepted]

        # 7. Update MTP pre_norm for next cycle
        pos = gamma if n_accepted == gamma else n_accepted
        self._mtp_pre_norm[uid] = verify_pre_norm[:, pos : pos + 1, :]

        # 8. Build all_tokens = [y, draft_0, ..., draft_{n_accepted-1}]
        draft_int_values = [int(v) for v in draft_concat[0].tolist()]
        all_tokens: list[tuple[int, mx.array]] = [(y_val, y_logprobs)]
        for i in range(n_accepted):
            all_tokens.append((draft_int_values[i], logprobs_all[i]))

        # 9. Update _next_tokens to the bonus for the next call
        gen_batch._next_tokens = mx.array([bonus_val])
        gen_batch._next_logprobs = [bonus_lp]
        mx.async_eval(gen_batch._next_tokens)

        # 10. Bookkeeping the parent normally does in _next()
        self._gen_tokens_counter += len(all_tokens)
        self._steps_counter += 1
        if self._steps_counter % 512 == 0:
            mx.clear_cache()

        # 11. Apply state machine + length checks per token, build responses
        responses = self._build_yielded_responses(uid, idx, all_tokens)

        first = responses[0]
        rest = responses[1:]
        if rest:
            self._token_buffer[uid] = deque(rest)
        elif first.finish_reason is not None:
            self._filter_finished_uid(uid)
            self._cleanup_uid(uid)

        return [first]

    def _build_yielded_responses(
        self,
        uid: int,
        idx: int,
        all_tokens: list[tuple[int, mx.array]],
    ) -> list[GenerationBatch.Response]:
        """Run state machine + length checks per token; build a Response each.

        Stops at the first ``finish_reason``-bearing token; tokens after a
        stop / length cutoff are discarded the same way the parent
        ``GenerationBatch.next()`` would discard them.
        """
        gen_batch = self._generation_batch
        state_machine = gen_batch.state_machines[idx]
        max_tokens_limit = gen_batch.max_tokens[idx]

        responses: list[GenerationBatch.Response] = []
        for token_int, logprob in all_tokens:
            gen_batch._num_tokens[idx] += 1
            finish_reason: str | None = None
            if gen_batch._num_tokens[idx] >= max_tokens_limit:
                finish_reason = "length"

            (
                gen_batch._matcher_states[idx],
                match_sequence,
                current_state,
            ) = state_machine.match(gen_batch._matcher_states[idx], token_int)
            if match_sequence is not None and current_state is None:
                finish_reason = "stop"

            gen_batch.tokens[idx].append(token_int)

            prompt_cache: list[Any] | None = None
            all_tokens_full: list[int] | None = None
            if finish_reason is not None:
                prompt_cache = gen_batch.extract_cache(idx)
                all_tokens_full = gen_batch.tokens[idx]

            responses.append(
                GenerationBatch.Response(
                    uid=uid,
                    token=token_int,
                    logprobs=logprob,
                    finish_reason=finish_reason,
                    current_state=current_state,
                    match_sequence=match_sequence,
                    prompt_cache=prompt_cache,
                    all_tokens=all_tokens_full,
                )
            )

            if finish_reason is not None:
                break

        return responses
