"""DSv4-specific MTP self-speculative decode glue.

Sits alongside the Qwen3.5-flavored ``mtp_module.py`` /
``mtp_batch_generator.py`` and reuses their machinery (the
``draft_tokens`` chain helper, the BatchGenerator scaffolding,
the per-uid pre-norm capture). The DSv4-specific pieces are:

* :class:`DSv4MTPPredictor` — thin wrapper around the model's
  pre-loaded ``model.model.mtp[mtp_idx]`` module. Unlike Qwen3.5's
  :class:`~.mtp_module.MTPPredictor`, weights are NOT loaded
  dynamically here; they live on the model already (via the
  modified ``deepseek_v4.sanitize`` and ``DeepseekV4Model.mtp``
  ModuleList). The predictor only adapts the call signature to
  the one ``draft_tokens`` expects.
* :func:`dsv4_speculative_forward` — verify-pass forward through
  the target. DSv4 uses standard KV caches (RotatingKVCache +
  PoolingCache wrapped in CacheList) with rollback via ``trim()``,
  so no GDN bookkeeping is needed; this is a thin wrapper over
  the model's ``__call__`` that hands back the captured pre-final-
  norm hidden via the ``MTPBatchGenerator``'s wrapped-norm side
  channel.
* :class:`DSv4MTPBatchGenerator` — overrides ``_speculative_next``
  to use the DSv4 helpers. Inherits per-uid pre-norm capture,
  token buffering, and BS=1 dispatch logic from the base
  :class:`~.mtp_batch_generator.MTPBatchGenerator`.

BS>1 batched-MTP is NOT yet enabled here — that's Phase 5.
"""
from __future__ import annotations

import logging
import os
from collections import deque
from typing import Any, Optional, Sequence

import mlx.core as mx

from .mtp_batch_generator import MTPBatchGenerator
from .mtp_module import draft_tokens

logger = logging.getLogger(__name__)

# Log acceptance distribution every N spec cycles when EXO_DSV4_MTP_LOG=1.
# Set to 0 / unset to silence.
_LOG_INTERVAL = int(os.environ.get("EXO_DSV4_MTP_LOG_INTERVAL", "0"))


class DSv4MTPPredictor:
    """Thin wrapper around ``model.model.mtp[mtp_idx]`` that exposes the
    ``predict`` API expected by :func:`mtp_module.draft_tokens`.

    Unlike Qwen3.5's ``MTPPredictor`` (which builds the MTP module
    dynamically from a separate weight file), DSv4's MTP module is
    already part of the loaded model object — it gets sharded by
    :class:`DeepseekV4ShardingStrategy.shard_model` alongside the main
    layers, and its weights come in via the main checkpoint (with
    ``deepseek_v4.sanitize`` keeping ``mtp.*`` after the patch landed
    in mlx-lm@<commit>). Construction here is essentially free.
    """

    def __init__(self, model: Any, mtp_idx: int = 0) -> None:
        self.model = model
        # For DSv4: outer `Model` owns `lm_head`; inner `model.model`
        # (DeepseekV4Model) owns `embed_tokens`, `norm`, and `mtp`.
        # Don't conflate the two — Qwen3.5's MTPPredictor used the
        # same name for both because that model's lm_head lives on
        # the inner model.
        inner = getattr(model, "model", None) or model.language_model.model
        self._inner = inner
        self.embed_tokens = inner.embed_tokens
        self.final_norm = inner.norm
        # lm_head: try outer model first (DSv4 layout), fall back to
        # inner / language_model (Qwen3.5-style layouts).
        self.lm_head = (
            getattr(model, "lm_head", None)
            or getattr(inner, "lm_head", None)
            or getattr(getattr(model, "language_model", None), "lm_head", None)
        )
        if self.lm_head is None:
            raise RuntimeError(
                "DSv4MTPPredictor: could not locate lm_head on the model. "
                "Checked model.lm_head, model.model.lm_head, model.language_model.lm_head."
            )

        if not hasattr(inner, "mtp") or len(inner.mtp) <= mtp_idx:
            raise RuntimeError(
                f"Model has no MTP module at index {mtp_idx}. Checkpoint "
                f"may be missing mtp.* weights, or num_nextn_predict_layers "
                f"is set to 0 in ModelArgs."
            )
        self.mtp_module = inner.mtp[mtp_idx]
        self._cache: Optional[Any] = None

    def reset_cache(self) -> None:
        """Reset the MTP attention's KV cache. Call when the target
        model's prompt cache resets or when starting a new sequence.
        """
        self._cache = self.mtp_module.make_cache()

    def predict(
        self,
        hidden_state: mx.array,
        token_ids: mx.array,
        return_hidden: bool = False,
        draft_mode: bool = False,
    ) -> Any:
        """Match the call signature of :meth:`MTPPredictor.predict`.

        Args:
            hidden_state: ``(B, S, D)`` post-hc_head, pre-final-norm
                hidden state captured from the target model (or from a
                previous chained MTP step).
            token_ids: ``(B, S)`` int — the token sampled at each
                position whose logits we want to predict the successor
                of.
            return_hidden: if True, also return the post-MTP-block,
                pre-final-norm hidden state for the next chained step.
            draft_mode: ignored for DSv4 (no truncated lm_head shortcut
                yet — the gain on the cluster is small and adds quant
                bookkeeping; can be added later if profiling justifies).

        Returns:
            ``logits`` (B, S, vocab) when ``return_hidden=False``,
            ``(logits, hidden)`` when True. If ``S == 1`` the returned
            logits are squeezed to ``(B, vocab)`` to match the Qwen3.5
            convention.
        """
        del draft_mode  # not yet implemented for DSv4

        if token_ids.ndim == 1:
            token_ids = token_ids.reshape(1, -1)
        B, S = token_ids.shape
        if hidden_state.ndim == 2:
            hidden_state = hidden_state.reshape(B, S, -1)

        # The MTP cache is persistent across requests in the predictor
        # instance. If batch size changes (request set added / removed
        # / c=1↔c=2 transition), the cached KV state is stale and would
        # crash a concatenate at the next attention update. Reset on
        # any batch-size mismatch — the next chained-draft step will
        # repopulate from scratch (one MTP forward of overhead, much
        # cheaper than a wrong-shape crash).
        if self._cache is not None:
            cached_keys = getattr(self._cache, "keys", None)
            if cached_keys is not None and getattr(cached_keys, "shape", None):
                cache_B = cached_keys.shape[0]
                if cache_B != B:
                    self._cache = None

        if self._cache is None:
            self.reset_cache()

        out = self.mtp_module(
            prev_hidden=hidden_state,
            next_token=token_ids,
            embed_tokens=self.embed_tokens,
            final_norm=self.final_norm,
            lm_head=self.lm_head,
            mask=None,
            cache=self._cache,
            return_hidden=return_hidden,
        )

        if return_hidden:
            logits, pre_norm_out = out
        else:
            logits = out
            pre_norm_out = None

        if S == 1:
            logits = logits.squeeze(1)

        if return_hidden:
            return logits, pre_norm_out
        return logits


def dsv4_speculative_forward(
    model: Any,
    inputs: mx.array,
    cache: Sequence[Any],
    captured: dict[str, mx.array],
    speculative: bool = False,
) -> tuple[mx.array, mx.array]:
    """Verify-pass forward for DSv4. Returns ``(pre_norm, logits)``.

    DSv4's caches are :class:`RotatingKVCache` + :class:`PoolingCache`
    (+ optional indexer pool) wrapped in :class:`CacheList`. Rollback
    on rejected drafts is via ``cache.trim(n_rejected)`` on each
    top-level cache (CacheList recursively trims its children) — no
    GDN bookkeeping needed, no manual layer iteration, no kernel
    monkey-patching. ``speculative`` is accepted for parity with the
    Qwen3.5 helper but currently ignored — callers do rollback after
    this returns.

    The pre-final-norm hidden is captured as a side effect of the
    wrapped final-norm in
    :meth:`MTPBatchGenerator._setup_hidden_capture`; this function
    just reads it back from the supplied ``captured`` dict.
    """
    del speculative  # rollback is caller's responsibility; trim() is enough

    logits = model(inputs, cache=cache)
    pre_norm = captured.get("pre_norm")
    if pre_norm is None:
        raise RuntimeError(
            "dsv4_speculative_forward: captured['pre_norm'] is empty — the "
            "MTPBatchGenerator's wrapped-norm side channel was not "
            "exercised. Did model() run without going through "
            "DeepseekV4Model.norm?"
        )
    return pre_norm, logits


class DSv4MTPBatchGenerator(MTPBatchGenerator):
    """``MTPBatchGenerator`` specialized for DSv4. Overrides
    :meth:`_speculative_next` to use the DSv4 verify-forward + cache
    rollback path. Adds **BS>1** support so c=2+ benefits from MTP.

    BS>1 strategy: **min-acceptance.** All caches share a single scalar
    offset, so different per-uid acceptance counts can't be rolled
    back independently without per-stream cache state (which DSv4's
    cache classes don't expose). Instead we take ``n_min = min(n_accepted)``
    across uids, advance every cache by ``n_min + 1`` (drafts beyond
    ``n_min`` are wasted), and yield ``n_min + 1`` tokens per uid.

    Throughput math at BS=2 with per-stream accept p=0.85, gamma=3:
      single-stream tokens/cycle  ≈ 1 + p + p² + p³ = 3.18
      min-of-2 tokens/cycle       ≈ 2 × E[min(γ_1, γ_2)] + 2 ≈ 2 × 1.7 + 2 = 5.4
      vs naive (no spec) at BS=2  = 2
      → ~2.7× tokens/step at BS=2 (vs 3.18× single-stream MTP).

    Quality is preserved by construction: discarded "extra" accepted
    drafts simply aren't yielded; the cache state matches what the
    target model would produce after ``n_min + 1`` non-spec steps.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Acceptance-rate counters (lightweight; only matter when
        # EXO_DSV4_MTP_LOG_INTERVAL > 0).
        self._spec_cycles: int = 0
        self._spec_total_accepted: int = 0
        self._spec_accept_hist: list[int] = [0] * (self.gamma + 1)

    def _record_acceptance(self, n_accepted: int) -> None:
        if _LOG_INTERVAL <= 0:
            return
        self._spec_cycles += 1
        self._spec_total_accepted += n_accepted
        if 0 <= n_accepted <= self.gamma:
            self._spec_accept_hist[n_accepted] += 1
        if self._spec_cycles % _LOG_INTERVAL == 0:
            mean = self._spec_total_accepted / self._spec_cycles
            hist = ",".join(
                f"{i}:{c}" for i, c in enumerate(self._spec_accept_hist)
            )
            logger.warning(
                f"[MTP] cycles={self._spec_cycles} "
                f"mean_accept={mean:.3f}/{self.gamma} "
                f"hist={hist}"
            )

    # ── BS>1 dispatch ──────────────────────────────────────────────────

    def _next(self, *args: Any, **kwargs: Any):
        """Override base BS=1-only spec dispatch with BS-N support.

        Eligibility for spec: at least one uid in gen_batch, no
        prefill pending, no unprocessed sequences. The BS=1 path
        keeps the parent's behavior; BS>1 takes the new batched
        path.
        """
        gen_batch = self._generation_batch

        # Drain buffered tokens first (one per call), per the parent
        # API. At BS>1 we may have multiple uids with buffered tokens
        # — drain whichever has them first; the gen_batch still
        # advances normally on subsequent calls.
        if len(gen_batch) >= 1:
            for uid in gen_batch.uids:
                if uid in self._token_buffer and self._token_buffer[uid]:
                    return [], self._yield_buffered(uid)

        spec_eligible = (
            self.gamma > 0
            and len(gen_batch) >= 1
            and len(self._prompt_batch) == 0
            and len(self._unprocessed_sequences) == 0
        )
        if spec_eligible:
            # All uids must be prefilled (have a captured pre_norm).
            uids = list(gen_batch.uids)
            need_first_step = [u for u in uids if u not in self._mtp_prefilled]
            if need_first_step:
                return self._first_step_and_capture_batch(uids)
            if len(uids) == 1:
                return [], self._speculative_next(uids[0])
            return [], self._speculative_next_batch(uids)

        # Fallback: standard non-spec path. Drop prefill flags so the
        # next BS=1 idle window re-captures from a clean forward.
        result = super(MTPBatchGenerator, self)._next(*args, **kwargs)
        for uid in self._generation_batch.uids:
            self._mtp_prefilled.discard(uid)
        return result

    def _first_step_and_capture_batch(
        self, uids: "Sequence[int]"
    ):
        """Run a standard decode step at BS=N and stash per-uid pre_norms.

        The wrapped final-norm captures ``(B, L, hidden)``; we slice
        the last position per batch entry into the per-uid dict.
        Prefilled uids whose pre_norm was already captured stay in
        the dict; freshly-prefilled ones get added.
        """
        prompt_responses, generation_responses = super(
            MTPBatchGenerator, self
        )._next()
        if not generation_responses:
            return prompt_responses, generation_responses

        decode_pre_norm = self._captured.get("pre_norm")
        if decode_pre_norm is not None:
            mx.eval(decode_pre_norm)
            B = decode_pre_norm.shape[0]
            gen_batch = self._generation_batch
            for b_idx, uid in enumerate(gen_batch.uids):
                if b_idx >= B:
                    break
                self._mtp_pre_norm[uid] = decode_pre_norm[b_idx : b_idx + 1, -1:, :]
                self._mtp_prefilled.add(uid)
        return prompt_responses, generation_responses

    def _speculative_next_batch(self, uids: "Sequence[int]"):
        """One verify/accept cycle at BS>1 using min-acceptance strategy.

        For each uid, draft γ tokens chained through a separate MTP
        cache. Stack drafts into a (N, γ) batch; build the verify
        input (N, γ+1) by prepending each uid's last sampled token;
        run a single batched verify forward; per-uid acceptance check;
        roll back caches by ``γ - n_min`` where ``n_min`` is the
        minimum acceptance count across uids; yield ``n_min + 1``
        tokens per uid (extras buffered for the parent's next-call
        contract).
        """
        gen_batch = self._generation_batch
        N = len(uids)
        gamma = self.gamma

        # Verify each uid has the prerequisites.
        ys: list[mx.array] = []
        y_logprobs_list: list[Any] = []
        pre_norms: list[mx.array] = []
        for uid in uids:
            y = gen_batch._next_tokens
            if y is None or not gen_batch._next_logprobs:
                # Fallback: parent's standard path.
                return gen_batch.next()
            pre = self._mtp_pre_norm.get(uid)
            if pre is None:
                return gen_batch.next()
            pre_norms.append(pre)

        # gen_batch._next_tokens is shape (N,) — the tokens just
        # sampled by the previous step (one per uid, in uids order).
        next_tokens_arr = gen_batch._next_tokens.reshape(N, 1)  # (N, 1)
        # gen_batch._next_logprobs is a list of length N.
        y_logprobs_list = list(gen_batch._next_logprobs)

        # 1. Draft γ tokens — chained, batched across uids.
        # Stack pre_norms into (N, 1, hidden_size).
        stacked_pre_norm = mx.concatenate(pre_norms, axis=0)  # (N, 1, hidden)
        draft_ids_list, draft_probs_list = self._draft_tokens_batched(
            stacked_pre_norm, next_tokens_arr, gamma, self.temp
        )
        # draft_ids_list[i] is mx.array shape (N,) — uid b's i-th draft.
        # draft_probs_list[i] is mx.array shape (N, vocab) at temp>0, else None.

        # 2. Build verify input (N, γ+1) and run a single batched
        #    target forward. Cache rollback uses trim() afterwards.
        draft_concat = mx.concatenate(
            [d.reshape(N, 1) for d in draft_ids_list], axis=1
        )  # (N, γ)
        verify_input = mx.concatenate(
            [next_tokens_arr, draft_concat], axis=1
        )  # (N, γ+1)
        verify_pre_norm, verify_logits = dsv4_speculative_forward(
            self.model,
            verify_input,
            gen_batch.prompt_cache,
            self._captured,
        )
        # verify_pre_norm: (N, γ+1, hidden), verify_logits: (N, γ+1, vocab)

        # 3. Per-uid acceptance check (min-strategy).
        target_tokens = mx.argmax(
            verify_logits[:, :gamma, :], axis=-1
        )  # (N, γ)

        if self.temp == 0:
            matches = mx.equal(target_tokens, draft_concat)  # (N, γ)
            all_next = mx.argmax(verify_logits, axis=-1)  # (N, γ+1)
            logprobs_all = verify_logits - mx.logsumexp(
                verify_logits, axis=-1, keepdims=True
            )  # (N, γ+1, vocab)
            mx.async_eval(matches, all_next, logprobs_all, verify_pre_norm)
            # Per-uid n_accepted = first-mismatch index.
            matches_arr = matches.tolist()  # list[list[bool]] (N, γ)
            n_accepted_per: list[int] = []
            for n in range(N):
                k = 0
                while k < gamma and matches_arr[n][k]:
                    k += 1
                n_accepted_per.append(k)
            n_min = min(n_accepted_per)

            # Compose all_tokens per uid. Bonus token comes from the
            # verify forward at position n_min+1 for every uid (since
            # we're advancing all caches by n_min+1).
            all_tokens_per: list[list[tuple[int, mx.array]]] = []
            draft_int = draft_concat.tolist()  # (N, γ) int
            all_next_arr = all_next.tolist()  # (N, γ+1) int
            for n, uid in enumerate(uids):
                row: list[tuple[int, mx.array]] = []
                # First yielded: y (the previously-sampled token).
                row.append((int(next_tokens_arr[n, 0].item()), y_logprobs_list[n]))
                # Then n_min accepted drafts from THIS uid's drafts.
                for k in range(n_min):
                    row.append((int(draft_int[n][k]), logprobs_all[n, k]))
                all_tokens_per.append(row)
            bonus_vals = [int(all_next_arr[n][n_min]) for n in range(N)]
            bonus_lps = [logprobs_all[n, n_min] for n in range(N)]
        else:
            # Temp>0 stochastic acceptance. Sequential per-uid but the
            # heavy lifting (verify forward) is already batched.
            # NOTE: a fully-batched stochastic verify is possible but
            # involves more bookkeeping; this path is rarely hot.
            n_accepted_per = []
            all_tokens_per = []
            bonus_vals: list[int] = []
            bonus_lps: list[Any] = []
            logprobs_all = verify_logits - mx.logsumexp(
                verify_logits, axis=-1, keepdims=True
            )
            for n, uid in enumerate(uids):
                accept_ratios: list[mx.array] = []
                for i in range(gamma):
                    p = mx.softmax(verify_logits[n, i] / self.temp, axis=-1)
                    q = draft_probs_list[i]
                    p_di = p[draft_concat[n, i]]
                    q_di = q[n, draft_concat[n, i]]
                    ratio = p_di / mx.maximum(q_di, 1e-10)
                    accept_ratios.append(mx.minimum(ratio**self.alpha, 1.0))
                uniforms = mx.random.uniform(shape=(gamma,))
                k = 0
                while k < gamma and uniforms[k].item() < accept_ratios[k].item():
                    k += 1
                n_accepted_per.append(k)

            n_min = min(n_accepted_per)

            for n, uid in enumerate(uids):
                row = [(int(next_tokens_arr[n, 0].item()), y_logprobs_list[n])]
                for k in range(n_min):
                    row.append((int(draft_concat[n, k].item()), logprobs_all[n, k]))
                all_tokens_per.append(row)
                # bonus token: temp-sampled at position n_min for this uid
                bonus = int(
                    mx.random.categorical(verify_logits[n, n_min] * (1.0 / self.temp)).item()
                )
                bonus_vals.append(bonus)
                bonus_lps.append(logprobs_all[n, n_min])

        # 4. Roll back target caches by (γ - n_min) — single scalar
        #    rollback works because every stream advances by the same
        #    amount under min-strategy.
        rollback = gamma - n_min
        if rollback > 0:
            for c in gen_batch.prompt_cache:
                if hasattr(c, "trim"):
                    c.trim(rollback)
                elif hasattr(c, "offset"):
                    c.offset -= rollback
            mtp_cache = self.mtp._cache
            if mtp_cache is not None:
                if hasattr(mtp_cache, "trim"):
                    mtp_cache.trim(rollback)
                elif hasattr(mtp_cache, "offset"):
                    mtp_cache.offset -= rollback

        # 5. Update per-uid pre_norm to verify_pre_norm at n_min for
        #    next cycle.
        for n, uid in enumerate(uids):
            self._mtp_pre_norm[uid] = verify_pre_norm[n : n + 1, n_min : n_min + 1, :]

        # 6. Stage bonus tokens for next call.
        gen_batch._next_tokens = mx.array(bonus_vals)
        gen_batch._next_logprobs = bonus_lps
        mx.async_eval(gen_batch._next_tokens)

        # 7. Bookkeeping.
        total_yielded = sum(len(t) for t in all_tokens_per)
        self._gen_tokens_counter += total_yielded
        self._steps_counter += 1
        if self._steps_counter % 512 == 0:
            mx.clear_cache()

        # 8. State machine + length checks per yielded token, per uid.
        responses_per: list[list[Any]] = []
        for n, uid in enumerate(uids):
            idx = gen_batch.uids.index(uid)
            responses_per.append(
                self._build_yielded_responses(uid, idx, all_tokens_per[n])
            )

        # Yield first response per uid; buffer the rest.
        first_responses: list[Any] = []
        for n, uid in enumerate(uids):
            row = responses_per[n]
            if not row:
                continue
            first = row[0]
            rest = row[1:]
            if rest:
                self._token_buffer[uid] = deque(rest)
            elif first.finish_reason is not None:
                self._filter_finished_uid(uid)
                self._cleanup_uid(uid)
            first_responses.append(first)

        return first_responses

    def _draft_tokens_batched(
        self,
        stacked_pre_norm: mx.array,
        next_tokens_arr: mx.array,
        gamma: int,
        temp: float,
    ) -> tuple[list[mx.array], list[Any]]:
        """γ chained MTP forwards at batch_size=N.

        Returns ``(draft_ids, draft_probs)`` matching the contract of
        :func:`mtp_module.draft_tokens` but with batch dim N.

        Uses ``self.mtp.predict(...)`` once per draft step — each call
        runs the MTP module forward at shape (N, 1).
        """
        draft_ids: list[mx.array] = []
        draft_probs: list[Any] = []
        h = stacked_pre_norm
        tok_arr = next_tokens_arr  # (N, 1)

        for i in range(gamma):
            logits, h = self.mtp.predict(
                h, tok_arr, return_hidden=True, draft_mode=False
            )
            # logits: at S=1, predict() squeezes to (N, vocab).
            if temp == 0:
                tok_arr = mx.argmax(logits, axis=-1).reshape(-1, 1)  # (N, 1)
                draft_ids.append(tok_arr.reshape(-1))
                draft_probs.append(None)
            else:
                q = mx.softmax(logits / temp, axis=-1)  # (N, vocab)
                tok_arr = mx.random.categorical(
                    logits * (1.0 / temp)
                ).reshape(-1, 1)  # (N, 1)
                draft_ids.append(tok_arr.reshape(-1))
                draft_probs.append(q)
            # h returned by predict at S=1 has shape (N, 1, hidden) —
            # fed straight into the next predict() call.

        return draft_ids, draft_probs

    # ── BS=1 path (preserved) ──────────────────────────────────────────

    def _speculative_next(self, uid: int):
        """One verify/accept cycle, DSv4 flavor.

        Mirrors :meth:`MTPBatchGenerator._speculative_next` but uses:
          * :func:`dsv4_speculative_forward` for the verify pass
            (no GDN handling, no kernel patching)
          * ``cache.trim(rollback)`` on each prompt-cache entry for
            rejected-draft rollback (CacheList recursively trims its
            inner caches)
        """
        gen_batch = self._generation_batch
        idx = gen_batch.uids.index(uid)

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

        # 1. Draft γ tokens via MTP — chained, fully lazy.
        next_token_arr = y.reshape(1, 1)
        draft_ids, draft_probs = draft_tokens(
            self.mtp, pre_norm, next_token_arr, gamma, temp
        )

        # 2. Verify forward over [y, draft_0, ..., draft_{γ-1}] through
        #    the target. DSv4 has no GDN — just a vanilla forward.
        draft_concat = mx.concatenate(
            [d.reshape(1, 1) for d in draft_ids], axis=1
        )  # (1, γ)
        verify_input = mx.concatenate(
            [next_token_arr, draft_concat], axis=1
        )  # (1, γ+1)
        verify_pre_norm, verify_logits = dsv4_speculative_forward(
            self.model,
            verify_input,
            gen_batch.prompt_cache,
            self._captured,
        )

        # 3. Acceptance check (lazy until item() in step 4).
        target_tokens = mx.argmax(verify_logits[:, :gamma, :], axis=-1)

        accept_ratios: list[mx.array] = []
        uniforms: Optional[mx.array] = None
        corrections: list[mx.array] = []
        bonus_token: Optional[mx.array] = None
        matches: Optional[mx.array] = None
        all_next: Optional[mx.array] = None

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

        # 4. Determine acceptance count.
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
        self._record_acceptance(n_accepted)

        # 5. Roll back the target's KV caches for the rejected drafts.
        #    DSv4's CacheList implements trim() recursively; raw caches
        #    expose offset for the legacy path.
        rollback = gamma - n_accepted
        if rollback > 0:
            for c in gen_batch.prompt_cache:
                if hasattr(c, "trim"):
                    c.trim(rollback)
                elif hasattr(c, "offset"):
                    c.offset -= rollback

            # Also roll back the MTP module's own cache by the same
            # amount: each rejected draft cycle advanced the MTP cache
            # one step too. n_accepted MTP steps land in the next
            # cycle's pre_norm; the rest are wasted.
            mtp_cache = self.mtp._cache
            if mtp_cache is not None:
                if hasattr(mtp_cache, "trim"):
                    mtp_cache.trim(rollback)
                elif hasattr(mtp_cache, "offset"):
                    mtp_cache.offset -= rollback

        # 6. Compute bonus token + logprobs.
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

        # 7. Update MTP pre_norm to the verify-pass hidden at the
        #    accepted position, ready for the next cycle.
        pos = gamma if n_accepted == gamma else n_accepted
        self._mtp_pre_norm[uid] = verify_pre_norm[:, pos : pos + 1, :]

        # 8. Build all yielded tokens: [y, accepted drafts...].
        draft_int_values = [int(v) for v in draft_concat[0].tolist()]
        all_tokens: list[tuple[int, mx.array]] = [(y_val, y_logprobs)]
        for i in range(n_accepted):
            all_tokens.append((draft_int_values[i], logprobs_all[i]))

        # 9. Stage the bonus for the next call.
        gen_batch._next_tokens = mx.array([bonus_val])
        gen_batch._next_logprobs = [bonus_lp]
        mx.async_eval(gen_batch._next_tokens)

        # 10. Bookkeeping.
        self._gen_tokens_counter += len(all_tokens)
        self._steps_counter += 1
        if self._steps_counter % 512 == 0:
            mx.clear_cache()

        # 11. State machine + length checks per yielded token.
        responses = self._build_yielded_responses(uid, idx, all_tokens)

        first = responses[0]
        rest = responses[1:]
        if rest:
            self._token_buffer[uid] = deque(rest)
        elif first.finish_reason is not None:
            self._filter_finished_uid(uid)
            self._cleanup_uid(uid)

        return [first]
