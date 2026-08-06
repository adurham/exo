#!/usr/bin/env python3
# pyright: reportPrivateUsage=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportArgumentType=false, reportUnknownLambdaType=false
# pyright: reportCallIssue=false, reportAny=false
"""Worker script for the GENUINELY-INDEPENDENT 2-process admission-race
regression test (``test_pp_admission_race_subprocess.py``).

WHY THIS EXISTS (the gap it closes): the pre-existing 2-process glue
test (``_pp_glue_subprocess_worker.py``) is real-transport but
LOCKSTEP -- one script decides, for BOTH ranks, exactly which call
each rank makes at each point in the sequence. Per
``docs/batched-decode-n2-admission-handoff-2026-08-05.md`` and design
doc Section 15, the N=2 deadlock observed on the real cluster
(``[jaccl] reliable_all_reduce_v2 deadline``) exists ONLY because
each rank's ``runner.py`` event loop decides INDEPENDENTLY, per loop
iteration and with no cross-rank synchronization, whether to do
``submit()``-side work (prefill's own single-request metaframe
send/recv, ``batch_axis=0``) or ``step()``-side work (the batched
glue's ``tick()``, which sends the SCHEDULER control-message wire
shape followed by a ``batch_axis=1`` metaframe). A lockstep driver
structurally cannot produce that divergence.

WHAT THIS WORKER DOES DIFFERENTLY: each rank runs its OWN loop,
mirroring ``runner.py``'s ``handle_generation_tasks()`` shape -- a
local queue of "work that has become available", drained on this
rank's OWN schedule, interleaved with ``tick()`` (the ``step()``
equivalent) every iteration, with a per-rank-seeded random jitter
sleep between iterations so the two ranks' schedules genuinely
diverge in wall time. Neither rank ever waits for or observes the
other's scheduling decision -- exactly the production shape.

The two ranks perform the SAME TOTAL AMOUNT of work (identical
iteration count, exactly one single-request prefill each, identical
number of ``tick()`` calls). Only the ORDERING/interleaving differs,
and only because each rank chose independently. Under a correct
design (an in-band, rank-0-decides/rank-1-reacts admission signal, per
the handoff doc's next-step #1) that difference must be harmless.
Under today's code it is not: the moment rank 0 issues prefill's
metaframe header (6 int32, ``pp_metaframe._HEADER_FIELDS``) on an
iteration where rank 1 issues the scheduler control header (5 int32,
``pp_scheduler_wire._HEADER_FIELDS``) -- or vice versa -- the two
ranks are running mismatched wire operations against each other, which
is the real-cluster deadlock's local, deterministic-to-detect
equivalent.

2026-08-06 UPDATE (the actual fix under test now): per
``pp_batched_decode_glue.py``'s module docstring "UPDATE
(2026-08-06...)" section, request B's prefill is no longer dispatched
by directly calling the single-request metaframe layers on this rank's
own local schedule -- it is dispatched via
``Rank0BatchedDecodeGlue.enqueue_prefill()`` (rank 0, pure in-memory,
zero wire I/O) and only actually RUNS once ``tick()`` (the SAME
single-writer call already used for decode-step traffic) grants it via
a real ``PrefillMessage``. Rank 1 NEVER independently decides to
prefill B -- it only runs B's prefill when its OWN ``tick()``
reactively receives that ``PrefillMessage`` and returns a
``PrefillGrant``. This worker therefore drives the FIXED call shape on
both ranks (``enqueue_prefill``/``tick()``-returns-``PrefillGrant``/
run-prefill-on-grant/``stage_local_cache``) under the SAME genuinely-
independent per-rank scheduling as before the fix -- proving the fix
holds even when the two ranks' local loops disagree about WHEN B
becomes available, not just proving the old (pre-fix) code races.

Both wire shapes come from the SAME, REAL, PRODUCTION code paths --
no mock, no simulated transport:
  * prefill  -> ``MetaFramedPipelineFirstLayer``/``LastLayer``
    (reached through the ``Batched*`` subclasses' own
    outside-``batch_step_scope`` fallback, i.e. exactly how a real
    runner reaches prefill with batched-decode layers installed at
    model-load time -- design doc Section 15, Attempt 1) -- now
    dispatched ONLY from inside ``tick()``'s ``PrefillGrant`` branch,
    never independently.
  * decode   -> ``Rank0BatchedDecodeGlue.tick()`` /
    ``Rank1BatchedDecodeGlue.tick()``.

Protocol with the parent: identical to the other subprocess workers --
write one JSON result file; the parent does all assertions.
"""

from __future__ import annotations

import json
import random
import sys
import time
import traceback

import mlx.core as mx
import mlx.nn as nn
import mlx.utils

# Per-iteration jitter bound (seconds). Small enough that the whole
# test stays fast, large enough that the two ranks' loops genuinely
# drift apart in wall time rather than accidentally staying in step.
_MAX_JITTER_SECONDS = 0.01

# Total loop iterations per rank. Identical on both ranks so the TOTAL
# work is symmetric -- only the interleaving is free to diverge.
_ITERATIONS = 10

# Earliest iteration at which request B's prefill is allowed to be
# picked off the local queue. Keeps request A admitted and decoding
# first, so every tick() before this point is real wire traffic
# (an idle tick() sends nothing and would desynchronize the counts for
# a reason unrelated to the race under test).
_B_EARLIEST_ITERATION = 3

# Probability that a rank, on an iteration where B's prefill IS
# available, chooses to do it now rather than tick() first. This is the
# whole point: the choice is LOCAL and unsynchronized, exactly like
# runner.py's per-iteration `self._work_queue.get_nowait()` vs
# `generator.step()` decision.
_PREFILL_PICK_PROBABILITY = 0.5


def _build_llama_model(seed: int):
    from mlx_lm.models.llama import Model as LlamaModel
    from mlx_lm.models.llama import ModelArgs

    args = ModelArgs(
        model_type="llama",
        hidden_size=256,
        num_hidden_layers=4,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=4096,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    mx.random.seed(seed)
    model = LlamaModel(args)
    params = model.parameters()
    new_params = mlx.utils.tree_map(
        lambda p: mx.random.normal(shape=p.shape, dtype=p.dtype)
        if isinstance(p, mx.array)
        else p,
        params,
    )
    model.update(new_params)
    mx.eval(model.parameters())
    return model, args.vocab_size


def _local_prefill(model, prompt):
    """Plain local prefill on the model's ORIGINAL (unwrapped) layers,
    before the pipeline layer swap -- matching the established
    prefill-before-layer-swap discipline of the other subprocess
    workers. Used only to obtain request A's per-rank cache half."""
    cache = model.make_cache()
    logits = model(prompt[None, :], cache=cache)
    mx.eval(logits)
    return cache, int(mx.argmax(logits[0, -1]).item())


def main() -> int:  # noqa: C901 - one linear scenario, split would obscure it
    rank = int(sys.argv[1])
    out_path = sys.argv[2]
    seed = int(sys.argv[3])

    result: dict[str, object] = {"rank": rank}
    # Per-rank INDEPENDENT scheduling RNG. Deliberately seeded
    # differently per rank (unlike the model seed, which must match)
    # so the two loops' choices and sleeps genuinely diverge -- this
    # is the mechanism that replaces the old harness's lockstep.
    schedule_rng = random.Random(seed * 7919 + rank * 104729)
    trace: list[str] = []

    try:
        group = mx.distributed.init(backend="ring")
        if group.rank() != rank:
            raise RuntimeError(
                f"MLX ring group.rank()={group.rank()} does not match "
                f"expected rank={rank} from argv"
            )
        if group.size() != 2:
            raise RuntimeError(f"expected group.size()==2, got {group.size()}")

        model, vocab_size = _build_llama_model(seed)

        sys.path.insert(0, "src")
        from typing import Any, cast

        from exo.worker.engines.mlx.auto_parallel import (
            _set_layers,
            get_inner_model,
            get_layers,
        )
        from exo.worker.engines.mlx.pp_batched_decode_adapter import (
            BatchedDecodeResponseAdapter,
        )
        from exo.worker.engines.mlx.pp_batched_decode_glue import (
            Rank0BatchedDecodeGlue,
            Rank1BatchedDecodeGlue,
        )
        from exo.worker.engines.mlx.pp_batched_decode_layers import (
            BatchedMetaFramedPipelineFirstLayer,
            BatchedMetaFramedPipelineLastLayer,
        )
        from exo.worker.engines.mlx.pp_batched_decode_runtime import (
            BatchedDecodeSession,
            RankOneMirrorSession,
        )

        def greedy(logits: mx.array) -> mx.array:
            return mx.argmax(logits, axis=-1)

        inner = get_inner_model(cast(nn.Module, cast(Any, model)))
        layers = get_layers(inner)
        mid = len(layers) // 2

        mx.random.seed(seed + 1000)
        prompt_a = mx.random.randint(0, vocab_size, shape=(5,))
        cache_a, first_token_a = _local_prefill(model, prompt_a)

        # Request B's prompt. B is the request that arrives MID-STREAM
        # and whose prefill is the one that goes over the REAL wire,
        # racing the ongoing batched decode -- the exact seam design
        # doc Section 15 identifies as the unfixed gap.
        mx.random.seed(seed + 2000)
        prompt_b = mx.random.randint(0, vocab_size, shape=(4,))

        my_slice = slice(None, mid) if rank == 0 else slice(mid, None)
        my_layers = list(layers[my_slice])
        my_layers[0] = BatchedMetaFramedPipelineFirstLayer(
            my_layers[0], r=rank, group=group
        )
        last_layer = BatchedMetaFramedPipelineLastLayer(
            my_layers[-1], r=rank, s=2, group=group
        )
        my_layers[-1] = last_layer
        _set_layers(cast(nn.Module, cast(Any, model)), my_layers)
        my_cache_a = cache_a[my_slice]
        # model.make_cache() is called AFTER _set_layers() already
        # replaced the model's layers with my_layers (the half-slice)
        # -- it already returns a cache list correctly sized for
        # my_layers, NOT the full original layer list. Slicing it
        # again with [my_slice] here would double-slice (out of range
        # on rank 1, since my_slice=(mid, None) applied a second time
        # to an already-halved list walks past its end) -- this was a
        # real bug in this worker's first draft, caught when this test
        # actually ran the fixed-code path for the first time (the
        # b_prefill_b's forward pass is the first real use of cache_b
        # via the model's real fa_idx-indexed cache lookup, which is
        # what surfaced the IndexError).
        cache_b = model.make_cache()

        def run_single_request_prefill_b() -> int:
            """One REAL single-request prefill through the REAL Phase
            0.5 metaframe layers (reached via the Batched* subclasses'
            no-batch-context fallback), i.e. the exact wire shape
            ``submit()``/``prefill()`` puts on the wire in production.
            Deliberately a DIFFERENT wire shape from ``tick()``'s
            scheduler control message + batch-axis metaframe -- that
            difference is what the race turns into a deadlock IF this
            is ever dispatched independently of a ``PrefillGrant``.
            Under the 2026-08-06 fix, this function is ONLY ever
            called from inside the ``PrefillGrant``-handling branch
            below -- never on this rank's own independent schedule.

            Returns the sampled first token (meaningful only on rank
            0, which gathers the real final hidden state/logits, same
            as ``_local_prefill``'s own convention above -- rank 1's
            return value is never read).
            """
            last_layer.is_prefill = True
            try:
                out = model(prompt_b[None, :], cache=cache_b)
                mx.eval(out)
            finally:
                last_layer.is_prefill = False
            return int(mx.argmax(out[0, -1]).item())

        rank0_glue: Rank0BatchedDecodeGlue | None = None
        rank1_glue: Rank1BatchedDecodeGlue | None = None
        if rank == 0:
            session = BatchedDecodeSession.new(max_concurrency=2)
            adapter = BatchedDecodeResponseAdapter(
                session=session, eos_ids=frozenset({999999})
            )
            rank0_glue = Rank0BatchedDecodeGlue(
                session=session, adapter=adapter, dst_rank=1, group=group
            )
            # submit()-equivalent for request A, before the loop starts
            # (A is the already-running request; B is the mid-stream
            # arrival the race is about).
            rank0_glue.enqueue_admission(
                request_id=1,
                cache_slot=0,
                prefilled_cache=my_cache_a,
                initial_token=first_token_a,
                sampler=greedy,
                max_tokens=1000,
            )
        else:
            mirror = RankOneMirrorSession.new(max_concurrency=2)
            rank1_glue = Rank1BatchedDecodeGlue(
                session=mirror, src_rank=0, group=group
            )
            rank1_glue.stage_local_cache(
                request_id=1, cache_slot=0, prefilled_cache=my_cache_a
            )

        # ------------------------------------------------------------
        # THE INDEPENDENT EVENT LOOP (the whole point of this worker).
        #
        # Mirrors runner.py's handle_generation_tasks(): every
        # iteration either drains one item of locally-available work
        # (the submit()/prefill side) or advances decode (the step()
        # side), decided PURELY from this rank's own local state and
        # its own RNG -- never from anything the peer rank did. The
        # jitter sleep makes the two ranks' wall-clock cadences
        # genuinely diverge, as two real runner processes' do.
        #
        # 2026-08-06 UPDATE (drives the FIX, not the pre-fix shape):
        # rank 0's independent local choice is now WHEN to call
        # ``enqueue_prefill`` (pure in-memory, submit()-equivalent,
        # zero wire I/O -- mirrors a real runner's work-queue drain
        # picking up request B at an arbitrary, locally-decided
        # iteration) versus calling ``tick()`` (the single-writer wire
        # call). ``tick()`` itself -- not this rank's own schedule --
        # decides WHEN the queued prefill actually gets announced over
        # the wire (see Rank0BatchedDecodeGlue.tick()'s priority
        # ladder: admission first, then decode, then a new prefill
        # announcement). Rank 1 has NO independent B-related decision
        # left AT ALL -- it only ever calls ``tick()``, and only runs
        # B's prefill when ``tick()`` itself reactively returns a
        # ``PrefillGrant``. That asymmetry (rank 0 still has a local
        # scheduling choice about ENQUEUEING; rank 1 has none about
        # PREFILLING) is precisely what the fix's "rank 0 decides,
        # rank 1 reacts" design produces -- if this test still races,
        # the fix has a real hole; if it doesn't, this is genuine
        # evidence the single-writer control channel is closing the
        # gap under real independent-schedule pressure, not just in a
        # lockstep test.
        # ------------------------------------------------------------
        tokens_a: list[int] = []
        b_enqueued = False
        b_registered = False
        b_prefill_done = False
        for iteration in range(_ITERATIONS):
            time.sleep(schedule_rng.uniform(0.0, _MAX_JITTER_SECONDS))

            if rank == 0:
                assert rank0_glue is not None
                b_available = (
                    iteration >= _B_EARLIEST_ITERATION and not b_enqueued
                )
                must_enqueue_b_now = (
                    b_available and iteration == _ITERATIONS - 2
                )
                do_enqueue_b_now = must_enqueue_b_now or (
                    b_available
                    and schedule_rng.random() < _PREFILL_PICK_PROBABILITY
                )
                if do_enqueue_b_now:
                    trace.append(f"it={iteration} enqueue_prefill_b")
                    rank0_glue.enqueue_prefill(
                        request_id=2,
                        cache_slot=1,
                        n_prompt_tokens=int(prompt_b.shape[0]),
                        single_request_fallback=False,
                    )
                    b_enqueued = True

                trace.append(f"it={iteration} tick")
                responses, admitted_id, grant = rank0_glue.tick(model)
                del admitted_id
                if 1 in responses:
                    tokens_a.append(responses[1].token)
                if grant is not None:
                    assert grant.request_id == 2
                    trace.append(f"it={iteration} run_grant_prefill_b")
                    first_token_b = run_single_request_prefill_b()
                    b_prefill_done = True
                    rank0_glue.enqueue_admission(
                        request_id=2,
                        cache_slot=grant.cache_slot,
                        prefilled_cache=cache_b,
                        initial_token=first_token_b,
                        sampler=greedy,
                        max_tokens=1000,
                    )
            else:
                assert rank1_glue is not None
                # 2026-08-06 fix (prefill forward-pass race, see
                # PrefillReadyMessage's own docstring in
                # pp_scheduler_protocol.py): rank 1 must call
                # mark_prefill_registered() on its OWN independent
                # schedule -- mirroring submit()'s real call site in
                # production, and matching rank 0's own independently-
                # randomized enqueue_prefill timing above -- for
                # tick() to ever return a real (non-NACK'd) grant.
                # Deliberately uses the SAME kind of independent
                # per-rank random gate as rank 0's b_available/
                # do_enqueue_b_now logic (not synchronized with it in
                # any way) so this test continues to genuinely
                # exercise the real timing race the ack/NACK handshake
                # exists to resolve, rather than always registering
                # suspiciously early/late relative to rank 0's own
                # schedule.
                b_registerable = (
                    iteration >= _B_EARLIEST_ITERATION and not b_registered
                )
                must_register_b_now = (
                    b_registerable and iteration == _ITERATIONS - 2
                )
                do_register_b_now = must_register_b_now or (
                    b_registerable
                    and schedule_rng.random() < _PREFILL_PICK_PROBABILITY
                )
                if do_register_b_now:
                    trace.append(f"it={iteration} mark_prefill_registered_b")
                    rank1_glue.mark_prefill_registered(2)
                    b_registered = True

                trace.append(f"it={iteration} tick")
                grant = rank1_glue.tick(model)
                if grant is not None:
                    assert grant.request_id == 2
                    trace.append(f"it={iteration} run_grant_prefill_b")
                    _ = run_single_request_prefill_b()
                    b_prefill_done = True
                    rank1_glue.stage_local_cache(
                        request_id=2,
                        cache_slot=grant.cache_slot,
                        prefilled_cache=cache_b,
                    )

        if not b_prefill_done:
            raise RuntimeError(
                f"rank={rank}: request B's prefill was never granted/run "
                f"within {_ITERATIONS} iterations -- the scenario did not "
                f"actually exercise the seam under test; increase "
                f"_ITERATIONS or _PREFILL_PICK_PROBABILITY rather than "
                f"treating this as a pass"
            )

        if rank == 0:
            result["tokens_a"] = tokens_a
        result["trace"] = trace
        result["ok"] = True
    except BaseException as e:  # noqa: BLE001 - report, don't crash silently
        result["ok"] = False
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        result["trace"] = trace

    with open(out_path, "w") as f:
        json.dump(result, f)

    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
