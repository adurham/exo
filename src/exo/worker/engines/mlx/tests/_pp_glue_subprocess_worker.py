#!/usr/bin/env python3
# pyright: reportPrivateUsage=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportArgumentType=false, reportUnknownLambdaType=false
# pyright: reportCallIssue=false, reportAny=false
"""Worker script for the real 2-PROCESS glue-layer correctness test
(test_pp_batched_decode_glue_subprocess.py). Drives
Rank0BatchedDecodeGlue/Rank1BatchedDecodeGlue through a real
submit()-shaped lifecycle: enqueue two requests (mimicking two
separate ExoBatchGenerator.submit() calls, one upfront, one staged
mid-stream), tick() repeatedly (mimicking ExoBatchGenerator.step()'s
single call site), and complete_request() one of them via the real
eviction protocol -- all over MLX's real ring backend across two
genuine OS processes, exactly matching this session's established
test_pp_batched_decode_subprocess.py pattern.

Protocol with the parent: identical to _pp_subprocess_worker.py --
writes one JSON result file, parent does all assertions.
"""

from __future__ import annotations

import json
import sys
import traceback

import mlx.core as mx
import mlx.nn as nn
import mlx.utils


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


def _prefill(model, prompt, vocab_size):
    del vocab_size
    cache = model.make_cache()
    logits = model(prompt[None, :], cache=cache)
    mx.eval(logits)
    first_token = int(mx.argmax(logits[0, -1]).item())
    return cache, first_token


def main() -> int:
    rank = int(sys.argv[1])
    out_path = sys.argv[2]
    seed = int(sys.argv[3])

    result: dict[str, object] = {"rank": rank}
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
        n_layers = len(layers)
        mid = n_layers // 2

        # Real prefill for TWO independent requests -- each on the
        # model's ORIGINAL unwrapped layers (matching this session's
        # established prefill-before-layer-swap discipline).
        mx.random.seed(seed + 1000)
        prompt_a = mx.random.randint(0, vocab_size, shape=(5,))
        cache_a, first_token_a = _prefill(model, prompt_a, vocab_size)

        mx.random.seed(seed + 2000)
        prompt_b = mx.random.randint(0, vocab_size, shape=(4,))
        cache_b, first_token_b = _prefill(model, prompt_b, vocab_size)

        if rank == 0:
            my_layers = list(layers[:mid])
            my_layers[0] = BatchedMetaFramedPipelineFirstLayer(
                my_layers[0], r=0, group=group
            )
            my_layers[-1] = BatchedMetaFramedPipelineLastLayer(
                my_layers[-1], r=0, s=2, group=group
            )
            _set_layers(cast(nn.Module, cast(Any, model)), my_layers)
            my_cache_a = cache_a[:mid]
            my_cache_b = cache_b[:mid]

            session = BatchedDecodeSession.new(max_concurrency=2)
            adapter = BatchedDecodeResponseAdapter(
                session=session, eos_ids=frozenset({999999})
            )
            glue = Rank0BatchedDecodeGlue(
                session=session, adapter=adapter, dst_rank=1, group=group
            )

            # submit()-shaped call #1: enqueue request A upfront.
            glue.enqueue_admission(
                request_id=1,
                cache_slot=0,
                prefilled_cache=my_cache_a,
                initial_token=first_token_a,
                sampler=greedy,
                max_tokens=50,
            )

            # tick() #1: admits A (queue non-empty, slot free). The
            # admission response's own token IS first_token_a (the
            # token A's prefill already produced) -- do not
            # pre-seed tokens_a with first_token_a separately, or
            # it gets double-counted.
            responses, admitted_id = glue.tick(model)
            assert admitted_id == 1, f"expected admission of 1, got {admitted_id}"
            assert responses[1].token == first_token_a
            tokens_a: list[int] = [responses[1].token]

            # tick() #2: A decodes solo (no pending admissions left).
            responses, admitted_id = glue.tick(model)
            assert admitted_id is None
            tokens_a.append(responses[1].token)

            # submit()-shaped call #2: enqueue request B MID-STREAM,
            # while A is already decoding -- the real scenario this
            # whole glue layer exists to support safely.
            glue.enqueue_admission(
                request_id=2,
                cache_slot=1,
                prefilled_cache=my_cache_b,
                initial_token=first_token_b,
                sampler=greedy,
                max_tokens=50,
            )

            # tick() #3: admits B (queue non-empty again, slot 1 free).
            # B's admission response's own token IS first_token_b (the
            # token B's prefill already produced) -- seed tokens_b
            # from the response, matching tokens_a's own pattern
            # above (never pre-seed separately from a value the very
            # next tick() call will also report, or it double-counts).
            responses, admitted_id = glue.tick(model)
            assert admitted_id == 2, f"expected admission of 2, got {admitted_id}"
            assert responses[2].token == first_token_b
            tokens_b: list[int] = [responses[2].token]

            # tick() #4-5: A and B decode together.
            for _ in range(2):
                responses, admitted_id = glue.tick(model)
                assert admitted_id is None
                tokens_a.append(responses[1].token)
                tokens_b.append(responses[2].token)

            # complete_request()-shaped call: A finishes, real eviction
            # protocol round-trip (EvictMessage/EvictAckMessage).
            glue.complete_request(1)

            # tick() #6-7: B continues solo after A's eviction.
            for _ in range(2):
                responses, admitted_id = glue.tick(model)
                assert admitted_id is None
                tokens_b.append(responses[2].token)

            result["tokens_a"] = tokens_a
            result["tokens_b"] = tokens_b
            result["ok"] = True
        else:
            my_layers = list(layers[mid:])
            my_layers[0] = BatchedMetaFramedPipelineFirstLayer(
                my_layers[0], r=1, group=group
            )
            my_layers[-1] = BatchedMetaFramedPipelineLastLayer(
                my_layers[-1], r=1, s=2, group=group
            )
            _set_layers(cast(nn.Module, cast(Any, model)), my_layers)
            my_cache_a = cache_a[mid:]
            my_cache_b = cache_b[mid:]

            session = RankOneMirrorSession.new(max_concurrency=2)
            glue = Rank1BatchedDecodeGlue(session=session, src_rank=0, group=group)

            # submit()-shaped call #1: stage A's locally-prefilled
            # cache BEFORE its admission arrives reactively over wire.
            glue.stage_local_cache(
                request_id=1, cache_slot=0, prefilled_cache=my_cache_a
            )

            glue.tick(model)  # admits A reactively
            glue.tick(model)  # A decodes solo

            # submit()-shaped call #2: stage B mid-stream.
            glue.stage_local_cache(
                request_id=2, cache_slot=1, prefilled_cache=my_cache_b
            )

            glue.tick(model)  # admits B reactively
            for _ in range(2):
                glue.tick(model)  # A+B decode together

            glue.tick(model)  # eviction of A (MSG_KIND_EVICT branch)

            for _ in range(2):
                glue.tick(model)  # B continues solo

            result["ok"] = True
    except BaseException as e:  # noqa: BLE001 - report, don't crash silently
        result["ok"] = False
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()

    with open(out_path, "w") as f:
        json.dump(result, f)

    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
