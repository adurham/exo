#!/usr/bin/env python3
# pyright: reportPrivateUsage=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportArgumentType=false, reportUnknownLambdaType=false
# pyright: reportCallIssue=false, reportAny=false
"""Worker script for the real 2-PROCESS batched-decode correctness
test (test_pp_batched_decode_subprocess.py). Each of the two real OS
processes launched by that test runs THIS script once, with its rank
(0 or 1) and model kind ("llama" or "dsv4") passed via argv, and
MLX_HOSTFILE/MLX_RANK already set by the parent for
``mx.distributed.init(backend="ring")``.

This is deliberately a standalone script (not importable test code)
because MLX's ring backend needs a real, separate OS process per rank
-- there is no way to get genuinely independent compile-decoration
threads (the exact thing Phase 1's DSv4 threading investigation this
session already root-caused) other than real process boundaries.

Protocol with the parent: writes ONE JSON line per rank to the file
path given as argv[3] on completion (or on any exception, with an
"error" key) -- the parent test reads both ranks' result files after
both subprocesses exit and does the actual correctness assertions.
This script does NOT assert anything itself; it only reports data.
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


def _build_dsv4_model(seed: int):
    import mlx_lm.models.deepseek_v4 as dsv4

    args = dsv4.ModelArgs(
        model_type="deepseek_v4",
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        n_shared_experts=1,
        n_routed_experts=8,
        q_lora_rank=32,
        qk_rope_head_dim=16,
        num_experts_per_tok=2,
        head_dim=16,
        compress_ratios=[0, 4],
        hc_mult=2,
        o_groups=2,
        o_lora_rank=32,
        index_n_heads=4,
        index_head_dim=16,
        index_topk=3,
        sliding_window=8,
        num_nextn_predict_layers=0,
        tie_word_embeddings=True,
    )
    mx.random.seed(seed)
    model = dsv4.Model(args)
    params = model.parameters()

    def randomize(x):
        if isinstance(x, mx.array) and mx.issubdtype(x.dtype, mx.floating):
            return mx.random.normal(shape=x.shape, dtype=x.dtype)
        return x

    model.update(mlx.utils.tree_map(randomize, params))
    mx.eval(model.parameters())
    return model, args.vocab_size


def main() -> int:
    rank = int(sys.argv[1])
    model_kind = sys.argv[2]
    out_path = sys.argv[3]
    seed = int(sys.argv[4])

    result: dict[str, object] = {"rank": rank}
    try:
        group = mx.distributed.init(backend="ring")
        if group.rank() != rank:
            raise RuntimeError(
                f"MLX ring group.rank()={group.rank()} does not match "
                f"expected rank={rank} from argv -- environment setup bug"
            )
        if group.size() != 2:
            raise RuntimeError(f"expected group.size()==2, got {group.size()}")

        if model_kind == "llama":
            model, vocab_size = _build_llama_model(seed)
        elif model_kind == "dsv4":
            model, vocab_size = _build_dsv4_model(seed)
        else:
            raise ValueError(f"unknown model_kind={model_kind!r}")

        sys.path.insert(0, "src")
        from typing import Any, cast

        from exo.worker.engines.mlx.auto_parallel import (
            _set_layers,
            get_inner_model,
            get_layers,
        )
        from exo.worker.engines.mlx.pp_batched_decode_layers import (
            BatchedMetaFramedPipelineFirstLayer,
            BatchedMetaFramedPipelineLastLayer,
        )
        from exo.worker.engines.mlx.pp_batched_decode_runtime import (
            BatchedDecodeSession,
            RankOneMirrorSession,
        )
        from exo.worker.engines.mlx.pp_scheduler_wire import (
            recv_step_message,
            send_step_message,
        )

        inner = get_inner_model(cast(nn.Module, cast(Any, model)))
        layers = get_layers(inner)
        n_layers = len(layers)
        mid = n_layers // 2

        def greedy(logits: mx.array) -> mx.array:
            return mx.argmax(logits, axis=-1)

        # Prefill on the model's ORIGINAL (full, unwrapped) layer list
        # -- the same model instance, before any layer is replaced with
        # a batched metaframe wrapper. Only AFTER prefill produces a
        # real full cache do we slice it and swap in this rank's own
        # half of the layers wrapped for decode -- matches every other
        # test in this session (e.g.
        # test_pp_batched_decode_over_real_wire.py's
        # _single_request_prefilled_cache, which deliberately uses a
        # SEPARATE unwrapped model instance for exactly this reason).
        mx.random.seed(seed + 1000)
        prompt = mx.random.randint(0, vocab_size, shape=(6,))
        cache = model.make_cache()
        logits = model(prompt[None, :], cache=cache)
        mx.eval(logits)
        first_token = int(mx.argmax(logits[0, -1]).item())

        if rank == 0:
            my_layers = list(layers[:mid])
            my_layers[0] = BatchedMetaFramedPipelineFirstLayer(
                my_layers[0], r=0, group=group
            )
            my_layers[-1] = BatchedMetaFramedPipelineLastLayer(
                my_layers[-1], r=0, s=2, group=group
            )
            _set_layers(cast(nn.Module, cast(Any, model)), my_layers)
            my_cache = cache[:mid]

            session = BatchedDecodeSession.new(max_concurrency=2)
            msg = session.admit_request(
                request_id=1,
                cache_slot=0,
                prefilled_cache=my_cache,
                initial_token=first_token,
                sampler=greedy,
            )
            send_step_message(msg, dst=1, group=group)

            tokens: list[int] = [first_token]
            for _ in range(4):
                prepared = session.prepare_step()
                send_step_message(prepared.message, dst=1, group=group)
                logits = session.run_forward(model, prepared)
                step_results = session.finish_step(prepared, logits)
                tokens.append(step_results[1][0])

            result["tokens"] = tokens
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
            my_cache = cache[mid:]

            session = RankOneMirrorSession.new(max_concurrency=2)
            recvd_admit = recv_step_message(src=0, group=group)
            session.admit_request(recvd_admit, cache_slot=0, prefilled_cache=my_cache)

            for _ in range(4):
                recvd = recv_step_message(src=0, group=group)
                n_active = len(recvd.entries)
                placeholder = mx.zeros((n_active, 1), dtype=mx.int32)
                mx.eval(placeholder)
                session.step(model, recvd, placeholder)

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
