#!/usr/bin/env python3
"""Local 2-rank repro/verification for EXO_PREFILL_CHUNK_OVERLAP correctness.

Exercises the REAL ``prefill_batched()`` chunk loop (the one in
``src/exo/worker/engines/mlx/generator/generate.py``) against a REAL
2-rank ``mx.distributed`` group on this single machine (loopback ring),
with a tiny synthetic DeepSeek-V4 model whose MoE/attention layers issue
genuine ``all_sum`` collectives -- so the barrier-vs-chunk-collective
interaction the race lives in is actually exercised, not stubbed out.

What it checks
--------------
For each prompt length (one an exact multiple of ``prefill_step_size``,
one with a remainder, both forcing multiple chunk-boundary crossings):

  1. Run ``prefill_batched`` with EXO_PREFILL_CHUNK_OVERLAP=0.
  2. Run it again, same process, same weights, fresh caches, with
     EXO_PREFILL_CHUNK_OVERLAP=1.
  3. Compare per-layer cache state and post-prefill decode logits
     between the two runs -- bit-exact where possible.
  4. Print each rank's own hashes so a cross-rank comparison can be made
     from the combined stdout of both ranks.

Weights are seeded identically on both ranks before init (otherwise any
cross-rank comparison is meaningless). No tokenizer / HF download: token
ids come from ``mx.random.randint``.

Run (2 real ranks, loopback ring -- NO cluster involvement):

    .venv/bin/mlx.launch -n 2 --backend ring \
        .venv/bin/python bench/local_repro_chunk_overlap.py

Single-process sanity mode (group=None, no collectives):

    uv run python bench/local_repro_chunk_overlap.py --single
"""

from __future__ import annotations

import hashlib
import os
import sys
from typing import Any, cast

import mlx.core as mx

SEED = 1234
PREFILL_STEP_SIZE = 32
# 128 = 4 * 32 exactly; 100 = 3 * 32 + 4 (remainder / odd last chunk).
# prefill_batched internally drops the last token, so the processed
# lengths are 127 and 99 -- both still cross chunk boundaries.
PROMPT_LENGTHS = (129, 100)
NUM_DECODE_STEPS = 4


def _shim_batch_pooling_cache() -> None:
    """Give ``BatchPoolingCache`` the carry attributes DSv4 reads.

    The installed mlx_lm defines ``_overlap_kv_carry`` /
    ``_overlap_gate_carry`` on ``PoolingCache`` but not on its batched
    sibling, so the batched DSv4 compressor path AttributeErrors on the
    first chunk. That is a pre-existing gap in the vendored mlx_lm and is
    orthogonal to the chunk-overlap race under test here; class-level
    ``None`` defaults restore the documented "no carry yet" semantics.
    """
    from mlx_lm.models.cache import BatchPoolingCache

    for attr in ("_overlap_kv_carry", "_overlap_gate_carry"):
        if not hasattr(BatchPoolingCache, attr):
            setattr(BatchPoolingCache, attr, None)


def make_tiny_dsv4() -> Any:
    """Tiny real DeepseekV4 model -- real MoE all_sum in the forward pass."""
    from mlx_lm.models.deepseek_v4 import Model, ModelArgs

    args = ModelArgs(
        model_type="deepseek_v4",
        vocab_size=2048,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=1,
        n_shared_experts=1,
        n_routed_experts=8,
        routed_scaling_factor=1.5,
        q_lora_rank=128,
        qk_rope_head_dim=32,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        hidden_act="silu",
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        head_dim=64,
        scoring_func="sqrtsoftplus",
        sliding_window=128,
        o_groups=2,
        o_lora_rank=128,
        index_n_heads=8,
        index_head_dim=32,
        index_topk=64,
        num_nextn_predict_layers=0,
        tie_word_embeddings=False,
        topk_method="noaux_tc",
        n_mtp_layers=0,
    )
    # Seed BEFORE weight init so both ranks build identical weights.
    mx.random.seed(SEED)
    model = Model(args)
    mx.eval(model.parameters())
    model.eval()
    return model


def _array_digest(a: mx.array) -> str:
    mx.eval(a)
    return hashlib.sha256(
        bytes(memoryview(a.astype(mx.float32)))  # pyright: ignore[reportArgumentType]
    ).hexdigest()[:16]


def _flatten_state(state: Any, out: list[mx.array]) -> None:
    if isinstance(state, mx.array):
        if state.size:
            out.append(state)
        return
    if isinstance(state, (list, tuple)):
        for item in cast("list[Any]", list(state)):
            _flatten_state(item, out)


def cache_digests(cache_layers: Any) -> list[str]:
    digests: list[str] = []
    for layer in cache_layers:
        arrays: list[mx.array] = []
        _flatten_state(cast(Any, layer).state, arrays)
        digests.append("+".join(_array_digest(a) for a in arrays) or "<empty>")
    return digests


def decode_logits(model: Any, cache: Any, start_token: int) -> list[mx.array]:
    out_logits: list[mx.array] = []
    cur = start_token
    for _ in range(NUM_DECODE_STEPS):
        out = model(mx.array([[cur]]), cache=cache)
        mx.eval(out)
        out_logits.append(out[0, -1])
        cur = int(mx.argmax(out[0, -1]).item())
    return out_logits


def run_once(
    model: Any,
    tokens: mx.array,
    group: Any,
    overlap: bool,
) -> tuple[list[str], list[mx.array], list[int]]:
    import exo.worker.engines.mlx.generator.batch_generate  # noqa: F401
    from exo.worker.engines.mlx.cache import make_kv_cache
    from exo.worker.engines.mlx.generator.generate import prefill_batched

    _shim_batch_pooling_cache()

    os.environ["EXO_PREFILL_CHUNK_OVERLAP"] = "1" if overlap else "0"

    def sampler(logits: mx.array) -> mx.array:
        return mx.argmax(logits, axis=-1)

    _, _, per_stream_caches, _ = prefill_batched(
        model,
        cast(Any, None),
        sampler,
        [tokens],
        [make_kv_cache(model)],
        group,
        None,
        None,
        prefill_step_size=PREFILL_STEP_SIZE,
    )
    cache = per_stream_caches[0]
    digests = cache_digests(cache)
    logits = decode_logits(model, cache, int(tokens[-1].item()))
    argmaxes = [int(mx.argmax(x).item()) for x in logits]
    return digests, logits, argmaxes


def main() -> int:
    single = "--single" in sys.argv
    group = None if single else mx.distributed.init(backend="ring")
    rank = 0 if group is None else group.rank()
    world = 1 if group is None else group.size()
    tag = f"[rank {rank}/{world}]"
    print(f"{tag} start (single={single})", flush=True)

    model = make_tiny_dsv4()

    all_ok = True
    for prompt_len in PROMPT_LENGTHS:
        mx.random.seed(SEED + prompt_len)
        tokens = mx.random.randint(0, 2048, (prompt_len,))
        mx.eval(tokens)
        kind = (
            "exact-multiple"
            if (prompt_len - 1) % PREFILL_STEP_SIZE == 0
            else "remainder"
        )
        n_chunks = -(-(prompt_len - 1) // PREFILL_STEP_SIZE)
        print(
            f"\n{tag} === prompt_len={prompt_len} ({kind}, step={PREFILL_STEP_SIZE},"
            f" {n_chunks} chunks) ===",
            flush=True,
        )

        d0, l0, a0 = run_once(model, tokens, group, overlap=False)
        d1, l1, a1 = run_once(model, tokens, group, overlap=True)

        cache_match = d0 == d1
        argmax_match = a0 == a1
        max_diff = max(
            float(mx.max(mx.abs(x - y)).item())
            for x, y in zip(l0, l1)  # noqa: B905
        )
        bit_exact = max_diff == 0.0

        print(f"{tag} overlap=0 cache digests: {d0}", flush=True)
        print(f"{tag} overlap=1 cache digests: {d1}", flush=True)
        print(f"{tag} cache_state_identical      = {cache_match}", flush=True)
        print(f"{tag} decode argmax overlap=0    = {a0}", flush=True)
        print(f"{tag} decode argmax overlap=1    = {a1}", flush=True)
        print(f"{tag} decode_argmax_identical    = {argmax_match}", flush=True)
        print(f"{tag} decode_logits_max_abs_diff = {max_diff:.3e}", flush=True)
        print(f"{tag} decode_logits_bit_exact    = {bit_exact}", flush=True)
        # Cross-rank comparison handle: identical across ranks means both
        # ranks saw the same reduced values (no mis-matched collective).
        print(f"{tag} XRANK overlap0_digest={'|'.join(d0)}", flush=True)
        print(f"{tag} XRANK overlap0_argmax={a0}", flush=True)

        case_ok = cache_match and argmax_match and bit_exact
        all_ok = all_ok and case_ok
        print(f"{tag} RESULT prompt_len={prompt_len}: "
              f"{'PASS' if case_ok else 'FAIL'}", flush=True)

    print(f"\n{tag} OVERALL: {'PASS' if all_ok else 'FAIL'}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
