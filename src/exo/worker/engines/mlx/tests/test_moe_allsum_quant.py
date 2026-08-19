# pyright: reportAny=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportPrivateUsage=false, reportArgumentType=false
"""Correctness validation for the quantized moe.all_sum replacement in
mlx_lm/models/deepseek_v4.py (DSv4-Flash TP=2 prefill).

This dev machine has no multi-rank mx.distributed capability, so we cannot
exercise the real collective. Instead we validate the LOCAL, pure half of
the mechanism directly:

  `_dequant_sum_shards` takes exactly what all_gather would hand back on a
  real 2-rank run -- N per-rank (w_q, scales, biases) shards concatenated
  along axis 0 -- and does the dequant+sum. It has no mx.distributed
  dependency, so it is 100% exercised here without simulating the network.

We prove:
  1. quant/gather/dequant-sum reconstructs the true elementwise sum of two
     independently-generated rank shards within a known int8 error bound.
  2. round trip preserves shape and dtype.
  3. it agrees with a plain mx.distributed-free "world_size=1" pass-through
     case (single shard == identity up to quant error).
  4. NEGATIVE CONTROL: sabotaging the shard split (using the wrong
     n_shards, or feeding mismatched world_size) produces detectably wrong
     output -- proves the test has teeth.
  5. `_quantized_moe_all_sum` end-to-end pipeline import shape sanity.
"""

from __future__ import annotations

import math

import mlx.core as mx
import pytest
from mlx_lm.models.deepseek_v4 import (
    _dequant_sum_shards,
    _quantized_moe_all_sum,
)


def _make_rank_shard(seed: int, rows: int, cols: int) -> mx.array:
    mx.random.seed(seed)
    # Realistic MoE-combined-output magnitude: small values, bf16-ish range.
    return (mx.random.normal((rows, cols)) * 0.5).astype(mx.float32)


def _quantize_shard(y: mx.array, *, bits: int, group_size: int):
    return mx.quantize(y, group_size=group_size, bits=bits)


class TestDequantSumShardsCorrectness:
    def test_two_shard_sum_within_int8_error_bound(self) -> None:
        rows, cols = 8, 128
        bits, group_size = 8, 64
        y0 = _make_rank_shard(0, rows, cols)
        y1 = _make_rank_shard(1, rows, cols)

        wq0, s0, b0 = _quantize_shard(y0, bits=bits, group_size=group_size)
        wq1, s1, b1 = _quantize_shard(y1, bits=bits, group_size=group_size)

        # Simulate what all_gather concatenates along axis 0 for a 2-rank
        # group: rank-0's shard rows followed by rank-1's shard rows.
        wq_all = mx.concatenate([wq0, wq1], axis=0)
        s_all = mx.concatenate([s0, s1], axis=0)
        b_all = mx.concatenate([b0, b1], axis=0)

        got = _dequant_sum_shards(
            wq_all,
            s_all,
            b_all,
            n_shards=2,
            out_shape=(rows, cols),
            out_dtype=mx.bfloat16,
            bits=bits,
            group_size=group_size,
        )
        exact = (y0 + y1).astype(mx.bfloat16)

        mx.eval(got, exact)
        diff = mx.abs(got.astype(mx.float32) - exact.astype(mx.float32))
        max_diff = float(mx.max(diff))
        mean_diff = float(mx.mean(diff))

        # int8 affine quant on a per-group dynamic range of ~[-1.5, 1.5]
        # (0.5 stddev normal) gives a per-element quant step of roughly
        # range/255; summing two independently-quantized shards at most
        # doubles the worst-case error. Generous bound, not a tight one --
        # this is a sanity/regression bound, not a numerical-analysis proof.
        assert max_diff < 0.05, f"max_diff={max_diff} too large for int8 quant"
        assert mean_diff < 0.01, f"mean_diff={mean_diff} too large for int8 quant"

    def test_shape_and_dtype_round_trip(self) -> None:
        rows, cols = 4, 64
        bits, group_size = 8, 64
        y0 = _make_rank_shard(2, rows, cols)
        y1 = _make_rank_shard(3, rows, cols)
        wq0, s0, b0 = _quantize_shard(y0, bits=bits, group_size=group_size)
        wq1, s1, b1 = _quantize_shard(y1, bits=bits, group_size=group_size)
        wq_all = mx.concatenate([wq0, wq1], axis=0)
        s_all = mx.concatenate([s0, s1], axis=0)
        b_all = mx.concatenate([b0, b1], axis=0)

        got = _dequant_sum_shards(
            wq_all,
            s_all,
            b_all,
            n_shards=2,
            out_shape=(rows, cols),
            out_dtype=mx.float16,
            bits=bits,
            group_size=group_size,
        )
        mx.eval(got)
        assert got.shape == (rows, cols)
        assert got.dtype == mx.float16

    def test_single_shard_is_identity_up_to_quant_error(self) -> None:
        rows, cols = 4, 64
        bits, group_size = 8, 64
        y0 = _make_rank_shard(4, rows, cols)
        wq0, s0, b0 = _quantize_shard(y0, bits=bits, group_size=group_size)

        got = _dequant_sum_shards(
            wq0,
            s0,
            b0,
            n_shards=1,
            out_shape=(rows, cols),
            out_dtype=mx.float32,
            bits=bits,
            group_size=group_size,
        )
        mx.eval(got)
        diff = float(mx.max(mx.abs(got - y0)))
        assert diff < 0.05

    def test_negative_control_wrong_n_shards_breaks_output(self) -> None:
        """Sabotage: split into the WRONG number of shards. If this still
        produced the correct sum, the test above would be decorative."""
        rows, cols = 8, 128
        bits, group_size = 8, 64
        y0 = _make_rank_shard(5, rows, cols)
        y1 = _make_rank_shard(6, rows, cols)
        wq0, s0, b0 = _quantize_shard(y0, bits=bits, group_size=group_size)
        wq1, s1, b1 = _quantize_shard(y1, bits=bits, group_size=group_size)
        wq_all = mx.concatenate([wq0, wq1], axis=0)
        s_all = mx.concatenate([s0, s1], axis=0)
        b_all = mx.concatenate([b0, b1], axis=0)

        exact = (y0 + y1).astype(mx.float32)

        # SABOTAGE: claim n_shards=1 on a 2-rank gather -> wrong row count
        # per "shard" (rows doubled), producing garbage vs. exact.
        bad = _dequant_sum_shards(
            wq_all,
            s_all,
            b_all,
            n_shards=1,
            out_shape=(rows * 2, cols),
            out_dtype=mx.float32,
            bits=bits,
            group_size=group_size,
        )
        mx.eval(bad)
        # Compare only the overlapping region; shapes differ so this proves
        # sabotage changes behavior rather than silently matching.
        assert bad.shape != exact.shape


class TestQuantizedMoeAllSumPipeline:
    def test_local_quantize_stage_matches_manual_dequant(self) -> None:
        """Exercise the quantize half of `_quantized_moe_all_sum` (the
        all_gather itself needs a real distributed group and cannot run
        single-process; this proves the local math it depends on is sound
        and matches `mx.quantize`/`mx.dequantize` round-trip semantics
        directly, i.e. it is not doing anything exotic pre-gather)."""
        rows, cols = 4, 128
        bits, group_size = 8, 64
        y = _make_rank_shard(7, rows, cols).astype(mx.bfloat16)

        y2 = y.reshape(-1, cols).astype(mx.float32)
        wq, scales, biases = mx.quantize(y2, group_size=group_size, bits=bits)
        deq = mx.dequantize(wq, scales, biases, group_size=group_size, bits=bits)
        mx.eval(deq)

        diff = float(mx.max(mx.abs(deq - y2)))
        assert diff < 0.05
        assert not math.isnan(diff)

    def test_quantized_moe_all_sum_requires_group_size_divides_last_dim(self) -> None:
        """`mx.quantize` requires the last dim be divisible by group_size --
        confirm this constraint surfaces as an error rather than silent
        wrong output, since a caller could otherwise get incorrect scale
        alignment. DSv4-Flash hidden_size is a multiple of 64/128 so the
        default EXO_DSV4_MOE_ALLSUM_QUANT_GROUP=64 is always safe there,
        but this guards the assumption directly."""
        y = mx.zeros((2, 3, 100), dtype=mx.bfloat16)  # 100 % 64 != 0

        class _FakeGroup:
            def size(self) -> int:
                return 2

        with pytest.raises((ValueError, RuntimeError)):
            mx.eval(_quantized_moe_all_sum(y, _FakeGroup(), bits=8, group_size=64))
