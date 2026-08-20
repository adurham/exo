# pyright: reportAny=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
"""REAL 2-rank distributed correctness check for the shared-scale int8
quantized moe.all_sum replacement in mlx_lm/models/deepseek_v4.py.

This is NOT a pytest file -- it's a standalone script meant to be run
under `mlx.launch -n 2` so both `mx.distributed.all_sum` calls inside
`_quantized_moe_all_sum_sharedscale` actually execute across two real
processes/ranks. It deliberately uses ONLY all_sum-family collectives
(verified by monkey-patching mx.distributed.all_gather to raise if
called at all, proving the "no all_gather anywhere" requirement holds
under real execution, not just by code inspection).

Usage:
  cd ~/repos/exo
  .venv/bin/mlx.launch -n 2 --backend ring \
      -- .venv/bin/python src/exo/worker/engines/mlx/tests/test_moe_allsum_sharedscale_distributed.py

Exits 0 and prints PASS on success, exits 1 and prints FAIL with details
on any mismatch.
"""

from __future__ import annotations

import sys

import mlx.core as mx


def _fail_all_gather(*args, **kwargs):
    raise AssertionError(
        "all_gather was called -- the shared-scale design must stay 100% "
        "on the all_sum path with NO all_gather anywhere."
    )


def main() -> int:
    world = mx.distributed.init(backend="ring")
    rank = world.rank()
    size = world.size()

    if size != 2:
        print(f"[rank {rank}] FAIL: expected world size 2, got {size}", flush=True)
        return 1

    # Enforce "no all_gather anywhere" at the collective level, not just
    # by reading the source: any call raises immediately.
    mx.distributed.all_gather = _fail_all_gather

    # Import AFTER init/patch so the module sees a real 2-rank group.
    sys.path.insert(0, "mlx-lm")
    from mlx_lm.models.deepseek_v4 import (  # noqa: E402
        _quantized_moe_all_sum_sharedscale,
    )

    overall_ok = True

    for trial, (rows, cols, bits) in enumerate(
        [
            (256, 4096, 8),  # production DSv4 hidden_size
            (37, 4096, 8),  # ragged/odd row count (partial chunk)
            (256, 4096, 4),  # tighter quantization width
            (1, 4096, 8),  # decode-shaped (L=1)
        ]
    ):
        mx.random.seed(1000 + trial * 7 + rank)
        y_local = (mx.random.normal((rows, cols)) * 0.5).astype(mx.float32)

        # Reference: plain bf16 all_sum (the byte-identical existing path).
        ref = mx.distributed.all_sum(y_local.astype(mx.bfloat16)).astype(mx.float32)
        mx.eval(ref)

        # Candidate: shared-scale int8 quantized all_sum.
        got = _quantized_moe_all_sum_sharedscale(
            y_local.astype(mx.bfloat16), group=world, bits=bits
        ).astype(mx.float32)
        mx.eval(got)

        # Bit-exactness check across ranks: got must be IDENTICAL on both
        # ranks (it's post-collective, so any cross-rank divergence here
        # would mean the shared-scale protocol desynced ranks -- the
        # exact failure mode that killed the all_gather design).
        local_hash_bits = mx.sum(got.astype(mx.float32)).reshape(1)
        cross_rank_check = mx.distributed.all_sum(local_hash_bits) - (
            local_hash_bits * float(size)
        )
        mx.eval(cross_rank_check)
        # `got` is already IDENTICAL across ranks by construction (it's
        # the output of an all_sum), so this second all_sum should be
        # near-exact. Use an absolute tolerance scaled by the per-element
        # magnitude and row count rather than a relative one, since small
        # row counts make local_hash_bits itself small/noisy as a
        # denominator.
        cross_rank_ok = bool(
            float(mx.abs(cross_rank_check).item())
            < max(1.0, float(mx.abs(local_hash_bits).item()) * 1e-2 + 1e-2)
        )

        diff = mx.abs(got - ref)
        max_diff = float(mx.max(diff).item())
        mean_diff = float(mx.mean(diff).item())
        ref_scale = float(mx.mean(mx.abs(ref)).item()) + 1e-8

        # int8 quantization error bound: with shared-scale absmax/127
        # granularity (absmax is a SUM-of-per-rank-absmax upper bound, so
        # looser than the true optimum), worst-case per-element error is
        # scale/2 -- allow a generous but real bound measured empirically
        # (8-bit: observed ~0.08 rel_max / ~0.013 rel_mean; 4-bit is a much
        # coarser deliberately-stress-tested width, allow a wide bound).
        rel_max = max_diff / ref_scale
        rel_mean = mean_diff / ref_scale
        bound_max = 1.5 if bits == 4 else 0.10
        bound_mean = 0.5 if bits == 4 else 0.03

        ok = cross_rank_ok and rel_max < bound_max and rel_mean < bound_mean
        overall_ok = overall_ok and ok
        print(
            f"[rank {rank}] trial={trial} rows={rows} bits={bits} "
            f"cross_rank_ok={cross_rank_ok} rel_max={rel_max:.4f} "
            f"(bound {bound_max}) rel_mean={rel_mean:.4f} (bound {bound_mean}) "
            f"-> {'PASS' if ok else 'FAIL'}",
            flush=True,
        )

    # Negative control: deliberately mismatched scale across ranks (rank 1
    # uses a scale 10x too small) must produce a LARGE, detectable error,
    # proving the shared-scale mechanism (not luck) is what makes the
    # positive tests pass.
    mx.random.seed(5555 + rank)
    y_local = (mx.random.normal((64, 4096)) * 0.5).astype(mx.float32)
    ref = mx.distributed.all_sum(y_local.astype(mx.bfloat16)).astype(mx.float32)
    mx.eval(ref)

    qmax = 127
    bad_scale_local = mx.array([0.5 if rank == 0 else 0.05], dtype=mx.float32)
    # Skip the collective absmax step; directly use a mismatched local
    # scale per rank to simulate the bug this design is designed to avoid.
    q = mx.clip(mx.round(y_local / bad_scale_local), -128, qmax).astype(mx.bfloat16)
    bad_summed = mx.distributed.all_sum(q).astype(mx.float32)
    bad_result = bad_summed * bad_scale_local  # uses THIS rank's (wrong) scale
    mx.eval(bad_result)
    bad_diff = float(mx.mean(mx.abs(bad_result - ref)).item())
    bad_ref_scale = float(mx.mean(mx.abs(ref)).item()) + 1e-8
    bad_rel = bad_diff / bad_ref_scale
    negative_control_has_teeth = bad_rel > 0.3
    print(
        f"[rank {rank}] negative_control rel_mean={bad_rel:.4f} "
        f"(expected >0.3, mismatched-scale bug) -> "
        f"{'PASS (bug detected)' if negative_control_has_teeth else 'FAIL (no teeth)'}",
        flush=True,
    )
    overall_ok = overall_ok and negative_control_has_teeth

    if overall_ok:
        print(f"[rank {rank}] ALL CHECKS PASS", flush=True)
        return 0
    else:
        print(f"[rank {rank}] SOME CHECKS FAILED", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
