"""CI-resident regression guard for the fused ``hc_expand`` Metal kernel.

Mirrors ``mlx-lm/tests/test_hc_expand_kernel.py`` from the adurham/mlx-lm
fork into exo's OWN test tree, converted from a standalone ``__main__``
script into real pytest test functions.

Why the mirror exists
---------------------
The upstream copy lives inside the ``mlx-lm`` submodule. ``pipeline.yml``
scopes pytest to ``src`` and never checks out submodules, so the submodule
copy has never executed in CI even once (same gap class fixed for
``BatchPoolingCache`` in commit ``8158c0f52``). Rather than pull mlx-lm's
whole suite in, the narrow correctness guard is mirrored here.

What it actually guards
------------------------
It imports ``_hc_expand_op`` and ``_make_hc_expand_kernel`` from the
**installed** ``mlx_lm`` — the one resolved by ``uv.lock`` /
``[tool.uv.sources]`` — not from the submodule source tree. That makes it
strictly stronger than the submodule copy: it fails both when the fused
kernel regresses in mlx-lm AND when exo's mlx-lm pin drifts back to a
revision that predates the kernel (in which case
``_make_hc_expand_kernel`` may be absent or return ``None``, and the tests
below fail loudly rather than silently skipping).

The fused ``hc_expand`` kernel is a custom Metal kernel (built via
``mx.fast.metal_kernel``) that computes the hyper-connection expand step in
a single dispatch instead of the reference ``_hc_expand_op`` composition of
ops. It must stay numerically equivalent to the reference within bf16
rounding tolerance across the shapes exercised below (production prefill,
an asymmetric deterministic case that catches transpose bugs, HC=2 template
generality, and decode L=1).

GPU/Metal requirement
----------------------
The kernel requires an actual Metal device. CI's pytest step
(``.github/workflows/pipeline.yml``) runs ONLY on the ``macos-26`` runner
matrix leg (``if: runner.os == 'macOS'``) and its own comment states it
needs GPU access for MLX — the runner has real Metal hardware. This test
module intentionally does NOT skip when Metal is unavailable; it FAILS,
because a silent skip would be a decorative guard. If this ever fails with
"Metal GPU not available" that is a signal the CI runner regressed, not a
reason to accept a skip.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import mlx.core as mx
import pytest
from mlx_lm.models.hyper_connection import (
    _hc_expand_op,  # pyright: ignore[reportPrivateUsage]
    _make_hc_expand_kernel,  # pyright: ignore[reportPrivateUsage]
)


def _require_metal() -> None:
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        pytest.fail(
            "Metal GPU not available -- this guard must EXECUTE, not skip, "
            "on CI's macOS runner. A missing GPU here means the runner "
            "environment regressed."
        )


def _rel_err(a: mx.array, b: mx.array) -> float:
    a_f = a.astype(mx.float32)
    b_f = b.astype(mx.float32)
    denom = mx.maximum(mx.abs(b_f), mx.array(1e-6))
    return float((mx.abs(a_f - b_f) / denom).mean())


def _max_abs(a: mx.array, b: mx.array) -> float:
    return float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max())


def _has_nan_inf(a: mx.array) -> bool:
    return bool(mx.any(mx.isnan(a) | mx.isinf(a)))


def _kernel_call(
    x: mx.array, residual: mx.array, post: mx.array, comb: mx.array
) -> mx.array:
    """Invoke the fused hc_expand Metal kernel with the shapes derived
    from the given inputs, mirroring the upstream script's dispatch."""
    kernel = _make_hc_expand_kernel()
    if kernel is None:
        pytest.fail("Metal GPU not available; kernel unbuildable.")

    batch, length, num_hc, dim = residual.shape
    call = cast(
        Callable[..., tuple[mx.array]],
        kernel,
    )
    (out,) = call(
        inputs=[x, residual, post, comb],
        template=[
            ("T", x.dtype),
            ("U", x.dtype),
            ("HC", num_hc),
            ("D", dim),
        ],
        grid=(batch * length * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(batch, length, num_hc, dim)],
        output_dtypes=[x.dtype],
    )
    return out


def test_hc_expand_kernel_realistic_production_shape() -> None:
    """Production prefill shape (B=1, L=2048, HC=4, D=4096) at realistic
    magnitudes must match the reference within bf16 rounding tolerance."""
    _require_metal()
    batch, length, num_hc, dim = 1, 2048, 4, 4096
    scale = 2.2
    mx.random.seed(0)
    x = (mx.random.normal(shape=(batch, length, dim)) * scale).astype(
        mx.bfloat16
    )
    residual = (
        mx.random.normal(shape=(batch, length, num_hc, dim)) * scale
    ).astype(mx.bfloat16)
    post = mx.random.uniform(-1, 1, shape=(batch, length, num_hc)).astype(
        mx.float32
    )
    comb = mx.random.uniform(
        -1, 1, shape=(batch, length, num_hc, num_hc)
    ).astype(mx.float32)

    ref = _hc_expand_op(x, residual, post, comb)
    got = _kernel_call(x, residual, post, comb)
    mx.eval(ref, got)

    rel = _rel_err(got, ref)
    assert rel <= 1e-3, f"mean relative error {rel:.3e} exceeds tolerance"
    assert not _has_nan_inf(got), "kernel output contains NaN/Inf"


def test_hc_expand_kernel_asymmetric_deterministic_catches_transpose_bugs() -> (
    None
):
    """Asymmetric deterministic input (distinct strides per axis) --
    catches transpose/axis-order bugs that random symmetric inputs miss."""
    _require_metal()
    batch, length, num_hc, dim = 1, 3, 4, 8
    x = (
        mx.arange(batch * length * dim, dtype=mx.float32).reshape(
            batch, length, dim
        )
        * 0.01
    ).astype(mx.bfloat16)
    residual = (
        mx.arange(batch * length * num_hc * dim, dtype=mx.float32).reshape(
            batch, length, num_hc, dim
        )
        * 0.005
    ).astype(mx.bfloat16)
    post = (
        mx.arange(batch * length * num_hc, dtype=mx.float32).reshape(
            batch, length, num_hc
        )
        * 0.1
    )
    comb = (
        mx.arange(
            batch * length * num_hc * num_hc, dtype=mx.float32
        ).reshape(batch, length, num_hc, num_hc)
        * 0.05
    )

    ref = _hc_expand_op(x, residual, post, comb)
    got = _kernel_call(x, residual, post, comb)
    mx.eval(ref, got)

    rel = _rel_err(got, ref)
    mabs = _max_abs(got, ref)
    assert rel <= 1e-3, f"mean relative error {rel:.3e} exceeds tolerance"
    assert mabs < 1e-1, f"max abs error {mabs:.3e} exceeds tolerance"


def test_hc_expand_kernel_hc_equals_two_template_generality() -> None:
    """HC=2 exercises a different template instantiation than the HC=4
    production shape -- the kernel is templated on HC, not hardcoded."""
    _require_metal()
    dim = 4096
    scale = 2.2
    x = (mx.random.normal(shape=(1, 64, dim)) * scale).astype(mx.bfloat16)
    residual = (mx.random.normal(shape=(1, 64, 2, dim)) * scale).astype(
        mx.bfloat16
    )
    post = mx.random.uniform(-1, 1, shape=(1, 64, 2)).astype(mx.float32)
    comb = mx.random.uniform(-1, 1, shape=(1, 64, 2, 2)).astype(mx.float32)

    ref = _hc_expand_op(x, residual, post, comb)
    got = _kernel_call(x, residual, post, comb)
    mx.eval(ref, got)

    rel = _rel_err(got, ref)
    assert rel <= 1e-3, f"mean relative error {rel:.3e} exceeds tolerance"


def test_hc_expand_kernel_decode_length_one_small_input_build_path() -> None:
    """L=1 is the decode-step shape and exercises a distinct small-input
    kernel build path from the prefill shapes above."""
    _require_metal()
    batch, length, num_hc, dim = 1, 1, 4, 4096
    scale = 2.2
    x = (mx.random.normal(shape=(batch, length, dim)) * scale).astype(
        mx.bfloat16
    )
    residual = (
        mx.random.normal(shape=(batch, length, num_hc, dim)) * scale
    ).astype(mx.bfloat16)
    post = mx.random.uniform(-1, 1, shape=(batch, length, num_hc)).astype(
        mx.float32
    )
    comb = mx.random.uniform(
        -1, 1, shape=(batch, length, num_hc, num_hc)
    ).astype(mx.float32)

    ref = _hc_expand_op(x, residual, post, comb)
    got = _kernel_call(x, residual, post, comb)
    mx.eval(ref, got)

    rel = _rel_err(got, ref)
    assert rel <= 1e-3, f"mean relative error {rel:.3e} exceeds tolerance"
