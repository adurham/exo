# Copyright © 2026 Adam Durham (hermes-gw)
"""``_qslice``: packed-column arithmetic for tensor parallelism.

The oQ loader must slice *quantized* tensors: the packed weight (uint32 words)
and the scale/bias columns, never a dequantized float. These tests pin the
arithmetic for every mode the pack uses -- affine at 2/3/8 bits (group 64) and
mxfp4/mxfp8 (group 32) -- against MLX's own dequantizer, and pin the refusals
that keep a rank from cutting a quantization group or a word in half.
"""

from __future__ import annotations

import unittest

import numpy as np
import mlx.core as mx

from synth import np_of
from mlx_lm.models.deepseek_v41 import oq_build as oq

MODES = [("affine", 2, 64), ("affine", 3, 64), ("affine", 8, 64),
         ("mxfp4", 4, 32), ("mxfp8", 8, 32)]
IN_F, OUT_F = 512, 256


def _q(mode: str, bits: int, group: int, shape=(OUT_F, IN_F), seed=0):
    mx.random.seed(seed)
    w = (mx.random.normal(shape) * 0.05).astype(mx.bfloat16)
    q = mx.quantize(w, group_size=group, bits=bits, mode=mode)
    mx.eval(*q)
    return q[0], q[1], (q[2] if len(q) > 2 else None)


class QSliceTest(unittest.TestCase):
    def test_in_axis_matches_dequantized_columns(self):
        """A sliced packed tensor dequantizes to exactly the full one's columns."""
        for mode, bits, group in MODES:
            w, s, b = _q(mode, bits, group)
            full = mx.dequantize(w, s, b, group_size=group, bits=bits, mode=mode)
            mx.eval(full)
            lo, hi = 128, 384
            sw, ss, sb = oq._qslice(w, s, b, bits=bits, group=group, mode=mode,
                                    axis="in", lo=lo, hi=hi)
            got = mx.dequantize(sw, ss, sb, group_size=group, bits=bits,
                                mode=mode)
            mx.eval(got)
            self.assertEqual(tuple(got.shape), (OUT_F, hi - lo))
            self.assertTrue(
                np.array_equal(np_of(got), np_of(full[:, lo:hi])),
                f"{mode}{bits}: in-axis slice is not the full tensor's columns")
            # packed columns follow elems*bits/32 exactly
            self.assertEqual(sw.shape[-1], hi * bits // 32 - lo * bits // 32)
            self.assertEqual(ss.shape[-1], hi // group - lo // group)

    def test_out_axis_matches_dequantized_rows(self):
        for mode, bits, group in MODES:
            w, s, b = _q(mode, bits, group)
            full = mx.dequantize(w, s, b, group_size=group, bits=bits, mode=mode)
            mx.eval(full)
            lo, hi = 128, 192
            sw, ss, sb = oq._qslice(w, s, b, bits=bits, group=group, mode=mode,
                                    axis="out", lo=lo, hi=hi)
            got = mx.dequantize(sw, ss, sb, group_size=group, bits=bits,
                                mode=mode)
            mx.eval(got)
            self.assertEqual(tuple(got.shape), (hi - lo, IN_F))
            self.assertTrue(np.array_equal(np_of(got), np_of(full[lo:hi])))

    def test_rank_slices_tile_the_full_tensor(self):
        """Concatenating the ranks' slices rebuilds the whole packed tensor."""
        for mode, bits, group in MODES:
            w, s, b = _q(mode, bits, group)
            for axis in ("in", "out"):
                n = IN_F if axis == "in" else OUT_F
                parts = [oq._qslice(w, s, b, bits=bits, group=group, mode=mode,
                                    axis=axis, lo=r * n // 4, hi=(r + 1) * n // 4)
                         for r in range(4)]
                ws = [p[0] for p in parts]
                mx.eval(*ws)
                cat = mx.concatenate(ws, axis=-2 if axis == "out" else -1)
                mx.eval(cat)
                self.assertTrue(mx.array_equal(cat, w),
                                f"{mode}{bits} {axis}: slices do not rebuild")

    def test_rejects_unaligned_slice(self):
        """A boundary inside a group/word is refused, not silently rounded."""
        w, s, b = _q("affine", 4, 64)
        for lo, hi in ((32, 128), (0, 96), (64, 192 + 32)):
            with self.assertRaises(ValueError):
                oq._qslice(w, s, b, bits=4, group=64, mode="affine",
                           axis="in", lo=lo, hi=hi)
        # mxfp4: lcm(32, 32) = 32, so 32 is legal there but 16 is not
        wm, sm, bm = _q("mxfp4", 4, 32)
        oq._qslice(wm, sm, bm, bits=4, group=32, mode="mxfp4", axis="in",
                   lo=32, hi=64)
        with self.assertRaises(ValueError):
            oq._qslice(wm, sm, bm, bits=4, group=32, mode="mxfp4", axis="in",
                       lo=16, hi=64)

    def test_rejects_out_of_range_and_bad_axis(self):
        w, s, b = _q("affine", 8, 64)
        with self.assertRaises(ValueError):
            oq._qslice(w, s, b, bits=8, group=64, mode="affine", axis="in",
                       lo=0, hi=IN_F + 128)
        with self.assertRaises(ValueError):
            oq._qslice(w, s, b, bits=8, group=64, mode="affine", axis="diag",
                       lo=0, hi=128)
        with self.assertRaises(ValueError):
            oq._qslice(w, s, b, bits=8, group=64, mode="affine", axis="out",
                       lo=128, hi=128)

    def test_rejects_scales_that_do_not_describe_the_weight(self):
        """A pack whose scales disagree with the packed geometry fails loudly."""
        w, s, b = _q("affine", 8, 64)
        with self.assertRaises(ValueError):
            oq._qslice(w, s[: OUT_F // 2], None if b is None else b[: OUT_F // 2],
                       bits=8, group=64, mode="affine", axis="in", lo=0, hi=128)

    def test_stacked_expert_slice_keeps_expert_axis(self):
        """3-D stacks slice along out/in only; the expert axis is untouched."""
        w, s, b = _q("affine", 2, 64, shape=(4, OUT_F, IN_F))
        sw, ss, sb = oq._qslice(w, s, b, bits=2, group=64, mode="affine",
                                axis="in", lo=128, hi=256)
        self.assertEqual(sw.shape, (4, OUT_F, 128 * 2 // 32))
        self.assertEqual(ss.shape, (4, OUT_F, 2))
        sw2, ss2, sb2 = oq._qslice(w, s, b, bits=2, group=64, mode="affine",
                                   axis="out", lo=128, hi=192)
        self.assertEqual(sw2.shape, (4, 64, IN_F * 2 // 32))
        full = mx.dequantize(w, s, b, group_size=64, bits=2, mode="affine")
        part = mx.dequantize(sw2, ss2, sb2, group_size=64, bits=2, mode="affine")
        mx.eval(full, part)
        self.assertTrue(np.array_equal(np_of(part), np_of(full[:, 128:192, :])),
                        "expert out-slice is not the full stack's rows")


class AffineProjFromQuantizedTest(unittest.TestCase):
    def test_matches_from_weight_on_the_same_values(self):
        """``from_quantized`` == ``from_weight`` of the same (dequantized) weight.

        This is the orientation/group guard: a transposed or group-permuted
        tensor would produce a completely different projection.
        """
        for mode, bits, group in MODES:
            w, s, b = _q(mode, bits, group)
            deq = mx.dequantize(w, s, b, group_size=group, bits=bits, mode=mode)
            mx.eval(deq)
            packed = oq.AffineProj.from_quantized(w, s, b, bits=bits, group=group,
                                                  mode=mode)
            if mode == "affine":
                ref = oq.AffineProj.from_weight(mx.contiguous(deq), bits, group)
                n = 8
                x = (mx.random.normal((n, IN_F)) * 0.1).astype(mx.bfloat16)
                a = packed(x)
                r = ref(x)
                mx.eval(a, r)
                cos = float(mx.sum(a * r) / mx.sqrt(mx.sum(a * a) * mx.sum(r * r)))
                self.assertGreater(cos, 0.999, f"{mode}{bits} cosine {cos}")

    def test_refuses_inconsistent_mode_and_biases(self):
        w, s, b = _q("affine", 8, 64)
        with self.assertRaises(ValueError):
            oq.AffineProj.from_quantized(w, s, None, bits=8, group=64,
                                         mode="affine")
        wm, sm, _ = _q("mxfp4", 4, 32)
        with self.assertRaises(ValueError):
            oq.AffineProj.from_quantized(wm, sm, b, bits=4, group=32,
                                         mode="mxfp4")

    def test_activation_dtype_follows_scales(self):
        """bf16 scales -> bf16 activations -> bf16 output (MLX's fast path)."""
        w, s, b = _q("affine", 4, 64)
        proj = oq.AffineProj.from_quantized(w, s, b, bits=4, group=64)
        self.assertEqual(proj._act, mx.bfloat16)
        y = proj(mx.zeros((2, IN_F), dtype=mx.bfloat16))
        mx.eval(y)
        self.assertEqual(y.dtype, mx.bfloat16)


if __name__ == "__main__":
    unittest.main()
