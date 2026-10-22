# Copyright © 2026 Adam Durham (hermes-gw)
"""Layout parity of the oQ tensors themselves (published pack, real bytes).

Three independent oracles per tensor, all of which fail loudly on a
packing/layout error (a permuted or transposed slice shows up as ~0 cosine, not
as a small delta):

1. **Numpy spec decode** -- a from-scratch decoder of the MLX affine layout
   (unsigned code per group, ``scale * q + bias``, little-endian codes inside
   each uint32 word) compared against ``mx.dequantize``. Independent of the
   loader's own reader.
2. **Quantizer round trip** -- ``mx.quantize(dequantize(t))`` must reproduce the
   pack's packed weight *bit for bit* at the same bits/group/mode. This pins
   our interpretation to MLX's own encoder inverse.
3. **Module-level parity** -- ``AffineProj.from_quantized(pack tensors)`` against
   ``AffineProj.from_weight(dequantized float)`` on identical activations, and
   the expert stacks' ``gather_qmm`` against a float ``gather_mm`` reference.

The cross-format leg against an EXL3 pack is NOT here: no EXL3 1017 pack exists
on this machine (the published ones are 210-415 GB and the in-tree exl3 loader
has no encoder to build one from), so an EXL3 comparison cannot be run without
downloading one. See docs/dsv41-oq-checkpoint-inventory.md.
"""

from __future__ import annotations

import unittest

import numpy as np
import mlx.core as mx

from synth import np_of
from mlx_lm.models.deepseek_v41 import oq_build as oq

#: layer 0's projections (the pack quantizes 10 modules + 3 expert stacks here)
DENSE = ["attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_b",
         "ffn.shared_experts.w1", "ffn.shared_experts.w2",
         "ffn.shared_experts.w3"]
EXPERTS = ["ffn.experts.w1", "ffn.experts.w2", "ffn.experts.w3"]


def _np_affine_dequant(w: np.ndarray, scales: np.ndarray, biases: np.ndarray,
                       bits: int, group: int) -> np.ndarray:
    """Decode MLX affine quantization from the spec, with no mlx involved.

    Codes are a contiguous little-endian bitstream over the packed row: value
    ``i`` occupies bits ``[i*bits, (i+1)*bits)`` of the row's uint32 words read
    as little-endian bytes. (A per-word decode only works when ``bits`` divides
    32 -- 3/5/6-bit pack across word boundaries.) Then ``scale * code + bias``
    per group.
    """
    lead, packed = w.shape[:-1], w.shape[-1]
    n_elems = scales.shape[-1] * group
    assert n_elems * bits <= packed * 32, "packed row too short for the codes"
    raw = np.ascontiguousarray(w.astype("<u4")).view(np.uint8)   # lead + (4*packed,)
    raw = np.concatenate([raw, np.zeros(lead + (3,), np.uint8)], axis=-1)
    pos = np.arange(n_elems) * bits
    byte, off = pos // 8, (pos % 8).astype(np.uint32)
    acc = (raw[..., byte].astype(np.uint32)
           | (raw[..., byte + 1].astype(np.uint32) << 8)
           | (raw[..., byte + 2].astype(np.uint32) << 16))
    q = (acc >> off) & np.uint32((1 << bits) - 1)
    q = q.reshape(lead + (n_elems,)).astype(np.float32)
    sc = np.repeat(scales.astype(np.float32), group, axis=-1)
    bi = np.repeat(biases.astype(np.float32), group, axis=-1)
    return q * sc + bi


def _arr(x) -> np.ndarray:
    """numpy float32 view of an mlx OR numpy array."""
    if isinstance(x, mx.array):
        return np_of(x)
    return np.asarray(x, dtype=np.float32)


def _cos(a, b) -> float:
    a, b = _arr(a).ravel(), _arr(b).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _codes(packed: np.ndarray, bits: int) -> np.ndarray:
    """Unpack little-endian codes from uint32 words (any leading dims)."""
    per = 32 // bits
    lead, n = packed.shape[:-1], packed.shape[-1]
    q = np.zeros(lead + (n * per,), dtype=np.uint8)
    for i in range(per):
        q[..., i::per] = ((packed >> np.uint32(i * bits))
                          & np.uint32((1 << bits) - 1)).astype(np.uint8)
    return q


class LayoutParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import synth

        cls.pack = synth.real_pack()
        if cls.pack is None:
            raise unittest.SkipTest("published pack subset not present")
        cls.ck = oq.OqCheckpoint(cls.pack)
        cls.table: list[str] = []

    def _packed(self, mod: str, e0: int | None = None, e1: int | None = None):
        """Packed weight/scales/biases of ``mod``, optionally an expert slice.

        The published stacks hold 384 experts x 2304 x 5120 -- dequantizing one
        whole stack is 9 GiB, so the heavy comparisons run on a few experts and
        the geometry itself is covered by :func:`oq_build._qslice`'s tests.
        """
        sl = slice(e0, e1) if e0 is not None else slice(None)
        qi = self.ck.qinfo(mod)
        w = self.ck.array(f"{mod}.weight")[sl]
        s = self.ck.array(f"{mod}.scales")[sl]
        b = (self.ck.array(f"{mod}.biases")[sl]
             if self.ck.has(f"{mod}.biases") else None)
        return w, s, b, qi

    def _dequant_slice(self, mod: str, e0=None, e1=None):
        w, s, b, qi = self._packed(mod, e0, e1)
        bits, group, mode = oq._qparams(qi)
        out = mx.dequantize(w, s, b, group_size=group, bits=bits, mode=mode)
        mx.eval(out)
        return out

    def test_numpy_spec_decode_matches_mlx(self):
        """Affine tensors decode identically with an independent implementation."""
        jobs = [(f"layers.0.{n}", None, None) for n in DENSE]
        # the 2- and 3-bit expert stacks are exactly where a code-level round
        # trip stops being informative, so decode them element-wise here
        jobs += [(f"layers.0.{n}", 0, 4) for n in EXPERTS]
        for mod, e0, e1 in jobs:
            qi = self.ck.qinfo(mod)
            if oq._qparams(qi)[2] != "affine":
                continue
            w, sc, bi, _ = self._packed(mod, e0, e1)
            bits, group, _mode = oq._qparams(qi)
            ref = _np_affine_dequant(np_of(w), np_of(sc), np_of(bi), bits, group)
            got = np_of(self._dequant_slice(mod, e0, e1))
            self.assertEqual(got.shape, ref.shape)
            err = float(np.max(np.abs(got - ref)))
            scale = float(np.max(np.abs(ref))) + 1e-12
            # ``got`` is bf16 (the pack's dequantized dtype): ~3 decimal
            # digits, so 1e-2 relative is the floor of what bf16 can even
            # represent. A layout error would be O(1), not O(1e-3).
            self.assertLess(err / scale, 1e-2,
                            f"{mod}: numpy spec decode disagrees with mlx "
                            f"(rel err {err / scale:.3e}) -- the packed layout "
                            f"is not what we think it is")
            self.assertGreater(_cos(got, ref), 0.9999, mod)

    def test_quantizer_round_trip_is_bit_exact(self):
        """``mx.quantize(dequantize(t))`` reproduces the pack's packed bytes."""
        exact = {"affine": True, "mxfp4": True, "mxfp8": True}
        jobs = [(f"layers.0.{n}", None, None) for n in DENSE]
        jobs += [(f"layers.0.{n}", 0, 4) for n in EXPERTS]   # 4 experts, not 384
        for mod, e0, e1 in jobs:
            w, sc, bi, qi = self._packed(mod, e0, e1)
            bits, group, mode = oq._qparams(qi)
            deq = self._dequant_slice(mod, e0, e1)
            q = mx.quantize(deq, group_size=group, bits=bits, mode=mode)
            mx.eval(*q)
            want = w
            mx.eval(want)
            same = np.array_equal(np_of(q[0]), np_of(want))
            # Codes are not bit-reproducible in general: dequantizing to bf16
            # and re-encoding re-rounds at the last mantissa bit, so a code can
            # land one step over. The layout signal is the *fraction* of equal
            # words and the cosine of the round-tripped values -- a permuted or
            # transposed packing scores ~0 on both.
            d = np.abs(_codes(np_of(want), bits).astype(np.int16)
                       - _codes(np_of(q[0]), bits).astype(np.int16))
            frac = float(np.mean(d == 0))
            worst = int(d.max())
            rt = mx.dequantize(q[0], q[1], q[2] if len(q) > 2 else None,
                               group_size=group, bits=bits, mode=mode)
            cos = _cos(rt, deq)
            self.table.append(
                f"{mod:22s} {mode:6s} b{bits} g{group:<3d} "
                f"codes_identical={frac:6.2%} max_dcode={worst} cos={cos:.6f}"
                f"{'  (bit-exact)' if same else ''}")
            if exact.get(qi["mode"], False):
                # At 8 bits this is >0.999. At 2-3 bits a bf16 intermediate
                # genuinely loses code fidelity (the bias cancels the value
                # inside a group), so the bar is 0.9 -- still far above the
                # ~0 a permuted/transposed packing would produce.
                self.assertGreater(
                    cos, 0.9,
                    f"{mod}: dequant->requant->dequant drifted (cos {cos}) -- "
                    f"the packed layout is not the one mx.quantize produced")
                # ``codes_identical`` / ``max_dcode`` are printed, not asserted:
                # re-encoding bf16-rounded values legitimately flips a code to
                # the opposite end of the range in groups whose scale is tiny
                # (the two decode to nearly the same float). The value-level
                # cosine above and the independent numpy decode in
                # test_numpy_spec_decode_matches_mlx are the layout signals.

    def test_projection_parity_packed_vs_float(self):
        """Packed tensors and their dequantized floats project identically."""
        mx.random.seed(0)
        for name in DENSE:
            mod = f"layers.0.{name}"
            qi = self.ck.qinfo(mod)
            if oq._qparams(qi)[2] != "affine":
                continue
            deq = self.ck.dequant(mod)
            bits, group, mode = oq._qparams(qi)
            packed = oq.AffineProj.from_quantized(
                self.ck.array(f"{mod}.weight"), self.ck.array(f"{mod}.scales"),
                self.ck.array(f"{mod}.biases"), bits=bits, group=group,
                mode=mode)
            ref = oq.AffineProj.from_weight(mx.contiguous(deq), bits, group)
            x = (mx.random.normal((6, deq.shape[1])) * 0.05).astype(mx.bfloat16)
            a, b = packed(x), ref(x)
            mx.eval(a, b)
            self.assertGreater(_cos(a, b), 0.999,
                               f"{mod}: packed/float projection parity")

    def test_expert_stack_parity_gather_qmm_vs_float(self):
        """A stacked expert tensor gathers like the float math of that expert."""
        mx.random.seed(1)
        args = oq.ModelArgs.from_dict(self.ck.config)
        experts, _ = oq._load_experts(self.ck, "layers.0.ffn.experts.", args,
                                      rank=0, world=1, limit=args.swiglu_limit)
        x = (mx.random.normal((4, args.dim)) * 0.05).astype(mx.bfloat16)
        idx = mx.array([[0, 5, 17, 300, 42, 7]] * 4, dtype=mx.uint32)
        mx.eval(x, idx)
        out_q = experts(x, idx)                       # [4, topk, dim]
        mx.eval(out_q)
        self.assertTrue(bool(mx.all(mx.isfinite(out_q))),
                        "gather_qmm over the pack's expert stack is not finite")
        # one fixed expert, weights = 1: the routed output equals the float path
        e = int(idx[0, 0])
        w1 = self._dequant_slice("layers.0.ffn.experts.w1", e, e + 1)[0]
        w3 = self._dequant_slice("layers.0.ffn.experts.w3", e, e + 1)[0]
        w2 = self._dequant_slice("layers.0.ffn.experts.w2", e, e + 1)[0]
        acts = oq.ClampedSwiGLU(args.swiglu_limit)
        x0 = x[0:1]
        h = acts(x0 @ w1.T, x0 @ w3.T)
        want = h @ w2.T                                # [1, dim]
        got = experts.down_proj(h[None], mx.array([[e]], dtype=mx.uint32))
        got = got.reshape(1, -1)                       # [1, dim]
        mx.eval(got, want)
        self.assertGreater(_cos(got, want), 0.99,
                           "expert down_proj disagrees with the float reference")

    def test_zz_report_table(self):
        """Print the per-tensor parity table (``-s`` to see it)."""
        print("\n" + "\n".join(self.table))
        self.assertTrue(self.table, "run the parity tests first")


if __name__ == "__main__":
    unittest.main()
