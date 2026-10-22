# Copyright © 2026 Adam Durham (hermes-gw)
"""End-to-end block and draft-head checks on the published pack.

Runs the REAL ``Block`` built by ``oq_build.build_block`` from layer 0 of the
published pack (all 10 quantized modules + the 3 routed expert stacks), and
compares it against a reference block whose dense quantized projections are
rebuilt from their *dequantized* floats through the existing
``AffineProj.from_weight`` path. A layout/orientation/grouping error in the
packed reader moves one block far away from the other; identical packing keeps
them at ~1.0.

The routed-expert stacks keep the pack's packed tensors in both blocks --
dequantizing 13.5 G values would cost +27 GiB -- so their float parity is
asserted per expert in ``test_layout_parity.py``
(``gather_qmm`` vs a float ``gather_mm``).

The draft head is checked too: the pack stores it in MXFP (mxfp4 experts,
mxfp8 dense), which is the path that would silently break if anything assumed
affine.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from synth import np_of
from mlx_lm.models.deepseek_v41 import oq_build as oq
from mlx_lm.models.deepseek_v41.cache import ModelCache
from mlx_lm.models.deepseek_v41.config import ModelArgs
from mlx_lm.models.deepseek_v41.hyper_connections import make_identity_pre_mix
from mlx_lm.models.deepseek_v41.model import SharedState


def _cos(a, b) -> float:
    a = np_of(a).ravel() if isinstance(a, mx.array) else np.asarray(a).ravel()
    b = np_of(b).ravel() if isinstance(b, mx.array) else np.asarray(b).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


class RefSwitchLinear(nn.Module):
    """Float reference for an expert stack: dequantize only the *touched* experts.

    Dequantizing all 384 experts of the published 1017 stacks would be 9 GiB per
    projection (27 GiB for w1/w2/w3), so the reference materializes just the
    experts a batch routes to and gathers those -- which is also what the
    packed ``gather_qmm`` computes, per expert.
    """

    def __init__(self, weight, scales, biases, *, bits: int, group_size: int,
                 mode: str):
        super().__init__()
        # named weight/scales/biases so the loader's parameter-tree
        # accounting sees the same checkpoint tensors as the real module
        self.weight, self.scales, self.biases = weight, scales, biases
        self.bits, self.group_size, self.mode = bits, group_size, mode

    def __call__(self, x, indices, sorted_indices: bool = False):
        ids = np.array(indices).reshape(-1)
        uniq = np.unique(ids)
        sel = mx.array(uniq, dtype=mx.uint32)
        w = mx.dequantize(self.weight[sel], self.scales[sel],
                          None if self.biases is None else self.biases[sel],
                          group_size=self.group_size, bits=self.bits,
                          mode=self.mode)
        w = mx.contiguous(w.swapaxes(-1, -2))               # [U, in, out]
        pos = mx.array(np.searchsorted(uniq, np.array(indices)).astype(np.uint32))
        return mx.gather_mm(x, w, rhs_indices=pos, sorted_indices=False)


def ref_quant_proj(ck, module: str, *, axis: str | None = None, lo: int = 0,
                   hi: int = 0):
    """Float reference for one dense projection (affine, same bits/group)."""
    qi = ck.qinfo(module)
    bits, group, mode = oq._qparams(qi)
    if mode != "affine":
        raise AssertionError(f"{module}: non-affine, no float reference")
    w = ck.dequant(module)
    if axis == "out":
        w = w[lo:hi]
    elif axis == "in":
        w = w[:, lo:hi]
    return oq.AffineProj.from_weight(mx.contiguous(w), bits, group)


class BlockForwardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import synth

        pack = synth.real_pack()
        if pack is None:
            raise unittest.SkipTest("published pack subset not present")
        cls.ck = oq.OqCheckpoint(pack)
        cls.args = ModelArgs.from_dict(cls.ck.config)
        cls.blk, cls.rep = oq.build_block(cls.ck, cls.args, 0)
        with mock.patch.object(oq, "_quant_proj", ref_quant_proj), \
                mock.patch.object(oq, "OqSwitchLinear", RefSwitchLinear):
            cls.ref, _ = oq.build_block(cls.ck, cls.args, 0)

    def _input(self, b: int = 1, s: int = 4, seed: int = 0) -> mx.array:
        mx.random.seed(seed)
        return (mx.random.normal((b, s, self.args.hc_mult, self.args.dim))
                * 0.05).astype(mx.bfloat16)

    def test_accounting_is_exact(self):
        rep = self.rep
        self.assertEqual(rep["wo_a"], "plain")
        self.assertEqual(rep["bias_vl"], "checkpoint")
        self.assertGreater(rep["tensors"], 40)
        self.assertEqual(rep["tp"], {"attn": False, "shared": False})

    def test_block_forward_is_finite(self):
        x = self._input()
        pre = make_identity_pre_mix(1, 4, self.args.hc_mult)
        out, ffn_pre = self.blk(x, pre, 0, ModelCache(self.args, 1, 64),
                                SharedState())
        mx.eval(out, ffn_pre)
        self.assertEqual(tuple(out.shape), (1, 4, self.args.hc_mult, self.args.dim))
        self.assertTrue(bool(mx.all(mx.isfinite(out))), "block output not finite")
        self.assertGreater(float(mx.max(mx.abs(out))), 0.0)
        # deterministic: a second run is bit-identical
        out2, _ = self.blk(x, pre, 0, ModelCache(self.args, 1, 64),
                           SharedState())
        mx.eval(out2)
        self.assertTrue(bool(mx.all(out == out2)))

    def test_block_matches_requantized_float_reference(self):
        x = self._input(seed=3)
        pre = make_identity_pre_mix(1, 4, self.args.hc_mult)
        a, _ = self.blk(x, pre, 0, ModelCache(self.args, 1, 64), SharedState())
        b, _ = self.ref(x, pre, 0, ModelCache(self.args, 1, 64), SharedState())
        mx.eval(a, b)
        cos = _cos(a, b)
        self.assertGreater(cos, 0.99,
                           f"oQ block vs float-requantized block: cos {cos:.5f}")
        rel = float(mx.max(mx.abs(a - b))) / (float(mx.max(mx.abs(b))) + 1e-9)
        self.assertLess(rel, 0.15, f"max relative deviation {rel:.3f}")

    def test_head_is_plain_and_matches_its_weight(self):
        lin = oq._build_head(self.ck, self.args, rank=0, world=1, group=None)
        w = self.ck.array("head.weight")
        mx.eval(w)
        self.assertTrue(np.array_equal(np_of(lin.weight), np_of(w.astype(lin.weight.dtype))))
        h = (mx.random.normal((2, self.args.dim)) * 0.02).astype(mx.bfloat16)
        y = lin(h)
        ref = h.astype(mx.float32) @ w.astype(mx.float32).T
        mx.eval(y, ref)
        self.assertGreater(_cos(y, ref), 0.999,
                           "head logits disagree with the plain bf16 weight")

    # -- the draft head (MXFP path) --------------------------------------
    def test_mtp_modules_use_their_own_modes(self):
        args = self.args
        experts, consumed = oq._load_experts(self.ck, "mtp.0.ffn.experts.",
                                             args, rank=0, world=1,
                                             limit=args.swiglu_limit)
        for proj in (experts.gate_proj, experts.up_proj, experts.down_proj):
            self.assertEqual(proj.mode, "mxfp4")
            self.assertEqual((proj.bits, proj.group_size), (4, 32))
            self.assertIsNone(proj.biases)
            self.assertEqual(proj.weight.shape[0], 128)   # draft is 128 experts
        self.assertIn("mtp.0.ffn.experts.w1.weight", consumed)
        x = (mx.random.normal((4, args.dim)) * 0.02).astype(mx.bfloat16)
        idx = mx.array([[0, 1, 2]] * 4, dtype=mx.uint32)
        y = experts(x, idx)
        mx.eval(y)
        self.assertTrue(bool(mx.all(mx.isfinite(y))), "mxfp4 gmm not finite")

        proj = oq._quant_proj(self.ck, "mtp.0.attn.wq_a")
        self.assertEqual(proj._mode, "mxfp8")
        self.assertEqual((proj._bits, proj._group), (8, 32))
        self.assertIsNone(proj.biases)
        self.assertEqual(proj._act, mx.bfloat16)
        y = proj((mx.random.normal((3, args.dim)) * 0.02).astype(mx.bfloat16))
        mx.eval(y)
        self.assertTrue(bool(mx.all(mx.isfinite(y))))

    def test_build_mtp_assembles_one_stage(self):
        """One DSpark stage from the pack (mtp.2's markov/conf are absent here)."""
        args = ModelArgs.from_dict(self.ck.config)
        args.n_mtp_layers = 1                 # only mtp.0 is in the local subset
        head = oq.build_mtp(self.ck, args, rank=0, world=1)
        self.assertEqual(len(head.stages), 1)
        st = head.stages[0]
        self.assertIsInstance(st.ffn.experts, oq.OqExperts)
        self.assertEqual(st.ffn.experts.gate_proj.mode, "mxfp4")
        self.assertIsInstance(st.attn.wq_a, oq.AffineProj)
        self.assertEqual(st.attn.wq_a._mode, "mxfp8")
        self.assertIsInstance(head.main_proj, oq.AffineProj)
        self.assertEqual(head.main_proj._mode, "mxfp8")


if __name__ == "__main__":
    unittest.main()
