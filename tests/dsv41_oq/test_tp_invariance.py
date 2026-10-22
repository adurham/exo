# Copyright © 2026 Adam Durham (hermes-gw)
"""TP invariance of the oQ build: world=2 must equal world=1.

Both ranks run in ONE process against a replayed ``all_sum``
(:class:`synth.ReplayCollective`): each rank's payload is recorded per call
index and the ``k``-th collective returns ``local + peers' k-th payloads``.
MLX graphs cannot be evaluated from two Python threads (it segfaults), so the
ranks run sequentially and the forward is re-run until the payload store is
complete -- at which point the last pass is numerically a real two-rank sum.

The models are the REAL ``Block``/``AffineProj``/``OqSwitchLinear``/
``OqExperts``/``ShardedHead`` classes that ``oq_build.build_block`` assembles
from a tiny checkpoint in the published pack's layout
(``synth.write_synth_pack``): no download, and every projection is quantized
exactly as the 1017 pack quantizes its own (mixed bits, group 64).

Covered, per the sharding contract:

* out-axis (rows) shards -- ``attn.wq_b``, ``shared_experts.w1/w3``, the expert
  gate/up stacks: each rank's output equals its rows of the world=1 output;
* in-axis (columns) shards -- ``attn.wo_b``, ``shared_experts.w2``, the expert
  down stack: the ranks' partial sums equal the world=1 output;
* the vocab-sharded head: one ``all_sum`` rebuilds the full logit row;
* the whole block, end to end.
"""

from __future__ import annotations

import os
import unittest

import numpy as np
import mlx.core as mx

import synth
from synth import np_of
from mlx_lm.models.deepseek_v41 import moe as dsmoe
from mlx_lm.models.deepseek_v41 import oq_build as oq
from mlx_lm.models.deepseek_v41.cache import ModelCache
from mlx_lm.models.deepseek_v41.config import ModelArgs
from mlx_lm.models.deepseek_v41.hyper_connections import make_identity_pre_mix
from mlx_lm.models.deepseek_v41.model import SharedState

WORLD = 2


def _cos(a: mx.array, b: mx.array) -> float:
    mx.eval(a, b)
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return float(mx.sum(a * b) / mx.sqrt(mx.sum(a * a) * mx.sum(b * b)))


class TPInvarianceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.path.join(os.environ.get("TMPDIR", "/tmp"), "oq_tp_pack")
        synth.write_synth_pack(root)
        cls.ck = oq.OqCheckpoint(root)
        cls.args = ModelArgs.from_dict(cls.ck.config)
        # exact fp32 MoE partials: the bf16 wire format would only add noise
        dsmoe._MOE_ALLSUM_BF16 = False
        synth.install_replay()
        cls.full, _ = oq.build_block(cls.ck, cls.args, 0)
        # the rank blocks are only used for per-projection checks in this class
        cls.rank = [
            oq.build_block(cls.ck, cls.args, 0, rank=r, world=WORLD,
                           group=synth.ReplayCollective(r, WORLD, {}, {}))[0]
            for r in range(WORLD)
        ]

    def _x(self, n: int, d: int, seed: int = 1) -> mx.array:
        mx.random.seed(seed)
        return (mx.random.normal((n, d)) * 0.05).astype(mx.bfloat16)

    def _assert_close(self, got, want, thr=0.999, what=""):
        mx.eval(got, want)
        self.assertEqual(got.shape, want.shape, what)
        self.assertGreater(_cos(got, want), thr, f"{what}: cosine too low")
        self.assertLess(float(mx.max(mx.abs(got - want))),
                        2e-2 * float(mx.max(mx.abs(want))) + 1e-6,
                        f"{what}: max abs diff too large")

    # -- shard geometry --------------------------------------------------
    def test_shard_geometry_chain_is_consistent(self):
        """Every sharded projection holds its rank's half, and the attn chain
        (``wo_a`` groups -> ``wo_b`` inputs) agrees -- the mistake that made an
        earlier revision slice nothing at all."""
        a = self.args
        for r in range(WORLD):
            blk = self.rank[r]
            at = blk.attn
            self.assertEqual(at.n_heads, a.n_heads // WORLD)
            self.assertEqual(at.n_groups, a.o_groups // WORLD)
            # wq_b: rows = this rank's heads * head_dim
            self.assertEqual(at.wq_b.weight.shape[0], at.n_heads * a.head_dim)
            # wkv stays replicated (it is the shared latent)
            self.assertEqual(at.wkv.weight.shape[0], a.head_dim)
            # wo_a: one contiguous run of groups, rows = groups * o_lora
            self.assertEqual(at.wo_a.weight.shape[0], at.n_groups * a.o_lora_rank)
            # wo_b consumes exactly what wo_a produced
            self.assertEqual(at.wo_b.weight.shape[1] * 32 // 8,
                             at.n_groups * a.o_lora_rank)
            # shared expert: intermediate width halved, down-proj input halved
            sh = blk.ffn.shared_experts
            self.assertEqual(sh.w1.weight.shape[0], a.moe_inter_dim // WORLD)
            self.assertEqual(sh.w2.weight.shape[1] * 32 // 8, a.moe_inter_dim // WORLD)
            # routed experts: same intermediate slice
            self.assertEqual(blk.ffn.experts.gate_proj.weight.shape[1],
                             a.moe_inter_dim // WORLD)
            self.assertEqual(blk.ffn.experts.down_proj.weight.shape[2] * 32 // 3,
                             a.moe_inter_dim // WORLD)

    # -- out-axis shards -------------------------------------------------
    def test_wq_b_rows_match(self):
        x = self._x(5, self.args.q_lora_rank)
        full = self.full.attn.wq_b(x)
        per = self.args.n_heads * self.args.head_dim // WORLD
        for r in range(WORLD):
            got = self.rank[r].attn.wq_b(x)
            self._assert_close(got, full[:, r * per:(r + 1) * per], 0.9999,
                               f"rank {r} wq_b rows")

    def test_shared_expert_out_rows_match(self):
        x = self._x(4, self.args.dim, seed=2)
        per = self.args.moe_inter_dim // WORLD
        for name in ("w1", "w3"):
            full = getattr(self.full.ffn.shared_experts, name)(x)
            for r in range(WORLD):
                got = getattr(self.rank[r].ffn.shared_experts, name)(x)
                self._assert_close(got, full[:, r * per:(r + 1) * per], 0.9999,
                                   f"rank {r} shared {name} rows")

    def test_expert_gate_up_rows_match(self):
        t = 3
        x = self._x(t, self.args.dim, seed=3)
        idx = mx.array([[1, 0, 2, 4, 6, 7]], dtype=mx.uint32)
        idx = mx.broadcast_to(idx, (t, idx.shape[1]))
        mx.eval(idx)
        per = self.args.moe_inter_dim // WORLD
        for proj in ("gate_proj", "up_proj"):
            full = getattr(self.full.ffn.experts, proj)(x, idx)
            for r in range(WORLD):
                got = getattr(self.rank[r].ffn.experts, proj)(x, idx)
                # gather_qmm layout is [tokens, topk, 1, out]
                self._assert_close(got, full[..., r * per:(r + 1) * per], 0.999,
                                   f"rank {r} expert {proj} rows")

    # -- in-axis shards --------------------------------------------------
    def test_wo_b_columns_sum(self):
        x = self._x(5, self.args.o_groups * self.args.o_lora_rank, seed=4)
        full = self.full.attn.wo_b(x)
        half = x.shape[-1] // WORLD
        got = sum(self.rank[r].attn.wo_b(x[:, r * half:(r + 1) * half])
                  for r in range(WORLD))
        self._assert_close(got, full, 0.9999, "wo_b columns")

    def test_shared_expert_down_columns_sum(self):
        x = self._x(4, self.args.moe_inter_dim, seed=5)
        full = self.full.ffn.shared_experts.w2(x)
        half = x.shape[-1] // WORLD
        got = sum(self.rank[r].ffn.shared_experts.w2(
            x[:, r * half:(r + 1) * half]) for r in range(WORLD))
        self._assert_close(got, full, 0.9999, "shared_experts.w2 columns")

    def test_expert_down_columns_sum(self):
        t = 2
        x = self._x(t, self.args.moe_inter_dim, seed=6)
        idx = mx.array([[0, 3, 5, 2, 1, 6]], dtype=mx.uint32)
        idx = mx.broadcast_to(idx, (t, idx.shape[1]))
        mx.eval(idx)
        full = self.full.ffn.experts.down_proj(x, idx)
        half = x.shape[-1] // WORLD
        got = sum(self.rank[r].ffn.experts.down_proj(
            x[..., r * half:(r + 1) * half], idx) for r in range(WORLD))
        self._assert_close(got, full, 0.999, "expert down_proj columns")

    # -- vocab-sharded head ---------------------------------------------
    def test_sharded_head_rebuilds_full_logits(self):
        h = self._x(2, self.args.dim, seed=7)
        full_head = oq._build_head(self.ck, self.args, rank=0, world=1, group=None)
        y_full = full_head(h)

        heads = {r: oq._build_head(self.ck, self.args, rank=r, world=WORLD,
                                   group=synth.ReplayCollective(r, WORLD, {}, {}))
                 for r in range(WORLD)}
        for hd in heads.values():
            self.assertIsInstance(hd, oq.ShardedHead)
            self.assertEqual(hd._p.weight.shape[0], self.args.vocab_size // WORLD)
        # ShardedHead writes its band into a zero-padded full-width row and
        # all_sums; summing the two padded bands is exactly that collective.
        voc = self.args.vocab_size
        per = voc // WORLD
        bands = []
        for r in range(WORLD):
            y = heads[r]._p(h).astype(mx.float32)
            lo = r * per
            bands.append(mx.pad(y, [(0, 0)] * (y.ndim - 1)
                                + [(lo, voc - lo - y.shape[-1])]))
        got = bands[0] + bands[1]
        self._assert_close(got, y_full, 0.9999, "vocab-sharded head bands")

    # -- the whole block -------------------------------------------------
    def test_full_block_forward_matches(self):
        b, s = 1, 3
        mx.random.seed(11)
        x = (mx.random.normal((b, s, self.args.hc_mult, self.args.dim))
             * 0.05).astype(mx.bfloat16)
        mx.eval(x)
        pre = make_identity_pre_mix(b, s, self.args.hc_mult)
        out_full, _ = self.full(x, pre, 0, ModelCache(self.args, b, 32),
                                SharedState())
        mx.eval(out_full)

        groups: dict = {}

        def run(r, g):
            groups[r] = g
            blk = oq.build_block(self.ck, self.args, 0, rank=r, world=WORLD,
                                 group=g)[0]
            return blk(x, pre, 0, ModelCache(self.args, b, 32), SharedState())[0]

        outs, passes = synth.run_replayed(run, WORLD)
        for r, out in enumerate(outs):
            self._assert_close(out, out_full, 0.999,
                               f"rank {r} block output (converged in {passes})")


if __name__ == "__main__":
    unittest.main()
