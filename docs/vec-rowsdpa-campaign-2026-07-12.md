# Verify-vec per-row-sdpa campaign — results and handoff (2026-07-12)

Follow-up to vec-serving-increment4-handoff-2026-07-12.md ("NEXT
CAMPAIGN: lossless ~34 via per-row-sdpa vec variant"). Ran both
increments the same day. Headline: **the per-row hypothesis is dead —
the vec drift is NOT in any batched-vs-single kernel the vec path
issues.** Full evidence below; prod default unchanged (loop champion).

## What landed (all committed and pushed)

- **mlx-lm `3db21ec`** — `EXO_DSV4_VERIFY_ROWSEQ_VEC_ROWSDPA=1`
  (level 1): every sdpa / fused-kernel call per row over the SAME
  q_rows/views/masks; batched projections + gathered ring views kept.
  Generalizes the flush-row fallback to all rows in all three vec
  functions.
- **mlx-lm `c211f5b`** — level 2 (`...ROWSDPA=2`): ALSO per-row
  q/kv projections, norms, rope-in, inverse-rope + o-proj tail
  (`_rowsdpa_project_rows` / `_rowsdpa_oproj_rows` — the loop's exact
  L=1 calls, stacked), and per-row `q_residual` into the sparse
  indexer. The only remaining batched piece is the gathered ring view +
  single manual end-state write.
- **exo `da56467f1`, `9b7e17a78`** — pin bumps + env plumbed through
  start_cluster.sh.
- Both levels: ldiff_cycles 192/6/48 gamma=2 **12/12 BITWISE CLEAN**
  (accept3/reject0/reject1 × LDIFF_BATCH_CACHE {all,ring,pool,0});
  level-1 all-caches leg also clean with steel-BI OFF; level-2 clean
  with LDIFF_FENCE=1. (Local runs on a 48GB machine with
  n_routed_experts shrunk 256→32 to fit — the full-size harness model
  jetsams a 48GB box. gamma=5 legs not run: repo bench/ldiff_cycles.py
  lacks the LDIFF_GAMMA generalization, which still lives only in
  m4-1:~/scratch/ldiff_cycles.py.)

## On-cluster results (2-node TP serving, DSv4-Flash, temp 0)

| config | t/s | gold gate vs loop+BI |
|---|---|---|
| loop champion (prod) | 29.4–29.7 | — (baseline) |
| vec batched (inc. 4) | 34.1 | 0/3 |
| vec + ROWSDPA=1 | **35.4–35.5** | 0/3 |
| vec + ROWSDPA=2 | 31.1–31.3 | 0/3 |

All legs `MLX_STEEL_BATCH_INVARIANT=1`, `dspark_identity_gen.py`
(3 prompts × 400 tok), first diffs at chars ~637/564/664 (≈150 tokens)
— the increment-4 signature, unchanged.

## The decisive evidence

1. **ROWSDPA=1 output == ROWSDPA=2 output, byte-identical 3/3** —
   across two separate deployments and a full restructuring of the
   projection graph. Per-row-izing projections/norms/rope/o-proj
   changed NOTHING. The drift is not in the batched projections, not
   in the batched attention (level 1 removed that), and the vec path
   is fully deterministic and restart-stable.
2. **loop+BI is restart-stable 3/3** (loopbi vs loopbi2, separate
   deployments). The gate is measuring a real, deterministic,
   systematic numeric difference between the vec and loop verify
   forwards — not a rare stochastic flip and not restart noise. The
   ~150-token onset is just where the first near-tie logit pair sits.
3. Every local reproduction attempt stays BITWISE CLEAN (batch-cache
   matrix, steel-BI off, serving fence on). Random weights / 6 layers /
   48 tokens / no TP cannot see it — consistent with the FULLBLOCK
   lesson (real-weight value dependence).

## Where the drift can still live (what L1 and L2 share)

- The **gathered ring views + manual two-segment end-state write** vs
  the loop's real `update_and_fetch` buffer (content-proven bitwise on
  the harness, value-suspect at real weights / TP shapes).
- The **shared pre-write `_ring_mask`** (row 0's `make_mask(1)`) for
  all rows vs the loop's per-row `_rowseq_row_mask` (all-true content
  either way, but built at different cache states).
- The **one L-row spec-stash entry** vs the loop's L 1-row entries
  (rollback re-push order; harness reject legs clean).
- **Graph segmentation / TP interaction**: vec builds one graph per
  layer verify vs the loop's L sub-graphs; under the serving fence and
  2-rank TP, kernel fusion boundaries differ. LDIFF_FENCE=1 (singleton
  group) did not reproduce; REAL 2-rank TP has never been harnessed.

## Recommended next steps

1. **On-cluster sub-op hash-diff forensics** — the FULLBLOCK
   methodology (per-row/sub-op hash diff at a diverging position, see
   the 2026-07-10 forensics that pinned the MoE M-dependence). Deploy
   vec+ROWSDPA=2 and loop side by side on the SAME prompt, dump
   per-layer attn-out hashes at the first diverging token, and bisect
   layer class (local/compressed/sparse) + sub-op. Everything upstream
   of the first differing sub-op is exonerated by determinism.
2. If the views/manual-write mechanics are implicated: replace the
   manual write with L real `_update_in_place` calls (keep gathered
   views only for the sdpa reads) — cheap to try, kills two suspects.
3. Longer term (unchanged from increment 4): BI-pin the fused sparse
   kernel and masked batched sdpa in adurham/mlx and re-enable full
   batching — the 35.5 t/s level-1 number shows the prize is real
   (+20% over champion) if the gate can be closed.
4. ROWSDPA=1 (35.5 t/s) supersedes the batched vec (34.1) as the
   **lossy opt-in**: same gate failure, more speed. Launch:
   `EXO_DSV4_VERIFY_ROWSEQ_VEC=1 EXO_DSV4_VERIFY_ROWSEQ_VEC_ROWSDPA=1
   MLX_STEEL_BATCH_INVARIANT=1 ./start_cluster.sh`. ROWSDPA=2 is
   strictly worse (31.2, same gate) — diagnostic value only.

## Operational notes (additions to increment 4's list)

- Gate outputs live on m4-1 in /tmp: rowsdpa_{1,2,3}.txt (level 1),
  rowsdpa2_* (level 2), loopbi_* / loopbi2_* (loop+BI controls). All
  byte-comparable; loopbi == loopbi2, rowsdpa == rowsdpa2.
- `bench/measure_tps.sh` is gitignored (`bench/*.sh`); the working copy
  on m4-1 is `~/measure_tps.sh`.
- Cluster restored to prod default (pure ./start_cluster.sh, loop
  champion, steel-BI 0) at the end of this campaign.

---

## RESOLUTION (same day, later session)

**Root cause found and fixed; vec+ROWSDPA=3 is the new champion.**

Layer-hash forensics (`EXO_DSV4_LAYER_HASH_SUBOPS=0..42`, gate prompt 1,
both legs): every hash identical through pos 128; first divergence at
**pos 129 — the first vec-engaged forward — at B02.attn_out** (first
CompressedAttention layer), with `attn_in` and ALL pre-forward
cache-state hashes (ring, pool, indexer) IDENTICAL. LocalAttention
layers matched everywhere. All three ROWSDPA levels produced
byte-identical output — the divergence was invariant to every cache/
mask/projection mechanic.

Cause: `DeepseekV4ShardingStrategy` replicates attention but sets
`attn.sharding_group` on Compressed/SparseCompressedAttention when
`EXO_DSV4_SEQ_SPLIT=1` (prod default) — so the loop's per-row
`__call__` ends with `elif self.sharding_group is not None:
all_sum(out)`, a distributed reduction **no vec path ever performed**
(increments 1-4 and ROWSDPA 1-3 alike). Deterministic rank-level
numeric difference on every compressed/sparse forward; the single-rank
ldiff harness (sharding_group=None) is structurally blind to it — which
is why four increments of bitwise-proven mechanics never moved the
serving gate.

Fix: `_rowsdpa_sharding_allsum` mirrors the loop tail in all five vec
tails (mlx-lm `095c98c`, exo `4b9932322`).

**Validation:** gold gate 3/3 IDENTICAL vs loop+steel-BI; 100K quality
rung 10/10 needle recall, decode 36.9 t/s, prefill 360 t/s; MTP-off
identity leg (see below). Short-ctx decode 33.7 t/s (vs 29.5 loop
champion, +14%); the all_sum costs ~1.8 t/s vs the pre-fix lossy 35.5.

Champion flip: start_cluster.sh defaults now
`EXO_DSV4_VERIFY_ROWSEQ_VEC=1`, `EXO_DSV4_VERIFY_ROWSEQ_VEC_ROWSDPA=3`,
`MLX_STEEL_BATCH_INVARIANT=1`.

Lesson for the file: single-rank harnesses can never catch rank-level
(distributed-reduction) asymmetries between code paths. Any alternate
serving path must replicate the loop's DISTRIBUTED tail, not just its
per-rank math — grep for `sharding_group` consumers before declaring a
path loop-exact.
