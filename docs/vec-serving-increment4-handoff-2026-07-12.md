# Verify-vec serving enablement (increment 4) — handoff 2026-07-12

## Objective
Make EXO_DSV4_VERIFY_ROWSEQ_VEC actually engage in serving, so the
vec+tau0 full-γ config can be measured. That is the remaining path to the
~38 t/s projection. Current champion: loop + tau=0.5 at **29.4 t/s**
(rollback-cost campaign closed same day — see
rollback-cost-campaign-handoff-2026-07-12.md closure note).

## The two blockers (both PROVEN today)

### 1. Vec never engages in serving
Serving converts caches at insert (mlx-lm generate.py `_make_cache`,
~line 922): RotatingKVCache → **BatchRotatingKVCache**, PoolingCache →
**BatchPoolingCache**. But every vec gate requires the PLAIN classes:
- `_rowseq_vec_ring_ok` (deepseek_v4.py ~4591): `isinstance(local_cache,
  RotatingKVCache)` — Batch ring is NOT a subclass → all three attention
  classes' vec paths no-op in serving.
- `_compressed_rowseq_vec_supported` (~4699): additionally requires
  `isinstance(pool_cache, PoolingCache)`.

Consequence: the task-#23 "serving gold gate 3/3" trivially passed (it
gated the loop), and the 16.5→27.6 t/s tau=0 measurement in the
exo_dspark_port memory CANNOT have been the current serving stack —
re-verify its provenance before trusting any vec serving numbers.

### 2. Vec ring write bypasses the spec stash
`_rowseq_vec_ring_apply` writes `cache.keys` manually and never feeds
`_spec_pushed`. Under cache-level rollback (prod default,
EXO_DSV4_SPEC_CACHE_ROLLBACK=1) a rejection then sees
`spec_pushed_rows()==0 != verify_len` → refusal → **commit-forward
fallback returns on every vec-engaged rejection** (the exact ~72ms cost
the task-#24 campaign just eliminated). ldiff-proven: γ=5 vec plain-ring
reject0 refuses with `rings=False pools=True` at cycle 2 (cycle 1 ran the
loop — the vec/loop alternation is offset-dependent).

## Increment-4 work plan

### 4a. Ring core (shared by all three attention classes)
- `_rowseq_vec_ring_ok`: also accept BatchRotatingKVCache with
  `keys is not None and keys.shape[0]==1 and keys.shape[2]==max_size and
  _offset >= max_size and _lengths is None`.
- `_rowseq_vec_slot_map`: unchanged; batch ring wraps to 0 (keep=0).
- `_rowseq_vec_ring_apply`: branch on class for the end-state write.
  Batch bookkeeping per L rows (mirror `_update_in_place`):
  `offset` (mx per-stream array) += L; `_offset` (python int) += L;
  `left_padding -= L` (steady state is rotated); `rotated = True`;
  `_idx = slots[-1]+1`; final
  `keys = mx.depends(keys, (left_padding, offset))`.
- **Stash feed in BOTH classes** (fixes blocker 2): when
  `cache._spec_stash_armed`, append `(kv, zero_width_values)` — values
  are (B,1,L,0) `_zero_values`-style; `rollback_spec_write` iterates rows
  within one entry, so a single L-row entry is fine. NOTE the POOL mixed
  gate requires single-row POOL stash entries — pools already push
  per-row via the real compressor calls; do not conflate the two.
- `save_spec_state`/`restore_spec_state` on the batch ring already cover
  everything the vec write mutates (keys copy, offset/left_padding
  copies, _offset/_idx/rotated) — verified.

### 4b. Mask parity (the bitwise-risk area)
Loop path on batch classes passes per-row ARRAY masks
(`_rowseq_row_mask` → `BatchRotatingKVCache.make_mask(N=1)`: window +
left_padding validity, `mx.roll`ed by _idx+1 — at steady state with
left_padding≤0 the content is ALL-TRUE, but the KERNEL class differs from
mask=None). Vec currently passes mask=None (correct only for plain rings,
whose N=1 make_mask returns None at full window).
- LocalAttention vec on batch rings: build per-row masks by READ-ONLY
  emulation of make_mask with _idx advanced per row (make_mask mutates
  nothing), stack to (L,1,1,W), pass to the batched sdpa. Whether
  batched-array-mask sdpa == per-row-array-mask sdpa bitwise under
  MLX_STEEL_BATCH_INVARIANT is UNKNOWN — the ldiff harness decides; if it
  drifts, that is a steel-BI pin gap to fix in mlx (sdpa mask
  specialization), not a reason to fudge the mask.
- CompressedAttention: the decode path extends the ring mask over pooled
  columns (`_extend_mask(mask, _dispatch_pmask(pool_cache, L, offset),
  width)` ~line 3610). Vec must replicate the extended array per row
  within its width groups. Also relax its gate to BatchPoolingCache
  (B==1) — the per-row `self.compressor(...)` calls are class-generic.
- SparseCompressedAttention: per-row pmask gathers already exist in the
  vec path; the ring-mask component needs the same treatment as
  LocalAttention. Check `BatchPoolingCache.make_mask` at N=1/B=1 (may be
  an array where the plain class returns None).

### 4c. Gate matrix (all on m4-1, cluster DOWN — see operational notes)
1. `~/scratch/ldiff_cycles.py 6000 4 48` with LDIFF_SPEC_STATE=1
   LDIFF_CACHE_ROLLBACK=1 LDIFF_RESTORE_AFTER_TRIM=1
   EXO_DSV4_VERIFY_ROWSEQ=1 EXO_DSV4_VERIFY_ROWSEQ_MIN_CTX=0
   EXO_DSV4_ROWSEQ_ROWMASK=1 EXO_DSV4_VERIFY_ROWSEQ_VEC=1
   MLX_STEEL_BATCH_INVARIANT=1 MLX_GEMV_BATCH_INVARIANT=1, sweeping
   LDIFF_GAMMA ∈ {2,5} × LDIFF_BATCH_CACHE ∈ {ring, pool, all, 0}.
   `ring` isolates 4a; `all` is the serving-classes gate. Every chain
   must be BITWISE CLEAN and refusal-free (the harness RAISES on
   cache-level refusal — that is the stash-compat gate).
2. Deploy (envs: EXO_DSV4_VERIFY_ROWSEQ_VEC=1 MLX_STEEL_BATCH_INVARIANT=1
   EXO_DSV4_DSPARK_CONF_TAU=0 + EXO_DSV4_MTP_PROFILE=64
   EXO_DSV4_RB_PROFILE=1 — all plumbed in start_cluster.sh as of exo
   HEAD): expect MTP-PROF verify to collapse from ~60ms (pruned loop) /
   ~112ms (full-γ loop), rb_commitfwd ABSENT, rollback <1ms.
3. Gold gate (dspark_identity_gen.py specON/specOFF legs — same-numerics:
   BOTH legs need MLX_STEEL_BATCH_INVARIANT=1), DSML battery, degen
   probe, measure_tps. Champion decision: vec+tau0 must beat 29.4 by
   enough to justify steel-BI's −5% global tax (it applies to EVERYTHING,
   including prefill).

## Operational notes (hard-won today, do not skip)
- The ldiff harness model is ~30GB bf16 (unquantized 4-layer DSv4 at full
  vocab). NEVER run it on m4-1 with the cluster loaded: jetsam SIGKILLs
  the 72GB runner, the m4-2 peer rank survives as a zombie holding ~118GB
  wired, and re-placement wedges. Teardown recipe (no reboot needed,
  twice proven): on both nodes `screen -S exorun -X quit; pkill -f
  "python -m exo"; sleep 3; pkill -9 -f multiprocessing.spawn`, verify
  `vm_stat | grep wired` drops to ~190K pages, then ./start_cluster.sh.
- ldiff harness env gotchas: it reads LDIFF_* flags, NOT the EXO_DSV4
  rollback envs; defaults drive the legacy known-dirty path. Its reject
  chunk builders were γ=2-hardcoded — fixed in the m4-1 scratch copy
  (LDIFF_GAMMA now generalizes; γ=2 token values unchanged). The fixed
  copy lives ONLY at m4-1:~/scratch/ldiff_cycles.py.
- m4-1 worktree ~/scratch/mlxlm-dspark currently has the mixed-rollback
  cache.py copied over 2dc4f85 (matches mlx-lm f00a9a9). `git stash list`
  is empty; reconcile with a real fetch/checkout of f00a9a9 before new
  edits.
- Deploy flow: commit → push origin → ./start_cluster.sh (~5 min).
  MTP-PROF dumps land in m4-1:~/.exo/exo_log/runner_log/stderr.log —
  APPEND-ONLY ACROSS RESTARTS: snapshot `stat -f %z` before a run and
  read `tail -c +OFFSET`, or stale dumps from prior sessions will
  masquerade as fresh (cost this campaign an hour).

## Current state (all pushed)
- exo main d96b11b44 + env plumbing commit (VERIFY_ROWSEQ_VEC / CONF_TAU
  forwards); mlx-lm main f00a9a9 (mixed-flush cache-level rollback).
- Prod: loop champion, pure defaults, 29.4 t/s, byte-lossless, full
  ladder green. rb_commitfwd = 0; rollback 0.79ms mean.
- Rollback is NO LONGER the vec+tau0 blocker; increments 4a/4b are.
