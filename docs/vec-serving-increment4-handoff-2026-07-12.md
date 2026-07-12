# Verify-vec serving campaign — handoff after increment 4 (2026-07-12)

## State (all committed and pushed)
- **Prod default: loop champion, byte-lossless, 29.4–29.7 t/s** (pure
  `./start_cluster.sh` defaults; full ladder green earlier today —
  rollback-cost campaign, see
  rollback-cost-campaign-handoff-2026-07-12.md).
- **Increment 4 LANDED** (mlx-lm `4ebd37f`, exo pin `d96b11b44`+): vec
  engages on serving's Batch* cache classes, feeds the spec stash, and
  carries loop-parity masks. Harness matrix **15/15 BITWISE**
  (LDIFF_GAMMA ∈ {2,5} × LDIFF_BATCH_CACHE ∈ {all, ring, pool, 0},
  accept3/reject0/reject1, cache-level rollback refusal-free).
- **Measured** (both opt-in via env, gated off by default):
  - vec + tau0.5: **34.1 t/s** (+16%), cycle 57.9ms, rollback 0.9ms,
    zero commit-forwards.
  - vec + tau0 (full-γ): 26.3 t/s — acceptance 2.39 tok/cycle doesn't
    pay for the 77ms verify. The old ~38 projection is dead; full-γ is
    not the win, pruned+vec is.
  - Opt-in launch: `EXO_DSV4_VERIFY_ROWSEQ_VEC=1
    MLX_STEEL_BATCH_INVARIANT=1 ./start_cluster.sh` (LOSSY, see below).
- **Serving gold gate FAILS 0/3 for vec** (dspark_identity_gen.py,
  same-numerics legs, both MLX_STEEL_BATCH_INVARIANT=1): coherent
  near-tie flips starting ~150 tokens in, all 3 prompts.
  **Control: loop + steel-BI is 3/3 IDENTICAL** — steel-BI and the
  serving stack are exonerated; the drift is vec's batched compute.
- Diagnosis: the real-weight **value-dependent batched-vs-single
  kernel class** (~1 flip per ~6k layer-forwards — matches the
  ~150-token onset at 43 layers). The 4-layer random-weight ldiff
  harness is statistically blind to it (the FULLBLOCK campaign learned
  the same lesson). Probe-proven instance of the class: mx.fast sdpa
  with an array mask containing REAL False entries is NOT
  batch-invariant even under steel-BI (~1e-3 batched-vs-single);
  all-true masks tested bitwise at probe values but are value-suspect.
  Prime suspects for the residual: the fused sparse gather-SDPA kernel
  (custom Metal, EXO_DSV4_SPARSE_FUSED_SDPA=1 default, NO BI pins) and
  mx.fast sdpa batched-with-mask at real values.

## NEXT CAMPAIGN: lossless ~34 via per-row-sdpa vec variant
Keep the batched projections + gathered ring views (the cheap 80% of
the vec win), but issue every sdpa/fused-kernel call PER ROW — kernel
class AND batch size identical to the loop's L=1 calls, bitwise by
construction, no dependence on kernel BI at all.
1. Add `EXO_DSV4_VERIFY_ROWSEQ_VEC_ROWSDPA=1` (or fold into the vec
   gate): in `_RowseqVecMixin.rowseq_vec`, `_compressed_rowseq_vec`,
   `_sparse_rowseq_vec`, replace the batched attention call with a
   per-row loop over the SAME q_rows/views/masks (the flush-row
   fallback already shows the exact shape — generalize it to all rows).
2. Measure: expect between 29.4 (loop) and 34.1 (full vec). If ≥ ~32,
   run the ladder; champion-flip only on gold gate 3/3 + battery +
   probe (steel-BI must be ON in both identity legs).
3. If the number disappoints, profile which batched op carried the win
   (projections vs sdpa) via EXO_DSV4_MTP_PROFILE + a variant that
   batches only projections.
4. Longer term (real 34+ lossless): BI-pin the fused sparse kernel and
   the masked sdpa batched path in mlx (adurham/mlx), then re-enable
   full batching. The probe in
   ~/scratch/probe_sdpa_mask_bi.py (m4-1) reproduces the masked-sdpa
   gap standalone.

## Landed mechanics (for code archaeology)
- `_rowseq_vec_ring_ok/_apply`: BatchRotatingKVCache at B==1 steady
  state; keep=0 wrap; mirrors `_update_in_place` bookkeeping (offset
  array += L, `_offset` += L, `left_padding -= L`, rotated, mx.depends).
- Spec-stash feed in the apply (both ring classes): one L-row entry,
  `(kv, zero-width values)`; without it every vec rejection refused
  cache-level rollback → 72ms commit-forward.
- **Offset in-place-mutation root cause** (the batch-ring drift): the
  ring apply's `offset += L` mutates the mx array object; graph nodes
  built AFTER the apply (the inverse rope) capture the post-push value
  → output roped at offset+L. Plain rings immune (int offset).
  Defensive copy at capture — same idiom as LocalAttention.__call__
  (~line 3440), which exists for exactly this reason.
- Loop-parity masks: batch rings emit an ARRAY decode mask at N=1
  (all-true at vec's precondition but the kernel class must match);
  compressed extends over pooled columns via `_extend_mask` (the batch
  pool's donation path relies on the MASK to hide the deferred slot —
  the plain class slices instead); sparse gathers per-row pmasks,
  None rows contribute all-true.
- Flush-row per-row fallback: rows with non-None pmask (~1 per
  compress_ratio) run the loop's exact L=1 call.

## Operational notes (hard-won, do not skip)
- The ldiff harness model is ~30GB bf16. NEVER run it on m4-1 with the
  cluster loaded: jetsam SIGKILLs the 72GB runner, the m4-2 peer rank
  survives holding ~118GB wired, placement wedges. Teardown (no reboot,
  proven 3×): on both nodes `screen -S exorun -X quit; pkill -f
  "python -m exo"; sleep 3; pkill -9 -f multiprocessing.spawn`, check
  `vm_stat | grep wired` drops to ~190K pages, then start_cluster.
- ldiff harness reads LDIFF_* flags (SPEC_STATE/CACHE_ROLLBACK/
  RESTORE_AFTER_TRIM=1 for the prod rollback twin), NOT the EXO_DSV4
  rollback envs; defaults drive the legacy known-dirty path. LDIFF_GAMMA
  generalization + reject chunk fixes live ONLY in
  m4-1:~/scratch/ldiff_cycles.py.
- The m4-1 worktree ~/scratch/mlxlm-dspark has deepseek_v4.py/cache.py
  scp'd over 2dc4f85; reconcile with a real fetch/checkout of 4ebd37f
  before new edits.
- MTP-PROF dumps: m4-1:~/.exo/exo_log/runner_log/stderr.log is
  APPEND-ONLY ACROSS RESTARTS — snapshot `stat -f %z` before a run and
  read `tail -c +OFFSET`, or stale dumps masquerade as fresh.
- measure_tps.sh reuses one prompt — a warm prefix-cache can return
  600 tok in 0.0s; use a fresh prompt for post-restart confirmations.
- start_cluster can land nodes on DIFFERENT commits if a push races the
  deploy (saw e41b437f vs b7876405); it stops at the consistency check —
  rerun after confirming the push propagated.
- Deploy envs all plumbed: EXO_DSV4_VERIFY_ROWSEQ_VEC,
  EXO_DSV4_DSPARK_CONF_TAU, MLX_STEEL_BATCH_INVARIANT,
  EXO_DSV4_MTP_PROFILE, EXO_DSV4_RB_PROFILE.
