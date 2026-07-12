# Rollback-cost campaign handoff (task #24) — 2026-07-12

## Objective
Cut the per-cycle speculative rollback cost from ~32ms to ~5ms. It is the
decode ceiling for BOTH serving configs. Payoff: loop champion 29→~33 t/s;
vec+tau0 → ~38 t/s projected.

## Current prod state (deployed, healthy)
- DSpark 3-block draft head DEFAULT ON (exo `b90fe53f5`): c=1 **28.8–29.4
  t/s**, byte-lossless, battery + degen probe green.
- Champion config = per-row rowseq loop + confidence pruning
  (`EXO_DSV4_DSPARK_CONF_TAU=0.5`), `VERIFY_ROWSEQ=1`, steel-BI OFF,
  spec-off at c≥2 (`MTP_C2_MAX_CTX=1`), `BS_MIN_ACCEPT=1` (do not touch —
  =0 is the known-corrupt mode, tripwired).
- Verify-vec (`EXO_DSV4_VERIFY_ROWSEQ_VEC=1`, mlx-lm `2dc4f85`) is
  PROVEN (bitwise vs loop in all ldiff regimes; gold gate 3/3) but gated
  OFF: at pruned L≈2.4 its win ≈ the −5% global cost of
  `MLX_STEEL_BATCH_INVARIANT=1` which it REQUIRES. It becomes profitable
  once rollback is cheap (then run tau=0, full-γ verifies).

## The problem, precisely
Cycle profile at γ=5 (EXO_DSV4_MTP_PROFILE=64, 2026-07-12):
`draft ≈11ms, verify 112ms (loop; vec collapses this), accept 1.4ms,
rollback mean 32ms (min 0.28ms — the no-flush path is ~free), total 156ms`.
At γ=5 every verify (6 rows) crosses a ratio-4 pool boundary ⇒ the
flush-attribution rollback runs EVERY cycle. With pruning (avg L≈2.4) it
fires most cycles.

Code: `exo/src/exo/worker/engines/mlx/speculative/dsv4_mtp.py`
`_speculative_next` step 5 (search "Pool-consistency discipline") — paths:
(a) no flush → plain trim (cheap ✓); cache-level
`PoolingCache.spec_rollback` (mlx-lm `models/cache.py`); (b) flush from
rejected token → `restore_meta` + re-accumulate / commit-forward.
Profile WHICH branch dominates first (likely restore/re-accumulate over
41 pools ×2 classes; `EXO_DSV4_SPEC_RB_LOG=1` exists for logging).

## The design candidate (structurally clean)
Verify-time **pool freeze + post-accept replay**:
- The tree-verify path ALREADY freezes pools during verify
  (`_POOL_FREEZE` / `_TREE_VERIFY_CTX` in `deepseek_v4.py` Compressor —
  returns the committed pooled prefix, mutates nothing).
- Linear verify could do the same: freeze pools for the γ+1-row forward,
  then REPLAY only the committed rows (n_acc+1) through the compressor
  (real per-row calls — the increment-2/3 doctrine: tiny ops, zero
  replication risk). Rejection then needs NO pool undo at all.
- Quality note: during the frozen verify, rows attend a pooled set that
  is 1–2 entries staler than sequential — this CHANGES bytes vs the
  current rollback discipline. The gold gate must therefore compare
  spec-on-frozen vs spec-off on the same numerics; expect the identity
  property to hold only if the freeze semantics are also applied... it
  will NOT be bitwise vs today's path. Decide: either accept a re-gated
  trajectory change (battery + needle + probe), or make freeze bitwise by
  replaying pool state BEFORE the bonus-token decode of the next cycle
  (sequential-equivalent visibility is the deferred-bump pattern — a
  frozen verify sees exactly what sequential row 0 sees; rows 1+ would
  have seen newer pools on flush cycles). Study
  `PoolingCache.accumulate_windows` decode mode + deferred bumps first.

## Gates (all exist, all on m4-1)
1. `bench/ldiff_cycles.py 6000 4 48` with prod envs — accept3/reject0/
   reject1 chains vs sequential (run loop AND vec configs).
2. Gold gate: `~/scratch/dspark_identity_gen.py <tag>` on spec-off vs
   spec-on same-numerics legs; `cmp` the outputs (3 prompts × 400 tok).
3. `bench/dsv4_dsml_battery.py --api http://localhost:52415/v1`.
4. `bench/c2_temp1_degen_probe.py --long-tokens 8000 --gen 1600` (c=2
   safety; spec stays off at c≥2).
5. `bash ~/measure_tps.sh` on m4-1 (champion baseline 28.8–29.4).

## Operational notes
- Deploy: commit → push origin → `./start_cluster.sh` from laptop repo
  root (~5 min; nodes git-reset + reinstall mlx-lm).
- mlx-lm worktree for harness runs on m4-1: `~/scratch/mlxlm-dspark`
  (git fetch + checkout, run with `PYTHONPATH=` + `~/scratch/mlxbi_venv`).
- Node-to-node bulk transfer: python http.server on the TB bridge
  (m4-1 = 192.168.200.1) — 5.3GB in ~4s. No node-to-node ssh.
- USER HOLD: no macOS upgrades / Metal 4 work (task #13).
- Full campaign history: memory `exo_dspark_port.md`,
  `exo_dsv4_c2_temp_degen.md`; tasks #19–#24.

## Lessons that keep paying (enforce in the new session)
- One variable at a time; re-run the failing repro after every candidate
  fix; an A/B exoneration is void if the lever was dormant.
- Batch the expensive ops, keep stateful sub-modules on their real
  per-row calls.
- temp=0/short batteries are blind to acceptance-spread and deep-ctx
  bugs — gate with the ragged temp-1.0 probe and long ldiff chains.
