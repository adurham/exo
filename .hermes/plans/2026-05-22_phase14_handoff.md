# Phase 14 Handoff — 2026-05-22

User target: **35 t/s at c=2 100K MTP-on, TOPK=512 + model defaults preserved.**

## TL;DR — current state

- **Baseline:** c=2 100K MTP-on γ=2 at FENCE_EVERY_N=4 is stable at **34.16 t/s
  mean σ=0.07 across 9/10 iters** (iter 0 = warmup, was 40.90 with asymmetric
  streams — outlier we don't understand yet, see Open Q3). This is the new
  production champion.
- **Gap to target:** ~0.84 t/s short of 35.0. Plan B levers were needed
  to close.
- **Eagle (Plan B.2) is FALSIFIED on this build.** K=8 catastrophically
  killed acceptance (iter 0 18.47, iter 1 with stream-0-dead at 0.62).
  K=1 sanity (should be bit-equivalent to hard-embed) ALSO failed
  (iter 0 21.48, asymmetric streams [11.52 / 9.96]). This means there's
  a **bug in the integration**, not just an OOD problem. The K=1 sanity
  check claude-code did was on shape math only, not pipeline behavior.
- **Cluster is DOWN** (cleanly killed) at end of session.

## What landed in production (shipped)

### Commit `9425b643` — Plan A: shared `common_generation_time_at_start`
- `src/exo/worker/engines/mlx/generator/batch_generate.py:1289+` —
  `_submit_batched_eligible` now captures ONE shared
  `generation_time_at_start` after all per-stream MTP cache prefills
  complete, assigns to every `_EngineTask`. Eliminates per-stream Δ
  in `gen_tps` measurement.
- Verified: per-stream Δ went from ~10 t/s pre-fix to ~0.05 t/s post-fix
  on bistable iters. Real attribution restored.
- 36/36 mlx engine unit tests pass.

### Commit `2e708e19` — `EXO_DSV4_FENCE_EVERY_N_LAYERS=4` new default
- `start_cluster.sh:226` — flipped from `8` (the c=1 optimum) to `4`
  (the c=2 optimum). Halves per-fence chain depth, breaks c=2 bistability.
- Rationale: c=2 has 2× per-collective payload (B=2 vs B=1) → higher
  peer-CQE tail probability per call → c=1 fence cadence becomes
  bistable at c=2. FENCE=4 → ~11 fence segments instead of ~6 → stall
  probability below threshold.
- Verified: 10/10 ≥33 agg ✓ (steady-state iters 1-9: mean=34.16 σ=0.07).

### Commit `3882458d` (mlx, branch `try/ack-qp-isolated`)
- Bootstrap barrier in BOTH `MeshGroup` ctors after `post_ack_recvs(0)`.
  Top-level uses `side_channel_->all_gather<int>(0)`, subgroup uses
  `exchange({Destination{}})`.
- `has_ack` guard in `ack_sync_pre()` (defensive — matches
  `ack_sync_post`'s pattern).
- Env-gate `MLX_JACCL_ACK_SYNC_PRE` for the ce5c64fd pre-barrier
  calls (default OFF).
- **Default OFF behavior is bit-identical to pre-bootstrap state.**
  Always-on bootstrap barrier just prevents a hypothetical future race.
- Cluster relaunch with default-off path works clean.
- `MLX_JACCL_ACK_SYNC_PRE=1` cluster comes up clean but DOES NOT
  resolve c=2 bistability (tested 4-iter, alternating good/bad).

### Commit `33c4a2d7` — Eagle soft-embedding code (CODE SHIPPED, BEHAVIOR BROKEN)
- `mlx-lm/mlx_lm/models/deepseek_v4.py` — `_EAGLE_CTX` side channel at
  line 179, MTP embed override at line 2486-2503 (branch
  `eagle-soft-emb` tip `c369073` on adurham/mlx-lm).
- `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:329-354,1536-1571`
  — `DSv4MTPPredictor.eagle_k` + `set_eagle_soft_emb()`; c=2 batched
  draft loop install/clear via try/finally.
- `src/exo/worker/engines/mlx/speculative/mtp_module.py:591-624,656-700`
  — `_compute_eagle_soft_emb()` helper + c=1 draft loop integration.
- `bench/mtp_eagle_microbench.py` — new single-node microbench (never ran;
  see Open Q1).
- **Default OFF (EXO_DSV4_MTP_EAGLE_K=0) is bit-identical to pre-Eagle**
  per design. The cluster RAN cleanly with K=0 and produced the 34.16
  baseline — so default-off path is verified safe.
- **K=1, K=8 BOTH BROKEN at c=2 100K** (see Open Q2).

### Commit `541061ee` — start_cluster.sh propagates `EXO_DSV4_MTP_EAGLE_K`
- Env wasn't being forwarded to the runner process. Added next to
  `EXO_DSV4_MTP` propagation.

## What's in working tree (NOT committed)

```
(nothing — all session edits were committed before kill)
```

`git status` should be clean except for the standard `M mlx` /
`M mlx-lm` submodule pointer chatter and the `?? .claude/ .hermes/`
untracked.

## Cluster final state

- **Killed cleanly** on both nodes (m4-1, m4-2) at session end (~18:32 CDT).
- Last cluster config running: `EXO_DSV4_MTP_EAGLE_K=1`, FENCE=4, γ=2,
  TOPK=512, mlx@try/ack-qp-isolated tip `3882458d`, mlx-lm@eagle-soft-emb
  tip `c369073`, exo@`541061ee`.
- To resume: just run `./start_cluster.sh` (no env overrides needed
  for default-off Eagle path; defaults match the new champion config).

## Open questions

### Q1: Microbench never ran — why?
- `bench/mtp_eagle_microbench.py` tries `mlx_lm.load(args.model)` which
  resolves the model id through HuggingFace Hub by default. On m4-1 the
  HF cache only has 36GB of the 143GB 6-bit model — XET tried to download
  the rest and hung after a few minutes.
- When I switched to a local path (`/Users/adam.durham/.exo/models/...`),
  the load FAILED because the mlx-community DSv4 variants ship WITHOUT
  MTP weights (45 missing `model.mtp.0.*` params). The cluster runtime
  patches MTP weights in at load time via `EXO_DSV4_MTP=1` + a separate
  patch script, but the microbench's plain `mlx_lm.load` doesn't.
- **Need:** either (a) port the MTP-patch logic into the microbench,
  (b) point at a pre-patched cache directory, or (c) skip microbench
  and use a cluster-side step-1 P(top-1) probe via the running model.

### Q2: K=1 should be bit-equivalent to hard-embed but isn't — bug location?
- Claude-code's local "K=1 collapses bit-exactly" sanity check tested
  shape contract only, not pipeline behavior.
- At c=2 100K with K=1, iter 0 came back asymmetric ([11.52 / 9.96])
  and slow (agg 21.48, vs FENCE=4 baseline iter 0 of 23.29 symmetric).
- Hypotheses to test in the new session:
  1. **`prev_logits` capture forces lazy → eager eval.** The chain
     previously kept logits lazy; now we materialize them at i=0 for
     soft_emb computation. Even at K=1 this changes the mlx scheduling.
     Fix candidate: emit `mx.async_eval` instead of holding refs.
  2. **`_compute_eagle_soft_emb` at K=1 isn't actually mathematically
     equivalent to `embed_tokens(argmax)`.** The softmax + argsort
     + take_along_axis + re-normalize chain might have a numerical
     difference at K=1 even though probs/sum=1.0. Need to verify.
  3. **`set_eagle_soft_emb(None)` in the try/finally clears the
     channel mid-loop, but the SAME loop iteration's `predict()` may
     have already captured a reference into mlx's compute graph.**
     Race between Python-side dict mutation and mlx evaluation.
  4. **DSv4MTPPredictor doesn't have `embed_tokens` exposed correctly.**
     The c=1 + c=2 paths do
     `getattr(self.mtp, "embed_tokens", None)`. If that returns
     something OTHER than the model's actual input embed (e.g. an MTP
     module's own embed), `_compute_eagle_soft_emb` is computing the
     wrong mixture.

### Q3: 10-iter FENCE=4 iter 0 was asymmetric (10.36 / 30.53, agg 40.90)
- **Iter 0 of a fresh-cluster bench has shown per-stream asymmetry in
  every recent run regardless of FENCE setting.** FENCE=4 iter 0 = 40.90
  (asymmetric high), FENCE=2 iter 0 = 27.14 (asymmetric mid), FENCE=8
  iter 0 = 33.93 (symmetric — but on 5-iter only).
- Plan A's shared `common_generation_time_at_start` should force Δ=0.
  But Δ on iter 0 keeps being non-zero.
- Hypothesis: the FIRST decode step may have a different code path
  (e.g. first `_next()` call goes through `_first_step_and_capture_batch`
  which runs a non-spec forward — different completion timing).
  Need to inspect.
- **Doesn't block the 35 t/s target since iter 0 is warmup.** But should
  be understood / fixed for measurement hygiene.

### Q4: Champion of `champion-2026-05-18-mtp-g2-acksync-32.3` (skill pitfall #46)
- Claude-code's analysis suggested this was a false-champion measured
  on a pyproject/uv.lock mismatch state — the named mlx commits
  (05545d38 + ce5c64fd) were on a feature branch never merged to main.
- Current c=1 baseline reproduces at **29.7 t/s** σ=0.09, not 32.29.
- The bootstrap-barrier fix (3882458d) PLUS `MLX_JACCL_ACK_SYNC_PRE=1`
  reproduces cluster startup AND a c=2 5-iter (4 iters: 30.74 warmup,
  34.18 ✓, 22.13 ❌, 22.82 ❌) — protocol fix didn't resolve bistability.
- **Skill pitfall #46 needs to be updated** to reflect that the
  documented champion was likely contaminated. The bootstrap-barrier
  fix from this session IS a real improvement (deployable, neutral by
  default, env-gated for opt-in testing of the protocol fix).

## What's needed to hit 35 t/s

The bistability problem is SOLVED (FENCE=4 → 34.16 stable). Closing
the remaining 0.84 t/s gap requires:

### Path A: Fix Eagle (preferred — minimal cluster churn)
1. **Debug Eagle K=1 regression** (Q2 above). Identify which of the
   four hypotheses is the actual bug. Most likely #1 (lazy/eager eval
   pattern change) or #4 (wrong embed_tokens reference).
2. Once K=1 reproduces baseline (~34.16), test K=2 and K=4.
3. If any K passes (lifts c=2 by ≥+0.84 t/s), ship.
4. If no K helps, abandon Eagle. The Phase 14 plan predicted +1.9% at
   γ=2 (= +0.65 t/s) so even WORKING Eagle alone is borderline for the
   35 target.

### Path B: γ=3 (Plan B.1 — never tested in this session)
- The Phase 14 plan predicted γ=3 alone is net-negative because tokens/cycle
  grows ~4.7% while wall grows ~12%.
- BUT — that prediction assumed the FENCE=8 chain depth where c=2 was
  bistable. At FENCE=4 the chain is shorter, so γ=3's extra hop might
  not blow up the same way.
- Quick to test (env-only change, no code): set `EXO_SPECULATIVE_GAMMA=3`
  and run a 3-iter c=2 100K.
- Risk: at γ=3 the chain depth at c=2 might re-introduce bistability.

### Path C: γ=3 + working Eagle stacked
- Phase 14 plan's combined estimate: +8% over γ=2 baseline = +2.7 t/s
  → ~36.9. Clears 35 with margin.
- Requires Eagle to actually work AND γ=3 to be stable at FENCE=4.

### Path D: Push the verify-side wall down
- Skill ref `references/2026-05-19-quality-and-verify-tail.md` and the
  Phase 14 plan's "B.3 c=1 ceiling lift" describe this — structural,
  weeks of work.
- Last resort.

## Action plan for next session

1. **Verify cluster comes up clean** on current HEAD `541061ee` with
   default env (no overrides). Should produce 34.16 σ=0.07 baseline.
2. **Debug Eagle K=1.** Start by inspecting:
   - `getattr(self.mtp, "embed_tokens", None)` — what is it? Run a
     quick assertion in `mtp_module.py:draft_tokens` that prints
     `id(embed_tokens)` vs `id(model.embed_tokens)` to verify they
     match. If they don't, fix the predictor's embed_tokens reference.
   - Add a Python assert at K=1: `soft_emb == embed_tokens(argmax_id)`
     within tolerance. If it doesn't hold, the helper has a numerical bug.
3. If Eagle proves dead-end, **try γ=3 inline** at FENCE=4 (Path B).
   Should be a ~30 min experiment: just env-override, 3-iter c=2 100K.
4. **Document what works in the skill** — `EXO_DSV4_FENCE_EVERY_N_LAYERS=4`
   default flip is a real production win; skill needs an update.

## Files / commits / state at handoff

```
exo HEAD:    541061ee config(start_cluster.sh): propagate EXO_DSV4_MTP_EAGLE_K
mlx-lm HEAD: c369073  feat(dsv4): Eagle soft-embedding side channel...
             (branch: eagle-soft-emb)
mlx HEAD:    3882458d fix(jaccl): bootstrap barrier in MeshGroup ctors...
             (branch: try/ack-qp-isolated)

uv.lock pins:
  mlx    -> github.com/adurham/mlx@try/ack-qp-isolated  (3882458d)
  mlx-lm -> github.com/adurham/mlx-lm@eagle-soft-emb    (c369073)

start_cluster.sh defaults: FENCE=4, TOPK=512, γ=2, BATCHED_PREFILL_RENDEZVOUS=2000ms
                            EAGLE_K not set → defaults to 0 (off)

Cluster: DOWN (cleanly killed at 18:32 CDT both nodes).
```

## Bench/probe scripts left behind

All on m4-1 `/tmp/`:
- `bench_c2_temp0_2iter.py` — quick K-sweep probe (2 iters)
- `bench_c2_temp0_5iter.py` — 5-iter validation
- `bench_c2_temp0_10iter_phase14a.py` — 10-iter formal validation
- `probe_c1_baseline.py` — c=1 baseline (3-iter)
- `probe_c1_oneshot.py` — c=1 single-iter
- `c2_validate10.log` — full 10-iter FENCE=4 result (34.16 σ=0.07 on iters 1-9)
- `c2_eagle_k8.log` — K=8 cluster bench result (broken: 18.47 / 10.83 asymmetric)
- `c2_eagle_k1.log` — K=1 cluster bench iter 0 only (broken: 21.48 asymmetric)
- `eagle_implementation_report.md` — claude-code's Eagle implementation
  report

All on laptop `/tmp/`: same set + `eagle_implementation_brief.md`,
`ce5c64fd_wedge_brief.md`, `ce5c64fd_wedge_report.md`,
`c2_bistability_brief.md`, `c2_bistability_report.md`.

## Memory updates needed

- Skill `exo-cluster-operations` pitfall #46: needs update per Q4
  (champion claim was likely contaminated; mlx fix branch fixes are
  real but bootstrap-barrier is the real shipped improvement).
- Skill recipe `references/2026-05-21-c2-100k-mtp-recipe.md` mentions
  FENCE_EVERY_N at the implicit default; should be updated to call
  out the new 4 default and the c=2 rationale.

## Quality preservation (user requirement check)

- `EXO_DSV4_INDEX_TOPK=512` — kept ✓
- `EXO_DSV4_MTP=1` (γ=2) — kept ✓
- `EXO_KV_CACHE_BITS=0` (bf16) — kept ✓
- Model: `mlx-community/DeepSeek-V4-Flash-8bit` — kept ✓
- No model-default overrides. The FENCE change is a cluster-coordination
  knob (not a model knob), so quality is unaffected.
