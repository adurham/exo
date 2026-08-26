# DSpark/MTP spec-decode verdict — C_s per-step profile (2026-08-26)

**Step 1 of the corrected spec-decode verdict protocol. Zero new cluster runs —
synthesized from existing [MTP] shadow-gate logs + code reading. This is the
highest-value deliverable of the campaign: it explains why observed spec-decode
tok/s is ~+2% (not the +55-80% the acceptance rate predicts) and points the fix
direction at the verify path, not draft tuning.**

## Headline: C_s = 3.20, not 1.2-1.4

Kimi's framing (consult `/tmp/consult_nondet_kimi.md`): for memory-bound decode,
a γ+1-token batched verify reads the same weights once, so the cost ratio
`C_s = (spec step time)/(serial step time)` should be ~1.2-1.4. At τ≈2.15
accepted tokens/step, predicted speedup = τ/C_s = +55-80%. Observed was only
+1.8% (M1 shadow gate) / -7% to +4% (Stage-2b/3 live A/B). The consult's
hypothesis: the verify path has ~2x overhead.

**Measured C_s from the M1 shadow-gate data (docs/p4v2-m1-shadow-gate-results-and-recovery-2026-08-24.md,
782 cycles, 383 at 100K):**

| quantity | value | source |
|---|---|---|
| baseline sequential tok/s @100K | 29.0 | M1 baseline (sequential) |
| serial step time | 34.48 ms/token | 1000/29.0 |
| draft forward (per cycle) | 11.3 ms | shadow `[MTP]` draft_ms mean |
| verify forward (per cycle, γ_mean=3.31) | 99.0 ms | shadow `[MTP]` verify_ms mean |
| spec cycle time D_cycle (draft+verify) | 110.3 ms | 11.3 + 99.0 |
| tokens committed/cycle (1 bonus + a accepted) | 3.256 | a=2.256 measured |
| **C_s = D_cycle / serial_step** | **3.20** | 110.3 / 34.48 |
| predicted speedup = τ/C_s = 3.256/3.20 | 1.018x (+1.8%) | matches observed |
| break-even a* = D_cycle·baseline/1000 − 1 | 2.199 | 110.3·29/1000−1 |
| measured a | 2.256 | clears break-even by 0.057 (2.6% of a*) → **HOLD** |

The measured C_s = **3.20** is ~2.3× Kimi's predicted 1.2-1.4 for a properly
batched verify. That single number is why +2% is observed instead of +55%.

## Why C_s = 3.20: the verify is ROW-SEQUENTIAL, not batched

The verify forward at `dsv4_mtp.py:2496-2524` builds `verify_input` of shape
`(N, γ+1)` — for γ=3 that is 4 rows (1 anchor + 3 drafts) — and runs ONE call to
`dsv4_speculative_forward` → `model(inputs, cache=cache)` (`dsv4_mtp.py:1420`).
At the call-site level this looks like a single batched forward.

But under `EXO_DSV4_VERIFY_ROWSEQ=1` + `EXO_DSV4_ROWSEQ_FULLBLOCK=1` (both set in
the production `dspark_prod` tmux env, confirmed via `ps eww` on both nodes
2026-08-26), the DeepSeek-V4 model's attention block runs **bitwise-sequential**
— (γ+1) separate per-row attention passes, not one batched matmul. The
measured verify_ms confirms this: it scales **linearly** with γ, 53.5 ms (γ=1)
→ 134.4 ms (γ=5), at **~20.2 ms/row @100K**. A truly batched verify would be
nearly flat in γ (weights read once, KV-cache read amortized). This is the
"HC kernels doing per-token GEMV loops instead of batched verify" that Kimi's
consult flagged as the prime suspect.

The rowseq constraint exists for a correctness reason, not by accident:
`dsv4_mtp.py:4838` documents that `EXO_DSV4_VERIFY_ROWSEQ` "makes the verify
forward bitwise-sequential, removing the drift this tried to patch." Batched
multi-row verify on DSv4's `SparseCompressedAttention` (which includes an
Indexer top-k search over the compressed KV cache whose cost grows with
context) produces ~1ulp logit drift that flips near-tied argmaxes across ranks
→ acceptance diverges → per-uid yield drift → cross-rank wedge. Rowseq is the
shipped correctness fix for that (commits 2026-08-02, `b9921962e`). The cost is
that the verify scales with γ instead of being nearly free — which is the
whole C_s problem.

## Per-cycle residual: ~8 ms of bookkeeping on top of draft+verify

Shadow wall = 117.5 ms/cycle vs 110.3 ms accounted (draft+verify). The ~8 ms
residual is rollback + fence drain/re-arm + cross-rank `broadcast_from_canonical`
for n_accepted+bonus sync (`dsv4_mtp.py:2607-2613`) + DSpark ctx append. This is
real overhead Kimi's "per-step host↔GPU syncs / per-step RDMA broadcast" item
predicted, but it is a minor term (8/110 = 7%) next to the rowseq verify (99 ms).

## What this means for the verdict

The corrected protocol's PROMOTE bar (median fixed-window delta ≥ +10% AND
lower 95% CI ≥ +5%) is an **arithmetic impossibility** at the current C_s.
At C_s=3.20 and a≈2.26, the mechanism is at break-even (a* = 2.199, measured
clears by 2.6% of a*). To clear +10% the acceptance would need to be
≈2.42 (at the same C_s) — a 7% relative acceptance lift, which is large for a
draft head that is already at 0.681 accept-rate. Equivalently, to clear +10%
at the current a=2.256, C_s would need to fall to ~2.78 — i.e. the rowseq
verify would need to be ~13% cheaper, which is inside the cost-model noise,
not a tuning win.

**The verdict is secondary to this finding.** The real story is that DSpark
speculation on this cluster is at the break-even knife-edge because the verify
forward is row-sequential (C_s=3.20), not because the draft head is
miscalibrated. The fix direction is **verify-path batching**, not draft tuning:
make the γ+1-row verify forward a single truly-batched attention pass (or
batch the parts of the per-row loop that don't need to be per-row, per the
`exo-perf-tuning` / `dspark-fullblock-context-scaling-cliff-2026-08-04.md`
next-step: isolate which piece of the per-row attention loop needs to stay
per-row, starting with the Indexer top-k search). That is the path to C_s≈1.3
and the predicted +55-80%.

## Cross-checks

- **Observed tok/s from existing /tmp/ab golden runs (spec-ON, 100K, temp=0):**
  golden_100k 26.98, golden_100k_r1 27.27, golden_100k_r2 24.88 (mean 26.4).
  Spec-OFF baseline (campaign doc) 28.46. Observed delta ≈ -7.3% — *worse* than
  the +1.8% projection. The gap is the natural-end-length confound the consults
  flagged (these runs stopped at 343/295/977 tokens; decode-s amortizes
  prefill+startup differently) plus the base nondeterminism. The fixed-window
  protocol (step 4) will resolve this — but the C_s arithmetic says even a
  clean fixed-window measurement will land near break-even, not +10%.
- **τ from ladder_trace.jsonl (525 cycles, today's spec-ON runs):** mean
  n_accepted=1.89, tokens/cycle=2.89, histogram {0:84, 1:112, 2:105, 3:224}.
  Consistent with the shadow-gate a=2.256 (different run, same order of
  magnitude). The 0-accept cycles (84/525 = 16%) are pure overhead — a full
  draft+verify cycle that commits only the 1 bonus token.
- **Gate A (temp=0 acceptance is strict argmax):** code reading confirms
  acceptance = `mx.equal(target_tokens, draft_concat)` where
  `target_tokens = mx.argmax(verify_logits[:, :gamma, :], axis=-1)`
  (`dsv4_mtp.py:2551-2566`); bonus = `argmax(verify_logits)` at position
  n_accepted (`:2565, :2589`). No probabilistic acceptance at temp=0. Detail
  in the Gate A section below. **Gate A = CLEAN.**
- **Per-row verify cost scales with context too:** the `dspark-fullblock-cliff`
  doc measured r1_verify_fwd=1455.8 ms at ~14K context (vs 99 ms at 100K warmup
  depths) — the Indexer top-k over the compressed KV cache grows with context.
  This is the same mechanism that produced the 15.9x collapse between depth
  500 and 14K in the 2026-08-04 investigation. At 352.6K the C_s will be worse.

## Gate A: temp=0/alpha=1.0 acceptance is strict argmax — CLEAN

The acceptance machinery is exact, by code reading (no cluster run needed):

- **Accepted draft tokens** (`dsv4_mtp.py:2551-2566`): `target_tokens =
  mx.argmax(verify_logits[:, :gamma, :], axis=-1)` is the target model's argmax
  at each of the γ draft positions. `matches = mx.equal(target_tokens,
  draft_concat)`. Acceptance proceeds position-by-position until the first
  mismatch (`:2572-2576`). The accepted token IS `target_tokens[n][k]` — the
  verify forward's own argmax — by construction. Not a sample, not a ratio
  test: strict equality.
- **Bonus/correction token** (`dsv4_mtp.py:2565, 2589`): `all_next =
  mx.argmax(verify_logits, axis=-1)` (or `argmax(logprobs_all)` under
  `EXO_DSV4_MTP_ACCEPT_LOGPROBS=1`, which normalizes via logsumexp first —
  same argmax up to tie-breaking). `bonus_vals[n] = all_next_arr[n][acc]` —
  the argmax at the position immediately after the last accepted draft. Strict
  argmax, not a sample.
- **temp=0 path** (`dsv4_mtp.py:2448, 2555`): `all_greedy = all(t == 0 for t
  in stream_temps)` gates the argmax branch. Any temp>0 stream takes the
  stochastic `:2655+` branch (rejection sampling with per-stream RNG) — but at
  temp=0 the ENTIRE batch uses argmax. **No probabilistic acceptance at temp=0.**
- **Cross-rank canonicalization** (`dsv4_mtp.py:2606-2613`): n_accepted and
  bonus are broadcast from canonical rank (combined 2N-int32 broadcast). This
  ensures rank-consistent committed tokens; it does NOT change the acceptance
  *rule* (still argmax equality), only which rank's argmax wins a near-tie.
- **Tie-reverify** (`dsv4_mtp.py:4851-4895`): RETIRED from prod
  (`EXO_DSV4_MTP_TIE_REVERIFY=0` in production env, confirmed via `ps eww`).
  When on, it re-ran a clean sequential forward on near-tie cycles and took
  ITS argmax — still strict argmax, just from a re-verified forward. Off in
  prod, so not a factor.

**Gate A verdict: CLEAN.** temp=0/alpha=1.0 acceptance is strict argmax equality
(accepted token == argmax of that verify forward's logits) at all three sites
(draft positions, bonus, tie-reverify). No probabilistic acceptance, no
floating-point ratio test. The EOS-ban (`EXO_DSV4_SPEC_EOS_BAN`, default OFF)
is a separate, documented losslessness concern (it forces next-best special
token at EOS-want positions) — OFF in the Stage-3 config we will run, so not a
Gate A factor.

## Sources

- `docs/p4v2-m1-shadow-gate-results-and-recovery-2026-08-24.md` (M1 measured
  numbers: draft 11.3 ms, verify 99.0 ms, a=2.256, D_cycle=110.3 ms, +1.8%)
- `docs/p4-scoping-mtp-for-tp-2026-08-24.md` (cost model C(k)=(k+1)·A+N+k·d,
  break-even a*=2.199, verify-floor-is-rowseq finding)
- `docs/dspark-fullblock-context-scaling-cliff-2026-08-04.md`
  (r1_verify_fwd=1455.8 ms @14K ctx, 15.9x collapse, Indexer top-k hypothesis)
- `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:1420` (verify forward),
  `:2496-2524` (verify_input build), `:2551-2566` (argmax accept), `:4838`
  (rowseq comment), `:4851-4895` (tie-reverify, retired)
- `/tmp/ab/ladder_trace.jsonl` (525 cycles, τ=2.89 from today's spec-ON runs)
- `/tmp/ab/golden_100k*.json` (observed spec-ON tok/s 24.88-27.27)
- Consults: `/tmp/consult_nondet_{glm,dsv4pro,kimi}.md` + `.reasoning`