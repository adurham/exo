# DSpark/MTP Speculative Decoding — Master Campaign History (2026-05 → 2026-08-28)

**Status:** CLOSED / PROMOTED TO PRODUCTION (2026-08-27, validated 2026-08-28)  
**Cluster:** 2× Mac Studio M4 Max, TB5/RDMA, DeepSeek-V4-Flash-0731 8-bit, TP=2 since
2026-08-16 (PP before)  
**Repo SHAs:** exo `75d2402dd` / mlx-lm `d098642`

This document traces the three-month arc from the PP-era MTP champion (30.8 tok/s c=1 @100K,
2026-06-04) through a sequence of correctness crises, the cluster migration from Pipeline
Parallel to Tensor Parallel (2026-08-16), the Aug-24 shadow-gate HOLD, and the Aug-26 REVERT
of the as-is rowseq config, to the Aug-27 depth-gated batched-verify PROMOTION (+36.71% @100K,
12/12 pairs) and the Aug-27/28 352K memory fix (+17.57% @352.6K, zero collapses), culminating
in Aug-28 correctness verification establishing that the system is byte-deterministic per
config — and that the historical "99.3% base nondeterminism" was a fresh-nonce harness artifact
in `build_prompt()` (dspark-352k-correctness-harness-verification-2026-08-28.md).

---

## 1. Current production state (authoritative, 2026-08-28)

### 1.1 Live environment flags

| Flag | Value | Notes |
|------|-------|-------|
| `EXO_SPECULATIVE` | 1 | |
| `EXO_DSV4_MTP` | 1 | |
| `EXO_DSV4_MTP_DEDICATED` | 0 | |
| `EXO_DSV4_DSPARK` | 1 | |
| `EXO_DSV4_DSPARK_NATIVE` | 1 | |
| `EXO_SPECULATIVE_GAMMA` | 3 | |
| `EXO_DSV4_VERIFY_BATCH` | 1 | ≥8K context path |
| `EXO_DSV4_VERIFY_BATCH_MIN_CTX` | 8192 | |
| `EXO_DSV4_VERIFY_ROWSEQ` | 1 | <8K path |
| `EXO_DSV4_DSPARK_TP_SHARD` | 1 | |
| `EXO_MLX_CLEAR_CACHE_INTERVAL` | 64 | |
| `EXO_DSV4_HC_COLLAPSE_KERNEL` | 1 | |
| `EXO_DSV4_HC_EXPAND_KERNEL` | 1 | |
| `EXO_DSV4_SPEC_EOS_BAN` | 0 | |
| `EXO_JIT_PLACEMENT_WAIT_SECONDS` | 120 | requires exo ≥ `75d2402dd` |

**Shipped-but-OFF:** `EXO_DSV4_DRAFT_EPILOGUE=0` (A/B pending); `EXO_DSV4_BOOKKEEP_FAST` off;
`EXO_DSV4_EXACT_TOPK_PARAM_CAP=64` (default unchanged, A/B pending).

### 1.2 Headline performance numbers

| Context | Config | tok/s | vs baseline | CI / notes |
|---------|--------|------:|-------------|------------|
| @100K | batched ON vs spec-OFF 27.15 | 36.63 | **+36.71%** | CI +28.26..+51.02%, 12/12 pairs |
| @352.6K | verbon3 ON vs stripped-OFF 24.19 | 28.44 | **+17.57%** | CI +8.7..+27.8%, 0/8 collapses |
| @352.6K | batched vs rowseq 22.93 | 28.95 | **+26.3%** | — |
| Soak @352.6K | batched, 4000 tok | 30.46 | mean_accept 2.438/3 (~81%) | 4000/4000 finish=length |

**Memory (post-fix):** swap peak 50–97 MB (was 1.37–1.76 GB); wired 92–93.5 GB (was 95.3–96.8 GB).  
**C_s:** 3.20 (rowseq) → 2.14 (batched) (dspark-cs-profile-2026-08-26.md).  
**Determinism:** byte-deterministic per config at depth incl. across relaunches; batched/rowseq/
spec-off are three deterministic different greedy trajectories — bounded 0.023%/row MoE residual,
first div char ~24 (batched vs rowseq) / ~54 (rowseq vs spec-off); output quality verified
34/34 factual claims (dspark-352k-correctness-harness-verification-2026-08-28.md).

---

## 2. Chronological campaign narrative

### 2.1 PP-era MTP/Eagle foundation (May–Jun 2026)

The campaign opens with the Eagle K=1 regression: commit `21ba40db` introduced a synchronously-
blocking 16 KB bf16 all_gather on the c=2 draft-chain critical path — a JACCL collective between
two `predict()` calls — causing a **17× slowdown** (a ~150 s c=2 100K iter became ~6.5 minutes).
The fix short-circuited the broadcast at K=1 by reusing the already-broadcast `tok_arr`; at K=1
the broadcast is algebraically unnecessary (eagle_k1_fix_report.md).

The correctness milestone was commit `c840bc2d` (2026-06-04), fixing a ~1 ulp batched-verify
logit difference that at temperature 0 flipped tied tokens and cascaded into spurious `</think>`
emissions. Fix: deterministic tie-break on the bonus token only (lowest token id among tokens
within eps=0.5 logits of the max). Correctness: 5/5 PASS; 100% absolute correctness across the
8-task AI-Stupid-Meter suite (mtp-tiebreak-losslessness-fix.md).

| Metric | Value |
|--------|-------|
| Champion tok/s (c=1, 100K, γ=2) | 30.77 mean / 30.8 median / σ=0.067 / 0/10 errors |
| mean_accept | 1.04/2 drafts (hist: 0:34%, 1:28%, 2:38%) |
| verify fraction of cycle | 93.4% of 87.5 ms total |
| γ sweet spot (PP-era) | 2 (γ=1: −6%; γ=3: −18%) |
| Forbidden levers | `EXO_KV_CACHE_BITS != 0`; `EXO_DSV4_INDEX_TOPK < 512` |

The champion config was tagged `validated-upstream-integration-20260604-153609` at commit
`652d8224` (deepseek-v4-mtp-performance.md).

Three c=2 MTP verify bugs were then fixed in mlx-lm (`cb7e3bd` wide ring, `aaac5c3`/`60a0a0c`
pooled mask, `491f6fe`/`5b00004` sparse-kernel accuracy — L>1 sparse verify routed per-position
through the accurate fused L=1 SDPA path), yielding ~20 t/s per stream at c=2 100K. A residual
**80% adversarial final-digit flip** (Bug-3: ~0.3 noise on pathological final-digit-before-EOS
needle) was documented but not fixed (deepseek-v4-c2-mtp-verify-fixes.md). Two additional B=2
bugs were fixed in June: seq-split all_gather reconstruction scrambled streams at B>1 (`8a9cdee`),
and MTP bootstrap offset read the ring cursor (~131) instead of logical position (~138696)
(`48a4a3c`). Gate removal commit `47fdf32a`. Post-fix prefill: **367 t/s aggregate @100K B=2**
(b2-mtp-resolution-2026-06-24.md).

**All PP-era absolute numbers are stale under TP.**

### 2.2 Correctness crisis + FULLBLOCK cliff (Jul–early Aug 2026)

On 2026-07-26, commit `2d696ff60` disabled speculation cluster-wide after a confirmed self-doubt
reasoning spiral on `math_digit_sum`: the model burned its entire 8192-token context budget re-
verifying its own output without converging. Root cause: **generic L>1 batched-verify numerics
drift across ALL THREE speculation mechanisms** — not DSpark-specific. Same divergence reproduced
under classic draft-model speculation and plain chained MTP (p4-scoping-mtp-for-tp-2026-08-24.md
§B).

The 2026-08-02 fix introduced `EXO_DSV4_VERIFY_ROWSEQ` + `EXO_DSV4_ROWSEQ_FULLBLOCK` (run each
verify row sequentially at L=1). `EXO_DSV4_MOE_PARTS_ROWSEQ=shared` (commit `b9921962e`) carried
a residual **0.023%/row** divergence — not bitwise-zero. Validated: 2× identical temp=0 reruns
of `math_digit_sum` → clean convergence.

The native-head plan (commit `99f5cda51`, `EXO_DSV4_DSPARK_NATIVE` gate, 2026-08-03/04) loaded
the -0731 checkpoint's own bundled head: **4705 `mtp.*` keys** across `mtp.0/1/2` on-disk. The
"81 params" figure is a post-`sanitize()` module count — never hardcode this; assert the actual
on-disk key count (dsv4-0731-dspark-native-head-plan-2026-08-03.md).

| Claim | Verdict |
|-------|---------|
| 15.9x context-scaling collapse @14K tok | RETRACTED same-day; fresh relaunch: 17.31 t/s @15K |
| "Rare / 1-in-8" stall characterization | CORRECTED: 2/11 = **~18%** GPU-idle stall @~2800 tok |
| Stall rate at 14K: 1461.5 ms/cycle verify | SURVIVED; acceptance during stall: 94% (not draft quality) |
| `DRAFT_AHEAD` as root cause | FALSIFIED: 2/10 collapses with it off (indistinguishable from 2/11 on) |
| Collapse mechanism = DSpark draft head | RETRACTED (Aug-24): FULLBLOCK verify MODE k-multiplier |

The morning MoE fix (commit `b9921962e`) was correct and unaffected by the cliff
(dspark-fullblock-context-scaling-cliff-2026-08-04.md).

### 2.3 TP-port scoping (Aug 21–24 2026)

Flag sweep (2026-08-21) confirmed SDPA row-split already active (3.7–7.6× at B>1 L≤8), all
decode-side flags (`EXO_DSV4_DECODE_NODE_DIET`, `EXO_DSV4_EXACT_TOPK`, `EXO_DSV4_SINGLE_GATHER`)
already ON in production; `EXO_DSV4_DSPARK_NATIVE` was identified as the one unexplored live
finding — production ran `DSPARK=1` but `DSPARK_NATIVE` unset, using the stale preview-vintage
local head (flag-sweep-completion-and-dspark-native-finding-2026-08-21.md).

T6 gate (2026-08-22): kernel-fix-only ceiling **1.29× < 1.5× gate threshold**; PP+DSpark
comparison baseline **5 weeks stale** (TP had improved from ~15–20 → 29.2–31.1 tok/s via async-
fence fix, collapsing the apparent gap). Port not recommended
(mtp-dspark-tp-port-decision-gate-2026-08-22.md).

P4-scoping (2026-08-24) issued **9 corrections** to prior claims and confirmed TP DSpark wiring
is live end-to-end; correct env: all three `EXO_SPECULATIVE=1` + `EXO_DSV4_MTP=1` +
`EXO_DSV4_DSPARK=1`. The `EXO_DSV4_MTP=1` hard-prerequisite was undocumented and the most
likely silent-failure mode (p4-scoping-mtp-for-tp-2026-08-24.md).

| Quantity | Value |
|----------|-------|
| Shadow M1 acceptance @100K (782 cycles, 383 @100K) | a=2.256 vs a*=2.199 |
| Verify cost @100K (γ_mean=3.31) | 99 ms (20.2 ms/row, +22% above cost model) |
| Projected throughput gain | +1.8% — HOLD band (a*≤a<a*+0.30) |
| Gate band margin | a=2.256 clears a*=2.199 by only 0.057 |

PM chose option 2: revert to production defaults, bank M1 data, M2+ shelved behind HOLD
(p4v2-m1-shadow-gate-results-and-recovery-2026-08-24.md).

### 2.4 Corrected A/B → REVERT (Aug 25–26 2026)

The corrected fixed-window protocol (256-tok window, `/v1/chat/completions`, temp=0, paired
time-adjacent) ran 12 ON + 12 OFF probes. Two bugs unmasked during staging: gamma env silently
ignored bug (`433dce6c1`); EOS-ban unmasking (`ebe272fd7`).

The **C_s profile** was the decisive analytical finding: C_s=3.20 because verify is row-sequential
(~20.2 ms/row @100K); at C_s=3.20 / a≈2.26, PROMOTE at +10% bar is an arithmetic impossibility.
Fix direction identified: verify-path batching (reduce C_s to ~2.0). The M1 shadow-gate HOLD
(+1.8%) was confirmed with real data (dspark-cs-profile-2026-08-26.md).

| Decision metric | Value | Bar | Pass? |
|-----------------|------:|-----|-------|
| Median % delta (ON−OFF) | +1.87% | ≥+10% | FAIL |
| 95% bootstrap CI | [−0.82%, +9.45%] | lower ≥+5% | FAIL |
| CI includes 0? | YES | no | FAIL — REVERT trigger |
| Tier-1 byte-identity | 2/3 (MoE residual) | all 7 | FAIL |
| Gate A (strict argmax) | CLEAN | — | PASS |

Phase-0 batch-invariance (2026-08-26) tested real DSv4-Flash shapes: dense bf16 GEMM and SDPA
fail 0-ulp (wq_b 131K ULP, wo_b 65K ULP, fused SDPA 65K ULP); quantized MoE passes; Indexer
query-dependent Jaccard 0.039 → full Kimi design (C_s~1.3) infeasible; indexer-stream-sharing
design chosen (verify-batch-phase0-2026-08-26.md, dspark-verdict-measurement-2026-08-26.md).

### 2.5 Promotion + opt round 1 (Aug 27 2026)

The indexer-stream-sharing G0 design crashed immediately on the first warmup request:
`ValueError: [broadcast_shapes] Shapes (1,1,3) and (1,1,2) cannot be broadcast` — the
`pmask`-not-None assumption violated; `_dispatch_pmask` returns a 3D pmask `(1,1,L_full)` that
was never sliced. In parallel ablation, stream-sharing also killed acceptance −19% (stale row-0
pool snapshots broke top-k sets for rows 1..L-1). REVERT (verify-batch-g0-fail-2026-08-27.md).

Superseded same day by the corrected depth-gated batched M=4 forward (submodule `dda9237`,
parent commit `6eba31ff1`) — pre-rowseq batched code path reintroduced without snapshot logic,
gated to ctx ≥ 8192 → **PROMOTED +36.71%** (dspark-mtp-production-baseline-2026-08-27.md).

| Metric | rowseq (pre-promotion) | batched (promoted) |
|--------|----------------------:|-------------------:|
| C_s | 3.20 | 2.14 |
| Verify cost (ms mean) | 83.76 | 60.60 |
| Acceptance (mean/γ) | 2.118/3 | 2.250/3 |
| @100K tok/s (median, 24 pairs) | ~28.3 | 36.63 (+36.71% vs spec-OFF 27.15) |
| Weakest / strongest pair | — | +14.17% / +56.17% |

G0'' correctness gate: batched-vs-rowseq drift 74.7% ≤ base-vs-base drift 99.3% → PASS. **Flag:**
the 99.3% denominator was later exposed as a fresh-nonce harness artifact (Aug-28); throughput
verdict is unaffected (dspark-352k-correctness-harness-verification-2026-08-28.md).

Concurrent investigations: 14K cliff = regime-mismatch artifact (1455.8 ms@14K was FULLBLOCK
regime; 99 ms@100K is batched regime; no same-regime 14K point exists).
`EXO_DSV4_EXACT_TOPK_PARAM_CAP` shipped env-gated default 64; same-regime A/B pending
(dspark-14k-cliff-investigation-2026-08-27.md). Draft-epilogue fusion shipped default-OFF
(theoretical +16%, cycle 73.7→62.9 ms); A/B deferred
(dspark-draft-epilogue-fusion-2026-08-27.md).

**352K memory regression opened:** 4/16 collapses ~1 t/s; swap 1.37–1.76 GB; root cause
10.13 GB replicated DSpark head → ~99.4 vs ~89.3 GB/node
(dspark-352k-memory-regression-2026-08-27.md).

### 2.6 352K close-out + verification (Aug 27–28 2026)

Three analysis docs quantified the collapse mechanism in sequence. Batched-verify transients:
+4.73 MB/sparse-layer, bounded O(10 MB) in-flight, second-order relative to the 10.13 GB head.
Allocator fragmentation REFUTED (hot-bucket reuse, one-time ~95 MB warmup, not cumulative).
Pool-growth partial amplifier: all 21 sparse layers grow in lockstep every ~1024 decode tokens
→ ~3.85 GB peak (vs 1.95 GB steady) on growth cycles, but collapse rate coarser than cadence →
amplifier only (dspark-352k-batched-verify-transients-2026-08-27.md,
dspark-352k-allocator-pool-analysis-2026-08-27.md).

Residency doc ranked the margin eaters per node: (1) DSpark head weights +10.13 GB, not sharded;
(2) MLX buffer-cache ratchet (monotonic, spec-ON-only, `CLEAR_CACHE_INTERVAL=0`); (3) ~11 MB
alloc+free per verify cycle from `SPEC_STATE_RESTORE` ring copies + pool-meta snapshots. Pool
storage corrected: 1.95 → **2.27 GB** (missed sparse indexer pool +0.45 GB)
(dspark-352k-residency-analysis-2026-08-27.md).

| Fix | Mechanism | Expected savings |
|-----|-----------|-----------------|
| `EXO_MLX_CLEAR_CACHE_INTERVAL=64` | eliminates MLX buffer-cache ratchet | zero code; 5–15% throughput trade-off |
| `EXO_DSV4_DSPARK_TP_SHARD=1` (commit `2d85ccdcb`) | shards DSpark FFN experts across TP group | ~3–3.5 GB/node |
| exo `75d2402dd` JIT wait fix | polls through ALL `JitPlacementUnavailableError` reasons | eliminates launch oscillation |

**verbon3 validation (arm C, n=8):** 0/8 collapses; median 28.44 vs stripped-OFF 24.19 =
**+17.57%** (CI +8.7..+27.8%); swap peak 97 MB; wired peak 92.0–93.5 GB
(dspark-352k-verification-preregister-2026-08-28.md).

**Correctness verification ALL PASS:** Tier-1 7/7 ×2, B0=B1 byte-identical across two separate
instance relaunches, S-triple byte-identical (S0=S1=S2 @352.6K), soak 4000/4000 tokens @30.46
tok/s (median gap 0.08 ms, max 376 ms), post-soak 7/7 (dspark-352k-correctness-harness-
verification-2026-08-28.md). Preregistered degenerate-envelope FAIL (base drift = 0.0%, any
deterministic path difference fails; Amendment 3) recorded honestly.

**Fresh-nonce artifact (Aug-28):** `build_prompt()` embeds a `uuid4` nonce per run → the
historical "99.3% base nondeterminism @100K" and "295 vs 977 tokens same prompt" claims were
harness artifacts. Re-test with fixed prompts: 100K spec-off pair byte-identical (979 chunks
each); 352.6K spec-off triple byte-identical. Base decode is byte-deterministic at depth under
fixed prompts (dspark-352k-correctness-harness-verification-2026-08-28.md).

---

## 3. Corrections & retractions ledger

| # | Claim | Where stated | Corrected by | Current truth |
|---|-------|--------------|--------------|---------------|
| 1 | 15.9x context-scaling collapse | dspark-fullblock-context-scaling-cliff-2026-08-04.md | Same-day MAJOR CORRECTION + Aug-24 reattribution | Retracted; fresh relaunch 17.31 t/s @15K; mechanism = FULLBLOCK verify MODE k-multiplier, not DSpark head |
| 2 | "DSpark/MTP not wired for TP" | PERFORMANCE_HISTORY.md §1 | p4-scoping-mtp-for-tp-2026-08-24.md §A | Wired all along; `EXO_SPECULATIVE=0 EXO_DSV4_MTP=0` env-disabled since 2026-07-26 |
| 3 | Zero "PP speculation using DSpark" log lines = missing capability | PERFORMANCE_HISTORY analysis | p4-scoping §A (batch_generate.py:3466) | Expected-by-construction under TP; log inside `isinstance(layer, PipelineLastLayer)` gate |
| 4 | "DSpark costs nothing to warm under TP" | start_cluster.sh:1712 | p4-scoping §A | 10.13 GB unified memory resident per node |
| 5 | γ=2 sweet spot | PERFORMANCE_HISTORY §5.4 | p4-scoping corrections | PP-era/batched-verify artifact; TP rowseq optimum k=1; batched-verify optimum γ=3 |
| 6 | "EXO_SPECULATIVE=0 permanent default" | exo-speculative-decode-correctness skill | p4-scoping §B; fixed 2026-08-02 | Historical; `start_cluster.sh:382` flipped back to `:=1`; promoted 2026-08-27 |
| 7 | "99.3% base nondeterminism @100K" + G0'' envelope logic | dspark-mtp-production-baseline-2026-08-27.md | dspark-352k-correctness-harness-verification-2026-08-28.md | Fresh-nonce harness artifact (`build_prompt()` uuid4 per run); throughput verdicts unaffected |
| 8 | "Rare/1-in-8" stall characterization | dspark-fullblock-context-scaling-cliff-2026-08-04.md | Same doc SEVERITY CORRECTION | 2/11 = **~18%** measured GPU-idle stall rate |
| 9 | 81 mtp params | dsv4-0731-dspark-native-head-plan-2026-08-03.md / early circulation | p4-scoping §A | **4705 on-disk keys** for -0731 across `mtp.0/1/2`; 81 is post-`sanitize()` module count |
| 10 | Pool storage 1.95 GB | dspark-352k-batched-verify-transients-2026-08-27.md | dspark-352k-residency-analysis-2026-08-27.md | **2.27 GB** — sparse indexer pool (+0.45 GB) missed in root-cause doc |
| 11 | 1455 ms@14K vs 99 ms@100K cliff | dspark-cs-profile-2026-08-26.md | dspark-14k-cliff-investigation-2026-08-27.md | Regime mismatch (FULLBLOCK rowseq vs batched); no same-regime 14K point exists |
| 12 | "Champion 31.5/32.3 t/s" May claims | Early campaign notes (`champion-2026-05-17-fenced`, `champion-2026-05-18-acksync`) | PERFORMANCE_HISTORY.md §5.1 | Not reproducible (32.3 redeploy cratered to 4.3 t/s); only σ-verified numbers count (30.06 σ=0.06; later 30.77 σ=0.067) |

---

## 4. Standing documentation gaps (audit findings)

- **(a)** `PERFORMANCE_HISTORY.md` §1 header (~lines 86–127) describes pre-promotion state; its own Aug-24 correction is now doubly superseded by the Aug-27 promotion. Needs a promotion-era correction block pointing at this master doc.
- **(b)** §5.3 FULLBLOCK entry + Aug-24 correction: "cluster runs DSpark OFF" chain is now doubly stale — spec has been ON since 2026-08-27.
- **(c)** §5.1 "ceiling ~30–35 tok/s" framing superseded: 36.63 tok/s @100K is the measured production number.
- **(d)** §5.2 thematic section lacks the promotion and 352K entries; they exist only in the chronological tail (~line 4486+) of `PERFORMANCE_HISTORY.md`.
- **(e)** `eagle_k1_fix_report.md` sits in the repo root, not `docs/` — should be moved.
- **(f)** June c=2 Bug-3 residual (80% adversarial final-digit flip) has never been re-tested under TP batched verify.
- **(g)** `SOAK_long` JSON field `n=3465` vs correctness doc "4000/4000 tokens" — chunk-vs-token count mismatch, cosmetic; no data error.
- **(h)** Concurrency (c=2) has **never** been validated under the promoted config — all campaign runs used c=1. At γ=3, c=2 gives B×L=8, exactly at the rowseq gate boundary (<8K path), and the ≥8K batched path is unvalidated at B=2.

---

## 5. Open items & watch list

| Item | Status | Next action |
|------|--------|-------------|
| Draft-epilogue A/B @100K + 352.6K | **CLOSED 2026-08-28: gate FAIL, stays OFF** | Byte-lossless (Tier-1 7/7, cross-arm identical both depths) but −0.35% @100K / −0.26% @352.6K; epilogue draft is synchronous, cost moved to accept phase (dspark-p1-draft-epilogue-ab-results-2026-08-28.md) |
| 14K same-regime A/B + `EXACT_TOPK_PARAM_CAP` validation | **CLOSED 2026-08-29: MIN_CTX=8192 SUPPORTED** | P4a batched 14K/32K = 56.1/54.5 ms vs P4b rowseq 78.7/75.2 ms (+40%/+38%); no crossover above 8K; 1455 ms claim retired (rowseq 14K = 78.7 ms same-stack) |
| `TP_SHARD` vs `CLEAR_CACHE=64` ablation | **CLOSED 2026-08-29: both fixes kept** | Shard = +4.84% @352.6K (dominant term, >10×); CLEAR_CACHE=64 = −0.35% (~free); 0 collapses any arm; arm-CACHE hash `d56b8dd1` deterministic within-arm but byte-divergent from TP_SHARD=1 arms (expected numerics, documented) |
| Remaining ~6.5–7 GB replicated head (attention/main_proj/markov) | Not sharded; headroom gap vs spec-OFF persists | Further sharding or harder quant; low priority, real risk |
| Fix #3 `SPEC_STATE_RESTORE` gating | Unimplemented; ~11 MB alloc+free per verify cycle continues | Gate to non-rotating-ring high-acceptance cycles; ~5.5 MB/cycle saving |
| jaccl `WC_ERR` segfault | n=1 auto-recovered; first ever in campaign; 2 verbon3 slots voided | Watch; escalate if recurs |
| 352.6K CI floor +8.7% < +15% bar | Collapse-elimination conclusive; throughput margin thin | State "margin thin" — do not claim full +15% bar |
| c=2 under promoted config | **Correctness legs DONE 2026-08-28**: spec legs clean (Bug-3 0/6, deep B=2 deterministic); TWO shared-generator bugs found (short-c2 degen + BS=2 abort reshape crash), both repro spec-OFF | Fix the two shared bugs (see dspark-p2-c2-validation-results-2026-08-28.md); then c=2 spec-OFF throughput control |
| G0'' methodology revision | Fresh-nonce artifact exposes denominator flaw | Adopt fixed-prompt + per-arm-determinism framing as standard correctness gate |
| mxfp4/mxfp8 numerics under axis-1 sharding long-run | Deployed but not long-soak validated under TP shard | Monitor production logs; escalate on any U+FFFD or repetition |

---

## 6. What's next (PM plan, priority-ordered)

### P1 — Draft-epilogue fusion A/B @100K + 352.6K

**CLOSED 2026-08-28 — GATE FAIL (throughput), flag stays default-OFF.**
Pre-registered campaign `dspark-p1p4-campaign-preregister-2026-08-28.md`; results
`dspark-p1-draft-epilogue-ab-results-2026-08-28.md`. Correctness perfect (Tier-1
7/7 byte-identical; ON/OFF cross-arm byte-identical at BOTH depths; all arms
internally deterministic; 0 collapses @352.6K). Throughput: −0.35% @100K,
−0.26% @352.6K. Mechanism confirmed engaged (consume-cycle draft 8.2→0.55 ms)
but the epilogue draft runs synchronously in the accept window (+7.9 ms) — no
overlap exists to hide it; cycle −1.8%, end-to-end ≈ 0. The theoretical +16%
assumed overlap the single-Metal-stream reality doesn't provide.

### P2 — c=2 validation under promoted config

**CLOSED (correctness legs) 2026-08-28 — two NEW shared-generator bugs found,
NEITHER in the spec path** (`dspark-p2-c2-validation-results-2026-08-28.md`):
(1) c=2 system+user short-prompt degeneration — `.</think>Paris` period-3 loop,
kill-switch at token 61, 3/3 spec-ON AND 2/2 spec-OFF (mechanism-independent,
c=1 clean); (2) BS=2 degen-abort reshape crash — mlx-lm `cache.py:2050
fetch_overlap_carry` `[reshape] size 2 into (1,1,1,1)` kills BOTH streams +
instance (availability bug, also spec-OFF). Spec-specific legs ALL CLEAN: deep
batched B=2 deterministic + zero contamination; **Bug-3 adversarial final-digit
0/6 flips under TP batched verify** (PP-era ~80% class not reproduced); c=2
spec-ON @100K = 10.0/9.7 tok/s per stream (19.7 aggregate; B=2 cycles are
per-row — BS>1 batched verify remains the Phase-5 TODO). Outstanding: c=2
spec-OFF throughput control (L2 leg cut at session wrap-up). c=2 NOT
production-ready until the two shared bugs are fixed; c=1 promoted config
unaffected.

### P3 — verbon3 ablation

**DONE 2026-08-29 (L3+L4+L6 anchor).** Full numbers in
PERFORMANCE_HISTORY 2026-08-29 entries + machine verdict
`/tmp/ab/p1p4/p3_verdict.json` (shared-window W=2097 recomputed across all
arms; `analyze_p3_final.py`):
- arm-BOTH (P1-OFF L0 trio): 30.446/30.482/30.519 (median 30.482)
- arm-SHARD (L3): 30.582/30.590/30.618 (median 30.590, σ=0.019 = 0.06% CV)
- arm-CACHE (L4): 29.000/29.008/29.035 (median 29.008, σ=0.018 = 0.06% CV)
- **CLEAR_CACHE=64 cost at depth: −0.35%** (windowed; 5–15% estimate
  refuted — the interval is ~free). **TP_SHARD contribution: +4.84%**
  (shard-off = arm-CACHE 29.008 vs arm-BOTH 30.482) — the dominant term
  of the bundled pair by >10×. Both fixes stay in production.
- Collapses 0/3 on every arm (arm-CACHE accepted-risk arm did not
  collapse; n=3 distinguishes ~0 from ~1 rates only).
- Memory: replicated head costs ~+4–5 GB/node settled wired (arm-CACHE
  97.3–101.5 GB peaks vs arm-BOTH 92.3–96.2); swap flat 50–65 MB.
- Byte-identity: arm-CACHE deterministic within-arm (hash `d56b8dd1`)
  but byte-divergent from the TP_SHARD=1 arms (`f08efa3f`) — first
  divergence one mid-reasoning coin flip at output char ~1967; expected
  numerics (head replication changes reduction order), NOT a correctness
  regression. Production (TP_SHARD=1) remains byte-lossless across the
  CLEAR_CACHE axis.
- Cross-session anchor (L6, production restore): see PERFORMANCE_HISTORY
  2026-08-29 L6 entry — fresh 352.6K within ±5% of arm-BOTH median
  validates arm-BOTH reuse.

### P4 — 14K same-regime verify-cost curve

**P4a (batched arm) DONE 2026-08-28** on the production launch (L0), fixed
prompts, MTP-PROF windowed per-run means (m4-1, `/tmp/ab/p1p4/p4_phase_extract.json`):

| ctx | regime (by MIN_CTX=8192) | verify ms | total ms | draft ms |
|-----|--------------------------|----------:|---------:|---------:|
| 4K | rowseq | 79.5 | 92.2 | 8.2 |
| 7.5K | rowseq | 78.3 | 91.3 | 8.4 |
| 9K | batched | **56.1** | 68.6 | 8.2 |
| 14K | batched | **56.1** | 68.7 | 8.2 |
| 32K | batched | 54.5 | 68.0 | 8.5 |
| 64K | batched | 59.2 | 72.0 | 8.2 |
| 100K | batched | 64.2 | 76.0 | 8.2 |
| 352.6K | batched | 84.8 | 96.5 | 8.2 |

**P4b (rowseq arm) DONE 2026-08-29** (L5, `VERIFY_BATCH=0
VERIFY_ROWSEQ=1`, same frozen prompts, warm-flush first per window-bias
note): rowseq verify 14K = **78.73 ms** (total 89.21, n=200), 32K =
**75.20 ms** (total 85.64, n=250).

**MIN_CTX placement verdict (pre-registered rule): SUPPORTED — keep 8192.**
Batched verify is cheaper at both crossover depths (14K: 56.1 < 78.7 =
+40.4% rowseq penalty; 32K: 54.5 < 75.2 = +38.0%); the two regimes are
parallel plates (rowseq ~78–80 ms flat 4K→32K, batched ~54–56 ms flat
9K→32K) with a one-time step at the 8K gate and NO crossover above 8K, so
no depth exists at which raising MIN_CTX above 8192 would pay. Floor
stays ≥8K regardless (correctness asymmetry, pre-registered). The
historical "1455.8 ms @14K" is retired with a same-stack rowseq
measurement: 78.7 ms (~18.5× lower; old number was FULLBLOCK-regime).
End-to-end corroboration: decode tok/s batched 41.3 vs rowseq 31.3 @14K,
39.3 vs 28.7 @32K; TTFT identical (prefill unaffected).

### P5 — Doc hygiene

**Rationale:** Multiple stale docs will mislead future analysis if not corrected soon.

**Actions:**
- Patch `PERFORMANCE_HISTORY.md` §1 (~lines 86–127) and §5.1/§5.2/§5.3 with correction blocks
  pointing at this master doc (dspark-mtp-master-history-2026-08-28.md).
- Move `eagle_k1_fix_report.md` from repo root into `docs/`.
- Patch `exo-speculative-decode-correctness` skill header — still describes PP-era config and
  states "cluster runs spec-off" in places that are now doubly stale.

### NOT FUNDED

The following were explicitly evaluated and declined:

- **Further head sharding beyond FFN** (attention/main_proj/markov remain replicated, ~6.5–7 GB):
  incremental win, real numerics risk under axis-1 sharding at group_size=32.
- **Custom rowseq-GEMM Metal kernels** for wq_b/wo_b/lm_head: not the dominant cost; fused SDPA
  fails 0-ulp (65K ULP), making the full C_s~1.3 Kimi design infeasible regardless.
- **Tree drafting revival:** verify-floor-dominated; May PP data; DSpark draft geometry at full
  block_size=5 is negative at most acceptance rates.
- **Context caps:** rejected on principle — production workload legitimately uses 352.6K+.

---

## Appendix A: Source-document index

All primary sources from the six era digests, chronological order.
`PERFORMANCE_HISTORY.md` (ongoing ledger, §5 + chronological tail ~line 4486+) is the running
record this master doc consolidates; it is noted separately and requires the §1/§5 correction
patches per gap (a)–(d).

| # | Filename | Date | One-line role | Verdict / outcome |
|---|----------|------|---------------|-------------------|
| 1 | `eagle_k1_fix_report.md` (repo root) | ~2026-05 | Diagnose + fix Eagle K=1 17× regression from `21ba40db` all_gather on c=2 draft critical path | Fix proposed (K=1 short-circuit); no commit SHA in doc |
| 2 | `docs/mtp-tiebreak-losslessness-fix.md` | 2026-06-04 | Root-cause + fix for ~1 ulp tiebreak losslessness violation in c=1 MTP (spurious `</think>`) | SHIPPED `c840bc2d`; c=1 champion 30.77/30.8 t/s; 100% correctness |
| 3 | `docs/deepseek-v4-mtp-performance.md` | 2026-06-04 | PP-era c=1 champion config performance writeup; all tuning knobs, acceptance metrics, quality gates | Tagged `validated-upstream-integration-20260604-153609`; stale under TP |
| 4 | `docs/deepseek-v4-c2-mtp-verify-fixes.md` | 2026-06-06/07 | Three c=2 MTP verify bugs: wide ring, pooled mask, sparse-kernel accuracy | SHIPPED `cb7e3bd` `aaac5c3` `60a0a0c` `491f6fe` `5b00004`; Bug-3 80% flip residual UNRESOLVED |
| 5 | `docs/b2-mtp-resolution-2026-06-24.md` | 2026-06-24 | Two additional B=2 bugs (seq-split all_gather, bootstrap offset); c≥2 gate removed | SHIPPED `8a9cdee` `48a4a3c` `47fdf32a`; prefill 367 t/s @100K B=2 |
| 6 | `docs/dsv4-0731-dspark-native-head-plan-2026-08-03.md` | 2026-08-03 | Plan + impl of -0731 native DSpark head loading; `Model.sanitize()` discovery | SHIPPED `99f5cda51`; gated OFF (`DSPARK_NATIVE=0` default) |
| 7 | `docs/dspark-fullblock-context-scaling-cliff-2026-08-04.md` | 2026-08-04 | FULLBLOCK cliff investigation; 15.9x claim self-retracted same day; ~18% stall confirmed | 15.9x RETRACTED; stall real @~2800 tok; DSpark OFF decision |
| 8 | `docs/flag-sweep-completion-and-dspark-native-finding-2026-08-21.md` | 2026-08-21 | Close flag-sweep; identify `DSPARK_NATIVE` as unexplored (preview-vintage head active in prod) | All flags swept; DSPARK_NATIVE deferred (decode-only, out of prefill scope) |
| 9 | `docs/mtp-dspark-tp-port-decision-gate-2026-08-22.md` | 2026-08-22 | T6 gate: MTP/DSpark TP-port priority decision | NOT recommended; 1.29× < 1.5× gate; 5-week-stale baseline |
| 10 | `docs/p4-scoping-mtp-for-tp-2026-08-24.md` | 2026-08-24 | Deep code-verified TP port scoping; v1 memo + v2 banner with 9 corrections | Conditional GO for k=2; corrects 9 prior claims; correct env = all 3 flags |
| 11 | `docs/p4v2-m1-shadow-gate-results-and-recovery-2026-08-24.md` | 2026-08-24 | M1 shadow gate data (782 cycles); incident recovery (wrong host, dropped env var) | HOLD (a=2.256 vs a*=2.199, +1.8%); cluster reverted |
| 12 | `docs/dspark-mtp-ab-preregister-2026-08-25.md` | 2026-08-25 | A/B pre-register + all stage results appended in-place (γ-bug, EOS-ban fix, corrected protocol) | REVERT (+1.87% median, CI [−0.82%, +9.45%] includes 0) |
| 13 | `docs/dspark-cs-profile-2026-08-26.md` | 2026-08-26 | C_s profile analysis: why spec-decode yields +2% not +55-80% | C_s=3.20; PROMOTE arithmetically impossible without verify-path batching |
| 14 | `docs/dspark-tier1-byte-identity-2026-08-26.md` | 2026-08-26 | Tier-1 byte-identity test on 7-prompt degen set (temp=0, 512 max_tokens) | Tier-1 PARTIAL (2/3); MoE 0.023%/row residual confirmed as mechanism |
| 15 | `docs/dspark-verdict-measurement-2026-08-26.md` | 2026-08-26 | 24-run fixed-window A/B measurement; corrected protocol | REVERT (decisive); closes Aug-24 HOLD with real data |
| 16 | `docs/verify-batch-phase0-2026-08-26.md` | 2026-08-26 | Batch-invariance micro-tests; design decision for indexer-stream-sharing | Indexer-stream-sharing design chosen; SDPA fail = C_s~1.3 infeasible |
| 17 | `docs/verify-batch-g0-fail-2026-08-27.md` | 2026-08-27 | G0 crash (broadcast_shapes mismatch) + stream-sharing −19% acceptance | REVERT; superseded same day by corrected M=4 batched forward |
| 18 | `docs/dspark-mtp-production-baseline-2026-08-27.md` | 2026-08-27 | Promotion record: depth-gated batched M=4 verify; 24-pair A/B | PROMOTED +36.71% @100K; 12/12 pairs; C_s 3.20→2.14 |
| 19 | `docs/dspark-14k-cliff-investigation-2026-08-27.md` | 2026-08-27 | Code-reading: 1455ms@14K vs 99ms@100K claim; regime-mismatch root cause | Cliff artifact; `EXACT_TOPK_PARAM_CAP` shipped gated; same-regime A/B pending |
| 20 | `docs/dspark-draft-epilogue-fusion-2026-08-27.md` | 2026-08-27 | Draft-epilogue fusion design + impl; overlaps 10.8 ms draft with prior cycle tail | SHIPPED code default-OFF; A/B deferred; theoretical +16% |
| 21 | `docs/dspark-352k-memory-regression-2026-08-27.md` | 2026-08-27 | 352K collapse investigation, root cause (10.13 GB replicated head), fix | Fix PROMOTED (`DSPARK_TP_SHARD=1` `2d85ccdcb`); pre-fix gate FAIL −17.87% |
| 22 | `docs/dspark-352k-batched-verify-transients-2026-08-27.md` | 2026-08-27 | Quantify batched-verify transient allocations per sparse layer (+4.73 MB) | Fragmentation REFUTED; pool growth partial amplifier; O(10 MB) second-order |
| 23 | `docs/dspark-352k-allocator-pool-analysis-2026-08-27.md` | 2026-08-27 | MLX Metal allocator fragmentation vs pool-growth analysis; pool corrected | Fragmentation REFUTED; pool 1.95 → 2.27 GB; ratchet mechanism identified |
| 24 | `docs/dspark-352k-residency-analysis-2026-08-27.md` | 2026-08-27 | Per-node steady-state + per-cycle residency; margin eater ranking; ranked fixes | 10.13 GB head primary; MLX ratchet secondary; fixes ranked and sequenced |
| 25 | `docs/dspark-352k-verification-preregister-2026-08-28.md` | 2026-08-28 | Pre-registered correctness + harness verification protocol for verbon3 @352.6K | Protocol defined; Amendments 2-3 degenerate-bar FAIL recorded honestly |
| 26 | `docs/dspark-352k-correctness-harness-verification-2026-08-28.md` | 2026-08-28 | Live correctness + soak verification against full stacked production config | ALL PASS (Tier-1 7/7, soak 4000/4000 @30.46, 34/34 factual); fresh-nonce exposed |
| 27 | `docs/dspark-352k-verification-runs-2026-08-28.json` | 2026-08-28 | Raw run records (9 entries) for 352.6K verification campaign | Median 28.44 vs 24.19 (+17.57%); 0 collapses; SOAK_long 30.46 tok/s |
