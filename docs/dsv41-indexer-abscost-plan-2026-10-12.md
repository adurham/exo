# DSv4.1 — Indexer absolute-cost workstream — EXECUTABLE PLAN (Fable consult #5, saved 2026-10-12)

**Provenance:** requested by the owner ("get a detailed plan from fable on that and save it — we'll execute in a new session"). Produced by the reference-model consult (Fable) on 2026-10-12, saved verbatim below with this header. **NOT executed.** The owning session starts at the "New-session handoff" section at the end.

**House-convention mapping:** the plan's deliverable paths (docs/perf/*.md, branch perf/indexer-profile) map to this repo's convention: put deliverable docs at `docs/dsv41-<name>-<date>.md` on the appropriate branch/main (see the restart-survival plan + buildnote precedents), branches named as stated. Everything else stands as written.

**Base state at save time (measured, live):** exo `deploy/next23b-qsplit` @ `60008939`, mlx-lm @ `6a5c78f4`; flags `EXO_DSV41_CKPT=1` + `DSV41_INDEXER_QSPLIT=1`; prefill ~300 rows/s @50K / ~291 @105K after the qsplit win (+10.5% / +11.9%, token-identity 5/5, battery clean). The launcher forwards only `DSV41_*` names (`EXO_DSV41_*` = silent no-op). HIER default ON since 2026-10-06.

---

# Indexer Absolute-Cost Workstream: Executable Plan

**Base state (MEASURED, live):** exo `deploy/next23b-qsplit` @ `60008939`, mlx-lm @ `6a5c78f4`. Flags: `EXO_DSV41_CKPT=1`, `DSV41_INDEXER_QSPLIT=1`. Prefill is ~300 rows/s at 50K and ~291 at 105K. The indexer is 31.1% of a 2048-row chunk at 100K and 42.9-44.2% deep (both measured pre-split).

**Plan shape:** Phase 1 and Phase 2 run offline. Then **one** combined live window, which needs the user's go: a Phase 1 observation pass with no boot, plus the Phase 2 treatment boot. Phase 3 is an offline decision doc plus first experiments. There are **two user-facing surfaces total** (M1 = live-window request, M2 = final report and Phase 3 recommendation), plus any pre-registered abort.

---

## 0. Before anything: resolve three ambiguities

Do this first, about 1 hour. These are the biggest risks of executing this mechanically.

### 0.1 Pin the definition of `nb`

The context says HIER is "1.71-2.05x slower at *small* nb (nb=16384)". But a 2048-row chunk split across 2 ranks gives nb_local = 1024. So `nb` is probably not "query rows per rank". It may be rows x heads, rows x batch, or a key-block count.

- Read the HIER code path in mlx-lm @ `6a5c78f4`.
- Write in the profile doc the exact formula for `nb` and for the score-tensor element count `S`, in terms of (query rows global, query rows local, idx heads, L).
- **The whole gate design depends on this. Do not proceed on an assumed definition.**

### 0.2 Pin "verify-ratio"

Search the repo docs and prior consult/plan docs for the term.

- **If undefined**, use this definition: *for MTP/speculative verify steps (and plain decode), the fraction of step wall-time spent in the indexer, plus the rows-per-verify and the acceptance ratio.* This is the small-nb, full-L regime the bypass targets, so it matters for Phase 2 EV.
- If the cluster does not run speculative decoding, the probe reduces to "decode-step indexer share at L in {50K, 100K}".

### 0.3 Locate the instrumentation points

Find:
- the indexer forward function(s): reference path vs HIER path;
- the qsplit band assignment;
- the top-k call;
- where the index-K cache lives (dtype, layout);
- the launcher env-forwarding code. Confirm the `DSV41_` prefix forwarding rule and note it in the doc.

**Deliverable:** section 0 of the profile doc (see house-convention mapping in the header) on worktree branch `perf/indexer-profile` off mlx-lm `6a5c78f4`.

**Abort-to-user condition:** if `nb` cannot be pinned unambiguously from code, stop and surface with the specific question. Everything downstream depends on it.

---

## Phase 1: Profiling bundle (about 1 day; offline except one no-boot observation pass)

### Bench hygiene (applies to every offline bench on studio2)

**Idle guard**
- Before each run, require no live request in flight and no GPU activity for at least 60 s. Check via the exo API or log tail, plus `ps` and GPU-busy sampling.
- Poll during the run, every 1-2 s. If traffic appears, **discard the sample**; do not just pause.
- Log every discard.

**Memory cap**
- studio2 holds roughly half the 552B model, so free unified memory is small.
- Compute peak bytes per bench config *before* running it. A reference-path score tensor at nb=16384 x L=130K in fp32 is about 8.5 GB.
- Skip any config whose predicted peak exceeds a stated cap. Measure free memory first, and set the cap at 50% of free.
- Never risk OOM-pressuring the live process.

**Statistics**
- Warm up >=3 runs, then take >=7 timed runs.
- Interleave A/B order (ABAB...), not AAA-BBB.
- Report median and IQR.
- Every number is labeled MEASURED / DERIVED / BOUND / MODELED.

**Inputs**
- Use **real activations** where possible. Capture `q_idx`, the index-K cache, and any per-head weights from one real layer at L ~ 50K and ~ 105K.
- Get these from the existing dump mechanism if one exists. Otherwise, write an offline replay harness that loads one layer's indexer weights plus a saved KV state; no node-side edits.
- Synthetic random data is only acceptable for timing, never for tie analysis.

### 1A. Indexer time split (score-GEMM / top-k / K-cache bytes) at L in {50K, 75K, 100K, 130K}

**Harness:** `bench/indexer_split.py` on the branch. It calls the indexer sub-ops of **both** paths (reference and HIER) at per-rank shapes after qsplit (nb_local, L).

**Two modes per config:**
1. **Barriered.** Put `mx.eval(...)` + `mx.synchronize()` after each sub-op: (i) index-K read/projection, if any, (ii) score GEMM plus head-weighting/ReLU/sum, (iii) top-k/argpartition, (iv) gather/scatter of selected indices.
2. **Unbarriered.** Time the whole indexer call end to end.

**Validity check:** the sum of barriered parts must be within +-15% of the unbarriered total.
- If it is, report the split as MEASURED.
- If not, report each part as BOUND (an upper bound on its standalone cost) and fall back to Metal capture (`mx.metal.start_capture` -> Xcode GPU trace; note: headless xctrace works on these nodes per the 2026-10-11 pack, command-buffer level), for one config (100K) only, to get per-kernel GPU time.

**Roofline (DERIVED):** for each sub-op, compute
- bytes moved: K-cache read = L x d_idx x bytes; score write/read = S x 4 for fp32 (or the tile equivalent for HIER);
- FLOPs = 2 x nb_local x L x H_idx x d_idx.

Compare against **measured** peaks on studio2:
- a 1 GB `mx.array` copy for stream bandwidth;
- a large fp16 matmul for FLOP/s.

Do not use spec-sheet numbers. Classify each sub-op as bandwidth-bound (>=60% of measured BW), compute-bound (>=60% of measured FLOP/s), or neither (overhead/latency-bound).

**Acceptance:** a table per L with ms and % of indexer time for each sub-op, the roofline class, and the validity-check result. At least the 100K and 130K rows must be MEASURED, not BOUND.

### 1B. nb x L crossover map (reference vs HIER)

**Grid**
- nb (in the pinned definition from 0.1) spanning: decode/verify (1-8 rows), tail-chunk sizes, the full chunk per rank (1024 rows equivalent), and the nb=16384 point that reproduces the known 1.71-2.05x.
- L in {4K, 8K, 16K, 32K, 50K, 75K, 100K, 130K}.
- Skip configs over the memory cap and record them as "not run (mem)".

**Measure:** the unbarriered indexer time for both paths, plus the peak memory delta for both. Use `mx.get_peak_memory()` reset per run.

**Reproduction check (mandatory):** the nb=16384 point must reproduce 1.71-2.05x +-20%, and the 100K full-chunk point must reproduce "neutral" (0.9-1.1x).
- If it does not, the harness is not representative. Fix it, or stop Phase 2 and surface.

**Output**
- A heatmap/table of speedup = t_HIER / t_ref against nb and L.
- The fitted crossover contour expressed as a threshold on S, in elements.
- Check whether a pure-S threshold fits. "Fits" means it misclassifies no cell where |speedup - 1| > 10%.
- If it doesn't fit, record the 2-parameter form needed, for example `S <= S* AND L <= L*`.

### 1C. Workload occupancy (needed for honest Phase 2 EV)

**Source:** existing live logs or traces of the real agentic workload. This is read-only.

**Measure:** time-weighted occupancy of the (nb, L) cells. That is, what fraction of total GPU wall time is spent in prefill full chunks, prefill tail chunks, decode steps, and verify steps, and at what L.

- If logs lack this, DERIVE it from turn-length and context-length distributions in the logs, and label it as such.

**Phase 2 EV (MODELED)** = sum over cells of occupancy x indexer share x (1 - 1/speedup), counting only cells where the reference path is faster.

### 1D. Replication check (offline code audit + one live observation)

**Code audit.** List every indexer-adjacent op still executed identically on both ranks after qsplit:
- index-K projection of new rows;
- top-k on full rows;
- all-gather of selected indices;
- per-head weights.

For each, estimate ms at 100K (DERIVED from 1A numbers).

**Band balance.** The causal mask means early rows in a band have a shorter effective L.
- Compute the per-rank work imbalance analytically for chunk position vs L (DERIVED). It is negligible at 100K and could matter at L ~ chunk size.
- Confirm with per-rank timing in the live observation (1E).

**Cross-rank agreement.** Confirm from code that both ranks make the same path decision (HIER vs reference) for a given chunk. This is a requirement carried into Phase 2.

### 1E. GPU-busy capture and verify-ratio probe (live, observation-only, no boot; runs inside the M1 window)

Run on the **current live build**, no config change, quiet-cluster precondition met:
- one scripted prefill at about 50K and one at about 105K (existing bench prompts);
- one decode run of >=256 tokens at about 100K.

**GPU-busy**
- Run `sudo powermetrics --samplers gpu_power -i 100` on **both** nodes, time-aligned.
- Report GPU active residency during prefill chunks and during decode.
- Optional: one `xctrace record --template 'Metal System Trace' --attach <pid> --time-limit 20s` on one rank during a 100K chunk, but only if the existing process permits attach without restart. If it needs a restart or env change, skip it; do not spend a boot.

**Per-rank timing.** Use existing log timestamps for per-chunk time per rank to quantify band skew from 1D.

**Verify-ratio** (per the definition from 0.2): verify/decode step time, rows per verify, acceptance, and the indexer share. Estimate the indexer share from 1B timings at those shapes (DERIVED) if live logs lack per-op timing.

**Acceptance:** GPU-busy % per phase per node, per-rank skew, and verify metrics, all labeled.

### Phase 1 stop conditions (pre-registered)

| Condition | Action |
|---|---|
| 1B reproduction check fails after one harness fix attempt (<=1.5 h) | Stop Phase 2 offline work. Write up. Surface at M1 with Phase 1 data only. |
| Memory cap prevents running any config >=100K | Proceed with BOUND-only 1A. Flag. |
| Total Phase 1 offline exceeds 1.5 days | Freeze what exists, write up, and move to Phase 2 only if 1B succeeded. |

### Phase 1 deliverables

- Profile doc on `perf/indexer-profile`: section 0 definitions, 1A-1E tables, roofline, crossover map, occupancy, Phase 2 MODELED EV, Phase 3 inputs (path per house-convention mapping in the header).
- Bench scripts committed on the branch and pushed to the fork.
- A PERFORMANCE_HISTORY.md entry, "profiling (no perf change)", same session.

**Phase 1 EV:** information only, 0% direct gain. Its value is that it decides whether Phase 2 is worth a boot and which Phase 3 lever (if any) to pursue.

---

## Phase 2: Small-batch HIER bypass (0.5-1 day offline + one live round)

### Go/no-go entering Phase 2 (from Phase 1)

Proceed only if **both** hold:
- (a) 1B shows a contiguous region where the reference path is >=1.15x faster than HIER;
- (b) the 1C MODELED e2e EV is >=1.0% on the real workload.

Otherwise, write "bypass not worth a boot" with the numbers and skip to Phase 3.

> **Honesty note:** at full prefill chunks at 50-100K, HIER is "neutral", so the bypass win comes from tail chunks, short-L turns, decode, and verify. The 2-3% estimate may be optimistic for this workload. 1C is what confirms or kills it.

### 2.1 Gate design

```
use_reference = BYPASS_ENABLED and S_global <= S_STAR [and L <= L_STAR if 1B required 2-param]
```

**Computed from global quantities** (global chunk rows, L), not local nb, so both ranks always take the same path. Divergent paths across ranks plus different tie-breaking would produce rank-dependent selection, which is unacceptable.

**Threshold choice.** Set S_STAR at the largest S where the reference path is >=1.10x faster in 1B across all L, with a ~25% safety margin toward HIER. HIER is the memory-safe default, so bias toward it.

**Hard memory ceiling, independent of speed.** Use the reference path only if the predicted score-tensor bytes are <= `DSV41_INDEXER_REF_MAX_BYTES`. Default to a value that cannot pressure memory at the live footprint; take it from the measured headroom in 1B.

**Env vars** (must be `DSV41_`-prefixed; `EXO_DSV41_` is a silent no-op):
- `DSV41_INDEXER_HIER_BYPASS=0|1` (code default 0);
- `DSV41_INDEXER_HIER_BYPASS_SMAX=<elements>`;
- `DSV41_INDEXER_REF_MAX_BYTES`.

**Engagement counter.** Count path decisions per (nb, L) bucket. Log a one-line summary every N chunks/steps and at shutdown, for example `HIER_BYPASS ref=123 hier=456`. This is mandatory, because the culture has already been bitten by silent no-ops.

### 2.2 Top-k tie-overlap / exactness validation (offline; mandatory, before any live round)

Run on real-activation dumps (>=2 contexts, >=4 layers spread across depth, both ranks' bands) and on adversarial synthetics (scores quantized to force exact ties; duplicated K rows), at every gated-in (nb, L) cell. Use the HIER path as the baseline, because HIER is what's live.

| Level | Check | Pass |
|---|---|---|
| L1 | Score values before selection, compared elementwise between paths | bit-identical (max abs diff = 0) |
| L2 | Selected index **sets** per row | identical on 100% of real-activation rows |
| L3 | If L1 fails | report the max abs diff and count rows whose k-th/(k+1)-th score gap is <= the observed diff (the at-risk rows) |
| L4 | Ties (synthetic) | characterize each path's tie-break rule (lowest index? tile order?) |
| L5 | Ordering of selected indices, if downstream is order-sensitive (sparse-attention gather + softmax is normally order-invariant, but **verify in code**) | identical order, or proven order-invariant |

**Decision**
- **L1 and L2 pass on all real data:** proceed.
- **L1 passes, synthetic ties diverge:** make the bypass path's tie-break match HIER's rule exactly. This must be a deterministic, documented rule; do not change HIER, because that would change live behaviour. Re-run until synthetic sets are identical.
- **L1 fails** (different reduction order produces different floats): attempt to make the reference path's reduction order match HIER's, <=2 h.
  - If that's not possible and L2 shows **any** real-row divergence, **abort Phase 2** as "not exact". Write it up.
  - If L1 fails but L2 shows 0 divergent real rows, do not treat that as safe. Proceed to the live token-identity gate, but pre-register that any token divergence aborts (no "probably a tie" waivers).

### 2.3 Bounded implementation

- Worktree branch `perf/hier-bypass` off mlx-lm `6a5c78f4`. Add an exo branch off `60008939` only if launcher or env changes are needed; ideally none.
- Change scope:
  - the gate function;
  - the env reads;
  - the engagement counter;
  - the tie-break alignment from 2.2, if needed.
- **No other edits.** No refactor of HIER.
- Unit tests:
  - the gate decision table: global vs local nb, memory ceiling;
  - flag-off is byte-identical in behaviour to `6a5c78f4` (path counter shows 100% HIER);
  - path-equality tests from 2.2 at 3 representative shapes.

**Offline verification**
- Re-run 1B at the gated cells with the flag on. Speedups must match 1B +-15%.
- Run the cells just above S_STAR and confirm HIER is still used (counter).

**Timebox:** 1 day for 2.1-2.3. If exceeded, stop, write up the state, and surface at M1 without a live request for Phase 2.

### 2.4 Quiet-cluster live validation protocol (inside the M1 window, after 1E)

**Request to the user at M1:**
- builds/shas;
- flags;
- boot budget: 1 treatment + 1 reserve, plus a mandatory revert boot if not shipped. Ask the user to confirm whether the revert boot counts against the budget;
- expected EV;
- gates.

**Sequence**

1. **Precondition.** Quiet cluster: no user session active, verified by API and log tail, held for >=5 min.
2. **Baseline (no boot)**, on the current live build:
   - token-identity reference set: the 5 standard prompts, greedy;
   - quality battery reference: reuse the stored reference if it is from this exact build;
   - perf: prefill at 50K and 105K, x3 each; decode at 100K, x3; plus one **short-context agentic turn** (L ~ 4-16K, small tail chunks), since that's where the bypass should show up.
3. **Treatment boot.** Same shas plus the `perf/hier-bypass` mlx-lm sha. Add `DSV41_INDEXER_HIER_BYPASS=1` and `DSV41_INDEXER_HIER_BYPASS_SMAX=<S_STAR>`.
4. **Mechanical gates** (both nodes, before any measurement):
   - `ps eww <pid>` shows `DSV41_INDEXER_HIER_BYPASS=1`, `DSV41_INDEXER_QSPLIT=1`, the expected SMAX, and the CKPT flag;
   - the running mlx-lm sha equals the expected one (from the version/log line);
   - no `EXO_DSV41_INDEXER_*` names present.
5. **Engagement check.** After one warm prompt, the counter shows ref>0 **and** hier>0 on both ranks, with identical counts across ranks. Equal counts are what confirm the global gate.
   - ref=0 -> abort (no-op).
   - Mismatched counts across ranks -> abort.
6. **Token identity:** 5/5 vs baseline. Any mismatch -> abort.
7. **Quality battery:** pass per the existing thresholds.
8. **Perf:** same set as baseline, same order. Ship criteria:
   - (a) no metric regresses beyond the baseline's own spread (max-min of the 3 runs) or 1%, whichever is larger;
   - (b) at least one of short-context turn time, decode tok/s, or deep prefill improves by >=1% MEASURED, outside the noise spread;
   - (c) peak memory per node increases <= the predicted ceiling (from `vm_stat`/process RSS before and after).

**Abort (auto-revert immediately to `60008939` + `6a5c78f4` with the current flags) on any of:**
- a mechanical, engagement, identity, or quality failure;
- an error or OOM;
- a regression beyond (a);
- the user's traffic arriving mid-window.

Use the reserve boot **only** if the treatment boot failed for infrastructure reasons, not to retry a failed gate with different settings.

**Outcomes**
- **Ship:** leave the treatment live. Record the new live state (shas, flags) in PERFORMANCE_HISTORY.md and the deploy doc.
- **Pass gates but fail (b):** revert. Keep the flag in code, default off. Document as "neutral, not shipped".

### Phase 2 deliverables

- Bypass buildnote (path per house-convention mapping): gate math, S_STAR derivation, the 2.2 exactness tables, live results, ship decision.
- Branch pushed to the fork.
- PERFORMANCE_HISTORY.md entry, same session.
- The deploy branch updated only if shipped.

**Phase 2 EV (MODELED, to be replaced by the 1C number)**
- e2e: +0.5% to +3%, centered around +1.5%.
- Larger on decode/verify-heavy and short-context turns; about 0 on deep full-chunk prefill.
- P(shippable >=1%) ~ 50-60%.
- P(exactness abort) ~ 15-20%.

---

## Phase 3: Profile-driven piece-2 decision (decision doc + first experiments only; 2-4 days if pursued)

Do not implement past the first experiments in this plan. Any live round for piece-2 is a **new plan and a new go**.

### Inputs

From 1A at 100K and 130K:
- the post-qsplit indexer share of chunk time, I;
- the sub-op split;
- the roofline classes.

From 1E: GPU-busy %.

**Amdahl reference (DERIVED from the pre-split figure):** a 43% share pre-split, halved by qsplit, gives ~0.215/0.785 ~ 27% post-split. **Replace this with the MEASURED I.**

### Decision table (pre-registered)

| Measured condition (deep, 100K+) | Decision |
|---|---|
| GPU-busy < 80% during prefill on either node | **Stop piece-2.** The bottleneck is dispatch, sync, or comm, not indexer kernels. Write it up as a redirect recommendation (out of scope). |
| I < 15% of chunk time | **Stop piece-2.** Ceiling <= ~7% e2e even at a 2x indexer speedup (DERIVED); not worth 2-4 days. |
| Top-k/select >= 40% of indexer time | **Lever A** (exact selection improvement) |
| K-cache read is bandwidth-bound (>=60% of measured BW) and >= 40% of indexer time | **Lever B** (index-K lower precision), **conditional on user approval of a non-exact change class** |
| Score GEMM is compute-bound and dominant | Neither lever fits. Document the compute ceiling (M4 Max has no fp8 matmul hardware). Stop piece-2 unless an int8 matmul path in MLX is measured >=1.5x on these shapes; one 2 h microbench is allowed to check. |
| Mixed / no sub-op >= 40% | Pick the lever with the larger MODELED e2e gain, A preferred on ties because it's exact. |

### Lever A: exact top-k/select improvement

**First experiment** (offline, <=1 day): a microbench of the current top-k (argpartition/topk, plus HIER's tile merge) against:
- (i) a two-pass threshold select: histogram on the top bits of the fp32 score, find the k-th value bucket, then a compacting gather. This is radix-select style, as a custom `mx.fast.metal_kernel`.
- (ii) tile-level pruning inside HIER: skip tiles whose max score is below the running k-th threshold, if not already done.

Run at the measured deep shapes with real-activation dumps.

**Exactness gate:** identical index sets **and** identical tie-break versus the live path on 100% of real rows plus the synthetic-tie suite, using the same L1-L5 as 2.2. A divergence is a hard stop for that variant.

**Continue criterion:** >=1.5x on the select sub-op at 100K+, implying >=3% MODELED e2e at depth. Otherwise stop the lever.

### Lever B: lower-precision index-K (only if bandwidth-bound)

**This is not exact.** Token identity will not hold, so it needs a different gate class:
- top-k recall versus the reference >= a pre-registered threshold (propose 99.5% mean, 98% p1 per row; the user must approve);
- the quality battery;
- plus a long-context task battery.

The user must explicitly accept this gate class before any live round. Surface it as a question at M2; do not assume.

**First experiment** (offline, <=1 day):
- quantize the index-K cache to int8 with per-row (or per-group) scales, or fp8-e4m3 emulated;
- measure (i) the score-GEMM + K-read time at 100K/130K, (ii) top-k recall on real dumps across layers and depths, (iii) the memory saving (a bonus: a larger context budget).

**Continue criterion:** >=1.3x on the indexer at depth **and** recall meeting the proposed threshold on all tested layers.

### Phase 3 deliverables

`PIECE2_DECISION` doc (path per house-convention mapping) containing:
- the filled decision table with MEASURED inputs;
- the chosen lever, or "stop";
- first-experiment results;
- a MODELED e2e band;
- a proposed bounded plan for the 2-4 days, if continuing.

Push the branch and add a PERFORMANCE_HISTORY.md note.

**Phase 3 EV (MODELED):**
- 0% (most likely single outcome, since the profile may kill it) to +8-15% deep prefill if a lever delivers 1.5-2x on a ~27% share.
- P(some lever reaches a live round) ~ 35-45%.
- Lever B carries quality risk and approval friction.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| `nb` misdefined, so the gate is built on the wrong axis | Step 0.1 is a hard prerequisite; the 1B reproduction check is the second guard. |
| Offline benches on studio2 perturb live serving, or OOM it | Idle guard with mid-run polling and discard; predicted-peak memory cap at 50% of free; skip large reference-path configs. |
| Barrier instrumentation distorts the split (it kills MLX lazy fusion) | +-15% sum check; fall back to Metal capture; label BOUND. |
| Microbench is not representative of in-model behaviour | Real activation dumps; reproduce the known 1.71-2.05x and "neutral" points before trusting anything. |
| Float reduction-order differences flip near-ties (a subtler issue than exact ties) | 2.2 L1 bitwise check plus at-risk row counting; no waivers on token divergence. |
| Ranks take different paths | Gate on global quantities; equal engagement counts across ranks are a live gate. |
| Silent env no-op (`EXO_` prefix) | `DSV41_` names only; ps-eww gate; engagement counter must be >0. |
| Bypass path reintroduces memory blowup at a mis-set threshold | Independent byte ceiling, plus a peak-memory live gate. |
| Bypass EV is smaller than hoped (workload dominated by deep full chunks) | 1C occupancy gate (>=1.0% MODELED) before spending a boot. |
| Band imbalance at short L erodes the qsplit gain there | Measured in 1D/1E. Document only; a fix is out of scope unless large (>5%). |
| Lever B silently becomes "accepted non-exactness" | Explicit user approval of the gate class at M2; no live round without it. |
| Partial or stale state at session end | Every exit path leaves a pushed branch, an updated doc, a PERFORMANCE_HISTORY entry, and the live cluster on either the recorded baseline or a fully-gated new state. |

## Whole-workstream early stop

Stop everything, write up, and surface if any of these happen:
- `nb` cannot be pinned (step 0);
- the 1B reproduction check fails and Phase 2 is skipped, **and** the Phase 3 table says stop;
- GPU-busy < 80% (the problem is elsewhere, so redirect);
- I < 15% post-split **and** the 1C bypass EV < 1%;
- any live abort caused by instability unrelated to the change (cluster health first);
- the user's traffic makes a quiet window unavailable for more than 2 days after the M1 request.

## Budget

| Item | Budget |
|---|---|
| Phase 1 (offline) | <=1.5 days of idle-guarded studio2 time |
| Phase 1 live (1E) | ~60-90 min observation, **0 boots** |
| Phase 2 (offline) | <=1 day |
| Phase 2 live | 1 treatment boot + 1 reserve (infra failure only), plus a revert boot if not shipped (counting per the user's answer at M1); ~2-3 h window including baseline |
| Phase 3 | <=2 days offline first experiments; **0 boots** |
| **Total** | <= ~4.5 working days; **<=2 boots** plus revert |

## Surfaces to the user

- **M1:** Phase 1 offline results, Phase 2 offline results, and the single live-window request (1E observation + Phase 2 treatment) with budget and gates.
- **M2:** live outcome, the Phase 3 decision doc, and the Lever B gate-class question if relevant.
- **Otherwise:** surface only on a pre-registered abort.

---

## New-session handoff

You are executing this plan (committed as `docs/dsv41-indexer-abscost-plan-2026-10-12.md`, saved from Fable consult #5). If a plan copy also exists at `docs/perf/INDEXER_WORKSTREAM_PLAN.md`, this file is authoritative.

**Live state:** exo `deploy/next23b-qsplit` @ `60008939` + mlx-lm @ `6a5c78f4`, flags `EXO_DSV41_CKPT=1` and `DSV41_INDEXER_QSPLIT=1`. The launcher forwards only `DSV41_*` names; `EXO_DSV41_*` names are silent no-ops. HIER (the tiled indexer) has been default ON since 2026-10-06. The indexer query-band split is shipped (+10.5% / +11.9% prefill, 5/5 token identity).

**Read first:**
1. PERFORMANCE_HISTORY.md (last ~5 entries);
2. the qsplit buildnote/plan doc;
3. the HIER introduction doc (2026-10-06);
4. the deploy/boot runbook (quiet-cluster, ps-eww, quality-battery procedures);
5. the indexer forward and HIER code in mlx-lm @ `6a5c78f4`.

**Start with step 0.** Pin `nb`, "verify-ratio", and the instrumentation points; record them in the profile doc on a worktree branch `perf/indexer-profile`.

Then run Phase 1 offline, then Phase 2 offline, then surface **once** at M1 with the live-window request. No node-side edits. Push to the fork. Label every number MEASURED, DERIVED, BOUND, or MODELED. Update PERFORMANCE_HISTORY.md in the same session as any result. Never leave the cluster in an unrecorded state.
