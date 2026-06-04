# DSv4 Verify-Forward Tail-Variance Investigation -- Path from 30.06 to 32+ t/s

> **For Hermes:** This is an INVESTIGATION-FIRST plan. The first phase is pure
> instrumentation + code reading; we do NOT touch the cluster code until the
> tail mechanism is characterized. Per pitfall #47 ("STOP BENCHING, READ
> CODE"), all benchmarks here are SHORT (2-iter, <=15 min) and gated on
> instrumentation showing what we expect. Long 10-iter benches only at the
> validation stage AFTER a candidate fix is built.

**Goal:** Identify and (if possible) eliminate the verify-phase tail variance
that today's profile showed at FENCE=43: verify mean=57.10ms with min=51.74ms
max=70.40ms (35% spread, N=120). If the tail is a chained-collective CQE
arrival pattern (same family as the May-17 draft-side stall), clipping it
should yield +1-2 t/s without any compile-boundary or kernel work. Target:
mean cycle 62.65ms -> <=59ms -> >=31.5 t/s.

**Baseline anchor:** `baseline-2026-05-18-mtp-g2-topk512-30.06` (mean=30.062
t/s sigma=0.059 10/10 clean, mlx-lm `ac26733`, exo `9be275d5`).

**Tech stack:** exo + mlx-lm + mlx (Apple forks). All instrumentation lives
in mlx-lm (Python -- easy deploy) unless we explicitly need mlx-side
counters. Python changes deploy via `uv lock --upgrade-package mlx-lm` then
cluster relaunch. Always verify deployment via the grep-in-venv check from
Task 0 of the previous plan.

---

## Critical Context (Read First)

### Today's negative results (do NOT re-attempt without microbench)

The 2026-05-18 verify-forward plan tried two levers and both regressed:

1. **Lever 1 take 2** (L<=8 fused-SDPA L-into-batch fold in
   `_sparse_pooled_attention`): 30.06 -> 28.9 t/s. The
   already-`@mx.compile(shapeless=True)` inner kernel beat `mx.fast.sdpa`
   at B*L=3 because of per-call Metal launch overhead. Reverted via
   mlx-lm `b4d9e410`.

2. **Lever 2** (fuse `_raw_post_attn` + `_raw_ffn_pre` into
   `_raw_attn_to_ffn`, inline `_hc_expand_op`): 30.06 -> 7.2-10.5 t/s.
   Catastrophic. Capturing `mx.fast.metal_kernel` (inside
   `HyperConnection._hc_kernel`) within another `mx.compile` boundary
   appears to trigger pathological behavior. Reverted via mlx-lm
   `ac267339`.

DO NOT REVISIT either of these without a single-layer microbench that
proves the in-context savings at gamma=2 verify shape first.

### FENCE=8 vs FENCE=43 (today's probe)

At FENCE=8 (script default for gamma=2): 29.5-29.6 t/s scored, profile
total 61.59ms (1.7% better than FENCE=43's 62.65ms but agg_tps lower).
FENCE=8 is NOT a free win on current HEAD. The May-17 champion at FENCE=8
claiming 31.5 t/s is not reproducible here -- that's a separate bisect
investigation (Phase D below, optional).

### Profile breakdown today (FENCE=43, N=120 cycles)

```
draft      4.54 ms (7.2%)   min=4.40, max=5.07,  range=0.67 ms (15%)
verify    57.10 ms (91.1%)  min=51.74, max=70.40 range=18.66 ms (35%) <- TARGET
accept     0.81 ms (1.3%)   min=0.64, max=2.57,  range=1.93 ms
rollback   0.20 ms (0.3%)   min=0.14, max=0.35
total     62.65 ms

agg_tps                    : ~30.06 t/s
verify mean / cycle ratio  : 91.1%   <- unchanged from prior plan
mean accepted              : 1.04/2 (alpha2 ~ 0.52)
```

The verify range (18.66 ms = ~33% of mean) is the largest absolute
variance in the cycle. Draft has 15% range but only 4.54 ms mean.
Verify is where the wall lives.


### The hypothesis we're testing

**H1: The verify-phase tail (52->70 ms) is the same chained-collective
peer-CQE arrival pattern that bit `draft_tokens` on May 17.** That fix
was the per-step `mx.eval(tok_arr)` fence between chained `predict()`
calls (mtp_module.py:654, commit `ce61e46b`). The mechanism described
in the comment at lines 612-622: gamma chained predicts queue up gamma
lazy `all_sum`s in the GPU/comm-stream command buffer; each subsequent
`all_sum` is gated on the previous one's CQE delivery; peer-CQE arrival
tail accumulates with chain depth.

The verify pass is ONE forward over 86 layers. Each layer's MoE has an
`all_sum` (deepseek_v4.py:1071). With `EXO_DSV4_FENCE_EVERY_N_LAYERS=43`,
only ~2 layers per forward get `mx.eval(y)`. That's an 86-deep
`all_sum`-chain partitioned into segments of length 43. **If each
segment has its own peer-CQE arrival tail, we'd expect the verify tail
to be correlated with the maximum segment latency, not just the mean.**

Falsifiable predictions:
- (a) Per-layer all_sum latency, plotted across the 86 layers, should
  show clusters of low-latency layers and tail-prone layers, with tails
  correlating to fence boundaries.
- (b) The 70 ms verify outliers should correlate with a specific
  layer-range exhibiting a high-tail (>1 ms over-mean) cycle.
- (c) Adding fences at every layer (FENCE=1, equivalent to the May-9
  Phase H Lever 1 behavior) should ELIMINATE the tail but worsen the
  mean (more sync overhead). The trade can then be quantified.

**Alternative hypotheses (must also be ruled out):**
- H2: Thermal -- Mac Studios throttling under sustained load.
  Falsifiable: tail should grow as the test progresses if thermal;
  check iter-over-iter variance in a 10-iter run.
- H3: Allocator-driven -- MLX `mx.compile` cache thrash at shape
  transitions. Falsifiable: verify shape is (1, 3) consistently; no
  shape transitions expected once warmed.
- H4: Cross-rank coordination -- one rank periodically slow due to a
  background macOS process. Falsifiable: per-rank latency should
  diverge during tails, not stay matched.
- H5: The expected mean IS 51.74 ms (today's verify min), and the
  RAISED-FROM-FLOOR pattern (most cycles 55-60ms) is the actual
  problem, not the 70ms tail. Falsifiable: build a verify-latency
  histogram; if multimodal with low-mode at 52ms and high-mode at
  57ms, that's a bimodal stall, not a tail.

Each phase below tests a subset of these.

---

## Architecture / Code Surface

Key files (read-only unless noted):

- `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:1308-1334` -- where
  verify is timed (`t_after_draft -> t_after_verify` brackets the entire
  `dsv4_speculative_forward` call).
- `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:75-145` -- the
  `_phase_timer` / `_PROFILE_INTERVAL` machinery. `EXO_DSV4_MTP_PROFILE=N`
  dumps every N cycles.
- `mlx-lm/mlx_lm/models/deepseek_v4.py:1071-1090` -- the per-layer
  `all_sum` + fence decision. Lines 1083-1086 are the
  `EXO_DSV4_FENCE_EVERY_N_LAYERS` gate; line 1087 `mx.eval(y)` is the
  fence point.
- `mlx-lm/mlx_lm/models/deepseek_v4.py:2274+` -- `DeepseekV4Model.__call__`
  the top-level forward. The mask + pipeline send/recv happen here.
- `mlx-lm/mlx_lm/models/deepseek_v4.py:1051+` -- `DeepseekV4MoE.__call__`,
  where each layer's `all_sum` lives.
- `src/exo/worker/engines/mlx/speculative/mtp_module.py:580-660` --
  reference for the May-17 draft-side fix; the comment block at 612-622
  is the canonical explanation of the chained-collective tail mechanism.

We may need to add (and later remove) lightweight per-layer probes to
MoE (or to the verify call site). All probe code is env-gated so
production runs are zero-cost.


---

## Phase A: Inventory Existing Probes (no code change, ~30 min)

**Objective:** Confirm whether existing probes already give us per-layer
all_sum latency. If yes, skip Phase B's instrumentation work. Pure code
reading -- no edits, no commit, no push.

### Step 1: Inventory

```
grep -n "mx.eval\|finalize\|perf_counter\|span(" mlx-lm/mlx_lm/models/deepseek_v4.py | head -40
grep -n "MLX_BUILD_PROBE\|MLX_OP_PROBE\|EXO_DSV4_MTP_PROFILE\|EXO_PROFILER" mlx-lm/mlx_lm/models/deepseek_v4.py
ls src/exo/worker/profiler*.py 2>/dev/null
```

Look for:
- A probe that fires per-layer (not per-section) and reports CPU-wall
  around the `mx.eval(y)` fence point.
- A probe that's zero-cost when its env var is unset.
- Documentation of what `MLX_BUILD_PROBE` captures vs what we need.

### Step 2: Decision

If we already have a probe at the layer/segment granularity, skip to
Phase B Step 5 (use it directly).

Expected outcome: `MLX_BUILD_PROBE` tracks CPU-side build-time per layer
SECTION (attn_pre, attn, post_attn, ffn_pre, ffn, post_ffn) but NOT
GPU-side collective wait time -- which is exactly what we need.
`EXO_PROFILER=spans` reports the right granularity but pitfall #17
makes it a perf-killer. So expect to proceed to Phase B.

### Phase A success criterion

Decision documented (use-existing OR build-new) + the existing-probe
output sample (if applicable) saved to `/tmp/allsum_inventory.txt`.

---

## Phase B: Add Per-Layer All_Sum Latency Probe (instrumentation, 2-3 hrs)

**Objective:** Add a per-segment all_sum CPU-wall probe gated on a new
env var `EXO_DSV4_ALLSUM_PROBE=1`. Dump per-segment latencies every N
cycles. Use this to characterize the tail.

The probe MUST be zero-cost when off (no env-var read in the hot path,
no extra `perf_counter()` calls). Pitfall #17 "profiler-hook-trap" is
relevant -- we must NOT add `mx.eval`s on the timing path because that
itself synchronizes and masks the very thing we're measuring.

### Step 1: Locate the fence site

`mlx-lm/mlx_lm/models/deepseek_v4.py:1071-1088`, inside
`DeepseekV4MoE.__call__`. The fence-gate uses `mx.eval(y)` which IS the
sync point we want to time.

### Step 2: Patch outline

At module scope (near existing env-gated probes):

```python
import os as _ap_os
_ALLSUM_PROBE_ENABLED = bool(_ap_os.environ.get("EXO_DSV4_ALLSUM_PROBE"))
_ALLSUM_PROBE_LOG_EVERY = int(
    _ap_os.environ.get("EXO_DSV4_ALLSUM_PROBE_LOG_EVERY", "50")
)
_ALLSUM_PROBE_ACC = {}    # layer_idx -> list[ms]
_ALLSUM_PROBE_CYCLES = 0
```

In `DeepseekV4MoE.__call__`, around the existing fence-gate:

```python
y = mx.distributed.all_sum(y, group=self.sharding_group)
_is_last = self.layer_idx == self._num_total_layers - 1
_is_fence_idx = (self.layer_idx % self._fence_every_n) == (self._fence_every_n - 1)
if _is_last or _is_fence_idx:
    if _ALLSUM_PROBE_ENABLED:
        import time as _ap_t
        _t0 = _ap_t.perf_counter()
        mx.eval(y)
        _ms = (_ap_t.perf_counter() - _t0) * 1000.0
        _ALLSUM_PROBE_ACC.setdefault(self.layer_idx, []).append(_ms)
        if _is_last:
            global _ALLSUM_PROBE_CYCLES
            _ALLSUM_PROBE_CYCLES += 1
            if _ALLSUM_PROBE_CYCLES % _ALLSUM_PROBE_LOG_EVERY == 0:
                # Compute and log per-layer p50/p99/max from the last N cycles.
                # Then reset _ALLSUM_PROBE_ACC for the next window.
                ...
    else:
        mx.eval(y)
```

Key design points:
- The probe captures CPU-wall around `mx.eval(y)` ONLY when a fence is
  taken. Non-fence layers stay 100% lazy.
- At FENCE=43 we get per-segment data ("how slow is the 43-layer
  segment ending at this fence?"). At FENCE=1 we get per-layer data.
  We'll run BOTH to triangulate.
- Probe write path is Python-only overhead (~us); doesn't perturb
  `mx.eval` itself.

### Step 3: Deploy + verify (Task 0 protocol)

```
cd mlx-lm
python3 -c "import py_compile; py_compile.compile('mlx_lm/models/deepseek_v4.py', doraise=True)"
git add mlx_lm/models/deepseek_v4.py
git commit -m "perf-probe(dsv4): env-gated per-segment all_sum latency probe"
git push origin main
cd ..
uv lock --upgrade-package mlx-lm
git add mlx-lm uv.lock
git commit -m "deps: bump mlx-lm for allsum probe"
git push origin main

# teardown + relaunch with:
EXO_DSV4_FENCE_EVERY_N_LAYERS=43 \
  EXO_DSV4_ALLSUM_PROBE=1 \
  EXO_DSV4_ALLSUM_PROBE_LOG_EVERY=20 \
  EXO_DSV4_MTP_PROFILE=20 \
  EXO_DSV4_MTP_LOG=1 EXO_DSV4_MTP_LOG_INTERVAL=50 \
  ./start_cluster.sh

# verify on BOTH nodes:
grep -c "_ALLSUM_PROBE_ENABLED" ~/repos/exo/.venv/lib/python3.13/site-packages/mlx_lm/models/deepseek_v4.py
# inference probe -> "4"
# small-prefill 2-iter bench -> 0 errors
```

### Step 4: 100K 2-iter diagnostic bench at FENCE=43

`--iterations 2 --warmup 1 --prompt-words 75000 --max-tokens 128 --timeout 3600`.
Expected wall ~13 min. Capture both bench JSON and the per-segment dumps.

### Step 5: 100K 2-iter diagnostic bench at FENCE=1

Same as Step 4 but with `EXO_DSV4_FENCE_EVERY_N_LAYERS=1`. DIAGNOSTIC
only -- expect mean t/s to drop. Per-layer latency data is interpretable
as per-layer cost.

### Step 6: Analyze

Build histograms of per-segment (FENCE=43) and per-layer (FENCE=1)
latencies. Look for:
- **Bimodal distribution** (two peaks: low fast-path mean and high
  tail-path mean) -> strongest H1 evidence.
- **Tail layers correlated with specific layer indices** -> points at
  a topology or routing-rank issue.
- **Tail latency growing across cycles** -> points at thermal (H2).
- **Tail layers DIFFERENT between fence boundaries and non-fence
  layers** -> useful for designing the fence-placement fix.

Write findings to `.hermes/plans/2026-05-18_allsum_tail_findings.md`.

### Phase B success criterion

Clear, falsifiable answer to: "where in the verify forward does the
18ms tail come from, and is it chained-collective in nature?"

### Phase B abort criterion

If the probe data can't identify the tail mechanism, go to Phase D
(orthogonal investigations) instead of attempting a fix that might
regress like Levers 1 and 2.


---

## Phase C: Targeted Fix (only if Phase B implicates chained-collective tail)

**Objective:** Apply the smallest possible code change that the data
shows should clip the tail without regressing the mean. NO fix without
data showing it should work.

### Possible fix shapes (decided AFTER Phase B data)

**C1: Adaptive fence frequency at tail-prone layers only.** If data
shows specific layer ranges produce the tail, add fences ONLY at those
ranges (e.g., layer 60-70) instead of uniform FENCE=43. New env:
`EXO_DSV4_FENCE_AT_LAYERS=60,65,70,85`. Default unchanged.

**C2: Mid-segment async-eval drain.** If the tail is buffer-buildup
within a 43-layer segment, insert a partial-drain
`mx.async_eval(some_intermediate)` at layer N/2 to peel off the queue
without blocking forward progress. Mirrors the draft-side
`mx.eval(tok_arr)` fix at lower frequency.

**C3: First-layer-of-segment eager-eval.** Some chained-collective bugs
show the FIRST collective in a chain as the slow one. If data shows
this, add a single `mx.eval` on the first all_sum result of each
segment to seed the pipeline.

**C4: Cross-rank wakeup ping.** If H4 (one rank periodically slow) is
implicated, send a tiny `mx.distributed.all_sum(mx.zeros(1))` ping
before the segment start to ensure both ranks are awake. Cheap (~us).

### Step 1: Choose variant based on Phase B data

Document the choice + the data row that supports it. Justify why this
variant is expected to clip the tail without growing the mean (the
tradeoff Levers 1/2 lost on today).

### Step 2: Microbench BEFORE cluster bench

Build `bench/allsum_chain_microbench.py` (new file) that runs
`mx.distributed.all_sum + mx.eval` in a chain matching the verify shape
(86 sequential all_sums on (B=1, L=3, hc_mult, D) tensors with the
chosen fence/drain pattern) and measures CPU-wall latency. If
microbench shows clear benefit, proceed. If not, do NOT proceed.

The microbench gate is **non-negotiable** -- it's the rule we missed on
Levers 1 and 2 today.

### Step 3: Deploy + Task 0 protocol

Same as Phase B step 3.

### Step 4: 2-iter diagnostic 100K bench

If both iters land within +/-0.5 t/s of baseline AND profile shows
verify max <65ms (was 70ms), proceed to Step 5. If either iter shows
regression OR profile tail unchanged, REVERT immediately (2 iters =
answer known, per pitfall #41).

### Step 5: 10-iter validation 100K bench

- 10 scored iters, all >= 30 t/s (note: >=30, not >=29 -- we're trying
  to BEAT baseline, not just not-regress).
- sigma < 0.3 (tighter than baseline's sigma=0.06; we want a real
  improvement, not variance compression).
- Mean > 30.5 t/s (a meaningful lift, not noise).
- Verify mean <56ms AND verify max <65ms in profile data.
- 0 errors.

### Phase C success criterion

10/10 clean >=30.5 t/s mean, verify max clipped per profile. Tag
`champion-2026-05-19-mtp-g2-topk512-XX.X` and push.

### Phase C failure / abort response

Revert. Document why the candidate failed (mean shifted? variance
unchanged? side effect?). Data from Phase B is preserved for future
attempts -- write a `_followup.md` in the plans dir.

---

## Phase D: Orthogonal Investigations (parallel, low-priority)

These can run in any order, independently of Phases A-C. Each is a
small standalone item.

### D1: 9 ms/cycle profile-vs-wall gap

Today's data: profile-sum = 62.65ms; agg_tps-implied per-cycle wall =
68.9 ms. Gap = ~9 ms/cycle = 13% of wall. Potentially recoverable.

**Hypothesis sources:** `finalize()` overhead at top-level forward,
`span()` overhead (env-gated but maybe leaking), per-cycle Python work
in `MTPBatchGenerator._speculative_next` not inside `prof.record`
brackets.

**Action:** Add a single `t_cycle_start_total` timer at the TOP of
`_speculative_next` (before draft) and `t_cycle_end_total` at the END
(after yield), bracketing accept/rollback/yield/etc. Dump the
difference vs the `total` profile bucket; the delta is the "untracked
overhead" we're hunting. Single 2-iter bench, 30 min.

If the gap is real (>5ms/cycle untracked), make a follow-up plan.

### D2: L_q==1 fast path A/B

The May-13 L_q==1 fused-SDPA fast path (commit `0bdaab8b`) was landed
on a microbench showing 1.24x, but today's Lever 1 take 2 results
suggest that microbench may have been misleading. Does the L_q==1 fast
path itself regress in-context vs the inner kernel?

**Action:** A/B-test via existing `EXO_DSV4_TOPK_FUSED` env (or add
`EXO_DSV4_SPARSE_LFAST_DISABLE` if needed). Two 2-iter benches with
L_q==1 fast path OFF vs ON. If OFF is faster, revert the May-13 patch.

### D3: May-17 champion bisect

Tag `champion-2026-05-17-mtp-g2-fenced-31.5` is reported at 31.5 t/s at
FENCE=8 gamma=2. Today at FENCE=8 we got 29.5-29.6 t/s. Somewhere in
the ~2 days of commits since that tag, a +1.5 t/s regression landed.

**Action:** standard `git bisect` between the tag and current HEAD, at
each step running a single 3-iter 100K bench at FENCE=8. ~15 commits,
~10 min each = 2-3 hours bisect work. Tag whatever commit introduces
the regression. The fix may be free (one-commit revert) or the
regression may be a feature we wanted; either way, knowing is better
than guessing.

### D4: Multi-acceptance MoE acceleration

Today: mean_accept=1.04/2 -> alpha2 ~ 0.52. MoE per-cycle work is
fixed; only way to raise tokens-per-cycle is raise alpha. Can we force
gamma=3 even though the model's `num_nextn_predict_layers=1`
historically capped at gamma=2? What's alpha3? If alpha3 > 0.4,
gamma=3 wins net.

**Action:** A 2-iter bench with `EXO_SPECULATIVE_GAMMA=3` and observe.
Plan-side this only if Phase C fails.


---

## Hard Constraints (Do NOT)

- **DO NOT** retry Lever 1 take 2 or Lever 2 from today's plan without
  an isolated single-layer microbench at gamma=2 verify shape that
  PROVES the in-context savings BEFORE touching cluster code.
- **DO NOT** modify the mlx submodule (C++) without going through the
  per-bench Task 0 deployment protocol AND understanding the mid-May
  4.3 t/s regression mechanism first.
- **DO NOT** remove the `mtp_module.py:654` per-step fence in
  `draft_tokens` -- confirmed twice to cause gamma=2 iter-1 stall.
- **DO NOT** set `EXO_PROFILER=spans` for steady-state perf
  measurement -- pitfall #17, perf-killer.
- **DO NOT** lower `EXO_DSV4_INDEX_TOPK` below 512 without an explicit
  quality regression budget.
- **DO NOT** claim a champion without >=10 iters at the production
  config, all >=30 t/s, sigma<0.3, 0 errors.
- **DO NOT** chain `sleep 280; check; sleep 280` patterns in the bench
  monitor; use active state-change polling.

---

## Files Likely To Change

- `mlx-lm/mlx_lm/models/deepseek_v4.py` -- Phase B probe insertion
  (env-gated, ~30 lines), Phase C fix (mechanism-dependent, ~10-50
  lines).
- `bench/allsum_chain_microbench.py` -- NEW file, Phase C step 2
  microbench to validate the fix mechanism in isolation.
- `uv.lock` -- auto-regenerated each mlx-lm bump.
- `mlx-lm` submodule pointer in exo -- auto-bumped.

**Do NOT modify:**
- `mlx/` submodule (C++).
- `start_cluster.sh` env defaults (FENCE=8 default stays -- we pass
  FENCE=43 on launch for THIS investigation since it matches today's
  baseline; the script default is correct for production).
- `src/exo/worker/engines/mlx/speculative/*.py` (except read-only
  reference reads).

---

## Tests / Validation

### Per-bench validation (Task 0 protocol from the previous plan)

For every mlx-lm change:
1. Local `py_compile` clean.
2. Deployment grep on both nodes returns >=1 hit on the patch's unique
   string (path:
   `~/repos/exo/.venv/lib/python3.13/site-packages/mlx_lm/models/deepseek_v4.py`).
3. Inference probe at small context returns sensible output (`"4"`).
4. Small-prefill 2-iter bench completes with 0 errors.
5. THEN 100K bench (2-iter diagnostic OR 10-iter validation per phase).

### Quality gate (Phase C validation only)

```
ssh ... "cd ~/repos/exo && uv run python3 bench/quality_probe_dsv4.py \
  --host localhost --port 52415 \
  --model mlx-community/DeepSeek-V4-Flash-8bit \
  --target-tokens 100000 --output ~/quality_phase_c.json"
```

Quality must hold: needle-found = True, response coherent.

### Statistics required for "champion" claim

>=10 scored iters, all >=30 t/s (>=0 above baseline mean 30.06),
sigma<0.3, 0 errors, profile verify max <65ms. Tag with the exact mean
value rounded to 1 decimal: `champion-2026-05-19-mtp-g2-topk512-XX.X`.

---

## Risks, Tradeoffs, Open Questions

### Risks

1. **Probe overhead could mask the signal.** The
   `mx.eval`-with-timing pattern is what we use to FENCE -- adding a
   `perf_counter` around it adds <1 us and should be invisible. If it
   isn't, the data is useless. **Mitigation:** validate by comparing
   probe-on profile numbers to probe-off (today's) baseline numbers; if
   mean cycle shifts >0.5 ms, the probe is perturbing and we need a
   different measurement technique (e.g., MLX-internal counter).

2. **Phase B may produce a clear mechanism but no clean fix.** The
   tail might be a fundamental cost (Thunderbolt 5 RDMA CQE
   notification latency floor). **Mitigation:** Phase B success
   doesn't require a fix -- it requires a CHARACTERIZATION. If no fix
   is feasible, document and close.

3. **Phase C variant choice might be wrong even with good Phase B
   data.** Today's session showed "this should work" ideas can regress
   catastrophically. **Mitigation:** microbench gate in Phase C Step 2
   is non-negotiable. If microbench doesn't show benefit, do NOT
   proceed to cluster bench.

4. **Bisect (Phase D3) could find the regression in a commit we don't
   want to revert.** **Mitigation:** stop on bisect mid-way if
   commit-message hints at this. Document the find without reverting.

### Tradeoffs

- **Probe-and-measure overhead.** Phase A+B together: ~30 min cluster +
  1-2 hours code reading + analysis. Worth it because today showed
  blind perf attempts cost just as much wall-time AND regress.
- **Phase B's per-layer probe is intentionally CPU-side timing.** It
  measures CPU wait on GPU/comm pipeline, not GPU intrinsic cost.
  Adequate for tail characterization; not for fine-grained kernel
  analysis.
- **No mlx C++ changes in this plan.** All investigation lives in
  Python so deploy + revert costs are low.

### Open Questions

1. Is there an existing per-layer profile that gives us the data
   without Phase B's instrumentation work? (Phase A answers.)
2. Does Mac Studio M4 Max's Thunderbolt-5-over-RDMA driver have its
   own latency-tail characterization we can pull from system logs?
3. If H1 is confirmed, is the cleanest fix architectural (drain queue
   more often) or mechanical (cross-rank wakeup ping)? Decide at
   Phase C step 1.

---

## Time Budget

| Phase | Description                                          | Wall (worst case) | Decision     |
|-------|------------------------------------------------------|-------------------|--------------|
| A     | Inventory existing probes                            | 30 min            | proceed/skip |
| B     | Add per-layer probe, deploy, 2 diagnostic benches    | 2-3 hours         | proceed/abort |
| C     | Microbench fix, deploy, 2-iter + 10-iter benches     | 3-4 hours         | champion/revert |
| D1-4  | Orthogonal investigations (parallel, each ~1 hr)     | 1-3 hours         | low priority  |

Total to "champion at 31+ t/s OR characterized non-fixable tail":
**6-10 hours.** Longer than today's plan estimated, but today's results
(4 hours, two reverts, no lift) suggest the investigation-first
discipline is worth the extra time.

---

## Quick Reference: Cluster Operations

### Launch cluster -- Phase B probe ON
```
cd /Users/adam.durham/repos/exo && \
  EXO_DSV4_FENCE_EVERY_N_LAYERS=43 \
  EXO_DSV4_ALLSUM_PROBE=1 \
  EXO_DSV4_ALLSUM_PROBE_LOG_EVERY=20 \
  EXO_DSV4_MTP_PROFILE=20 EXO_DSV4_MTP_LOG=1 EXO_DSV4_MTP_LOG_INTERVAL=50 \
  ./start_cluster.sh
```

### Launch cluster -- Phase B per-layer (FENCE=1) diagnostic
Same as above but `EXO_DSV4_FENCE_EVERY_N_LAYERS=1`.

### Launch cluster -- Phase C candidate fix
```
cd /Users/adam.durham/repos/exo && EXO_DSV4_FENCE_EVERY_N_LAYERS=43 ./start_cluster.sh
```
(probe OFF, fence at the production setting).

### Bench launch template (2-iter diagnostic)
```
SSH_OPTS="-i ~/.ssh/exo_cluster -o IdentitiesOnly=yes"
ssh $SSH_OPTS adam.durham@192.168.86.201 "
  screen -dmS bench_phase_<X> zsh -l -c '
cd ~/repos/exo
uv run python3 bench/concurrent_bench.py \
  --host localhost --port 52415 \
  --model mlx-community/DeepSeek-V4-Flash-8bit \
  --concurrency 1 --iterations 2 --warmup 1 --max-tokens 128 \
  --prompt-words 75000 --timeout 3600 \
  --json-out ~/bench_phase_<X>_\$(date +%s).json \
  --label phase_<X> \
  2>&1 | tee ~/bench_phase_<X>.log
'"
```

### Active iter poll (state-change only)
```
LAST=0
while true; do
  cnt=$(ssh ... "if ! pgrep -f concurrent_bench >/dev/null; then echo DEAD; exit; fi; grep -cE 'iter=' ~/bench_phase_<X>.log")
  [[ "$cnt" == "DEAD" ]] && break
  if (( cnt > LAST )); then
    LAST=$cnt
    ssh ... "grep -E 'iter=' ~/bench_phase_<X>.log | tail -1"
  fi
  sleep 30
done
```

### Probe-log analysis (Phase B step 6)
```
ssh ... "grep -E 'ALLSUM-PROBE' ~/exo.log | tail -500" > /tmp/allsum_dumps.txt
python3 bench/allsum_chain_microbench.py --analyze /tmp/allsum_dumps.txt
```

---

## Notify-the-User Discipline

Discord pings on (per pitfall #45):
- Phase A complete (existing probe sufficient OR Phase B starts)
- Phase B probe analysis ready (regardless of conclusion)
- Phase C candidate identified + microbench passes (before deploy)
- Phase C 10-iter bench done (champion or revert)

All command/payload stays in chat; Discord = ping only.

---

## Remember

- A 2-iter 100K bench is ~13 min. A 10-iter is ~70 min. Use 2-iter for
  decisions; 10-iter only for champion claims.
- Investigation-first: Phase A+B before any Phase C cluster bench.
- All probe code is env-gated AND zero-cost when off. Production code
  must remain unchanged in steady-state hot paths.
- Don't re-attempt the compile-boundary collapse pattern (Levers 1/2)
  without a single-layer microbench gate.
- Champion criteria are TIGHTER than the previous plan's baseline gate:
  >=30 t/s (not >=29), sigma<0.3 (not sigma<0.5), mean >30.5 (a real lift).
