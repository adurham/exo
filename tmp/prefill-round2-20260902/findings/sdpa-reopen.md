# sdpa-reopen — Reopening the '2.029x fully closed' SDPA attribution (READ-ONLY, 2026-09-02)

**Mode:** read-only analysis. Zero cluster contact, zero code changes, zero commits.
**Repo HEAD:** `cb1f91903` (main). Every number below carries a `file:line` or doc citation.
**Verdict in one line:** the 'FULLY closed' claim in `docs/dsv4-4096-regression-root-cause-2026-08-19.md` rests on a **per-call vs per-token units conflation**; the cluster's own sync AND non-sync data show **superlinear per-call SDPA scaling (~3.1–4.1x for a 2x row doubling)**, the isolated benchmark's 2.0x was per-CALL and therefore predicted a per-token ratio of ~1.0x, not 2.0x — and the depth degradation is a **different, largely already-explained phenomenon** (known O(P) terms account for ~86% of the measured slope).

---

## (a) What the original measurement ACTUALLY measured

Source: `docs/dsv4-4096-regression-root-cause-2026-08-19.md` (all four sections). The measurement was:

- **A/B on the live 2-node cluster** (not a microbench): `EXO_PREFILL_STEP_SIZE=2048` vs `4096`, `EXO_DSV4_SEQ_SPLIT=1` (standing default, `deepseek_v4.py:230`), same **12,068-token** matched prompt, third iteration run with `EXO_PROFILER=spans EXO_PROFILER_SYNC_SPANS=1` (the first two iterations were non-sync).
- **Units are ms/TOKEN** (span `total_ms` ÷ total prompt tokens — verified in round 1; the naive "half as many chunks cancels" objection does not apply, and a 2x per-token jump at fixed total tokens is a real 2x wall increase).
- SEQ_SPLIT halves each chunk's query rows across the 2 TP ranks (`deepseek_v4.py:4749-4753`, band slice), so per-rank query rows L_q: **1024 → 2048** when STEP_SIZE doubles. KV/pool sides stay FULL on both ranks by design (`deepseek_v4.py:4734-4738`).
- Measured: `attn.sdpa` 0.4153 → 0.8428 ms/token (**2.029x**), `attn.sdpa.compressed` 0.2745 → 0.6477 (**2.359x**).
- The "linear" reference it was matched against: `bench/sdpa_subtile_microbench.py`, which measured `ratio_2048_over_1024` = `single_call_2048_ms / single_call_1024_ms` at **fixed KV length** (`bench/sdpa_subtile_microbench.py:195-197`) ≈ **1.86–2.00x**. That is a **per-call** ratio.

**What the span actually contains** (important for interpretation): `attn.sdpa` fires 23x/chunk — 21 `SparseCompressedAttention` layers (tiled at `_SPARSE_SDPA_TILE=128`, `deepseek_v4.py:322`, single upfront gather `deepseek_v4.py:4953-4964`, per-tile `_sparse_pooled_attention_inner` which is `@mx.compile(shapeless=True)`, `deepseek_v4.py:1480`) **plus 2 `LocalAttention` layers** (compress_ratio=0, NO seq-split, full-L rows, sliding-window-only KV; code-map `docs/dsv4-sdpa-subtiling-code-map-2026-08-19.md:90-95`). `attn.sdpa.compressed` (20 layers) is one fused `mx.fast.scaled_dot_product_attention` over `[RotatingKVCache local ≤ sw+L | pooled P]` (`deepseek_v4.py:4433`, `:4452`; local cache is `RotatingKVCache(max_size=128)`, `deepseek_v4.py:7348-7352`) — so its KV is **bounded local + O(P) pooled**, per-token cost has a genuine O(P) term but the local term is depth-independent.

## (b) The specific crack in the 'fully closed' claim

**The claim:** "attn.sdpa's sync-mode ratio (2.029x) matches the isolated laptop microbenchmark's linear-scaling prediction (~2.0x) almost exactly… SDPA scales linearly… no hidden, unattributed SDPA-kernel-level cost" (`dsv4-4096-regression-root-cause-2026-08-19.md:171-188`).

**The crack — units conflation.** The two 2.0x numbers are in different units:

- The microbenchmark's ~2.0x is **per-call** (one 2048-row call vs one 1024-row call, KV fixed).
- The cluster's 2.029x is **per-token**. With SEQ_SPLIT, chunk count halves (6 chunks @2048 vs 3 @4096 for a 12,068-token prompt; remainder chunks in both arms), so calls halve while tokens-per-call double. The conversion is:

```
ms/token ratio = (n_calls_4096 × percall_2048row) / (n_calls_2048 × percall_1024row)
               = (n_calls_4096/n_calls_2048) × per-call ratio
               = 0.5 × per-call ratio
```

- Therefore **2.029x ms/token ⇒ per-call ratio ≈ 4.06x**. Arithmetic check from the doc's own raw numbers: 0.8428 × 12,068 = 10,171 ms over 63 calls = 161.4 ms/call, vs 0.4153 × 12,068 = 5,012 ms over 126 calls = 39.8 ms/call. Ratio **4.06x**. (Call-count assumption verified in code: one `attn.sdpa` span per sparse/local layer per chunk per rank; sparse-class sub-tiling happens INSIDE the span, so it does not multiply call count; 220K span profile call counts — 2530 `attn.sdpa` / 2200 `compressed` over ~110 chunks — confirm 23+20 per chunk.)
- **Linear per-call scaling predicts ms/token ratio ≈ 1.0x** — 2x per-call cost, half as many calls, same total tokens. The doc's "matches almost exactly" compares a per-token ratio to a per-call ratio and concludes the mechanism is confirmed. It is not.
- A code-accurate forward model at 12K (sparse: per-tile cost independent of L_q × L_q/128 tiles = linear; compressed: L_q × (~2175 local + P≈3,017 pooled) = linear-in-L_q with mild O(P); LocalAttention: full-L, KV=128 window, linear) predicts **~1.0–1.1x per-token** — nowhere near 2.029x.

**The original 3.15x "mystery" was never resolved — only relabeled.** The SECOND UPDATE's non-sync data independently shows avg per-call 95,245 µs @4096 vs 30,218 µs @2048 = **3.15x per-call** (directly reported, not derived), and non-sync ms/token 1.78x ⇒ per-call ≈ 3.56x. Both sync (≈4.06x derived) and non-sync (3.15x direct) agree the cluster per-call scaling is **~3.1–4.1x for a 2x row doubling — superlinear** — while the isolated fixed-KV bench says 2.0x. The THIRD UPDATE "solved" the 3.15x by attributing it to lazy-eval leakage and declaring the sync-mode 2.029x as confirmation of linearity — but 2.029x ms/token IS the same superlinear per-call signal, seen through a halved call count.

**Reductio / plausibility checks on the sync numbers themselves (method-discipline §1):** 10.17 s of `attn.sdpa` inside a ~30 s 12K sync-mode prefill ≈ 30% of wall vs 13.6% at 220K — elevated, a flag for sync-mode inflation, but not mathematically impossible, and the non-sync data shows the same signature, so the reductio does not kill the measurement.

**Sync-mode bias direction (method-discipline #3):** sync mode adds a per-CALL constant overhead (drain + `mx.synchronize()`), which **dilutes** the per-call ratio at larger L_q (constant overhead is a smaller fraction of a bigger call) — the bias runs *opposite* to the observed direction. Serialization at span boundaries destroys overlap and could inflate absolute per-call times, but it inflates both arms; and it cannot manufacture a 2x-per-call discrepancy vs the isolated bench that the non-sync run also shows. Sync mode does not rescue the 'linear' claim.

**Candidate mechanisms checked and ruled out (or bounded) for the 4.06x per-call:**
- Ragged last chunks (12,068 = 5×2048+1828 / 2×4096+3876): both arms carry one remainder chunk; ≲10–15% distortion, cannot turn 4.06 into 2.0.
- `mx.compile` recompiles: the sparse inner kernel is `shapeless=True` — one compiled kernel for all widths. Compressed path is not compiled but is a single fused kernel.
- LocalAttention (no seq-split) at 4x rows: per-call linear at fixed 128-window KV; ms/token ratio 1.0x; small span share anyway.
- Depth confound (the 4096 arm's chunks at marginally deeper mean depth): both arms span the same token range with evenly distributed chunk centers; the doc's own ~5% bound stands (`dsv4-4096-regression-root-cause-2026-08-19.md:98-103`).
- KV-length mismatch between arms: KV/pool full on both ranks; rotating local cache bounded at 128 + L − 1. Not it.

**Honest bottom line for (b):** the 2.029x number is real as a per-token doubling, but the conclusion drawn from it ("matches linear prediction, nothing to chase") is invalid. The cluster evidence says per-call SDPA cost on the live 2-node cluster scales ~quadratically-ish in per-rank query rows at 12K context, which the isolated single-node benchmark does not reproduce — i.e., there IS an unattributed, environment- or shape-dependent SDPA-level cost, exactly the thing the closure declared nonexistent. What it is (allocator pressure at 2x gathered-tensor footprint, Metal occupancy/tile-quantization at L_q=2048, cross-rank interaction, or residual measurement artifact in BOTH modes) is genuinely open — the reopen is warranted, and the per-call ratio (~3.1–4.1x) is the number to chase, not ms/token.

## (c) Could the 2.029x and the prefill-vs-context degradation be the SAME phenomenon?

**Short answer: no — the numbers separate them cleanly, with an 86%-explained account for the degradation.**

The degradation (round-1 item2, `tmp/prefill-round1-20260902/findings/item2-instrumentation.md:137-156`): per-chunk TOTAL wall 4777 → 4843 → 5000 ms at 89K/150K/250K; regression slope ≈ **0.00281 ms/chunk per token of depth = 5.8 ms/chunk per 2048-token step** (recomputed here: intercept 4644 ms, R²=0.992 — consistent with round 1's 4637/0.995 within rounding of the depth proxy).

Attribute the slope to the known O(P) terms using the 220K span profile (`docs/dsv4-220k-prefill-span-profile-2026-08-18.md:82-91`, ~5667 ms/chunk TOTAL):
- `attn.sdpa.compressed` 11.8% ≈ 669 ms/chunk. Its pooled-P part (fused SDPA over [~2175 local | P pooled]; local part depth-independent) at P≈55,000, assuming roughly half its cost is P-proportional ≈ 334 ms → **d/dstep = 334/55000 × 512 ≈ 3.1 ms/chunk/step**.
- `attn.indexer` 4.0% ≈ 227 ms/chunk; score GEMM `(B,L,D)@(B,D,P)` (`deepseek_v4.py:3849`) is almost entirely P-proportional ≈ 193 ms → **≈ 1.8 ms/chunk/step**.
- `attn.sdpa` (sparse) gathers fixed top-k=512 per row (`index_topk=512`, `deepseek_v4.py:870`) → **≈ 0 ms/chunk/step**. Collectives: payload ∝ L×H×D per chunk, depth-independent → ≈ 0.

**Sum ≈ 4.9–5.0 of the measured 5.8 ms/chunk/step (~86%).** The residual ~0.9 ms/chunk/step is within the noise of the three-point fit (n=3, single run per depth) plus second-order terms (pool visibility growth, argpartition cost growing weakly with P). The depth degradation is, to first order, **already explained by known O(P) attention terms** — no exotic mechanism required.

Now test whether that same O(P) mechanism could also have produced the 2.029x at 12K: **it cannot.**
- An O(L_q·P) term: per-token ratio across the A/B = P'/P ≈ 1.0 (+ ≤5% depth confound). Predicts ~1.0x, not 2.029x.
- Quantitatively at 12K: P ≈ 12,068/4 ≈ 3,017 pooled rows. The compressed P-part per token scales from the 220K figure as 0.163 ms/token × (3017/55000) ≈ **0.009 ms/token ≈ 2.2% of the measured 0.4153 ms/token**. Even if the entire O(P) cost had doubled between arms, it could move the ratio by ~2%, not 102%.
- Conversely, the 2.029x mechanism (whatever it is — superlinear in L_q) would, if it scaled with P too, have to add hundreds of ms/chunk at 250K (doubling L_q at 12K "explains" ~875 ms/chunk via the sdpa spans' Δms/token × 2048); the ENTIRE measured 89K→250K per-chunk growth is 223 ms. So the 4096-step-size regression mechanism and the depth slope cannot share one dominant term.

**Verdict: two different phenomena.** (1) The depth degradation = the designed-in O(P) prefix-attention + indexer-scoring terms, mostly expected behavior, capped at ~406→427 tok/s spread by the 3-point fit. (2) The 12K A/B anomaly = a per-call superlinearity in the SDPA spans at doubled per-rank rows, mechanism unknown, unexplained by O(P), unexplained by sync mode, contradicted by the isolated fixed-KV microbenchmark. The 'fully closed' doc conflated (2)'s units and declared (1)'s territory clean in the same stroke.

## (d) Pre-registered re-measurement design at 250K

**Question:** is per-call SDPA cost on the live cluster linear in per-rank query rows at deep context, and does the same anomaly exist at 250K that showed at 12K?

**Primary comparison arm (cleanest probe): `EXO_DSV4_SEQ_SPLIT=0` at STEP_SIZE=2048.** This doubles per-rank query rows (1024→2048 effective, since SEQ_SPLIT=0 runs attention fully replicated: `deepseek_v4.py:230`, gate at `:4740-4745`) at **identical chunk count, identical chunk boundaries, identical depths** — removing the depth-alignment confound entirely. The STEP_SIZE=4096 arm is retained as the secondary arm for continuity with the 12K result, at the cost of a coarser depth grid.

**Instrumentation (exact):**
- Model-internal spans: `EXO_PROFILER=spans EXO_PROFILER_LEVEL=1` (launcher env, allow-listed `start_cluster.sh:1660`).
- **Sync on:** `EXO_PROFILER_SYNC_SPANS=1` (read at `mlx-lm/mlx_lm/profiler.py:215`; `mx.synchronize()` at both span boundaries, `profiler.py:221-232`).
- **Watchdog raised — MANDATORY pairing:** `EXO_RUNNER_HANG_TIMEOUT_SECONDS=600` (default 45 s, `src/exo/worker/runner/supervisor.py:81`; allow-listed `start_cluster.sh:1592`; 300 s was sufficient for the 12K run, 600 s gives margin at 250K where per-chunk walls are 5+ s and sync-mode multiplies the progress-callback gap). Without this the runner gets SIGKILLed mid-request — the exact failure documented at `dsv4-4096-regression-root-cause-2026-08-19.md:150-161`.
- **Batched production path only:** `prefill_batched` (`src/exo/worker/engines/mlx/generator/generate.py:1269`, called at `batch_generate.py:3068`). Do NOT accept numbers from the eager `stream_generate` fallback — the paths differ materially (per-chunk barrier/eval_cache/clear_cache structure at `generate.py:1496-1527` exists only on the batched path). Verify the run used it by requiring the `[MEM] batched prefill chunk` log lines (`generate.py:1537-1540`) and `Starting batched prefill:` (`generate.py:1400`) in both nodes' logs.
- Capture: `attn.sdpa`, `attn.sdpa.compressed`, `attn.indexer`, `attn.gather`, `moe.switch_mlp`, `moe.all_sum`, `attn.all_gather` + the auto-dump at prefill completion (no SIGUSR1 — per `dsv4-4096-regression-root-cause-2026-08-19.md:213-220`; the mistimed SIGUSR1 crashed a rank once already).
- **Both per-call AND per-token views recorded and compared** (the round-1 lesson): per-call = span `total_ms` ÷ span call count; per-token = `total_ms` ÷ 250,019. The analysis is per-call-first; ms/token is the derived, secondary view.
- Prompt: fresh random word-salad + embedded secret needle, new seed per run; verify `usage.cached_tokens: 0` and a single clean `Prefill progress: 0/` start per node inside the window (methodology per `dsv4-220k-prefill-span-profile-2026-08-18.md:29-53`).

**Arms (all at 250,019-token fresh prompts, standing baseline otherwise: `MLX_JACCL_DATA_RECV_POOL=0`, `EXO_DSV4_SEQ_SPLIT_MIN_L` untouched):**

| arm | STEP_SIZE | SEQ_SPLIT | purpose | chunks | per-rank rows |
|---|---|---|---|---|---|
| A (control) | 2048 | 1 | standing config baseline | 123 | 1024 |
| B (primary probe) | 2048 | **0** | same chunks/depths, rows 1024→2048 | 123 | 2048 |
| C (continuity) | 4096 | 1 | replicates the 12K A/B geometry at 250K | 62 | 2048 |

**Pre-registered outcome bands (fixed BEFORE any run):**

Measured quantity: per-call ratio R = per-call@2048rows ÷ per-call@1024rows for `attn.sdpa` (and separately `attn.sdpa.compressed`), on the sparse-branch steady-state calls only (exclude each run's first 4 chunks; both arms' early chunks take the dense-concat branch below topk).

1. **REFUTES same-phenomenon AND confirms per-call linearity at depth:** R ∈ [1.8, 2.2] for both spans (isolated-bench prediction 1.86–2.00). Then the 12K anomaly was shallow-context-specific (e.g. small-prompt fixed-cost contamination across 3–6 chunks) and the 2.029x closure is wrong in reasoning but right in conclusion; depth degradation stays attributed to the O(P) terms per §c.
2. **CONFIRMS the 12K anomaly persists at depth (superlinear per-call):** R ≥ 2.7 for `attn.sdpa` in the B-vs-A comparison (i.e., ≥35% above linear, outside the ±15% ragged-chunk distortion bound). CONFIRMS-same-phenomenon requires additionally: R growing with depth (compare R at arm A vs C, and the within-run per-chunk-call trend at 250K vs the same statistic at a 12K control run), AND the depth-slope of the sdpa spans alone exceeding ~1.5 ms/chunk/step — i.e. the L_q-superlinearity and the P-linear terms are entangled. If R ≥ 2.7 but depth-independent (same R at 12K and 250K), it is the SAME phenomenon as the 12K anomaly but DISTINCT from the depth degradation — report it as an isolated kernel-shape effect worth ~1.3–1.6% of prefill wall at standing config, not a depth story.
3. **INDETERMINATE:** 2.2 < R < 2.7, or arm B's quality gate fails (needle answered incorrectly → discard, rerun once; two failures → stop, report measurement invalid), or either run trips the watchdog despite 600 s.
4. **Guard rails:** arm B (SEQ_SPLIT=0) is expected to be ~7–10% SLOWER end-to-end (June 2026 measured +18-19% for enabling it) — that is fine; the deliverable is R, not tok/s. Reductio before quoting any per-call figure: calls × per-call ≤ measured total wall for that span category, and span total ≤ chunk wall (5000 ms at 250K standing config).

**Honest cluster-time estimate:** 3 arms × 1 run at 250K ≈ 615 s each clean; +10–15% profiler tax; sync mode adds substantial serialization (the 12K sync run's attn.sdpa alone absorbed ~30% of wall) — budget **~15 min/run**. Plus 3 relaunches (~5 min each) and one optional 12K control re-run for the depth-dependence check (+~10 min incl. relaunch). **Total ≈ 75–90 minutes of cluster time**, one operator, single session. Repeat each arm only on quality-gate failure (band 3).

---

## Sources
- `docs/dsv4-4096-regression-root-cause-2026-08-19.md` (all four sections — the claim under review)
- `bench/sdpa_subtile_microbench.py:181-199` (per-call ratio definition), `docs/dsv4-sdpa-subtiling-code-map-2026-08-19.md` (attention classes, seq-split gates, LocalAttention no-split)
- `mlx-lm/mlx_lm/models/deepseek_v4.py`: `:230-231` (SEQ_SPLIT default), `:322` (tile 128), `:1480` (shapeless compile), `:3849` (indexer GEMM), `:4433/:4452` (compressed KV concat), `:4749-4753` (band), `:4865-5017` (sdpa spans), `:7336-7352` (RotatingKVCache 128 / cache classes)
- `docs/dsv4-attention-kernel-efficiency-2026-08-18.md` (production shapes, 61.7%/79.1% ceilings, KV=2175 local bound)
- `docs/dsv4-220k-prefill-span-profile-2026-08-18.md` (span shares at depth)
- `tmp/prefill-round1-20260902/findings/item2-instrumentation.md` (bench_*.json per-chunk walls, 4637/0.00288 regression, TP full-chunk confirmation)
- `docs/PERFORMANCE_HISTORY.md:7740-7754` (round-1 flag on the 2.029x crack), `:895-905` (191K retest context)
- `mlx-lm/mlx_lm/profiler.py:197-236` (sync spans), `src/exo/worker/runner/supervisor.py:65-81` (watchdog), `start_cluster.sh:88,1660-1661,1589-1592` (env allow-list)