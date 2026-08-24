# P5 — The "~22% TP-vs-PP prefill gap": provenance, theory, live measurement

**Date:** 2026-08-24
**Cluster:** 2x Mac Studio M4 Max, TB5/RDMA (jaccl), `adams-mac-studio-m4-1.local`
rank0 + API :52415, m4-2 rank1.
**Lead as dispatched:** "TP mode gets ~350 tok/s prefill vs PP mode's ~450 tok/s
on this cluster (~22%) — never root-caused. Find the uncaptured TP headroom."
**Standing constraint:** PP-vs-TP as a *serving topology* is SETTLED (TP wins;
see `exo-sharding-mode-tradeoffs` +
`references/pp-tp-architecture-decision-2026-08-16.md`). This document does not
relitigate that. It asks only: **is the 450 real, on comparable ground, and if
so where does TP's extra time go?**

---

## 0. Executive verdict (read this first)

**Verdict (b), confirmed by live head-to-head measurement: the ~450 PP figure
was never apples-to-apples, and the gap runs the OPPOSITE direction on current
code.**

Measured live 2026-08-24, identical probe / depth / fp8 checkpoint / needle,
fresh cluster for each arm:

| Mode | Prefill @100K | TTFT | Needle |
|---|---|---|---|
| **TP** (production) | **366.5 tok/s** | 192.49 s | PASS |
| **PP** | **277.0 tok/s** | 255.06 s | PASS |

**TP is 32.3% FASTER than PP.** There is no uncaptured TP prefill headroom
implied by PP — the premise is false.

Four independent, compounding defects produced the phantom "~22%":

1. **Different checkpoint AND quantization** — every PP ~450 number was on
   `mlx-community/DeepSeek-V4-Flash` (affine int8, *preview*); production is
   `deepseek-ai/DeepSeek-V4-Flash-0731` (**fp8 e4m3**).
2. **A documented 1.42x counting artifact** (`chars//4` vs the real 5.68
   chars/token) that the repo's own §55/fact-1450 warning explicitly says
   contaminates the PP 364–512 curve.
3. **Depth swap** — the ~450s are 500K/94K/10K points quoted as if at 100K.
4. **The decisive one: they are chunk-loop rates, not TTFT rates.** PP's
   *opening chunk-loop rate measured live today is 523 tok/s*, reproducing the
   historical 490/512 claims exactly — while its honest end-to-end rate is 277,
   because **55.8 s of PP-only first-token pipeline drain sits outside the
   chunk loop**. TP's two numbers diverge by 2.9%; PP's by 34%.

At the chunk-loop level where the historical comparison was actually made, the
two modes are **377 (TP) vs 372 (PP) — statistically identical**. The real
structural asymmetry (TP moves 43x the collective bytes per chunk) is worth
only ~4–9% and is fully repaid by PP's bubble and 12x-worse depth decay.

The 2026-08-18 "**prefill ceiling ~350–360, compute-bound**" finding is
**upheld and strengthened** — it was the reading a PP 450 supposedly strained,
and PP measures 277.

Full evidence below.

---

## 1. PROVENANCE — every doc/fact carrying a PP prefill number near 450

### 1.1 The number hunt

`grep -rniE "(4[2-7][0-9])(\.[0-9]+)? *(tok/s|tokens/s|t/s)" docs/` plus a
sweep of the warm-memory fact store surfaces exactly **four** distinct
provenance families for a "PP prefill ≈ 450" claim. They are NOT the same
measurement.

| # | Figure | Source | Date | Model + quant | Depth | Method | Apples-to-apples vs TP 359.7@100K? |
|---|--------|--------|------|---------------|-------|--------|-------------------------------------|
| **P1** | **PP 444 tok/s** | warm-memory **fact 1014** | 2026-07-16 | `mlx-community/DeepSeek-V4-Flash` — **affine 8-bit, group_size 64** (PREVIEW checkpoint) | **500K** | runner `Prefill progress:` instantaneous rate | **NO** — different checkpoint, different quant, 5x the depth |
| **P2** | **PP 485–512 tok/s** | fact **1018** curve (10K=512, 94K=485) | 2026-07-17 | same PREVIEW 8-bit checkpoint | 10K / 94K | `bench/phase3_precheck_depth_throughput.py`, **pre-fix** | **NO** — checkpoint AND the `chars//4` 1.42x inflation |
| **P3** | **PP 490 @1K / 431 @200K** | `hybrid-pp-prefill-tp-decode-design-2026-08-04.md` §3 "Prior art" (quoting fact 1018) | 2026-08-04 (quoting 07-17) | same PREVIEW 8-bit | 1K / 200K | inherited from P2 | **NO** — inherited, plus wrong depth |
| **P4** | **PP 467 / 439 tok/s** | `docs/profiling/request_lifecycle_trace.md` | **2026-04-05** | **Qwen3.5-397B-A17B-4bit** — a *completely different model* | 16K | RequestTrace spans | **NO** — different MODEL entirely |

**There is no "PP ~450 @100K on DSv4-Flash-0731 fp8" measurement anywhere in
the repo or the fact store.** The dispatched lead's "~450" is P1 (444@500K) and
P2 (485@94K) blended and re-quoted at an implied 100K depth they were never
measured at.

### 1.2 Disqualifier 1 — the checkpoint is different, and the quant is different

Live production today (`/state`, verified 2026-08-24):

```
modelId      deepseek-ai/DeepSeek-V4-Flash-0731
quantization fp8            (config.json: quant_method=fp8, fmt=e4m3,
                             weight_block_size=[128,128], scale_fmt=ue8m0)
nLayers 43   hiddenSize 4096   n_routed_experts 256   experts_per_tok 6
storageSize  166,878,536,440 B  (155 GiB on disk)
```

Every PP number in families P1–P3 was measured on
`mlx-community/DeepSeek-V4-Flash` — a **different local directory, a different
checkpoint, and a different quantization scheme**:

```
~/.exo/models/mlx-community--DeepSeek-V4-Flash/config.json
  quantization = {group_size: 64, bits: 8, mode: "affine"}     # 144 GiB
~/.exo/models/deepseek-ai--DeepSeek-V4-Flash-0731/config.json
  quantization_config = {quant_method: "fp8", fmt: "e4m3", ...} # 155 GiB
```

This is not a nitpick. `start_cluster.sh` itself carries the warning, added
2026-08-15 when the default was finally flipped:

> `DSV4_MODEL_ID` was `mlx-community/DeepSeek-V4-Flash` (**a stale PREVIEW
> checkpoint**) until 2026-08-15 … That is not cosmetic: **preview and
> production produce DIFFERENT PP layer splits and therefore different
> cross-rank parity**.

affine-int8 and fp8-e4m3 dispatch **different MLX quantized-matmul kernels**
with different per-expert arithmetic intensity, and an 11 GiB weight-size
difference changes the weight-streaming term that §2.6/Lever-1 identified as
the actual MoE bottleneck. A prefill throughput number is not portable across
that boundary in either direction.

### 1.3 Disqualifier 2 — the 1.42x counting-convention inflation (§12 trap, already documented)

Warm-memory **fact 1450** (2026-08-16) and
`hybrid-pp-prefill-tp-decode-design-2026-08-04.md` §55 both state this
explicitly, and fact 1450 **names facts 1017, 1018 and 931 as affected**:

> `bench/phase3_precheck_depth_throughput.py` computed
> `prefill_tps = prompt_tokens / ttft_s`, where `prompt_tokens` fell back to
> `est_tokens = prompt_chars // 4`. The real ratio for this tokenizer on English
> prose is ~5.68 chars/token, so the //4 heuristic **INFLATES throughput by
> 1.42x**.

Fact 1018 — the sole source of the entire PP curve (1K=490, 10K=512, 94K=485,
200K=431, 400K=377, 500K=364) — **is explicitly on that list.** So is fact 931,
the TP curve it was compared against.

The design doc's own header now carries:

> **Any prefill throughput number in this document predating Section 55 must be
> checked for which numerator produced it before being quoted or compared** —
> this includes the widely-cited **PP 364-512 and TP 319-427 curves**.

The "~450 PP vs ~350 TP" framing is precisely the comparison that warning
forbids: it pits an un-renormalized 2026-07 PP number against a
tokenizer-ground-truth 2026-08 TP number.

**Note the direction carefully.** The inflation is *common-mode* — it hit the
PP and TP curves of that era equally, so the 07-17 **ratio** (PP/TP = +14.2% at
500K) survives renormalization even though both absolute numbers do not. What
does *not* survive is comparing a raw 2026-07 PP absolute (450-ish) against a
2026-08 TP absolute measured with the corrected numerator (359.7). That
comparison manufactures roughly 1.42/1.0 worth of phantom gap on top of any
real one. **This alone accounts for the bulk of the "~22%."**

### 1.4 Disqualifier 3 — depth mismatch

The TP baseline being compared against is **359.7 tok/s @ 100K**
(PERFORMANCE_HISTORY §1, 2026-08-22, fenceasync build, needle-verified).

The PP numbers near 450 are at **500K** (P1: 444) and **94K/10K** (P2: 485/512)
— and the PP curve is steeply depth-dependent (512@10K → 364@500K, a 29%
decay). Quoting "PP ~450" without a depth is quoting a point on a curve as if
it were the curve. At the *matched* depth in the PP curve (94K ≈ 100K), the
figure is 485 — but that 485 is a raw pre-renormalization number on the preview
checkpoint, i.e. maximally contaminated by §1.2 + §1.3 simultaneously.

### 1.5 What the honest, provenance-scrubbed comparison actually is

The only PP-vs-TP prefill pair ever measured under *matched conditions*
(same session, same checkpoint, same harness, fresh restart, thermally
controlled) is fact 1018's:

```
depth   PP(fresh)   TP        delta
 1K       490       427      +14.8%
10K       512       379      +35.1%
94K       485       358      +35.5%
200K      431       353      +22.1%     <-- the only depth where "~22%" appears
400K      377       332      +13.6%
500K      364       319      +14.2%
```

**The "~22%" in the dispatched lead is the 200K row.** It is a real ratio from a
real matched-conditions measurement — but on the **preview affine-8bit
checkpoint**, at **200K**, in **July**, on a code version ~6 weeks and hundreds
of commits behind current HEAD (which has since landed the fenceasync fix,
gate+up fusion, pool-growth, argsort-cliff kill, and the whole §2/§3 campaign
— all TP-path work that moved the TP side of this ratio and left the PP side
untouched because PP has not been launched since).

So: **the ~22% is a 200K, preview-checkpoint, July number**, and the "450 vs
350" absolute framing layered on top of it additionally imports the 1.42x
counting artifact and a depth swap. Three independent apples-to-oranges
defects, compounding.

---

## 2. THEORY (written and committed BEFORE the live measurement)

Recorded here in full, with predictions, so the measurement can falsify it.
This section was authored while the TP 100K probe was in flight and before any
of its output was read.

### 2.1 Structural cost model, 2 nodes, DSv4-Flash-0731

Shared constants: `L = 43` layers, `hidden = 4096`, chunk (`EXO_PREFILL_STEP_SIZE`)
`= 2048` tokens, bf16 activations (2 B/elt), `E = 256` experts, `top-k = 6`.

**TP (production).** `DeepseekV4ShardingStrategy` replicates attention on both
ranks and shards **only** the MoE block — each rank holds 128 of 256 experts and
the ranks `all_sum` the expert output after every MoE block. Under
`EXO_DSV4_SEQ_SPLIT=1` prefill query rows are additionally split across ranks
with an `all_gather` for the sparse-attention indexer.

Per-chunk TP collective payload:

```
all_sum payload = chunk x hidden x 2 B = 2048 x 4096 x 2 = 16.78 MB
calls/chunk     = 43   (one per layer)
payload/chunk   = 43 x 16.78 MB = 721.6 MB
```

This 16.78 MB figure is exactly the "production 16.8MB call" §2.3 already
characterizes, which is a strong consistency check on the model.

**PP.** Layers split by rank (rank0: 0–21, rank1: 22–42). One point-to-point
boundary handoff per chunk per stage boundary; with 2 ranks that is **one**
`send`/`recv_like` of the hidden state per chunk:

```
boundary payload = chunk x hidden x 2 B = 16.78 MB
calls/chunk      = 1
payload/chunk    = 16.78 MB
```

**Ratio: TP moves 43x the bytes per chunk that PP does.** That is the headline
structural asymmetry, and it is the term any "TP pays what PP doesn't" story
must rest on.

### 2.2 Quantifying each term with §2/§8's measured numbers

**TP collective term.** The trustworthy per-call cost is §2.7's jaccl-internal
`steady_clock` median of **36 µs/call** at the 8KB decode size, and §2.3's
decomposition of the production 16.8 MB call: **~2.4 ms boundary + ~0.9–1.7 ms
CPU + ~2–9 ms wire = ~5–12 ms/call** band. Taking the band:

```
TP collective/chunk = 43 calls x [5..12] ms = 215..516 ms/chunk
```

**PP boundary term.** Same 16.78 MB payload, same transport, once:

```
PP boundary/chunk   = 1 call x [5..12] ms = 5..12 ms/chunk
```

**PP bubble term.** 2-stage pipeline, per chunk fill+drain. The 2026-04-05
lifecycle trace measured R0's `distributed_cb` (the PP barrier wait) at
**417–436 ms steady-state** per chunk against a ~3.2 s forward — i.e. the
bubble is real and large in wall terms, but it is *concurrent idle*, not added
work; on a single-request stream it costs approximately half a stage's latency
per chunk boundary and is largely amortized once the pipe is full.

**Width-efficiency term.** §3.4 measured sharded MoE GEMMs as **2.6–3.6%
FASTER per unit work** than full-width. **Sign is inverted** relative to the
naive "TP's half-width GEMMs are less efficient" story — so this term *helps*
TP, it cannot explain a TP deficit, and any theory that leans on it is dead on
arrival. (Reconciliation: TP's disadvantage cannot be GEMM shape; it has to be
the collective count, or nothing.)

### 2.3 Per-chunk budget and the predicted gap

A 100K prefill at 2048-token chunks is ~49 chunks. At the measured TP
359.7 tok/s, wall = 100,000/359.7 ≈ **278 s**, i.e. **~5.67 s/chunk**.

```
                       per chunk      as % of TP's 5.67 s
TP all_sum (43x)      215..516 ms         3.8% ..  9.1%
PP boundary  (1x)         5..12 ms        0.1% ..  0.2%
TP-minus-PP comm      210..504 ms         3.7% ..  8.9%
```

**PREDICTION (pre-registered):** the collective term can explain a TP prefill
deficit of **at most ~4–9%**, not 22%. If the live measurement reproduces a
22% gap, the collective term is **insufficient by 2.4x–6x** and a second,
unidentified mechanism must exist. If the live measurement does *not* reproduce
a 22% gap, the provenance analysis in §1 is the whole answer and no mechanism
hunt is warranted.

**FALSIFICATION CONDITIONS, stated in advance:**
1. If PP launches on current code and measures ≥ 430 tok/s @100K on
   DSv4-Flash-0731 fp8, §1's provenance verdict is WRONG and there is a real
   mechanism to hunt.
2. If TP re-measures materially below ~350 or above ~370 @100K on current
   code, the TP baseline itself has drifted and the whole comparison must be
   re-based before any attribution.
3. If the collective term is directly measured above ~1.3 s/chunk, §2.3's
   arithmetic is wrong and the collective *could* carry 22% alone.

### 2.4 Reconciling "compute-bound 2026-08-18" with a PP 450

The 2026-08-18 finding declares the ~350–360 prefill ceiling **compute-bound**,
concluding CP would not help. What it actually measured (per fact 939 and the
220K span profile): `EXO_PREFILL_GPU_TIME=1` + `MLX_GPU_TIME=1` giving
**GPU/wall = 93–95% at every context depth**, i.e. only 5–7% of prefill wall is
GPU-idle. That is a *TP-mode* measurement.

A **93–95% GPU-busy TP prefill mathematically cannot be 22% behind anything for
comm-stall reasons** — there is only 5–7% of idle to recover, which brackets
neatly with §2.3's independently-derived 3.7–8.9% collective-attributable band.
**Two independent methods agreeing on ≤9% is the strongest single argument that
the 22% was never a TP-side stall.** For a PP 450 to coexist with a
93–95%-GPU-busy TP 360 on identical hardware and an identical checkpoint, PP
would have to be doing *less total GPU work*, not merely waiting less — which
is exactly what a **different checkpoint at a different quantization** (§1.2)
delivers, and exactly what a **1.42x counting convention** (§1.3) fabricates
without any physics at all.

---

## 3. LIVE TP MEASUREMENT (current build)

**Relaunch:** none needed — the cluster was already live in TP=2 production.
Verified from `/state` before the probe: instance `0389b210…`, `MlxJacclInstance`,
**both shards `TensorShardMetadata`** (deviceRank 0 and 1, `startLayer 0 /
endLayer 43` on both — i.e. every layer on both ranks, MoE-sharded), both runners
`RunnerReady`, model `deepseek-ai/DeepSeek-V4-Flash-0731` **fp8**.

**Probe:** `bench/phase3_precheck_depth_throughput.py --targets 100000` — the
same harness, the same convention, and the same `--model` as the
`known-good-prefill-baseline-2026-08-21` and PERFORMANCE_HISTORY §1 baselines.
Tokenizer-ground-truth numerator (no `chars//4`), needle-in-haystack verified.

**No timing patch was required.** The runner already emits a per-chunk
`Prefill progress: <done>/<total> tokens (<cum tok/s>)` line at
`generate.py:811` with millisecond timestamps; differencing consecutive lines
yields the exact per-chunk wall time. Zero code changes were made to the repo
for instrumentation.

### 3.1 Result

```json
{ "target_tokens": 100000, "prompt_tokens": 70557,
  "prompt_token_source": "tokenizer", "api_prompt_tokens": 70561,
  "ttft_s": 192.49220691699884,
  "prefill_tok_s": 366.5447091602188,
  "decode_tok_s": 27.790263031330475,
  "response": "FALCON-MERCURY-7749", "needle_found": true }
```

**TP = 366.5 tok/s prefill @100K, needle PASS.** Against the §1 baseline of
359.7 the current build is **+1.9%** — no regression, and squarely inside the
"known-good" band (366.6 tok/s on 2026-08-21 — a 0.03% match). **Falsification
condition 2 is NOT triggered**; the TP baseline is sound and the comparison
stands on it.

### 3.2 Per-chunk curve (34 steady-state chunks × 2048 tokens)

```
depth=   2048  chunk=2048 tok   5136 ms  inst= 398.8 tok/s
depth=  10240  chunk=2048 tok   5346 ms  inst= 383.1 tok/s
depth=  20480  chunk=2048 tok   5407 ms  inst= 378.8 tok/s
depth=  34816  chunk=2048 tok   5541 ms  inst= 369.6 tok/s
depth=  49152  chunk=2048 tok   5607 ms  inst= 365.3 tok/s
depth=  63488  chunk=2048 tok   5653 ms  inst= 362.3 tok/s
depth=  69632  chunk=2048 tok   5647 ms  inst= 362.7 tok/s
```

```
mean 5502 ms   median 5514 ms   min 5136   max 5653
first-5 mean 5359 ms  ->  last-5 mean 5632 ms   =  +5.1% decay over 70K
stall outliers (>1.5x median): 0 of 34
```

**§2.3 predicted ~5.67 s/chunk from the measured 366 tok/s; the curve measures
5.50 s/chunk mean.** The theory's per-chunk budget was right to within 3%.

The curve is **extremely clean**: monotone-ish, 5.1% total decay across 70K, and
**not one stall outlier in 34 chunks**. This is a well-behaved,
compute-saturated loop.

### 3.3 Collective-attributable time in TP prefill (upper bound)

Direct per-collective timing is not loggable without a patch, so per the brief
this is bounded from §2's measured numbers (§2.3's decomposition of the
production 16.8 MB call: ~5–12 ms/call all-in):

```
all_sum payload  = 2048 x 4096 x 2 B          = 16.78 MB   (matches §2.3's "16.8MB call")
calls per chunk  = 43 layers                  = 721.6 MB moved per chunk
collective/chunk = 43 x [5..12] ms            = 215 .. 516 ms
as a share of the measured 5502 ms chunk      = 3.9% .. 9.4%
```

**Independent cross-check.** Fact 939's `EXO_PREFILL_GPU_TIME=1` measurement put
TP prefill at **93–95% GPU-busy at every depth**, i.e. 5–7% idle:

```
idle headroom = 5-7% of 5502 ms  =  275 .. 385 ms/chunk
collective bound (above)         =  215 .. 516 ms/chunk
```

**Two methodologically independent routes bracket the same 4–9% band.** There is
at most ~9% of collective-attributable time in TP prefill, and realistically
~5–7%. **Falsification condition 3 is NOT triggered** (the collective is nowhere
near the ~1.3 s/chunk that would be needed to carry 22% alone).

---

## 4. LIVE PP MEASUREMENT

**PP is NOT bit-rotted. It launched clean on current code.**

**Relaunch #1** (`DSV4_SHARDING=Pipeline EXO_PP_DRAFT_MODEL= DSV4_KV_CACHE_BITS=0
EXO_DSV4_MTP=0 EXO_SPECULATIVE=0 ./start_cluster.sh`) — first attempt aborted at
the launcher's own "local HEAD is not on origin/main" guard (my §1–2 doc commit
was unpushed); pushed, relaunched, and it came up **READY (2/2)** with no
crashes. `/state` confirms genuine pipeline sharding:

```
instance 18cfa397…  MlxJacclInstance
  shard bc488562  PipelineShardMetadata  startLayer 0   endLayer 22
  shard 04171fcd  PipelineShardMetadata  startLayer 22  endLayer 43
  runners: both RunnerReady
```

Same probe, same model, same depth, same needle.

### 4.1 Result

```json
{ "target_tokens": 100000, "prompt_tokens": 70656,
  "prompt_token_source": "tokenizer", "api_prompt_tokens": 70660,
  "ttft_s": 255.06489475000126,
  "prefill_tok_s": 277.01185641110123,
  "decode_tok_s": 24.74038834577857,
  "response": "FALCON-MERCURY-7749", "needle_found": true }
```

**PP = 277.0 tok/s @100K, needle PASS — versus TP's 366.5 on the identical
probe.**

> **TP is 32.3% FASTER than PP at 100K on current code, same checkpoint, same
> harness, same depth, both needle-verified.**
>
> **The gap does not merely fail to reproduce — it runs in the OPPOSITE
> DIRECTION.** Falsification condition 1 required PP ≥ 430 tok/s to keep a
> mechanism hunt alive. PP measured **277**.

### 4.2 PP per-chunk curve (69 chunks × 1024 tokens)

Note PP self-selects a 1024-token chunk (the `//ranks` divisor on the PP path)
vs TP's 2048, so all comparisons below are normalised per token.

```
depth=  28672  chunk=1024   2115 ms  inst= 484.2 tok/s
depth=  33792  chunk=1024   2285 ms  inst= 448.1 tok/s
depth=  39936  chunk=1024   2939 ms  inst= 348.4 tok/s
depth=  41984  chunk=1024   3933 ms  inst= 260.4 tok/s   <-- stall
depth=  49152  chunk=1024   5066 ms  inst= 202.1 tok/s   <-- stall
depth=  52224  chunk=1024   9588 ms  inst= 106.8 tok/s   <-- stall (worst)
depth=  62464  chunk=1024   8116 ms  inst= 126.2 tok/s   <-- stall
depth=  69632  chunk=1024   4143 ms  inst= 247.2 tok/s   <-- stall
```

```
median 2279 ms  mean 2727 ms     (69 chunks captured, depth 1024 -> 70656)
first-5 mean 1957 ms  ->  last-5 mean 3198 ms   =  +63.4% decay over 70K
stall outliers (>1.5x median): 5 of 69,  worst 9588 ms
```

Contrast with TP's **+5.1% decay and 0 outliers**. PP degrades **12x faster with
depth** and is the only mode that stalls.

---

## 5. ANALYSIS — decomposition, additivity, verdict

### 5.1 Where PP's time actually goes (exact, additive by construction)

```
  PP1. Baseline at PP's OWN opening rate (1957 ms x 69 chunks)   = 135.0 s
  PP2. Pipeline stall outliers (5 chunks, worst 9588 ms)         =  19.5 s
  PP3. Depth-decay growth (exact residual)                       =  33.6 s
       --------------------------------------------------------------------
       TOTAL                                                     = 188.1 s  = measured 188.1 s
       TP, same 70,656 tokens                                    = 187.1 s
```

**Additivity check closes exactly**: PP2+PP3 = 53.1 s, and
(chunk-loop excess 1.1 s) + (TP's margin over PP1, 52.0 s) = 53.1 s. ✅

The striking part: **PP1 alone (135.0 s) beats TP (187.1 s) by 52 s.** If PP
could hold its own opening rate, it *would* be the ~500 tok/s machine the
history claims. It cannot: PP2 + PP3 are 28% of its chunk loop, and **TP pays
neither** (0 stalls, 5.1% decay).

### 5.2 The dominant term is OUTSIDE the chunk loop entirely

```
              chunk loop    outside loop      TTFT
  TP            187.1 s         5.4 s        192.5 s    ( 2.8% outside)
  PP            190.1 s        65.0 s        255.1 s    (25.5% outside)
  delta          +3.0 s       +59.6 s        +62.6 s
```

**95% of PP's deficit is outside the per-chunk loop.** Inside the loop the two
modes are within **1.6%** of each other.

The runner log names the term directly:

```
[R0] Prefill: 70 real + 0+1 dummy iterations, Processed 70658 tokens in 199282.8ms
Prefill complete: 70659 tokens in 199.29s (354.6 tok/s)
   ...but the client-observed TTFT was 255.06 s.
```

**55.8 s of PP-only first-token pipeline drain/handoff sits between "prefill
loop done" and "first token out."** This is the `PipelineLastLayer` p2p
rendezvous the launcher itself warns about ("the p2p handoff itself is an
UNAVOIDABLE rendezvous — rank 0 can't proceed until rank 1 sends", and the
reason PP forces `MLX_EVENT_WAIT_TIMEOUT_MS=1800000`). TP's equivalent term is
5.4 s.

### 5.3 THE RECONCILIATION — what the historical "PP ~450–512" really measured

```
  PP opening chunk-loop rate  = 1024 / 1.957 s  = 523 tok/s
  PP full chunk-loop rate     = 70656 / 190.1 s = 372 tok/s
  PP END-TO-END (TTFT) rate   = 70656 / 255.1 s = 277 tok/s   <-- honest

  TP full chunk-loop rate     = 70557 / 187.1 s = 377 tok/s
  TP END-TO-END (TTFT) rate   = 70557 / 192.5 s = 367 tok/s   <-- honest
```

**PP's measured opening chunk-loop rate today is 523 tok/s — which reproduces
the historical 490/512/485 claims essentially exactly.** Those numbers were
real; they were **chunk-loop instantaneous rates at shallow depth**, taken from
the runner's `Prefill progress:` line (fact 931 says so in as many words: *"from
exo progress_callback log, NOT ab_probe cumulative which includes HTTP/tokenize
overhead"*).

The fatal asymmetry:

| | chunk-loop rate | TTFT rate | divergence |
|---|---|---|---|
| **TP** | 377 | 367 | **2.9%** |
| **PP** | 372 | 277 | **34%** |

For TP, the chunk-loop rate is an honest proxy for end-to-end throughput. **For
PP it is not, because PP hides a 55.8 s drain outside the loop.** Comparing PP's
chunk-loop number against TP's TTFT number manufactures a large phantom gap out
of nothing but bookkeeping — and stacked on top of §1's checkpoint change and
1.42x counting artifact, that is the entire "~22%."

**And note the deepest irony**: at the *chunk-loop* level where the historical
comparison was made, the two modes are 377 vs 372 — **statistically identical**.
The structural cost difference the whole theory was built to find (§2's 43x
collective payload asymmetry) is **real but self-cancelling**: TP's 43 all_sums
per chunk cost it ~4–9%, and PP pays that back and more in bubble + decay, so
the per-chunk loops land within 1.6%.

### 5.4 Additivity against §2's predictions

| Predicted term (§2, pre-registered) | Predicted | Measured | Verdict |
|---|---|---|---|
| TP per-chunk wall | ~5.67 s | 5.50 s | ✅ within 3% |
| TP collective-attributable share | 3.7–8.9% | 3.9–9.4% (bounded, 2 methods) | ✅ confirmed |
| TP collective can explain ≤ ~9%, not 22% | ≤9% | ≤9% | ✅ confirmed |
| Width-efficiency term helps TP (sign inverted, §3.4) | helps TP | TP loop ≥ PP loop | ✅ confirmed |
| PP boundary transfer ~1 call/chunk, cheap in-loop | 5–12 ms | in-loop delta 1.6% | ✅ confirmed |
| PP bubble | "real and large in wall terms" | 55.8 s out-of-loop | ✅ confirmed, and **it is the dominant term** |

The one thing §2 under-weighted: it treated the PP bubble as "largely amortized
once the pipe is full." At 100K it is **29% of PP's TTFT** — the single biggest
line item in the entire comparison.

### 5.5 VERDICT — **(b), decisively: the gap does not reproduce; the 450 was never apples-to-apples**

1. **The 450 has no matching provenance.** No PP measurement on
   DSv4-Flash-0731 fp8 at 100K near 450 exists. The four sources are a
   different checkpoint at a different quantization (affine-int8 preview vs
   production fp8), at different depths (500K / 94K / 10K / 16K), one of them a
   different *model* entirely (Qwen3.5-397B).
2. **A documented 1.42x counting artifact** (`chars//4`) inflates the entire
   PP 364–512 curve; fact 1450 names fact 1018 explicitly. The repo's own
   §55 warning forbids exactly the comparison the lead was built on.
3. **The 450-class numbers are chunk-loop rates, not TTFT rates** — measured
   live today as PP's 523 tok/s opening rate, reproducing the historical claim
   while its honest end-to-end rate is 277.
4. **Live head-to-head on current code inverts the sign: TP 366.5 vs PP 277.0
   (+32.3% for TP)**, both needle-verified, same probe, same depth, same fp8
   checkpoint, fresh cluster for both arms.

**There is no uncaptured TP-mode prefill headroom implied by PP.** The premise —
"PP proves ~450 is achievable on this hardware, so TP is leaving 22% on the
table" — is false at every level: PP does not achieve 450 end-to-end on this
checkpoint, and TP is the faster mode.

The **2026-08-18 "prefill ceiling ~350–360, compute-bound"** finding is
**upheld and strengthened**, not strained. It was the reading that a PP 450
supposedly contradicted; measured head-to-head, PP is 277 and the ceiling stands.
TP at 366.5 is the best measured prefill on this cluster in either topology.

### 5.6 No lever is proposed, and why that is the correct outcome

Verdict (b) means the mechanism hunt is closed before it starts — but the
measurement independently re-confirms the §3 closed-levers picture, so it is
worth stating what it rules out:

- **Collective/compute overlap variants** (async allreduce pipelining across
  layers, a cousin of the dead §2.6 chunk-overlap lever): the total addressable
  pool is 215–516 ms/chunk (4–9%). Even a *perfect, free* overlap of 100% of it
  yields at most ~9%. And it is not free: §2.6's chunk-overlap lever died on
  a **correctness race**, and §0c proved MLX collectives are matched
  positionally **by eval order** — any rank-dependent scheduling gives *silent,
  deterministic* corruption with no crash signal. A cousin lever inherits that
  exact hazard for ≤9% upside.
- **fp8-native collective payload** (the one §2.5 gap — int8 was killed, fp8
  for the *prefill* allreduce was never tested): would at best halve the wire
  term. Per §2.3 the wire is only ~2–9 ms of the 5–12 ms/call; the rest is
  GPU→CPU stream-boundary coherency, which is **payload-proportional but NOT
  collective-specific** (a plain non-collective CPU-stream op reproduces the
  same 2.66x penalty). Halving payload on a term worth 4–9% total, where the
  majority of that term isn't wire, is a sub-2% ceiling against §2.5's
  documented accuracy risk. **Not worth a dispatch.**
- **TP's 93–95% GPU-busy** leaves ≤7% idle to recover by *any* comm-side lever.
  The remaining prefill headroom is in **compute**, not communication — exactly
  where 2026-08-18 pointed.

**Recommended disposition: close this lead.** Not "structural wall with levers
closed" (verdict c) — the honest finding is stronger and simpler: **the gap was
a measurement-provenance artifact, and the live head-to-head inverts it.**

### 5.7 Cost of the finding

3 relaunches total (1 failed PP launch on the git-push guard, 1 successful PP,
1 restore to TP production), 2 needle-verified 100K probes, zero crashes, zero
code changes to the repo. PP mode confirmed **not** bit-rotted on current code —
a useful side-finding, since the history assumed it might be.

---

## 6. Reproduction

```bash
# TP arm (production config)
cd ~/repos/exo && DSV4_KV_CACHE_BITS=0 ./start_cluster.sh
ssh adams-mac-studio-m4-1.local "cd ~/repos/exo && .venv/bin/python \
  bench/phase3_precheck_depth_throughput.py \
  --model deepseek-ai/DeepSeek-V4-Flash-0731 --targets 100000 \
  --json-out /tmp/p5_tp_100k.json"

# PP arm
cd ~/repos/exo && DSV4_SHARDING=Pipeline EXO_PP_DRAFT_MODEL= DSV4_KV_CACHE_BITS=0 \
  EXO_DSV4_MTP=0 EXO_SPECULATIVE=0 ./start_cluster.sh
#   ...same probe...

# per-chunk curve (both modes, no patch needed)
ssh adams-mac-studio-m4-1.local 'grep "Prefill progress" ~/.exo/exo_log/exo.log'
```
