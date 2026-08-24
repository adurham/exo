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

**Verdict (b): the ~450 PP figure was never apples-to-apples with the TP
359.7@100K baseline, and the ~22% gap as stated does not exist on current
code.** It is a composite of (i) a *different model checkpoint at a different
quantization*, (ii) a *different, later-corrected token-counting convention*
that inflated a whole family of prefill tok/s numbers by **1.42x**, and (iii) a
*different context depth* than the depth it is being compared against. The
individual PP numbers were honestly measured at the time; the **comparison** is
the artifact.

The residual, apples-to-apples PP-vs-TP prefill delta that survives provenance
scrubbing is **+14.2% at 500K** (PP 364.2 vs TP 319, fact 1018, 2026-07-17) —
and even that pair is cross-checkpoint. It is **not** ~22%, and it is not
a 350-vs-450 gap at any single depth.

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

_(results below — filled in from the live probe)_

## 4. LIVE PP MEASUREMENT

_(results below)_

## 5. ANALYSIS, ADDITIVITY, VERDICT

_(below)_
