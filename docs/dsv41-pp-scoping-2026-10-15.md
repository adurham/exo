# DSv4.1 pipeline-parallel scoping — Gate 0 (2026-10-15)

**Status:** offline scoping study, Gate 0 only. **GATE 0 KILLED** — no full study run.
**Branch:** `p20/pp-scoping-2026-10-15` (based on `origin/main` `4f18c8bd7`).
**Live cluster:** untouched. **0 boots, 0 config changes, 0 node-side edits, 0 model-executing requests.**
All cluster contact read-only (`/state`, logs); the study is otherwise a code + document read.

**Labels.** `MEASURED` = read from a log/db or a cited instrument · `DERIVED` = arithmetic on MEASURED
inputs · `BOUND` = ceiling/floor · `MODELED` = cost model · `ESTIMATE` = judgment without a measured input.
Every number carries a source. Old-era (DSv4-Flash) figures carry the §55 / fact-1450 counting-convention
caveat (a `chars//4` numerator inflated that whole era's prefill tok/s by 1.42×); **ratios from that era
survive renormalization, absolutes do not.**

**Method / provenance.** Read: owner brief `docs/dsv41-scaling-alternatives-brief-2026-10-15.md`; hybrid
design `docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md` (CURRENT STATE + §§107–108 only);
`docs/dsv41-tp3-scaling-readiness-2026-10-11.md` (+2026-10-12 addendum, `p20/tp3-scaling-readiness`);
`docs/dsv41-big-append-census-2026-10-13.md` + `docs/benchmarks/big-append-census-2026-10-13/census.csv`;
`docs/mtp-dspark-tp-port-decision-gate-2026-08-22.md`; `docs/jaccl-internal-timing-allsum-transport-fast-2026-08-21.md`;
`docs/moe-all-sum-skew-vs-comms-2026-08-19.md`; **`docs/p5-tp-prefill-gap-2026-08-24.md`** (the pivotal
matched PP-vs-TP head-to-head); `PERFORMANCE_HISTORY.md` (recent entries). Engine code read at the
submodule pins of `origin/main`: `mlx-lm` `6cc9c1e87` (`mlx_lm/models/deepseek_v41/{attention,indexer,compressor,cache,model,moe,mtp,spec,engram,exl3_build,config}.py`)
and `src/exo/worker/engines/mlx/dsv41/load.py`. Checkpoint config (MEASURED, `docs/benchmarks/phase4-exl3-prod-mtp-compat-2026-09-28/raw/p7c.out:49-50`):
40 body layers + 3 MTP, `kv_source_layer_ids=[2,8,14,20]`, `compress_ratios=[0,0,2×18(2..19),1×20(20..39),0,0,0]`,
`index_source_layer_ids=[2,8,14,20,24,28,32,36]`, `candidate_source_layer_id=20`, `engram_layer_ids=[1,14]`,
`dim=5120`, `head_dim=512`, `index_head_dim=128`, `o_groups=8`, `hc_mult=4`, `vocab=129280`.

---

## 0. VERDICT — GATE 0 KILLS (item 1 decisive; item 2 fires; item 3 corroborates; item 4 clean)

| Gate 0 item | Result | Criterion | Fires? |
|---|---|---|---|
| 1. Decode inequality `C_d ≥ S_d/2 + 2h` | **FAILS** — `C_d ≈ 2.8–5.0 ms` vs required `≥ 9.3 ms` (and `S_d/2 ≈ 13.5–27 ms` on the expert slice alone) | (a) decode below ~95 % of 34.6 tok/s | **YES** |
| 2. Calibration of `edge ≈ C + R/2` | **FAILS** — matched live head-to-head on the same old model finds the sign **inverted** (TP +32.3 %); the model still predicts a large positive edge | (b) calibration miss > 3 pp, no corrected model restores a positive edge | **YES** |
| 3. Bubble-weighted prefill gain | **Model's own optimistic edge gives +5.98 % weighted (< +8 %)**; calibrated edge gives −8.36 % | (c) weighted gain < ~+8 % | **YES** |
| 4. Dependency direction | **No backward edge** under a contiguous ascending split; all edges forward | (d) non-eliminable backward dependency | no |

**One-line verdict.** The engine refusal in `load.py` is a code state, not physics — the cross-layer state
*can* be transported (item 4). But PP does not pay: it **doubles per-round decode latency** because a decode
round cannot be pipelined across stages (item 1), and its only measured matched-conditions prefill
comparison on this cluster shows it **slower than TP, not faster** (item 2). The prefill upside the brief
hypothesised (+14–18 % @50–105 K) traces to a 2026-07 measurement that a 2026-08-24 matched head-to-head
**inverts**. Study closed; nothing further built or modelled.

---

## Gate 0 · Item 1 — DECODE INEQUALITY (the decisive kill)

### Setup (from engine code + measured profiles)

A single decode round under the current TP+MTP production build runs `γ+1 = 4` verify rows through the body
in one forward, plus a cheap 3-stage draft head; the round is *one host sync* (`mlx_lm/.../spec.py:149,154`).

- Decode round **95 ms / 3.28 tokens → 34.6 tok/s** `[MEASURED, owner brief §1; live next27]`.
- **Verify forward ms-by-rows**, measured on this 2×M4-Max TP=2 cluster `[MEASURED, spec.py:76-77]`:
  `{1:58.5, 2:74.9, 3:87.9, 4:97.7, 5:111.7, 6:120.1}`. At γ=3 the round runs **4 verify rows = 97.7 ms** —
  i.e. **the verify forward is essentially the whole round**; the draft is ~ms.
- Collectives per round `[DERIVED, code]`: `attn.all_sum` + `moe.all_sum` per layer (`attention.py:229-231`,
  `moe.py`) × 40 body layers + 1 head `all_sum` (`exl3_build.py`, `ShardedHead`) = **81**, plus a few on the
  3-layer draft head.
- Per-collective transport, in production `[MEASURED, jaccl-internal-timing-2026-08-21.md:97-107]`:
  median **36.1/36.0 µs** @8192 B; p95 165/143 µs. Downstream correction `[MEASURED, PERFORMANCE_HISTORY 2026-08-22]`:
  the true collective cost is **2.9–5.3 % of decode wall**; the earlier 21 % / ~4094 µs sync-span figures
  are **methodology artifacts** (a sync-span captures the collective *plus* queued-GPU drain and dispatch
  outside the call — jaccl-internal-timing:5-15).

### Both sides of the inequality

Parity requires (owner brief; algebra re-derived and confirmed):
`PP round ≈ S_d + R_d + 2h + draft` vs `TP round ≈ S_d/2 + R_d + C_d + draft`
⇒ **PP holds parity only if `C_d ≥ S_d/2 + 2h`.**

- `R_d` (replicated work) is topology-neutral in wall time: TP pays all 40 layers' fixed overhead in
  parallel on both nodes; PP pays 20+20 serially — both are 40 layers' worth of wall. Cancels. `[DERIVED]`
- **`C_d` ≈ 81 × ~36–70 µs = 2.9–5.7 ms**; the direct measurement (2.9–5.3 % of the 95 ms round) gives
  **`C_d ≈ 2.8–5.0 ms`** `[MEASURED share × MEASURED round, DERIVED]`. (Do **not** bound `C_d` by
  81 × p95 = 13 ms — the calls do not all hit p95 together; that bound inflates the PP-favourable corner.)
- **`2h`** ≈ 0.1–0.5 ms: one activation handoff per round of the hc-expanded hidden state
  `[b, n, hc_mult=4, dim=5120]` bf16; at n=4 that is **~164 KB** `[DERIVED, model.py:256 broadcast + cache.py]`.
- **`S_d/2`** (per-node sharded decode compute per round): the MoE expert slice is `4.46 GB/rank` at m=4
  (`PERFORMANCE_HISTORY` P5 verify roofline @91 K) `[MEASURED]`; its pure-roofline read is **9.0 ms** at the
  measured 497 GB/s, and it runs at **~1.5–3× its floor → 13.5–27 ms** `[MEASURED × range, DERIVED]`; plus the
  sharded attention/dense slices (2.04 GB/rank) → **`S_d/2 ≈ 13.5–40 ms`** `[DERIVED, ESTIMATE upper]`.

### Verdict

Bar = 95 % of 34.6 = **32.87 tok/s → round ≤ 99.8 ms**. PP is killed if `S_d/2 + 2h − C_d > ~4.7 ms`;
with `C_d ≤ 5 ms` and `2h ≈ 0.5 ms` that is **`S_d/2 > ~9.3 ms`** `[DERIVED]`.

```
S_d/2 =  9.0 ms (experts at pure roofline), C_d = 5 ms -> PP round  99.1 ms -> 33.1 tok/s  (at the bar)
S_d/2 = 13.5 ms (experts at 1.5x floor),   C_d = 5 ms -> PP round 103.6 ms -> 31.7 tok/s  BELOW
S_d/2 = 27.0 ms (experts at 3.0x floor),   C_d = 5 ms -> PP round 117.1 ms -> 28.0 tok/s  BELOW
S_d/2 = 40.0 ms (+sharded attn/dense),     C_d = 5 ms -> PP round 130.1 ms -> 25.2 tok/s  BELOW
```

**KILL (criterion a).** The inequality fails by ~3–10×: `C_d` (~3–5 ms) is nowhere near
`S_d/2 + 2h` (~14–40 ms). Modeled PP decode is **~25–33 tok/s, central ~28–31**, i.e. **at or below the
95 % bar** — only the single most PP-favourable corner (full-width experts at pure roofline, zero sharded
dense/attention scaling) grazes the bar at ~33 tok/s. This is the expected shape: **the current TP is not
collective-bound** (the 2026-08-22 async-fence fix lifted TP decode ~18.5 → 29.2–31 tok/s
`[MEASURED, PH:638]`), so PP's structural win — removing the collective — buys ~3–5 ms, while PP's
structural cost — serialising the two stages' compute — costs `S_d/2`.

**Historical direction, corroborating.** On the *old* model PP+DSpark single-request decode measured
**27–33 tok/s** — but against a then-TP of **15–20 tok/s** (`docs/mtp-dspark-tp-port-decision-gate-2026-08-22.md`)
that was *collective-bound*. Against today's 34.6 that same 27–33 band lands **at or below parity**.
Standing directive (2026-08-16 settlement) is unchanged: **MTP/DSpark decode goes through TP.**

**Escape hatch — priced separately, never as baseline parity.** *Speculative pipelining* (stage 1 drafts
round k+1 while stage 2 verifies round k) is the only route that could hide the intra-round bubble. It is
**blocked at baseline** by a data dependency — round k+1's draft needs round k's committed tokens — so it
requires either a re-architected draft/verify split across stages or accepting draft staleness. **High-effort,
unbuilt, un-modelled here; not counted.**

**Scope caveat.** Item 1 covers **single-stream decode only** (the owner's Requirement-3 case, and the
brief's framing). Under concurrency c≥2 a *batched* PP pipeline can fill a stage's bubble with another
request's rows — but that is the batched-PP design that was never shipped (§107 lists cross-phase
concurrency + cancellation among the hybrid's costs), so it is a separate, larger build, not a Gate-0 parity
argument. Concurrency is **out of scope here and stated, not implied.**

---

## Gate 0 · Item 2 — CALIBRATION of `edge ≈ C + R/2` (the model is untrustworthy)

**The model under test** `[owner brief §2.4, DERIVED]`: PP prefill edge ≈ `C + R/2 − h`, where `C` = the
collective-sync share of chunk wall, `R` = the replicated row-local share (PP halves it: each stage runs
half the layers), `h` ≈ negligible handoff. Applied to the **current** engine's instruments
(`C` 16.9 % @16 K → 6.9 % @340 K in-model; `R/2` ≈ 4.9–5.2 % @50–100 K) it yields **+14–18 % @50–105 K,
tapering ~+7 % deep**.

**The stated anchor** `[MEASURED, old era]`: closure-math pair **PP 364 vs TP 319 tok/s @500 K = +14.1 %**
(`hybrid-…-design-2026-08-04.md:224,254`). The model is untrustworthy unless it reproduces +14.1 % within ~3 pp.

### What the anchor actually is — a void measurement, not a reproduction target

`docs/p5-tp-prefill-gap-2026-08-24.md` is a **matched-conditions live head-to-head** on the *same old model*
(`deepseek-ai/DeepSeek-V4-Flash-0731`, fp8, same probe, same depth, fresh cluster each arm, needle-verified
both arms). Result at 100 K:

| Mode | chunk-loop rate | TTFT rate |
|---|---:|---:|
| **TP** (production) | **377 tok/s** | 366.5 tok/s |
| **PP** | **372 tok/s** | 277.0 tok/s |
| delta | **PP −1.3 %** | **TP +32.3 %** |

`[MEASURED, p5-tp-prefill-gap-2026-08-24.md §3-§5]`

- **At the chunk-loop level — the level `C + R/2` models — PP and TP are within 1.6 % (PP slightly slower).**
  The +14.1 % does not reproduce; the sign is inverted.
- `p5` attributes the old PP curve to **four compounding provenance defects** `[MEASURED]`: (1) a *different
  checkpoint and quantization* — every PP ~450 number was on `mlx-community/DeepSeek-V4-Flash` affine-int8
  preview, production is fp8 e4m3, and the launcher itself warns the two produce **different PP layer splits
  and different cross-rank parity**; (2) the documented **1.42× `chars//4` inflation** (fact 1450 names the
  whole PP 364–512 curve); (3) a **depth swap** (500 K/94 K/10 K quoted as if at 100 K); (4) **chunk-loop
  rate vs TTFT rate** — PP hides **55.8 s of first-token pipeline drain outside the chunk loop** (25.5 % of
  its TTFT), which `C + R/2` does not model at all.
- PP also decays **12× faster with depth** and is the only mode that stalls (5/69 chunks > 1.5× median,
  worst 9.6 s) `[MEASURED, p5 §4.2]`.

### Corrected chain of reasoning (as ruled, do not record it as "anchor failed the arithmetic")

1. The **+14.1 % anchor is void** on provenance (`p5` §1): different checkpoint/quant/splits + the 1.42×
   numerator + depth swap + chunk-loop-vs-TTFT mix. A void anchor makes the test **undefined**, not *failed*.
2. The **replacement anchor** is the matched **chunk-loop** head-to-head — **PP −1.3 %** — because `C + R/2`
   models *chunk-loop* cost, not the out-of-loop drain.
3. Plugging the old model's 100 K profile (`C`, `R`) into `C + R/2` still predicts a **large positive edge
   (~+10–15 %)** against a measured **−1.3 %** — a **miss far beyond 3 pp**.
4. There is a **mechanism** for the miss, and it is a known in-repo artifact: `C` is read from **sync-span
   instruments**, and this project already proved those **inflate cost** on decode (21 % → the real 3–5 %).
   The model's own inputs are contaminated in the direction that manufactures a PP edge.

**KILL (criterion b).** The `C + R/2` model is **untrustworthy**: its anchor is void, its inputs
(sync-span collective shares) are known-inflated, and a matched same-model head-to-head contradicts its
sign. **No corrected model restores a positive PP prefill edge** — the honest corrected model is
`edge ≈ (real C ≈ 3–5 %) + R/2 − (out-of-loop drain + 12×-worse depth decay)`, which is **≤ 0** on this
cluster. The brief's own caveat ("partially attention-replication-driven — already fixed in current TP, so
today's deep edge is expected smaller") points the same way; `p5` shows it is not merely smaller, it is
**negative at the chunk loop** the model describes.

---

## Gate 0 · Item 3 — BUBBLE-WEIGHTED PREFILL GAIN (corroborating)

For `n` chunks, PP costs `(n+1)·T_stage` vs TP `n·T_tp`; with per-chunk gain `g = T_tp/T_stage − 1`, PP wins
only above **`n ≈ 1/g`** `[DERIVED]`. Current measured TP rates `[MEASURED, owner brief §1]`: **300.1 tok/s
@50 K, 291.5 @105 K cold; ~191 marginal @~490 K.** 2048 rows/chunk. Effective gain
`= (n/(n+1))·(1+g) − 1`.

**Two gain values are shown** — the *model's own optimistic* `g = +14 %/chunk`, and the *calibrated*
`g = −1.3 %/chunk` (item 2, `p5` chunk-loop). Workload weights are the real census distribution
(`docs/benchmarks/big-append-census-2026-10-13/census.csv`, `exo_new_v4` = cluster-side prefill rows;
270 seed calls, 4,290,206 rows) `[MEASURED, DERIVED]` — the only on-disk prefill-length distribution
(`out/calls_all.csv` from the census machine is not present locally; the CSV covers ≥20 K-cloud-miss seeds,
so it **over-represents large prefills** and the weighted figure below is therefore **optimistic for PP**).

| Prefill size | calls | rows | mean rows/call | n (chunks) | eff. gain @ g=+14 % | eff. gain @ g=−1.3 % |
|---|---:|---:|---:|---:|---:|---:|
| <4 K | 178 | 217,517 | 1,222 | 0.6 | **−57.4 %** | −63.1 % |
| 4–13 K | 23 | 160,310 | 6,970 | 3.4 | **−11.8 %** | −23.7 % |
| 13–25 K | 24 | 504,004 | 21,000 | 10.3 | +4.0 % | −10.1 % |
| 25–50 K | 32 | 1,106,879 | 34,590 | 16.9 | +7.7 % | −6.9 % |
| 50–105 K | 6 | 357,923 | 59,654 | 29.1 | +10.3 % | −4.6 % |
| 130–500 K | 6 | 1,435,007 | 239,168 | 116.8 | +13.1 % | −2.2 % |
| >500 K | 1 | 508,566 | 508,566 | 248.3 | +13.6 % | −1.7 % |
| **workload-weighted** | 270 | 4,290,206 | — | — | **+5.98 %** | **−8.36 %** |

- **Crossover** `n = 1/g`: **7.1 chunks ≈ 14.5 K rows** under the model's own +14 % — and **never** under the
  calibrated −1.3 % (PP loses at every length). Below ~14.5 K rows PP is net slower; **178 of 270 real calls
  (66 %) are under 4 K rows** and pay the full bubble.
- **1-chunk follow-up** (the prefix-cache case, ~2 K rows): PP is **1.75× slower** (model) / **2.03× slower**
  (calibrated). A follow-up is the common turn shape; the old +14.1 % @500 K was *bubble-free* and says
  nothing about it.
- `p5` adds a multiplier the brief's model omits: **PP self-selects a 1024-row chunk** (the `//ranks`
  divisor) vs TP's 2048 — so PP's bubble count is **2× n** at equal prompt length, worsening every row above.
- **Cold 50–105 K** (the brief's headline band): model **+10.3 %** at n≈29 — already below the brief's
  +14–18 % and, with the model's ±10 pp uncertainty, indistinguishable from the bar.

**KILL (criterion c).** Even granting the model's own optimistic +14 %/chunk, the **workload-weighted gain is
+5.98 %, below the ~+8 % bar** (because 66 % of the workload is ≤2 chunks and pays the bubble). Under the
calibrated edge it is **−8.36 %**. The brief's "+14–18 % @50–105 K" is the *large-prefill corner only* and
does not survive weighting by the real distribution.

---

## Gate 0 · Item 4 — DEPENDENCY DIRECTION (clean — does not fire)

Method: read the per-forward `SharedState` and the `LayerCache` machinery at `mlx-lm 6cc9c1e87`, plus
`src/exo/worker/engines/mlx/dsv41/load.py` for the refusal and `tp_geometry`.

**Key structural fact** `[MEASURED, model.py:11-16,71-82]`: "Cross-layer sharing flows through a per-forward
`SharedState` … kv sources publish their compressed-KV cache and index-key cache, index sources publish
their top-k selection, the candidate source publishes its block mask, and every layer in between consumes
the most recent value. **Layers run top-down, so every source writes before its consumers read.**" Consumers
resolve their source via `config.py:124-144` — `kv_source_for` / `index_source_for` return
`max(srcs ≤ layer_id)`, i.e. **the most recent source at-or-below**. With a **contiguous ascending
layer-range split**, every source index ≤ every consumer index ⇒ **every cross-layer edge is forward
(stage 1 → stage 2). No backward edge was found**, so **criterion (d) does not fire.**

| # | Element | Produced by | Consumed by | Direction | Shape | Bytes / 2048-row chunk | Overlap-able | Eliminable? |
|---|---|---|---|---|---|---|---|---|
| 1 | `shared.topk_idxs` (index top-k) | index sources 2,8,14,20,24,28,32,36 | ratio>0 layers below the next source | **forward** | `[b,n,512]` int32 | 4.19 MB (one live slot) | yes (needs the prior top-k) | no (consumed in-block) |
| 2 | `shared.candidates` (block mask) | candidate source 20 | index sources 24–36 | **forward** | block-level mask (indexer.py:538 "never allocates `[b,n,nb]`") | small `[ESTIMATE]` | yes | no |
| 3 | `shared.kv_src_cache` → `comp_kv` ref | kv sources 2,8,14,20 | ratio>0 layers 2–39 | **forward** | `[b,⌈tok/ratio⌉,512]` bf16 | **history**, append-only | yes | no |
| 4 | `shared.index_src_cache` → `index_k` ref | index-key owners 2,8,14,20 | index sources below | **forward** | `[b,⌈tok/ratio⌉,128]` bf16 | **history**, append-only | yes | no |
| 5 | `comp_kv` **history** (layer 20, ratio 1) | layer 20 | layers 21–39 attend over it (`attention.py:209`) | **forward** | `[b,tok,512]` bf16 | ~1.0 GiB @1 M `[DERIVED, cache.py:19-29]` | yes (prefix slice) | no |
| 6 | `index_k` **history** (layers 2,8,14,20) | key owners | index sources 24–36 read layer-20 keys | **forward** | `[b,⌈tok/ratio⌉,128]` bf16 | ~0.25–0.5 GiB each @1 M | yes | no |
| 7 | `cache.engram_ids` | `EngramHasher` at fwd start | engram layers 1,14 | **forward** | `[b,capacity]` int64 | history | yes | no |
| 8 | engram hashes | fwd start from `input_ids` | engram layers 1,14 | **forward** | `[b,n,2,(G−1)·H]` int64 | small | yes | **yes** (recompute from token ids locally) |
| 9 | Activation handoff (stage boundary) | stage 1 last layer | stage 2 first layer | **forward** | `[b,n,4,5120]` bf16 (hc-expanded) | **83.9 MB** | yes | no |
| 10 | DSpark draft taps | body layers in `dspark_target_layer_ids` | MTP head | **forward** | `[b,n,dim]` per tap | — (head goes on the last stage) | — | — |
| 11 | LM head | — | — | — | vocab-sharded | — | — | last stage |

**Notes / refinements (recorded; moot given the kill, kept for any future reopen).**
- **The cleanest split is exactly at layer 20** — stage 1 = 0–19, stage 2 = 20–39. Source 20 (kv *and*
  index *and* candidate) then lives in stage 2, so **no source history crosses the boundary at all** and the
  "co-locate layer-20 caches at stage 2" memory cost disappears. The brief's proposed **22/21 split crosses
  the 20–23 consumer group** and *does* require shipping layer-20 state; 20/20 or 24/16 are boundary-aligned.
- **History caches are a cross-chunk concern, not a backward edge.** With chunks pipelined, stage 1 writes
  chunk t+1 while stage 2 reads chunk t; any cross-boundary history read must be a **versioned append-only
  prefix** (the `compress_len` slice), never a shared mutable buffer.
- **Not fully audited (flagged, not claimed clean):** inter-forward *temporal* backward edges — whether any
  high layer persists state (candidate/top-k caches, indexer state, draft-head state) that a *lower* layer
  reads in a **later** forward or chunk. The within-forward ordering is guaranteed by the docstring; the
  temporal ordering was not exhaustively traced. The draft head's interaction with source caches was likewise
  not fully traced. **These are the two items a build would have to close first.**
- **Memory note (possible PP upside, moot here).** Under TP the compressor/index caches are **replicated on
  both ranks** (~3.2 GiB per rank at 1 M bf16 `[MEASURED, cache.py:19-29]`); under PP each stage holds only
  its own layers' caches, so a 20/20 split would **~halve** the cluster total — a real but modest upside that
  does not offset items 1–3.

---

## 1. Closure — what was and was not done

- **Gate 0 run in full** (all four items); three of four fire. Per the pre-registered criteria, **the full
  study was not run** — no boundary-placement modelling, no memory-fit table, no TP-native (SP + overlap)
  pricing, no old-machinery reuse audit, no gain-vs-effort table. Those were contingent on Gate 0 passing.
- **Non-disturbance.** 0 boots; 0 config changes; 0 node-side file edits; **0 model-executing requests**; no
  generation curl. Cluster contact: **none** — all inputs are in-repo code + documents (the live `/state` and
  the census CSV were the only cluster-adjacent artifacts, both already on disk). Live build untouched.
- **NEEDS-GO items (none executed; listed for the record).**
  1. The one measurement that would make item 1 airtight without building PP: on **one** node, time a single
     layer's `switch_mlp` + sharded dense/attention ops at **m=4** at **full width** vs **TP half width**;
     ×40 gives `S_d/2` directly. If ≥ 10 ms, item 1 is decisive beyond inference. **Requires a node-side
     bench (NEEDS-GO).**
  2. Re-measure PP end-to-end on the **current** DSv4.1 build at one depth, to replace the borrowed old-model
     `p5` result with a same-model number. **Requires `DSV4_SHARDING=Pipeline` (a config change / boot —
     NEEDS-GO).** Not needed for the kill: item 1 alone closes it.
- **What would change the answer.** A credible spec-pipelining design that hides the intra-round bubble
  (item 1 escape hatch) **and** a same-model matched PP-vs-TP prefill measurement that reproduces a positive
  chunk-loop edge (item 2). Neither exists; both are high-effort and unmodelled.
- **Relationship to the closed record.** This **upholds** the 2026-08-16 decision (TP for both phases;
  `hybrid-…-design-2026-08-04.md` CURRENT STATE + §107) and **adds** a measured reason it is right for
  prefill too, on this cluster — not merely for the co-residency/concurrency reasons §107 gave. The standing
  owner directive that MTP/DSpark decode goes through TP is **not disturbed**.

## 2. Labels recap

MEASURED: 95 ms/round, 34.6 tok/s, verify ms-by-rows, jaccl medians, collective-share 2.9–5.3 %, 4.46 GB
expert slice, 497 GB/s, `p5` head-to-head (366.5/277.0, 377/372, +32.3 %, −1.3 %, 55.8 s drain, 12× decay),
census distribution, checkpoint config, code facts (source lists, ratios, engram ids, `SharedState` order).
DERIVED: `C_d` ms, `S_d/2` range, `2h` bytes, all item-3 table numbers, weighted +5.98 % / −8.36 %, crossover
7.1 chunks/14.5 K rows, 1.75×/2.03×, 3.2 GiB caches. BOUND: `C_d ≤ 5 ms` (direct measurement, not 81×p95).
MODELED: the `C + R/2` model itself and the +14–18 % it produces. ESTIMATE: candidates mask size, `S_d/2`
upper ~40 ms (adds sharded dense/attention), the handoff size (~164 KB/round).
