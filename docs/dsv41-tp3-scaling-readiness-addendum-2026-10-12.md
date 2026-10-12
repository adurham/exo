# DSv4.1 — TP3 RE-MODEL ADDENDUM (2026-10-12): is a 3rd node worth engineering *now* that the indexer split is live?

**Date:** 2026-10-12 · **Author:** p20 campaign (delegated, offline)
**Live tree:** exo `deploy/next23b-qsplit @ 60008939d` + mlx-lm `perf/dsv41-indexer-qsplit @ 6a5c78f`, 2-rank tensor-parallel (`worldSize=2`, jaccl/RDMA, one TB5 cable pair).
**Updates:** `docs/dsv41-tp3-scaling-readiness-2026-10-11.md` (branch `p20/tp3-scaling-readiness @ 11fe07ff7`).
**Labels:** every number is `[MEASURED]`, `[DERIVED]`, `[MODELED]` or `[BOUND]`. **No 3-node cluster exists — every `w=3` figure below is MODELED.**

---

## VERDICT

- **Performance (revised).** Over the **current live 2-node baseline**, a 3rd node is modeled at **≈ +24 % to +39 % prefill (centre ≈ +30 %)** across 25–130 K — a sign flip from the original study's **flat-to-worse** — *gated on a small query-axis pad* that lets the now-live indexer split engage at `n % 3 != 0`.
- **Support (unchanged in kind).** Still requires the TP3 readiness package — **≈ 15–32 eng-days**, critical path = transport N>2 / ARQ (**3–6 d**) — **plus the pad**. The indexer split removed the dominant *performance* blocker, not the transport / geometry / launcher prerequisites.

## 0. Why this addendum

The indexer query-band split (`DSV41_INDEXER_QSPLIT`) is now **live-proven on the pair**: `deploy/next23b-qsplit @ 60008939d` (+ fixed mlx-lm `6a5c78f4`) passed — token identity **5/5** (incl. the round-1 `identity_3` divergence) and live A/B **+10.5 % @50 K / +11.9 % @105 K rows/s**, battery CLEAN (`docs/dsv41-indexer-qsplit-live-ab2-2026-10-12.md`; PH + results #2 on `main @ 06d134c95`). The split bands the indexer's **query rows** across ranks; its reconstruction is an **N-way zero-padded `all_sum`** already parametric in rank count (`indexer.py:196-214`). The original study's verdict — *flat-to-worse at `w=3`* — was **dominated by the replicated, O(context) indexer** (3.6 % @16 K → ~43 % @340 K). With the indexer now splittable, that dominant term changes sides. This addendum re-models the band.

## 1. UPDATED replicated-vs-splittable partition (w2 → w3) @ ~100 K

**MOVED to splittable:** the **indexer** (`Indexer` score + top-k + emit) = **31.1 % @100 K** `[MEASURED]` (measure pack Task B). Under the split it is band-shardable exactly 3 ways (`rows = n // world`); source/consumer layers stay band-consistent (`indexer.py:196-214`; buildnote §"N-node implications").

**REMAINS replicated** (shares of the current per-rank wall @100 K):

| replicated term | share @100 K | source |
|---|---:|---|
| hyper-connections (`hc_*`) | **5.2 %** | `[MEASURED]` measure pack Task B; scope doc §2 |
| RMSNorm (attn/ffn/final/q/kv) | **1.7 %** | `[MEASURED]` |
| `attn.wq_a` (shared q latent, no head axis) | **~0.55 %** | `[DERIVED]` from dense-qkv 4.3; scope doc §3a |
| `attn.wkv` (single MQA KV head) | **~0.22 %** | `[DERIVED]` same |
| router gate (`ffn.gate` score + topk) | **0.5 %** | `[MEASURED]` (MoE routing gate) |
| compressor (CSA/HCA) | **0.08 %** | `[MEASURED]` |
| **KV keys** (rope + fp8/fp4 fake-quant + window/comp cache) | **not separately resolved** — folded into `sdpa` / dense-qkv | `[BOUND]` scope doc §2 |
| **embeddings** `[129280,5120]` = **1.318 GB bf16** `[DERIVED]` | **0 compute** (row-indexed gather) | memory, not wall |
| `dispatch / GPU-idle` (~9.6 % = WALL−Σ) | treated fixed | `[BOUND]` |

**Remaining replicated fraction of the current live wall** `[DERIVED]`: **9.7 % @100 K**, **9.4 % @130 K** (11.0 % @25 K, 10.4 % @50 K); **≈ 6.8 % projected @340 K** (indexer 43 %). Sources: measure pack Task B fold; scope doc §2/§4 (the non-indexer row-local replicated set = **8.17 %** = HC 5.2 + norms 1.7 + wq_a 0.55 + wkv 0.22 + router 0.5). ⇒ post-split the **splittable fraction ≈ 0.90** of the wall @100 K.

## 2. RE-MODELED TP3 band

Framework (the study's): per-rank chunk wall `T(w) = R + S/w`; speedup `= T(2) / T(3)`. Class ratios: **splittable = 2/3** (w2 full/2 → w3 full/3); **replicated = 1**; **indexer = 1 (full) if the split does NOT engage, `/3` with the pad**. Baseline = **current live 2-node** (qsplit ON; the indexer is already banded at w2).

| depth | indexer share `[MEASURED/MODELED]` | **BEFORE** (idx replicates at w3) `[MODELED]` | **AFTER** (idx splits 3-way, + pad) `[MODELED]` |
|---:|---:|---:|---:|
| 25 K | 13.9 % | **1.25×** (+24.6 %) | **1.42×** (+42.2 %) |
| 50 K | 21.9 % | **1.16×** (+15.6 %) | **1.43×** (+42.6 %) |
| 100 K | 31.1 % | **1.06×** (+5.9 %) | **1.43×** (+43.1 %) |
| 130 K | 33.9 % | **1.03×** (+3.1 %) | **1.43×** (+43.2 %) |

- **BEFORE reproduces the study's verdict.** With no pad the indexer (its depth-dominant replicated term) cannot split, so the band goes **flat at depth (+3 % @130 K)** — the published "flat-to-worse". (Study's own before-band: hard 1.04–1.19×, cheap 0.78–0.86×, centre +5–12 % hard `[MODELED]`.)
- **AFTER** applies the w3 **collective-growth + slowest-of-3 straggler haircut** `1/(1 + c(g−1))` with `c ∈ [0.10, 0.15]` (collective share of the wall `[BOUND]` 1–24 %) and `g ∈ [1.3, 2.0]` (N>2 routes large `all_reduce` to reduce-scatter + all_gather: +33 % wire, 2→4 steps) → ×0.87–0.97 ⇒ **≈ +24 % to +39 % (centre ≈ +30 %)**.
- **Calibration** `[MEASURED]`: the live w2 delta (`1/(1 − α/2) = 1.119 @105 K`) implies a w2 indexer share **α ≈ 0.21**; the harness fold gives 31.1 % — the same **unreconciled divergence** the study flagged. The model brackets it (0.21–0.31); at w3 the indexer is halved again vs today (full/2 → full/3).
- **Decode:** unchanged — the split is length-gated (`n > 16`), decode stays replicated → **flat ± a few %**, small regression risk from extra round-trips `[MODELED]`.
- **Residual uncertainties** `[MODELED]`: (i) **sync growth at w3** — 1.3–2× per collective on a 1–24 % `[BOUND]` base; (ii) **slowest-of-3** `E[max of 3]` ⇒ +0.29σ per collective; (iii) **`sdpa` (13.6 %)** may be partly *replicated* (per-row KV gather is bandwidth-bound) → its split share is a **floor**; (iv) **uneven heads 22/21/21** adds ~5 % imbalance on the attention classes; (v) **±10 %** per-class resolution (measure pack closure −9.6 %).
- **What the original study would have said:** with the indexer splittable, the study's own `T = R + S/w` yields the AFTER row — it would have flipped the verdict from flat-to-worse to clearly positive; it said flat-to-worse *only because* its dominant replicated term (the indexer) could not split.

## 3. UPDATED w3 requirements + effort (was ≈ 14–30 eng-days)

| # | item | was | **now** | note |
|---|---|---|---:|---|
| R1 | **transport N>2 — ARQ / reliable layer** (`size_==2` gate, *throws* for N>2; `mesh_impl.h:706-709,1171-1175,1590`) | 3–6 d | **3–6 d** | **CRITICAL PATH**, unchanged |
| R2 | **w3 raise tensors** — `_block_bounds` `n % (8·world)` (`attn.wq_b` 2048t, `attn.wo_b` 512t, `head` 8080t; `exl3_build.py:316-326`) + uneven **22/21/21** heads & 3-3-2 groups (64 %3, 8 %3) + vocab `//world` floor (129280 %3 = 1) | 3–11 d | **3–11 d** | geometry, unchanged |
| R3 | **launcher / topology** — ~15 two-node hardcodes (`start_cluster.sh`) + 2 cables/rank + 2 distinct HCAs (6 QPs, `max_qp=3`) | 4–8 d | **4–8 d** | + cables |
| R4 | **NEW — query-axis pad** so the qsplit engages at `n % world != 0` (prod chunks **2048 %3 = 2**, **128 %3 = 2**); fence at `indexer.py:194`; the N-way `all_sum` already supports any N | — | **1–2 d** | buildable offline |
| R5 | **NEW — 3-node validation round**: 5-prompt token-identity gate + quick quality battery on 3 ranks (mirrors the qsplit round) | — | **1–2 d** | needs the hardware (+1 boot) |
| R6 | placement/API predicate, park/coord N-generalization, 3-node transport + geometry tests (suite today covers only 3-node *rejections*) | ~5–13 d | **~5–13 d** | offline |
| | **total** | | **≈ 15–32 eng-days** | critical path **R1 (3–6 d)** |

## 4. Pre-purchase vs hardware-only validation

- **Validatable pre-purchase (offline, no hardware):** R2 geometry / divisibility math, uneven-head layout, vocab-floor fix; **R4 pad + its `w=3` unit tests** (CPU-only; the shipped qsplit suite already exercises a `world=3` case at 0/8); R6 placement predicate, launcher hardcode list, park/coord generalization, 3-node transport unit tests.
- **Only validatable with the hardware:** HCA/QP fit (2 distinct HCAs for 6 QPs), 3-node direct-link route/relay bring-up, the true MLX-level `all_gather` cap, end-to-end 3-rank `mx.distributed.init` correctness + hang-mode, **R5** the 3-node token-identity / quality battery, and the live 3-node A/B that would replace this model's calibration.

## 5. Owner bottom line

**Support** for a 3-node setup still requires the TP3 readiness package — updated **≈ 15–32 eng-days**, critical path the **transport N>2 (ARQ) generalization at 3–6 d** — **plus** the new **query-axis pad (1–2 d)** so the now-live indexer split engages at `w=3`; the indexer work is **necessary, not sufficient**. **Performance** is the revised **MODELED** band: **≈ +24 % to +39 % (centre ≈ +30 %) prefill** across 25–130 K over today's live 2-node baseline — a sign flip from the original study's flat-to-worse, which held *only because* its depth-dominant term (the replicated indexer) could not split. **Honest caveat:** this re-model is **MODELED against a MEASURED 2-node calibration** (+10.5 % @50 K / +11.9 % @105 K, token identity 5/5); **no 3-node hardware exists**, so the band — and the w3 sync / straggler / decode terms — are projections to be confirmed by an R5 validation round on real hardware.

## Provenance

- **OFFLINE: 0 boots, 0 node contact, 0 cluster changes.** Sources: qsplit live A/B #2 (`docs/dsv41-indexer-qsplit-live-ab2-2026-10-12.md`, `main @ 06d134c95`); qsplit buildnote N-node section (`docs/dsv41-indexer-qsplit-buildnote-2026-10-11.md`, `p20/dsv41-indexer-qsplit @ 60008939`); attention-sharding scope (`docs/dsv41-attention-sharding-scope-2026-10-11.md`, `p20/attention-sharding-scope @ 0adf3f0d`); prefill measure pack (`docs/dsv41-prefill-measure-pack-2026-10-11.md`); original TP3 study (`docs/dsv41-tp3-scaling-readiness-2026-10-11.md`, `p20/tp3-scaling-readiness @ 11fe07ff7`).
