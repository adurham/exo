# DeepSeek-V4.1-Flash — alternative quant/variant survey for the 2× M4 Max exo/MLX cluster

**Date:** 2026-10-11 · **Author:** Phase-20 variant survey (web + HF API only) · **Scope:** find any published artifact that suits this cluster *better* than the current incumbent `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`.

**Constraints honored:** Hugging Face Hub REST/raw API + web only; **no cluster contact** (live deploy running on the nodes); **no model downloads** (metadata/API only, sizes read from the HF tree API `lfs.size`); no GUI automation; work isolated to worktree `/private/tmp/p20-variant-survey`. All sizes are exact byte sums from `GET /api/models/<id>/tree/main?recursive=1`, reported in both GiB and decimal GB.

---

## 1. Verdict (verdict-first)

**NO — there is no artifact today that fits this hardware *better* than the incumbent.** The incumbent is the best-fitting published artifact for a 2×128 GB unified-memory box that must keep the uncensored behaviour.

- Every artifact that **fits** the memory budget (≤ ~205–215 GB total) is either
  (a) the **same EXL3-trellis format** as the incumbent → identical format/decode cost, no hardware win, or
  (b) **not uncensored** (a plain quant of the base model) → behaviour change, or
  (c) **architecture-changed** (REAP expert-pruned) and shipped in a **non-standard packed layout** that the forked dsv41 engine cannot load as-is.
- The only artifacts in the **native-format (decode-win) class** — MLX affine 2-bit g64 — all **exceed the budget** (238.8–259.7 GB). This is the *exact* result the fork already measured: **Q1d GATE 1** found native `q4g64` is faster (+32.6 % wall / +43.6 % GPU) but **no native arm fits 128 GiB**; **Q1e** closed the q3 arm as memory-blocked on the live wired limit. **A published native-format checkpoint does not change that arithmetic.**
- **Conditional alternative, only if the user trades away a constraint:**
  - Trade **uncensored → base**: `Krishando/DeepSeek-V4.1-Flash-EXL3-2.86bpw` (213.8 GB, Hessian-traced quality-first EXL3) or `jeet0733/...EXL3-C4h36` (210.6 GB). Same format cost as incumbent → **no hardware win, only a quality/behaviour swap.** Not "better suited to the hardware."
  - Trade **uncensored + no-pruning → base + REAP-pruned + custom loader**: `rapid-mlx/DeepSeek-V4.1-Flash-REAP-2bit-MLX` (212.9 GB) is the *only* native-format artifact that fits — a decode-side win class, but it prunes 48/384 experts per layer and needs the publisher's experimental packed loader.

**Bottom line: keep the incumbent.** The hardware-limiting lever remains the MoE ragged small-M shape (below) and the memory budget, not the availability of a different checkpoint.

### Format-cost facts that apply to every row (stated once)

- **(i) MoE ragged shape — UNCHANGED for all candidates.** 40 layers, 384 routed experts, top-6, ~37 rows/expert at 2048-token chunks is **architecture** and is identical in every V4.1-Flash variant. No checkpoint swap can change it.
- **(ii) Decode/verify small-M format cost.** EXL3 3-bit trellis decode is the **worst case measured**. Native MLX affine/mxfp4 arms measured **~2.47× faster on the same small shapes** in the fork's dense-slice tests (Q1e-era). → Native-format candidates (MLX affine / mxfp4) are a **decode-side win class**; EXL3-trellis candidates are **not** (same cost as incumbent).
- **(iii) Prefill — do not promise gains.** The in-fork dense-slice re-quant (`DSV41_DENSE=affine6`) moved prefill **~0**. No candidate here should be expected to move prefill; treat prefill as flat.
- **(iv) Engram caveat (fit math).** The base checkpoint is 510.3 GB because ~189–202 GiB of it are the two **Engram** n-gram lookup tables (layers 1/14) — data, not weights, never matrix-quantized. **Every EXL3/MLX pack excludes them**; they are fetched from the base (shards 47/48, ~95 GiB each) or NVMe-offloaded at serve time. Engram is byte-identical across all variants and its handling is unchanged by any swap → it does **not** change the format question, but it is why the "~197 GiB" packs serve on 128 GiB nodes.

---

## 2. Ranked shortlist

Ranked by **net suitability for this specific hardware** (fit ∧ format-decode class ∧ loadability ∧ behaviour). "Δ vs incumbent": incumbent = `dealignai/...UNCENSORED-EXL3-2.9bpw`, **196.21 GiB / 210.7 GB**, EXL3 trellis mul1 avg 2.9 bpw, head 6-bit / MTP 4-bit, sha `8a27b35f…`, 2026-09-17.

| # | Repo id (revision date) | Total (GiB / GB) | TP2 per-rank | Format family | Loads in fork as-is? | Honest expected impact vs incumbent | Risk |
|---|---|---|---|---|---|---|---|
| 0 | **`dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`** (2026-09-17) | 196.21 / 210.7 | ~98 / 105 | EXL3 trellis 2.9 bpw | **Yes — incumbent** | baseline | none (uncensored kept) |
| 1 | `manateelazycat/...UNCENSORED-TensorFold-EXL3` (2026-10-04) | 196.21 / 210.7 | ~98 / 105 | EXL3 trellis 2.9 bpw | Yes (standard EXL3) | **≈ zero** — byte-identical footprint; same trellis cost; uncensored is the dealignai abliteration, re-packed for TensorFold | mirror/repack; no reason to switch |
| 2 | `rapid-mlx/...REAP-2bit-MLX` (2026-10-05) | 198.31 / 212.9 | ~99 / 106 | **MLX affine 2-bit g64** (native) | **No** — packed/stacked layout; publisher's experimental loader only | **Decode-side win class** (~2.5× at tiny M per dense-slice precedent); prefill flat. But **336/384 experts retained** → real architecture change | **base model (no uncensored)**; REAP prune (mean retained routing mass 99.99 %, worst-layer 99.86 %); text-only (vision+MTP omitted); non-standard loader |
| 3 | `Krishando/...EXL3-2.86bpw` (2026-10-11) | 199.14 / 213.8 | ~100 / 107 | EXL3 trellis 2.86 bpw | Yes (standard EXL3, `quant_method: exl3`, mcg) | **No hardware win** — same trellis cost; denser/quality-optimized (Hessian-traced on ~1.03M tokens); only routed experts compressed | **base (no uncensored)**; at the top edge of the fit budget |
| 4 | `jeet0733/...EXL3-C4h36` (2026-09-19) | 196.18 / 210.6 | ~98 / 105 | EXL3 trellis | Yes (assumed standard EXL3) | **No hardware win** | **base**; undocumented recipe (`C4h36` unexplained) |
| 5 | `jayleaton/...EXL3-2.8bpw` (2026-10-09) | 189.83 / 203.8 | ~95 / 102 | EXL3 trellis, **per-expert ragged widths 2–5 bit** | **No** — "cannot be loaded by stock exllamav3 or vLLM"; needs TensorFold Zig engine | Smaller & higher measured fidelity (KL 0.0469 vs Mia 2.9's 0.0594; MMLU-200 88.5 % vs 87.5 %) but needs a different engine | **base**; engine swap required; excluded Engram |
| 6 | `Mia-AiLab/...EXL3-2.9bpw` (2026-09-13) | 196.21 / 210.7 | ~98 / 105 | EXL3 trellis 2.9 bpw | Yes | **≈ zero** | **base** — the reference pack the incumbent was built against |
| 7 | `mlx-community/...MLX-2bit` (2026-09-11) | 222.43 / 238.8 | ~111 / 119 | **MLX affine 2-bit g64** (native) | Yes (standard affine) | Decode-side win class | **Does not fit** (+~24 GB over); **base** |
| 8 | `TensorFold/...MLX-2bit-MTP` / `0xSojalSec/...MLX-MAC` (2026-10-01 / 2026-09-10) | 222.43 / 238.8 | ~111 / 119 | MLX affine 2-bit g64 | Yes | Decode-side win class | **Does not fit**; base |
| 9 | `orcarouter/...Uncensored-MLX` 2bit (2026-10-02) | 241.91 / 259.7 (repo holds 2/3/4-bit: 241.9/405.9/516.1 GiB) | ~121 / 130 | **MLX affine** (native) | Unknown — **repo is gated** (auth required) | Would be the *ideal* decode-win + uncensored combo | **Does not fit** (+~49 GB); **gated**; 3-bit/4-bit dirs far larger |
| 10 | `Mia-AiLab/...EXL3-3.0bpw` (2026-09-12) | 204.21 / 219.3 | ~102 / 110 | EXL3 trellis 3.0 bpw | Yes | No win | **Over budget** (~+8 GB); base |
| — | `deepseek-ai/DeepSeek-V4.1-Flash` (official, 2026-10-01) | 475.27 / 510.3 | — | **FP8** e4m3fn block[32,32] + FP4 routed experts | Different loader path | — | **Does not fit** (2.4× budget) |

### Also surveyed (for completeness — none fit, none beat the incumbent)

- **Uncensored, too big:** `dealignai/...UNCENSORED-FP8` 475.27 GiB (the canonical abliterated source; **510.3 GB — no**); `dealignai/...ABLITERATED-FP8` (redirect, 2026-09-22); `Solstice-AI/...UNCENSORED-MXFP4` 283.02 GiB (303.9 GB — no); `MomonV/...UNCENSORED-NVFP4` 491.10 GiB; `UltimateIntent/...Abliterated-EXL3-3.51bpw` 428.49 GiB; `com-kotobalabs/...Uncensored-EXL3-2.0bpw` 333.53 GiB (surprisingly *larger*, not smaller — no); `satgeze/...EXL3-2.0bpw-Ablit-EngramQ4` 241.98 GiB / 259.8 GB (abliterated but ships Engram-Q4 inline → over); `pqhaz/...heretic` 477.17 GiB (directionally-ablated, **native FP8** 510 GB); `distributedcognition/...abliterated`, `s-zaizen/...Abliterated`, `securepeak/...Abliterated` (713 GiB!), `mIn-MAXsystem-Htz/...Abliterated`, `schwyzquant|Openintelligent123|…-UNCENSORED-FP8` (all ≤510 GB FP8). `neko-legends/...uncensored-engram-4x-spark` 188.83 GiB is **Engram shards only**, not a servable model.
- **Base, too big / exotic:** `gnahzhang/...LSQ3-G128-MLX` 420.53 GiB (package; native 3-bit LSQ but oversized + needs pinned kernel); `nvidia/...NVFP4` 491.10 GiB; `intel-ai|amd Quark-MXFP4` ~474 GiB; `pipenetwork/...MLX-mixed-4_8bit` 398.25 GiB; `Jundot/...oQ3e/oQ4e-mtp` 330.9/402.4 GiB; `OpensourceWTF/...MTPLX-streaming-q2` 376.25 GiB; `GreenBitAI/...4bit-paged` 406.10 GiB; `drowzeys/...TR3-Hybrid` 409.89 GiB; `0xSero/...Spark` 430.19 GiB; `littlecedar/...2.9bpw-with-engram` 385.05 GiB; `meshapplied/...Ternary-Latest` 131.57 GiB / 141.3 GB (small but **ternary overlay, needs Goethe engine**, base); `wenzhouwu/YoungAi-...` 105.98 GiB (vector-quantized, exotic, base); `bertholomus/...TensorFold-TP2-2xGB10` 0.62 GiB (**recipe only**, not weights).
- **GGUF-only — *requires conversion; engine does not load GGUF* (low priority):** `antirez/deepseek-v4.1-flash-gguf` 824.5 GiB; `audreyt/...Abliterated-GGUF` 681.2 GiB; `dongnhdev/...Q2-EngramQ4-GGUF` 249.0 GiB; `pyrodog/...UNCENSORED-DwarfStar-Q2` 341.5 GiB (Gated? no — downloadable); `mxxm-t|kernelpool|JigSawPT|smalinin` GGUF; `Adolphsson/...UNCENSORED-GGUF` 0.91 GiB (pointer/partial only).
- **No fitting artifact found from:** `mlx-community` (only 2-bit/4-bit + DSpark-drafter), `inferencerlabs` (V4.1 non-Flash only), `Vontra` (org empty/renamed). A ~3–3.5 bpw native MLX quant of V4.1-Flash **does not currently exist** in `mlx-community` or elsewhere.

**Sweep method:** HF search API across `search=DeepSeek-V4.1-Flash`, `=V4.1-Flash`, `=DeepSeek-V4.1`, plus filters `mlx`, `mxfp4`, `uncensored`, `abliterated`; author sweeps (`dealignai` full org, `mlx-community`, `Vontra`, `inferencerlabs`); sorted by `lastModified` (~120 distinct repos enumerated). Sizes = exact byte sums from the tree API. One search tool glitch (a `jq` parse error on one repo) was retried with a different call and resolved; no value was fabricated.

---

## 3. What would have to exist to beat the incumbent (ideal-artifact spec)

Nothing published today clears all of these; the winning artifact would be:

1. **Uncensored / abliterated** (preserve the user's deliberate choice) — same weight-level abliteration class as the dealignai finetune.
2. **Native MLX format** — affine **g64 at ~2.4–2.8 bpw** (or mxfp4), i.e. the *decode-side win class* proven at ~2.47× on ragged small-M in the fork's dense-slice tests. This is the only lever that touches the measured bottleneck.
3. **Total resident footprint ≤ ~198 GiB / ≤ ~212 GB** (TP2 ≈ ≤106 GB/rank within the ~125.8 GB wired limit), with the ~189–202 GiB **Engram tables kept out of resident memory** (NVMe/streamed — unchanged from today).
4. **Standard safetensors layout** loadable by the fork's existing `dsv41` native path (`mx.quantize`/`gather_qmm` affine, q8g64 shared experts, affine6 dense) — no exotic packed/stacked layout, no third-party engine.
5. **No expert pruning** (keep all 384 routed / top-6) — the ragged shape is architecture and must stay byte-equivalent.

**If that artifact existed**, expected impact: decode-side win ≈ the dense-slice precedent (~2× order at tiny M, **medium-high confidence**); **prefill ≈ flat** (dense requant precedent, **low confidence of any gain**); memory fit within budget (**high confidence**); behaviour preserved (**high confidence** — it is the same abliteration re-quantized). The nearest published thing to this is `orcarouter/...Uncensored-MLX` 2-bit — but it is **gated and ~259.7 GB (43 GB over budget)**, and has no 2-bit sibling that fits. The nearest *fitting* native-format thing is `rapid-mlx/...REAP-2bit-MLX` — but it is base (not uncensored), REAP-pruned, and non-standard.

**Cheapest real path to the ideal** (not a Hub download): in-fork **re-encode** of the *existing* uncensored resident weights is blocked by memory (Q1d/Q1e); the only routes are (a) the streaming/tiering memory stack the fork already flagged as the reopen condition, or (b) publishing a *≤198 GiB native-format quant of the dealignai abliteration* (no such artifact exists on the Hub as of 2026-10-11).

---

## 4. Sources (Hub metadata, retrieved 2026-10-11)

| Repo id | lastModified | Exact size (bytes) | Format (from card/config) |
|---|---|---|---|
| `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw` | 2026-09-17 | 210,677,064,447 | EXL3 trellis, mul1, 2.9 bpw, head 6 / MTP 4 |
| `manateelazycat/DeepSeek-V4.1-Flash-Uncensored-TensorFold-EXL3` | 2026-10-04 | 210,677,064,320 | EXL3 2.9 bpw (TensorFold repack) |
| `rapid-mlx/DeepSeek-V4.1-Flash-REAP-2bit-MLX` | 2026-10-05 | 212,937,714,756 | MLX affine 2-bit g64; REAP 336/384 |
| `Krishando/DeepSeek-V4.1-Flash-EXL3-2.86bpw` | 2026-10-11 | ~213.8 GB (tree sum 199.14 GiB) | EXL3, `mcg`, 2.86 bpw |
| `jeet0733/DeepSeek-V4.1-Flash-EXL3-C4h36` | 2026-09-19 | ~210.6 GB (196.18 GiB) | EXL3 |
| `jayleaton/DeepSeek-V4.1-Flash-EXL3-2.8bpw` | 2026-10-09 | 203,831,795,286 | EXL3 ragged 2–5 bit; TensorFold Zig engine |
| `Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw` / `-3.0bpw` | 2026-09-13 / 09-12 | 210.7 / 219.3 GB | EXL3 2.9 / 3.0 bpw |
| `mlx-community/DeepSeek-V4.1-Flash-MLX-2bit` | 2026-09-11 | 238,834,784,075 | MLX affine 2-bit g64 |
| `TensorFold/DeepSeek-V4.1-Flash-MLX-2bit-MTP` | 2026-10-01 | 238,834,784,075 | MLX affine 2-bit g64 |
| `orcarouter/DeepSeek-V4.1-Flash-Uncensored-MLX` | 2026-10-02 | 1,249,762,110,163 (2/3/4-bit) | MLX affine (gated) |
| `deepseek-ai/DeepSeek-V4.1-Flash` | 2026-10-01 | 510,313,368,913 | FP8 e4m3fn block[32,32] + FP4 experts |
| `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8` | 2026-09-11 | ~510 GB | FP8 native + abliteration |
| `Solstice-AI/DeepSeek-V4.1-Flash-UNCENSORED-MXFP4` | 2026-09-30 | ~303.9 GB | MXFP4 |
| `satgeze/…EXL3-2.0bpw-Ablit-EngramQ4…` | 2026-10-04 | 259.8 GB | EXL3 2.0 bpw + Engram Q4, abliterated |
| `com-kotobalabs/Deepseek-v4.1-Flash-Uncensored-EXL3-2.0bpw` | 2026-09-24 | ~358 GB | EXL3 2.0 bpw |
| `meshapplied/DeepSeek-V4.1-Flash-Ternary-Latest` | 2026-10-10 | 141.3 GB | ternary-g128-hadamard-gptq (Goethe) |

*(~120 further repos enumerated; all classified into the "also surveyed" buckets above. Full raw JSON retained in this survey's session, not committed.)*

**Precedent cited:** `docs/PERFORMANCE_HISTORY.md` — *Q1d GATE 1 (experts requant)* (native q4g64 +32.6 % wall / +43.6 % GPU but **no native arm fits 128 GiB**) and *Q1e* (q3 arm memory-blocked on the live wired limit; reopen only via streaming/tiering or a ≤3.7 bpw native format). These are the on-box measurements behind the "decode-side win class / prefill ~flat" statements.

**No-cluster-contact affirmation:** this survey was produced entirely from the Hugging Face Hub REST/raw API and the local repo history. No request was made to any exo/cluster node; no model files were downloaded; only Hub metadata (tree listings, configs, READMEs) was read.
