# DSv4.1 — native-experts FIT & QUALITY scoping memo (2026-10-11)

Author: Phase-20 PM (delegation). Branch `p20/native-experts-fit-scope` (worktree
`/private/tmp/p20-fit-scope`, off `origin/main` `af3d48dd6`). **OFFLINE DESIGN ONLY — no build,
no node contact, no requant started.** Inputs: the P20 prefill measurement pack
(`p20/prefill-measure-pack` @ `302421ebe`, doc `docs/dsv41-prefill-measure-pack-2026-10-11.md`),
the MoE autopsy + secondary (`p20/dsv41-moe-autopsy` @ `9032dd4ed`), the Q1d/Q1e raw on
`deploy/phase20-campaign` (`git show 0808e3507…:<path>`), and a read-only engine code pass
(fingerprint/park/loader) at `origin/main`.

**Question this memo answers:** given that native MLX-affine experts beat the EXL3 2.9 bpw
trellis kernel by **22.1–23.0 % WALL** on the expert block at real prefill routing (pack Task A,
≥20 % gate, thin +2 pp margin), can *any* native arm be made to FIT the 2×128 GiB node, what
would it buy e2e, what quality gate would it have to clear, what engine work is needed, and is
any of it worth doing?

---

## 0. VERDICT-FIRST — recommended decision path (details in §1–§4)

1. **The lever is real at the block, but the *e2e* ceiling is small and NO native arm fits
   today.** Amdahl: expert block ≈ **33 %** of the chunk (pack Task B) × 22 % block win ⇒
   **≈ 7.3 % e2e** *if every MoE layer went native*. Nothing goes native today.
2. **The fit and the quality are in direct tension.** The quality-acceptable arm (q4g64, cos
   **0.9881**) is **+54.6 GB/rank** over EXL3 and cannot fit more than ~7 of 40 layers
   (≈ **1.3 % e2e**). The arm that could fit by a knife-edge (q3g128, +12.2 GB) is the
   **lossy** one (cos **0.9423**).
3. **RECOMMENDED (ordered, bounded): DO NOT spend engine days yet.**
   - **Step 1 (OWNER, days=0):** pick the fit route — (a) wired-limit raise *with risk analysis*,
     (b) shrink the demand (demand-hygiene below), or (c) accept a ≤3.24 bpw quality-risky format.
     This is the gate on everything else.
   - **Step 2 (bounded, ~1–2 offline days):** measure the demand-hygiene reclaims (§1.2 —
     DSpark-taps retention is the big one, plausibly ≈3 GB at 102K). If that reclaims ≥0.5–1 GB,
     the **all-40-layer q3g128** arm (currently over by only 0.4 GB) flips into fit, and a *partial
     q4g64 mix* widens.
   - **Step 3 (OWNER):** only if a fit route exists, run the quality gate (§2) — source decision +
     full-model two-node battery. **q3g128's 0.9423 is an owner judgment** and is not assumed
     acceptable here.
   - **Step 4 (only after 1–3):** the engine work (§3, **~10–16 eng-days**) — a moderate, well-bounded
     change (the native forward/module already exists), dominated by the fingerprint/park coupling.
   - **If no fit route survives Step 1/2: close as "does not fit / not worth it"** at the current
     128 GiB memory + production wired limit. The honest read: even fully fitted, this lever is
     **~6–7 % e2e** for a quality-gated re-quant, ~10–16 engine-days, and (route a) a wired-limit
     risk — a **marginal** trade against the ~1 GB-of-margin operating point.

---

## 1. FIT OPTIONS — exact byte math vs realistic headroom

### 1.0 The exact baseline (per rank, TP=2 intermediate-width slice) — all verified

| quantity | value | source |
|---|---|---|
| per-rank routed-expert weight count | `N_rank = 271,790,899,200` | q1d_memfit.py |
| **EXL3 experts/rank** | **98.253 GB** (91.505 GiB) = trellis 97.675 + signs 0.578 | q1d-memfit-and-fetch.md |
| — EXL3 layer split | **k=2 on layers 18–22 (5 layers) = 8.566 GB → 1.713 GB/layer; k=3 on the other 35 = 89.687 GB → 2.562 GB/layer** | same |
| non-expert **REST**/rank | **10.710 GB** (MTP-ffn 3.408 + replicated 6.499 + shared 0.429 + MTP-other 0.187) | same |
| KV @91K | 0.297 GB (slope 3200 B/token) | same |
| **W_live (wired limit)** | **120000 MiB = 125.829 GB = 117.19 GiB, BOTH nodes** | q1e a2_nwpeak.json; `start_cluster.sh:1474` `DSV4_WIRED_LIMIT_MB` |
| incumbent live weights (`active`) | 104.7 GB; resident peak 114.0 GB; MLX-allocator peak 124.833 GB (≈116.26 corrected, ×1.0737 over-report) | q1e a2_nwpeak.json |

### 1.1 MLX affine byte model — **bias INCLUDED** (the Q1e correction)

`mx.quantize` affine emits fp16 scales **and** fp16 biases ⇒ **eff bpw = bits + 2·16/gs**
(measured on real weights, `q1e/a3_packing.json`). Native bytes depend only on shape, so they are
**uniform across layers** (unlike EXL3, which is k=2 on 5 layers).

| arm | eff bpw | experts/rank | GB/layer | **Δ vs EXL3 layer** (k=3 / k=2) |
|---|---:|---:|---:|---:|
| **EXL3** 2.9 bpw | ~2.9 | **98.253** | 2.562 / 1.713 | — |
| **q4g64** | 4.50 | **152.883** | 3.822 | **+1.260 / +2.109** |
| q4g32 | 5.00 | 152.88→*169.9* | 4.247 | +1.685 / +2.534 |
| q5g64 | 5.50 | 178.36→*186.9* | 4.671 | +2.109 / +2.958 |
| **q3g64** | 3.50 | **118.909** | 2.973 | +0.411 / +1.260 |
| **q3g128** | 3.25 | **110.416** | 2.760 | **+0.198 / +1.047** |

> **CORRECTION carried into this memo (important).** The P20 pack / early Q1d headline quotes
> q4g64 experts = **144.39 GB = "+46 GB/rank over EXL3"**. That used the **wrong 4.25 bpw** (bias
> omitted). Q1e corrected it to **4.5 bpw ⇒ 152.883 GB/rank = +54.6 GB over EXL3** (the doc's own
> per-layer footprint, 3.82 GB, already encoded 4.5). **Use +54.6, not +46.** Every native arm is
> therefore *further* over than the pack's headline suggests.

### 1.2 "Headroom" — what the ~2–15 GB/rank actually is

There is no single headroom number; it depends on the domain:

| reading | incumbent peak (GB) | **spare to W_live 125.83** | note |
|---|---:|---:|---|
| weights-only (EXL3 experts+REST+KV) | 109.26 | **16.57** | most generous; ignores demand |
| resident (OS footprint peak @102K) | 114.0 | **11.8** | Q1e P_resident = 9.3 non-weight + 2 margin |
| MLX-allocator (corrected) | ≈116.26 | **9.6** | the wired-limit domain; Q1e P_alloc ≈ 20.1 + 2 |
| Q1e pre-registered rule headroom (for the q3g128 arm) | demand 15.81 | **15.41 − 15.81 = −0.4** | B_arm 110.42; over by 0.4 (no margin) / 2.4 (with margin) |

So a working figure is **~2 GB (allocator-strict, with margin) to ~15 GB (the q3g128 arm's
weights-only headroom)** — matching the brief. The incumbent already runs within ~1 GB of the
wired limit on the allocator reading; **any** arm must also carry the prefill transient (9.3 GB
at 102K) *outside* the weights, which is why the fits are so tight.

**Demand-hygiene reclaims (cheap, format-independent) — the lever that moves the knife-edge:**
- **DSpark-taps retention (largest).** `Conversation.prefill` retains *every chunk's* DSpark taps
  (~30.7 KB/row) until prefill returns (flagged-not-fixed, `PERFORMANCE_HISTORY.md` Q2-gamma).
  At ~102K rows that is **≈3.1 GB** of avoidable transient. Freeing it per-chunk (after the chunk
  is consumed) is a hygiene fix that plausibly reclaims **~1–3 GB** — enough **alone** to flip the
  all-40-layer q3g128 arm (over by only 0.4 GB) into fit. *Measure the exact retention first.*
- **Tiled-indexer `[b,n,nb]` transient** (~0.5–1 GB): the HIER pass exists to remove exactly this
  row materialization at depth; a chunk-size or nb-gated HIER floor reclaims it at shallow depth.
- **Smaller prefill chunk** cuts the 9.3 GB transient proportionally (a lever, but it costs chunk
  throughput); **quantizing shared experts** below q8g64 saves ≤0.2 GB (shared is only 0.429 GB).

### 1.3 OPTION 1 — per-layer mixed format

Strategy: **convert the 35 k=3 layers first** (cheapest Δ — the 5 k=2 layers are already light in
EXL3 and expensive in native: skip them unless you must). Block win ≈ **(n/40) × 22 %**; e2e ≈
**(n/40) × 22 % × 0.33 = (n/40) × 7.26 %**.

| n native (of 40) | f | block win | **e2e** | q4g64 needs | q3g128 needs |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.5 % | 0.55 % | **+0.18 %** | 1.26 GB | 0.20 GB |
| 3 | 7.5 % | 1.65 % | **+0.54 %** | 3.78 GB | 0.59 GB |
| 7 | 17.5 % | 3.85 % | **+1.27 %** | 8.82 GB | 1.39 GB |
| 11 | 27.5 % | 6.05 % | **+2.00 %** | 13.9 GB | 2.18 GB |
| 35 (all k=3) | 87.5 % | 19.25 % | **+6.35 %** | 44.1 GB ✗ | 6.93 GB ✓ |
| 40 (all) | 100 % | 22.0 % | **+7.26 %** | 54.6 GB ✗ | 12.17 GB (knife-edge) |

Reading: **q4g64 mixing buys ~7 layers ≈ 1.3 % e2e** within the resident headroom — i.e. the
quality-acceptable format is effectively useless as a speed lever. **q3g128 mixing buys the whole
35-layer set (≈6.35 % e2e) and all-40 is over by 0.4 GB** — but it is the lossy format (§2).

*e2e estimate ±*: the 22 % block win is a single-node standalone bound; the live round overlaps the
expert block with comm, so realized e2e is **≤ the table** (the pack's Amdahl caveat). Treat each
row as a **ceiling**. *Risk*: a mixed model is **numerically inconsistent across depth** (some layers
affine, some trellis) — the battery (§2) is mandatory, not optional; and rank-asymmetric policy would
silently corrupt the all_sum (§3 R5).

### 1.4 OPTION 2 — q3g128-fit fixes (reclaim ~0.5–1 GB; §1.2)

Applying the demand-hygiene reclaims moves the all-40 q3g128 arm from **over by 0.4 GB** to
**fitted with real margin**, which is the only route to a **full-layer** native arm (§1.3, 7.26 %
e2e ceiling). *e2e estimate*: up to the 7.3 % Amdahl ceiling (HIER-era chunk). *Risk*: it does not
touch the **quality** blocker (cos 0.9423) — the arm that most wants to fit is the least likely to
pass the battery. It also **worsens** the allocator reading unless the transient stays out of the
peak (streamed build, §3 R2).

### 1.5 OPTION 3 — wired-limit raise (the documented reopen trigger)

The Q1e reopen route (a): a deliberate `iogpu.wired_limit_mb` raise above 120000 MiB
(`start_cluster.sh:1474` `DSV4_WIRED_LIMIT_MB`, read back at :1478) **with a risk analysis**.
Raising it ~2 GB makes all-40 q3g128 fit; more makes a partial q4g64 mix fit. *Risk (material)*:
on a 137.44 GB unified-memory box the wired limit reserves GPU-wired (unswappable) memory; raising
it trades headroom for the whole OS (other processes, page cache) and can provoke OOM-kill of the
runner under a cold deep prefill (the Q2-gamma note already flags a **cold single-request prefill
crossing the wired limit ≈328K tokens / 266–286K resident** even at the *current* limit).
**Prerequisite:** the demand-hygiene headroom (Option 2) must be measured **first** — you cannot
choose a safe new limit without the true post-hygiene demand. *e2e estimate*: unlocks Option 1/2's
ceiling (up to ~7.3 %). Treat as **owner-gated, not agent-gated.**

### 1.6 OPTION 4 — combinations

- **Demand hygiene + wired-limit raise + q3g128 (all-40):** the maximal fit route → **up to ~7.3 %
  e2e** ceiling; quality-gated (cos 0.9423) and carrying both the wired-limit risk and the
  quality risk.
- **Demand hygiene + partial q4g64 mix:** quality-acceptable format, but the hygiene reclaims only
  widen the mix from ~7 to ~9–10 layers (**≈1.6–1.8 % e2e**) — still small.
- **Demand hygiene + q3g128 on the 35 k=3 layers only (leave k=2 EXL3):** **fits with ~4.8 GB
  margin** at the *current* wired limit (105.22 experts + 15.81 demand = 121.03 < 125.83) → **≈6.35 %
  e2e** ceiling with **no wired-limit raise** — the best fit/risk compromise *if* q3 quality clears.

---

## 2. QUALITY PATH — the mandatory gate for ANY experts re-quant

### 2.1 The battery that exists in-repo today

`bench/dsv41_quality_battery/` is **the ship gate for precision changes on the DSv4.1 path**
(`battery.py:8`, `README.md:8-13`; `runbook.md:1-8`). It is the tool behind the shipped "R8a"
battery. It runs:
- **Needle recall** — 6 probes (`NEEDLE_PROBES`, `battery.py:88-108`) on a warmed 350K-token
  prefix (`build_deep_prompt`, `:565-595`).
- **Free-form prose + degeneration detectors** — 20 multilingual prompts (`FREE_PROSE`,
  `:114-171`), scored by `run_detectors` (`:460-484`): HIGH-confidence `cross_script_glue`
  / U+FFFD / repetition / self-doubt → **DIRTY**; `same_script_glue` is **advisory** (REVIEW only).
- **Tool calling** — 10 probes (`TOOL_PROBES`, `:201-232`), `check_tool_call` (`:616-679`).
- **Parked-restore recall** — `recall_teal` (`:917-971`).

**A PASS requires** (two layers, `aggregate` `:1022-1029` + `compare.py:14-27,234-240`):
- per-phase **CLEAN** = `prose DIRTY == 0 ∧ needle_miss == 0 ∧ tool_miss == 0 ∧ park.recall_teal`;
- `compare.py` vs a **frozen A reference** returns **no FAIL** — FAIL = any *new* high-confidence
  detector hit, a DIRTY-count rise, a needle/tool regression, or lost park recall
  (`compare.py:140-228`). Exit 0=PASS / 2=REVIEW / 1=FAIL.
- Precedent bar: needles **6/6**, tools **10/10**, prose **0 DIRTY / 0 REVIEW**, park
  `recall_teal=True`, no fails vs frozen `g3` (next18 ship, `PERFORMANCE_HISTORY.md:11606-11608`).
- **The gate bites**: it FAILED the q1b affine6 round on a deterministic `t2` tool-format flip →
  NO SHIP (`df33ec83f`, `3f170c955`).

### 2.2 The "ORIGINAL bf16 source" — ★ OWNER DECISION (likely unsatisfiable)

The requirement is to re-quant from the **original** source. Q1d established the pre-EXL3 original
is **`dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8`** — public, **510.3 GB, fp8 e4m3fn + FP4
experts** (block [32,32], ue8m0); the plain `…-UNCENSORED` (**bf16**) name **404s**. Three original
shards (layers {0,20,39}, 22.18 GB) were fetched and **sha256 bit-exact** — retained *specifically*
"for Round 2's recon-source-vs-original penalty". **No bf16 original exists** (in-repo or on the
Hub). ⇒ the measured cosines (0.9881 / 0.9423) are **EXL3-relative**, not original-relative.
**OWNER must choose the reference source:** (i) the FP8/FP4 original (experts are fp4 there —
dequant is itself lossy), (ii) the EXL3 reconstruction (what the current cosines use — EXL3's own
error baked in, unmeasured), or (iii) materialize a bf16 source (a new artifact; still an fp4→bf16
upcast for experts).

### 2.3 Quantifying the q3g128 cos-0.9423 concern

The number is the pack's **A6b** method (R=4 natural routing, layer 20), **native arm vs the EXL3
production fused kernel's output** — i.e. vs the **EXL3 2.9 bpw trellis reconstruction**, *not*
bf16 truth. Both arms are built from the same EXL3-decoded weights, and **A6a** (reconstructed-fp16
"TRUE-W" vs the EXL3 kernel on a 24-expert subset) is **near-exact** (cos 0.9999995) — so 0.9423 is
a **real affine-format quantization effect, not a harness bug**.
- **What it means:** ~**34.7 % relative-L2 error on the routed-expert-block output** (fp64-scored),
  a *uniform affine-grid* error class at 3-bit/group-128 with fp16 scale+bias (eff 3.25 bpw).
- **Context:** the dense-slice precedent's q4/q5/q6 cosines *to the bf16 EXL3-decoded W* are
  0.9959 / 0.9990 / 0.9998 (`PERFORMANCE_HISTORY.md:11661-11678`) — the q3g128 experts arm sits **far
  below even the worst dense arm**. For scale: the token-63 precedent shows a **bf16 tie-break
  (margin 0.0, zero re-rank)** can flip a greedy token downstream; a **0.94-cosine / 34.7 % rel-L2
  expert perturbation is orders of magnitude larger**, so a material output change is *expected*.
- **What is UNKNOWN:** the end-to-end battery effect (never measured for any native arm); the error
  vs the *original*; and — a **double bind** — the only arm that fits (q3g128) is **both** the lossy
  arm **and** memory-blocked, while the accurate arm (q4g64) does not fit at all.
- **Do not decide this.** "Is 0.9423 acceptable?" is an **owner judgment** requiring battery evidence.

### 2.4 The mandatory gate — CHECKLIST (★ = OWNER decision point)

1. **Re-quant from the declared original source** — ★ *which source* (§2.2).
2. **Full-model TWO-NODE run** — not a single-layer bench (every battery run drives the live 2-node
   cluster API, `battery.py:61-62`; `runbook.md:101-103` gates on `READY (2/2)`). The 22 % win and
   all cosines are **standalone single-node module bounds** with an Amdahl "e2e ≤ this" caveat.
3. **Battery PASS** (`compare.py` PASS; phase CLEAN) — ★ *is "battery CLEAN" sufficient for a
   0.94-cosine re-quant?* (the governing ruling `PH:11580-11582` was made for a bf16-tie-break-class
   numeric change, not a 34.7 % rel-L2 one).
4. **`compare.py` vs a FROZEN A reference** — ★ *which* frozen reference, and must it be re-frozen
   from the current production build.
5. **Identity/parity evidence class** (co-required with the battery): post-boot **canary healthy both
   nodes**; **installed-module verify on BOTH nodes**; **production kernel md5 BEFORE==AFTER**;
   launcher "Nodes synchronized on commit X" → `READY (2/2)`; post-ship **parity smoke reproduces a
   known number**; **push/alignment guard** (the studios only run *pushed* commits). ★ *any additional
   bar* for a lossy re-quant (the repo's EXL3-build target names "cosine ≥0.999/layer + argmax
   agreement on next 64 greedy tokens", `docs/deepseek-v41-exl3-plan.md:181-186`).
6. **Fit precondition** — ★ *the reopen route* (wired-limit raise / demand shrink / ≤3.24 bpw format)
   — quality gating only matters **after** a fit route is chosen.
7. **Is ~7 % e2e worth the risk?** — ★ owner.

---

## 3. ENGINE-WORK estimate — load per-layer mixed expert formats (NO build done)

Findings from a read-only engine pass (file:line in the worker note; verified spot-checks below).

**Today the routed experts are ALWAYS EXL3 trellis** — `build_block` unconditionally does
`blk.ffn.experts = Exl3Experts(load_experts(ck, layer_id, rank=rank, world=world))`
(`mlx-lm/mlx_lm/models/deepseek_v41/exl3_build.py:684`, verified). A per-tensor format switch exists
**only for the dense roster** (`DSV41_DENSE_POLICY`, `exl3_build.py:548-644`, verified) — **not** for
experts. **But the machinery is largely a reuse:** the native forward already exists
(`mlx_lm.models.switch_layers.QuantizedSwitchLinear:27`, `BatchedSwitchGLU:204` + `fuse_weights():229`,
verified), and `MoE.__call__` is format-agnostic (`moe.py:167`) ⇒ **no forward edit**. The TP2
intermediate slice (per-rank H=1152, 16/64/128-aligned) and the `_all_sum_tail` collective are
**unchanged** by format mixing.

| # | workstream | days |
|---|---|---:|
| 1 | native expert module (reuse `BatchedSwitchGLU` + `fuse_weights`) | 0.5–1 |
| 2 | loader: reconstruct→fp16→transpose→`mx.quantize` **streamed**, TP slice | 1.5–2 |
| 3 | per-layer config + `build_block` switch + plumbing (`DSV41_DENSE_POLICY` clone) | 1–1.5 |
| 4 | allocator/transient: streamed build, `clear_cache` ordering, peak measurement | 1–1.5 |
| 5 | **fingerprint + park admission**: format digest, `KV_ALLOWLIST`, G4 identity field | 1.5–2 |
| 6 | tests (per-layer selection, numerics, TP2 all_sum parity, fingerprint/refusal, park admit) | 2–3 |
| 7 | integration/fit harness (peak hooks, A/B wiring) | 1–2 |
| | **TOTAL (engine)** | **≈8.5–13.5 → call it 10–16 with contingency** |

**Top risks (ordered):**
- **R1 — FIT (dominant, and it is NOT an engine risk).** §1: q4g64 all-native = 152.88 GB/rank ≫
  W_live; a per-layer q4g64 mix buys only ~7 of 35 layers; q3g128 fits only at the knife-edge (and
  is lossy). **Resolve the fit route BEFORE spending the engine days.**
- **R2 — load transient.** Reconstruct-to-fp16 is unavoidable (the source is fp8/fp4, not bf16); a
  non-streamed build adds **+13.6 GB/layer peak (fatal)**; streamed is safe but adds ≈**160–200 s**
  to load and must `mx.clear_cache()` between layers (the `AffineProj` precedent documents a measured
  ~1.5 GiB residue without it, `exl3_build.py:98-103`).
- **R3 — fingerprint invisibility (the new-site-identity interaction).** The 5-field fingerprint
  (`model_fp`, `tokenizer_fp`, `kv_code_fp`, `mlx_version`, `layout_fp`) hashes config/index + weight
  **names/sizes**, and **source** of allowlisted modules — **but does NOT hash the expert FORMAT**
  (verified: `fingerprint.py` `_model_fp` docstring "config.json + weight index + tensor names/sizes
  + quant config"; `compute_fingerprint` returns exactly those 5 fields). The cross-rank identity
  digest is `_BUILD_IDENTITY_FIELDS = ("model_fp","tokenizer_fp","kv_code_fp","mlx_version")` —
  **`layout_fp` deliberately excluded** (verified, `coord.py:128-132`, `build_identity_digest`
  `:183-200`). ⇒ **an env-only format toggle on the same code would change NONE of these fields**, so a
  mixed-format build could **silently restore parks written by a uniformly-EXL3 build** — the audit's
  stale-logits class. **Admitting the new format as a compatible site requires a new format digest in
  the fingerprint AND in the G4 `_BUILD_IDENTITY_FIELDS`, and (if a new module is added) an entry in
  `KV_ALLOWLIST`** (else the CI `allowlist_problems` guard trips, `fingerprint.py:700-721`). The new
  rank-invariant **SITE-identity park** is otherwise orthogonal — but the format digest must join the
  rank-invariant set too.
- **R4 — numerics/quality.** A mixed model is inconsistent across depth; per-layer cosine ≠ e2e
  quality — the battery is mandatory.
- **R5 — TP2 rank symmetry.** The format policy must resolve **identically on both ranks**; a
  rank-asymmetric env would sum partials from different quantisations (silently wrong, and **not**
  caught by `kv_code_fp` since the source is identical). Needs a rank-agreement axis.
- **R6 — kernel/compile + dtype.** The new `gather_qmm` path changes the first-chunk compile
  (warmup) and needs fp16 activations matching the scales dtype or `gather_qmm` falls to a ~10×
  slower path (`moe.py:74-78`).

**Park is format-agnostic in its files** — it serialises only KV/cache state
(`win_kv/comp_kv/index_k/carry/engram_ids/draft/snapshot/anchor`), no expert bytes; the **only**
coupling is the fingerprint gate (R3).

---

## 4. RECOMMENDED DECISION PATH (ordered, bounded)

**Recommendation: do NOT authorize the engine build yet.** Sequence:

1. **[OWNER, 0 days]** Choose the fit route (§1.5/1.6): (a) wired-limit raise + risk analysis,
   (b) demand-hygiene shrink, or (c) accept a ≤3.24 bpw format. *This gates everything.*
2. **[bounded offline, ~1–2 days]** Measure the demand-hygiene reclaims (§1.2), DSpark-taps first.
   If ≥0.5–1 GB is reclaimed, the all-40 q3g128 arm (over by 0.4 GB) fits, and the k=3-only
   q3g128 variant fits with ~4.8 GB margin at the **current** limit.
3. **[OWNER]** If a fit route exists, run the quality gate (§2.4): source decision + full-model
   two-node battery. **Do not assume q3g128's 0.9423 is acceptable.**
4. **[only then]** Dispatch the engine work (§3, ~10–16 days), led by the fingerprint/park coupling
   (R3).
5. **If no fit route survives 1–2:** **close** the native-experts requant as *does-not-fit / not
   worth it* at 128 GiB + the production wired limit. The honest economics: **~6–7 % e2e ceiling**
   (Amdahl, single-node bound) for a **quality-gated** re-quant, **10–16 engine-days**, and — on the
   wired-limit route — added memory-breach risk, against an operating point already within ~1 GB of
   the limit. **The lever is real at the block; the e2e prize is small.**

---

## Provenance / integrity
- **Task F is offline design only — 0 boots, 0 config changes, no cluster contact, no requant.**
- All byte math re-derived from `q1d_memfit.py` + `q1e/a3_packing.json` (via `git show 0808e3507…`);
  the q4g64 bpw correction (4.25 → 4.5) carried from Q1e.
- Engine claims spot-verified on `origin/main`: `exl3_build.py:684` (unconditional EXL3 experts),
  `exl3_build.py:556-569` (`DSV41_DENSE_POLICY` dense-only), `switch_layers.py:27/204/229`,
  `fingerprint.py` (5 fields, `_model_fp` = names/sizes), `coord.py:128-132`
  (`_BUILD_IDENTITY_FIELDS` excludes `layout_fp`).
- Companion: `docs/dsv41-pmp-tasb-reconciliation-2026-10-11.md` (Task G, branch
  `p20/pmp-tasb-reconciliation`) reconciles the indexer share used in the pack's block-share input.

---

## OWNER DECISION — 2026-10-11

**The native-experts build is NOT authorized.** The lever is real at the expert block (+22.1–23.0 % WALL, pack Task A, ≥20 % gate) but the *e2e* ceiling is only **≈6–7 %** (Amdahl: ~33 % expert-block share × 22 % block win), the engine cost is **≈10–16 eng-days** (§3), and the arm that would fit is the materially **lossy** one (q3g128, cos 0.9423) while the quality-acceptable arm (q4g64, cos 0.9881) does not fit at 2 nodes. **Owner ruling: the projected ~6–7 % prefill gain for 10–16 engine-days plus a quality tradeoff is judged not worth it.** **Engine spend and the requant remain CLOSED.** The quality-battery question (§2) is **moot unless the reopen condition below is met.**

---

## REOPEN CONDITION — third node (owner-raised)

The one configuration that changes the calculus is a **third node (TP3)**, spreading the routed experts across 3 Macs instead of 2. Verified 3-node fit math on the repo's own byte facts (Q1d `docs/benchmarks/phase20-throughput/raw/pricing/q1/q1d/q1d-memfit-and-fetch.md`; Q1e `…/q1/q1e/{a3_packing,a2_nwpeak}.json`; `PERFORMANCE_HISTORY.md` Q1d/Q1e):

| quantity | value | basis |
|---|---:|---|
| EXL3 experts (both nodes) | **196.0 GB** = 98.253 GB/rank @TP2 | Q1d header |
| q4g64 experts (both nodes) | **305.8 GB** = 152.883 GB/rank @TP2 (= EXL3 **+54.6/rank**) | Q1e, 4.5 bpw |
| non-expert REST (both nodes) | **14.55 GB** = 10.71 GB/rank @TP2 | Q1d |
| q4 model total | ≈ **320.3 GB** | 305.8 + 14.55 |
| **TP3 per-node weights** | ≈ **106.8 GB** | 320.3 ÷ 3 |
| non-weight peak (allocator) | ≈ **20.1 GB** | Q1e a2 |
| **TP3 per-node peak** | ≈ **127 GB** | 106.8 + 20.1 |
| W_live (wired limit) | **125.829 GB** (iogpu.wired_limit_mb 120000) | q1e a2; `start_cluster.sh:1474` |
| **verdict** | **≈1 GB OVER** | same knife-edge class as today (incumbent 124.833 GB allocator peak, ≈1 GB *under*) |

The **lossless DSpark-taps reclaim (≈3 GB @102K)** — `Conversation.prefill` retains every chunk's taps until prefill returns (flagged-not-fixed; `PERFORMANCE_HISTORY.md` Q2-gamma) — would tip the arm to fitting (**≈1.9 GB margin**). *(Reconciliation: the brief's aggregate form — (98.09+54.6)×2 + 14.5, ÷3, +20.1 ⇒ ≈126.8 GB — and the repo-fact form (≈126.9 GB) agree to ≈0.13 GB; both land ≈1 GB over. The repo's exact figures are EXL3 **98.253** GB/rank [not 98.09], q4g64 **152.883** GB/rank = **+54.63**/rank [Q1e 4.5-bpw correction; the pack's +46 used the wrong 4.25 bpw], and REST **14.547** GB both-nodes.)*

**3-node sync scaling is UNQUANTIFIED (estimate-grade).** The current 2-node sync cost is itself only loosely bounded (1–24 % of prefill chunk wall, unmeasured in the last attribution; ~2–4 % as an overlapable-class estimate; decode collective-removal bound ≤8.4 %). A 3-node ring adds collective steps/messages per layer while per-rank compute shrinks, so the added-compute-to-throughput conversion is expected **sub-linear — plausibly +20–40 % prefill rather than the naive +50 %, decode flat-to-slightly-worse.** Quantifying it needs a TB5 all-sum microbench at real per-layer shapes on the current pair **plus** the 3-way link work.

**Caveats — TP3 is NOT a drop-in:**
1. The current **jaccl/TB5 transport is a two-node build**; a third node needs **3-way topology work**.
2. **Divisibility:** the experts are width-sharded on the **intermediate width H = 2304** (→1152 at TP2; a 3-way split is **768 — clean**; `loader.py::_slice_intermediate`, phase-12 geometry). The **hidden width 5120 is NOT divisible by 3** (1706.67), but it is *not* the sliced axis today; any policy that did slice it would need padding / uneven-split handling. *(The brief labelled 5120 the "intermediate dim"; per the Q1d config the intermediate is 2304 and 5120 is `hidden_size`.)*
3. The **engine's TP3 support is unproven** — the §3 work list assumed TP2.
4. A **fresh quality battery** (§2) is required regardless: the fit math is orthogonal to the q3g128/q4g64 quality question, and any fit arm must still clear the ship gate.
