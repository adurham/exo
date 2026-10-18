> Companion evidence pack for `WRITEUP.md` (raw capture, scrubbed with the same
> placeholder map as the writeup).

# EVIDENCE PACK (task A1) - homelab exo/MLX 2-node cluster writeup

## RAW STATUS (as of 2026-10-18)

This pack is mostly repo-doc citations (still fully checkable) plus live SSH probes. Pack
sha256 (first 16): `6b5d81bdb7b6cc92`.

**EDIT POLICY from 2026-10-18:** corrections are APPEND-WITH-LABEL, never in-place
replacement. A sourced figure is never overwritten by a recomputed one. The original keeps its
basis; the recomputed value is appended with its own basis; if the original's basis cannot be
identified, it is tagged PROVENANCE-WEAK.

**SOLE-SURVIVOR:** figures whose only surviving source is this pack (the live SSH probe rows —
`ibv_devinfo`, `sysctl`, `top`, `ifconfig`, serials). A dispute over one cannot be adjudicated
by re-reading raw.


Produced: 2026-10-16 23:27 
Author: subagent A1 (docs + git + read-only SSH). Append-only file per brief.
Repos (READ-ONLY, none modified): /Users/<user>/repos/exo (+ mlx and mlx-lm submodules).

Method / provenance rules used here:
- Every row carries a source path + line, or is labelled as a command output captured at the timestamp above.
- Numbers are classified by CONVENTION: "fresh full-feed" prefill = prompt_tokens / TTFT;
  "delta/incremental" prefill = rate over an appended span; "end-to-end" = wall clock incl. TTFT;
  "fixed-window" decode = tokens/decode_s over a fixed generation window; "ms/round" = per
  speculative verify round (includes verify), NOT decode-only steady state.
- A number stated in two places with different values is reported BOTH times and flagged CONFLICT.
- Estimates are labelled ESTIMATE; anything with no raw run behind it is flagged PROVENANCE-WEAK.

Live-vs-doc note: the cluster was serving during this capture. Read-only SSH probes
(`system_profiler`, `sw_vers`, `sysctl`, `top -l 1`, `ifconfig`, `ps`) were used; no launch,
kill, restart or model call was issued by this task.

---

## D1. PERFORMANCE NUMBERS

### D1.1 Curated core table (rows personally re-read and confirmed against the source line)

| # | metric | value | ctx depth | date | commit/tag | source (path:line) | convention | flags |
|---|---|---|---|---|---|---|---|---|
| 1 | prefill | 366.6 tok/s | 100K | 2026-08-21 | tag known-good-prefill-20260821-165048 | docs/PERFORMANCE_HISTORY.md:152 | fresh full feed (needle PASS) | superseded |
| 2 | prefill | 351.5 tok/s | 300K | 2026-08-21 | same tag | docs/PERFORMANCE_HISTORY.md:153 | fresh full feed (needle PASS) | superseded |
| 3 | prefill | 331.6 tok/s | 500K (nominal; real ~352.6K) | 2026-08-21 | same tag | docs/PERFORMANCE_HISTORY.md:154 | fresh full feed (needle PASS) | CONFLICT depth-label |
| 4 | decode | 17.48 tok/s | 100K | 2026-08-21 | same tag | docs/PERFORMANCE_HISTORY.md:155 | decode-only | async fence SILENTLY BROKEN |
| 5 | decode | 18.60 tok/s | 300K | 2026-08-21 | same tag | docs/PERFORMANCE_HISTORY.md:156 | decode-only | async fence SILENTLY BROKEN |
| 6 | decode | 17.26 tok/s | 500K (nominal; real ~352.6K) | 2026-08-21 | same tag | docs/PERFORMANCE_HISTORY.md:157 | decode-only | async fence SILENTLY BROKEN |
| 7 | decode | 18.6-18.9 tok/s | short (512-tok prompt) | 2026-08-21 | same tag | docs/PERFORMANCE_HISTORY.md:158 | decode-only | async fence SILENTLY BROKEN |
| 8 | prefill | 359.7 tok/s | 100K | 2026-08-22 | config EXO_DSV4_MOE_FUSED_GATE_UP=1 + EXO_DSV4_FENCE_ASYNC=1 | docs/PERFORMANCE_HISTORY.md:72 | fresh full feed (needle PASS) | - |
| 9 | prefill | 348.2 tok/s | 300K | 2026-08-22 | same | docs/PERFORMANCE_HISTORY.md:73 | fresh full feed (needle PASS) | - |
| 10 | prefill | 324.1 tok/s | 500K (nominal) | 2026-08-22 | same | docs/PERFORMANCE_HISTORY.md:74 | fresh full feed (needle PASS) | CONFLICT vs row 3 |
| 11 | decode | 26.91 tok/s | 100K | 2026-08-22 | same | docs/PERFORMANCE_HISTORY.md:75 | decode-only (needle PASS) | - |
| 12 | decode | 24.44 tok/s | 300K | 2026-08-22 | same | docs/PERFORMANCE_HISTORY.md:76 | decode-only (needle PASS) | - |
| 13 | decode | 21.51 tok/s | 500K (nominal) | 2026-08-22 | same | docs/PERFORMANCE_HISTORY.md:77 | decode-only (needle PASS) | CONFLICT vs p3 doc (23.48 @352,599) |
| 14 | decode | 29.2-31.1 tok/s | short (512-2000 tok prompt) | 2026-08-22 | same | docs/PERFORMANCE_HISTORY.md:78 | decode-only | - |
| 15 | decode | 25.5 t/s benign / 19.5 t/s agentic | 20K / 91K real-session replay | 2026-10-08 (pre-ship) | exo f4bb14746 | docs/known-good-decode-baseline-20261008.md:11 | ms/round (800-tok rounds incl. verify), NOT decode-only | CONVENTION-AMBIGUOUS |
| 16 | decode | 31.5 t/s benign / 23.7 t/s agentic | 20K / 91K | 2026-10-08 SHIPPED | exo 576e9d279, mlx-lm 3bf8316, tag known-good-decode-next17-20261008-160746 | docs/known-good-decode-baseline-20261008.md:12 | ms/round (118.56 / 130.07 ms per round) | CONVENTION-AMBIGUOUS |
| 17 | decode | 30.2 t/s agentic | 91K | 2026-10-08 | same + DSV41_INDEXER_HIER=0 | docs/known-good-decode-baseline-20261008.md:13 | ms/round (101.06 ms) | CONVENTION-AMBIGUOUS |
| 18 | decode (spec ON) | 36.63 tok/s median, fixed window | 100K | 2026-08-27 PROMOTED | tag dspark-mtp-prod-20260827-1745; mlx-lm dda9237, exo 6eba31ff1 | docs/dspark-mtp-production-baseline-2026-08-27.md:18 | fixed-window, temp=0, 256-tok window, 24 paired runs | vs 27.15 spec-OFF = +36.71%, 12/12 pairs |
| 19 | decode (spec OFF) | 27.15 tok/s median | 100K | 2026-08-27 | same | docs/dspark-mtp-production-baseline-2026-08-27.md:18 | fixed-window | - |
| 20 | decode (spec ON) | 28.44 median vs 24.19 stripped-OFF = +17.6% | 352.6K | 2026-08-28 | EXO_DSV4_DSPARK_TP_SHARD=1 + EXO_MLX_CLEAR_CACHE_INTERVAL=64; fix 2d85ccdcb | docs/dspark-mtp-production-baseline-2026-08-27.md:84 | fixed-window, 8 genuine runs | zero collapses; CI [+8.7%,+27.8%] thin |
| 21 | decode | 29.17 / 33.22 / 31.78 / 36.74 / 26.83 / 13.36 tok/s | 2,216 / 21,855 / 68,002 / 115,614 / 273,622 / 564,926 | 2026-09-22 | DeepSeek-V4-Flash-Vision-Exp, prod config (vec+ROWSDPA=3+VERIFY_BATCH=1+steel-BI+ATTN_ALLSUM=0) | docs/decode-depth-ladder-to-565k-memory-wall-2026-09-22.md:15-20 | decode-only (decode_tps from bench/long_decode_probe.py), needle PASS every rung | DIFFERENT CHECKPOINT than rows 1-14 |
| 22 | prefill | 213.5 / 397.6 / 422.3 / 423.3 / 409.0 / 371.8 tok/s | same 6 depths | 2026-09-22 | same | docs/decode-depth-ladder-to-565k-memory-wall-2026-09-22.md:15-20 | fresh feed, as reported by probe | - |
| 23 | peak GPU wired mem | 96.78 / 98.15 / 98.75 / 101.12 / 103.75 / 115.66 GB | same 6 depths | 2026-09-22 | same | docs/decode-depth-ladder-to-565k-memory-wall-2026-09-22.md:15-20 | peak resident, % of 120.6 GB wired limit | at 564,926 tok: 95.9% of limit |
| 24 | decode (WRONG vs CORRECT) | 27.03 vs 27.12 / 23.26 vs 25.65 / 18.96 vs 24.39 / 15.75 vs 24.91 / 12.31 vs 28.83 t/s | 20 / 605 / 1771 / 3521 / 7021 | 2026-09-22 | same | docs/decode-flat-across-depth-36tps-at-115k-2026-09-22.md:13-19 | LEFT = wall-clock incl. prefill (MEASUREMENT BUG); RIGHT = decode-only | canonical example of the convention trap |
| 25 | decode | 13.36 and 24.15 tok/s, same night, same nominal depth | 565K | 2026-09-23 | - | docs/overnight-results-two-mechanisms-and-stochastic-collapse-2026-09-23.md:38-39 | fixed-window | BIMODAL / stochastic collapse band - report as a band, not a point |
| 26 | verify cost | 83.76 -> 60.60 ms mean | 100K | 2026-08-27 | MTP-PROF n=1550 | docs/dspark-mtp-production-baseline-2026-08-27.md:35 | per-verify-round | C_s 3.20 -> 2.14 |
| 27 | prefill (spec era) | 17.5-18.6 tok/s decode, spec OFF entirely | 100-500K | 2026-08-21 | known-good-prefill-baseline-2026-08-21 | docs/dspark-mtp-production-baseline-2026-08-27.md:99-100 | decode-only | explicitly recorded as SUPERSEDED by row 18 |

Note on rows 21-23: this ladder runs on **DeepSeek-V4-Flash-Vision-Exp**, not the
DeepSeek-V4.1-Flash-EXL3 2.9bpw checkpoint the cluster now serves. Rows 1-20 are the EXL3 /
Flash-0731 lineage. The two families must not be blended in one curve.

### D1.2 Full extracted corpus table (210 rows, machine-extracted)

All 210 rows below carry a source path:LINE. Citations were auto-verified: 196/210 rows had a
row number appearing verbatim on the cited line; 14 were off-by-a-few (mostly the child citing a
markdown table HEADER line instead of the data row) and were auto-repaired to the nearest line
containing the value; 2 remain approximate and are marked. Rows flagged PROVENANCE-WEAK or
CONVENTION-AMBIGUOUS by the extractor are preserved verbatim - do not mix conventions.

# Extracted measured-throughput table — exo docs corpus (evidence pack)

All numbers below are REAL measured values printed in /Users/<user>/repos/exo/docs/**. Source paths are relative to the exo repo root. Columns: metric | value | ctx depth | date | commit/tag | source (path:line) | convention | flags. "prefill tok/s" in this corpus almost always means prompt_tokens/TTFT on a fresh full feed (tokenizer-ground-truth) via `bench/phase3_precheck_depth_throughput.py`, unless flagged. "decode tok/s" means decode-window-only (TTFT excluded) unless flagged.

## 1. Known-good decode/prefill baselines — DSv4-Flash-0731, TP=2 Mac Studio M4 Max

| metric | value | ctx depth | date | commit/tag | source (path:line) | convention | flags |
|---|---|---|---|---|---|---|---|
| prefill tok/s | 366.6 | 100K | 2026-08-21 | tag known-good-prefill-20260821-165048; exo 2eada90c3; mlx 1c591e105; mlx-lm 5e88545a33ad... | docs/known-good-prefill-baseline-2026-08-21.md:26 | fresh full feed; prompt_tokens/TTFT (end-to-end TTFT rate), tokenizer ground truth, needle PASS | none |
| prefill tok/s | 351.5 | 300K | 2026-08-21 | same | docs/known-good-prefill-baseline-2026-08-21.md:27 | same | none |
| prefill tok/s | 331.6 (repeats 333.9/334.0/331.6 across 3 runs) | 500K (real 352.6K-depth footprint) | 2026-08-21 | same | docs/known-good-prefill-baseline-2026-08-21.md:28,33 | fresh full feed TTFT rate | CONVENTION-AMBIGUOUS (depth labeled "500K" but real token count ≈352.6K; see B1/p3-synthesis) |
| decode tok/s | 17.48 | 100K | 2026-08-21 | same | docs/known-good-prefill-baseline-2026-08-21.md:26 | decode-only; spec OFF; async fence silently broken | none (superseded 2026-08-22) |
| decode tok/s | 18.60 | 300K | 2026-08-21 | same | docs/known-good-prefill-baseline-2026-08-21.md:27 | decode-only | none (superseded) |
| decode tok/s | 17.26 | 500K (nominal; real ≈352.6K) | 2026-08-21 | same | docs/known-good-prefill-baseline-2026-08-21.md:28 | decode-only | CONVENTION-AMBIGUOUS (depth label vs real: 352,599-tok prompt later measured as the "500K" point) |
| decode tok/s | ~18.6-18.9 | 512-tok prompt | 2026-08-21 | same | docs/PERFORMANCE_HISTORY.md:158 (known-good-prefill table mirrors :155-157 for the three depth rows) | decode-only short-ctx | none |
| prefill tok/s | 359.7 | 100K | 2026-08-22 | tag known-good-decode-fenceasync-20260822 (exo+mlx-lm) | docs/PERFORMANCE_HISTORY.md:72 | fresh full feed TTFT rate | none |
| prefill tok/s | 348.2 | 300K | 2026-08-22 | same | docs/PERFORMANCE_HISTORY.md:73 | fresh full feed TTFT rate | none |
| prefill tok/s | 324.1 | 500K | 2026-08-22 | same | docs/PERFORMANCE_HISTORY.md:74 | fresh full feed TTFT rate | CONVENTION-AMBIGUOUS (500K nominal vs ~352.6K real) |
| decode tok/s | 26.91 | 100K | 2026-08-22 | same | docs/PERFORMANCE_HISTORY.md:75 | decode-only; async fence genuinely engaged | none |
| decode tok/s | 24.44 | 300K | 2026-08-22 | same | docs/PERFORMANCE_HISTORY.md:76 | decode-only | none |
| decode tok/s | 21.51 | 500K (nominal) | 2026-08-22 | same | docs/PERFORMANCE_HISTORY.md:77 | decode-only | CONFLICT + CONVENTION-AMBIGUOUS (nominal depth; B1 re-measured real 352.6K at 23.48, +9.2%, methodological gap) |
| decode tok/s | 29.2-31.1 | 512-2000 tok prompt | 2026-08-22 | same | docs/PERFORMANCE_HISTORY.md:78 | decode-only short-ctx | none |
| prefill tok/s | 339 @500K "earlier multi-week campaign baseline" | 500K | ~2026-06/07 (earlier campaign) | none stated | docs/known-good-prefill-baseline-2026-08-21.md:36; docs/PERFORMANCE_HISTORY.md:140 | unspecified | PROVENANCE-WEAK (no raw run cited in either doc; used only as a parity reference) |

## 2. Async-fence fix validation (decode real measurements, 2026-08-22)

| metric | value | ctx depth | date | commit/tag | source (path:line) | convention | flags |
|---|---|---|---|---|---|---|---|
| decode tok/s | ~18.5 → 29.2-30.9 (+58-67%) | short ctx (~512-2000 prompt) | 2026-08-22 | none stated | docs/async-fence-fix-validated-2026-08-22.md:13-15 | decode-only, bench/decode_probe.py | none |
| decode tok/s | 30.70 / 30.91 / 30.91 / 30.89 / 30.85 (5 reps) | 512 prompt, 300 max tokens | 2026-08-22 | none stated | docs/async-fence-fix-validated-2026-08-22.md:80 | decode-only | CONVENTION-AMBIGUOUS (300-token window is sub-400; startup-heavy regime; doc notes per-request fence window share) |
| decode tok/s | 29.37 / 29.17 / 29.21 (3 reps) | 2000 prompt, 500 max tokens | 2026-08-22 | none stated | docs/async-fence-fix-validated-2026-08-22.md:81 | decode-only | none |
| prefill tok/s | ~350 (confirm-prefill gate) / decode ~30 at real depth | — | 2026-08-22 | none stated | docs/baseline-locked-decode-fence-fix-20260822.md:7-8 | directive value, not a run | PROVENANCE-WEAK (operating targets, not measured output in this doc) |

## 3. MoE all_sum family — prefill baselines & NOP ablation (2026-08-19/20)

| metric | value | ctx depth | date | commit/tag | source (path:line) | convention | flags |
|---|---|---|---|---|---|---|---|
| prefill tok/s (baseline) | 162.5 | 12,066-12,069 tok | 2026-08-19 | none stated | docs/moe-all-sum-dominant-cost-2026-08-19.md:18 | fresh feed, chunk-loop aggregate; quality intentionally broken | CONVENTION-AMBIGUOUS (bench-only tok/s with broken quality; per doc explicit) |
| prefill tok/s (NOP all_sum off) | 427.5 | same | 2026-08-19 | none stated | docs/moe-all-sum-dominant-cost-2026-08-19.md:18 | same | CONVENTION-AMBIGUOUS (NOP ablation, output garbage; harness validity check only) |
| prefill tok/s (baseline) | 167.3 | 38,066-38,067 tok | 2026-08-19 | none stated | docs/moe-all-sum-dominant-cost-2026-08-19.md:19 | fresh feed | none |
| prefill tok/s (NOP) | 430.3 | same | 2026-08-19 | none stated | docs/moe-all-sum-dominant-cost-2026-08-19.md:19 | same (2.57x, share 61.1%) | CONVENTION-AMBIGUOUS |
| prefill tok/s (cold, first run) | 152.7 | "same depth" (12K class) | 2026-08-19 | none stated | docs/moe-all-sum-dominant-cost-2026-08-19.md:43 | fresh feed, cold start | none (cold-start flagged by authors) |
| prefill tok/s (warm control) | 162.5 | same | 2026-08-19 | none stated | docs/moe-all-sum-dominant-cost-2026-08-19.md:45 | fresh feed, warm | none |
| prefill tok/s cumulative | ~165-168 (sharedscale) vs baseline ~162-172 | 2048-token steps | 2026-08-19 | none stated | docs/moe-allsum-sharedscale-live-test-no-speedup-2026-08-19.md:19-21 | fresh feed | none |
| prefill tok/s | ~168-169 vs baseline ~162-172 (statistically identical) | — | 2026-08-19 | none stated | docs/moe-allsum-sharedscale-CORRECTED-final-2026-08-19.md:26-28 | fresh feed | none |
| prefill tok/s cumulative | ~167-171 (chunk overlap) — unchanged from baseline | 2048-token steps | 2026-08-20 | none stated | docs/prefill-chunk-overlap-live-test-2026-08-20.md:24 | fresh feed | none |
| prefill tok/s cumulative | ~166-169 (jaccl sz3 change) | 2048-token steps | 2026-08-20 | none stated | docs/jaccl-sz3-tested-no-improvement-2026-08-20.md:21-23 | fresh feed | none |
| decode tok/s (OPT-8 broadcast change) | 111 vs 144 (23% slower) | — | 2026-08-19 | none stated | docs/gpu-util-vs-allsum-cost-reconciled-2026-08-19.md:78 | unspecified | CONVENTION-AMBIGUOUS (doc states "why" without window detail); referenced repeatedly (also docs/PERFORMANCE_HISTORY.md:456) |

## 4. Dispatch-count fusion A/Bs (decode, 2026-08-21)

| metric | value | ctx depth | date | commit/tag | source (path:line) | convention | flags |
|---|---|---|---|---|---|---|---|
| decode tok/s | Fusion ON 18.879 (σ=0.158) vs OFF 18.328 (σ=0.173), n=8 each (+3.01%) | ~512 tok prompt, 300 max tok | 2026-08-21 | none stated | docs/PERFORMANCE_HISTORY.md:1155-1156; docs/moe-gate-up-fusion-validated-2026-08-21.md:44-45 | decode-only | CONVENTION-AMBIGUOUS (sub-400-token window; bench/decode_probe.py) |
| prefill tok/s | 362.8 / 348.5 / 331.1 vs same-night baseline 368.7 / 351.9 / 333.2 (parity) | 100K/300K/500K | 2026-08-21 | none stated | docs/moe-gate-up-fusion-validated-2026-08-21.md:24-25 | fresh full feed TTFT rate | CONVENTION-AMBIGUOUS (500K nominal vs ~352.6K real) |
| decode tok/s | gate+up + qa_kv 18.789 (σ=0.141) vs gate+up alone 18.879 (σ=0.158) vs baseline 18.328 (σ=0.173), n=8 | ~512 tok prompt | 2026-08-21 | none stated | docs/qa-kv-fusion-no-measurable-gain-2026-08-21.md:54-56 | decode-only | CONVENTION-AMBIGUOUS (short window) |
| decode tok/s | 18.3 baseline (roofline input) | — | 2026-08-21 | none stated | docs/decode-roofline-dispatch-bound-2026-08-21.md:32 | decode-only ("tonight's A/B probes") | none |

## 5. hc_expand / hc_collapse fused kernels (prefill A/Bs, 2026-08-24/25)

| metric | value | ctx depth | date | commit/tag | source (path:line) | convention | flags |
|---|---|---|---|---|---|---|---|
| prefill tok/s | Arm A (OFF) 359.89 (355.03, 364.75); Arm B (ON) 373.80 (373.18, 374.43); Δ +13.91 (+3.87%) | ~70.5K real (100K target) | 2026-08-24 | exo deb1c8a6d (default flip; earlier e3df799c0), mlx-lm 7a1a4e8 (per docs/PERFORMANCE_HISTORY.md:719-723) | docs/hc-expand-kernel-ab-2026-08-24.md:13-15 | prefill tok/s = offline-tokenized prompt_tokens / TTFT wall (doc :344) | none |
| decode tok/s | 26.34-27.14 across A/B arms (structurally weak) | 70.5K | 2026-08-24 | same | docs/hc-expand-kernel-ab-2026-08-24.md:382-384 | decode-only, 1.5-2.5 s window | CONVENTION-AMBIGUOUS (window too short; doc says "structurally weak") |
| prefill tok/s | 300K: ON mean 356.72 vs OFF 352.21 (+1.28%); run rows 355.27/358.16 vs 352.25/352.16 | 211,797/211,022/211,774/211,379 real | 2026-08-24 | none stated | docs/hc-expand-depth-verification-2026-08-24.md:35-46 | prompt_tokens/TTFT | none |
| prefill tok/s | 500K: ON 336.74 vs OFF 333.91 (+0.85%, n=1/arm); OFF run also logged 336.8/336.7 tok/s end-to-end in exo.log at 352K | 352,670/352,431 real | 2026-08-24 | none stated | docs/hc-expand-depth-verification-2026-08-24.md:39-47,310,325 | prompt_tokens/TTFT | CONVENTION-AMBIGUOUS (n=1 per arm; log-line value is a chunk loop rate) |
| decode tok/s | ON 300K 31.08/25.03; OFF 300K 24.80/25.23; ON 500K 22.82; OFF 500K 23.95 | same depths | 2026-08-24 | none stated | docs/hc-expand-depth-verification-2026-08-24.md:35-40 | decode-only | CONVENTION-AMBIGUOUS (short windows; informational) |
| decode tok/s | 28.69 production probe | — | 2026-08-24 | none stated | docs/hc-expand-depth-verification-2026-08-24.md:199-199 | decode-only ("finish_reason implied by max_tokens (64)") | CONVENTION-AMBIGUOUS (64-token window, depth not stated; captured from production log) |
| prefill tok/s (hc_collapse) | Arm A mean 376.1681 (376.1051/376.2310), Arm B mean 383.2700 (383.9832/382.5568); Δ+1.888%; threshold 381.8106 | 70,390-70,625 real | 2026-08-25 | none stated in doc | docs/hc-collapse-kernel-ab-2026-08-25.md:22-30,283-284 | offline-tokenized prompt tokens / TTFT (doc :247) | none |
| decode tok/s (hc_collapse arms) | A 26.3937/26.5416; B 27.3900/30.2687 (2-run spread 2.88 tok/s — informational) | 70.4-70.6K | 2026-08-25 | none stated | docs/hc-collapse-kernel-ab-2026-08-25.md:265-268,387-396 | decode-only | CONVENTION-AMBIGUOUS (short windows) |
| prefill tok/s (hc_collapse depth) | 300K: ON 364.9700 vs OFF 357.9329 (+1.9660%); 500K: ON 345.5335 vs OFF 339.1177 (+1.8919%); 2K prod smoke 276.2442 | 211,670/211,239/352,277/352,070 real; 1,520 | 2026-08-25 | none stated | docs/hc-collapse-depth-verification-2026-08-25.md:55-69 | prompt_tokens/TTFT | none (n=1/arm at depth, doc discloses) |
| decode tok/s (hc_collapse depth) | 300K: ON 25.1234 / OFF 24.7175; 500K: ON 22.8427 / OFF 22.6526; smoke 29.3494 | same | 2026-08-25 | none stated | docs/hc-collapse-depth-verification-2026-08-25.md:55-59 | decode-only | CONVENTION-AMBIGUOUS (short windows) |

## 6. Prefill campaign measurements (Jun-Aug 2026)

| metric | value | ctx depth | date | commit/tag | source (path:line) | convention | flags |
|---|---|---|---|---|---|---|---|
| prefill tok/s | c=1 500K: 167 avg before → 251 avg after, never below 200 | 500K | 2026-06-24 | mlx-lm 453daa5, exo d26dc013 | docs/prefill-throughput-breakthrough-2026-06-24.md:87,312 | prefill loop aggregate (1993s for 500K cold) | none |
| prefill tok/s aggregate | c=2 B=2: 144 → 317 aggregate | 500K | 2026-06-24 | same | docs/prefill-throughput-breakthrough-2026-06-24.md:270,313 | B=2 concurrent aggregate | none |
| prefill tok/s | 158 per stream at B=2 | 500K | 2026-06-24 | same | docs/prefill-throughput-breakthrough-2026-06-24.md:354 | per-stream | none |
| prefill tok/s | 349 t/s fast chunks but 142 average (bimodal; reverted OPT-8) | 380K region | 2026-06-24 | same | docs/prefill-throughput-breakthrough-2026-06-24.md:191 | chunk-loop rate | CONVENTION-AMBIGUOUS (chunk-loop vs end-to-end distinction explicit in doc) |
| prefill tok/s | B=2 step=256: 46 (catastrophic) | — | 2026-06-24 | same | docs/prefill-throughput-breakthrough-2026-06-24.md:358 | aggregate | none |
| prefill tok/s | 367/353/340/318 aggregate (B=2; quality clean) | 100K/300K/500K(? depth set per run) | 2026-06-24 | same | docs/b2-quality-handoff-2026-06-24.md:37-37; docs/b2-mtp-resolution-2026-06-24.md:59 | B=2 aggregate | CONVENTION-AMBIGUOUS (depth↔value mapping not stated in line) |
| prefill tok/s | 100K: 255→353 (+38%); 495K ~200→306; 727K: 215; decode 29.0 mean | 100K/495K/727K | ~2026-07-01 | MOE_KERNEL_HANDOFF.md (argpartition etc.) | docs/PERFORMANCE_HISTORY.md:784-789 | fresh feed; staged 255→289→295→353 | none |
| prefill tok/s | SEQ_SPLIT=1: 220,318 tok/614.5s = 358.6 | 220K real | 2026-08-18 | none stated | docs/dsv4-220k-prefill-seqsplit-ab-2026-08-18.md:31 | fresh full feed wall | none |
| prefill tok/s | SEQ_SPLIT=0: 191,330 tok/688.775s = 277.8 | 191K real | 2026-08-18 | none stated | docs/dsv4-220k-prefill-seqsplit-ab-2026-08-18.md:35-36 | fresh full feed wall | none |
| prefill tok/s | STEP_SIZE=4096: 331.2 @191K (~8% slower than 358.6) | 191K | 2026-08-18 | none stated | docs/dsv4-prefill-step-size-4096-retest-2026-08-18.md:60-65 | fresh feed | none |
| prefill tok/s | CLEAR_CACHE_INTERVAL=2: 332.8 @179,720 tok | ~180K | 2026-08-19 | none stated | docs/dsv4-clear-cache-interval-2-test-2026-08-19.md:33 | fresh feed | none |
| prefill tok/s | unprofiled clean: 220,321 tok/612s = 360; profiled ~308-329 (10-15% profiler tax) | 220K | 2026-08-18 | none stated | docs/dsv4-220k-prefill-span-profile-2026-08-18.md:58,199-200 | fresh feed | none |
| decode tok/s (indexer p_block regression) | 13.67 @100K → 11.55 @300K → 10.03 @500K (regression); restored 17.39; prefill 363.0/347.9/328.9 vs same-night baseline 368.7/351.9/333.2 | 100K/300K/500K | 2026-08-21 | none stated | docs/indexer-pblock-decode-regression-2026-08-21.md:26-31,46-47 | decode-only / prefill TTFT | CONVENTION-AMBIGUOUS (500K nominal) |
| prefill tok/s + decode | 100K: 368.7 / 17.95; 300K: 351.9 / 18.50; 500K: 333.2 / 17.16 (needle PASS) | 100K/300K/500K | 2026-08-21 | none stated | docs/prefill-trace-instrumentation-findings-2026-08-21.md:76-78 | fresh feed TTFT + decode-only | none (matches known-good within noise) |
| prefill tok/s | 366.7 steady through untraced control; 378-385 during Instruments capture | ~70K region | 2026-08-23 | commit fc954293 deployed | docs/p2-xctrace-prefill-collective-wedge-2026-08-23.md:47,106,204-205 | fresh feed | none |
| prefill tok/s | 428.7 on a 349K-token request (real production prefill) | 349K | 2026-09-14 | none stated | docs/p13-switch-mlp-subphase-attribution-2026-09-14.md:200 | fresh feed | none |

## 7. PP-era numbers (hybrid-pp design doc, 2026-08-04 era) — contaminated conventions

| metric | value | ctx depth | date | commit/tag | source (path:line) | convention | flags |
|---|---|---|---|---|---|---|---|
| prefill tok/s (PP) | PP 364 @500K, TP 319 @500K (era curve 427/379/358/353/332/319 for TP; PP 490/512/485/431/377/364) | 1K-500K | 2026-07-17 (quoted) | none stated | docs hybrid-pp-prefill-tp-decode-design-2026-08-04.md:224-226; docs/p5-tp-prefill-gap-2026-08-24.md:173-179 | chunk-loop instantaneous rate (runner Prefill progress), NOT TTFT | PROVENANCE-WEAK + CONVENTION-AMBIGUOUS — preview affine-8bit checkpoint, chars//4 numerator inflates 1.42x (fact 1450, p5 §1.3); never apples-to-apples |
| prefill tok/s (renormalized, honest) | 225 / 214 / 202 at 100K/300K/500K (PP mode, current code) | 100K/300K/500K | 2026-08-04 | none stated | docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md:55; also :9210-9211 | end-to-end TTFT rate, tokenizer-ground-truth | none |
| prefill tok/s | PP pipeline-mode ladder: 100K 320 / 300K 304 / 500K 286 | 100K/300K/500K | 2026-08-04 | none stated | docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md:8552-8554 | fresh feed (harness token count) | none |
| prefill tok/s | post-transport-fix re-measure: 100K 224.5 / 300K 213.2 | 100K/300K | 2026-08-04 | none stated | docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md:8730-8731 | fresh feed | none |
| decode tok/s (PP era) | ~0.47-0.65 tok/s per session (stable slow mode); ~23-24 tok/s fast path (23.32 retracted as non-repro; 22.32 also unstable; 22.61/22.84 @14K on S71 build; 19.99 @14K config-broken at depth) | 14K-300K | 2026-08-04 | S71/S73 build b55ecd252 | docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md:8938-8945,10158-10171,10856-10869,10936-10939,11151,11289-11323,11576-11577,11625-11627 | decode-only client wall | PROVENANCE-WEAK + CONVENTION-AMBIGUOUS (bimodal stall artifact; doc retracts its own fast-path claims repeatedly; slow mode is the stable measured value) |
| decode tok/s | PP single-request 24.63 vs 25.30 @ short ctx; 2,078 tok: 0.49→24.09 on degenerate repetitive content | 2K-300K | 2026-08-04 | none stated | docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md:8937-8938,9344-9346 | decode-only | CONVENTION-AMBIGUOUS (content-driven swing documented) |
| decode tok/s (Qwen3.5-397B-4bit, PP) | short cold 37-38.2 / warm 47.7-48.9; long gen 49.3-52.9; 16K TTFT 36.6-40.6 s; prefill 439 / 467 @16K; ~50.6 decode @257 tok | 25 / 16K prompt | 2026-04-05 | none stated | docs/profiling/request_lifecycle_trace.md:87,152,216-232 | RequestTrace spans | PROVENANCE-WEAK + CONVENTION-AMBIGUOUS — different model entirely; PP pipeline era; quoted only to disqualify the "PP≈450" comparison (p5 §1.1 P4) |

## 8. P5 — TP vs PP head-to-head on current code (2026-08-24)

| metric | value | ctx depth | date | commit/tag | source (path:line) | convention | flags |
|---|---|---|---|---|---|---|---|
| prefill tok/s | TP 366.5 (TTFT 192.49 s) | 70,557 real (100K target) | 2026-08-24 | none stated | docs/p5-tp-prefill-gap-2026-08-24.md:27,341-346 | fresh full feed TTFT rate, tokenizer ground truth, needle PASS | none |
| decode tok/s | TP 27.79 | same run | 2026-08-24 | none stated | docs/p5-tp-prefill-gap-2026-08-24.md:345 | decode-only (usage) | CONVENTION-AMBIGUOUS (post-prefill 64-max-token window implied by probe; doc does not state window) |
| prefill tok/s | PP 277.0 (TTFT 255.06 s) | 70,656 real | 2026-08-24 | none stated | docs/p5-tp-prefill-gap-2026-08-24.md:28,430-436 | fresh full feed TTFT rate | none |
| decode tok/s | PP 24.74 | same run | 2026-08-24 | none stated | docs/p5-tp-prefill-gap-2026-08-24.md:434 | decode-only (usage) | CONVENTION-AMBIGUOUS |
| prefill tok/s | TP full chunk-loop 377 vs PP full chunk-loop 372 vs PP opening 523 | 70.5K | 2026-08-24 | none stated | docs/p5-tp-prefill-gap-2026-08-24.md:526-531 | chunk-loop rate (NOT TTFT) | CONVENTION-AMBIGUOUS — doc exists precisely to separate these; PP hides ~65 s drain outside the loop (255.06 s TTFT - 70,656/372) |
| prefill tok/s per-chunk | TP instantaneous 398.8 @2K → 362.7 @69.6K; PP 484.2 @28.7K → 106.8 @52K (stall) | 2K-70K | 2026-08-24 | none stated | docs/p5-tp-prefill-gap-2026-08-24.md:358-364,454-462 | chunk-loop instantaneous | CONVENTION-AMBIGUOUS |

## 9. P3 family: decode anchors and pool-grow A/B (DSv4-Flash-0731 TP, 2026-08-23)

| metric | value | ctx depth | date | commit/tag | source (path:line) | convention | flags |
|---|---|---|---|---|---|---|---|
| decode tok/s | 29.63 (usage) / 27.32 (events); p50-implied steady-state 31.39 | 520 | 2026-08-23 | exo HEAD 6bc843bfc | docs/p3-worker-b1-live-depth-anchors-2026-08-23.md:24,148-149,162 | decode-only (≥60 s window, EOS banned, finish=length) | none; two estimators in one doc — usage (headline) vs events |
| decode tok/s | 27.94 (usage) / 27.77 (events); p50-implied 29.17 | 100,026 real | 2026-08-23 | same | docs/p3-worker-b1-live-depth-anchors-2026-08-23.md:25,183-184,197 | decode-only | none |
| decode tok/s | 23.48 (usage) / 23.33 (events); p50-implied 25.54 | 352,599 real (the depth prior benches labeled "500K") | 2026-08-23 | same | docs/p3-worker-b1-live-depth-anchors-2026-08-23.md:26,218-219,232 | decode-only | none |
| decode tok/s | 29.63 / 27.94 / 23.48 (anchor table reprise) | 520 / 100,026 / 352,599 | 2026-08-23 | same | docs/p3-synthesis-500k-decode-decay-decomposition-2026-08-23.md:36-38 | decode-only | none (clean short→deep decay −20.8%) |
| decode tok/pool-grow arm A | 28.09 @100,022; 23.50 @352,602 | 100K / 352.6K | 2026-08-23 | exo HEAD 6bc843bfc | docs/p3-followup-poolgrow-ab-2026-08-23.md:485-487 | decode-only (usage) | none |
| decode tok/s (pool-grow arm B, raw =256) | 29.06 @100,023; 25.80 @352,601 | same | 2026-08-23 | same | docs/p3-followup-poolgrow-ab-2026-08-23.md:486,488 | decode-only | none |
| decode tok/s (shipped gated default POOL_GROW) | 28.94 @100,067 (34.55 ms/tok); 25.32 @352,643 (39.49 ms/tok) | 100K / 352.6K | 2026-08-23 | same | docs/p3-followup-poolgrow-ab-2026-08-23.md:953-954 | decode-only (usage; events estimator 28.73/25.31 agrees at depth) | none |

## 10. DSpark/MTP campaign — champions and A/Bs (PP-era to promotion)

| metric | value | ctx depth | date | commit/tag | source (path:line) | convention | flags |
|---|---|---|---|---|---|---|---|
| decode tok/s | 30.77 mean / 30.8 median, σ=0.067, 0/10 errors, MTP-on γ=2 c=1 | 100K | 2026-06-04 | tag validated-upstream-integration-20260604-153609; exo 652d8224 | docs/dspark-mtp-master-history-2026-08-28.md:80,86-87 | aggregate decode c=1 100K | none |
| decode tok/s | 30.6 mean / 30.7 median / min 29.9 / max 30.8 (integrated champion re-bench) | 100K | 2026-06-04 | same | docs/deepseek-v4-mtp-performance.md:19,26,63 | same | none (dup of champion) |
| decode tok/s | MTP-off correctness mode ~27 t/s | 100K | 2026-06-04 | same | docs/deepseek-v4-mtp-performance.md:38 | decode | CONVENTION-AMBIGUOUS (approximate "MTP-off ~27 t/s" reference) |
| decode tok/s (staging ladder) | control 15.4 → step128 15.5 → fence16 17.3 → +MTP 21.6 | 100K | 2026-06-04 | same | docs/deepseek-v4-mtp-performance.md:80-83 | decode | none |
| decode tok/s (~15.6 sequential reference) | PP+DSpark 27-33 tok/s single-request vs TP ~15.6 sequential | 100K (PP era) | 2026-06→07 | none stated | docs/superdeepseek-2xdgx-speedup-levers-2026-08-19.md:27 | decode | PROVENANCE-WEAK (quoting fact 1116/1410; PP-era, not reproducible under TP) |
| decode tok/s (token-tree) | tree 29.95 ≈ baseline 30.06; tuning variants all [29.7, 29.95] | 75K bench / 100K needle | 2026-05-19/20 | none stated | docs/PERFORMANCE_HISTORY.md:1445-1451 | decode c=1 | none |
| decode tok/s (Eagle K=8 no-renorm) | 28.7998 → 29.0375 mean (+0.24) | 100K c=1 | 2026-05-24 | none stated | docs/PERFORMANCE_HISTORY.md:1382-1383 | decode-only | none |
| decode tok/s (champion claims, NOT reproducible) | 31.5 claimed then 29.5 on retest; 32.3 claimed then cratered to 4.3 on redeploy; real reproducible 30.06 σ=0.06 | 100K | 2026-05-17/18 | champion-2026-05-17-fenced / champion-2026-05-18-acksync (tags) | docs/PERFORMANCE_HISTORY.md:1334-1338; docs/dspark-mtp-master-history-2026-08-28.md:273 | decode | PROVENANCE-WEAK (failed to reproduce; ledger entry 12) |
| decode tok/s (γ=2 K=1 FENCE=4 champ) | 34.19 σ=0.05, 5/5 clean | 100K c=2? (γ campaign era) | 2026-05-23 | none stated | docs/PERFORMANCE_HISTORY.md:1489-1490 | decode aggregate | CONVENTION-AMBIGUOUS (doc era mixes c=1/c=2 framing; "0.81 short of 35 target") |
| decode tok/s (drain elimination) | 22.9 → 34.5-34.6 aggregate at c=2 100K MTP-on | 100K c=2 | 2026-05-20/21 | none stated | docs/PERFORMANCE_HISTORY.md:1392 | aggregate | none |
| decode tok/s | 34.16 t/s c=2 100K "champion" quality-bug era | 100K c=2 | 2026-05 era | none stated | docs/PERFORMANCE_HISTORY.md:1895 | aggregate | PROVENANCE-WEAK (later undermined by quality bug; see §9 of PH) |
| decode tok/s (rowseq campaign, July) | loop champion 29.4-29.7; vec batched 34.1; vec+ROWSDPA=1 35.4-35.5; ROWSDPA=2 31.1-31.3 (gold gate 0/3 byte-dirty) | 100K, 400-tok gens | 2026-07-12 | none stated | docs/vec-rowsdpa-campaign-2026-07-12.md:37-40 | fixed-window decode, temp 0 | CONVENTION-AMBIGUOUS (throughput real, output byte-divergent vs loop — quality gate failed) |
| decode tok/s | prod loop 29.4-29.7; vec+tau0.5 34.1; vec+tau0 26.3 | 100K | 2026-07-12 | none stated | docs/vec-serving-increment4-handoff-2026-07-12.md:4-16 | fixed-window decode | none |
| decode tok/s (post-divergence-fix) | 36.9 decode / 360 prefill @ a rung; short-ctx 33.7 (vs 29.5 loop champ); later 39.1 decode / 359 prefill; 36.1-36.3 short-ctx | 100K-rung / short ctx | 2026-07-12 | mlx-lm 095c98c + exo 4b9932322 (fix) | docs/vec-rowsdpa-campaign-2026-07-12.md:141-171 | decode | CONVENTION-AMBIGUOUS (era values; superseded by 2026-09-22 vec-wash finding) |
| decode tok/s (rowseq followups) | prod lossy 27.3 → lossless stack 26.7-26.8; c=2 15.3 t/s/stream | 600-tok window / c=2 | 2026-07-10 | mlx-lm pinned 1fe020ed; exo uv.lock | docs/dsv4-rowseq-followups-plan-2026-07-10.md:297,309,334-351 | fixed-window decode | none |
| decode tok/s | @122K: row-seq 24.6 vs 19.1 sequential vs 26.2 uncorrected | 122K | 2026-07-10 | same | docs/dsv4-rowseq-followups-plan-2026-07-10.md:29 | decode | none |
| decode tok/s (c=2 MTP era July) | ~20 tok/s per stream (~40 aggregate) c=2 100K MTP-on γ=2 | 100K c=2 | 2026-06-06/07 | mlx-lm cb7e3bd/aaac5c3/60a0a0c/491f6fe/5b00004 | docs/deepseek-v4-c2-mtp-verify-fixes.md:21-23 | per-stream ~ c=2 aggregate | CONVENTION-AMBIGUOUS |
| prefill tok/s aggregate | 367 @100K B=2 post-fix (b2 gate removal) | 100K B=2 | 2026-06-24 | exo 8a9cdee/48a4a3c; gate removal 47fdf32a | docs/dspark-mtp-master-history-2026-08-28.md:96; docs/b2-mtp-resolution-2026-06-24.md:173 (table) | B=2 aggregate | none |

## 11. DSpark FULLBLOCK cliff (2026-08-04, retracted/reframed 2026-08-24)

| metric | value | ctx depth | date | commit/tag | source (path:line) | convention | flags |
|---|---|---|---|---|---|---|---|
| decode tok/s | DSpark ON+FULLBLOCK 27.56 @depth≈500 → 1.73 @depth≈14,253 (15.9x "collapse"); DSpark OFF 25.11 → 19.57 (1.28x normal) | ~500 / ~14K | 2026-08-04 | none stated | docs/dspark-fullblock-context-scaling-cliff-2026-08-04.md:49-50 | decode-only | CONFLICT — same doc later: re-measured 17.31 @15K clean on fresh relaunch (:338-339); band narrowed; Aug-24 reattribution to verify MODE (PERFORMANCE_HISTORY.md:1513-1538) |
| decode tok/s | moe-parts-rowseq win only near-zero ctx: 27.48 mean @300-500 tok | 300-500 tok | 2026-08-04 | none stated | docs/dspark-fullblock-context-scaling-cliff-2026-08-04.md:41 | decode | none |
| decode tok/s | stall runs: 18.03 / 16.88 clean, 1.28 collapse (3rd run); 19/20 clean at 23.9-25.4, one dip 10.98 | ~15K | 2026-08-04 | none stated | docs/dspark-fullblock-context-scaling-cliff-2026-08-04.md:361-362,604-607 | decode | PROVENANCE-WEAK (stochastic GPU-idle stall class; n-capture) |

## 12. MoE gate+up contradiction (2026-06-26 vs 2026-08-21)

| metric | value | ctx depth | date | commit/tag | source (path:line) | convention | flags |
|---|---|---|---|---|---|---|---|
| decode tok/s | 37.2 baseline vs 35.8 fused (−3.8%) at c=2 aggregate | c=2 (depth not stated) | 2026-06-26 | none stated | docs/dsv4-decode-stall-2026-06-26.md:5-6,80,175-176 | c=2 aggregate decode | CONFLICT — contradicts the 2026-08-21 +3.01% c=1 short-ctx result (PH §4.4:1234-1237 documents the contradiction; not resolved: "something else changed between these two measurements") |
| decode tok/s (aggregated c=2) | 0.42 t/s (both streams near-zero output, quality broken) | c=2 200K needle | 2026-06-26 | none stated | docs/dsv4-decode-stall-2026-06-26.md:203 | aggregate incl. failure | none (failure-mode measurement) |

## 13. DSpark/MTP production promotion + 352K (2026-08-27/28)

| metric | value | ctx depth | date | commit/tag | source (path:line) | convention | flags |
|---|---|---|---|---|---|---|---|
| decode tok/s | batched-ON median 36.63 vs spec-OFF 27.15 (+36.71%), CI [+28.26,+51.02]%, 12/12 pairs | 100K | 2026-08-27 | tag dspark-mtp-prod-20260827-1745; mlx-lm dda9237, exo 6eba31ff1, docs 56be3ea18 | docs/dspark-mtp-production-baseline-2026-08-27.md:18-21,42-43; docs/PERFORMANCE_HISTORY.md:4985 | 256-token fixed-window decode tok/s (~28.3 rowseq median) | none |
  - STATISTIC: +36.71% is the source doc's PAIRED-difference statistic. Ratio of the two displayed medians is 36.63/27.15 = +34.92%. Both are correct; they are different estimators.
| decode tok/s (verdict REVERT, rowseq width-3, pre-promotion A/B) | ON median 28.30 (IQR 27.44-29.90) vs OFF median 27.49 (+1.87%, CI [−0.82,+9.45] → FAIL) | 100K | 2026-08-26 | stage 2b commit 433dce6c1 | docs/dspark-verdict-measurement-2026-08-26.md:59-60; docs/PERFORMANCE_HISTORY.md:4865 | 256-token fixed-window | PROVENANCE-WEAK on the percentage: ratio of the displayed medians is 28.30/27.49 = +2.95%, not +1.87%; the +1.87% basis is not recoverable from the stated pair |
| decode tok/s (stage 2 width-5, pre-A/B) | 27.57 @100K (−3.1% vs baseline 28.46); 22.13 @352.6K (−11.7% vs 25.07) → REVERT | 100K / 352.6K | 2026-08-25 | none stated | docs/dspark-mtp-ab-preregister-2026-08-25.md:844-845 | decode (usage+events recorded) | none |
| decode tok/s (stage 2b width-3) | +13.3% @100K over a 12.46 s truncated window = NOT clean (retracted as inflated; later full result 27.57) | 100K | 2026-08-25 | commit 433dce6c1 | docs/dspark-mtp-ab-preregister-2026-08-25.md:844,893-897,1126-1134 | decode (truncated window) | PROVENANCE-WEAK + CONVENTION-AMBIGUOUS (early-stop truncated window; doc explicitly flags) |
| decode tok/s @352.6K (pre-fix gate, collapses included) | ON n=16 median 19.87 (4/16 collapsed to ~1; healthy runs 16.9-31.1) vs stripped-OFF n=9 median 24.19; gate −17.87% FAIL | 352.6K | 2026-08-27 | none stated | docs/dspark-352k-memory-regression-2026-08-27.md:169-174; docs/PERFORMANCE_HISTORY.md:5023-5026 | fixed-window decode @352.6K | none |
| decode tok/s (residency forced-OFF control) | 23.92 / 23.45 / 23.68, zero collapses | 352.6K | 2026-08-28 | none stated | docs/dspark-352k-memory-regression-2026-08-27.md:181 | fixed-window decode | none |
| decode tok/s (verbon3 arm C validation) | 8 runs: 26.29, 34.77, 28.83, 28.05, 30.52, 30.89, 27.97, 19.50 → median 28.44 vs stripped-OFF 24.19 = +17.57%, CI [+8.7,+27.8]% | 352.6K | 2026-08-28 | EXO_DSV4_DSPARK_TP_SHARD (exo 2d85ccdcb); JIT fix 75d2402dd | docs/dspark-352k-memory-regression-2026-08-27.md:191; docs/dspark-mtp-production-baseline-2026-08-27.md:84-86 | fixed-window decode, 0/8 collapses | none |
| decode tok/s (soak) | 30.46 @4000/4000 tokens, finish=length, median gap 0.08 ms | 352.6K | 2026-08-28 | same | docs/dspark-352k-correctness-harness-verification-2026-08-28.md:106; docs/dspark-mtp-master-history-2026-08-28.md:51,245-246 | soak aggregate decode | none |
| decode tok/s (combined B0 spec-off) | 28.95 fixed-window | 352.6K | 2026-08-28 | none stated | docs/dspark-352k-correctness-harness-verification-2026-08-28.md:104; docs/dspark-352k-verification-preregister-2026-08-28.md:51 (28.95 bar line) | fixed-window decode | none |
| decode tok/s (P3 verbon3 ablation arms) | arm-BOTH 30.482 median; arm-SHARD 30.590 (σ=0.019); arm-CACHE 29.008 (σ=0.018); CLEAR_CACHE=64 = −0.35%; TP_SHARD = +4.84% | 352.6K shared window W=2097 | 2026-08-29 | production TP_SHARD=1 stack | docs/dspark-mtp-master-history-2026-08-28.md:344-350; docs/PERFORMANCE_HISTORY.md:5245-5290 | fixed-window decode, 3 runs/arm | none |
| decode tok/s (P4b rowseq crossover) | batched 41.3 vs rowseq 31.3 @14K; 39.3 vs 28.7 @32K (end-to-end corroboration) | 14K / 32K | 2026-08-29 | same production stack | docs/dspark-mtp-master-history-2026-08-28.md:395-396 | end-to-end decode | CONVENTION-AMBIGUOUS ("end-to-end corroboration" — decode window basis not spelled out) |
| decode tok/s (c=2 spec-ON) | 10.01-10.07 (stream A) / 9.65-9.71 (stream B), aggregate 19.66-19.78; vs c=1 spec-ON 37.7 | 100K | 2026-08-28 | none stated | docs/dspark-p2-c2-validation-results-2026-08-28.md:22,51; docs/dspark-mtp-master-history-2026-08-28.md:332 | per-stream decode W=831 | CONFLICT — c=1 spec-ON @100K: 37.7 here vs 36.63 median (2026-08-27 promotion) — same metric/context, different values (mean vs median / different probe; both real) |
| decode tok/s (draft-epilogue A/B) | −0.35% @100K, −0.26% @352.6K (byte-lossless, gate FAIL) | 100K / 352.6K | 2026-08-28 | exo `EXO_DSV4_DRAFT_EPILOGUE` (stays OFF) | docs/dspark-p1-draft-epilogue-ab-results-2026-08-28.md:44; docs/dspark-mtp-master-history-2026-08-28.md:294 | fixed-window decode | none |
| decode tok/s (rollback cost fix) | 29.0-29.3 held; rollback 0.79 ms mean | 100K c=1 | 2026-07-12 | mlx-lm f00a9a9; exo d96b11b44 | docs/rollback-cost-campaign-handoff-2026-07-12.md:1,11 | decode | none |
| decode tok/s (M1 shadow gate) | 8.51 @100K — shadow-mode designed cost, NOT a regression (retractable) | 100K | 2026-08-24 | exo 6dcd61e4a / mlx-lm 0854b39 (machine-verified state) | docs/p4v2-m1-shadow-gate-results-and-recovery-2026-08-24.md:41-49; docs/PERFORMANCE_HISTORY.md:3785-3787 | decode incl. shadow draft+verify | CONVENTION-AMBIGUOUS (shadow mode inflates by construction; not production throughput) |
| verify cost (per-cycle ms, not tok/s) | γ_mean=3.31 @100K: 99 ms (20.2 ms/row); T6 baseline 34.25 ms/token | 100K | 2026-08-24 | none stated | docs/dspark-mtp-master-history-2026-08-28.md:152 | per-cycle verify | CONVENTION-AMBIGUOUS (ms/row, included for completeness: current decode wall 29.2 tok/s; ceiling 37.6 kernel-fix-only = derived, excluded) |

## 14. Vision-Exp era: vec-vs-loop and decode depth ladder (DSv4-Flash-Vision-Exp, 2026-09-22)

| metric | value | ctx depth | date | commit/tag | source (path:line) | convention | flags |
|---|---|---|---|---|---|---|---|
| decode tok/s | vec (prod default) 27.39 vs loop 27.33 (+0.2%, noise) on Vision-Exp | short ctx, 400-tok gens, temp 0 | 2026-09-22 | mlx-lm pinned 5c5328b (merge-base holds 095c98c); exo 4b9932322 present | docs/vec-divergence-verified-fixed-and-depth-is-the-real-lever-2026-09-22.md:52-54 | decode-only? | CONVENTION-AMBIGUOUS (doc does not state decode-only vs e2e for these 400-tok arms) |
| decode tok/s | generation length effect at identical depth: 400 tok = 27.08 vs 800 tok = 33.20 | same | 2026-09-22 | same | docs/vec-divergence-verified-fixed-and-depth-is-the-real-lever-2026-09-22.md:93-94 | decode-only? | CONVENTION-AMBIGUOUS (same basis as above) |
| decode tok/s | 29.5 (loop, retired -0731 checkpoint) → 33.7-35.5 (vec, same comparison) | short ctx | 2026-07-12 (retired checkpoint) | none stated | docs/vec-divergence-verified-fixed-and-depth-is-the-real-lever-2026-09-22.md:56-57 | decode | CONFLICT vs Vision-Exp 27.39/27.33 — cross-checkpoint comparison, documented as non-transferable |
| decode tok/s | 2,216→29.17; 21,855→33.22; 68,002→31.78; 115,614→36.74 (needle verified, decode_s 20.6-16.3 s) | 2.2K/21.9K/68K/115.6K | 2026-09-22 | none stated | docs/decode-flat-across-depth-36tps-at-115k-2026-09-22.md:33-36 | server-reported decode_tps (probe's own), decode-only | none — but SUPERSEDED at depth (below) |
| decode tok/s + prefill tok/s | 2,216→29.17/213.5; 21,855→33.22/397.6; 68,002→31.78/422.3; 115,614→36.74/423.3; 273,622→26.83/409.0; 564,926→13.36/371.8 (mem 115.66 GB = 95.9% of wired limit) | six-rung ladder | 2026-09-22 | none stated | docs/decode-depth-ladder-to-565k-memory-wall-2026-09-22.md:15-20 | server decode_tps + prefill tok/s from probe | CONFLICT (565K value CONFLICT-qualified below) |
| decode tok/s (WRONG harness, retracted example) | 27.03 / 23.26 / 18.96 / 15.75 / 12.31 (wall-clock incl. prefill) vs correct decode-only 27.12 / 25.65 / 24.39 / 24.91 / 28.83 | 20 / 605 / 1,771 / 3,521 / 7,021 | 2026-09-22 | none stated | docs/decode-flat-across-depth-36tps-at-115k-2026-09-22.md:15-19 | wall-clock tok/s (WRONG, documented as measurement bug) | PROVENANCE-WEAK (retracted instrument; kept as the canonical anti-example) |
| decode tok/s (content swing) | 30.5 (prose) to 45.9 (list-reverse) at fixed ~10.5K depth | ~10.5K | 2026-09-22 | none stated | docs/decode-depth-ladder-to-565k-memory-wall-2026-09-22.md:32-33 | decode-only | none (documented content confound) |
| decode tok/s | 250K reproduces at 26.72-26.83 (two independent runs); 273K 26.8; 565K ~13.4 | 250K/273K/565K | 2026-09-22 | none stated | docs/deep-context-refault-mechanism-confirmed-2026-09-22.md:6,98 | decode_tps | none |
| decode tok/s | 273,622 → 26.83 (earlier); 564,926 → 13.36 (earlier, peak 115.66 GB); same night: 564,928 → 24.15 | 273.6K / 565K | 2026-09-23 | none stated | docs/overnight-results-two-mechanisms-and-stochastic-collapse-2026-09-23.md:13-14,38-39; docs/overnight-final-indexer-hypothesis-falsified-corrected-curve-2026-09-23.md:75 | decode_tps | CONFLICT/bimodal — 565K measured 13.36 and 24.15 within one night (stochastic collapse band; both real) |
| decode tok/s | 284,831 → 28.94 (250K target rung) | 285K | 2026-09-23 | none stated | docs/overnight-results-two-mechanisms-and-stochastic-collapse-2026-09-23.md:80; docs/overnight-final-indexer-hypothesis-falsified-corrected-curve-2026-09-23.md:67 | decode_tps | none |
| decode tok/s (~30K real value, n=4) | 34.89 ± 0.64 (the 46.30 t/s earlier reading was a single outlier; retracted) | ~30K | 2026-09-23 | none stated | docs/OVERNIGHT-CLOSEOUT-FINAL-2026-09-23.md:86; docs/overnight-final-indexer-hypothesis-falsified-corrected-curve-2026-09-23.md:22; docs/overnight-corrections-dspark-sizing-and-content-driven-tps-2026-09-23.md:42-45 | decode_tps, ladder rungs 31-34 flat 30K→122K | CONFLICT — 46.30 (outlier, retracted) vs 34.89 (real) |
| decode tok/s (5-rung top-k ladder) | 45K/60K/76K/98K/121K → smooth ~30-34, no step (indexer-cliff hypothesis falsified) | 45K-121K | 2026-09-23 | none stated | docs/overnight-final-n6-250k-bistability-2026-09-23.md:64; docs/OVERNIGHT-CLOSEOUT-2026-09-23.md:73 | decode_tps | none |
| decode tok/s (n=6 250K mean) | 26.18 mean with 1 collapse in 6 | ~250K | 2026-09-23 | none stated | docs/overnight-ratchet-depth-specific-prefix-cache-good-2026-09-23.md:60 | decode_tps | none |
| decode tok/s | 28.99 @276,427 and 27.70 @273,626 (target-b final re-measurement); cycle-cost decomposition 30K = 34.10 t/s / 56.5 ms-cyc vs 273K = 27.29 t/s / 69.5 ms-cyc | 276.4K / 273.6K / 30K | 2026-09-23 | tree at origin/main edc6b43c4 | docs/OVERNIGHT-CLOSEOUT-FINAL-2026-09-23.md:50; docs/OVERNIGHT-CLOSEOUT-2026-09-23.md:47-49 | decode_tps | none |
| decode tok/s | 26.68 (freshboot non-collapse run); stochastic-band runs 16.43 / 16.83 in same sweep | ~250K | 2026-09-23 | none stated | docs/freshboot-control-inverts-model-chunk-size-dominant-2026-09-23.md:68; docs/OVERNIGHT-CLOSEOUT-2026-09-23.md:44 | decode_tps | none (stochastic slow mode band) |
| prefill tok/s (freshboot control) | fresh + 2048 chunks = 370.3; fresh + 128 chunks = ~148 | deep feed | 2026-09-23 | none stated | docs/freshboot-control-inverts-model-chunk-size-dominant-2026-09-23.md:77-78 | fresh feed | none |
| decode tok/s (incomparable workload) | 9.73 @270,816 tokens (needle task, non-comparable to the 27-29 baseline) | 270.8K | 2026-09-23 | none stated | docs/target-b-attribution-sdpa-compressed-not-indexer-2026-09-23.md:63 | decode | CONVENTION-AMBIGUOUS (different workload; doc says so) |
| decode tok/s (adaptive-prefill experiment) | 25.75 @638K (peak 87.87 GB, best result; ~2/3 boot state / ~1/3 chunk per decomposition) | 638K | 2026-09-23 | tree at edc6b43c4 | docs/OVERNIGHT-CLOSEOUT-FINAL-2026-09-23.md:43-44 | decode_tps | none |
| decode tok/s (post-fix check) | 13.12 @565K — inside the pre-fix collapse band; fix NOT sufficient | 565K | 2026-09-23 | none stated | docs/verify-565k-fix-not-sufficient-2026-09-23.md:16 | decode_tps | none |
| decode tok/s (FENCE_ASYNC claim, quoted) | 28.9 → 37.0 t/s c=1 "per its own A/B (2026-07-02 in source comment)" | — | 2026-07-02 (quoted second-hand) | none stated | docs/OVERNIGHT-CLOSEOUT-2026-09-23.md:52-53; docs/fence-async-28pct-claim-traced-to-artifact-2026-08-22.md:66; docs/overnight-250k-cycle-cost-and-fence-status-2026-09-23.md:33 | unspecified | PROVENANCE-WEAK + CONFLICT — traced to artifact; the real fence measurement was 18.5→29.2-30.9 (+58-67%, short ctx) on 2026-08-22; no matching 28.9→37.0 raw run exists in docs |

## 15. P-series decode bench (2026-08-31 → 09-03) and the 20-vs-34 t/s mystery (Vision-Exp era)

| metric | value | ctx depth | date | commit/tag | source (path:line) | convention | flags |
|---|---|---|---|---|---|---|---|
| decode tok/s | P06 Phase A (lm_head mxfp8 SHIP, 3 reps/arm, 1200-tok windows): ON 34.35/34.12/34.28 (median 34.28) vs OFF 32.35/33.09/32.11 (median 32.35) = +1.93 tok/s, +6.0%; prefill unchanged 378.3 vs 378.0 | 100K prefill / 1200-tok decode window | 2026-08-30 | none stated | docs/PERFORMANCE_HISTORY.md:5842,5863-5867 | decode-only, long_decode_probe, needle | none |
| decode tok/s | P11 measured median 30.71 = −10.41% vs 34.28 reference (decode gap re-audit) | same | 2026-08-31 | none stated | docs/PERFORMANCE_HISTORY.md:7345-7355 | decode-only | none |
| decode tok/s | P12 (QUERY_TILED_SDPA + EXACT_TOPK_PREFILL ON) 32.18 vs P13 (both OFF) 36.08; gap 3.90 tok/s; flags worth 34.76 (QTILE) + ~4.30 (TOPK) tok/s of prefill; prefill medians 417.21 → 378.15 (−9.36%) | same bench depth | 2026-08-31→09-01 | P06 mxfp8 stack | docs/PERFORMANCE_HISTORY.md:7715,7614-7676 | decode-only medians; prefill TTFT | CONVENTION-AMBIGUOUS |
| decode tok/s | P05 32.49 (NOT-THIS-FLAG); P09-era 32.06; outlier boot 30.06; between-boot spread 6.02 tok/s; P15 warmup 12.00 (rejected) | same bench depth | 2026-08-31→09-03 | none stated | docs/PERFORMANCE_HISTORY.md:7732,7791,7929,7958-7986,8080 | decode-only medians, between-boot A/B | PROVENANCE-WEAK (boot-to-boot variance dominates sub-~3 tok/s deltas; doc mandates relabeling) |
| decode tok/s (entropy study) | natural high-entropy prose 33.58 = pooled repetitive 33.58; random letter-strings 28.66 (−12.6%) | matched depth | 2026-09-01 | none stated | docs/PERFORMANCE_HISTORY.md:8138 | decode-only | PROVENANCE-WEAK on the percentage: ratio of the displayed pair is 28.66/33.58 = −14.7%, not −12.6%; the −12.6% basis is not recoverable |
| decode tok/s (temperature study) | temp=1.0 32.535 vs temp=0.8 pooled 33.579 (−3.11%, Welch p=0.101) | matched depth | 2026-09-01 | none stated | docs/PERFORMANCE_HISTORY.md:8157 | decode-only | none |
| decode tok/s (real-session vs bench) | real session decode @140-160K n=16: 34.06 mean vs benchmark 31.84 @150K (+2.22) | 140-160K | 2026-09-01 | none stated | docs/PERFORMANCE_HISTORY.md:8169 | decode-only (both) | none |
| decode tok/s (convention reconciliation) | same traffic measured as: decode-only 33.00; +TTFT/prefill 22.44; full wall 21.82; visible-answer-only 9.04 | 140-160K | 2026-09-01 | none stated | docs/PERFORMANCE_HISTORY.md:8172-8175 | same payload, four conventions | CONVENTION-AMBIGUOUS (intentionally — this is the convention-clarification measurement) |
| decode tok/s (wrong-model artifact) | 271.4/345.1/276.8 (median 276.8) "100K decode" + "decode 200-223 tok/s" + 469.6/322.8 off 97-token samples | 100K | 2026-08-30 | none stated | docs/PERFORMANCE_HISTORY.md:5798-5806,5853-5858 | unknown instrument | PROVENANCE-WEAK + RETRACTED (wrong-model harness artifact; 97-token samples; explicitly discredited) |

## 16. LMHEAD_MXFP8 / DSpark-native era (Vision-Exp, 2026-09-13/14)

| metric | value | ctx depth | date | commit/tag | source (path:line) | convention | flags |
|---|---|---|---|---|---|---|---|
| decode tok/s (DSpark-native regression/fix) | broken local head 10.59/10.97/10.97 → native head 27.25/31.09/32.31; at 100K decode recovered to 36-43 | ~2K-class bench + 100K | 2026-09-12/14 | EXO_DSV4_DSPARK_NATIVE default flipped (805afbb018bad838b114d08035907d7df470e2ac) | docs/PERFORMANCE_HISTORY.md:9426-9436 | decode (bench) | CONVENTION-AMBIGUOUS (bench depth rows not fully stated; 100K row is separate) |
| decode tok/s (mxfp8 merge validation) | 38.62 post-merge vs 37.61 pre | 100K | 2026-09-13 | mxfp8 merge | docs/incidents/lmhead-mxfp8-cross-lingual-glue-defect-2026-09-13.md:101; docs/PERFORMANCE_HISTORY.md:9470 | decode-only, live A/B | none |
| decode tok/s (mxfp8 cost re-measure) | 37.05 / 37.65 vs 38.62 mxfp8-on baseline (~2.5-4% cost); fresh remeasure bf16-only 39.06 (2-3 tok/s run-to-run variance) | 100K-class | 2026-09-13/14 | none stated | docs/incidents/lmhead-mxfp8-cross-lingual-glue-defect-2026-09-13.md:101-106; docs/lmhead-mxfp8-defect-and-fallback-investigations-2026-09-14.md:147-148 | decode-only live | CONVENTION-AMBIGUOUS (2-3 tok/s natural variance documented) |
| decode tok/s (conditional fallback candidate) | 37.38 live vs fresh bf16 baseline 39.06 | same | 2026-09-14 | fallback patch | docs/lmhead-mxfp8-defect-and-fallback-investigations-2026-09-14.md:147-148,317 | decode-only live A/B | none |

## 17. Incidents: vision regression, c=2 drift, GPU throttle, launcher (2026-09-16→09-28)

| metric | value | ctx depth | date | commit/tag | source (path:line) | convention | flags |
|---|---|---|---|---|---|---|---|
| decode tok/s (vision control triple) | A2 text 11.21; B1 IMAGE 14.10; C1 text-after 11.11 | ~2K-class | 2026-09-21 | none stated | docs/incidents/hang-watchdog-false-positive-and-vision-regression-2026-09-21.md:567-569 | decode (controlled A/B) | none |
| decode tok/s (turn-2 with image in history) | no-image 23.46 vs image-in-history 24.36 (no real regression) | same boot | 2026-09-21 | none stated | docs/incidents/hang-watchdog-false-positive-and-vision-regression-2026-09-21.md:584-585 | decode | none |
| decode tok/s (GPU throttle resolved) | ~4-7 → 33.3 (293 tokens / 8.8 s) on standard text request | same | 2026-09-21 | none stated | docs/incidents/hang-watchdog-false-positive-and-vision-regression-2026-09-21.md:555-555 | decode | none |
| decode tok/s (degenerate-content probes) | 3.5 ("count 1-5"), 6.2 ("numbers 1-20"), 0.95 (~157 s) on same boot vs 20.44/20.66 regression-probe baseline | ~2K | 2026-09-21 | none stated | docs/incidents/hang-watchdog-false-positive-and-vision-regression-2026-09-21.md:397-408 | decode, short max_tokens | PROVENANCE-WEAK (degenerate short outputs; doc warns these are not comparable) |
| c=2 spec-verify drift incident | per-stream 5.171 ± 0.012; AGG 10.342 ± 0.023; prefill 215.87 ± 1.13 tok/s | c=2 | 2026-09-16 | none stated | docs/incidents/c2-spec-verify-structural-drift-2026-09-16.md:56-58 | decode-only + prefill | none |
| decode tok/s (launcher incident sanity) | 21.8 measured vs documented 20.86 mean baseline | ~2K | 2026-09-28 | none stated | docs/incidents/ssh-quoting-executed-cluster-launcher-2026-09-28.md:61 | decode (server-reported) | none |

## 18. DSv4-Flash "500K ladder" serving handoff (July 2026 era)

| metric | value | ctx depth | date | commit/tag | source (path:line) | convention | flags |
|---|---|---|---|---|---|---|---|
| prefill tok/s + decode tok/s | 4.7K: 152/27.6; 32K: 151/28.0; 64K: 152/27.4; 128K: 152/24.2; 256K: 150/21.3; 500K: 141.6/17.2 (c=1, MTP on) | ladder | 2026-07-06 | none stated | docs/dsv4-c2-serving-handoff-2026-07-06.md:441-450 | fresh feed prefill; decode-only | CONVENTION-AMBIGUOUS (decode basis not spelled out; values consistent with decode-only convention) |
| decode tok/s | post-fence-gating: 29.5 → 38.3/37.4 (+29%) at 4K rung band | 4K | 2026-07-02-06 era | none stated | docs/dsv4-c2-serving-handoff-2026-07-06.md:662,725-727 | decode | none |
| decode tok/s | 586K rung: prefill 331, decode 25.94 (MTP) → 27.04 after exact top-k (+4.2%); interpolation ~26.5 @500K | 586K | 2026-07-06/07 | mlx-lm 4d87751 (EXO_DSV4_EXACT_TOPK) | docs/dsv4-c2-serving-handoff-2026-07-06.md:985,1028,1031 | prefill fresh; decode-only | none |
| decode tok/s | needle workload at TRUE 500K runs 28.94 vs word-count 27.04 @586K | 500K/586K | 2026-07-07 | same | docs/dsv4-c2-serving-handoff-2026-07-06.md:1159-1160 | decode | CONVENTION-AMBIGUOUS (workload-dependent; doc states) |
| c=2 decode t/s/stream | ~20 @MTP-eract; 7.8 @1500-tok pairs; MTP-on 14.3 stream (28.6 agg incl prefill); pools-correct 15.3 | 100K c=2 | 2026-07-06 | none stated | docs/dsv4-c2-serving-handoff-2026-07-06.md:27,112,415; docs/dsv4-rowseq-followups-plan-2026-07-10.md:7 | per-stream | CONVENTION-AMBIGUOUS (aggregate vs per-stream mixing) |
| KV quant micro-measure | bf16 KV 11.4 vs 4-bit KV 11.9 tok/s/stream (c=2 100K MTP=0, sparse-attn regime) | 100K c=2 | 2026-05-09 | start_cluster.sh knobs commit 65655ced | docs/kv-cache-architecture.md:159-160 | per-stream decode | none |

## 19. TurboQuant / quantization decode rows (2026-06, DSv4-Flash)

| metric | value | ctx depth | date | commit/tag | source (path:line) | convention | flags |
|---|---|---|---|---|---|---|---|
| decode tok/s | baseline (4-bit dequant + fused SDPA) 40.9; TurboQuant 4-bit qmm 37.0 (−10%); 3-bit qmm 38.3 (−6%); 3-bit dequant+inverse-rotate+fused SDPA 36.5 (−11%); ALL REGRESSED | not stated in rows | 2026-06 | none stated | docs/turboquant-integration.md:691-694 | decode | CONVENTION-AMBIGUOUS + PROVENANCE-WEAK (rows do not state depth/window; era c=2 family, PH §7:1652-1728 lists related values incl. c=2 14.3 MTP-on 28.6 agg) |
| decode tok/s (coding-eval samples) | 4bit 35.8; 4bit-qkv 35.6; nvfp4 35.6-37.0; arch_design 31.5-31.7; debug_fix 34.9 | coding-eval tasks | 2026-06 | none stated | docs/PERFORMANCE_HISTORY.md:1690-1694 | decode across samples | CONVENTION-AMBIGUOUS |

## 20. DSv4.1 benchmark campaign (2026-09-26 → 10-01)

| metric | value | ctx depth | date | commit/tag | source (path:line) | convention | flags |
|---|---|---|---|---|---|---|---|
| decode tok/s (server-reported) | 10 runs: 20.01-21.82, median 20.72, mean 20.86, σ=0.659 | 2000 (2331 prompt rows) | 2026-09-26 | none stated | docs/benchmarks/phase0-v4flash-2k-2026-09-26/README.md:47 | server generation_tps, decode-only (1400-tok windows) | none |
| prefill tok/s | 10 runs: 313.15-372.00, median 334.78, mean 337.83 | 2000 | 2026-09-26 | same | docs/benchmarks/phase0-v4flash-2k-2026-09-26/README.md:48 | fresh feed | none |
| plain decode tok/s (DSv4.1 first TP=2) | 9.25-9.30 (107-108 ms/step); try2 9.02 | 23-token prompt | 2026-09-28 | MLX fork 0.32.3.dev20260918+603f16eb7 | docs/benchmarks/phase14-dsv41-first-tp2-decode-2026-09-28/README.md:13,30-32 | plain greedy decode | none |
| plain decode tok/s (body speedup) | 9.25 → 16.32 tok/s (fused hc + engram async + TP head split) | short prompt | 2026-09-28 | mlx-lm feat/dsv41-exl3 @ 14d1c74 | docs/benchmarks/phase15-dsv41-body-speedup-2026-09-28/README.md:9,34 | plain greedy decode | none |
| spec decode tok/s | plain 16.5; γ=2 22.3; γ=3 22.7; γ=4 20.2; per-prompt @γ3: 21.6/25.4/20.9 | 96-tok windows | 2026-09-28 | same branch | docs/benchmarks/phase16-dsv41-dspark-spec-2026-09-28/README.md:13-18 | mean over 3 prompts | none |
| spec decode tok/s (verify-cost round) | γ3 23.5 (per-prompt 22.4/26.5/21.6); plain 16.9; γ2/3/4 = 23.3/23.5/21.2 (run 3) | same | 2026-09-28 | same | docs/benchmarks/phase17-dsv41-verify-cost-2026-09-28/README.md:8-9,59 | mean over 3 prompts | none |
| one-sync + adaptive gamma | 22.9 mean; +1.1 tok/s mean on 2 of 3 prompts (γ per prompt) | 96-tok windows | 2026-09-28 | same | docs/benchmarks/phase18-dsv41-adaptive-gamma-2026-09-28/README.md:1,16,20 | mean tok/s | none |
| prefill tok/s (exo integration gate; DSv4.1) | 2,048: 9.9 (incl. first-run compile); 8,192: 73.7; 16K: crashed (527 GPU timeouts) | 2K/8K/16K | 2026-09-28 | none stated | docs/benchmarks/phase19-dsv41-exo-integration-gate-2026-09-28/README.md:23-25 | fresh feed | CONFLICT (production V4 comparison "~430 @100K" stated in :27-28; see CONFLICT note vs old-stack 359-366) |
| prefill/decode (M1 window) | 8K: 130.4 prefill / 1.6 decode right after; 16K: 72.7 / 5.0 (collapse explained in phase21 as compile) | 8K/16K | 2026-09-28 | mlx-lm feat/dsv41-exl3 acf9b55 | docs/benchmarks/phase20-dsv41-m1-window-2026-09-28/README.md:9-10 | fresh feed + post-prompt decode | none (decode value retracted/explained by phase21) |
| prefill/decode (context scaling) | 8K 129-130; 16K 73→129 with 512-row chunks; steady post-prompt decode 12.8-13.2 @8K/16K | 8K/16K | 2026-09-29 | same branch | docs/benchmarks/phase21-dsv41-context-scaling-2026-09-29/README.md:10-11,33-34 | fresh feed + steady-state decode | none |
| prefill tok/s (served production through API) | 2K: 218.6 (TTFT 9.389 s); 8K: 223.9 (TTFT 36.598 s); 16K: refused (16384 cache cap) | 2K/8K/16K | 2026-10-01 | exo 6909d121, mlx-lm ad6f9d3 | docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/README.md:58-62 | TTFT to first content token, fresh feed, prefix cache off | none |
| prefill tok/s (harness) | 8K: 239.1/245.6 (run) and 237.8/244.1 (run2, cold); 16K: 239.6/238.4 and 238.1/237.4 (±1% @16K, 3.28% @8K) | 8K/16K | 2026-10-01 | same | docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/README.md:84-87 | engine_prefill harness, fresh feed | none |
| prefill tok/s (chunk-size lever) | chunk 4096: 8K 256.3/253.9 (+9.7%/+4.9%), 16K 249.4/248.2 (+7.1%); chunk 8192 @16K 229.9/229.6 (worse); peaks 115.3-117.4 GB | 8K/16K | 2026-10-01 | same | docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/README.md:148-149 | harness, fresh feed | none |
| prefill tok/s (ops sweep) | MLX_MAX_OPS_PER_BUFFER 24: 235.1/240.5; 5: 236.0/245.4 & 240.6/242.6 — neutral vs 200 band 237.8-246.2 / 237.4-239.6 | 8K/16K | 2026-10-01 | same | docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/README.md:190-194 | harness | none |
| TTFT restore check | 16K-prompt 2K TTFT re-measure 9.75 s vs 9.39 s pre-window | 2K | 2026-10-01 | same | docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/README.md:254-255 | TTFT through API | none |
| other: SSD read bandwidth (iostat) | 6206 MB/s (m4-1) / 6284 MB/s (m4-2), p10-p90 6064-6314; soak p25 medians n2 6243 vs n1 6076 | — | 2026-09-28 | none stated | docs/benchmarks/phase10-planb-gate-2026-09-28/README.md:45-46; docs/benchmarks/phase11-stage1-fstar-2026-09-28/README.md:28 | device iostat, 35 / 39 min soak | other-metric row (the doc's 26.9/39.3-40.0 tok/s "ceilings" are DERIVED from this; excluded per rule (a)) |
| decode cycle rate (production Vision-Exp, Prometheus-derived) | 972.1 tok / 44 samples = 22.09 tok/s; 2.056 tok/cycle → 93.1 ms/cycle | production serving | 2026-09-28 | none stated | docs/benchmarks/phase11-stage1-fstar-2026-09-28/raw/p41-stage1-fstar.md:25-26 | cycle aggregate rate from session records | PROVENANCE-WEAK + CONVENTION-AMBIGUOUS (documented as DERIVED anchor; no raw run preserved in repo) |
| decode ceiling (MoE-only microbench) | 40-layer MoE-only decode ceiling 24.1 → 34.0 tok/s | microbench | 2026-09-28 | none stated | docs/benchmarks/phase9-exl3-spike-2026-09-28/README.md:30 | kernel microbench | CONVENTION-AMBIGUOUS (component-level, not end-to-end) |
| port projection (9.84) | plain decode/token 101.6 ms → 9.84 tok/s (phase3 port measurement) | 8-layer harness | 2026-09-27 | none stated | docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/README.md:124 | single-node harness | CONVENTION-AMBIGUOUS (8-layer subset ≠ full model; downstream falsified as EXL3-build figure in phase8:23,51) |

## 21. 1M-context / deep prefill campaign (DSv4.1 1M serving, 2026-10-03 → 10-07)

NOTE: this section's prefill rows carry a documented convention split — "fresh" = full feed from empty session; "delta" (r500/r750/r1m) = incremental prefill of the remaining rows on top of an existing session; also "wire tok/s" vs "rows/s" corrections (PH:10890-10913). Rates are quoted as tok/s (wire) except where the note says rows/s.

| metric | value | ctx depth | date | commit/tag | source (path:line) | convention | flags |
|---|---|---|---|---|---|---|---|
| prefill tok/s (full re-prefill mislabeled as delta) | r160 (160K cold): 782.8 s = 204 tok/s; r500 "110 tok/s at 500K"; r750 "75 tok/s at 750K"; r1m ≈56 tok/s — ALL LATER DECLARED ARTIFACTS (1-row-chunk frame bug) | 160K/500K/750K/1M | 2026-10-04→10-05 | build 7a3096e85 era | docs/PERFORMANCE_HISTORY.md:10173 (r750), :10300-10301,10303,10374-10390 (corrections) | full feeds mis-read as delta | PROVENANCE-WEAK + CONFLICT — artifact numbers 110/75/56 vs corrected framefix numbers below. NOTE: rows/s ≡ tokens in this source (PH:10398 "100,001 tok in 407.7 s = 245.3 rows/s"), so the 110 tok/s vs 103.6 rows/s pair IS a genuine same-unit conflict, not a unit mismatch. |
| prefill tok/s (wedge era) | 268.7 wedge check; then ~56 tok/s effective on same feed shape | 160K | 2026-10-05 | none stated | docs/PERFORMANCE_HISTORY.md:10330-10333 | fresh feed | PROVENANCE-WEAK (declared driver-state wedge artifact) |
| prefill rows/s (framefix, delta) | r500 delta completes at 103.6 rows/s (pre-fix 2 h+ stall) | 500K delta | 2026-10-05 | framefix build | docs/PERFORMANCE_HISTORY.md:10415 | incremental delta feed, rows/s | none |
| prefill (delta 100K→200K, M2 HIER A/B) | wire: 338.8 → 390.2 tok/s (+15.2%); base 160K build 714.9 s = 223.8 tok/s | 100K→200K delta | 2026-10-06 | next4 build (sync spans 0207d7c0f/66e7344) | docs/PERFORMANCE_HISTORY.md:10484,10523-10528 | incremental delta, wire tok/s | CONVENTION-AMBIGUOUS (wire vs row rate — corrected globally at PH:10890-10913; the two bases share a floor) |
| prefill (M2 deep-delta) | +26.7% at 160K→500K delta (win grows with depth) | 500K delta | 2026-10-06 | M2 default-ON (mlx-lm main) | docs/PERFORMANCE_HISTORY.md:10514 | delta | none |
| prefill tok/s (fresh comparator) | fresh 100K: 243.0 vs 243.2 (M2 arm comparator) | 100K | 2026-10-06 | M2 build | docs/PERFORMANCE_HISTORY.md:10544-10545 | fresh full feed | none |
| prefill tok/s (soak rungs, final build era) | r160 cold: 720.5 s / 222.1 (M2 build) → 688.7 s / 232.3 (+4.6%, sparse-attn build) → 675.4 s / 236.9 (+2.0%) → 646 s / 247.7 (soak12 final) → 608 s / 263.2 (+6.3%, consumer-skip) → 266.4 (post-final −0.9%) | 160K | 2026-10-06→07 | next4 → next6 → next7 → next9 8d4fbdb60/mlx-lm 872e0e4 → next13 | docs/PERFORMANCE_HISTORY.md:10606,10743,10844,11069,11237,11296-11297 | fresh full feed (r160) | none |
| prefill tok/s (fresh 100K session arc) | 222 → 274.0 (+23.4%) → 281.0 (+2.6%) → 283.0 (+0.7%); feed 378.9 s/100,001 rows = 263.9 | 100K | 2026-10-06→07 | same progression | docs/PERFORMANCE_HISTORY.md:10938,11055,11197,11296 | fresh full feed | none |
| prefill (delta rows/s, consumer-skip) | r500: 143.1 → 186.2 rows/s (+30.1%); r750: 146.4 rows/s (+57.3%); r160 +8.6%; fresh +2.6%; depth ladder +6.3%/+29.0%/+57.3%/+65.5% | 160K-1M delta | 2026-10-07 | consumer-skip shipped (5a986da; next13) | docs/PERFORMANCE_HISTORY.md:11189,11212,11243,11262,11523 | incremental delta, rows/s | none |
| prefill tok/s (MoE_ALLSUM_BF16 lever) | fresh +4.1% → promoted default-ON | 100K fresh | 2026-10-07 | bf16 all_sum promotion | docs/PERFORMANCE_HISTORY.md:11013 | fresh feed | none |
| prefill (chunk 4096 on new stack) | chunk=4096: fresh 263.6 (+4.4%), delta 413.3 (+2.9%) (PV32=0); on final stack a wash | 100K fresh + delta | 2026-10-06→07 | next8-era builds | docs/PERFORMANCE_HISTORY.md:10816,11289 | fresh + delta | CONVENTION-AMBIGUOUS (delta value is wire tok/s; see PH:10890 correction) |
| prefill (prefix-cache contamination) | fake 1914 tok/s "fresh" reading (reused filler matched parked session); comparator read 255.6 @128.8K actually contaminated | ~129K | 2026-10-06 | none stated | docs/PERFORMANCE_HISTORY.md:10976-10982,11067 | fresh-looking but cache-hit | PROVENANCE-WEAK + RETRACTED (instrument artifact; doc documents the fake reading explicitly) |
| delta/wire correction row | "delta 418 tok/s" is a wire number, not a row rate; real floor shared across bases | — | 2026-10-06 | none stated | docs/PERFORMANCE_HISTORY.md:10890-10913 | convention correction | CONVENTION-AMBIGUOUS (kept as the doc's own flag) |

## 22. next17/next18 decode latency rounds (2026-10-08/09) — known-good decode baselines

| metric | value | ctx depth | date | commit/tag | source (path:line) | convention | flags |
|---|---|---|---|---|---|---|---|
| decode latency + t/s (agentic replay) | production f4bb14746: 145.75 ms/round = 25.5 t/s (benign 20K g3); agentic 91K = 157.10 ms = 19.5 t/s | 20K / 91K (800-tok rounds, γ=3) | 2026-10-08 | exo f4bb14746 (before) | docs/known-good-decode-baseline-20261008.md:11; docs/PERFORMANCE_HISTORY.md:11437 | median ms/round over 8/6 reps; t/s = derived from round latency incl. verify | CONVENTION-AMBIGUOUS (t/s is round-derived end-to-end, not steady-state decode-only) |
| decode latency + t/s (next17 shipped) | benign 118.56 ms = 31.5 t/s; agentic 130.07 ms = 23.7 t/s; HIER=0 split: agentic 101.06 ms = 30.2 t/s | same | 2026-10-08 | tag known-good-decode-next17-20261008-160746; exo 576e9d279 + mlx-lm 3bf8316; branch deploy/phase20-campaign @ 0b10977ab | docs/known-good-decode-baseline-20261008.md:12-13,26; docs/PERFORMANCE_HISTORY.md:11438-11439 | median ms/round | CONVENTION-AMBIGUOUS (same basis as above) |
| decode t/s (post-ship parity smoke) | benign 3×20K g3 = 118.60 ms median / 31.4 t/s (reproduces 118.56 to 0.04 ms) | 20K | 2026-10-08 | same | docs/known-good-decode-baseline-20261008.md:20; docs/PERFORMANCE_HISTORY.md:11453-11454 | median ms/round | CONVENTION-AMBIGUOUS |
| decode latency + t/s (P2 bf16-vs-fp32 all_sum A/B) | bf16 (=1, shipped): benign 118.53 ms / 32.36 t/s; agentic 129.90 ms / 23.62 t/s; fp32 (=0): 121.21 ms / 31.06 t/s; agentic 133.22 ms / 23.63 t/s | 20K / 91K | 2026-10-08 | production 576e9d279 + mlx-lm 3bf8316 | docs/PERFORMANCE_HISTORY.md:11509-11510 | median ms/round, cross-boot A/B | CONVENTION-AMBIGUOUS (round-derived t/s) |
| decode latency + t/s (R1b lever-2 relaunch) | agentic 91K: 130.38 → 99.30 ms (−31.08, disjoint); benign 118.86 → 94.93 ms | 20K/91K | 2026-10-08 | L2-full build | docs/PERFORMANCE_HISTORY.md:11574-11575 | median ms/round | none |
| decode latency + t/s (R1c ship round) | benign 20K: 94.94 ms / 39.13 t/s; agentic 91K: 99.46 ms / 31.03 t/s (reproduces R1b) | 20K/91K | 2026-10-09 | tag known-good-decode-next18-20261009-001052; exo fb4f9290b (<user>) + mlx-lm 16830e1 | docs/PERFORMANCE_HISTORY.md:11608-11609 | median ms/round parity smoke | CONVENTION-AMBIGUOUS |
| other: dense EXL3 achieved BW | m=1 GEMV 69.3 GB/s, m=4 GEMM 53.4 GB/s (real weights); read-roofline 497 GB/s measured | — | 2026-10-08 | dense track | docs/PERFORMANCE_HISTORY.md:11588,11534 | kernel microbench | other-metric row |

## 23. MiniMax + upstream comparison rows (2026-04 era)

| metric | value | ctx depth | date | commit/tag | source (path:line) | convention | flags |
|---|---|---|---|---|---|---|---|
| decode tok/s (MiniMax-M2.7) | pre-blocks baseline 18.60; MLX_SDPA_BLOCKS=88 + fused attn: 24.26 (min 24.19 max 24.32, n=5); v2 25.44 (+4.9%); fused SDPA 32.92 (+35.7%) | 50K c=1 | 2026-04-24 | none stated | docs/minimax-rdma-moe-validation-2026-04-24.md:10,14,118,129,144 | decode-only, mean of 5 | none (different model family, exo repo history) |
| decode tok/s (MiniMax post-session) | 26.14 → 27.86 @50K (MLX_SDPA_BLOCKS=88); 8-bit KV 27.79 vs bf16 27.84 (zero effect) | 50K | 2026-04-24 | exo 6ae331fe forwards | docs/minimax-decode-optimization.md:276-277,187 | decode-only | none |
| decode tok/s (c=2 era, DSv4-Flash 6bit) | fork 34.6 c=1 on 2× M4 Max RDMA TP vs rlt PR 30.1 single-node | 100K-class | 2026-04-26 | mlx-lm pin <user>/mlx-lm:main (Blaizzy variant) | docs/upstream-prs.md:47,165 | decode c=1 | PROVENANCE-WEAK (comparative claims in PR table; upstream-sync re-validate 30.7 c=1 100K MTP+γ=2 at :146; 27.3 median at :120) |
| decode tok/s (sparse SDPA rewrite) | 32K: 29.3 → 29.6 (+1.0% noise); 100K: 27.3 → 27.7 (+1.5%); matmul rewrite 4.2× wall at 32K (415→98 s, decode unchanged 28.8) | 32K/100K | 2026-04-28 era | mlx-lm 87f4625 | docs/upstream-prs.md:195; docs/fork-notes.md:185-187 | decode-only | none |
| decode tok/s (fork-notes May sweep) | c=1 100K: 15.4 (control) → 17.6 (step128 era) → 29.47 (+91%; mean 29.47 ± 0.08/±0.05 across restarts); prefill end-of-loop 243.6-243.8 @89,407 | 100K | 2026-05 | lock pins 98ebfde → 87f4625 → f4dd9e7 | docs/fork-notes.md:588,677-678,816,894 | decode-only + prefill | none |

# Notes / caveats by file group

**§1 baselines**
- The "500K" depth label in the DSv4-Flash benches is a nominal fill target: the real prompt landed at 352.599K (docs/p3-worker-b1-live-depth-anchors-2026-08-23.md:204-204). Every "500K" prefill/decode figure in §1-§5 is therefore at ~352.6K real depth. Quoting "500K" without this note re-labels the number.
- 2026-08-21 decode figures (17.48/18.60/17.26) were measured under the silently-broken async fence — real at the time, not representative after 2026-08-22 (docs/PERFORMANCE_HISTORY.md:160-165).
- 339 @500K "earlier-campaign baseline" is cited as a parity anchor only; no raw run is preserved (PROVENANCE-WEAK).

**§2-§5 prefill/kernel ABs**
- Corpus-standard prefill convention = prompt tokens / TTFT wall ("prefill tok/s = offline-tokenized prompt tokens / TTFT" — docs/hc-collapse-kernel-ab-2026-08-25.md:247, docs/hc-expand-kernel-ab-2026-08-24.md:344). Fresh full feed unless a doc says chunk-loop.
- hc_expand gain decays with depth (+3.87% @70.5K → +1.28% @300K → +0.85% @500K-target); hc_collapse does not (+1.89→1.97% flat). Do not quote the +3.87% headline for prefill generally.
- Decode cells inside the hc kernel A/Bs came from 1.5-2.5 s windows — the docs themselves call them "structurally weak" (CONVENTION-AMBIGUOUS).

**§7-§8 PP era**
- The July PP "450-class" numbers are chunk-loop instantaneous rates on the preview affine-8bit checkpoint with a chars//4 1.42x-inflated numerator; the live 2026-08-24 honest rates are TP 366.5 vs PP 277 end-to-end at the same depth checkpoint (docs/p5-tp-prefill-gap-2026-08-24.md:577-601). Never quote PP 364-512 or TP 319-427 without that frame.
- PP per-session fast-path decode (22-24 tok/s) was repeatedly retracted; only the ~0.5-0.65 slow band is robust.

**§9-§10 DSpark/MTP**
- Fixed-window (256-tok) decode is the corpus metric for spec A/Bs; whole-generation tok/s differs. The 2026-08-27 promotion number (36.63) is a fixed-window median vs spec-OFF 27.15 (same-window) — do not compare it to whole-generation rungs (41.3 @14K corroboration rows) without saying so.
- CONFLICT: c=1 spec-ON @100K appears as 36.63 (24-pair median, docs/dspark-mtp-production-baseline-2026-08-27.md:18) and 37.7 (c=2-validation doc's c=1 reference, docs/dspark-p2-c2-validation-results-2026-08-28.md:51). Both measured; different probe/statistic.
- CONFLICT: pre-promotion rowseq median @100K ~28.3 vs spec-OFF 27.15 vs fixed-window arm medians 28.30/27.49 — same era, related but not identical statistics; the 28.3 figure is master-history's rowseq-column value (doc :200).

**§13-§14 352K / Vision-Exp**
- 565K decode is bimodal (13.36 vs 24.15 same night — stochastic collapse band, docs/overnight-results-two-mechanisms-and-stochastic-collapse-2026-09-23.md:38-39). Any single 565K/250K number must carry the band.
- The "28.9 → 37.0 t/s" FENCE_ASYNC figure is a source-comment A/B (2026-07-02) with no preserved raw run and was traced to an artifact; the 2026-08-22 real fence measurement is 18.5 → 29.2-30.9 (+58-67%, short ctx) (docs/fence-async-28pct-claim-traced-to-artifact-2026-08-22.md:66). Mixing these two in a writeup would be wrong.
- real-session vs bench decode at matched depth: 34.06 (n=16 real) vs 31.84 (bench @150K) — decode engine fine; perceived ~20 t/s is the full-wall convention (docs/PERFORMANCE_HISTORY.md:8169-8175).

**§20 DSv4.1 benchmarks**
- phase19/21 claim "Production V4 prefills ~430 tok/s at 100K" as the reference for the NEW model-era production (Vision-Exp measured 397.6-423.3 @21K-115K in the 2026-09-22 ladder); the OLD DSv4-Flash-0731 stack measured 359.7-366.6 @100K. Two checkpoints; both real; do not blend.
- phase22 production-vs-harness gap (218.6-223.9 served vs 237.8-246.2 harness @8K) comes from API/TTFT overhead; the README states it.
- The phase10 "26.9 / 39.3-40.0 tok/s" figures are bandwidth-derived ceilings (iostat GB/s ÷ MB/token), not run throughputs — excluded from rows; only the measured GB/s and trace hit rates were kept (rule (a)).

**§21 1M soak**
- "110/75/56 tok/s" soak-1 numbers are declared artifacts (mixed chunk-pool frames); corrected family is the 2026-10-05 framefix + consumer-skip rows/s ladder. Delta rates and fresh rates must never be mixed in one curve (docs/PERFORMANCE_HISTORY.md:10374-10413).
- Wire-vs-row correction (418 wire ≠ row rate) is the doc's own standing caveat.

**§22 next17/18**
- t/s in these rows = tokens-per-round / median round latency; it includes verify and round overheads and is NOT the decode-only steady-state rate used elsewhere — flag CONVENTION-AMBIGUOUS on every row in this group.

**General cross-checks performed**
- Conflicting-value pairs were captured as separate rows with CONFLICT/flags: (1) decode @500K/nominal 21.51 vs 23.48 @352.6K (methodological gap, docs/p3-synthesis-500k-decode-decay-decomposition-2026-08-23.md:41-43); (2) gate+up fusion −3.8% (c=2, 2026-06-26) vs +3.01% (c=1, 2026-08-21); (3) c=1 spec-ON 36.63 vs 37.7; (4) 565K 13.36 vs 24.15; (5) ~30K 46.30-outlier vs 34.89 real; (6) FENCE_ASYNC +28% claim vs artifact-traced reality; (7) soak-1 110/75/56 vs corrected framefix rows/s; (8) vec speedup July (-0731) 33.7-35.5 vs Vision-Exp wash 27.39/27.33; (9) PP-era prefill curve pre-renormalization vs honest 225/214/202; (10) fake 1914 tok/s prefix-cache reading vs real comparators.
- Estimates, targets, projections and roofs (e.g., "predicted +5-12%", roofline % ceilings, phase5/phase10 I/O ceilings, deepseek-v41-exl3-plan ±40% planning ranges, draft-epilogue theoretical +16%, P4a projected tok/s) were excluded per rule (a); the two bandwidth GB/s microbenches and one iostat row kept are labeled "other" metrics, not tok/s.

### D1.3 Convention conflicts and weak-provenance items (do not merge these in a writeup)

- CHECKPOINT SPLIT: "Production V4 prefills ~430 tok/s at 100K" (DSv4.1-phase docs) refers to the
  Vision-Exp production stack (ladder rows 397.6-423.3 @21K-115K), NOT the DSv4-Flash-0731 /
  EXL3 lineage (359.7-366.6 @100K). Both are real; blending them is an error.
- DEPTH LABEL: every T1-era "500K" row is NOMINAL. Real prompt length 352,599 tokens. Decode
  @nominal-500K is reported as 21.51 tok/s (2026-08-22 fenceasync, docs/PERFORMANCE_HISTORY.md:77)
  and as 23.48 tok/s @352,599 (docs/p3-worker-b1-live-depth-anchors-2026-08-23.md:26;
  docs/p3-synthesis-500k-decode-decay-decomposition-2026-08-23.md:38).
- PREFILL @nominal-500K appears as 331.6 (08-21, docs/PERFORMANCE_HISTORY.md:154), 324.1 (08-22,
  :74), 333.91 (hc-expand OFF 08-24), 339.12 (hc-collapse OFF 08-25) - different days, configs and
  depth residuals.
- "FENCE_ASYNC +28%, 28.9 -> 37.0 t/s": PROVENANCE-WEAK. It survives only in a 2026-07-02 source
  comment, quoted second-hand in three docs, and was traced to an artifact
  (docs/fence-async-28pct-claim-traced-to-artifact-2026-08-22.md:66). The real 2026-08-22 fence
  measurement is 18.5 -> 29.2-30.9 short-context, +58-67%
  (docs/async-fence-fix-validated-2026-08-22.md:13-15).
- MoE gate+up fusion: CONFLICT, documented unresolved - 37.2 -> 35.8 (-3.8%) at c=2 2026-06-26
  (docs/dsv4-decode-stall-2026-06-26.md:175-176) vs +3.01% (18.879 vs 18.328, c=1 short-ctx,
  2026-08-21) (docs/moe-gate-up-fusion-validated-2026-08-21.md:44-45);
  docs/PERFORMANCE_HISTORY.md:1234-1237 records the contradiction as unresolved.
- 565K decode is BIMODAL (13.36 vs 24.15 same night). Any single-point 565K figure is unsafe.
- known-good-decode-20261008 rows (D1.1 #15-17) are median ms/round over 800-token rounds and
  INCLUDE verify. They are not decode-only steady state and must not be placed in the same column
  as rows 4-14.
- Retracted artefacts catalogued in the corpus (do NOT reuse): 271.4/345.1/276.8 tok/s "100K decode"
  from 97-token samples (2026-08-30); a 1914 tok/s "fresh" reading that was a prefix-cache hit;
  46.30 t/s single outlier vs real 34.89 +/-0.64 @~30K; soak-1 deep-prefill 110 tok/s @500K / 75 @750K
  declared an artifact of a 1-row-chunk framing bug; May "champions" 31.5/32.3 tok/s not reproducible
  (32.3 cratered to 4.3).

---

## D2. FORK DIVERGENCE COUNTS

Command form: `git rev-list --count <upstream-ref>..HEAD` and `... --no-merges <upstream-ref>..HEAD`,
run in each repo at the timestamp above. Upstream ref per repo is the remote that points at the
real upstream (exo: `upstream`; mlx: `origin` = ml-explore/mlx; mlx-lm: `upstream`).

| repo | path | branch | HEAD SHA | upstream ref | upstream SHA | commits ahead | commits ahead, --no-merges |
|---|---|---|---|---|---|---|---|
| exo | /Users/<user>/repos/exo | deploy/next28 | 5a18b81debe3e4d1b82370d473614d0b4be82bf2 | upstream/main (exo-explore/exo) | bc412e2230f395e652c23426a5eb88dfa515769b | 2607 | 2568 |
| mlx | /Users/<user>/repos/mlx | integrate/upstream-20261016 | b177818386215ea578c16f8d8de06706e06bcff8 | origin/main (ml-explore/mlx) | b6642e604843776705b57b1eaecb6f57e519b793 | 277 | 253 |
| mlx-lm | /Users/<user>/repos/exo/mlx-lm | main | 676f803ed28f2703adb7a16f74a407baccf05066 | upstream/main (ml-explore/mlx-lm) | 93e6d4a6447b3080861985947f0820e76fcb957d | 527 | 485 |

Note on mlx: `mlx` exists BOTH as a standalone checkout at /Users/<user>/repos/mlx and as the
git submodule of exo at /Users/<user>/repos/exo/mlx. `git submodule status` in exo reports
`b177818386215ea578c16f8d8de06706e06bcff8 mlx` - the same SHA as the standalone checkout HEAD, so
the two are consistent. mlx-lm's gitlink is `676f803ed28f2703adb7a16f74a407baccf05066`.

Commits touching mlx's jaccl mesh.cpp specifically
(`git log --oneline -- mlx/distributed/jaccl/lib/jaccl/mesh.cpp`):

| measure | value |
|---|---|
| total commits touching that file (all history) | 33 |
| of those, in the fork divergence `origin/main..HEAD` | 32 |

(The file lives at `mlx/distributed/jaccl/lib/jaccl/mesh.cpp` - the upstream-refactored
`lib/jaccl` layout, not a top-level `mlx/jaccl/`. A path lookup for `mlx/jaccl/mesh.cpp` returns 0
commits and would silently understate the work.)

exo working tree at capture: one untracked directory, `bench/dsv41_quality_battery/results/`
(`git status --porcelain`); no tracked modifications from this task (read-only).

---

## D3. PINNED SHAs (deployed cluster)

From `~/repos/exo/uv.lock` (last modified 2026-10-16 18:35; 540,955 bytes):

| package | pin | uv.lock line |
|---|---|---|
| mlx (darwin) | version 0.32.3.dev20261002+b1778183, source git https://github.com/<user>/mlx.git?branch=main#b177818386215ea578c16f8d8de06706e06bcff8 | uv.lock:1382-1384 (also :438, :1522) |
| mlx-lm | source git https://github.com/<user>/mlx-lm.git?branch=main#676f803ed28f2703adb7a16f74a407baccf05066 (version declared 0.32.0) | uv.lock:1499-1501 |
| exo | HEAD 5a18b81debe3e4d1b82370d473614d0b4be82bf2, branch deploy/next28, 2026-10-16 12:05:58 -0500, subject "deploy(next28): stage-2 upstream sync - exo merge 9fe0719a + mlx b1778183 + mlx-lm 676f803e" | git rev-parse HEAD |

The mlx/mlx-lm uv.lock SHAs MATCH the submodule gitlinks recorded in exo (verified above), which is
the invariant pyproject.toml:117-119 says CI enforces
(`src/exo/shared/tests/test_submodule_lock_pin_alignment.py`): the committed gitlink and the uv.lock
pin must be the same SHA.

From `~/repos/exo/pyproject.toml`:
- line 90: `{ git = "https://github.com/<user>/mlx.git", branch = "main", marker = "sys_platform == 'darwin'" }`
- line 120: `mlx-lm = { git = "https://github.com/<user>/mlx-lm.git", branch = "main" }`
- lines 91-100: on Linux the mlx package comes from prebuilt wheels on
  `rltakashige/mlx-jaccl-fix-small-recv` (mlx 0.32.0, CUDA 12/13 variants) - i.e. the darwin pin is
  the fork; the linux pin is a third-party wheel. Relevant if the writeup says "mlx is our fork".

Live check, read-only SSH (`~/repos/exo/.venv/bin/python` on node0):
- `mlx 0.32.5.dev20261016+b17781838`  (note: import-time version string differs from the uv.lock
  declared `0.32.3.dev20261002+b1778183`; the git SHA suffix b17781838 is the same commit)
- `mlx-lm 20261017.dev46+g676f803ed`
- `exo 0.3.70`

---

## D4. MEMORY BUDGET MATH

Source of truth for the model config: the checkpoint on both nodes,
`~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw/config.json`
(read via read-only SSH; 4,482 bytes, mtime 2026-09-24 00:40, identical on both nodes).

### D4.1 Weight bytes

| quantity | value | source |
|---|---|---|
| on-disk checkpoint, measured | `du -sk` = 205,739,672 KiB = 210,677,420,032 B = **210.68 GB** (196.22 GiB readable); 39 safetensors shards (corrected 2026-10-18 from an earlier miscount of 40; see REPRO-RUN-2026-10-18.md) | `du -sk` on node0 |
| on-disk checkpoint, as stated in docs | **210.58 GB** ("196.2 GiB = 210.7 GB" in `docs/deepseek-v41-exl3-plan.md:116`) | 0.1 GB discrepancy vs the measured 210.68 GB — NOT rounding: 210.58 GB = 196.12 GiB and 196.2 GiB = 210.67 GB, so the doc's GB/GiB pairing is internally inconsistent. Which mechanism produced it is unrecoverable (raw gone); likely conversion or transcription. |
| per-rank weights at TP=2, measured from shard headers + the sharding rule the running code applies | **108.76 GB/rank** | `docs/deepseek-v41-memory-budget-correction-2026-09-27.md:22` |
| per-rank under "experts split, everything else replicated" | 112.56 GB/rank | same doc:21 |
| naive file/2 | 105.29 GB/rank | same doc:23 |

Per-rank breakdown (same doc:43-49): routed experts 98.02 GB (width-sharded); MTP/DSpark 3 DSpark
stages 3.84 GB (width-sharded); shared expert 0.40 GB (width-sharded); attention + embed + head +
router + vision + other 6.50 GB (REPLICATED). `DeepseekV4ShardingStrategy`
(`src/exo/worker/engines/mlx/auto_parallel.py`) width-shards three things, not one: routed experts,
shared expert, and `mtp.ffn.*` plus (under `EXO_DSV4_DSPARK_TP_SHARD`, default 1 since 2026-08-27)
every DSpark stage's `.ffn`.

### D4.2 KV bytes at 1M tokens - the MLA math, done explicitly

Assumption set (all from the live checkpoint's config.json unless noted):
- text_config: `num_hidden_layers = 40`, `head_dim = 512`, `window_size = 128` (sliding_window),
  `index_head_dim = 128`, `index_n_heads = 32`, `index_topk = 512`, `max_position_embeddings = 1048576`.
- `num_key_value_heads = 1`. There is NO `kv_lora_rank` key: V4/V4.1 use the MLA absorbed form, one
  shared 512-d KV latent per position, not per-head K/V. Confirmed in code:
  `mlx_lm/models/deepseek_v41/config.py:34` ("attention (MLA): one shared 512-d KV vector per
  position") and `mlx_lm/models/deepseek_v41/cache.py:80`
  (`self.win_kv = mx.zeros((bsz, self.window, args.head_dim), ...)`).
- `compress_ratios` (first 40 of 43) = `[0,0, 2 x18 (layers 2-19), 1 x20 (layers 20-39)]`.
- `kv_source_layer_ids = [2, 8, 14, 20]` -> ratios: layer 2 = 2, layer 8 = 2, layer 14 = 2, layer 20 = 1.
- `index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]` -> ratios 2,2,2,1,1,1,1,1.
- Storage dtype: the quantized-grid buffers `win_kv`, `comp_kv`, `index_k` are stored in **bf16
  always** (2 bytes/element), even though the values are pre-quantized (win_kv fp8-e4m3, comp_kv fp4
  e2m1 latents x e4m3 scales, index_k fp4 e2m1 x ue8m0 scales). Rationale in code: bf16 is lossless
  for <=6-significant-bit products. `mlx_lm/models/deepseek_v41/cache.py:19-30`.
- Consumer layers hold a reference to their source's buffer, so `comp_kv` is stored once per source,
  not once per layer.

At 1,048,576 tokens, per session, per node (cache is NOT TP-sharded - it is replicated; see
`docs/dsv41-restart-rankfix-buildnote-2026-10-11.md:55-65`):

| component | rows | formula | bytes |
|---|---|---|---|
| window ring (`win_kv`), all 40 layers | 128 per layer | 40 x 128 x 512 x 2 B | 5,242,880 B = 5.0 MiB (constant in L) |
| compressed KV (`comp_kv`), 3 sources at ratio 2 | ceil(1M/2) = 524,288 each | 3 x 524,288 x 512 x 2 B | 1,610,612,736 B = 1.50 GiB |
| compressed KV (`comp_kv`), 1 source at ratio 1 | 1,048,576 | 1 x 1,048,576 x 512 x 2 B | 1,073,741,824 B = 1.00 GiB |
| index keys (`index_k`), 3 sources at ratio 2 | 524,288 each | 3 x 524,288 x 128 x 2 B | 402,653,184 B = 0.375 GiB |
| index keys (`index_k`), 5 sources at ratio 1 | 1,048,576 each | 5 x 1,048,576 x 128 x 2 B | 1,342,177,280 B = 1.25 GiB |
| **total, one 1M session** | | | **4,434,427,904 B = 4.13 GiB** | PROVENANCE-WEAK + DERIVED (arithmetic from config dims; no raw run behind it, as L356) |

Cross-check against the code's own claim: `cache.py:29` states "a 1M session's compressed/index
caches are ~3.2 GiB in bf16, ~6.4 GiB fp32". CONVENTION-AMBIGUOUS: 3.2 GiB is reproduced exactly
if `index_k` is counted only on the 4 kv-source layers (3 x ratio-2 + 1 x ratio-1 = 0.625 GiB, total
3.13 GiB ~= 3.2 GiB) rather than on all 8 `index_source_layer_ids` (which gives 4.13 GiB). Which
convention the running build uses was NOT resolved by this task; treat "1M KV" as 3.2-4.2 GiB and
say which counting rule you used.

Other KV-per-token figures that appear in the corpus, with their conventions:
- "FP4 main KV at 890 B/token" for V4.1 (`docs/deepseek-v41-exl3-plan.md:68,261;
  docs/deepseek-v41-memory-budget-correction-2026-09-27.md:98`). At 1M that is ~0.93 GB. This is a
  different (smaller) accounting than the table above; the derivation is not given in the docs.
- ~6.9 KB/token resident, measured on the V4-Flash production config
  (`docs/p3-reviewer-r1-verification-2026-08-23.md:272`: "6936 / 6896 / 6891 B/token"), giving
  2.43 GB at 352.6K (`docs/p3-synthesis-500k-decode-decay-decomposition-2026-08-23.md:69`).
- 11.7 KB/token cold-prefill slope measured live
  (`docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md:12292-12296`).
- A "naive dense-MLA" upper bound of 43 layers x 512 dims x 2 B = 44 KB/token
  (`docs/p3-reviewer-r1-verification-2026-08-23.md:276-278`); the corpus also records that a claimed
  "~74x smaller" ratio was REFUTED and the real ratio is 6.39x. Use 6.39x, not 74x.
- `docs/decode-roofline-dispatch-bound-2026-08-21.md:26-27` "3.56 GB/token/node" is ACTIVE WEIGHT
  bytes per decode step, not KV. Do not file it under KV.

### D4.3 Activation / runtime overhead estimate

No direct measurement of activation memory was found. The corpus carries an ESTIMATE range, stated
as borrowed from V4-Flash plus V4.1 additions (Engram hot-row LRU 2-4 GB, FP4 KV, indexer/candidate
scratch): **16 GB (best) to 22 GB (worst) per rank** (
`docs/deepseek-v41-memory-budget-correction-2026-09-27.md:88-99`). Label ESTIMATE, not measurement.
`docs/deepseek-v41-exl3-plan.md:259` gives the older pair "72 GB weights per rank, 88 GB peak, so
~16 GB per rank of KV, DSpark, indexer scratch and MLX allocator overhead".

### D4.4 Headroom per node

Live readings (read-only SSH, both nodes, same timestamp):

| quantity | node0 | node1 |
|---|---|---|
| `sysctl hw.memsize` | 137,438,953,472 B = 128 GiB | same |
| `sysctl iogpu.wired_limit_mb` | **120000** (120 GB) | **120000** |
| `top -l 1` PhysMem | 127G used (109G wired, 1010M compressor), 94M unused | 127G used (109G wired, 1084M compressor), 92M unused |
| largest exo python process RSS | 16,294,080 KB = 15.5 GiB (pid 25115) | 15,658,288 KB = 14.9 GiB (pid 35168) |

CONFLICT: every doc in the corpus pins `iogpu.wired_limit_mb = 115000` -> 120.6 GB
  - UNIT RULE: this pack renders MiB as GB throughout. Under that rule 115000 MiB = 120.6 GB and the live 120000 MiB = 125.83 GB. The L634 cell's bare "120 GB" is the decimal-MB reading of the same number; both are stated, the MiB rule is used for arithmetic.
(`docs/PERFORMANCE_HISTORY.md:7436`, `docs/decode-depth-ladder-to-565k-memory-wall-2026-09-22.md:22`,
`docs/deepseek-v41-exl3-plan.md:52`, `docs/overnight-closeout-levers-exhausted-2026-09-23.md:95`).
The LIVE value on both nodes is 120000. Any "% of wired limit" figure copied from a doc is computed
against 120.6 GB, whereas live is 120 GB - a ~0.5% difference, but state which one you use.

Arithmetic, every term labelled:
- usable wired budget/node = 120 GB (live sysctl; docs say 120.6 GB).
- weights/node = 108.76 GB (measured, D4.1).
- KV/node at 1M tokens = 3.2-4.2 GiB per session (D4.2), i.e. 3.4-4.5 GB decimal.
- activation + scratch + allocator = 16-22 GB (ESTIMATE, D4.3).
- => headroom = 120 - 108.76 - 3.4..4.5 - 16..22 = **-8.2 GB (best case) to -15.3 GB (worst case)**.
  The budget does NOT close at 1M context. This independently reproduces the gap column in
  `docs/deepseek-v41-memory-budget-correction-2026-09-27.md:88-95` (gap 8.84 / 10.84 / 12.84 /
  14.84 GB for 16/18/20/22 GB overhead) to within the KV-term convention difference, and it is why
  that doc prescribes streaming the coldest 9.0-15.1% of expert bytes.
- Live corroboration of the pressure: both nodes show 127G used of 128 GiB with ~1 GB unused while
  serving.

---

## D5. HARDWARE FACTS

Read-only SSH, captured at the timestamp at the top of this file.

| field | node0 | node1 |
|---|---|---|
| Model Name | Mac Studio | Mac Studio |
| Model Identifier | Mac16,9 | Mac16,9 |
| Model Number | MODELNO | MODELNO |
| Chip | Apple M4 Max | Apple M4 Max |
| Total cores | 16 (12 performance, 4 efficiency) | 16 (12 performance, 4 efficiency) |
| Memory | 128 GB | 128 GB |
| System Firmware Version | 20457.1.29 | 20457.1.29 |
| OS Loader Version | 20457.1.29 | 20457.1.29 |
| `sw_vers` ProductVersion | macOS 27.0 | macOS 27.0 |
| `sw_vers` BuildVersion | 26A428 | 26A428 |
| `uname -a` kernel | Darwin 27.0.0, root:xnu-13432.1.9~1/RELEASE_ARM64_T6041, built Tue Aug 11 21:05:48 PDT 2026 | same |
| `hostname` | node0 | node1 |
| `scutil --get LocalHostName` | node0 (DRIFTED from hostname - see conflicts) | node1 (DRIFTED) |
| `scutil --get ComputerName` | node0 | (not captured) |
| Hardware UUID | HWUUID-A | HWUUID-B |
| Serial Number (system) | SERIAL-A | SERIAL-B |
| Activation Lock Status | Enabled | Disabled |
| Thunderbolt RDMA device names | rdma_en3, rdma_en4 (both `status: active`) | same |
| TB/USB4 bus speeds seen | Bus 3 up to 120 Gb/s (no device); Bus 2 and Bus 1 80 Gb/s, device connected | (not captured in detail) |
| TB MAC-ish addresses (en3 / en4) | aa:bb:cc:00:00:04 / aa:bb:cc:00:00:05 | (not captured) |

MLX / stack versions: see D3 - live `mlx 0.32.5.dev20261016+b17781838`,
`mlx-lm 20261017.dev46+g676f803ed`, `exo 0.3.70`, pinned in uv.lock to mlx b1778183 / mlx-lm 676f803e.

Thunderbolt RDMA device names and the max_qp evidence (grep of docs/ for `ibv_devinfo`):

| claim | evidence |
|---|---|
| two RDMA-capable cables, `rdma_en3` + `rdma_en4`, both PORT_ACTIVE | docs/dual-cable-topology-and-qp-budget-2026-08-21.md:73; docs/phase3-cluster-validation-blocked-resolved-2026-08-20.md:16; docs/DSV4_VISION_PORT_PLAN_PHASE34.md:29 |
| Apple's TB RDMA HCA reports `max_qp = 3` PER DEVICE, verified live via `ibv_devinfo -v` on both nodes (`max_qp:3, max_cq:6, ...`) | docs/dual-cable-topology-and-qp-budget-2026-08-21.md:61-62; docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md:7697, :14029, :14200 |
| so a second cable doubles the QP budget 3 -> 6 per peer | docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md:14313, :14406 |
| the correct one-cable pairing is ASYMMETRIC: node0 reaches node1 via its own rdma_en3, node1 reaches node0 via its own rdma_en4 | docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md:14476-14480, :14606 |
| no RC and no UD transport available; UC only | docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md:14220-14228 |
| live launcher/bench scripts write `[[null, "rdma_en3"], ["rdma_en4", null]]` to `p47_ibv.json` | docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/scripts/p22_prefill_launch.sh:14; docs/benchmarks/phase14-dsv41-first-tp2-decode-2026-09-28/scripts/p47_launch.sh:5 |

CONFLICT to flag: `hostname` (node0/2) disagrees with `LocalHostName`
(M4-4/M5) on the same boxes; the ssh config comment records the mDNS names drifting and the aliases
being repointed to Tailscale FQDNs on 2026-10-03 (`~/.ssh/config`), and
`docs/launcher-address-drift-repair-2026-09-28.md:22` records the same drift. macOS version also
drifts across the corpus: docs cite 26.4 (25E241) in `docs/iogpu-residency-set-abort.md:5,74`,
macOS 27.0 (26A5388g) in `docs/incidents/2026-08-09-section17-concurrency-wedge-sample.txt:48`, and
live is 27.0 (26A428).

---

## D6. SCRUB LIST (private details to redact before publishing)

Exact strings. "present in repo" = the string already appears in tracked repo files (counted with
`grep -rIl` over exo/docs + exo/AGENTS.md + exo/start_cluster.sh + exo/pyproject.toml).

Hostnames / mDNS names:
- `node0`, `node1` (present in repo: 7 files; e.g.
  docs/p3-worker-c-attn-kernel-walltime-2026-08-23.md:195, docs/launcher-address-drift-repair-2026-09-28.md:22)
- `node0`, `node1` (drifted LocalHostNames, live)
- `node0` (ComputerName, live)
- ssh aliases `node0`, `node1` (present: 53 files)
- ssh aliases `node0`, `node1`, `node0-lan`, `node1-lan`
- `node0.local`, `node1.local` (compat shims in ~/.ssh/config)

Tailscale / VPN:
- tailnet suffix `tailnet.ts.net` (present: 1 file, docs/launcher-address-drift-repair-2026-09-28.md:75)
- FQDNs `node0.tailnet.ts.net`, `node1.tailnet.ts.net`
- tailnet IPs `100.64.0.1` (m4-1, 12 files), `100.64.0.2` (m4-2, 5 files), `100.64.0.3`
  (macbook-m4), `100.64.0.4`, `100.64.0.5` (work laptop)
- host alias `gateway`, `gateway.tailnet.ts.net`

LAN IPs:
- `192.168.1.201`, `192.168.1.202` (launcher-hardcoded studio IPs; present: 12 files, e.g.
  docs/handoff-2026-08-08-section22.md:35, docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md:5203)
- `192.168.1.48` (node0), `192.168.1.47` (node1) - current DHCP leases
- `192.168.1.74` (stale lease recorded in docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md:5209)
- `100.64.0.6`, `100.64.0.7`, `100.64.0.8`, `100.64.0.9`, `100.64.0.10`,
  `100.64.0.11`, `100.64.0.12`, `100.64.0.13`, `100.64.0.14`, `100.64.0.15` (other
  entries in ~/.ssh/known_hosts), `10.0.0.2`, `172.16.0.1`, `172.16.0.10-12`, `172.16.0.20`,
  `172.16.0.250`, `172.16.0.40-50`, `192.168.1.2`, `[homeassistant.local]:2222`

MAC addresses (live):
- `aa:bb:cc:00:00:01` (en0, m4-1), `aa:bb:cc:00:00:02` (en1)
- `aa:bb:cc:00:00:03` (en2), `aa:bb:cc:00:00:04` (en3, TB/RDMA), `aa:bb:cc:00:00:05` (en4, TB/RDMA),
  `aa:bb:cc:00:00:06` (en5)
- `aa:bb:cc:00:00:07`..`aa:bb:cc:00:00:0a` (en6-en9)

Hardware identifiers (live, and NOT in the repo - 0 files):
- Serial Numbers `SERIAL-A` (m4-1), `SERIAL-B` (m4-2)
- Hardware UUIDs `HWUUID-A` (m4-1), `HWUUID-B` (m4-2)
- Provisioning UDIDs `UDID-A` (m4-1), `UDID-B` (m4-2)
- Model Number `MODELNO`

Thunderbolt device / domain UUIDs (live, from `system_profiler SPThunderboltDataType`, m4-1):
- TB bus UIDs `TBUID-1`, `TBUID-2`, `TBUID-3`
- Domain UUIDs `TBDOM-1` (Bus 3),
  `TBDOM-2` (Bus 2), `TBDOM-3` (Bus 1),
  `TBDOM-4` (peer Mac Studio on Bus 2)
- TB Service UUIDs `TBSVC-1`, `TBSVC-2`

Local username in paths:
- `adam.durham` (macOS short user on the studios; appears in 63 files as `/Users/<user>/...`)
- `adam` (ssh User in ~/.ssh/config)
- Any `/Users/<name>/...` path in docs (5+ files, e.g. docs/PERFORMANCE_HISTORY.md)

Email / committer identity:
- global git email `<user>@example.com` (~/.gitconfig)
- exo repo-local committer `<user> <<user>@example.com>` (`git config user.email`)
- author name `<user>`

Other identifiers to consider redacting:
- GitHub org/user fork URLs `github.com/<user>/exo`, `github.com/<user>/mlx`, `github.com/<user>/mlx-lm`
- `rltakashige/mlx-jaccl-fix-small-recv` (third-party wheel source pinned in uv.lock/pyproject)
- key filenames `~/.ssh/cluster_key`, `~/.ssh/cluster_key.pub`, `~/.ssh/id_ansible`
- 1Password agent socket path `~/Library/Group Containers/<team>.com.1password/t/agent.sock`

Note: the two serial numbers, hardware UUIDs, provisioning UDIDs and TB domain UUIDs are NOT present
in the repo (grep count 0). They would leak only if the writeup pastes `system_profiler` output,
which is exactly what D5 above contains - scrub D5 before publishing.

---

## D7. OPEN PROBLEMS (genuine unresolved items)

Extracted by grepping docs/ for TODO / unresolved / open / known issue / not yet fixed / still open /
INCONCLUSIVE, plus a read of PERFORMANCE_HISTORY.md section 13 "Open / never-finished threads"
(around line 2261). Roadmap/plan language and anything the doc itself marks resolved or superseded
was excluded. Items where a later doc closes something an earlier doc listed open are called out in
D9.


## D8. NEGATIVE RESULTS (closed investigations)

# Writeup evidence pack (child): negative results + open problems -- exo docs

Source corpus: /Users/<user>/repos/exo/docs/ (read-only). Built 2026-10-16.
Method: read PERFORMANCE_HISTORY.md sections 7.2, 7.3, 9.0, 13 (lines 2261-3914) and the
Quick-reference closed-lever table (lines 3916-3948), then grepped docs/ for:
'NEGATIVE', 'negative result', 'DEAD END'/'dead end', 'no improvement', 'no measurable',
'regress', 'rejected', 'falsified', 'did not help', 'failed', 'no speedup', 'artifact',
'retraction', 'TODO', 'still open', 'unresolved', 'open thread', 'known issue',
'not yet fixed', 'never fully', 'INCONCLUSIVE', 'open question', 'blocked', 'unexplained',
'root cause unknown'. Every row ends with path:LINE. Roadmap/plan language and items the
corpus marks superseded/resolved are excluded from List B.

## List A -- NEGATIVE RESULTS (closed investigations: this lever does not work)

| # | Short name | What was tried (one line) | Verdict + key measured number | Source |
|---|---|---|---|---|
| A1 | Quantized/shared-scale int8 `moe.all_sum` wire payload | Replace bf16 all_sum payload with shared-scale int8 to cut wire bytes (multi-doc chain 2026-08-19) | NEGATIVE: live test hung the collective (:294); corrected measurement 265 vs 178 ms/call = 1.49x SLOWER (:308); structurally infeasible -- all_sum is a true elementwise partial-sum reduction, zero-pad trick only works for gathers | PERFORMANCE_HISTORY.md:308 (also :294, :283; moe-allsum-quant-live-test-failed-2026-08-19.md:9; moe-allsum-sharedscale-CORRECTED-final-2026-08-19.md:55; moe-allsum-quant-root-cause-and-closure-2026-08-19.md:79) |
| A2 | `all_sum` NOP bypass ablation | Use the built-in all_sum NOP file-toggle as a differential ablation to measure true unsync collective cost | Unsafe: destabilized the cluster, needed a fresh relaunch; "Do not use the `all_sum` NOP toggle for throughput measurement again" | docs/allsum-ablation-unsafe-2026-08-21.md:88 (verdict :1, :63) |
| A3 | `attn.all_gather` subgroup lever | Replace the ~2x-wire-bytes all_sum workaround (8.5% of prefill wall) with a real subgroup all_gather | Negative: wedge still reproduces post-transport-fix (empty response, needle FAIL); the all_sum cost is confirmed real/necessary; lever a dead end for now | docs/allgather-lever-negative-result-2026-08-21.md:35 (conclusion :73) |
| A4 | `EXO_DSV4_FUSED_SOFTMAX` | Fused Metal softmax kernel for sparse/pooled attention (OPT-12), first live A/B | Real correctness break: needle FAIL at 100K (confused/repetitive output), isolated to the flag by controlled rerun; reverted, stays OFF | docs/fused-softmax-negative-result-2026-08-21.md:37 (header :1) |
| A5 | `EXO_DSV4_INDEXER_PBLOCK` (small p_block) | Tile indexer pooled-axis scoring to bound peak allocation at high context | Prefill-neutral (no measurable prefill win at 16384/262144) but real decode regression at depth vs flat ~17-18 tok/s baseline; "do not enable in production at any small/moderate block size" | docs/indexer-pblock-decode-regression-2026-08-21.md:74 (measured :27, :31) |
| A6 | TurboQuant KV compression | Test TurboQuant variants vs dequantize-then-SDPA for KV cache attention | NEGATIVE, every config regressed: baseline 40.9 tok/s; 4-bit+qmm 37.0 (-10%), 3-bit+qmm 38.3 (-6%), 3-bit+dequant+rot 36.5 (-11%) | PERFORMANCE_HISTORY.md:1661-1668 (also 2198) |
| A7 | Quantized MoE all_sum (section umbrella) | The section 7.3 summary of the quantized-collective family as an aggregate lever | "comprehensively tested and closed NEGATIVE across 8 docs" (see A1) | PERFORMANCE_HISTORY.md:1682 |
| A8 | "~22% TP-vs-PP prefill gap" | Chase the long-standing TP ~350 vs PP ~450 prefill lead to ground with provenance archaeology + first live head-to-head | ARTIFACT: gap does not reproduce, it INVERTS -- TP 366.5 vs PP 277.0 tok/s at 100K = TP +32.3%; manufactured by 4 compounding provenance defects (different checkpoint/quant, 1.42x chars//4 artifact, depth swap, chunk-loop-rate-vs-TTFT-rate) | PERFORMANCE_HISTORY.md:1803 (live verdict :1820) |
| A9 | Token-tree speculative drafting | Implement + tune tree drafting against linear baseline (planned phases, May 2026) | NEGATIVE across a multi-week arc: correctness broken at production config first, then after fixes tree showed NO throughput lift over linear baseline (29.7-29.95 tok/s variants; ~6% of wall commit-forward overhead ate the gain) | PERFORMANCE_HISTORY.md:1435, 1445, 1447 |
| A10 | Hybrid PP-prefill/TP-decode phase swap | Swap phases between TP and PP per mode; also expert-locality placement | Rejected twice, independently; based on a 1.42x chars//4 counting artifact; PP structurally idles one node; hybrid swap and expert-locality both ruled architecturally dead | PERFORMANCE_HISTORY.md:1138-1144 |
| A11 | Expert co-location to cut TP all_sum traffic | Co-locate experts to reduce cross-rank all_sum volume | No effect: all_sum traffic volume is independent of expert placement | PERFORMANCE_HISTORY.md:3934 |
| A12 | MoE tile-geometry retune (bm > 16) / MAXBE widening | Retune MoE kernel tile geometry for headroom | Confirmed dead end at production's real shape, independently re-verified (July "falsified" verdict transfers; kernel already at/above ceiling) | docs/moe-tile-geometry-retune-dead-end-2026-08-18.md:79-89 (header :1) |
| A13 | Fused top-k / fused indexer kernel / wq_a+wkv fusion | Fuse small attention-path ops (indexer topk kernel, wq_a+wkv) to cut dispatch overhead | Noise-level or measurably slower once pipelined; wq_a+wkv correct but no measurable decode gain at c=1; fused topk alternatives all worse | PERFORMANCE_HISTORY.md:3930 (also :792, :2196; docs/qa-kv-fusion-no-measurable-gain-2026-08-21.md:68) |
| A14 | `EXO_PREFILL_STEP_SIZE=4096` on DSv4/TP | Double prefill step size at long context | ~7-8% end-to-end regression, confirmed twice; root cause is SDPA cost blowup at doubled sequence length | PERFORMANCE_HISTORY.md:3931 (docs/dsv4-4096-regression-root-cause-2026-08-19.md:15; docs/dsv4-prefill-step-size-4096-retest-2026-08-18.md:58) |
| A15 | bf16 to fp16 compute dtype on DSv4 | fp16 compute for quantized_matmul documented +7% on Qwen3.5, blanket-considered | 7x decode slowdown on DSv4 -- JACCL/RDMA transport lacks fp16 support; reverted; model/kernel-specific | PERFORMANCE_HISTORY.md:1704-1710 (verdict :1705) |
| A16 | `MLX_JACCL_RELIABLE_MAX_SZ=3` | Bump RDMA reliable max size to chunk size 3 | Statistically identical to sz=2, clean null; do not re-attempt | PERFORMANCE_HISTORY.md:3933 (docs/jaccl-sz3-tested-no-improvement-2026-08-20.md:21, :50) |
| A17 | `MLX_METAL_FAST_SYNCH=1` (+ `MLX_EVENT_WAIT_POLL_US` sweep) | Fast Metal sync primitive / poll-interval sweep on stream boundary overhead | 1.5x SLOWER with 70x more variance (dead end, don't retest); poll sweep flat, no effect | PERFORMANCE_HISTORY.md:246-249 |
| A18 | `EXO_PREFILL_CHUNK_OVERLAP` | Overlap prefill chunk compute with next-chunk work on live cluster | Throughput lever measured FLAT -> declared DEAD/CLOSED with standing do-not-re-litigate decision (correctness race was separately fixed) | PERFORMANCE_HISTORY.md:2286-2289 (quick-ref :3935) |
| A19 | DSpark FULLBLOCK verify mode at depth | FULLBLOCK rowseq verify at production depth (as corrected 2026-08-24 by P4v2) | 15.9x throughput collapse (27.56 -> 1.73 tok/s), indicts the FULLBLOCK verify MODE at k=4, not the DSpark head (outlier occurred at 94% acceptance) | docs/dspark-fullblock-context-scaling-cliff-2026-08-04.md:49 (corrected reading: PERFORMANCE_HISTORY.md:3936) |
| A20 | Nesting `mx.fast.metal_kernel` inside `mx.compile` | Compile a custom Metal kernel inside another mx.compile region | CATASTROPHIC 3-4x regression; do not nest | PERFORMANCE_HISTORY.md:1259-1262 |
| A21 | switch_mlp "27.7% of peak BW" headroom (T3) | Pipelined microbench that priced the MoE decode kernel at 27.7% of peak / 38.9% of wall | RETRACTED as measurement artifact (serial-sync mx.eval inside loop): re-measured 116-117us/call = ~404 GB/s = 74% of peak (87% independent); "NO kernel optimization lever exists here" | docs/switch-mlp-bandwidth-artifact-retraction-2026-08-22.md:1 (numbers :56) |
| A22 | FENCE_ASYNC historical "+28%" claim | Use the 2026-07-02 async-fence +28% figure as an available lever estimate | Artifact: traced to MTP-PROF, a tool whose own comments describe forced per-boundary syncing (same artifact class as sync-span profiler) | docs/fence-async-28pct-claim-traced-to-artifact-2026-08-22.md:15 (PH cross-ref :2923-2927) |
| A23 | HyperConnection training-gate 4.32x "async-fence-shaped bug" | Microbench found _hc_ops 1517us vs _hc_kernel 351us/call, looked like a live mis-gate | False lead: direct production verification showed all four gate conditions clear the fast path -- the kernel already fires in production; NOT a bug | docs/hyperconnection-training-gate-false-lead-2026-08-22.md:35 (4.32x), :91 (retraction) (PH :2556-2585) |
| A24 | `gather_qmv_rhs_lhs` kernel extension for prefill MoE M > 1 | Extend the gather kernel to prefill batch shapes for small-M headroom | Implemented, deployed, found NO-GO: real headroom exists only at decode B=1, not prefill's larger M | PERFORMANCE_HISTORY.md:2276-2277 |
| A25 | hc_expand bf16-comb cast (1.41x microbench win) | Cast tiny `comb` tensor to bf16 instead of upcasting large residual (1.41x speedup, ~1.3pp of wall) | REJECTED on quality: ~1.08% mean relative error, stable 1.08-1.14% across 7 seeds (0.125 max-abs); fp32-accum variant WORSE (1.30-1.37%); only <0.2% variant is 1.6x SLOWER (5018 vs 3104us); P5 CLOSED, no shippable variant | docs/hc-expand-rejection-relitigated-multiseed-2026-08-22.md:63-69 (PH :2626-2644) |
| A26 | lm_head mxfp8 argmax-flip fixes (alternatives + margin fallback) | Conditional margin-based bf16 fallback per row to dodge mxfp8 lm_head argmax flips; also alternative quant schemes | Rejected: every MLX-supported alternative scheme still shows the defect; the implemented+live-tested fallback FAILED both correctness and throughput bars | docs/lmhead-mxfp8-defect-and-fallback-investigations-2026-09-14.md:51 (Investigation B verdict :87-89) |
| A27 | Indexer top-k acceptance cliff at 65,536 | Hypothesis that indexer top-k threshold causes an acceptance/throughput cliff; 5-rung direct ladder test | FALSIFIED: no step; ladder flat 31-34 t/s from 45K to 122K across the predicted crossover | docs/overnight-final-indexer-hypothesis-falsified-corrected-curve-2026-09-23.md:28 (verdict :1; docs/overnight-closeout-levers-exhausted-2026-09-23.md:43) |
| A28 | "tok/s is content-driven" throughput model | Attribute tok/s variance to prompt content across depth ladders | Artifact of comparing across scripts; within one script, depth is the only variable | docs/overnight-closeout-levers-exhausted-2026-09-23.md:47 |
| A29 | 46.30 t/s "healthy 30K" anchor | Single-run anchor figure for collapse analysis | Measurement correction: single outlier; true value 34.89 t/s (n=4, stdev 0.64) | docs/overnight-closeout-levers-exhausted-2026-09-23.md:49 |
| A30 | TP=2 width-sharding as a skinny-GEMM efficiency tax | Hypothesis that width sharding hurts per-unit GEMM efficiency | Sign inverted: no penalty; sharded is 2.6-3.6% FASTER per unit work | PERFORMANCE_HISTORY.md:3944 (also :1086-1097) |
| A31 | Attention-path KV-read bytes as the 500K decode decay owner | Code-derived per-depth byte inventory of attention reads (1930 B/token slope) | Ruled out as one mechanism: predicts only +1.64 ms/token 100K to 352.6K vs live +6.80 ms/token; all_sum payload proven L-independent | docs/p3-worker-a-kv-read-inventory-2026-08-23.md (summary/verdict PERFORMANCE_HISTORY.md:3034, :3050-3051) |
| A32 | Memory residency / expert-weight paging from disk | Hypothesis that deep decode pays per-token expert paging; checked pageins/swap | Ruled out: pageins delta 1.0 MB, zero swapins over a full decode; shard fully resident 87.4 GB; ps RSS was a tool gap, not a real problem | docs/memory-residency-check-ruled-out-2026-08-22.md:1 (PH :2905-2914) |
| A33 | Comm/compute overlap as a missing lever | Reviewer-listed idea: overlap TP all_sum with compute | Closed: genuine overlap already exists in the production path; item closed by prior work | docs/comm-compute-overlap-already-exists-2026-08-21.md:105 (conclusion :98-105) |
| A34 | Dequant-to-fp16 + dense matmul MoE arm | Replace quantized expert GEMM path with dequant-to-fp16 + dense matmul | 1.8-2.1x SLOWER at block level (0.47-0.55x) -- killed; gather_mm geometrically impossible at shape | PERFORMANCE_HISTORY.md:11150 |
| A35 | July "82 TFLOP/s" MoE kernel claim | Quote the July 82 TFLOP/s figure as enormous MoE kernel headroom | RETIRED: max measured anywhere 9.58 TF; 82 TF is 5.4x the 15.14 TF silicon ceiling -- physically impossible (missing-drain/L2 artifact class) | PERFORMANCE_HISTORY.md:11151 |
| A36 | 7.26 ms client-to-server boundary lever | Price the apparent 7.26 ms client-server boundary as addressable | ARTIFACT: 101.06 - 93.8 subtracts mismatched clocks/points; ~0 ms addressable; closed by pricing | PERFORMANCE_HISTORY.md:11689 |
| A37 | Cheap footprint levers (DSpark non-FFN sharding; leaf snapshot retention 3 to 1) | Recover GB/rank at depth for the 565K memory wall from cheap levers | Measured and exhausted: 0.12 GB and 0.17 GB respectively, both ~17x too small vs the ~5 GB/rank required | docs/overnight-closeout-levers-exhausted-2026-09-23.md:54-62 (verdict :65-67) |

## List B -- OPEN PROBLEMS (genuinely unresolved as of the doc's date)

| # | Unresolved item (one line) | Current status as stated | Doc date | Source |
|---|---|---|---|---|
| B1 | Prefill ~340K cliff: live mechanism never directly reproduced | gc_limit-threshold model is best standing explanation but NOT live-confirmed; k_eff unmeasured (bounded 2-5); bimodal 8-32s stall amplitude at Studio scale UNREPRODUCED | 2026-09-14 | PERFORMANCE_HISTORY.md:2943-2953 (docs/prefill-cliff-mechanism-2026-08-24.md:48, :877) |
| B2 | +1.67..+2.52 ms/tok decode depth residual unexplained | All of it on-GPU busy work; collective ruled out on both axes; leading candidate (inter-layer pipelining bias) unproven | 2026-08-24 | PERFORMANCE_HISTORY.md:3619-3626 (docs/p01a-allsum-arrival-skew-at-depth-2026-08-29.md:103: residual "remains unattributed...") |
| B3 | Collective-wait measurement honest hole: a wait hidden inside a command buffer reads as busy | Clean closer (direct CPU-side collective timer via EXO_DSV4_ALLSUM_PROBE) exists but forces a blocking eval in place of the production async fence, perturbs what it measures, and was not run | 2026-08-24 | PERFORMANCE_HISTORY.md:3633-3640 |
| B4 | Poolgrow win attribution: concat elimination vs make_mask duty-cycle amplification | Not isolated (R2's slice-the-pad variant rejected as shipping form; never run as experiment); no step-size sweep; gates were batch=1, MTP/DSpark OFF | 2026-08-23 | PERFORMANCE_HISTORY.md:3581-3586 |
| B5 | Prefill FLOPs roofline ~1.40x headroom tail not decomposed | 28.8% non-GEMM remainder minus 9.5% all_sum slice never decomposed with decode-style rigor; sdpa/sdpa.compressed ceiling denominators never audited for masking/MLA-absorption FLOP inflation | 2026-08-22 | PERFORMANCE_HISTORY.md:2446-2457 |
| B6 | Decode roofline floor excludes KV-cache/attention read cost | Methodology gap flagged not resolved: 500K attribution figure likely overstated once KV-read cost counted | 2026-08-22 | PERFORMANCE_HISTORY.md:2309-2318 |
| B7 | True per-kernel GPU attribution structurally unavailable in current traces | xctrace/Metal System Trace yields only generic "Compute" channel names; needs mx.metal.start_capture or Xcode GPU Frame Capture, which stayed unreachable in the SSH-only environment | 2026-08-22/2026-09-14 | PERFORMANCE_HISTORY.md:2344-2352 (also :3745) |
| B8 | Which sub-cause dominates the 34x all_sum span-vs-transport gap (eval-fence vs dispatch vs Python scheduling) | Attributed jointly but "not yet further decomposed, now the correct next target"; MoE kernel-trace item kept open for that split | 2026-08-21/22 | PERFORMANCE_HISTORY.md:3015-3017 (also :2897-2899) |
| B9 | MoE expert-GEMM prefill kernel headroom real but not priced/harvested | Loop-2 verdict (iii) INCONCLUSIVE: 51-60% of calibrated ceiling, headroom exists, tested alternatives don't harvest it; prescribed per-segment M/N/K + cycles autopsy (one log line) not yet executed | 2026-10-07 | PERFORMANCE_HISTORY.md:11149 (prescribed step :11394-11395; gate rationale :11374-11376) |
| B10 | Cross-rank all_sum straggler tail asymmetry root cause (4.2x rank0-leaning in rare severe calls) | Root cause not investigated (small impact); verdict "not conclusively ignorable" at ~1.7% of decode wall (upper bound); bounded follow-up only if decode wins needed | 2026-08-22 | PERFORMANCE_HISTORY.md:2472-2493 |
| B11 | Rank1 Metal GPU timeout crash forensics: latent MoE-kernel fragility under allocator stress not separable from tracer finalize memory pressure | Best-supported mechanism is tracing-procedure memory pressure, but candidate 2 is not separable with generic channels; untraced deep (500K) decode jetsam/swap risk explicitly told to be measured directly, not inferred | 2026-08-23 | PERFORMANCE_HISTORY.md:3273-3286 |
| B12 | Do 0.5-20ms decode gaps align with per-layer moe.all_sum/fence boundaries? | Per-token gap-rate check (~20-22 gaps/token) did not cleanly confirm the expected 43/token; inconclusive rather than refuted | 2026-08-22 | PERFORMANCE_HISTORY.md:2793-2799 |
| B13 | B>1 batched verify + c=2 speculative decode production-readiness | BS>1 batched verify remains the Phase-5 TODO; c=2 spec-OFF throughput control outstanding (L2 leg cut); c=2 NOT production-ready until two shared bugs fixed | 2026-08-28 | docs/dspark-mtp-master-history-2026-08-28.md:333-335 |
| B14 | N=2 concurrent-admission transport crashes: root cause unidentified | Same SEAM suspected as bugs #1/#5 but the other traffic not yet identified; two different crash signatures (parse error vs ECONNRESET) not yet distinguished (PP-era scope) | 2026-08-05 | docs/batched-decode-n2-admission-handoff-2026-08-05.md:1091-1109 |
| B15 | PP-era jaccl C++ transport-layer bugs: long-session corruption (first-send loss on 16380-byte UC chunks) and p2p_retry_exchange stall | "NOT YET FIXED" -- needs a dedicated jaccl-focused investigation; poison-WR probe armed but never fired; p2p stall "NOT resolved by this fix, still open and genuinely separate" | 2026-08-04 | docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md:4444-4446 (NOT-YET-FIXED state), :10307,10751 (16380-byte detail), stall item :8233-8241 |
| B16 | ~11 ms/round unaccounted in DSV4.1 adaptive-gamma accounting | "Where the ~11 ms/round unaccounted goes is still open (not the sync)" | 2026-09-28 | benchmarks/phase18-dsv41-adaptive-gamma-2026-09-28/README.md:25 |
| B17 | V4.1 full-body runtime unknowns: prefetch overlap must be engineered (60-80% assumed); full-model routing concentration unverified; real end-to-end numbers incl. acceptance need the runtime | Still open per consult, unchanged | 2026-09-28 | benchmarks/phase8-fork-verification-2026-09-28/README.md:80-88 |
| B18 | Deep-context decode degradation: mmap page re-faulting mechanism CONFIRMED, fix not attempted | Mechanism confirmed (page-ins measured, swap/compressor ruled out); "Fix: NOT yet attempted. Next action is lever 1 above." | 2026-09-22 | docs/deep-context-refault-mechanism-confirmed-2026-09-22.md:97 (status :95-99) |
| B19 | Cause of the multi-TB SSD/sampler throughput dip in phase-11 stage-1 fstar | Environment correlate not found; candidates SSD-internal thermal/media or macOS memory-pressure; "Left open -- it matters only to a streamed design" | 2026-09-28 | benchmarks/phase11-stage1-fstar-2026-09-28/README.md:97 (context :90-96) |
| B20 | 2026-09-23 overnight closeout explicit OPEN/NOT-DONE list: fresh-boot control for adaptive-prefill result; attn.sdpa.compressed investigation; wired-limit 115000 to 124000 experiment (needs explicit user decision, wedge risk); hang-watchdog deeper fix (progress vs stall); chunk-size-change quality gate | Explicitly listed as OPEN / NOT DONE, not run | 2026-09-23 | docs/OVERNIGHT-CLOSEOUT-FINAL-2026-09-23.md:66-81 (items :68, :72, :73, :77, :81) |
| B21 | MoE-at-depth interplay vs allocator/memory-pressure as the depth-residual owner | "Candidate 1 verdict: INCONCLUSIVE"; overall "inconclusive rather than closed, but for different reasons than 'wrong'" | 2026-08-29 | docs/p02c-moe-allocator-depth-residual-2026-08-29.md:176 (verdict :380) |
| B22 | Token-63: does the L2-full lever's bf16 tie-break propagate to the token-63 output change? | "propagation to token 63 NOT pinned": lever mechanistically sufficient via downstream near-tie flip, but path 53 to 63 unobserved; no output-affecting change ships on this evidence | 2026-10-09 | PERFORMANCE_HISTORY.md:11658 (header :11649) |
| B23 | DSpark FULLBLOCK depth collapse treated as unresolved pre-condition for enabling DSpark | p4-scoping lists it as "YES if DSpark is enabled. 15.9x collapse, worse than no speculation at depth. Unresolved." (see contradiction C2 for the corpus-wide tangle) | 2026-08-24 | docs/p4-scoping-mtp-for-tp-2026-08-24.md:1074 |
| B24 | Perf-history section-13 checklist rows still flagged open: 340K cliff root cause (above), Instruments trace of moe.switch_mlp GatherQMM internals | switch_mlp internals bullet was answered for gather/scatter (P13) but the standing per-kernel-attribution limitation and Xcode GUI unreachability remain (see B7, B9) | 2026-08-22/2026-09-14 | PERFORMANCE_HISTORY.md:3003-3007 (resolution path :3673-3702; limiter note :3745) |

Section-13 threads deliberately left OUT of the table (superseded/resolved by later corpus
items, kept for traceability): the MTP/DSpark TP-port premise's cheap PP+DSpark re-validation
sweep "not executed this session" (PERFORMANCE_HISTORY.md:2545-2549, superseded by
docs/p4-scoping-mtp-for-tp-2026-08-24.md), and the clock-synced two-rank skew test
(INCONCLUSIVE at PERFORMANCE_HISTORY.md:2814-2860, decisively resolved 2026-09-15 at :2864-2903).
Also excluded as scope-corrected non-production threads: PP-only decode-stall chains
(PERFORMANCE_HISTORY.md:2973-2984, :2985-2992), Qwen3.5-scoped DeltaNet/dual-stream items
(:2993-3002). Similarly excluded as doc-internal supersession: T5's 9s-window probe artifact
(closed by the probe-bug find, PERFORMANCE_HISTORY.md:3081-3109) and C2's unresolved decision
rule (resolved next day by the same-build two-depth capture, :3590-3617).

## Section 13 cross-check (line 2261)

PERFORMANCE_HISTORY.md section 13 "Open / never-finished threads" (PERFORMANCE_HISTORY.md:2261)
contains: switch_mlp 27.7% (retracted at :2354-2366), T7 prefill roofline 1.40x tail (genuinely
open, :2446-2457), T4 straggler tail (bounded open, :2472-2494), T5 occupancy (answered,
:2477-2525), T6 MTP/DSpark port gate (premise re-validation, :2545-2549), T10 decomposition
(CLOSED, :2615-2674), async-fence root cause + fix (RESOLVED, :2676-2716), still-open triple
(gap-fence alignment inconclusive :2793-2799; clock-sync (resolved :2864); kernel labels
:2802-2812), prefill-cliff items (B1), the bullet ledger of scope-corrected/removed threads
(:2963-3002), switch_mlp internals bullet (:3003-3007), plus P3-era adds (B2, B3, B4, B11).

## Contradictions noticed (both sides with line numbers)

1. C1 -- hc_expand: rejected for numeric divergence on 2026-08-22 vs a hc_expand fused kernel
   SHIPPED on 2026-08-24. Closed-negative side: docs/hc-expand-rejection-relitigated-multiseed-2026-08-22.md:63-69
   ("rejection upheld... no shippable variant exists in this design space"; P5 CLOSED) and
   PERFORMANCE_HISTORY.md:2626-2644. Ship side: docs/hc-expand-kernel-ab-2026-08-24.md:3 ("VERDICT --
   SHIP (default flipped, cluster now production on kernel-on)") and :18 (+2.31%, every framing clears
   the +1.5% threshold). Reconciliation: different levers -- the rejected one was a bf16-cast numeric
   change (fp32-exact class NOT shipped), while the shipped kernel is a fused kernel that is
   fp32-exact (:709 of PERFORMANCE_HISTORY.md notes "fp32-exact class, NOT the bf16-comb variant
   rejected earlier"). Residual wrinkle: the kernel's depth A/B at 300K/500K was INCONCLUSIVE --
   docs/hc-expand-depth-verification-2026-08-24.md:3, :46-47.
2. C2 -- DSpark FULLBLOCK "15.9x collapse": three discordant verdicts coexist in the corpus.
   (a) docs/dspark-fullblock-context-scaling-cliff-2026-08-04.md:326-356: same-day MAJOR CORRECTION --
   the sustained, context-scaling 15.9x collapse "simply is not present in this re-test" (line 354);
   only a narrow ~2800-token band anomaly reproduced. (b) PERFORMANCE_HISTORY.md:3936 quick-ref keeps
   a corrected 2026-08-24 (P4v2) verdict indicting the FULLBLOCK verify MODE at k=4 ("the outlier
   occurred at 94% acceptance"). (c) docs/p4-scoping-mtp-for-tp-2026-08-24.md:1074 still lists the
   collapse as "Unresolved" for DSpark enablement. Any writeup must pick one frame explicitly.
3. C3 -- MTP c>=2 degenerate-output root cause: listed "Root cause unknown -- needs separate
   investigation" in docs/b2-quality-handoff-2026-06-24.md:185, but marked RESOLVED the same day in
   docs/b2-mtp-resolution-2026-06-24.md:4 ("Two root-cause bugs fixed, both verified"). Old-doc open
   language must not be quoted as still-open.
4. C4 -- Speculative-decoding throughput ceiling, doc-internal supersession:
   PERFORMANCE_HISTORY.md:1318 frames "the throughput ceiling here has proven very hard to move past
   ~30-35 tok/s", while the UPDATE immediately below (:1321, 2026-08-27) declares that framing
   superseded by depth-gated batched verify at 36.63 tok/s (+36.71%), production default. The
   negative framing survives only as historical context.
5. C5 -- minor, same class: the "~22% TP-vs-PP prefill gap" is carried as a live lead in older prose
   but closed as an ARTIFACT in section 9.0 (PERFORMANCE_HISTORY.md:1803) with the sign inverted
   (:1820); quoting any pre-2026-08-24 PP ~450 figure would re-import the artifact.

## Counts

- List A (negative-result rows): 37
- List B (open-problem rows): 24 (numbered B1-B24; table re-counted programmatically after generation)
---

## D9. CONFLICT REGISTER (numbers/claims that disagree across the corpus)

| # | conflict | side A | side B | resolution |
|---|---|---|---|---|
| 1 | wired memory limit | 120000 MB live (`sysctl iogpu.wired_limit_mb` on both nodes) | 115000 MB (120.6 GB) in every doc, e.g. docs/PERFORMANCE_HISTORY.md:7436 | use the live value and say so; % -of-limit figures in docs are against 120.6 GB |
| 2 | hostname vs LocalHostName | `hostname` = node0/-2 (live) | `scutil --get LocalHostName` = M4-4 / M4-5 (live); docs/launcher-address-drift-repair-2026-09-28.md:22 | known drift, aliases repointed to Tailscale FQDNs 2026-10-03 |
| 3 | hc_expand | rejected and rejection UPHELD (multi-seed) 2026-08-22, docs/hc-expand-rejection-relitigated-multiseed-2026-08-22.md:63-69 | a fused Metal hc_expand kernel SHIPPED 2026-08-24 (+2.31%, default flipped), docs/hc-expand-kernel-ab-2026-08-24.md:3,18 | two different levers (bf16-comb cast vs fp32-exact fused kernel); the kernel's own 300K/500K depth A/B was INCONCLUSIVE (docs/hc-expand-depth-verification-2026-08-24.md:3,46-47) |
| 4 | "~22% TP-vs-PP prefill gap" | was treated as a real deficit | closed as an ARTIFACT with sign inverted: TP 366.5 vs PP 277.0 = TP +32.3%, docs/PERFORMANCE_HISTORY.md:1803,1820 | do not re-import any pre-2026-08-24 "PP ~450" figure |
| 5 | MoE gate+up fusion sign | -3.8% at c=2 2026-06-26 (docs/dsv4-decode-stall-2026-06-26.md:175-176) | +3.01% at c=1 short-ctx 2026-08-21 (docs/moe-gate-up-fusion-validated-2026-08-21.md:44-45) | unresolved; PERFORMANCE_HISTORY.md:1234-1237 records it as a contradiction |
| 6 | KV bytes at 1M | computed 4.13 GiB counting all 8 index sources (this pack, D4.2) | ~3.2 GiB stated in mlx_lm/models/deepseek_v41/cache.py:29 (= counting only the 4 kv-source layers) | convention difference, not a wrong number; state the counting rule |
| 7 | 565K decode | 13.36 tok/s (docs/overnight-results-two-mechanisms-and-stochastic-collapse-2026-09-23.md:38-39) | 24.15 tok/s same night (docs/overnight-final-indexer-hypothesis-falsified-corrected-curve-2026-09-23.md:75) | bimodal / stochastic collapse; report a band |
| 8 | 500K depth label | every T1-era row labelled 500K | real prompt 352,599 tok (docs/p3-worker-b1-live-depth-anchors-2026-08-23.md:26) | label the real depth |
| 9 | DSpark FULLBLOCK 15.9x collapse | source doc major-corrects it as not reproduced (docs/dspark-fullblock-context-scaling-cliff-2026-08-04.md:326-356) | still listed as "Unresolved" for DSpark enablement in docs/p4-scoping-mtp-for-tp-2026-08-24.md:1074 | report both, with dates |
| 10 | checkpoint identity | DeepSeek-V4-Flash-0731 / V4.1-EXL3 lineage (rows 1-20) | DeepSeek-V4-Flash-Vision-Exp (rows 21-23) | different model; do not blend |
| 11 | on-disk checkpoint size | 210.68 GB measured (`du -sk`) | 210.58 GB in docs/deepseek-v41-exl3-plan.md:116; live /state `storageSize=210,713,013,432 B` (210.71 GB logical, evidence-live.md:22) | rounding; prefer the measured value |
| 12 | import-time mlx version | `mlx 0.32.5.dev20261016+b17781838` live | `0.32.3.dev20261002+b1778183` in uv.lock:1383 | same git SHA (b1778183*); version string is set at build time |
| 13 | MTP c>=2 degenerate output | listed open, "root cause unknown" docs/b2-quality-handoff-2026-06-24.md:185 | RESOLVED same day, two root-cause bugs fixed, docs/b2-mtp-resolution-2026-06-24.md:4 | closed - do not quote the open language |
| 14 | speculative-decode "ceiling ~30-35 tok/s" | docs/PERFORMANCE_HISTORY.md:1318 | superseded immediately at :1321 by depth-gated batched verify 36.63 tok/s | treat :1318 as history |

---

## D10. METHOD / LIMITS OF THIS PACK

- The 210-row corpus table (D1.2) was machine-extracted by a single pass over docs/. It is a
  candidate set with citations, not a hand-audited table: 196/210 citations auto-verified, 14
  auto-repaired, 2 approximate. The curated 27-row table (D1.1) was re-read line by line by the
  author of this pack and should be treated as the higher-confidence subset.
- Deliberately excluded from D1.1 as non-measurements: roofline/theoretical ceilings (896 tok/s
  roofline, 26.9/39.3-40.0 I-O ceilings), pre-registration PROMOTE bars, and planning ranges
  (+/-40% in docs/deepseek-v41-exl3-plan.md).
- Nothing in the repos was modified. Only read-only commands were issued; the cluster processes were
  not touched.
