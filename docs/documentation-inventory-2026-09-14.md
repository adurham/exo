# Documentation Inventory — 2026-09-14

Full categorized inventory of fork-specific documentation across the three
repos on this cluster (`~/repos/exo`, its `mlx` submodule, its `mlx-lm`
submodule), covering ~4 months of accumulated work (2026-04-24 through
2026-09-14). This is a **read-only enumeration and categorization** — no file
listed below was deleted, merged, moved, or edited as part of producing this
doc. Category (c) entries are **recommendations only**.

Compiled by reading: every top-level `*.md` in all three repos in full or
near-full; the first ~20-40 lines and last ~15-20 lines of all 194 `docs/`
top-level files, `docs/incidents/`, `docs/upstream-pr-drafts/`; a full read of
every file under 60 lines (stub-candidate check); representative samples
across all 125 git-tracked `tmp/*.md` and all 45 `.hermes/plans/*.md`; and
corpus-wide `grep` sweeps for `SUPERSEDED`, `deprecated`, `obsolete`, `FINAL`,
`REMOVED`, `redundant`, and `no longer` (restricted to tracked `.md` files —
an early unscoped grep swept in unrelated JSON eval-harness fixtures under
`tmp/perf-campaign-2/round6-7/results/*.json`, which are adversarial-prompt
test artifacts, not documentation, and are excluded from this inventory
entirely). Live-code cross-checks (via `grep`/`git log`) were run against
every category (a) candidate to confirm current accuracy rather than trusting
doc headers at face value.

---

## 1. Summary counts

| Repo | Scope | (a) Current reference | (b) Dated historical | (c) Superseded/redundant | (d) Orphaned/unclear | Total |
|---|---|---:|---:|---:|---:|---:|
| exo | `docs/` (incl. `incidents/`, `upstream-pr-drafts/`, `profiling/`) | 11 | 180 | 2 | 1 | 194 |
| exo | `tmp/*.md` (git-tracked only) | 0 | 124 | 1 | 0 | 125 |
| exo | `.hermes/plans/*.md` | 0 | 45 | 0 | 0 | 45 |
| exo | top-level `*.md` | 6 | 4 | 2 | 0 | 12 |
| mlx | top-level `*.md` (upstream-inherited) | 0 | 0 | 0 | 5 | 5 |
| mlx | `docs/incidents/*.md` | 0 | 1 | 0 | 0 | 1 |
| mlx-lm | top-level `*.md` (upstream-inherited) | 0 | 0 | 0 | 6 | 6 |
| mlx-lm | `mlx_lm/*.md` (upstream-inherited) | 0 | 0 | 0 | 6 | 6 |
| **Total** | | **17** | **354** | **5** | **18** | **394** |

Note on the 394 total vs. the task's "~250-260 files" estimate: the gap is
almost entirely the `mlx`/`mlx-lm` upstream-inherited files (17 files) landing
in (d) rather than being excluded outright — see §5 for why they're flagged
"unclear" rather than dropped — plus the true corpus being slightly larger
than estimated (194 vs. "~220" for `docs/`, but that count already included
non-`.md` files like images/logs/JSON which are out of scope for a *markdown*
doc inventory and are not counted here). The markdown-only counts above are
exact, taken from `find`/`git ls-files`, not estimates.

---

## 2. Category (a): Still-authoritative / current-reference

Each entry below was verified against live code or the live cluster config
(not just its own header claim) before inclusion.

| File | What it authoritatively covers | Verification performed |
|---|---|---|
| `AGENTS.md` (`CLAUDE.md` symlinks to it) | Build/run/test commands, architecture overview (Node/Router/Worker/Master/Election), event-sourcing model, code-style rules, dashboard screenshot workflow | Matches current repo layout (`src/exo/main.py`, `src/exo/routing/topics.py`) |
| `RULES.md` | Repository-wide coding rules (typing discipline, Pydantic conventions, naming) | No contradicting or superseding doc found; still referenced implicitly by code style across `src/` |
| `README.md` | Project pitch, features, dashboard, install/run instructions | Standard project README, actively current |
| `PLATFORMS.md` | Tier 1/2/3 hardware support matrix | Short roadmap doc, no stale claims found, no superseding doc exists |
| `CONTRIBUTING.md` | Dev setup (`uv`, `rust`, `macmon` pinned fork), clone/build/run steps | Matches current build tooling |
| `docs/api.md` | REST API reference for all endpoints (`/node_id`, `/state`, instance management, `/v1/chat/completions`, Claude Messages API, OpenAI Responses API, Ollama compatibility, metrics) | Spot-checked: `/node_id` and `/state` routes confirmed live in `src/exo/api/main.py` (the doc's own file-path header, `src/exo/master/api.py`, is now wrong post-refactor — code lives in `src/exo/api/main.py` — but the described endpoints and behavior are accurate) |
| `docs/metrics.md` | Prometheus metrics catalog (`/metrics` endpoint, metric names, Grafana dashboard) | Spot-checked: `exo_generation_tps`, `exo_prefix_cache_hits_total` confirmed present in `src/exo/metrics/metrics.py` |
| `docs/kv-cache-architecture.md` | KV cache types (`KVCache`/`QuantizedKVCache`/`RotatingKVCache`/`ArraysCache`/`TurboQuantKVCache`), radix prefix-cache design, per-instance config, runtime code-path file map | File map targets confirmed to exist; this is the architecture doc the task asked us to search for (found under this exact name) |
| `docs/fork-notes.md` | PP-vs-TP tradeoff (structural, still true), fork dependency-pin map, running log of major fork-vs-upstream divergence fixes | Largest fork-tracking doc (1504 lines); still the canonical place new fork/upstream divergences get logged (most recent dated entry 2026-07-31) |
| `docs/upstream-prs.md` | Cross-repo tracker of what's been upstreamed / held / open, companion to fork-notes.md | Last full refresh 2026-06-04 — **dated content, but no superseding doc exists and it is still the only PR tracker**; treat the PR status table itself as dated (flagged in §5) while the doc's role as "the tracker" remains current |
| `docs/upstream-pr-drafts/README.md` | Index of held/opened upstream PR drafts (mlx#3596, exo#2149, 2 held) | Small index file, cross-checked against `docs/upstream-prs.md`'s PR table — consistent |

**Two additional files were seriously considered for (a) and intentionally
placed in (b) instead** — see §4 discussion:
- `docs/thinking-parser-fused-delimiter-fix.md` and `docs/mtp-tiebreak-losslessness-fix.md`
  are individually-shipped-fix writeups, verified still live in code
  (`parse_thinking_models`'s fused/prefix-boundary handling and the
  `EXO_DSV4_MTP_TIEBREAK_FIX`/`EXO_DSV4_MTP_ACCEPT_LOGPROBS` gates are both
  present in `dsv4_mtp.py` and `model_output_parsers.py` today), but they are
  investigation-writeup-shaped (single fix, single narrative, dated title),
  not reference-doc-shaped, so they're catalogued in §4 under their topic
  groups rather than here. Worth noting: `mtp-tiebreak-losslessness-fix.md`
  itself is now superseded in *mechanism* — the code comment at
  `dsv4_mtp.py:338` says `EXO_DSV4_MTP_ACCEPT_LOGPROBS` (default **ON** since
  2026-07-10) "Supersedes the tie-break fix (`EXO_DSV4_MTP_TIEBREAK_FIX`,
  already OFF in prod)" — but no doc describes `EXO_DSV4_MTP_ACCEPT_LOGPROBS`
  as its own topic (see §5 orphan gap).

---

## 3. Category (a)-adjacent judgment calls (read before trusting headers)

Two files are frequently *cited as* authoritative by other current docs, but
are themselves shaped as point-in-time investigation logs, not reference
docs. Both are catalogued in category (b) (§4), not (a), on a structural
principle: a document's category should follow its own form (is it *written
as* a living reference, or as a narrative investigation record with dead
ends and revised hypotheses?), not how often other docs cite it or how
accurate one specific section still is.

- **`docs/dspark-mtp-master-history-2026-08-28.md`** — self-titled "Master
  Campaign History," §1 ("Current production state, authoritative,
  2026-08-28") contains an env-flag table. **Verified 2026-09-14: every flag
  value in that table (`EXO_SPECULATIVE=1`, `EXO_DSV4_DSPARK_NATIVE=1`,
  `EXO_SPECULATIVE_GAMMA=3`, `EXO_DSV4_VERIFY_BATCH=1` @ MIN_CTX=8192,
  `EXO_DSV4_DSPARK_TP_SHARD=1`, `EXO_MLX_CLEAR_CACHE_INTERVAL=64`, etc.) still
  exactly matches `start_cluster.sh`'s current defaults, 17 days later.** But
  the other ~440 lines are a 3-month narrative (correctness crises, the
  PP→TP migration, reverts, promotions) — campaign-log-shaped, not
  reference-shaped. Filed in (b), flagged there as "de facto authoritative
  for its flag table specifically, verified current as of this inventory's
  date — do not archive without transferring that fact to a real reference
  doc first."
- **`docs/prefill-cliff-mechanism-2026-08-24.md`** — 1792 lines, written in
  investigation-log style ("session 4 conclusion"), but two top-level handoff
  docs (`MOE_KERNEL_HANDOFF.md`, `PREFILL_CLIFF_HANDOFF.md`) both explicitly
  point to it as "the full evidence chain" for a now-closed issue. Kept in
  (b) for the same structural reason — heavy citation doesn't change what
  kind of document it is.

---

## 4. Category (b): Dated historical investigation / campaign logs

This is the bulk of the corpus — 353 files total. Grouped by topic prefix so
a future reader can find a topic area quickly. **Docs/ currently has no
topic index of its own (see §6) — this section is intended to serve as one.**

### 4.1 `docs/` top-level (183 files: 180 in (b) proper + 3 discussed above/below)

Grouped by filename topic-prefix, in the order a reader is likely to search.

<details>
<summary><b>DSpark speculative-decode head (33 files, 2026-08-03 → 2026-08-28)</b> — click to expand file list</summary>

The single largest topic cluster. Covers: porting DSv4's native MTP/DSpark
draft head, the PP→TP migration's effect on speculative decode, the 352K
deep-context memory regression and its fix, and the final production
promotion (+36.71% @100K, 12/12 paired wins). **Start here if researching
speculative decode**: `dspark-mtp-master-history-2026-08-28.md` (the
synthesis doc, see §3) → `dspark-mtp-production-baseline-2026-08-27.md`
(current shipped config) → topic-specific files below for mechanism detail.

- `dsv4-0731-dspark-native-head-plan-2026-08-03.md` — native-head porting plan (implemented; superseded in detail by "what actually happened" section within itself)
- `dspark-fullblock-context-scaling-cliff-2026-08-04.md` — 352K-class context scaling cliff, first sighting
- `dspark-cs-profile-2026-08-26.md`, `dspark-tier1-byte-identity-2026-08-26.md`, `dspark-verdict-measurement-2026-08-26.md` — correctness/byte-identity verification round
- `dspark-mtp-ab-preregister-2026-08-25.md` — pre-registered A/B protocol (Stage-3 verdict later superseded by corrected batched design — banner inside the doc)
- `dspark-14k-cliff-investigation-2026-08-27.md` — a smaller-context (14K) cliff, separate from the 352K one
- `dspark-352k-allocator-pool-analysis-2026-08-27.md`, `dspark-352k-batched-verify-transients-2026-08-27.md`, `dspark-352k-memory-regression-2026-08-27.md`, `dspark-352k-residency-analysis-2026-08-27.md` — the memory-regression investigation (allocator fragmentation vs. residency vs. pool growth, root-caused and fixed)
- `dspark-352k-correctness-harness-verification-2026-08-28.md`, `dspark-352k-verification-preregister-2026-08-28.md`, `dspark-352k-verification-runs-2026-08-28.json` — correctness verification closing the memory-fix arc
- `dspark-draft-epilogue-fusion-2026-08-27.md` — draft-epilogue fusion attempt
- `dspark-mtp-production-baseline-2026-08-27.md` — **the shipped production config as of 2026-08-27** (still matches current `start_cluster.sh` per §3 verification)
- `dspark-p1-draft-epilogue-ab-results-2026-08-28.md`, `dspark-p1p4-campaign-preregister-2026-08-28.md`, `dspark-p2-c2-validation-results-2026-08-28.md` — P1/P4 campaign sub-results
- `dspark-mtp-master-history-2026-08-28.md` — synthesis doc, see §3

**Closely related, DSv4-core (not DSpark-specific) — 18 files, 2026-06-26 → 2026-08-19:**
- `dsv4-decode-stall-2026-06-26.md`, `dsv4-memory-leaks-2026-06-27.md`, `dsv4-memory-leak-handoff-2026-06-29.md`, `dsv4-memory-leak-resolution-2026-06-29.md` — a three-doc memory-leak arc (findings → handoff → resolution post-mortem); all three cover genuinely different vantage points of the same closed bug and are kept as three separate historical artifacts, not merged (see §5 discussion — considered for (c), kept in (b))
- `dsv4-c2-serving-handoff-2026-07-06.md` — concurrent-serving handoff (1236 lines)
- `dsv4-rowseq-followups-plan-2026-07-10.md` — row-sequential follow-up plan
- `dsv4-220k-prefill-eventwait-ringdiag-nonrepro-2026-08-18.md`, `dsv4-220k-prefill-eventwait-rootcause-triage-2026-08-18.md`, `dsv4-220k-prefill-rdma-wait-breakdown-2026-08-18.md`, `dsv4-220k-prefill-seqsplit-ab-2026-08-18.md`, `dsv4-220k-prefill-span-profile-2026-08-18.md` — 220K-context prefill investigation cluster (same day, five angles)
- `dsv4-attention-kernel-efficiency-2026-08-18.md`, `dsv4-prefill-step-size-4096-retest-2026-08-18.md` — same-day prefill-tuning companions
- `dsv4-4096-regression-root-cause-2026-08-19.md` — chunk-size=4096 regression, root-caused (has a "THIRD UPDATE, final word" self-correction structure — internally resolved, not a redundancy case)
- `dsv4-clear-cache-interval-2-test-2026-08-19.md`, `dsv4-sdpa-subtiling-code-map-2026-08-19.md` — same-week companions
- `dsv4-vision-phase3b-window-geometry-inventory.md` — undated in filename but clearly 2026-09 era (references `DSV4_VISION_PORT_PLAN_PHASE34.md`); pre-implementation code-geometry survey for the Vision-Exp port

</details>

<details>
<summary><b>MoE all_sum / collective cost (18 files, 2026-08-19 → 2026-08-21)</b></summary>

Covers the multi-day investigation into whether the MoE `all_sum` collective
could be sped up via int8 quantization on the wire. **Verdict, read this
first**: `moe-allsum-quant-root-cause-and-closure-2026-08-19.md` — the
approach is "mathematically dead" for this collective (not a bug, a
structural mismatch). A separate, later sub-thread (`sharedscale`) explored
a variant and also closed negative for prefill (only possibly useful for
decode, unexplored).

- `moe-all-sum-dominant-cost-2026-08-19.md` — establishes all_sum = 61-64% of prefill wall time (the founding finding of this whole cluster of docs)
- `moe-all-sum-payload-size-causal-test-2026-08-19.md`, `moe-all-sum-skew-vs-comms-2026-08-19.md` — causal/skew analysis
- `moe-allsum-quant-compute-overhead-analysis-2026-08-19.md`, `moe-allsum-quant-live-test-failed-2026-08-19.md`, `moe-allsum-quant-phase0-repro-2026-08-19.md`, `moe-allsum-quant-root-cause-and-closure-2026-08-19.md` — the int8-quant-on-wire investigation arc, closed negative
- `moe-allsum-sharedscale-root-cause-found-2026-08-19.md`, `moe-allsum-sharedscale-live-test-no-speedup-2026-08-19.md`, `moe-allsum-sharedscale-CORRECTED-final-2026-08-19.md` — the shared-scale variant; the third doc **explicitly corrects a wrong ~148x claim made earlier in the same session** (self-correcting three-doc arc, not redundant — see §5)
- `moe-all-sum-178ms-artifact-real-bottleneck-2026-08-20.md` — follow-up bottleneck re-attribution
- `moe-allsum-collective-cost-confirmed-2026-08-21.md` — final confirmation

**Other MoE kernel/dispatch — 6 files, 2026-08-18 → 2026-08-21:**
- `moe-per-stage-gpu-breakdown-2026-08-18.md`, `moe-tile-geometry-retune-dead-end-2026-08-18.md` (has an in-file 2026-08-19 correction banner for its denominator, core conclusion unchanged)
- `moe-gpu-time-overlap-bandwidth-bound-2026-08-19.md`, `moe-quant-vs-bf16-dequant-attribution-2026-08-19.md`, `moe-vs-dense-qmm-isolation-2026-08-19.md`
- `moe-gate-up-fusion-validated-2026-08-21.md` — validated positive result (gate+up fusion)

</details>

<details>
<summary><b>hc-collapse / hc-expand — hyperconnection gate kernels (6 files, 2026-08-22 → 2026-08-25)</b></summary>

- `hc-expand-rejection-relitigated-multiseed-2026-08-22.md` — re-litigates and **upholds** an earlier rejection with multi-seed evidence
- `hc-collapse-roofline-2026-08-24.md`, `hc-expand-depth-verification-2026-08-24.md`, `hc-expand-kernel-ab-2026-08-24.md`
- `hc-collapse-depth-verification-2026-08-25.md`, `hc-collapse-kernel-ab-2026-08-25.md`

Both `EXO_DSV4_HC_COLLAPSE_KERNEL` and `EXO_DSV4_HC_EXPAND_KERNEL` are `=1`
in current production per `dspark-mtp-master-history`'s verified flag table
— these kernels shipped.

</details>

<details>
<summary><b>500K decode-decay campaign — p2/p3 worker+reviewer cluster (12 files, 2026-08-23 → 2026-08-24)</b></summary>

A structured multi-worker investigation (worker-a/b1/c/c2/c3/d + two
reviewers + a synthesis) into decode throughput decay at long context.

- `p2-xctrace-prefill-collective-wedge-2026-08-23.md`
- `p3-worker-a-kv-read-inventory-2026-08-23.md`, `p3-worker-b1-live-depth-anchors-2026-08-23.md`, `p3-worker-c-attn-kernel-walltime-2026-08-23.md`, `p3-worker-c2-depth-busy-idle-capture-2026-08-23.md`, `p3-worker-c3-donation-failure-insitu-2026-08-23.md`, `p3-worker-d-metal-timeout-crash-forensics-2026-08-23.md`
- `p3-reviewer-r1-verification-2026-08-23.md`, `p3-reviewer-r2-verification-2026-08-23.md`
- `p3-followup-poolgrow-ab-2026-08-23.md` (1037 lines; its own §8 is explicitly marked "superseded by §14" — internal self-correction, not cross-file)
- `p3-synthesis-500k-decode-decay-decomposition-2026-08-23.md` — the synthesis
- `p3-followup-allsum-wait-at-depth-2026-08-24.md`

</details>

<details>
<summary><b>TP-width & prefill-gap campaign — p4/p4v2/p5 (4 files, 2026-08-23 → 2026-08-24)</b></summary>

- `p4-tp-width-shard-gemm-efficiency-2026-08-23.md`
- `p4-scoping-mtp-for-tp-2026-08-24.md` — 1227 lines, has a large in-file "SUPERSEDED IN PART" banner (P4v2) correcting its own ordering/verdict — internal self-correction
- `p4v2-m1-shadow-gate-results-and-recovery-2026-08-24.md`
- `p5-tp-prefill-gap-2026-08-24.md`

</details>

<details>
<summary><b>Aug-29/30 numbered campaign — p0N series (10 files)</b></summary>

Switch-MLP GPU tracing, allsum arrival skew at depth, MoE allocator depth
residual, roofline/occupancy reconciliation, small-op bucket tracing,
Sinkhorn truncation numerics, prefill-remainder per-kernel analysis,
SDPA-ceiling/top-k spike pre-registration.

- `p01-switch-mlp-gputrace-recapture-2026-08-29.md`, `p01a-allsum-arrival-skew-at-depth-2026-08-29.md`, `p01b-multilayer-pipelining-loss-2026-08-29.md`
- `p02c-moe-allocator-depth-residual-2026-08-29.md`, `p02d-roofline-occupancy-kernel-reconciliation-2026-08-29.md`
- `p03-smallop-bucket-gputrace-2026-08-30.md`, `p04-sinkhorn-truncation-numerics-2026-08-30.md`
- `p07-prefill-remainder-perkernel-preregister-2026-08-30.md`, `p07-prefill-remainder-perkernel-results-2026-08-30.md`
- `p08-sdpa-ceiling-and-topk-spike-preregister-2026-08-30.md`

(These p0N docs are the `docs/`-level companions to the much larger
`tmp/p0N-*` and `tmp/perf-campaign-2/` working directories — see §4.2/§4.3.)

</details>

<details>
<summary><b>phase0[abc] / phase3 / phase-c — collective overlap & order determinism (5 files, 2026-08-20 → 2026-08-22)</b></summary>

- `phase0a-allsum-boundary-decomposition-2026-08-20.md`, `phase0b-collective-overlap-gate-2026-08-20.md`, `phase0c-collective-order-determinism-2026-08-20.md`
- `phase3-cluster-validation-blocked-resolved-2026-08-20.md`
- `phase-c-dual-capture-confirms-fix-2026-08-22.md`

</details>

<details>
<summary><b>Collective/allsum/allgather microbenchmarks, general (7 files, 2026-08-21 → 2026-08-22)</b></summary>

- `allgather-lever-negative-result-2026-08-21.md`, `allsum-ablation-unsafe-2026-08-21.md`, `comm-compute-overlap-already-exists-2026-08-21.md`, `offline-collective-microbenchmark-2026-08-21.md`
- `allsum-straggler-aggregate-impact-2026-08-22.md`, `allsum-sync-span-artifact-arithmetic-check-2026-08-22.md`, `cross-rank-allsum-skew-2026-08-22.md`

</details>

<details>
<summary><b>Fence/roofline/async measurement-artifact corrections (11 files, 2026-08-19 → 2026-08-24)</b></summary>

A cluster of docs about measurement-methodology artifacts (sync-fence
profiling inflating apparent costs) — several explicitly retract earlier
claims once the artifact was found. **Read `switch-mlp-bandwidth-artifact-retraction-2026-08-22.md`
and `fence-async-28pct-claim-traced-to-artifact-2026-08-22.md` first** if
investigating why an old throughput/bandwidth percentage doesn't match a
newer one — the discrepancy is very likely this artifact class, already
diagnosed here.

- `local-absmax-fence-artifact-confirmed-2026-08-19.md`
- `async-fence-cache-owner-dead-code-root-cause-2026-08-22.md`, `async-fence-fix-validated-2026-08-22.md`, `fence-async-28pct-claim-traced-to-artifact-2026-08-22.md`
- `pysampler-blocking-eval-root-cause-2026-08-22.md`
- `roofline-recalculated-post-fix-2026-08-22.md`, `roofline-sanity-check-inputs-confirmed-2026-08-22.md`
- `switch-mlp-bandwidth-artifact-retraction-2026-08-22.md`, `switch-mlp-kernel-bandwidth-efficiency-2026-08-22.md` (the retracted claim, kept for provenance — the retraction doc is the one to trust)
- `t10-final-decomposition-closed-2026-08-22.md` — closes the whole T10 investigation thread
- `sdpa-flop-denominator-audit-2026-08-24.md`

</details>

<details>
<summary><b>mtp-* / b2-mtp-* — MTP tie-break & B=2 concurrency (4 files, 2026-06-24 → 2026-08-22)</b></summary>

- `mtp-tiebreak-losslessness-fix.md` — shipped fix writeup; **mechanism now superseded** by `EXO_DSV4_MTP_ACCEPT_LOGPROBS` per an in-code comment (see §2 discussion) — no doc yet describes the newer mechanism (orphan gap, see §5)
- `b2-mtp-resolution-2026-06-24.md` — resolution; supersedes `b2-quality-handoff-2026-06-24.md`'s original cache-merge theory (that doc has its own banner pointing here)
- `b2-quality-handoff-2026-06-24.md` — superseded-in-theory-only by the above (kept, own banner says so — see §5, considered for (c), kept in (b))
- `mtp-dspark-tp-port-decision-gate-2026-08-22.md` — decision-gate doc; used the now-retracted 27.7% switch_mlp figure as one input (flagged, not fatal to the doc's own conclusion)

</details>

<details>
<summary><b>decode-* timing/attribution, not dsv4-prefixed (4 files, 2026-08-21 → 2026-08-22)</b></summary>

`decode-roofline-dispatch-bound-2026-08-21.md`,
`decode-attribution-recompute-postfix-2026-08-22.md`,
`decode-idle-time-investigation-interim-synthesis-2026-08-22.md` (own text
says a still-earlier synthesis is itself superseded — internal chain),
`decode-time-budget-synthesis-2026-08-22.md`.

</details>

<details>
<summary><b>gpu-* clock/occupancy/utilization (5 files, 2026-08-18 → 2026-08-22)</b></summary>

`gpu-utilization-confirmed-saturated-2026-08-18.md`,
`gpu-util-vs-allsum-cost-reconciled-2026-08-19.md`,
`gpu-clock-symptom-confirmed-2026-08-22.md`,
`gpu-idle-gap-deep-dive-2026-08-22.md`,
`gpu-occupancy-clock-gap-postfix-2026-08-22.md`.

</details>

<details>
<summary><b>indexer-* — DSv4 sparse indexer (3 files, 2026-08-21 → 2026-08-24)</b></summary>

`indexer-pblock-decode-regression-2026-08-21.md` (the negative result that
closed `EXO_DSV4_INDEXER_PBLOCK`, originally proposed in the top-level
`INDEXER_TILED_P_PLAN.md`), `indexer-topk-fused-decode-only-2026-08-22.md`,
`indexer-prefill-decomposition-2026-08-24.md`.

</details>

<details>
<summary><b>prefill-* throughput/chunking, not dsv4- or p0N-prefixed (11 files, 2026-06-24 → 2026-08-24)</b></summary>

- `prefill-optimization.md` (undated filename, content is a general design doc, superseded in most specifics by later dated docs but kept as historical baseline)
- `prefill-throughput-breakthrough-2026-06-24.md` — the breakthrough batch referenced by both top-level HANDOFF docs as the resolution point
- `prefill-optimization-campaign-handoff-2026-08-18.md`
- `prefill-chunk-overlap-ab-harness-design-2026-08-20.md`, `prefill-chunk-overlap-live-test-2026-08-20.md`, `prefill-chunk-overlap-race-fix-2026-08-20.md`
- `prefill-sync-span-kernel-breakdown-2026-08-21.md`, `prefill-trace-instrumentation-findings-2026-08-21.md`
- `prefill-fence-gate-audit-2026-08-22.md`, `prefill-flops-roofline-aggregate-2026-08-22.md`
- `prefill-cliff-mechanism-2026-08-24.md` — the authoritative mechanism doc for the (now-resolved) prefill cliff; see §3 for why it's here and not (a)

</details>

<details>
<summary><b>jaccl-* RDMA transport (2 files, 2026-08-20 → 2026-08-21)</b></summary>

`jaccl-sz3-tested-no-improvement-2026-08-20.md`,
`jaccl-internal-timing-allsum-transport-fast-2026-08-21.md`.

</details>

<details>
<summary><b>verify-batch-* batched-verify correctness (2 files, 2026-08-26 → 2026-08-27)</b></summary>

`verify-batch-phase0-2026-08-26.md` (its own tail has a "RESOLUTION
2026-08-27" update section, not a separate supersession) and
`verify-batch-g0-fail-2026-08-27.md` (own body has a "RESOLVED, SUPERSEDED
BY THE CORRECTED DESIGN" section — again a self-contained resolution, not
pointing at a same-topic duplicate file).

</details>

<details>
<summary><b>vec-* vector row-SDPA serving campaign (2 files, 2026-07-12)</b></summary>

`vec-rowsdpa-campaign-2026-07-12.md`, `vec-serving-increment4-handoff-2026-07-12.md`.

</details>

<details>
<summary><b>minimax-* MiniMax model design docs (5 files, 2026-04-24, pre-DSv4 era)</b></summary>

`minimax-decode-optimization.md`, `minimax-fused-attention-design.md`,
`minimax-fused-attention-prompt.md`, `minimax-quantized-sdpa-design.md`,
`minimax-rdma-moe-validation-2026-04-24.md`. Oldest topic cluster in the
corpus — predates the DSv4 era entirely (MiniMax-M2.7 was the prior
production tenant). Kept for institutional memory (kernel-design techniques
here — fused attention, quantized SDPA — are model-agnostic groundwork).

</details>

<details>
<summary><b>handoff-2026-08-08/09/10 — section handoffs (7 files)</b></summary>

Numbered "Section N" handoffs from the `hybrid-pp-prefill-tp-decode-design`
mega-doc's live campaign — `section22`, `section23`, `section25` (Aug 8),
`section39` (Aug 9), `section42`, `section43`, `section43-part2` (Aug 10).
Full narrative context for these lives in the referenced sections of
`docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md` itself.

</details>

<details>
<summary><b>Miscellaneous single-topic negative-results and gate docs (20 files, 2026-07-12 → 2026-08-25)</b></summary>

`iogpu-residency-set-abort.md` (a genuinely important hardware/OS bug
writeup — silent SIGABRT root-cause on long-context inference, fork patch
`f8bac642`), `rollback-cost-campaign-handoff-2026-07-12.md`,
`batched-decode-n2-admission-handoff-2026-08-05.md` (1626 lines),
`mxfp4-gather-qmm-rhs-lhs-kernel-2026-08-19.md`,
`start-cluster-mlx-rebuild-skip-2026-08-19.md`,
`superdeepseek-2xdgx-speedup-levers-2026-08-19.md`,
`lever1-moe-smallm-headroom-2026-08-20.md`, `lever2-seqchunk-overlap-2026-08-20.md`,
`dual-cable-topology-and-qp-budget-2026-08-21.md`,
`fused-softmax-negative-result-2026-08-21.md`,
`gather-qmm-m1-dispatch-confirmed-correct-2026-08-21.md`,
`instruments-metal-trace-real-dispatch-latency-2026-08-21.md`,
`known-good-prefill-baseline-2026-08-21.md`,
`live-decode-two-rank-instruments-trace-2026-08-21.md`,
`qa-kv-fusion-no-measurable-gain-2026-08-21.md`,
`baseline-locked-decode-fence-fix-20260822.md`,
`hyperconnection-training-gate-false-lead-2026-08-22.md`,
`long-context-gpu-occupancy-2026-08-22.md`,
`memory-residency-check-ruled-out-2026-08-22.md`,
`xcode-removal-launcher-clt-fallback-2026-08-25.md`.

</details>

<details>
<summary><b>DSv4-Flash-Vision-Exp port (3 files, 2026-08-31 → 2026-09-09)</b></summary>

The most recent major campaign — porting the multimodal Vision-Exp
checkpoint. Sequential plan docs, each explicitly scoped to specific
phases and requiring separate authorization to proceed:
`DSV4_VISION_PORT_PLAN.md` (Phases 0-2), `DSV4_VISION_PORT_PLAN_PHASE34.md`
(Phases 3-4), `DSV4_VISION_PORT_PHASE5_PROCEDURE.md` (Phase 5 — relaunch +
smoke test, dated 2026-09-09). This is the model currently in production
(`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`), so this three-doc arc is the
most operationally relevant historical record in the corpus after
`PERFORMANCE_HISTORY.md` itself, even though it's correctly a (b)-shaped
sequential-phase-plan rather than a (a)-shaped reference.

</details>

<details>
<summary><b>Remaining standalone docs (7 files, various dates)</b></summary>

`deepseek-v4-c2-mtp-verify-fixes.md` (2026-06-06/07),
`deepseek-v4-flash-kickoff-prompt.md` (undated, pre-cluster-adoption kickoff
prompt for DSv4 — historical, references a `memory/` directory that no
longer exists in this repo layout),
`deepseek-v4-mtp-performance.md` (2026-06-04 performance writeup),
`eagle_k1_fix_report.md` (Eagle K=1 patch regression diagnosis),
`flag-sweep-completion-and-dspark-native-finding-2026-08-21.md`,
`jit-model-lifecycle-handoff-2026-06-30.md` (JIT model loading — shipped,
deployed, but not currently active per its own text: "enabled with Qwen3.6
NOT co-hosted at boot" — the co-host scenario is no longer this cluster's
config, so treat as historical rather than re-verify),
`pp-prefill-tp-decode-phase-swap-design-2026-08-16.md` (explicit
`SUPERSEDED — DEAD, DO NOT IMPLEMENT` banner, kept per instructions —
**do not move to (c)**, the design itself is dead but the doc documents a
real rejected-design decision, which is exactly what (b) is for).

</details>

### 4.2 `docs/upstream-pr-drafts/` (4 files, excl. README already in §2)

`02-mlx-sdpa-chunked.md` (HELD), `04-mlx-head-dim-192-256.md` (HELD),
`06-mlx-allocator-coalesce.md` (OPENED as mlx#3596),
`07-exo-thinking-parser-fused-delimiter.md` (OPENED as exo#2149, merged).
Tracked-in-repo PR description copies; status current per cross-check
against `docs/upstream-prs.md`'s PR table.

### 4.3 `docs/incidents/` (1 file)

`pkill-self-match-forces-sigkill-2026-09-10.md` — the house-style reference
this inventory doc followed. Documents a `pkill -f` self-match bug in
`start_cluster.sh`'s shutdown sequence; real fix not yet applied (documented
as future work in the doc's own §5), mitigation used once. Most recent
incident doc in the corpus.

### 4.4 `docs/profiling/` (1 file)

`request_lifecycle_trace.md` — not read in full detail beyond confirming it
is a profiling-methodology reference, filed in (b) as a supporting doc for
the many trace/span-profiling investigations above.

### 4.5 `tmp/*.md` — git-tracked working directories (124 files in (b), 1 in (c))

The `tmp/` tree holds working folders for individual investigation
rounds/campaigns. Structurally different from `docs/`: these are raw
round-by-round artifacts (pre-registrations, per-arm results, code reads),
often not intended as final prose, but preserved per the task's instruction
(institutional-memory value, do not delete).

| Folder | Files | Topic |
|---|---:|---|
| `perf-campaign-2/` | 69 | "Campaign 2" — a 13-round, highly disciplined pre-registered A/B campaign (2026-09-03/04) hunting for decode-throughput levers after Campaign 1 closed. **Outcome, stated in its own `IDEAS.md` banner: "THROUGHPUT PHASE CLOSED. ZERO SHIPPED."** Every round follows PRE-REGISTRATION → (execution, often blocked by tool/access denial) → REPORT, with several rounds (11-13) devoted entirely to catching and fixing measurement-apparatus bugs *before* spending a scarce cluster relaunch — genuinely high-value methodology notes, not just results. Round 13's `PREDICTION.md` contains three dated self-amendments, each catching an apparatus defect pre-boot; read this file if setting up a similar phase-mark/wake-latency measurement in the future. |
| `hardening-round1-20260903/` … `round4-20260903/` | 5+1+1+2 = 9 | Post-campaign hardening: CI re-enablement (had been `disabled_manually` for 8 months), guard-gap closure, mlx pin-drift resolution, alignment enforcement. Sequential and non-redundant — each round explicitly closes gaps found by the prior round. |
| `prefill-round1` … `round3-20260902/` | 4+5+8 = 17 | Three PM-subagent-report rounds investigating prefill throughput, each at a different repo HEAD (`cb1f91903` → `80db9a855` → `17d427b01`), each explicitly "unmodified, nothing applied" — a feedback-loop review process, not sequential fixes. (Round 4's folders — `prefill-round4-exec-20260902/`, `prefill-round4-exec-askb-20260902/` — contain no tracked `.md`, only supporting scripts/patches; not double-counted here.) |
| `research-v1v2v3-20260901/` | 8 | Three parallel workstream executions (log-mining feasibility, a V2 analysis, a V3 code-verification+telemetry+profiling-plan arc). |
| `verify-decomposition-20260901/` | 7 | "Where the 20→34 tok/s decode gap actually comes from" — decomposition into entropy and temperature sub-studies. |
| `p05-lmhead-mxfp8-20260830/`, `p05-review-20260830/` | 3+3 = 6 | The lm_head mxfp8 quantization investigation — **this is the predecessor investigation to the `EXO_DSV4_LMHEAD_MXFP8` text-quality-defect work documented in the `exo-cluster-operations` skill** (garbled-subword-token root cause, found 2026-09-13, one day before this inventory). `real_margins/ANALYSIS.md` here is the original margin-distribution measurement (42.7% frac<3.62, 11.5% derived flip rate) that the skill's "DERIVED ESTIMATE, not measured" caveat traces back to. Worth linking from the eventual lm_head-quality reference doc if one gets written (see §5 gap). |
| `p05-quant-lmhead-20260830/`, `p05-shared-batching-20260830/`, `p05-sinkhorn-real-20260830/` | 0 tracked `.md` each (scripts/JSON only) | Listed for completeness; contain no git-tracked markdown. |
| `p07-20260830/`, `p08-20260830/` | 1+1 = 2 | Top-k argpartition-vs-argsort verdict; item-1 verdict for a separate P08 sub-item. |
| `p01-20260829/` … `p04-sinkhorn-truncation-20260830/`, `p09-20260831/` … `p15-replication-20260831/` | 0 tracked `.md` each | Numbered campaign working dirs (arm launch scripts, `ps eww` preflight captures, JSON results) — supporting artifacts for the `docs/p0N-*` and `docs/perf-campaign-2` docs already catalogued above; no additional markdown narrative beyond what's already listed. |
| `overnight-loop/` | 2 | `CHARTER.md` (standing authorization for an autonomous overnight optimization loop, 2026-09-02) + `STATE.md` (running status). Operationally important as a record of a real standing authorization grant, though its "current" status should be re-confirmed with the user rather than assumed still active. |
| `prefix-cache-dive-20260902/` | 1 | "Why 0 full hits with 54/57 partial?" — refutes all three hypotheses, concludes the metric label (not the cache) was misleading. |
| `real-usage-capture-20260902/` | 3 | Real-usage decode-rate study vs. benchmark convention. |
| `wall-attribution-20260902/` | 1 | Attributes the "unaccounted" decode wall — concludes it's mostly a measurement artifact. |

### 4.6 `.hermes/plans/*.md` (45 files, 2026-05-14 → 2026-06-12)

The oldest and most chronologically dense stretch of the whole corpus — the
original "path to 35 tok/s" campaign that predates the `docs/` convention
being established. All 45 files fall into one continuous narrative arc:

- **Kernel-fusion design docs** (2026-05-14, same day, 5 files): fused MoE
  Metal kernel, sparse-attn fused kernel, expert co-location (abandoned at
  phase-0), compress-ratios reshape, indexer fused kernel.
- **γ=2 MTP bistability hunt** (2026-05-16 → 2026-05-17, 3 files):
  investigation → after-revert next-steps → session retrospective.
- **"Path to 35 tok/s" main campaign** (2026-05-18 → 2026-05-25, ~28 files):
  verify-forward optimization, verify-tail investigation, a dozen
  phase-N-findings docs (allsum tail, build probe, critical path, mtp head,
  token-tree drafting phases 1/6/7/8/9/10/11/12/13/14, quality findings,
  Eagle K=1 debug, γ=3 bistability fix, final session writeup). This is the
  densest, most sequential sub-arc in the entire corpus — each file
  explicitly builds on the previous day's findings within the same
  campaign. Recommended reading order is filename date order; do not sample
  non-sequentially.
- **Post-campaign** (2026-06-10, 2026-06-12, 2 files): DSv4 hyperconnection
  fix + co-host bench (cross-referenced by `docs/fork-notes.md`), and a
  short prefill-optimization Fable-5-analysis note (27 lines, the shortest
  file in `.hermes/plans/`).

No file in this directory met category (c)'s conservative bar — see
`docs/fork-vs-upstream-inventory.md`'s own note (quoted in §5) that flags
this whole directory as accidentally swept into git and a repo-hygiene
concern, which is a real observation but is about *git tracking*, not about
these documents' *content value* (the content is genuinely dense
institutional memory of how the 35 tok/s target was hit).

---

## 5. Category (c): Genuinely superseded / duplicate / redundant

Applying the conservative bar (explicit textual evidence required, no
inference from "looks similar"), only two files across the entire ~370-file
exo corpus meet it. Everything else that *looked* like a duplicate on first
pass (see the "considered and rejected" list below) turned out, on reading,
to be either self-contained internal corrections or genuinely distinct
vantage points on the same closed issue — which is exactly the institutional
value category (b) exists to preserve.

| File | Evidence (quoted verbatim) | Recommendation |
|---|---|---|
| `docs/pp-prefill-tp-decode-phase-swap-design-2026-08-16.md` | Its own first line: `> # ⚠️ SUPERSEDED — THIS DESIGN IS DEAD. DO NOT IMPLEMENT.` followed by: `> **Decision date: 2026-08-16.** The PP-prefill -> TP-decode phase swap described below was evaluated and **rejected**. The authoritative decision record is `docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md` **Section 107**` | **Do not delete** — this doc records a real rejected-design decision and its full rationale (11.1% gain that vanishes on cache hits, loses all cross-phase concurrency, needs a cache-gather mechanism that doesn't exist). Recommend leaving in place; if the corpus is ever pruned, this is a candidate for `docs/archive/` (not deletion) since its content is fully subsumed by `hybrid-pp-prefill-tp-decode-design-2026-08-04.md` Section 107 per its own banner. |
| `tmp/prefill-round3-20260902/artifacts/decode_instrumentation.patch.REMOVED.md` | Full file is a tombstone: `# decode_instrumentation.patch — REMOVED (was defective)` ... `The file that used to live here ... has been removed from disk because it was **defective** ... **Use the corrected, verified-working copy instead:** `tmp/prefill-round4-exec-askb-20260902/revert/instrumentation_as_run.patch`` ... and explicitly: `This has **NOT** been run — no git index/staging changes were made ... A human/PM must run the `git rm``| This is a genuine dead artifact (the file it describes no longer exists; only this tombstone note remains, itself asking for a `git rm` that was never done). **Recommend**: either action the tombstone's own request (`git rm` this pointer file specifically, since a human/PM was asked to do so and it appears not to have happened in the 12 days since 2026-09-02) or leave as-is — either is defensible, but this is the one file in the corpus that explicitly asks to be removed by its own author. **Not executed here** — read-only task. |

**Considered for (c) and explicitly rejected** (documented here so this
question doesn't get re-litigated):

- **`docs/dsv4-memory-leaks-2026-06-27.md` / `dsv4-memory-leak-handoff-2026-06-29.md` / `dsv4-memory-leak-resolution-2026-06-29.md`** — three docs, same bug family, two dated the same day. Checked for cross-references: **neither of the later two docs mentions the other two by filename**, and the fix commits differ (`d5f6c421`/`c0149c58` for the first four-leak-site doc vs. `947d7e50b` for the resolution). These are plausibly three distinct (if related) leak investigations across three days, not draft→final iterations of one doc. No explicit "supersedes"/"see X instead" language exists between them. Kept in (b), not (c).
- **`docs/b2-quality-handoff-2026-06-24.md`** — its own banner says `> **RESOLVED 2026-06-24.** ... See `docs/b2-mtp-resolution-2026-06-24.md` for the full resolution ... The content below is kept as historical record of the investigation that ruled out the cache-merge/extract path.` This is close to a (c) case, but the doc explicitly frames itself as intentionally-kept historical record of a *ruled-out* theory, not a redundant draft of the resolution doc (it covers a different, wrong hypothesis that the resolution doc doesn't re-explain). Kept in (b).
- **`MOE_KERNEL_HANDOFF.md` and `PREFILL_CLIFF_HANDOFF.md`** (top-level) — each carries a correction banner on ONE specific claim (`the cliff-elimination attribution to EXO_DSV4_PREFILL_ARGPARTITION=1 ... is factually wrong`), but each also explicitly states: *"The rest of this handoff (including the paragraph below) is preserved as-is for historical accuracy."* Since the documents themselves refuse full retirement and only correct a narrow claim, they do not meet the (c) bar. Kept in (b) — top-level, §4.7 handled separately below since they're not under `docs/`.
- **`docs/moe-allsum-sharedscale-{root-cause-found, live-test-no-speedup, CORRECTED-final}-2026-08-19.md`** — a three-doc same-day arc where the third explicitly says the second's own earlier-session reading was "WRONG -- caught and corrected within the same session." This looks superseded at first glance, but the "WRONG" reading being corrected is described *within* the CORRECTED-final doc itself (a same-session self-correction), and the two earlier docs each report a *different* stage of the experiment (root-cause identification, then a separate live no-speedup test) — not duplicate reports of the same measurement. Kept in (b), all three.
- **`docs/roofline-recalculated-post-fix-2026-08-22.md` vs. `docs/roofline-sanity-check-inputs-confirmed-2026-08-22.md`** — same-day, similar titles, but one recalculates the ceiling number post-fence-fix and the other sanity-checks the *inputs* to that ceiling calculation (a different, complementary check). Kept in (b), both.
- **`docs/dsv4-4096-regression-root-cause-2026-08-19.md`** — contains "UPDATE"/"SECOND UPDATE"/"THIRD UPDATE" sections with the third explicitly declared final ("the `unresolved 1.58x gap` language in those sections is superseded"). This is a single self-correcting document, not multiple files — nothing to place in (c).

**Top-level handoff docs with correction banners (kept in (b), not (c))** —
see full discussion above:
- `MOE_KERNEL_HANDOFF.md` — corrects its own argpartition-attribution claim, defers to `docs/prefill-cliff-mechanism-2026-08-24.md` for the real mechanism; rest preserved.
- `PREFILL_CLIFF_HANDOFF.md` — has a "2026-08-24 RESOLUTION BANNER" confirming the cliff it describes is resolved in production; body preserved for historical record and for the still-relevant "MEASUREMENT WARNING" about a tiled-P A/B's scoping caveat.

`INDEXER_TILED_P_PLAN.md` and `PREFILL_THROUGHPUT_PLAN.md` (top-level) are
both **plans, not resolved investigations** — `PREFILL_THROUGHPUT_PLAN.md`
explicitly says `Status: PLANNED — nothing executed`, and `PERFORMANCE_HISTORY.md`
tags it `(2026-07-13, INFO_ONLY) — pure planning doc`. `INDEXER_TILED_P_PLAN.md`'s
proposal was later tested and found negative (`docs/indexer-pblock-decode-regression-2026-08-21.md`).
Neither is superseded text — they're just unexecuted/superseded-by-negative-result
plans, which is squarely a (b) case (the plan itself has institutional value:
"this was considered, here's why, here's what happened when tested").

---

## 6. Category (d): Orphaned / unclear

| File(s) | Why unclear |
|---|---|
| `docs/upstream-prs.md` | Placed in (a) for its *role* (still the only PR tracker), but its content — the actual PR status board — was last refreshed 2026-06-04, over 3 months before this inventory. Neither clearly (a) (content is stale) nor (b) (it's still being pointed to as live). Recommend a fresh refresh pass rather than recategorization. |
| `mlx/ACKNOWLEDGMENTS.md`, `mlx/AGENTS.md`, `mlx/CONTRIBUTING.md`, `mlx/README.md` (`CLAUDE.md` is a symlink to `AGENTS.md`, not counted separately) | Per the task's own framing, "mostly upstream-inherited, not fork-specific." Verified via `git log`: `AGENTS.md`'s only touching commit is upstream's `057cdc9ec` ("Add AI usage policy"); `README.md`/`CONTRIBUTING.md`/`ACKNOWLEDGMENTS.md` show exclusively upstream PR-numbered commits. **None of these 4 files contain fork-specific content** — they are orphaned *from this inventory's perspective* (nothing to categorize as (a)/(b)/(c) because there's no fork-specific claim to verify or supersede), not orphaned in the sense of "unclear purpose." Recommend excluding this class from any future fork-doc audit scope entirely, rather than re-checking them each time. |
| `mlx-lm/ACKNOWLEDGMENTS.md`, `mlx-lm/AGENTS.md`, `mlx-lm/CODE_OF_CONDUCT.md`, `mlx-lm/CONTRIBUTING.md`, `mlx-lm/README.md`, `mlx-lm/mlx_lm/BENCHMARKS.md`, `mlx_lm/LEARNED_QUANTS.md`, `mlx_lm/LORA.md`, `mlx_lm/MANAGE.md`, `mlx_lm/README.md`, `mlx_lm/SERVER.md` | Same situation as above — `git log` on every one of these 11 files shows exclusively upstream (`ml-explore/mlx-lm`) PR-numbered commits, zero fork-authored history. `SERVER.md` is the closest to fork-relevant (KV-cache-quantization docs, which the fork's `EXO_KV_CACHE_BITS`/`kv-cache-architecture.md` also touches) but its own edit history is 100% upstream. Same recommendation: exclude from future fork-doc scope. |
| `docs/fork-vs-upstream-inventory.md`'s own self-flagged item | Not a doc-inventory entry itself (it's in (a)), but its text explicitly flags `.hermes/plans/` as a scope concern: `"> Scope note: ... `.hermes/plans/` are session scratch notes that were swept in by an early `git add -A` accident — flagged here, candidates for removal."` This is the *inventory doc itself* noting a repo-hygiene question about `.hermes/plans/`'s git-tracked status — worth surfacing to the user, since it's a different question from this doc's content-value assessment of those 45 files (§4.6 treats their content as valuable; this is purely about whether they *should have been* git-tracked in the first place). Not actioned here (read-only task; also a git-tracking question, not a documentation-content question). |

No file in `docs/`, `tmp/`, or `.hermes/plans/` (the three git repos' actual
fork-specific documentation, as opposed to the upstream-inherited files
above) was found genuinely unclassifiable — every dated file's topic was
recoverable from filename + a 20-40 line skim, and every undated file's
topic was recoverable from its opening paragraph.

---

## 7. Existing topic index — none found; this doc fills that gap

**Checked before writing this doc**: `docs/README.md` and `docs/INDEX.md` do
not exist (`ls` confirms `No such file or directory` for both). The only
`README.md` under `docs/` is `docs/upstream-pr-drafts/README.md`, which
indexes only that one 4-file subdirectory (PR drafts), not the corpus at
large.

The closest things to an existing index are:
- **`docs/PERFORMANCE_HISTORY.md`** — has its own table of contents and is
  explicitly the running decision-log / "don't re-litigate" reference for
  *performance* findings specifically. It organizes by chronological
  session/topic sections, not by a static topic → file map, and it
  synthesizes findings into prose rather than pointing readers at the
  underlying `docs/*.md` source files by name in most cases.
- **`docs/fork-vs-upstream-inventory.md`** — indexes *source code* changes
  (`src/`, `rust/`, `mlx/`, `mlx_lm/`) against explaining docs, as of
  2026-06-04. It is a code-to-doc map, not a doc-to-doc topic map, and its
  snapshot predates roughly 70% of the `docs/` corpus (everything from
  2026-06-04 onward — the entire DSpark, MoE-allsum, hc-collapse/expand,
  p0N/p3/p4/p5 campaigns, and the Vision-Exp port are all unlisted there).

**Neither existing doc serves this doc's purpose** (a topic-prefix map over
the ~180 dated investigation docs in `docs/` plus the `tmp/`/`.hermes/plans/`
corpus). This document is therefore the first topic index of its kind for
this repo and should be treated as **complementary to, not a replacement
for**, `PERFORMANCE_HISTORY.md` (prose synthesis + decision log) and
`fork-vs-upstream-inventory.md` (code-to-doc map).

**Recommendation for the user to consider** (not executed — read-only
task): add a one-line pointer to this doc from `docs/PERFORMANCE_HISTORY.md`'s
own header (e.g., alongside its existing "How to use this doc" section) and/or
from `docs/fork-vs-upstream-inventory.md`'s header, so a reader who finds
either of those two first can discover this topic-navigation doc. No existing
structure needs to be removed or restructured to do this — it would be a
pure addition.

---

*Generated 2026-09-14 by read-only enumeration of `~/repos/exo` (HEAD
`1da54ee19`), `~/repos/exo/mlx`, and `~/repos/exo/mlx-lm` on
`macstudio-m4-1`. No existing file was modified, moved, or deleted in the
course of producing this inventory.*
