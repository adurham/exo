# DSv4.1 — scaling alternatives: third node + pipeline-parallel — owner brief (2026-10-15)

**Status:** decision-support record, owner-requested ("document this please").
**No action triggered; nothing changed on the cluster, configs, or code.** All
measurements read-only. Live build unchanged: `deploy/next27-ringfix` @ `c758bcce5`.

**Context:** prefill is the owner's target axis ("I don't need to scale decode —
with MTP I'm getting plenty; I need to scale prefill"). This brief records (1) the
third-node assessment as of 2026-10-15, and (2) the pipeline-parallel re-examination
the owner raised the same day.

---

## 1. Third node — speed case vs capacity case

### 1.1 Speed case — assessed WEAK by owner

- Modeled gain: **+24–39 % prefill** (centre ≈ +30 %) over the current 2-node
  baseline; decode flat ± a few %. Gated on the TP3 support package + a 1–2 d
  query-axis pad. `[MODELED — p20/tp3-scaling-readiness addendum, 2026-10-12]`
- Felt translation `[DERIVED]`: cold ~100K feed ≈ 6.1 → 4.6 min; a session like
  2026-10-15's ≈ prefill 12.3 → ~9.1 min of model time (−7 % model time, ~4–5 % of
  total wall).
- Support cost `[ESTIMATE, from the study]`: ≈ **15–32 eng-days** (transport N>2
  3–6 d = critical path; w=3 geometry 3–11 d; launcher/topology 4–8 d; pad 1–2 d;
  validation round 1–2 d) **+ hardware** (same-gen M4 Max, 128 GB, ≥1 TB SSD,
  matching macOS; **two** TB5 cables, one to each existing node; two distinct HCAs
  at `max_qp=3`).
- **Owner assessment (recorded, 2026-10-15):** "it only gets me 30 % prefill
  boost … that's not a lot, esp for the cost of an additional node."
- Anchor correction (owner, same day): the old "400+ t/s prefill" figures belong
  to the **smaller DSv4-Flash era** (350–450 t/s range) — **not a DSv4.1 target**;
  DSv4.1 (552B-class, ~211 GB served) is expected to run lower. Current live cold
  measurements: **300.1 / 291.5 rows/s @ 50K / ~105K** `[MEASURED, qsplit live A/B #2]`.
- For completeness — the 2-node prefill path is **measured-exhausted**: indexer
  abs-cost workstream closed STOP (2026-10-12: bypass EV 0.015 %; K-read dead
  0.3–1.3 %; int8 probe 1.34–1.35× below its 1.5× bar); prefill GPU-busy
  **92.6–96.8 %** (kernels are the wall — no host slack); MoE kernel project parked
  permanently (kill gate #2); small-n partition trick shelved (+2.9 %); dense quant
  owner-closed. Remaining 2-node items: SP-hybrid **~+4 %** wall (12–25 d, unbuilt,
  ranked "consider only"); one unstarted fused-kernel follow-up candidate (needs
  its own plan).

### 1.2 Capacity case — the owner's actual rationale

- A 3rd node = a full **~126 GB wired budget DSv4.1 never touches** → room for a
  **secondary model running alongside** (owner's stated use: e.g. a current Qwen
  that can "consult" DSv4.1 / take consult + aux roles).
- Measured 2026-10-15 (both nodes, `/state`, read-only): `ramAvailable` ≈
  **19.97 / 20.11 GB** of 137.4 GB per node; under `iogpu.wired_limit_mb=120000`
  (125.83 GB) the GPU-side headroom after DSv4.1's 112–116 GB peak is
  ≈ **10–14 GB** `[DERIVED]` — only tiny aux models fit (a Qwen3.5-0.8B and a
  Qwen3-Coder-Next-4bit are already on the node disks; nothing consult-grade fits
  today). This is why the old Qwen3.6-35B co-host era ended — DSv4.1 solo absorbs
  the budget.
- On a dedicated node: 35B-8bit class (36 GB) trivial; considerably larger feasible.
- Plumbing options:
  - **(a) Standalone** — node 3 runs its own single-node server; register as a
    second local provider in Hermes. No changes to the main cluster. ~An afternoon.
  - **(b) Integrated** — node 3 joins the existing cluster (unified placement/JIT;
    keeps the TP3 path open). Needs the **launcher/topology half** of the TP3
    package (~4–8 d `[ESTIMATE]`), not the w=3 geometry/transport half. Placement
    behavior for the 2-rank DSv4.1 model on a 3-node cluster is unverified (the
    study's placement predicate reportedly filters 3-cycles for this model —
    5120 % 3 ≠ 0 — verify on the day).
- Hermes-side wiring: model routing — consult-tool target (currently Fable/cloud),
  aux tasks (title/approval/background_review currently ollama-cloud), delegation
  roles. Standard config (hermes-config-administration conventions).
- **Recorded verdict: buy = "second brain + headroom + TP3 optionality"; the
  prefill bump is a side effect, not the case.** Cross-ref: the 2026-10-13
  big-append census independently listed "the 3rd node" among the owner levers.

---

## 2. Pipeline-parallel (PP) — re-examined at owner request (2026-10-15)

### 2.1 Owner's recollection vs the record

- "I stopped work on that before because it was nowhere near as mature — it
  couldn't cancel in-flight requests or support c=2" — **CONFIRMED as recorded**:
  the closure names "all cross-phase concurrency (Requirement 1) and cancellation,
  which TP provides natively" among the hybrid's costs
  (`docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md` §107; decision memo
  2026-08-16).
- "clean split … pay the RDMA sync tax once … much better prefill throughput at
  the cost of worse decode" — **CONFIRMED for the DSv4-Flash era**: PP's structural
  prefill edge (each node owns whole layers; one activation handoff per stage
  boundary; zero per-layer sync) vs TP's per-layer `all_sum` tax. Old-config
  measurements: PP ≈ 444 t/s @500K (July PP config); final 500K closure math:
  **PP 364 vs TP 319** `[MEASURED, old era]`. Decode worse on PP.

### 2.2 Why it is closed (not merely immature)

- **2026-08-16 decision (authoritative in-repo record: the hybrid doc's CURRENT
  STATE header + §§107–108; the phase-swap doc is marked SUPERSEDED):** **TP for
  BOTH prefill and decode.**
  - Hybrid PP-prefill → TP-decode dead: co-residency impossible (weight-layout
    union ≈ 125.4 GB of 128 GB → needs a per-request sequential swap); swap priced:
    **best case ~11.1 % on a cold 500K request, ZERO on prefix-cache follow-ups
    (−36 s net on a 2K follow-up)**; break-even 51K–96K NEW prefill tokens; costs
    all cross-phase concurrency + cancellation; needs a cross-rank cache gather
    that never existed; adds a second failure domain mid-request. "The swap
    converts a miss into a smaller miss."
  - Expert-locality placement also dead (§108: TP width-shards experts — every
    rank holds all experts at half width; nothing to bind).
- A standing decision memo exists (operator skill archive:
  `~/.hermes/skills/.archive/exo-sharding-mode-tradeoffs/references/pp-tp-architecture-decision-2026-08-16.md`)
  recording the decision and specific rejected re-openings.

### 2.3 For DSv4.1 specifically, PP does not exist

- The engine refuses multi-rank pipeline sharding (verified in the live tree,
  2026-10-15): `src/exo/worker/engines/mlx/dsv41/load.py` —
  *"DSv4.1 does not support multi-rank Pipeline sharding: its layers share
  per-forward state and cross-layer caches (compressed KV, index top-k, engram ids)
  that cannot cross a layer-range boundary. Place it with Tensor sharding (the
  model card sets supports_tensor=true)."*
- So unlike DSv4-Flash (PP existed, ran, was measured), a DSv4.1 layer-range split
  would be a **from-scratch engine feature**: transport of the cross-layer state
  (indexer top-k computed at source layers / reused by consumer layers, candidate
  layers, compressed-KV carry, engram ids) across the stage boundary. Unscoped; no
  estimate exists.
- **Decode premise check** ("if MTP can fill that gap"): PP+DSpark speculative
  decode measured **27–33 t/s** single-request (2026-07-23 sweep, fork-notes; the
  then-TP baseline it was compared against is now known stale); current production
  TP+MTP: **34.6 t/s** decode (2026-10-15 session on next27) `[MEASURED]`. **No
  decode gap exists for MTP to fill — PP+MTP ≈/below current TP+MTP.** Also on
  record: the standing directive from the 2026-08-16 settlement — MTP/DSpark
  decode goes through TP.
- **If the owner ever wants PP examined as a DSv4.1 question anyway:** first step =
  an offline scoping pass (mirror of the TP3 / attention-sharding studies): can the
  engine take a layer-range split with explicit cross-layer state transport;
  modeled prefill/decode band; effort vs decode risk. **NOT started; requires
  owner go.**

---

## 3. Recorded state

- Nothing changed anywhere (cluster, configs, code); all measurements read-only;
  live build unchanged.
- Open items, for the record: (1) 3rd node as capacity buy — plumbing (a) or (b);
  (2) SP-hybrid (~+4 %, 12–25 d) on the pair; (3) fused-kernel follow-up candidate
  (needs its own plan); (4) DSv4.1-PP scoping pass (unstarted; §2.3).
- Provenance: tp3 studies (`p20/tp3-scaling-readiness` + addendum 2026-10-12);
  attention-sharding scope (`p20/attention-sharding-scope`); indexer abs-cost
  closeouts (`5984ddad8`, `d53666166`); PERFORMANCE_HISTORY 2026-10-11/12/13
  entries; live `/state` 2026-10-15; session `20261015_143731_a6c67d` (build
  `next27-ringfix`). Owner statements recorded with intent (verbatim where quoted).

### 2.4 Owner follow-up (2026-10-15, same day)

Owner note recorded: **"but it could be built."** — correct, and stated plainly for the record: the engine's refusal (§2.3) is a statement about the current code, not an impossibility. A DSv4.1 pipeline build is an open owner-option.

- **Build shape** (from the code facts on file): ~22/21 layer split, each stage full-width (the layer-subset hook in `load.py` + the world=1 path already exist — today as test facilities); boundary-aligned placement of the 8 indexer source→consumer groups so top-k reuse stays intra-stage; transport across the boundary of the cross-layer state the loader names (top-k indices ~[rows, 512] int32 × 8 source layers, compressed-KV carry, engram ids, candidate-layer state) plus the activation handoff; spec-decode across stages (never existed on this engine); and the maturity items the old PP path lacked (cancellation, cross-phase concurrency — the batched-PP design, never shipped).
- **Honest expected value** (only measured anchors: old model, old configs): PP prefill edge ~+14 % at 500K (364 vs 319); hybrid realized ~11 % best case cold-500K / zero on cached follow-ups; decode worse. DSv4.1 makes parts harder (bigger model → worse co-residency/swap math; the cross-layer transport is new cost the old PP never paid) and current TP has had 3+ months of optimization since. Expected class `[guess]`: single-digit-to-low-teens % prefill, plus decode risk and a reopened-architecture cost — vs TP3's +24–39 % (needs a node) and SP's ~+4 % (2-node, cheap). It is the only remaining double-digit 2-node prefill candidate, and the least scoped.
- **Entry point**: the offline scoping pass (mirror of the TP3 / attention-sharding studies) — can the engine take boundary-aligned layer ranges + explicit state transport; modeled band on DSv4.1; effort. **Unstarted; requires owner go.**

#### Estimate refinement — PP prefill decomposition (2026-10-15, owner-requested)

Decomposition-grade estimate `[DERIVED/GUESS]`, applied to the measured current-TP rates:

`PP edge ≈ (collective-sync share + per-chunk barrier + HALF the replicated row-local share) − (stage imbalance + handoff + cross-layer state transport)`

- **Removes**: collective sync ~9–12 % of chunk wall @~100K (falling with depth; in-model instruments 16.9 % @16K → 6.9 % @340K; TP3 doc bound 1–24 %); per-chunk `mx_barrier` ~1–2 %; replicated/2 ≈ 4.9–5.2 % @50–100K (PP runs each duplicated op once on one node, not twice in parallel — only HALF the replicated share is saved).
- **Adds**: stage imbalance ~1–3 % (largely pipelined away across chunks), one activation handoff per chunk (negligible), new cross-layer state transport ~1–3 % (guess).
- **Result `[DERIVED]`:** **≈ +14–18 % at 50–105K (≈ 335–355 t/s vs 291–300 measured cold), tapering to ~+7 % deep (≈ 205 vs 191 marginal @~490K)**; uncertainty ≈ ±10 points.
- **Cross-checks / caveats:** only hard anchor is old-era PP 364 vs TP 319 @500K = +14.1 % `[MEASURED, old model]` — partially attention-replication-driven (already fixed in current TP), so today's deep edge is expected smaller; a high-end sync share would push it up. Assumes a clean build (boundary placement vs the 8 indexer source→consumer groups; small state-transport cost). Decode side not included. Scoping pass remains the way to a modeled band.


## 4. Update — 2026-10-15 (later): owner GO on the scoping study

Owner directive: consult Fable, then run the PP scoping study ("go for it"). Fable reviewed and RESTRUCTURED the plan: **Gate 0 first** — the decode-parity inequality bounded theoretically with zero build (flagged as the likely single kill), old-model calibration of the decomposition within about 3pp, pipeline-fill-bubble-weighted prefill gain using the real prefill-length distribution, and a backward-dependency check over all cross-layer state; the full study runs only if Gate 0 passes. Kill criteria pre-registered (decode below about 95 percent of 34.6 tok/s; calibration miss beyond 3pp; workload-weighted gain below about +8 percent; non-eliminable backward edge). Fable also required: pricing of the TP-native alternative (sequence parallelism + compute/comm overlap) as the opportunity-cost baseline, and a gain-vs-effort table for the owner's build-commit ruling later. Study dispatched offline — branch p20/pp-scoping-2026-10-15; deliverable docs/dsv41-pp-scoping-2026-10-15.md + PH entry; 0 boots; live untouched.
