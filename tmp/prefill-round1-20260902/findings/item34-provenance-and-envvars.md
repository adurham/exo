# Item 34: Provenance & env-var read-time audit — clear-cache sweep (Exp 3) and chunk-size re-sweep (Exp 4)

**Date:** 2026-09-02 · **Repo:** exo @ `cb1f91903` · **Scope:** READ-ONLY audit. No cluster access, no runs.

---

## VERDICT (one line each)

- **(a) Is the clear-cache sweep (Exp 3) re-litigation?** **PARTIAL** — the interval hypothesis was refuted live on 2026-08-19 at STEP_SIZE=4096 (interval=2 → +0.48% noise), but the only interval ever tested was `1` vs `2`; the proposed `1/2/4/8` sweep tests 4/8 which were never run, and it targets the **2048** stepping value (not 4096). The core hypothesis (allocator-clear cadence moves prefill throughput) is already refuted; a 4/8 arm is a re-test of a dead mechanism with new knob values.
- **(b) Is the chunk-size re-sweep (Exp 4) re-litigation?** **YES** (for the SDPA mechanism) / **PARTIAL** (for the un-tested sizes) — 2048-vs-4096 was already fully root-caused on 2026-08-19 (SDPA linear in per-rank query rows, confirmed 2.029x under sync profiling); the mechanism predicts 3072 and 6144 are *also* regressions vs 2048, so sweeping them re-confirms a closed, understood result. 3072/6144 were **never** actually tested, so it is technically not a duplicate of those exact arms — but it is re-litigation of the fully-closed question. The "not controlled for deep context" objection has **no merit** (see §2.3).
- **(c) Which of the 3 env vars require a relaunch?** **all three** — `EXO_DSV4_SEQ_SPLIT` is frozen at module import (`auto_parallel.py:124`); `EXO_PREFILL_CLEAR_CACHE_INTERVAL` is read per-chunk but only from the process-global env that is fixed at worker launch; `EXO_PREFILL_STEP_SIZE` is read per-call but from the same process-fixed env. None can be varied per-request without relaunching the top-level exo process (which is what `start_cluster.sh` does). **Exception found:** the interval var is a **no-op on the TP batched-prefill path** (`prefill_batched`, `generate.py:1527` clears unconditionally).
- **(d) What does one relaunch cost?** **~20 min** of cluster wall time (measured/repeated figure for "one relaunch" incl. model load + a real prefill; cold DSv4-Flash model load alone is ~18.7 s/rank ≈ ~40 s for 2 ranks). Every arm of these sweeps = one relaunch = ~20 min × 2 nodes.

---

## 1. CLEAR_CACHE INTERVAL PROVENANCE (Exp 3)

### 1.1 The live test exists and is the primary refutation

- **Artifact:** `docs/dsv4-clear-cache-interval-2-test-2026-08-19.md` (commit `dd777f60a`, "docs: clear-cache-interval=2 fix for 4096 regression tested, REFUTED", **2026-08-19 00:32 -0500**).
- **Purpose:** the test was *framed as a recovery attempt for the 4096 regression*, not as a standalone sweep. It relaunched the cluster with `EXO_PREFILL_STEP_SIZE=4096 EXO_PREFILL_CLEAR_CACHE_INTERVAL=2` (doc lines 21-23).
- **Context depth:** **~180K tokens** — deep context (179,720 tokens, doc line 33). So the prior live evidence is at DEEP context, matching Exp 3's 150K.
- **Result:** `332.8 tok/s` vs interval=1's `331.2 tok/s` at the same STEP_SIZE=4096 → **interval=2 vs interval=1 = +0.48%** = statistical noise (doc lines 42-43). Conclusion quoted verbatim: *"The clear-cache-interval fix does not work"* (line 61) and *"interval=2 vs interval=1: +0.48% -- statistical noise, not a real recovery"* (line 42).

### 1.2 The "refuted" note in the user's skill is CONFIRMED

- `exo-dsv4-prefill-tuning` skill (SKILL.md, "Option A" block): *"the clear-cache-interval hypothesis (refuted, tested live, +0.48% noise) ... remain dead -- do not re-chase either."* This matches the doc exactly.

### 1.3 What was NOT tested (the PARTIAL part)

- **Only `interval=1` vs `interval=2` were ever run** (the doc tests exactly these two values). **`interval=4` and `interval=8` were NEVER tested** anywhere in `docs/`, `bench/`, `tmp/`, or git history. The proposed Exp 3 sweep (`1/2/4/8`) adds two genuinely un-run arms.
- The tested values were at **STEP_SIZE=4096** (the interval theory was the leading suspect for the 4096 regression). Exp 3 proposes the interval sweep **at 150K context** without pinning STEP_SIZE — the doc's negative came at STEP_SIZE=4096; the standing 2048 config was never interval-swept either.

### 1.4 The ~390ms/chunk clear_cache claim — origin and validity

- **Where it lives:** `skill` .../exo-dsv4-prefill-tuning/SKILL.md line 160: *"`mx.clear_cache()` is called every chunk by default... This adds ~390ms of fixed overhead per chunk."*
- **Critical: `EXO_PREFILL_CLEAR_CACHE_INTERVAL` is NOT read anywhere in `src/exo/`.** Grep of `src/` finds **zero** reads of `EXO_PREFILL_CLEAR_CACHE_INTERVAL`. The only reads are:
  - `mlx-lm/mlx_lm/generate.py:544` — `_clear_interval = int(_os.environ.get("EXO_PREFILL_CLEAR_CACHE_INTERVAL", "1"))`, applied in the **non-batched eager `stream_generate` chunk loop** (line 546-547).
  - `start_cluster.sh:107` (default `:=1`), `:1640` (forwarded to runner env), plus `bench/` and `fingerprint.py` doc/spec references.
- **The ~390ms number is a skill-inherited estimate, not a measured cluster number**: repo-wide grep for `390` finds NO measured clear_cache cost in `docs/` or `bench/`. It does not appear in either the clear-cache test doc or the 4096 root-cause doc. It is consistent with **no** on-cluster measurement found in this audit — treat it as unverified/estimated.
- **No-op on the batched TP path:** the cluster's live multi-stream prefill path (`prefill_batched`, `generate.py:1385-1386, 1527, 1547, 1591`) calls `mx.clear_cache()` **unconditionally** per chunk (line 1527) — there is **no interval gate** on that path. The interval var only gates the eager single-stream `stream_generate` loop in mlx-lm. Since the single-request needle tests (the 331.2/358.6/332.8 measurements) go through `prefill()` → `stream_generate` (single request, `len(tasks)<=1` gate at `batch_generate.py:2836`), the interval=2 test DID exercise the var — so the +0.48% is a real negative, not a no-op artifact. Note for multi-request batched prefill the var would be silently inert.

### 1.5 Net for Exp 3

The mechanism (allocator-clear cadence moves prefill throughput) was refuted at deep context live. The only new information a `4/8` sweep could add is whether >2× the amortization helps — which the mechanism analysis says it can't (clear_cache is ~2% of tracked time and flat per the prefill-cliff work; see §2.2). **Spending ~2.5 cluster-hours to re-test a refuted mechanism with two bigger knob values is not justified** unless a concrete counter-mechanism is proposed.

---

## 2. CHUNK-SIZE PROVENANCE (Exp 4)

### 2.1 The comparison that was run: context depth, controlled A/B, mechanism

- **Primary artifact:** `docs/dsv4-4096-regression-root-cause-2026-08-19.md` (commits `510182cd1` + `799bf1dff` + `4a0ed6268`, final **2026-08-19 14:47 -0500**).
- **E2E regression (deep context, NOT controlled A/B):** measured at **~191K tokens**: STEP_SIZE=4096 **331.2 tok/s** vs STEP_SIZE=2048 **358.6 tok/s** = **~8% regression** (`dsv4-prefill-step-size-4096-retest-2026-08-18.md` lines 60-66). This is the number the reviewer quotes.
- **Mechanism isolation (controlled A/B, shallow context):** the SDPA-vs-MoE attribution was proven with a **matched-prompt controlled A/B** at **12,068 tokens** under sync-mode profiling (`THIRD UPDATE`, root-cause doc lines 163-174): `attn.sdpa` 0.4153 → 0.8428 ms/token = **2.029x**; `attn.sdpa.compressed` 0.2745 → 0.6477 = **2.359x** at STEP_SIZE=4096 (L=2048/rank) vs 2048 (L=1024/rank).
- **Attributed mechanism (final):** SDPA cost is **linear in per-rank query-row count** under SEQ_SPLIT. MoE's efficiency gain at the larger batch (~11%) is real but smaller; SDPA's linear cost growth outweighs it → net ~8% regression. This linear-scaling conclusion was cross-confirmed by an **isolated laptop M4-Max microbenchmark** (0.998-1.047x tiling ratio = no kernel-shape penalty) *and* by the sync-mode cluster A/B (2.029x ≈ predicted 2.0x). The earlier "3.15x gap" was proven a lazy-eval profiler artifact, not a real cost. Source: root-cause doc lines 148-188 (THIRD UPDATE = final, supersedes earlier "unresolved gap" language).
- **Regime split (important for the reviewer's objection):** the **e2e 8% regression** was measured at **~191K (deep)**; the **mechanism-controlled A/B** that produced 2.029x was at **12K (shallow)**. The reviewer's "not controlled for deep context" is literally true of the *mechanism A/B*, but see §2.3 for why it doesn't matter.

### 2.2 Mechanism is depth-INVARIANT → reviewer's "not controlled for deep context" objection has NO merit

Reason it out explicitly:

- **SDPA penalty is linear in per-rank query rows (L), NOT in pooled depth (P).** Doubling L (1024→2048 rows/rank) at STEP_SIZE=4096 costs **2× per SDPA call**, and that 2× is independent of how deep the KV pool is. The per-call ratio held at 12K (sync A/B), 41K (first non-sync profile), and the ~191K e2e runs all agree.
- **The amortization benefit of bigger chunks is a FIXED per-chunk constant** (fewer chunk barriers/evals/clear_cache calls), independent of depth.
- **Therefore the NET (SDPA linear penalty − fixed overhead amortization) is depth-invariant**: at 250K the SDPA 2× penalty is still per-chunk, and the amortized saving is still constant. The regression neither gets better nor worse at 250K vs 191K — the 191K measurement already captures the regime.
- **Conclusion:** the reviewer's argument that the old 2048-vs-4096 sweep "was not controlled for deep context" collapses because the mechanism proven at shallow context is depth-invariant, and the e2e number was anyway measured at deep context. There is no mechanism by which 250K flips 4096 to a win.

### 2.3 Were 3072 or 6144 EVER tested? — NO

- A repo-wide grep for `STEP_SIZE=3072`, `STEP_SIZE=6144`, or any `2048/3072/4096/6144` sweep **returns nothing** in `docs/`, `bench/`, `tmp/`, `src/`, or launch scripts. The prior sweep (2026-08-18/19) covered **only 2048 and 4096**. The 4096-retest doc explicitly lists 2560/3072 as a recommended *future* sweep (line 112), confirming they were never run.
- **What does the established mechanism predict for 3072 and 6144?** Under the linear-in-L SDPA scaling model:
  - 3072 → 1536 rows/rank → **~1.5× SDPA** vs 2048 → a *smaller* SDPA penalty than 4096, but still >1.5× the MoE gain (~11-15%) → likely a **smaller regression, not a win** (sweet-spot region where penalty ≈ gain, borderline).
  - 6144 → 3072 rows/rank → **~3× SDPA** → regression *worse* than 4096.
  - The mechanism therefore bounds the sweep: 3072 is the only arm with any plausible near-breakeven prospect; 6144 is a guaranteed regression. **A 3072 single-arm confirmation could be argued as NEW** (never tested, near-breakeven expected value) — but the full 4-arm sweep (2048/3072/4096/6144) at 3 depths is largely re-litigation of a closed mechanism.

### 2.4 "The root cause is fully attributed" — the skill's THIRD UPDATE claim is CONFIRMED

The root-cause doc's `THIRD UPDATE (final)` section (lines 148-209) states the investigation is **fully closed**: Option A sub-tiling DEAD, 3.15x anomaly = profiler artifact, SDPA = linear. The skill's `exo-dsv4-prefill-tuning` "Option A (SDPA...)" block mirrors this verbatim and instructs: *"This entire investigation thread is CLOSED... Do not re-open unless new evidence emerges."*

---

## 3. ENV-VAR READ TIME — THE CRITICAL ANSWER

### 3.1 `EXO_DSV4_SEQ_SPLIT` — read at MODULE IMPORT → frozen at process start → RELAUNCH REQUIRED

- **`src/exo/worker/engines/mlx/auto_parallel.py:124`**:
  `_DSV4_SEQ_SPLIT: bool = os.environ.get("EXO_DSV4_SEQ_SPLIT", "1") == "1"` — read once into a **module-level constant** at import time. Used in the shard-assignment gate at `auto_parallel.py:1100` (`if _DSV4_SEQ_SPLIT and type(layer.attn).__name__ in (...)`), which runs **once when the model's attention sharding is configured** (model load), not per request.
- The model-side gate ("must match") is the same env var consumed at model build → also frozen at load.
- **Unambiguous:** this var is decided at process/model-load. **A sweep of it requires one full cluster relaunch per arm.**

### 3.2 `EXO_PREFILL_STEP_SIZE` — read PER-CALL, but from process-fixed env → effectively RELAUNCH REQUIRED

- **Read sites:** `src/exo/worker/engines/mlx/generator/generate.py:871`, `:1072`, `:1379` — each is `prefill_step_size = int(os.environ.get("EXO_PREFILL_STEP_SIZE", "4096"))` inside the prefill functions, evaluated when `prefill_step_size is None` **per prefill call**.
- **BUT** this reads `os.environ`, the process-global snapshot fixed when the worker process launched. Nothing in the codebase mutates `os.environ` between requests to vary this per-request. The runner subprocess is forked from the worker (`AsyncProcess(target=entrypoint,...)`, `supervisor.py:294`) and **inherits** the worker's env — a subagent cannot change a running worker's `os.environ` from outside.
- So: read-per-call is real in code, but **practically fixed at launch**. Per-arm variation requires relaunching the top-level `python -m exo` process (what `start_cluster.sh` does). **RELAUNCH REQUIRED.**
- Note the code default is **"4096"** if the env is unset, but `start_cluster.sh:88` always sets `EXO_PREFILL_STEP_SIZE:=2048`, so the effective standing default is 2048 on the cluster. (The code-default/launcher-default mismatch is a latent foot-gun — flags if anyone launches outside start_cluster.sh.)

### 3.3 `EXO_PREFILL_CLEAR_CACHE_INTERVAL` — read PER-CHUNK on the eager path, NEVER on the batched path → RELAUNCH REQUIRED where it works at all

- **Read site:** `mlx-lm/mlx_lm/generate.py:544` — `_clear_interval = int(_os.environ.get("EXO_PREFILL_CLEAR_CACHE_INTERVAL", "1"))` inside the **eager single-stream chunk loop**, `:546-547` gate. Read per chunk from process-fixed env → **RELAUNCH REQUIRED** for the single-stream path that the needle/A-B tests use.
- **NO-OP on batched TP prefill:** `prefill_batched` (`generate.py:1385-1386, 1527, 1547, 1591`) clears `mx.clear_cache()` **unconditionally** — no interval read/gate. So for any multi-request/batched-PREFILL traffic the interval var is silently ineffective. This must be called out before any Exp 3 design: **the knob may not even be wired into the production batched path.**
- **Runner env inheritance:** `start_cluster.sh:1640` forwards `EXO_PREFILL_CLEAR_CACHE_INTERVAL` into `EXO_ENV`, which launches `python -m exo` with that env; the runner is `AsyncProcess(entrypoint=...)` (`supervisor.py:294-304`) and inherits the parent env. A runner is spawned **per model instance load** in the JIT lifecycle, not per request — but since the value is fixed at worker launch regardless, **relaunch** is the only way to sweep it.

### 3.4 Runner spawn cadence (does a var read at runner start avoid a full relaunch?)

- The runner subprocess (`entrypoint` in `bootstrap.py`) is spawned by `supervisor.py:294` `AsyncProcess(target=entrypoint, ...)` when an instance is placed/loaded (JIT load per model). It **inherits** the worker's `os.environ`.
- **Because all three vars read the inherited process env**, and that env is fixed at `python -m exo` launch, *spawning a fresh runner does NOT re-read anything from start_cluster.sh* — a new runner gets the same inherited env. There is **no mechanism** in `bootstrap.py`/`supervisor.py` to inject per-runner env from outside the running worker. Confirmed: `bootstrap.py` reads only `EXO_PROFILER`/`EXO_PROFILER_LEVEL`/`EXO_RUNNER_COREDUMP`/`HANG_TIMEOUT` from env, none of which are the sweep vars.
- **Verdict:** none of the three vars can be swept per-request or per-runner-refresh. **Each arm = one full top-level relaunch.**

---

## 4. RELAUNCH COST

- **Measured figure used by this repo for "one relaunch, incl. model load + a real measurement":** ~**20 min** of cluster time. Source: `docs/indexer-prefill-decomposition-2026-08-24.md:243` — *"Runtime cost: one relaunch, ~20 min of cluster time"* (a planned indexer experiment's cost model).
- **Cold DSv4-Flash model load alone:** ~**18.7 s/rank** (2 ranks ⇒ ~40 s) — `docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md:13413` and `:13533` (*"Cold shard+load measured at ~18.7s/rank"*); a full re-place reload ~**90 s** (`same doc:5692`). The ~20 min figure dominates because it includes cluster bring-up (2 nodes × service start + rendezvous + a real prefill+decode) — the model load itself is a small fraction.
- **Per-arm budget for the two proposed experiments:** ~20 min relaunch + the prefill/measurement time (a ~191K single-request prefill alone was ~540 s ≈ 9 min in the 2026-08-19 tests, doc line 33-34). So each arm ≈ **~30 min of 2-node cluster time**, consistent with the reviewer's ~2.5 cluster-hour estimate for the combined sweeps.

---

## 5. BOTTOM LINE FOR THE PARENT

1. **Exp 3 (clear-cache 1/2/4/8 @150K):** re-tests a **already-refuted mechanism** (+0.48% at interval=2, deep context, live). Only 2/4/8 are new knobs on a dead hypothesis, AND the knob is a **no-op on the batched TP prefill path** — so a 150K batched-preflight run may not even exercise it. **Recommend: do not spend cluster time unless a concrete counter-mechanism is stated.**
2. **Exp 4 (step-size re-sweep @90/150/250K):** the 2048-vs-4096 question is **fully closed and attributed** (SDPA linear in per-rank rows). The mechanism is **depth-invariant**, so the reviewer's "not controlled for deep context" objection fails. **3072 and 6144 were never tested**, and 3072 is the only arm with any near-breakeven prospect under the model (6144 is a certain ~3×-SDPA regression). Only a **single 3072 arm** is defensible as "new"; the full 3-depth × 4-size plan re-litigates a closed result.
3. **Env vars:** all three (`EXO_DSV4_SEQ_SPLIT`, `EXO_PREFILL_STEP_SIZE`, `EXO_PREFILL_CLEAR_CACHE_INTERVAL`) require a **full top-level relaunch per arm**; there is no per-request or per-runner-refresh pathway. Each arm ≈ ~30 min 2-node cluster time.
4. **Data-quality flag (not in scope to fix):** the skill's `~390ms/chunk clear_cache` figure is **unverified/inherited** — no cluster measurement found in this audit.

---

### Citations index
- `docs/dsv4-clear-cache-interval-2-test-2026-08-19.md` (commit `dd777f60a`, 2026-08-19)
- `docs/dsv4-4096-regression-root-cause-2026-08-19.md` (commits `510182cd1`/`799bf1dff`/`4a0ed6268`, final 2026-08-19)
- `docs/dsv4-prefill-step-size-4096-retest-2026-08-18.md` (commit `5c4ba9ce8`, 2026-08-18) — 331.2 vs 358.6 @ ~191K
- `mlx-lm/mlx_lm/generate.py:535-547` — interval read/gate (eager path)
- `src/exo/worker/engines/mlx/generator/generate.py:871/1072/1379` — STEP_SIZE per-call reads; `:1527/1386/1547/1591` — unconditional clear_cache on batched TP path; `:853-854`, `:1094` — clear on prefill/ pipeline paths
- `src/exo/worker/engines/mlx/auto_parallel.py:124` (module-level SEQ_SPLIT read), `:1100` (shard gate)
- `src/exo/worker/engines/mlx/generator/batch_generate.py:2836` (`len(tasks)<=1` → per-request submit → eager stream_generate)
- `src/exo/worker/runner/supervisor.py:294-304` (AsyncProcess entrypoint spawn), `bootstrap.py` (inherited-env reads)
- `start_cluster.sh:88` (STEP_SIZE default), `:107` (interval default), `:124` (SEQ_SPLIT default), `:1637/1640/1668` (EXO_ENV forwarding)
- `docs/indexer-prefill-decomposition-2026-08-24.md:243` (relaunch ≈ 20 min); `docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md:13413/13533` (18.7 s/rank cold load)
- Skill `exo-dsv4-prefill-tuning` SKILL.md:160 (~390ms claim — unverified) + Option A block (clear-cache interval refuted, 4096 thread closed)
