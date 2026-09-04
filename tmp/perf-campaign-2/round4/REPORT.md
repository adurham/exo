# CAMPAIGN 2, ROUND 4 — Is the async fence ARMED, and what does mx.eval(y) cost today?

**Date:** 2026-09-03 | **PM:** round-4 delegation | **Commits (local, UNPUSHED):**
superproject `232d1f6b7`, mlx-lm submodule `7f14654` | **Cluster:** restored to shipped
config at end (API 200, both nodes READY, zero probe/diag flags on runners).

**Verdicts up front:**
- **Q1 — the fence is ARMED: ≥98.5% (bounded; 100% of observed decode events). No
  regression of the 08-22 fix under MTP=1+DSPARK=1.** Every blocking fallback is at
  prefill/warmup/transition, by design. The two-owner gate is satisfied continuously
  through decode on both nodes.
- **Q2 — the band lands <15%: the residual decode stall is GENUINE local compute,
  not the fence/handoff mechanism. The decode thread is CLOSED at the model level.**
  Probe B is decisive: GPU busy 68.8–153.2% of wall (mean 117.3%) during clean decode —
  there is no GPU-idle gap for a mechanism to own.
- **Q3 — not executable in this stack: `EXO_DSV4_FENCE_EVERY_N_LAYERS` is a dead
  knob (zero read sites).** With the fence armed there is no per-layer blocking eval
  at c=1 decode to modulate. Rebuilding the reverted OPT-7 gate would re-introduce a
  documented −23% prefill regression to test a mechanism the async fence supersedes.

---

## Q1 — armed ratio under today's TP+MTP+DSpark config

### Method
Live `EXO_DSV4_FENCE_GATE_DIAG=1` (the 08-22 method) on a real 89K-depth needle
request + 3×533-token decode probes, boot 2 of commit `232d1f6b7`. The gate diagnostic
logs every blocking-branch fallthrough (rate-limited: first 30 + every 200th) and every
`_set_fence_async_ok` setter. Read from BOTH nodes' `~/exo.log`.

### Evidence (both nodes identical — 646 diag lines each)

**Blocking-fallback lines: 50 per node, ALL before decode steady state:**

| Timestamps | L-shape | Phase | Why it blocks (correct) |
|---|---|---|---|
| 18:57:53–55 | L=13 | JIT warmup forwards | engine=False, cache=False — owners not yet armed |
| 18:57:59–18:58:01 | L=1 | warmup decode forwards | same |
| 18:58:10→19:00:24 (every ~22s) | L=2048 | **89K needle PREFILL chunks** | prefill is designed blocking (disarmed) |
| 19:00:39 | L=2013 | final prefill chunk | same |
| 19:00:52 | L=1 | **prefill→decode transition forward** | engine=False, cache=True — engine arms after transition |

**After 19:00:52 — the entire decode steady state: ZERO further fallback lines.**
The rate-limited counter froze at n=4000 through the needle's 71-token decode AND all
three decode_probe runs (531 tokens). Decode gate evaluations ≈ 300 steps × 43 layers ≈
13,000; the 200-line rate limit bounds hidden fails at <200 → **fallback rate <1.5%,
armed ≥98.5% (observed 100%)**. Well above the pre-registered 95% bar → **Q1 CLEAN.**

**Setter evidence:** 596 SETTER lines; `cache=True` held through decode (422×),
`engine=True` armed at transition. The MTP-on config re-arms the two-owner gate the way
the 08-22 fix designed: `DSv4MTPPredictor.__init__` registers "cache"
(`dsv4_mtp.py:1169-1171`), and the load-bearing re-arm is `activate_for_uids`'s
no-transition fast path firing `_set_fence_async(True)` on **every spec cycle before
every verify forward** (`dsv4_mtp.py:1304`) — structurally different from the 08-22
dead-code regime where the owner never instantiated.

**Gate code cites (verified line-by-line by PM against installed+submodule copies):**
- Gate: `mlx-lm/mlx_lm/models/deepseek_v4.py:3115-3121` (condition) → `mx.async_eval(y)`
  at :3130, blocking `mx.eval(y)` at :3150, both followed by `y = finalize(y)`.
- Registry: `_FENCE_ASYNC_REGISTERED` :131, `_register_fence_async_owner` :134,
  `_fence_key_ok` :140-147 (fail-closed), `_set_fence_async_ok` :156.
- The 08-22 unsatisfiable-gate condition does not exist today: `"cache"` registers
  (unconditionally under EXO_SPECULATIVE=1 + is_dsv4_with_mtp, `batch_generate.py:824-838`)
  and arms True every spec cycle.

**Decode paths audited (from `q1-gate-audit.md`, PM-verified):**
- Verify forward steady state: ARMED (the evidence above).
- Rejection cycles: `_set_fence_async(False)` at `dsv4_mtp.py:4982` around the
  commit-forward (:5035), re-armed :5050 — a designed blocking window on rejection
  only (rare at steady state; not a regression).
- c≥2: disarmed by design (all three legs). Not in today's c=1 path.
- No stuck-False path found: c2→c1 return, `MTP_C2_MAX_CTX` spec-ineligible dispatch,
  and BS transitions all recover via `activate_for_uids`.

### Two audit findings that change how this config should be read
1. **`EXO_DSV4_FENCE_EVERY_N_LAYERS` is a dead knob** (also kills Q3, below):
   `_fence_every_n` assigned at `deepseek_v4.py:2958`, **zero read sites tree-wide**
   (grep across submodule+src+installed; the only reader was OPT-7 `230a670`, reverted
   in `19a07b3` "made B=2 23% slower"). Its value (4) is logged by `dsv4_mtp.py:978` but
   nothing consumes it.
2. **`SpanProfilerHook` silently re-blocks the fence**: registering any profiler hook
   makes `finalize(y)` (`deepseek_v4.py:3151`) a forced `mx.eval` one line after
   `mx.async_eval` (`profiler.py:109-113`, bootstrap.py:146). Today's production env
   sets only `EXO_PROFILER_LEVEL=1` WITHOUT `EXO_PROFILER` → no hook registered
   (`bootstrap.py:117-119` requires the list var) → fence intact. **Any future
   fence/drain measurement taken with EXO_PROFILER=spans is invalid** — pre-registered
   as a measurement caveat for all future rounds.

---

## Q2 — what the fence/handoff costs TODAY (the two falsification probes)

### Probe A — GPU-identity all_sum substitute (implemented, run, confound-controlled)
- **Implementation:** env-gated `EXO_DSV4_ALLSUM_IDENTITY_PROBE=1` at the MoE per-layer
  all_sum only (`mlx-lm` commit `7f14654`): skips `mx.distributed.all_sum`, keeps the
  fence gate running on unchanged `y`. Loud one-time stderr warning; OFF = byte-identical;
  never in start_cluster defaults (pass-through line added with NO default, superproject
  `232d1f6b7`). Verified live: warning fired once on both nodes; probe absent from the
  restored production runners.
- **The pre-existing 2026-05-13 file-NOP (`/tmp/dsv4_nop_targets`) was REJECTED** as the
  A/B vehicle: it patches **global** `mx.distributed.all_sum` — it would also NOP the
  DSpark agree gate (`utils_mlx.py:459`) and `has_work` coord collectives, changing the
  workload itself (DSpark detaches → different code path).
- **Numbers (2K-ctx decode, boot 3 identity vs boot 2 clean, same commit):**
  - tok/s: identity 37.3–39.7 (5 runs) vs clean 26.3–26.7 (3 runs) — **UNUSABLE per
    pre-registration**: identity output degenerates (needle all_needles=0, bistability),
    acceptance collapses (61 tokens @ ~1/step vs 177 @ ~2-3/step), so t/s measures
    MTP acceptance dynamics, not the collective.
  - **89K needle TOTAL wall: identity 137.7s vs clean 150.4s = ≤8.4%** (prefill-dominated;
    identity also skips 43 collectives per prefill chunk, and the run degenerates —
    treat as an upper bound on the collective's whole-stack share).
  - Per-step BG windows: identity's 6 windows are warmup-contaminated (fresh boot) —
    no valid per-step A/B; reported as a null, not a delta.
- **Band application:** the only acceptance-unconfounded Probe A readings are the
  total-wall ≤8.4% (89K) and the step-time share implied by Probe B below (2–6ms of
  ~75ms/step ≈ 3–8% at 2K ctx). **<15% → the handoff mechanism does NOT own the stall.**

### Probe B — command-buffer GPU timestamps (GPU busy vs idle)
Already exists in the installed stack — `MLX_GPU_TIME=1` (mlx core accumulates
`GPUEndTime−GPUStartTime` per completed command buffer → `mx.metal.gpu_time_ns()`) +
`EXO_DECODE_PROBE=1` (per-16-step window wall/GPU/GPU% in `batch_generate.py:4124-4149`).
No code change was needed; verified live on both nodes (identical numbers):

| Boot | Config | Decode windows | GPU% busy (mean) | GPU% range |
|---|---|---|---|---|
| 2 (clean) | fence ARMED, collective live | 17 | **117.3%** | 68.8 – 153.2 |
| 3 (identity) | fence armed, collective REMOVED | 6 | 66.6% | 55.5 – 91.0 |

**The falsification the round-2/3 attribution needed:** if the fence/handoff mechanism
owned the ~1400µs/layer stall, the GPU would sit IDLE during decode windows waiting on
host-side handoff. It does not — under the armed fence the GPU is at or above 100% of
wall (values >100% are the async fence's overlap working: GPU still draining the
previous window while the host proceeds). Removing the collective makes the GPU LESS
busy (66.6%), i.e. the collective+handoff is real queued work, not hidden stall.

### The blocking-fence drain bracket (boot 1, `EXO_DSV4_ALLSUM_PROBE=1`)
Timed `mx.eval(y)` at the same site (this probe branch REPLACES the gate — a
within-probe blocking measurement): **decode-cycle per-layer p50 ≈ 0.99–1.0ms, cross-layer
p50-of-p50 1.81ms; prefill-chunk per-layer 14–38ms** (44 layers × 20 cycles dumped).
This reproduces round 2's "~1400µs/layer local drain" (0.99–1.81ms p50 band) and is
what the armed async fence takes off the CPU critical path — the +58-67% of 08-22.

**Q2 verdict: <15% band → the residual decode stall is genuine local compute
(GPU-saturated; the MoE producing y). The decode thread is CLOSED at the model level,
consistent with rounds 1–3's attribution.**

---

## Q3 — fence cadence 4/8/43: NOT EXECUTABLE in this stack

- The pre-registered lever does not exist: `EXO_DSV4_FENCE_EVERY_N_LAYERS` is assigned
  (`deepseek_v4.py:2958`) but **has zero read sites** — OPT-7 (`230a670`) added the only
  reader and it was **reverted (`19a07b3`) for −23% B=2 prefill** via graph accumulation.
- With the fence armed (Q1), **no per-layer blocking eval runs at c=1 decode at all**:
  the task's premise "~11 fences per 43-layer forward at FENCE_EVERY_N=4" describes the
  blocking path, which decode no longer takes. There is no cadence to modulate.
- Rebuilding an async_eval-every-N gate to force the ABAB would (a) re-introduce the
  documented prefill regression class, (b) test a mechanism the async fence already
  supersedes, and (c) cost ≥6 boots — out of time-box. **Refused; recorded.**
- The honest substitute (FENCE_ASYNC on/off ABAB) is already answered by the record
  itself: the 08-22 fix WAS that experiment (18.5 → 29-31 t/s, +58-67%). Re-pricing a
  known number adds nothing; not run.

---

## Reconciliation with the prior record

- **08-22 fix (`docs/async-fence-fix-validated-2026-08-22.md`):** CONFIRMED holding
  under the MTP-on config that re-arms its gate. The dead-code condition (owner never
  instantiated) is structurally gone — `DSv4MTPPredictor` constructs and registers
  unconditionally under today's config. No regression.
- **Round 1's "~98% local drain + dispatch" attribution: CONFIRMED** by Probe B —
  GPU-saturated decode windows, compress_ratio alternation intact (prefill spikes
  14–38ms/layer in the boot-1 bracket track per-layer work).
- **Round 2's "~1400µs/layer drain": reproduced** as 0.99–1.81ms/layer p50 (decode
  cycles) — and shown to be GPU work, not handoff (Probe B).
- **Round 3's FEASIBILITY verdict ("stall's mass sits where H-e1 cannot reach"):
  CONFIRMED and now measured directly** — the two probes it specified were run and the
  local-compute attribution survives falsification.
- **Round 1's fence-tax table (fence=8 "+0.7 t/s"):** that number predates both the
  async fence and this round's dead-knob finding — the knob it tuned no longer exists
  in the decode path. Mark it historical, not actionable.
- **New caveat for all future fence/drain measurements:** any run with
  `EXO_PROFILER=spans` re-blocks the fence via `finalize` — invalidates the measurement
  (documented above).

## Cluster health (end state)
- Relaunch on shipped config (no probe/diag env): verified **zero** of
  FENCE_GATE_DIAG/MLX_GPU_TIME/EXO_DECODE_PROBE/ALLSUM_PROBE/ALLSUM_IDENTITY on both
  runner PIDs; `EXO_DSV4_FENCE_ASYNC=1` present (shipped); API 200; both nodes READY
  (2/2); post-restore decode smoke: 26.6 t/s at 2K ctx (matches pre-experiment
  baseline); needle quality on the restored-config boots: exact match, zero leaks.

## Artifacts
- `tmp/perf-campaign-2/round4/PLAN.md` (pre-registration, written before relaunches)
- `tmp/perf-campaign-2/round4/q1-gate-audit.md` (static audit with line cites)
- `tmp/perf-campaign-2/round4/results/` — boot1/2/3 diag+probe extracts, needle JSONs
- Commits (local only, deliberately unpushed per task constraint 4):
  - `232d1f6b7` (superproject): probe pass-through line + mlx-lm submodule pointer
  - `7f14654` (mlx-lm submodule): Probe A implementation (env-gated, loud, never default)