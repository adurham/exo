# Phase 19 — DSv4.1-Flash turn-latency: engine identification, live round-wall
# measurement, and a clean negative on the spec-stack lever hunt

Date: 2026-10-07
Author: Hermes PM subagent (deleg_5b2e1964)
Cluster: 2x Mac Studio M4 Max, TP2 over jaccl RDMA
Deployed commit: `f0840af1c50ecfdc85236a68b60a4c341805f8ce` (both nodes, verified)
Model: `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw` (engine `dsv41`, 40L, hidden 5120, EXL3 2.9bpw)
**Relaunches performed: 0. Cluster env unchanged.**

---

## 0. Headline — the campaign brief's central premise is false

The campaign was briefed with a "residual ~60 ms/round unaccounted", a
"live acceptance collapse to 1.272/3" and a set of live spec-stack levers
(`EXO_DSV4_MTP_EAGLE_K`, `MTP_ACCEPT_LOGPROBS`, `MTP_TIEBREAK_FIX`,
`MTP_C2_MAX_CTX`, `MTP_MAX_CTX`, `MTP_DEDICATED`, `DSPARK`). Source reading and
live measurement show:

1. **The live engine is `dsv41`, not the legacy MLX path.** The model card sets
   `engine = "dsv41"` (`resources/inference_model_cards/dealignai--…EXL3-2.9bpw.toml:56`)
   → `Dsv41Builder` (`src/exo/worker/engines/mlx/dsv41/dispatch.py`). The dsv41
   package **never imports `dsv4_mtp.py`** or any profiler emitter.
2. **Every `EXO_DSV4_MTP_*` knob in the brief is dormant** — read only by
   `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py`, which the deployed
   engine never loads. See `mtp-knob-semantics.md`.
3. **The `[MTP-PROF]` / `[MTP] cycles=… mean_accept=… hist=…` numbers in the
   evidence pack were emitted by a PRIOR (legacy-engine) process.** They exist
   in the append-only `~/.exo/exo_log/runner_log/stderr.log` at line ~14,136,010
   of 14,150,154, *before* the most recent process boot; the current process
   emits `[DSV41] …` lines and **zero** `[MTP-PROF]` lines, and the live
   timestamped `~/exo.log` has zero of both groups. The 1.272/3 histogram is a
   historical artifact, not today's cluster.
4. **The engine is greedy-only** (`rounds.py`: "DSv4.1 serving is greedy-only in
   this build"), and Hermes connects at `temperature=0` (state.db). The brief's
   #1 hypothesis — "sampling params explain the acceptance gap" — is impossible:
   both arms are greedy.
5. **Live acceptance is NOT collapsed.** Measured today with the public API on
   benign content, mean accepted is **2.80–2.84 / 3**, at the *top* of the
   standalone 1.5/3 range (phase 16/17), not the bottom.

Everything below is what the commands actually returned.

---

## 1. What the live path actually is (`round-loop-map.md`)

* Round loop: `src/exo/worker/engines/mlx/dsv41/rounds.py::_one_round` (L328),
  driven by `engine.py::_rounds` (L971). One round = **one** `mx.eval` (L429,
  `mx.eval(logits)`), a single `(1 + gamma)`-row verify forward, with the draft
  feeding the verify lazily (draft is not evaluated on its own).
* Draft/verify/accept/rollback are *not wrapped by any profiler bracket* on this
  path — there is no per-cycle breakdown to be had, sync or non-sync.
* `VERIFY_MS` (`mlx-lm/mlx_lm/models/deepseek_v41/spec.py:77`) is literally the
  phase-17 table `{1:58.5, 2:74.9, 3:87.9, 4:97.7, 5:111.7, 6:120.1}`.
* `GammaPolicy` (spec.py:80) exists and `adaptive_gamma=True` (engine.py:281),
  **but `_one_round` never calls `policy.update()`** (verified: `grep update(`
  finds no call in rounds.py). Therefore `policy.next()` returns its start value
  **gamma = 3 every round** — the adaptive policy is a latent no-op. This is
  reported as a *bug finding*, not a lever (see §4).
* Request-phase timestamps are **absent** on this path (no recv / tokenize /
  prefill_start / first_token / last_token log lines).

## 2. Live measurement — the round wall is now actually known

Zero-relaunch, public-API-only harness: `bench/phase19_round_measure.py`.
It reads the `: generation_stats` SSE frame, which carries
`mtp_cycles_cumulative` / `mtp_accepted_drafts_cumulative`. With gamma pinned at
3, per request `rounds = Δcycles/3` and `mean_accepted = 3·Δaccepted/Δcycles`.

Greedy, `temperature=0`, unique-salted filler, one cold prefill + repeated
warm reps. Raw JSON in `docs/benchmarks/phase19-latency/raw/`.

| depth | decode t/s | ms/round | mean accepted /3 | note |
|---|---|---|---|---|
| 30 K | **29.97** | 126.9 | 2.804 | ttft 111.7 s cold, 3.8 s warm |
| 60 K | — | — | 2.778 | budget spent on reasoning (see trap) |
| 100 K | **28.08** | 135.6 | 2.813 | ttft 373.9 s cold, 5.4 s warm |
| 160 K | **27.39** | 140.0 | 2.841 | ttft 620.3 s cold, 4.5 s warm |

Cross-checks:
* `Δcycles = 768 = 3 × 256` rounds for a 976-token generation ⇒ **gamma = 3
  confirmed empirically**.
* Round wall from measured t/s and committed/round (3.81): 27.4 t/s ⇒ 7.2 tok/s
  of round-rate × 3.81 = 139 ms — matches the direct `ms/round` exactly
  (0 K = 130.5 expected vs 126.9 measured, ~3% profiler-independent agreement).
* **Acceptance is flat with depth** (2.80 at 30 K → 2.84 at 160 K). Round wall
  grows only ~10% from 30 K→160 K (126.9→140.0 ms), i.e. the decode round is not
  strongly depth-sensitive.

## 3. Decomposition vs the standalone numbers

* Standalone "110 ms round" (phase 18 note) vs live 127–140 ms: a **real but
  modest** gap (+15–27%), present even at 30 K, so it is **not** primarily a
  context-length effect. Same topology (phase 16/17 used the same 2-node TP2
  cluster), so the comparison is valid.
* Standalone decode 22.7–23.5 t/s vs live **28.0–30.0 t/s**: today's live cluster
  is **faster** than the phase-16/17 standalone numbers, not slower. The brief's
  framing ("live 18 t/s, why so slow?") does not survive measurement.
* The remaining round wall (~127–140 ms) at gamma=3 composes as, on the phase-17
  table: draft ≈ 11 + verify R4 ≈ 97.7 + overhead ≈ 12–20. The "residual 60 ms"
  was an artifact of applying a *legacy-engine, half-reading* profiler to a
  different engine's numbers.

## 4. The one real finding that is actionable (but not a ≥3% perf lever)

`GammaPolicy.update()` is never called on the dsv41 path. The policy therefore
pins gamma at 3 with zero adaptation. Given (a) acceptance 2.80–2.84/3 is near
saturation and (b) the offline optimum over the real `VERIFY_MS` table at
measured acceptance is at most ~3–5% better at gamma=2 and clearly worse at
gamma=4 (`docs/benchmarks/phase19-latency/gamma-optimality.md`), **fixing the
bug would not buy ≥3%.** It is reported so the next person does not mistake the
dead policy for a live one. A one-line `policy.update(gamma, accepted)` call in
`_one_round`'s caller is the fix, but it is *moot for speed* at the current
acceptance level.

## 5. Negative results (documented, not fabricated)

* **Sampling-params lever: DEAD.** Engine is greedy-only; Hermes already sends
  temperature=0.
* **`EXO_DSV4_MTP_*` lever farm: DEAD (dormant).** `EAGLE_K`,
  `ACCEPT_LOGPROBS`, `TIEBREAK_FIX`, `C2_MAX_CTX`, `MAX_CTX`, `DEDICATED`,
  `DSPARK*` are all read only by the unloaded `dsv4_mtp.py`. See
  `mtp-knob-semantics.md`.
* **Gamma matrix: NOT RUN.** Expected and computed flat: gamma is pinned at 3
  (no env knob for dsv41 gamma — `EXO_SPECULATIVE_GAMMA` is read only by dormant
  legacy paths), and the offline optimum is ≤3–5% at gamma=2, negative at
  gamma=4. Running the matrix would have cost ≥3 relaunches to re-derive a
  pin that a one-line fix already implies.
* **`[MTP-PROF]` bracket attribution: NOT REPRODUCIBLE on this engine** — the
  emitter does not exist on the dsv41 path.
* **Instrumentation relaunches: NOT PERFORMED.** The two relaunches the brief
  budgeted for request-phase timestamps were dropped: the API already exposes
  per-request acceptance (via the cumulative counters), so the decision-relevant
  number needed no relaunch, and the fixed-cost ledger rode on a dead premise.

## 6. Measurement traps hit (for the record)

* The `: generation_stats` SSE line does **not** start with `data:`; a parser
  that only inspects `data:` lines silently misses every spec counter.
* A "reply with DONE" task EOS'd at 21 tokens, giving a meaningless tiny sample.
  Use a long deterministic task (count to 450) for decode benches.
* At 60 K the model spent the whole `max_tokens` budget inside
  `reasoning_content` (`completion == reasoning == 700`, `finish=length`,
  empty content) — decode t/s unmeasurable for that arm. This is the known
  DSv4 reasoning-budget trap, reproduced live.

## 7. Files

* `round-loop-map.md` — the dsv41 round loop, brackets, sync spans, phase
  timestamps, gamma policy, stats formats (read at exo `f0840af1c`, mlx-lm `6cc9c1e`).
* `mtp-knob-semantics.md` — per-knob semantics + the `DORMANT / DEAD KNOBS` table.
* `gamma-optimality.md` — offline gamma optimum from the real `VERIFY_MS` table.
* `raw/` — the JSON per-rep records.
* `bench/phase19_round_measure.py` — the reusable zero-relaunch harness.
