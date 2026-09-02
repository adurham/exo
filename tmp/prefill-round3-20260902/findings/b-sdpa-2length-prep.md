# Round 3 — SDPA Two-Length Per-Call Timing (PREP ONLY)

**Status:** PREP. Zero cluster runs, zero src/ edits, zero commits in this round.
**Repo HEAD (verified every file:line against this commit):** `17d427b01`
**mlx-lm submodule HEAD:** `37260bbd` (known-good-poolgrow-default-20260823-10-g37260bb)

## 0. What this round decides

Round 2 reopened the SDPA closure on a **per-call vs per-token units conflation**:
the cluster's 2.029x was `ms/token`, while the isolated microbenchmark's 2.0x was
per-call. Because doubling `EXO_PREFILL_STEP_SIZE` halves the call count while
doubling tokens/call, `ms/token ratio = 0.5 × per-call ratio`, so the cluster's
**measured** per-call ratio was **~4.06x** for a 2x per-rank row doubling, not
the 2.0x "linearity" the closure claimed (full reopening argument in
`tmp/prefill-round2/findings/sdpa-reopen.md`; PM report section 3).

This round asks one question: **is that ~4.06x per-call a real multiplicative
constant (matters at deep context / 250K) or a fixed per-call overhead that
amortizes away at longer context (does not matter)?** The instrument below times
the SDPA call directly at exactly two context lengths — 12K and 64K — because
64K is far enough for O(P)-scaling (real constant) vs O(1)-per-call (fixed
overhead) to diverge. **250K is NOT run; a sweep is NOT built.** Design fixed by
PM; not redesigned here.

### Pre-registered decision rule (restated verbatim, PM-fixed, do not alter)

Let `R(L) = per-call@2048rows / per-call@1024rows` at context length L (definition
below).

- **REAL MULTIPLICATIVE CONSTANT** iff `R(12K) ∈ [3.0, 5.0]` **and**
  `R(64K) ∈ [3.0, 5.0]` **and** `|R(64K)-R(12K)| / R(12K) ≤ 25%`.
- **FIXED-OVERHEAD ARTIFACT** iff `R(64K) ≤ 2.2` **or** `R(64K) < 0.6 × R(12K)`.
- else **INDETERMINATE**.

## 1. The SDPA call site(s) — exact file:line at HEAD `17d427b01`

Both `attn.sdpa`-tagged spans live in
`mlx-lm/mlx_lm/models/deepseek_v4.py` (note: **not** `src/`; the model is in the
vendored mlx-lm submodule). Distinguishing the two:

### 1a. `attn.sdpa.compressed` — NOT instrumented for the ratio (at ceiling)
`CompressedAttention.__call__`, `deepseek_v4.py:4478` — spans a single fused
`mx.fast.scaled_dot_product_attention` over `[RotatingKVCache local | pooled]`.
This is **Apple's fused fast-SDPA kernel**, already measured at ~79.1% of ceiling
(`docs/dsv4-attention-kernel-efficiency-2026-08-18.md`); the round-2 reopening did
**not** find the anomaly here, and it is off the per-call anomaly's path. It is
**excluded** from the ratio. (It IS still tagged `compressed` in our probe for
context, but the decision uses only local vs sparse.)

### 1b. `attn.sdpa` — THE SITE THE RATIO MEASURES
`SparseCompressedAttention.__call__`, span at `deepseek_v4.py:4865`. Inside this
ONE span (a single per-chunk SDPA region) three branches fire, all through the
same `_sparse_pooled_attention` / `scaled_dot_product_attention` code:

- `local` branch (`pooled.shape[1]==0`, `:4867-4876`): `LocalAttention` layers
  (compress_ratio=0). **NO seq-split** — runs **full-L query rows, L_q=2048** at
  `EXO_PREFILL_STEP_SIZE=2048`.
- `compressed`-inside-sparse branch (`pooled≤index_topk`, `:4879-4890`).
- `sparse` branch (`:4893-5017`): the 21 `SparseCompressedAttention` layers, via
  `_sparse_pooled_attention` -> `_sparse_pooled_attention_inner`
  (`deepseek_v4.py:1481`, `@mx.compile(shapeless=True)`). Under `EXO_DSV4_SEQ_SPLIT=1`
  (`deepseek_v4.py:4748-4753` band slice) these run **banded per-rank rows,
  L_q=1024** at STEP_SIZE=2048 on a 2-rank cluster.

**Rationale for instrumenting these two:** a single 2048-token prefill chunk
fires **21 sparse (1024 rows) + 2 local (2048 rows)** SDPA calls at **identical
depth** with **identical per-rank KV/pool**, so
`R(L) = mean(local_ms) / mean(sparse_ms)` is the same **1024→2048 per-rank row
doubling** the round-2 ~4.06x came from — measured **without varying
`EXO_PREFILL_STEP_SIZE` and without a sweep**. At 64K the pooled KV is ~5× deeper,
so a fixed per-call overhead becomes a smaller fraction of each call → R falls
toward 2–3 (overhead); a real multiplicative constant keeps R ≈ 4 (see §0 bands).

The tag selection logic in the probe distinguishes `local` vs `compressed` vs
`sparse` inside the span (same branch test as the code).

## 2. The instrumentation (env-gated patch) — how lazy-eval is handled

Delivered as `artifacts/sdpa_2length_timing.patch` (NOT applied). It adds:

1. **Module gate** `EXO_DSV4_SDPA_CALL_PROFILE` (env-gated, **no-op when off** —
   one module-level bool read per call site, following the established
   `EXO_DSV4_MTP_PROFILE` / `EXO_PROFILER` idiom). A helper
   `_sdpacall_mark_start()` returns `perf_counter()` when on, else `0.0`.
2. **At each `attn.sdpa` site:** `_sdp_t0 = _sdpacall_mark_start() if _SDPA_CALL_PROFILE else 0.0`
   at span entry, and after the SDPA output is produced:
   `_sdpacall_record(out, layer_idx, _sdp_t0, tag)`.
3. **The lazy-eval sync (MANDATORY):** `_sdpacall_record` calls **`mx.eval(out)`**
   on the SDPA output **immediately before** the end `perf_counter()` timestamp.
   This is the whole point of direct timing: MLX is lazy, so a bare
   `perf_counter` around the call measures only **ENQUEUE** (graph build), not
   execution. The existing `finalize(x)` / `span()` helpers in
   `mlx_lm/mlx_lm/profiler.py:109-113` are **NO-OPS when no profiler hook is
   registered**, so the probe does its **own** `mx.eval(out)` — it cannot be
   silently misattributed to graph-build time. The comment at the gate and in the
   patch states this explicitly.
4. Emits `[SDPA-CALL] <tag> L=<idx> ms=<float>` per call to runner stderr
   (→ `~/exo.log` `Runner stderr:` lines). No SIGUSR1, no autodump dependency.

**Patched sites in `deepseek_v4.py` (HEAD line numbers):**
- `SparseCompressedAttention` `attn.sdpa` span — start at `:4865`, record after
  `out = finalize(out)` at `:5018` (tag = local/compressed/sparse by branch).
- `LocalAttention` `attn.sdpa` span — start at `:4293`, record after
  `out = finalize(...)` at `:4304` (tag = local, full-L rows).

`artifacts/deepseek_v4.instrumented.py` is the fully-instrumented copy for
reference; the `.patch` reproduces it exactly (`git apply --check` clean against
HEAD, applied copy byte-identical to the instrumented file).

## 3. Runbook — gotchas, batched path, env, cluster time

`artifacts/sdpa_2length_run.sh` contains the exact copy-pasteable commands and
env for both arms. Key guarantees baked in:

- **GOTCHA — profiler sync vs watchdog.** If anything enables
  `EXO_PROFILER_SYNC_SPANS=1`, it MUST be paired with
  `EXO_RUNNER_HANG_TIMEOUT_SECONDS=600` (default 45 s watchdog in
  `src/exo/worker/runner/supervisor.py` SIGKILLs the runner mid-run at 12K+
  because sync-mode serialization blows the progress-callback gap). This round
  does NOT need sync-spans (our probe does its own `mx.eval(out)`), but the
  runbook still exports `EXO_RUNNER_HANG_TIMEOUT_SECONDS=600` defensively and
  documents the pairing.
- **GOTCHA — batched path only.** Must measure `prefill_batched`
  (`src/exo/worker/engines/mlx/generator/generate.py:1269`, called at
  `batch_generate.py:3068`), never the eager `stream_generate` fallback. The
  runbook rejects any run whose log lacks `Starting batched prefill:`
  (`generate.py:1401`) / `[MEM] batched prefill chunk` (`generate.py:1538`).
- **GOTCHA — no SIGUSR1.** A mistimed SIGUSR1 crashed a cluster rank in a prior
  session. Let each request finish; read the auto-emitted `[SDPA-CALL]` lines from
  `~/exo.log`. The patch and runbook say "no SIGUSR1" outright.
- **GOTCHA — chunk halving.** `generate.py:497` `prefill_step_size // min(4, group.size())`
  is on the **PP** loop only (`_pipeline_parallel_prefill_steps`, `generate.py:443`).
  `prefill_batched` does **not** halve — it uses the full `prefill_step_size`
  (`generate.py:1454`, `n_to_process = min(prefill_step_size, max_length-offset)`),
  so at STEP_SIZE=2048 the batched path already yields the intended L_q=2048
  full-step chunks; under seq-split the sparse band is L_q=1024 per rank. Know
  which path you're on before quoting a row count: **this measurement is TP
  (`prefill_batched`), not PP**, so the halving gotcha does not apply here — it's
  noted so a future reader doesn't mis-attribute 1024 chunks to a bug.
- **Chunk count sanity:** 12K prompt → 6 chunks (5×2048 + remainder); 64K → 31
  chunks (31×2048 + remainder). n≥5 SDPA calls per (length, tag) is trivially met
  (135+ sparse / 12 local at 12K; 660+ / 62 at 64K per rank).
- **Env-gate must reach the runner.** `EXO_DSV4_SDPA_CALL_PROFILE` is new; the
  runbook flags that `start_cluster.sh` must allow-list it (add
  `[ -n "${EXO_DSV4_SDPA_CALL_PROFILE:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_SDPA_CALL_PROFILE=$EXO_DSV4_SDPA_CALL_PROFILE"`),
  otherwise the runner strips it and the probe no-ops (verified the allow-list
  pattern at `start_cluster.sh:1660-1661`, `:2018`).

### Estimated cluster time (both arms)
12K ≈ 30–40 s clean prefill (+probe serialization small); ~5 min incl. relaunch and
the rendezvous + harness overhead per the round-1 measurements. 64K ≈ 90–170 s
clean prefill; with relaunch ~8–10 min. **Total ≈ 15–20 min of cluster time, one
operator, single session** — two launches (one per arm) at `EXO_PREFILL_STEP_SIZE=2048`.
No sweep, no 250K, no extra arms.

## 4. Analysis — per-call absolute ms, warmup, reductio

`artifacts/sdpa_2length_analyze.py`:
- **Parses** `[SDPA-CALL]` lines, keeps global stream order.
- **Splits arms** at `--split-calls` (12K arm call count = 2 warmup +
  6×23 + 9 remainder = **149**; 64K = 2 + 31×23 + 9 = **724**; the analyzer takes
  the explicit index so the operator can adjust).
- **Buckets warmup**: the FIRST call per (tag, arm) is separated (Metal kernel
  compile + allocator warmup) and reported separately, excluded from the n≥5
  steady-state sample (this is the "FIRST call at each length" stratification the
  PM required).
- **Reports per-call ABSOLUTE ms** (mean/median/min/max, n per tag). ms/token
  appears ONLY as an explicitly-labeled derived secondary (µs/token rows), never
  the primary metric.
- **Reductio (fails loudly):** `sum over tags(calls × mean_ms)` must be ≤ the
  arm's wall clock (`--prefill-wall-sec`), else the per-call numbers are
  mathematically impossible and the arm is discarded. Built into the script so
  no per-call number can be quoted without passing it.
- **Decision rule** (§0) applied verbatim, printing the REACHED verdict.

Validated end-to-end on synthetic logs: CASE A (R12≈3.92, R64≈3.89) → **REAL
MULTIPLICATIVE CONSTANT**; CASE B (R64≈2.05) → **FIXED-OVERHEAD ARTIFACT**;
over-budget wall → **reductio fires and arm discarded**.

## 5. What each outcome DECIDES (plain statement)

- **REAL MULTIPLICATIVE CONSTANT (~4x per-call, holds at BOTH 12K and 64K):** the
  superlinear per-call SDPA cost is intrinsic and does NOT amortize with context.
  It matters at 250K and is the mechanism behind any deep-context SDPA blowup —
  next action is a real kernel-level investigation (the round-2 "mechanism
  unknown" is confirmed open and worth chasing), NOT a config change.
- **FIXED-OVERHEAD ARTIFACT (R collapses at 64K):** the ~4.06x was a fixed
  per-call overhead (allocator, Metal occupancy, dispatch) that amortizes away;
  it does **not** matter at 250K. The original "SDPA scales linearly for
  deep-context purposes" closure is then *reasoning-wrong but conclusion-right*
  for the deep-context question, and the depth degradation stays attributed to
  the designed-in O(P) terms (~86% explained) — no action.
- **INDETERMINATE:** neither band cleanly met (noisy, wrong path taken, watchdog
  trip, or reductio reject). Re-run once; if still indeterminate, the two-length
  design did not separate the hypotheses under this cluster's variance and a
  longer-contrast design (but still not a sweep) must be proposed to the PM.

## 6. Deliverables (all under `tmp/prefill-round3-20260902/`)
- `artifacts/sdpa_2length_timing.patch` — env-gated instrumentation (NOT applied;
  `git apply --check` clean against HEAD, byte-identical to the instrumented ref).
- `artifacts/deepseek_v4.instrumented.py` — the instrumented file for review.
- `artifacts/sdpa_2length_run.sh` — exact runbook, both arms.
- `artifacts/sdpa_2length_analyze.py` — parse / warmup / per-call / reductio /
  decision.
- `findings/b-sdpa-2length-prep.md` — this doc.

## 7. Constraints honored
- **No cluster contact** (no ssh, no benchmarks, no inference requests).
- **No src/ edits, no commits, no pushes.** `git status` on `src/` and
  `mlx-lm/mlx_lm/models/deepseek_v4.py` is clean at HEAD `17d427b01`; the only
  tracked working-tree modifications are pre-existing `tmp/p12-relaunch-*.log`
  files from an earlier session.
- Instrumentation delivered as a patch + reference copy; nothing applied.
