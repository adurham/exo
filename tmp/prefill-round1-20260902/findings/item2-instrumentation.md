# item2-instrumentation — What prefill per-chunk overhead is measurable EXISTING-logs-only

**Date:** 2026-09-02 · **Repo HEAD:** `cb1f91903` (main) · **Mode:** read-only, zero cluster contact.
**Provenance discipline:** every number carries a `file:line` or artifact citation; where a quantity cannot
be determined read-only it is marked **REQUIRES CLUSTER RUN**. Nothing below was fabricated.

---

## VERDICT (3–5 bullets, each a hard number with citation or an explicit "not measurable")

1. **The 390 ms/chunk claim is UNSOURCED — it appears ONLY as a prose comment, never as a measurement,
   and contradicts already-recorded real data.** Its two appearances are both documentation comments
   authored in commit `c0d012d2` (Adam Durham, 2026-06-21, "Context-adaptive prefill chunk sizing"):
   `mlx-lm/mlx_lm/generate.py:438-440` (`"...per-chunk fixed overhead (43 layers x kernel launches x RDMA
   all_sum x eval x clear_cache ~390ms)"`) and `start_cluster.sh:89-91` (same wording, 2026-06-21).
   A repo-wide grep for `390ms` finds **zero** docs/, bench/, or tmp/ measurement; the number is a
   code-comment assertion, not a recorded measurement. Worse, the ONE real instrumented measurement
   of that exact bracket contradicts it: `docs/prefill-trace-instrumentation-findings-2026-08-21.md` §"Real
   trace-data findings" measured `prefill.clear_cache` + `prefill.barrier` + `prefill.mem_checkpoint` at
   **single-digit ms TOTAL (worst case ~35 ms) against 190–1057 s prefill walls — under 0.02% of wall** —
   i.e. the named "fixed overhead" bracket (including `mx.clear_cache()`) was measured to be ~1-2 orders of
   magnitude smaller than 390 ms/chunk at 100K-500K context. Report the 390 ms claim as unsourced and
   directly at odds with a recorded measurement.

2. **No prefill per-chunk span data exists in the 2026-09-01 depth-scan logs.** The `prof_*.txt` files are
   `[MTP-PROF]` dumps — a decode/speculative-decode cycle profiler. Every file emits exactly the phases
   `B=1 draft/verify/accept/rollback/rb_drain/rb_gate/rb_pool/rb_ring/rb_snap/rb_tail/total` (verified in
   `tmp/verify-decomposition-20260901/raw/prof_089k_n1.txt`; identical key set in all 6 files). A grep for
   any prefill token (`TRACE`, `prefill_batched`, `.forward`, `all_sum`, `clear_cache`, `kernel`) across all
   `prof_*.txt` returns **zero hits** (rc=1). The `[MTP-PROF]` profiler lives in the decode loop
   (`dsv4_mtp.py:790-850`, `_PhaseTimer.dump`) and never touches prefill. **Existing depth-scan logs contain
   zero prefill span data — do not manufacture an estimate from them.**

3. **Prefill wall CAN be measured from existing data, and gives per-chunk TOTAL (not fixed) times.**
   Using `bench_*.json` real prompt counts and measured prefill tps (all 5 scored iterations per depth,
   median), with real chunk size = FULL 2048 (TP does NOT halve — see §5):
   - **89.4K tokens → 210.2 s prefill wall → 4777 ms per-chunk TOTAL** (44 chunks)
   - **150.0K tokens → 358.4 s → 4843 ms per-chunk TOTAL** (74 chunks)
   - **250.0K tokens → 615.1 s → 5000 ms per-chunk TOTAL** (123 chunks)
   These are TOTAL per-chunk numbers (compute + collective + overhead + everything), labeled as such and
   **not decomposable**. See §4.

4. **The fixed-vs-variable split from a 3-point regression gives a total-per-chunk intercept of
   ~4637 ms/chunk — NOT separable into a fixed-overhead component, and nowhere near 390 ms except as
   ~8% of intercept.** Regression of per-chunk wall vs mean context depth (n=3, R²=0.995) →
   **intercept ≈ 4637 ms/chunk, slope ≈ 0.00288 ms/chunk/token** (= ~5.9 ms per 2048-token step growth).
   The intercept ~4637 ms is an UPPER BOUND on any depth-independent fixed overhead AND conflates ALL
   depth-independent cost (model fixed cost, warm pipeline, etc.), so it is not an estimate of
   launch+sync+collective overhead. 390 ms would be **~8.4% of the intercept** — but the intercept is
   overwhelmingly real compute, not the claimed overhead mechanism. n=3, single-run each — treat the
   ~4637 ms figure as an upper bound with heavy caveats, not a measured fixed overhead. See §6.

5. **What spans exist TODAY on the prefill path (TP), and the key gap for any real measurement:**
   `request_trace` records per-chunk named spans `prefill_batched.chunk<i>.forward`,
   `.barrier`, `.distributed_cb`, `.eval_cache`, and there is a `T("prefill_batched.clear_cache")` span
   PLUS two T-spans for clear-cache. There is **no** named span separating `mx.eval([c.state...])`
   sync cost, **no** span per individual `all_sum` collective (those are the model-internal spans
   produced only under `EXO_PROFILER=spans`, currently unset), and **no kernel-launch count** capture.
   Full enumeration and gaps in §2.

---

## §1 Instrumentation landscape (start_cluster.sh / source)

Profiler-related env vars and what they control (sink = where output lands):

| Env var | Default | What it turns on | Output sink |
|---|---|---|---|
| `EXO_PROFILER` | unset (off) | comma list of hook variants: `spans` (per-span wall accumulator, replaces `EXO_MINIMAX_TRACE`), `layer_memory` (per-layer Metal mem snapshot; level 2 = pre+post layer) | dumps on SIGUSR1 or atexit (SpanProfilerHook) |
| `EXO_PROFILER_LEVEL` | `1` | snapshot depth for `layer_memory` | — |
| `EXO_PROFILER_SYNC_SPANS` | unset | forces `mx.synchronize()` at every span boundary in SpanProfilerHook (real GPU kernel time, not lazy-graph) | same dump |
| `EXO_DSV4_MTP_PROFILE` | `0` | per-cycle decode-phase timer (`_PhaseTimer`), emits `[MTP-PROF]` every N cycles | `logger.warning` → runner stderr → `~/exo.log` (`Runner stderr:` lines) |
| `EXO_DSV4_RB_PROFILE` | unset | adds rollback sub-phase boundary sync to MTP-PROF (`rb_snap/rb_gate/rb_drain/rb_ring/rb_pool/rb_commitfwd/rb_tail`) — decode-only | same |
| `EXO_TRACING_ENABLED` | `false` | turns on `request_trace`/`T()` span recording (per-request timeline) | `logger.info` `[TRACE] Request timeline:` → `~/exo.log` |
| `EXO_DSV4_MTP_PROFILE` + `EXO_DSV4_RB_PROFILE` | — | the two vars the depth-scan used (=50 and =1) | — |
| `EXO_PREFILL_STEP_SIZE` | `2048` (`start_cluster.sh:88`) | TP chunk size | — |
| `EXO_PREFILL_CLEAR_CACHE_INTERVAL` | `1` (=every chunk) | how often prefill loop calls `mx.clear_cache()` | — |

**Which are PREFILL-side:** `EXO_TRACING_ENABLED` (records `prefill_batched.chunk*.forward/barrier/eval_cache`
and the `prefill_batched.clear_cache` T-span) and `EXO_PROFILER=spans` (inside-model spans like
`moe.all_sum`, `attn.sdpa` — these DO cover prefill because prefill runs the same forward). `EXO_DSV4_MTP_PROFILE`
is **decode-only** (see VERDICT 2).

The `[MTP-PROF]` sink: `dsv4_mtp.py:819-846` `logger.warning` → runner stderr captured into `~/exo.log`
(greppable as the `Runner stderr: [MTP-PROF]` lines in the prof logs). The `request_trace` sink:
`trace.py:86-111` `logger.info("[TRACE] Request timeline:")`.

## §2 Spans on the prefill path today (TP driver `prefill_batched`)

Locations in `src/exo/worker/engines/mlx/generator/generate.py` (the fork's TP driver; the mlx-lm
`generate.py` is the serial fallback and the PP driver is `_pipeline_parallel_prefill_steps` in the same file):

- **forward** span: `generate.py:1459-1462` `request_trace.record("prefill_batched.chunk{chunk_idx}.forward(...)")`.
- **barrier** span: `:1496-1500` `mx_barrier(group)` under `request_trace.record("...barrier")`.
- **distributed_cb** span: `:1502-1507`.
- **eval_cache** (the per-chunk mx.eval sync): `:1509-1526` — `mx.eval([c.state for c in batched_cache])`
  recorded as `...eval_cache` (or `mx.async_eval` when `EXO_PREFILL_CHUNK_OVERLAP=1`).
- **`mx.clear_cache()` is called EVERY chunk** — `generate.py:1527` (unconditional, inside the while loop,
  after eval). This is the "clear_cache every chunk" the skill note describes. It is NOT itself wrapped in a
  timed span (only the pre-entry `T("prefill_batched.clear_cache")` at `:1385` and the post-extract one at
  `:1590` bracket the pre/post clear, ~0 for the in-loop one). **So today a per-chunk `clear_cache` time is
  measurable only as part of the unbracketed gap or via `EXO_PROFILER=spans`; there is no dedicated span.**
- **No span around (a)** the per-chunk `mx.clear_cache()` at `:1527`, **(b)** the `mx.eval` sync separately
  from eval_cache (they ARE the same record), **(c)** individual `all_sum` collectives (these are model-internal
  spans only present under `EXO_PROFILER=spans`: `moe.all_sum`, `attn.all_gather` etc.), **(d)** kernel launch
  counts (no counter exists). 

**Chunk-size halving is PP-only and does NOT apply to TP:** the PP driver `_pipeline_parallel_prefill_steps`
does `prefill_step_size = prefill_step_size // min(4, group.size())` at `generate.py:497`. The TP driver
`prefill_batched` uses the FULL step: `n_to_process = min(prefill_step_size, max_length-offset)` at
`generate.py:1454`, with `prefill_step_size = EXO_PREFILL_STEP_SIZE` (default 2048 via `:1378-1379`, default
line `start_cluster.sh:88`). **On the TP cluster, real chunk = 2048 tokens.** (Also matches the
`exo-dsv4-prefill-tuning` skill: "TP prefill hc/residual ops run at FULL L=2048; the halving is PP-loop-only,
`generate.py:497`".)

**PP driver spans** (for completeness): `_pipeline_parallel_prefill_steps` records `prefill.chunk<i>.forward`
(`:572`), `.distributed_cb` (`:578`), `.flush_sends` (`:582`), `.eval_cache` (`:586`), `.contiguous` (`:599`),
plus `prefill.post_loop_token` (`:656`). PP-only; not the cluster's path.

## §3 Does the depth-scan log contain prefill timing? NO.

All 6 `prof_*.txt` files are `[MTP-PROF]` decode-phase dumps. Verified:
- 156 `[MTP-PROF]` lines in `prof_089k_n1.txt`; the only phases present are `B=1
  draft/verify/accept/rollback/rb_drain/rb_gate/rb_pool/rb_ring/rb_snap/rb_tail/total` (full set, see VERDICT 2).
- Zero occurrences of `prefill`, `TRACE`, `forward`, `all_sum`, `clear_cache`, `kernel` in any `prof_*.txt` (grep rc=1).
- The `_PhaseTimer` that emits these is decode-cycle-only (`dsv4_mtp.py:790`).

These logs capture decode cycle decomposition (draft=8.6ms, verify=62.2ms, etc. at 89K) — valuable for the
decode-side decomposition this scan was built for, **but contain no prefill per-chunk timing**.

## §4 Prefill wall from existing data (bench_*.json)

Real prompt token counts, measured prefill tps (`prompt_tps` field, all 5 scored iterations per depth), and
derived per-chunk TOTAL wall (chunks = ceil((prompt_tokens−1)/2048), since prefill_batched processes
`prompt[:-1]` and TP uses full 2048):

| depth | real prompt tokens | prefill wall (s) | prefill tok/s | chunks @2048 | per-chunk TOTAL wall (ms) |
|---|---|---|---|---|---|
| 089k | 89,408 | 210.2 | 425.4 | 44 | **4777 ms** |
| 150k | 150,013 | 358.4 | 418.6 | 74 | **4843 ms** |
| 250k | 250,019 | 615.1 | 406.5 | 123 | **5000 ms** |

Provenance: `tmp/verify-decomposition-20260901/raw/bench_{089k,150k,250k}.json` — `by_concurrency["1"].iterations[].per_request[].prompt_tokens` + `.prompt_tps`; wall = tokens/tps; median over the 5 scored iterations. These are TOTAL per-chunk numbers (all cost), NOT fixed overhead — do not claim they decompose. The measured `prompt_tps` here (~425/419/407) are higher than the renormalized 225/214/202 in the skill — because these `bench_*.json` `prompt_tps` use the server's own tokenizer count (89,408 etc. are real tokens), and these are cumulative wall-based — different from the skill's renormalized figures; the key use here is wall and per-chunk total, which are internally consistent (wall samples per depth are tight: 209.15–210.29 s).

**Cross-check** against independent span-profile: 220K span profile (`docs/dsv4-220k-prefill-span-profile-2026-08-18.md`) measured 220,321 tokens in 612 s clean wall = ~360 tok/s, ~108 chunks @2048 → **~5667 ms/chunk TOTAL**, consistent with the 4777–5000 ms band here.

## §5 Can fixed vs variable be separated from existing data? Only as an upper bound.

3-point linear regression (per-chunk TOTAL ms vs mean context depth ≈ ½·prompt_tokens, since context grows
0→P across chunks), n=3, single run each:

```
intercept ≈ 4637 ms/chunk     (the depth-independent component)
slope     ≈ 0.00288 ms/chunk/token  (≈ 5.9 ms per 2048-token step)
R² = 0.995
```

**Interpretation and caveats (state these verbatim):** the intercept ~4637 ms is an UPPER BOUND on any fixed
per-chunk overhead, but it CONFLATES ALL depth-independent cost — including the model's own minimum
(forward compute at ~zero context), warm pipeline, and any amortized startup — not just launch + all_sum +
eval sync + clear_cache. It is NOT a measurement of the 390 ms mechanism. The claimed 390 ms would be
**~8.4% of this intercept.** And the only directly-measured instance of that mechanism's bracket
(prefill.clear_cache+barrier) is under 0.02% of wall (§VERDICT 1 / `prefill-trace-instrumentation-findings-2026-08-21.md`).
The slope (~5.9 ms growth per 2048-token step) is consistent with real compute growth (compare
`per-chunk-timing-method.md` 2026-08-16: `forward` slope ~+2.03 ms/chunk at 1024-token chunks on the PP path,
= ~4 ms per 2048 — same order). **Bottom line: existing data can bound the fixed component only at roughly
"between <0.02% of wall and the ~4637 ms intercept"; it cannot prove any specific fixed overhead, and 390 ms
is neither supported nor the dominant term.**

**To actually measure fixed-vs-variable cleanly (REQUIRES CLUSTER RUN):** enable `EXO_TRACING_ENABLED=1`
(records the `prefill_batched.chunk*.forward/barrier/eval_cache` spans + the clear_cache bracket) and/or
`EXO_PROFILER=spans EXO_PROFILER_SYNC_SPANS=1` (model-internal per-span, incl. `moe.all_sum`, `attn.sdpa`),
and add a dedicated span around the in-loop `mx.clear_cache()` at `generate.py:1527` (currently none exists
— see §2). Even better: a chunk-interval sweep (e.g. 1024 vs 2048 vs 4096 at a fixed depth) isolates the
depth-fixed cost from the per-chunk-fixed cost. This is a real-instrumentation next step, not resolvable
read-only.

---

## Summary of gaps (what would need a new measurement)

- **No prefill span data** exists in the depth-scan logs (decode-only `[MTP-PROF]`).
- **No dedicated span** around per-chunk `mx.clear_cache()` (`generate.py:1527`) — need to add one.
- **No per-collective or kernel-launch timing** without `EXO_PROFILER=spans` (unset during the scan).
- **390 ms is a code comment (2026-06-21, `c0d012d2`), not a measurement**, and contradicts the
  2026-08-21 instrumented measurement (<0.02% of wall for that exact bracket).

**Files created:** `tmp/prefill-round1-20260902/findings/item2-instrumentation.md` (this file).
**Files modified:** none. **Cluster contact:** none.
