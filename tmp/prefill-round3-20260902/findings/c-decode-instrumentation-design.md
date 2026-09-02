# Decode-side discriminator, round 3: pitfall-proof direct per-collective instrumentation (DESIGN ONLY)

Date: 2026-09-02 · Repo HEAD: **`17d427b01`** (`docs(perf): PREFILL ROUND 2 …`)
Round-2 baseline HEAD was `80db9a855`. **Drift check: HEAD moved 4 commits (docs-only
commit chain ending in the round-2 REPORT commit); both candidate line ranges are
byte-identical to round 2 — zero drift.** Scope: design only. No src/ edit, no cluster
run, no ssh, no commit, no push. `git status --short src/` verified empty after
producing the patch artifact.

Supersedes §2 of `tmp/prefill-round2-20260902/findings/decode-discriminator.md`.
Round 2's design was directionally right (direct timing, reuse of existing sinks,
env-gated no-op) but its per-collective records were **duration-only with no stratum
key, no warmup flag, no rank id, and no disjointness proof** — a frontier reviewer
(fable) named four pitfalls that would each independently invalidate the measurement.
This round closes all four *structurally in the code*, not in prose.

---

## 1. Verified file:line table at HEAD `17d427b01` (both exact, zero drift from `80db9a855`)

### Candidate A — fenced coordination collectives (`dsv4_mtp.py`, `_next()` BS>1 dispatch)

| What | file:line (verified at 17d427b01) |
|---|---|
| `_next()` override begins | `dsv4_mtp.py:2197` |
| Gate line (`if coord_group … size()>1 … len(gen_batch)>=1`) | `:2259` |
| Step 1: presence bitmask build | `:2263-2268` |
| **A1 collective** `counted = mx.distributed.all_sum(presence_arr, group=coord_group)` | `:2269` |
| `mx.eval(counted)` — the sync point (already present) | `:2270` |
| `.tolist()` + uid-intersection local logic | `:2271-2291` |
| filter + `_cleanup_uid` | `:2292-2300` |
| **A2 collective** `synced = mx.distributed.all_max(mx.array(gen_batch._num_tokens), group=coord_group)` | `:2306-2308` |
| `mx.eval(synced)` — the sync point (already present) | `:2309` |
| `gen_batch._num_tokens = synced.tolist()` | `:2310` |
| End of fenced block (task range 2259-2310) | `:2310` ✓ exact |

Sink: `_mtp_trace_log` — `dsv4_mtp.py:1876-1891`. Opens
`/tmp/dsv4_mtp_trace_rank_{rank}_pid{pid}.log` (per-rank, `ab`, buffering=0), gated on
`EXO_DSV4_MTP_TRANSITION_TRACE=1` at each call site (gate reads at `:2213`, `:2279`).
Verified side effects of the sink: increments `_mtp_trace_seq` and sets
`self._mtp_drift_step: int = 0` (`:1891`). Grep confirms `_mtp_drift_step` has **no
read site anywhere in the file** — the assignment is dead bookkeeping, so reusing the
sink introduces zero behavior change.

### Candidate B — `agree_on_tasks` / `agree_on_cancellations_fast` (`batch_generator.py`)

| What | file:line (verified at 17d427b01) |
|---|---|
| `class BatchGenerator(Engine)` | `:400` |
| Call site 1 `with T("batch_gen.agree_on_tasks"): self.agree_on_tasks()` | `:678-679` |
| Call site 2 `with T("batch_gen.agree_on_cancellations_fast"): self.agree_on_cancellations_fast()` | `:719-720` |
| **BatchGenerator** `def agree_on_tasks` | `:507` |
| `coord = get_coord_group(self.group)` | `:523` |
| **B1a gate** `if not mx_any(len(self._maybe_queue) > 0, coord): return` | `:524-525` |
| **B1b slow path** `mx_all_gather_tasks(...)` | `:527` |
| Existing `[PROF]` idiom (`_dt > 0.005` + `EXO_TRACING_ENABLED`) | `:532-534` |
| **BatchGenerator** `def agree_on_cancellations_fast` | `:561` |
| `coord = get_coord_group(self.group)` | `:578` |
| **B2a fast gate** `if not mx_any(has_anything, coord): return` | `:580-581` |
| **B2b** `mx_any(has_cancel_all, coord)` | `:584` |
| **B2c slow path** `mx_all_gather_tasks(...)` | `:587` |

**Duplicated `agree_*` methods — verified by grep at HEAD `17d427b01`:**

```
94:  class SequentialGenerator(Engine)
159:    def agree_on_tasks(self) -> None            <- SequentialGenerator copy
187:    def agree_on_cancellations(self) -> None
208:    def agree_on_cancellations_fast(self) -> None
400: class BatchGenerator(Engine)
507:    def agree_on_tasks(self) -> None            <- BatchGenerator copy  [INSTRUMENT THIS]
561:    def agree_on_cancellations_fast(self) -> None                       [INSTRUMENT THIS]
```

The patch touches ONLY the `:507` / `:561` copies (they are textually distinguishable:
the BatchGenerator copy carries the coord-group docstring block and the
`upstream PR #2048` comment; the patch's context anchors on those). The
SequentialGenerator copies at `:159`/`:208` are left untouched — they serve a
different, non-batched-decode engine path and would pollute the decode-rate counts.

Primitives (verified for lazy-eval handling):
- `mx_any` = `utils_mlx.py:1757-1764`: one `mx.distributed.all_sum(..., stream=CPU)` +
  **`mx.eval(num_true)` + `.item()` inside the primitive** — fully synchronous.
- `mx_all_gather_tasks` = `utils_mlx.py:2240-2329`: count `all_gather` materialized via
  `.tolist()` (`:2276-2277`), payload `all_gather` materialized via
  `.reshape(...).tolist()` (`:2315-2317`) — fully synchronous.
- Candidate A's collectives each already carry an in-place `mx.eval` (`:2270`, `:2309`).

**Key structural fact for the overhead question:** every timed interval in this design
already ends at an existing, unconditional materialization point. The instrumentation
adds **zero** new `mx.eval`/`mx.synchronize` calls.

---

## 2. Disjointness proof (pitfall 4): can A nest inside B or vice versa?

**Finding: NO nesting is possible in either direction on the decode path. The two
candidate timer sets are structurally disjoint — no exclusion rule or subtraction rule
is needed.** Evidence chain, with file:line:

The runner-level decode loop is `BatchGenerator.step()` (runner-side wrapper,
`batch_generator.py:662`) → `self._gen.step()` at `:849` (the `ExoBatchGenerator`
in `generator/batch_generate.py:686`) → its dispatch at `batch_generate.py:4204-4231`:
- `self._batched_decode_active` → `_step_batched_decode()` (`:3978`),
- else pp-spec → `_step_pp_spec()` (`:3543`),
- else plain → `self._mlx_gen.next()` (`:4229`).

**Candidate A's timers live inside `_next()`** (`dsv4_mtp.py:2197`, the
`DSv4MTPBatchGenerator` override; the `gen_batch.next()`/`._next()` machinery of
mlx-lm `BatchGenerator`). **Candidate B's timers live inside `agree_on_tasks()` /
`agree_on_cancellations_fast()`**, whose decode-rate call sites are `:678-679` and
`:719-720` — i.e., **in the code that runs BEFORE `self._gen.step()` is invoked**
(`:849` follows them in the same `step()` body; see `:835`'s own comment: "…
`self._gen.step()` (issuing TP all_reduces from the model …" — the agree calls are
textually and temporally before it).

Forward direction (B inside A): `agree_on_tasks`/`agree_on_cancellations_fast` are
never called from anywhere inside `_next()` or the spec decode cycle. Grep of
`dsv4_mtp.py` for `agree_on` yields **zero call sites** (only the `:2321` comment
mentioning the downstream callback name). The only `agree_on_cancellations_fast` /
`agree_on_tasks` call sites in the file are `batch_generator.py:143-144` (Sequential
warmup), `:244/:340/:351/:355` (Sequential step), `:491-492` (BatchGenerator warmup),
`:678/:719` (BatchGenerator.step — **the decode-rate site**), `:1176/:1272/:1275/
:1344/:1360/:1363` — and the latter group are the per-task callback closures defined
in `_start_task` (`on_generation_token` at `:1237`, invoked from
`batch_generate.py:4266-4268` **after** the `next()`/`_step_*` call at `:4206-4229`
returns, and from the pp-spec loop at `:2344`/`:2677`/`:3534` after the cycle's
forward returns). So the B collectives fire either before `step()`'s engine call or
after it returns — never during A's fenced block.

Reverse direction (A inside B): A's fenced block sits at the top of `_next()`
(`:2259-2310`) before any forward; `agree_on_tasks` does no model work and calls no
generator method that reaches `_next()` — its body is `get_coord_group` + `mx_any` +
`mx_all_gather_tasks` + queue bookkeeping (`:523-534`), verified by full source read.

**Residual overlap considered and dismissed:** the counter-gated callback
`agree_on_cancellations()`/`agree_on_tasks()` pairs (`:1272/:1275`) and the pp-spec
loop's cadence calls (`batch_generate.py` `:2344`, `:2677`) run in the same *thread*
but are sequenced after the decode cycle returns, so their intervals cannot bracket an
A interval. Even in the worst interleaving, the instrumented primitives
(`mx_any`/`mx_all_gather_tasks` at `:507/:561`) contain no call into the generator
object, so no A collective can execute inside a B timer.

**Conclusion:** A and B timers can never be nested or overlap on any single thread of
execution; per-call records from A and B are independent, additive cost measurements
with no double-counting. The design therefore needs no subtraction rule — but it does
keep the two sinks at explicitly different levels (A = per-collective JSONL records
inside the MLX generator; B = per-primitive `[PROF]` logger lines at the runner
wrapper), which makes any accidental future re-nesting visible in the record stream
(different `collective`/event namespaces) rather than silently merged.

---

## 3. The four pitfalls — structural handling (with the actual code)

### 3.1 Lazy eval (enqueue ≠ execution)

**Structural handling: no new sync is added; instead each timer's END is placed
immediately after the interval's own existing, unconditional materialization point,
so the measured duration includes the real execution of that collective and nothing
else.**

- **A1** (`dsv4_mtp.py:2269-2270`): start timestamp immediately before
  `all_sum(presence_arr, ...)`, end timestamp immediately after `mx.eval(counted)`.
  What is eval'd: **`counted`** — the 1024×int32 all_sum output, the exact array whose
  cross-rank reduction the call exists to produce. The subsequent `.tolist()` at
  `:2272` reads a value already materialized by that eval, so stopping the clock after
  `mx.eval` captures the full wire+drain cost without including local list-building.
- **A2** (`:2306-2309`): start before `all_max(mx.array(gen_batch._num_tokens), ...)`,
  end after `mx.eval(synced)`. Eval'd: **`synced`** — the per-uid `_num_tokens`
  all_max output. The `synced.tolist()` at `:2310` is downstream of the eval and is
  excluded (it is Python-side copy work, not collective cost).
- **B1a/B2a** (`mx_any`): the start is placed immediately before the `mx_any(...)`
  call and the end immediately after it returns. Because `mx_any` itself ends with
  `mx.eval(num_true)` + `.item()` (`utils_mlx.py:1763-1764`), the measured interval
  ends at a real materialization — this is the structural fix. The code does **not**
  add another `mx.eval`; it exploits the primitive's own.
- **B1b/B2c** (`mx_all_gather_tasks`): same principle — the primitive materializes its
  output via `.tolist()` (`utils_mlx.py:2276-2277`, `:2315-2317`); the timer ends when
  it returns, i.e., after the final gather's bytes have been materialized on the CPU
  stream.

Why this is safe where `EXO_PROFILER_SYNC_SPANS=1` was not: the sync-span profiler
*inserts* syncs at dozens of span boundaries × 43 layers × every chunk, inflating
collective times and tripping the 45s hang watchdog. Here every interval already ends
at a sync that production executes unconditionally — the instrumentation only observes
it. See §7 for the honest caveat about what "already synchronous" does and does not
cover on this stack.

### 3.2 Cross-rank clock skew (durations only, structurally enforced)

**Every emitted record contains: a duration in µs, a rank id for labeling, a coord
group size, and context scalars. No record contains an absolute timestamp of any kind
— no `time.time()`, no `perf_counter()` origin, no shared-epoch field. There is
nothing in the schema to subtract across ranks.**

- A's records carry `us`, `warmup`, `local_num_tokens` (or `depth_committed`),
  `coord_rank`, `coord_size`, `batch_size`, plus the sink's own per-rank `seq` counter
  (per-process, monotonically increasing — a *sequence*, not a clock).
- B's `[PROF]` lines carry `*_us`, `warmup=`, `depth=`, `rank=`, `queue=/n_local=`.
- The one place a naive design would be tempted to emit a timestamp — correlating an
  A record with a B record, or rank0 with rank1 — is instead handled by the analysis
  rule: **within-rank sequence numbers establish ordering; cross-rank correlation is
  done on counts and duration distributions, never on time coordinates.** The record
  schema's comment in the patch states this explicitly at the emit site so a future
  editor extending the schema sees the rule where they're typing.

### 3.3 First-call warmup (bucketed in the record itself)

Each timed site keeps a per-instance occurrence counter, initialized to zero, and
emits `"warmup": "warmup"` on the counter's first increment and `"warmup": "steady"`
thereafter. The counter is only touched when profiling is on (strict no-op when off).
Concretely, in the patch:

- A: `self._coord_sync_all_sum_calls` / `self._coord_sync_all_max_calls`, initialized
  in `DSv4MTPBatchGenerator.__init__` (next to the existing trace handles), read in
  the emit blocks: `self._mtp_trace_log(..., {"warmup": "warmup" if
  self._coord_sync_all_sum_calls == 0 else "steady", ...})` then
  `self._coord_sync_all_sum_calls += 1`.
- B: `_agree_on_tasks_calls`, `_agree_on_tasks_gather_calls`,
  `_agree_cancel_fast_calls`, `_agree_cancel_gather_calls` as `field(default=0,
  init=False)` on `BatchGenerator`, with the identical `== 0` test in each emit.

Analysis rule (pre-registered): every record with `warmup="warmup"` is excluded from
steady-state means and reported separately (typically one all_sum, one all_max, and a
handful of agree_* first-gathers per process — JACCL QP setup, first-touch pages, lazy
coord-group construction). No mean in §5's bands is computed over warmup records.

### 3.4 Disjoint placement — settled by §2's proof

No nesting exists, so no exclusion/subtraction rule is required. The design still
enforces two structural guards against *future* drift:

1. A and B use different sinks and different event namespaces
   (`decode_coord_collective_us` JSONL events vs `[PROF] agree_*` log lines) — any
   future refactor that moves one timer inside the other's interval will show up as
   implausible duration distributions (the outer would grow by the inner's mean),
   detectable by the §5 "calls × per-call ≤ wall" reductio that round 2 already
   mandated and that caught the 178 ms/call fabrication.
2. Neither timer wraps a span that contains the other's call site: A's interval is
   strictly inside `_next()` before any yield/forward; B's intervals are strictly
   inside the agree methods before `_gen.step()` begins. There is no text location
   where both could be live simultaneously.

---

## 4. Record schema (exact)

### Candidate A — JSONL via `_mtp_trace_log` (file `/tmp/dsv4_mtp_trace_rank_{rank}_pid{pid}.log`)

```json
{"seq": 1234, "event": "decode_coord_collective_us",
 "collective": "upstream_sync_all_sum",
 "us": 1187.4, "warmup": "steady",
 "local_num_tokens": [30145, 30145],
 "coord_rank": 0, "coord_size": 2, "batch_size": 2}
```
```json
{"seq": 1235, "event": "decode_coord_collective_us",
 "collective": "upstream_sync_all_max",
 "us": 1210.9, "warmup": "steady",
 "depth_committed": 30145,
 "coord_rank": 0, "coord_size": 2, "batch_size": 2}
```

`seq` = per-process monotonic counter (ordering within a rank only). `us` = the
duration. `depth_committed` = stratum key for A2 (the value being max'ed IS the
committed depth, per round 2's refinement). For A1 the stratum key is
`max(local_num_tokens)` (computed at analysis time). No timestamp field exists.

### Candidate B — `[PROF]` logger lines (runner stderr → `~/exo.log`), one shape per primitive

```
[PROF] agree_on_tasks.gate_us=842.3 warmup=steady depth=30145 rank=0 queue=0
[PROF] agree_on_tasks.gate_us=831.7 warmup=steady depth=30145 rank=0 queue=0 slow_path=1
[PROF] agree_on_tasks.all_gather_us=3120.4 warmup=steady depth=30145 rank=0 n_local=1
[PROF] agree_on_cancellations_fast.gate_us=851.2 warmup=steady depth=30145 rank=0 local_cancels=0
[PROF] agree_on_cancellations_fast.all_gather_us=2980.6 warmup=steady depth=30145 rank=0 n_local=2
```

Note B1a/B2a emit on BOTH branches (fast-path return and slow-path continuation) —
the `mx_any` gate fires unconditionally every step, so its cost is measured every
step; `slow_path=1` marks the calls that continue into the all_gather. The existing
`_dt > 0.005` threshold-and-suppress pattern in the old slow-path `[PROF]` line is
deliberately NOT reused for the new lines: suppressing sub-5 ms samples would bias the
gate-cost distribution (the ~1.2 ms signal must clear the threshold, not hide under
it). The old `:532-534` line remains as-is (it is a legacy aggregation, not the
instrument). Log volume: ~4 lines/step at 25-60 Hz ≈ 100-240 lines/s while profiling
is on — grep-able by the exact `[PROF] agree_on_` prefix, off entirely when the env
var is unset.

---

## 5. Env gating, strata, bands (pre-registered, restated verbatim where decided)

### Env vars introduced (exact names)

| Variable | Effect |
|---|---|
| `EXO_DSV4_DECODE_COLLECTIVE_PROFILING=1` | **The single new gate.** Read ONCE at module import in both files (module constants `_DECODE_COLLECTIVE_PROFILING` in `dsv4_mtp.py`, `_DECODE_AGREE_PROFILING` in `batch_generator.py`). When unset/0: hot-path cost is one boolean check per collective; no counters advance; no records are written. |
| `EXO_DSV4_MTP_TRANSITION_TRACE=1` | Pre-existing gate for the A-side SINK. The new A records ride `_mtp_trace_log`, which is individually gated on this var at the call site — so an approved run sets BOTH vars. (Candidate B needs only the new var.) |

Per the `EXO_DSV4_MTP_PROFILE` convention: read-once module constants, `== "1"`
string comparison, strict no-op otherwise. No new machinery, no new files, no new
config surface.

### Depth stratification (decided round 2 — implemented via the record's depth field)

S1 0-4K, S2 4-16K, S3 16-32K, S4 32K+ (committed decode depth: `max(_num_tokens)` for
A — the all_max's own payload; `min(_gen_num_tokens_depths())` for B — conservative
across uids). Decision is made on **S3-S4 means**, no post-hoc threshold tuning.
Motivation unchanged: MTP acceptance decays 1.411 → 1.312 → 1.226 with depth, so
shallow strata underestimate per-call coord cost.

### Pre-registered bands (verbatim from round 2 — NOT altered)

- **Candidate A costs**: `(A.all_sum + A.all_max) ≥ 600 µs/call` AND `≥ 10%` of
  decode-step wall in S3-S4.
- **Inconclusive**: strata straddle a band boundary by ±20%.
- (Round 2's B bands retained for completeness: B fast-gate costs at ≥300 µs/call AND
  ≥5% of step wall; B slow-path all_gather at ≥600 µs/call AND ≥10% of full-request
  decode wall; "neither costs" if the combined total <500 µs/call AND <8% of step
  wall. Any mean <200 µs treated as noise. No post-hoc threshold adjustment.)

Decode-step wall for the %-of-step denominators comes from the pre-existing
`request_trace.record("decode.step.mlx_next", ...)` spans (`batch_generate.py:4208/
4213/4231`) — already wired, no new plumbing.

---

## 6. Overhead budget and THE judgment call: does the sync requirement perturb the measurement?

### Bare instrumentation overhead (the round-2 estimate, still valid)

Two `perf_counter` calls + one dict/list build + one small file append or
`logger.info` per call: ~40-100 ns of timing cost against the ~1.2 ms/call signal =
0.003-0.008%. The record emission adds ~1-3 µs (JSONL line build + unbuffered write;
`logger.info` formatting ~2-5 µs). Against a 1.2 ms signal that is ≤0.4% —
negligible, and it is also *off* the timed interval (emission happens AFTER the end
timestamp; the next call's start timestamp is a fresh `perf_counter`, so emission
cost lands between intervals, not inside any measured duration).

### The judgment call: NO new sync is forced, and that is the load-bearing design fact

Round 2's brief assumed pitfall 1 would require the design to *insert* an
`mx.eval()` before the end timestamp, and correctly worried that a forced synchronize
"can serialize work that was previously overlapped." **This design does not need to
insert any sync, because every measured interval already terminates at an
unconditional materialization that production performs anyway:**

- A1: `mx.eval(counted)` at `:2270` — existing, unconditional, in-place.
- A2: `mx.eval(synced)` at `:2309` — existing, unconditional, in-place.
- B1a/B2a: `mx_any` internally does `mx.eval` + `.item()` (`utils_mlx.py:1763-1764`)
  — the primitive is born synchronous; every production call already pays this.
- B1b/B2c: `mx_all_gather_tasks` materializes both gathers via `.tolist()`
  (`:2276-2277`, `:2315-2317`).

So the instrumentation's measured durations are the *production* durations, not
upper bounds inflated by added syncs, and nothing that previously overlapped is
serialized by the instrumentation itself. This is the structural answer to the
perturbation question, and it is why this design differs from
`EXO_PROFILER_SYNC_SPANS=1` (which inserts new syncs and is documented in the skill
as both inflating collective times and tripping the 45s hang watchdog).

### Honest caveat — what "already synchronous" does NOT cover on this stack

The stack fact remains: every collective here pins to `JACCLGroup::
communication_stream()`'s single owned CPU-only stream (`AllReduce::eval_gpu` is a
hard throw in this mlx fork), so each collective is a real GPU→CPU→GPU
drain-and-refill. Two consequences stated honestly:

1. **The measured ~1.2 ms/call already INCLUDES the pipeline drain that the
   collective's own sync forces.** That is the correct quantity for the question
   being asked ("what does the coord protocol cost decode?") — the drain is a real,
   unconditional production cost, not an instrumentation artifact. What the design
   must NOT claim is that the number isolates "wire time"; it measures
   drain+wire+refill+eval. The band thresholds (600 µs) were set against this
   inclusive definition in round 2 and are unchanged.
2. **The instrumentation cannot eliminate an overlap that the code itself created**
   (e.g., if some future change made the presence-bitmask build overlap the previous
   collective's drain), because the start timestamp sits before the enqueue of the
   wrapped op only — anything already-overlapped *outside* the interval is untouched.
   The one theoretical perturbation left: the added ~1-3 µs Python-side record
   emission after the end timestamp slightly lengthens the gap before the NEXT
   collective's start. At 25-60 Hz decode that is ≤0.015% of the inter-call period —
   far below run-to-run variance, and it does not touch any measured interval.

### Detecting perturbation if it exists anyway (pre-registered check, cheap)

The on/off end-to-end comparison from round 2, made concrete:

- **Detection**: one approved run sets `EXO_DSV4_DECODE_COLLECTIVE_PROFILING=1`; the
  existing production telemetry (per-request decode tok/s at matched depth strata,
  from `usage` + wall clock — the skill's counting rules apply) is compared against
  the immediately preceding un-instrumented runs at the same depths. Gate: median
  decode tok/s delta within ±2% ⇒ instrumentation does not perturb the regime; a
  larger delta ⇒ report the timing data as suspect and do not draw the §5 verdict
  from it (re-plan with a lower-emission sink, e.g. accumulate in-memory and flush
  every Nth call, before any re-run).
- **Watchdog safety**: unlike sync-spans profiling, this design inserts no new sync
  points, so the `HANG_TIMEOUT_SECONDS` trip mechanism (dozens of spans × 43 layers
  × per-chunk syncs pushing inter-callback gaps past 45s) does not apply — the added
  per-step work is microseconds. The runbook still pairs the run with
  `EXO_RUNNER_HANG_TIMEOUT_SECONDS=300` as cheap insurance, and `EXO_PROFILER_SYNC_SPANS`
  stays **unset** (never needed for this design; turning it on would contaminate the
  very measurement).

---

## 7. Runbook for a future APPROVED run (no cluster contact made this round)

**Never repeat the two documented incidents**: no `SIGUSR1` to any PID (the
profiler-PID crash risk; this design needs no signals at all — A's records stream to
the JSONL file continuously, B's lines stream to stderr), and no
`EXO_PROFILER_SYNC_SPANS=1` anywhere in this workflow.

1. **Preconditions** (checked, not assumed): no active generation task
   (`curl <api-node>:52415/state` → write to a file first, then parse — the piped
   one-liner corruption pitfall); `git -C ~/repos/exo log --oneline -1` matches the
   HEAD this design verified (`17d427b01` or later with the two candidates re-verified
   by grep: `grep -n "counted = mx.distributed.all_sum" dsv4_mtp.py` → `:2269`,
   `grep -n "def agree_on_tasks" batch_generator.py` → two hits, `:159` + `:507`; if
   either moved, re-run the drift check in §1 before applying).
2. **Apply the patch** (one command, on a branch or main per supervisor policy):
   `git apply artifacts/decode_instrumentation.patch` (validated with
   `git apply --check` at HEAD `17d427b01` this round), then the repo gates
   (`uv run basedpyright`, `uv run ruff check` — expect zero NEW errors; the patch
   adds no new imports beyond `cast` in `batch_generator.py`).
3. **Deploy via the normal git path** (`exo-cluster-deployment` mechanics: commit,
   push, `git reset --hard` on both studios, relaunch) — never hand-edit on nodes.
4. **Launch with**: `EXO_DSV4_DECODE_COLLECTIVE_PROFILING=1
   EXO_DSV4_MTP_TRANSITION_TRACE=1 EXO_RUNNER_HANG_TIMEOUT_SECONDS=300` plus the
   standing production env (spec-on config per the current promoted defaults).
   `EXO_PROFILER_SYNC_SPANS` unset. Verify the env survived the relaunch with
   `ps eww` on both ranks (the silently-dropped-env incident).
5. **Drive decode to depth**: run real conversations to ≥32K committed depth
   (S4), ≥3 reps; a shorter 4-16K pass covers S2. Note which node holds rank1
   (`ps eww | grep pp_rank`) — per-rank JSONL files land on each rank's own node
   (`/tmp/dsv4_mtp_trace_rank_{rank}_pid{pid}.log`); check BOTH hosts before calling
   any sink empty (the rank1-logs-elsewhere pitfall).
6. **Collect**: scp the JSONL files from both nodes; grep
   `'[PROF] agree_on_'` from both nodes' `~/exo.log`. Parse with a script that
   (a) drops `warmup="warmup"` records, (b) buckets by the strata of
   §5, (c) computes per-rank means separately — never merging ranks' durations into
   one cross-rank average without keeping the rank label, and never comparing
   sequence numbers across ranks (they are per-process counters).
7. **Sanity reductio before quoting any number** (round 2's rule, kept):
   `calls × per-call ≤ measured decode wall` per stratum; also confirm
   `agree_on_tasks.gate` fires ~once per step() (frequency check against
   `decode.step.mlx_next` trace counts) — a silent factor-of-2 here is exactly the
   class of error the round-2 178 ms/call reductio caught.
8. **Verdict** per §5 bands on S3-S4 means. Record the on/off throughput delta from
   §6 as the perturbation verdict alongside.
9. **Cleanup**: revert the patch (or keep it dormant — it is a strict no-op when the
   env is unset; the supervisor decides), delete `/tmp/dsv4_mtp_trace_*` from both
   nodes, and append the result to `PRE-REGISTRATION.md` (append-only).

Cluster-time estimate (unchanged from round 2): ~1.5-2 h total (one deploy ~9 min +
2-3 decode runs to S4 at ~15-25 min each), zero run-failure risk from the
instrumentation itself (behavior unchanged when off; when on, no new sync points and
no skipped collectives).

---

## 8. What this design deliberately does NOT do

- No ablation arms, no NOP-ing of load-bearing collectives (settled — see the task
  brief's safety table; `agree_on_tasks` is the only `_queue` filler,
  `agree_on_cancellations_fast` is the 133.7s-bug fix, A's fence prevents LEN_ERR
  wedges).
- No new eval/synchronize anywhere (the central overhead judgment in §6).
- No cross-rank timestamp emission (§3.2 makes it unrepresentable).
- No changes to the SequentialGenerator copies, the legacy `[PROF]` aggregation
  lines, `mx_any`/`mx_all_gather_tasks` internals, or any call-site control flow.
- No signal-based dumps, no sync-span profiler, no watchdog exposure beyond the
  insurance value of the raised hang timeout.