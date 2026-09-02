# Decode-side discriminator: instrumentation design (PREPARE ONLY)

Date: 2026-09-02 · Repo HEAD: `80db9a855` (`docs(perf): PREFILL ROUND 1…`)
Scope: **design only**. No code applied, no cluster run, no ssh, no commits.
All `file:line` references were re-verified against the working tree at that HEAD —
the two given line ranges were **both exact**; nothing drifted.

Goal: discriminate whether either of two coord-subgroup collectives costs
decode wall time (and which one), rather than merely confirming "collectives
cost something." The two call sites already carry `T(...)` spans, but those
spans are only dumped at prefill-end (`generate.py:960`) and only record the
*full function* body, not per-collective internals — so a dedicated,
coarser-but-complete design is needed below.

---

## 1. Verified file:line table (both accurate at HEAD `80db9a855`)

### Candidate A — fenced coordination collectives (`dsv4_mtp.py` `_next()`, BS>1 dispatch)
| What | file:line |
|---|---|
| `_next()` override begins | `dsv4_mtp.py:2197` |
| `sync_group = self._get_sharding_group()` | `:2247` |
| `coord_group = get_coord_group(sync_group)` | `:2258` |
| **Gate line** (`if coord_group … size()>1 … len(gen_batch)>=1`) | `:2259` |
| Step 1 — build 1024-int32 presence bitmask | `:2263-2268` |
| **Step 1 collective** `counted = mx.distributed.all_sum(presence_arr, group=coord_group)` | `:2269` |
| `mx.eval(counted)` (synchronous drain) | `:2270` |
| `.tolist()` + uid-intersection local logic | `:2271-2291` |
| `gen_batch.filter(keep_indices)` + `_cleanup_uid` | `:2292-2300` |
| **Step 2 collective** `synced = mx.distributed.all_max(mx.array(gen_batch._num_tokens), group=coord_group)` | `:2306-2308` |
| `mx.eval(synced)` | `:2309` |
| `gen_batch._num_tokens = …` assignment | `:2310` |
| End of fenced block (given range 2259-2310 is **exact**) | `:2310` |

The block runs once per `_next()` call (~25–60 Hz at decode). Both collectives
are **already synchronous** — each already does `mx.eval(...).tolist()` in
place — so wrapping them in `perf_counter()` adds **no** extra
synchronization point and measures the real (already-synced) wall cost.

### Candidate B — `agree_on_tasks` + `agree_on_cancellations_fast` (`batch_generator.py`)
| What | file:line |
|---|---|
| `class BatchGenerator(Engine)` | `:400` |
| **Call site 1** `with T("batch_gen.agree_on_tasks"): self.agree_on_tasks()` | `:678-679` |
| **Call site 2** `with T("batch_gen.agree_on_cancellations_fast"): self.agree_on_cancellations_fast()` | `:719-720` |
| `def agree_on_tasks` (BatchGenerator) | `:507` |
| `coord = get_coord_group(self.group)` | `:523` |
| **B1a**: `if not mx_any(len(self._maybe_queue) > 0, coord): return` | `:524` |
| **B1b** slow-path: `mx_all_gather_tasks(...)` (all_gather) | `:527` |
| `def agree_on_cancellations_fast` (BatchGenerator) | `:561` |
| `coord = get_coord_group(self.group)` | `:578` |
| **B2a fast gate**: `if not mx_any(has_anything, coord): return` | `:580` |
| **B2b** slow path: `mx_any(has_cancel_all, coord)` | `:584` |
| **B2c** slow path: `mx_all_gather_tasks(...)` (all_gather) | `:587` |

⚠ There are **two** copies of `agree_on_tasks` and `agree_on_cancellations_fast`
in this file — `SequentialGenerator` (lines 159/187/208) and `BatchGenerator`
(507/536/561). The decode-path call sites that matter are in
**`BatchGenerator.step()`** (line 662), which dispatches `_step_batched_decode()`
/ PP-spec decode. **Instrument the BatchGenerator copies only** — the
SequentialGenerator copy is a different, non-decode path and would pollute counts.

Primitives used (all coord subgroup, CPU stream):
`mx_any` = `utils_mlx.py:1757-1764` (one `mx.distributed.all_sum` + `mx.eval`),
`mx_all_gather_tasks` = `utils_mlx.py:2240+` (one count `all_gather` + payload
all_gather: `utils_mlx.py:2272-2277`). Note the prior investigation (~1.2 ms/
call) and this file's own comment (`:516`) flag that driving JACCL all_gather at
decode rate can corrupt return buffers — a strong reason **not** to rely on the
ablation (Section 3) and to prefer direct timing (Section 2).

---

## 2. Direct per-collective timestamp deltas (PRIMARY — recommended)

### Why it is the right primary (and beats ablation here)
- Both candidates' NOP arm is **unsafe** (Section 3), so the ablation is
  disqualified as a safe primary on this exact pair. Direct timing has zero
  behavior change when the toggle is off and only adds `perf_counter` calls when on.
- The collectives are already synchronous (`mx.eval` inside), so the timing wrap
  measures the true wall cost *without adding the distortion that
  `EXO_PROFILER_SYNC_SPANS=1` introduces* (per
  `decode-collective-cost-and-unsafe-ablation-2026-08-21.md` § "UNSAFE technique").
- Direct timestamp deltas have beaten derived/sampled numbers twice in this campaign.

### Overhead analysis (vs. the ~1.2 ms/call reference signal)
- `time.perf_counter()` on macOS arm64 costs **~20–50 ns** per call. Two
  bracketing calls (start+stop) = **~40–100 ns** per collective. Over a 1.2 ms
  signal that is **0.003–0.008%** — four orders of magnitude below measurement
  noise. A decode `_next` runs at ~25–60 Hz; accumulating two counters per call
  adds a bounded, single-int-per-call cost that is immeasurable.
- No `mx.eval` is added anywhere; we wrap *around* the existing sync points.
  This is load-bearing for validity: the `comm-compute-overlap` skill note
  warns that a `perf_counter` wrap must not create a new synchronization point
  that serialises the async model path — here the wrapped calls are already
  synchronous, so nothing new is serialised.

### Dump mechanism — reuse existing no-op-when-off machinery (no new machinery)

**Candidate A:** reuse the existing per-rank JSONL `_mtp_trace_log`
(`dsv4_mtp.py:1876-1891`), already gated on
`EXO_DSV4_MTP_TRANSITION_TRACE=1` and writing to
`/tmp/dsv4_mtp_trace_rank_${rank}_pid${pid}.log`. This is the natural sink:
per-rank, already wired, already a no-op when the env is unset.

**Candidate B:** reuse the existing `[PROF]` logger idiom already present in
this exact class at `batch_generator.py:532-534` and `:557-559` (both gated on
`EXO_TRACING_ENABLED`), which is itself routed through the standard logger
(stderr) — i.e. the same transport as the `SpanProfilerHook` stderr auto-dump.
Alternatively (if a tighter per-decode aggregation is wanted) accumulate into
`request_trace.record(...)` — but the `[PROF]` logger line is simpler and
already the established pattern in this file and fires per-call.

### Exact insertion points + proposed diff (NOT applied)

**File `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py`** — inside `_next()`,
around the Step-1 and Step-2 collectives (lines 2269 / 2306). `time` is already
imported at `:34`.

```diff
             presence_arr = mx.array(local_presence, dtype=mx.int32)
+            _t_a1 = time.perf_counter()
             counted = mx.distributed.all_sum(presence_arr, group=coord_group)
             mx.eval(counted)
+            self._mtp_trace_log("upstream_sync_allsum_us", {
+                "us": round((time.perf_counter() - _t_a1) * 1_000_000),
+                "uids": list(gen_batch.uids),
+                "num_tokens_before": list(gen_batch._num_tokens),
+            })
             n_ranks = coord_group.size()
```
```diff
             if len(gen_batch) >= 1:
+                _t_a2 = time.perf_counter()
                 synced = mx.distributed.all_max(
                     mx.array(gen_batch._num_tokens), group=coord_group
                 )
                 mx.eval(synced)
+                self._mtp_trace_log("upstream_sync_allmax_us", {
+                    "us": round((time.perf_counter() - _t_a2) * 1_000_000),
+                })
                 gen_batch._num_tokens = cast(list[int], synced.tolist())
```
Accumulation: the two `_mtp_trace_log` lines write one JSON line per
call — the analysis phase aggregates `us` by depth stratum (Section 3).
Do **not** put anything between `all_sum`/`all_max` and their own `mx.eval`;
the wrap intentionally spans call→eval→(not tolist, which for A is deferred to
`gen_batch._num_tokens = …tolist()` — but both evals are the sync point).

**File `src/exo/worker/runner/llm_inference/batch_generator.py`** —
`BatchGenerator.step()`. The two `T(...)` spans at 678 and 719 already wrap
the full functions; replace the outer spans with spans that also split the
internal fast-path gate from the slow-path payload so the discriminator can
tell "cheap mx_any every call" apart from "rare all_gather":

```diff
-        with T("batch_gen.agree_on_tasks"):
+        with T("batch_gen.agree_on_tasks"):
             self.agree_on_tasks()
```
(add inside `agree_on_tasks`, `:523-527`):
```diff
         coord = get_coord_group(self.group)
+        _t_b1a = time.perf_counter()
         if not mx_any(len(self._maybe_queue) > 0, coord):
+            _dt = time.perf_counter() - _t_b1a
+            if os.environ.get("EXO_TRACING_ENABLED", "false").lower() in ("true", "1"):
+                logger.info(f"[PROF] agree_on_tasks.fast_mx_any={_dt*1000:.3f}ms")
             return
+        _dt = time.perf_counter() - _t_b1a
+        if os.environ.get("EXO_TRACING_ENABLED", "false").lower() in ("true", "1"):
+            logger.info(f"[PROF] agree_on_tasks.mx_any={_dt*1000:.3f}ms")
+        _t_b1b = time.perf_counter()
         agreed, different = mx_all_gather_tasks(self._maybe_queue, coord)
+        _dt = time.perf_counter() - _t_b1b
+        if os.environ.get("EXO_TRACING_ENABLED", "false").lower() in ("true", "1"):
+            logger.info(f"[PROF] agree_on_tasks.all_gather={_dt*1000:.3f}ms")
```
Similarly inside `agree_on_cancellations_fast` at `:578-587`: wrap the fast-gate
`mx_any` at `:580` and the slow all_gather at `:587` with the identical `[PROF]`
idiom (names `agree_on_cancellations_fast.fast_mx_any` / `.all_gather`). The one
line at `:527`/`:587` already uses this exact pattern for the slow paths
(`:532-534`); extend it to the fast gate and lower the `0.005` threshold to
something like `0.0002` (200 µs) so the ~1.2 ms signal clears it and the 
negligible fast-path cost still logs.

**Ablation fallback note:** if direct timing is ever blocked (e.g. the ~1.2 ms
signal is buried under step-to-step variance and the depth-strata means are
indistinguishable), Section 3's ablation is the fallback — but note both its
unsafety caveats and its higher cluster time.

---

## 3. Independent-ablation design (FALLBACK)

Four arms — **A-only-off, B-only-off, both-off, baseline**. A combined
both-only ablation discriminates nothing; independent toggles let the 2×2
difference attribute the delta.

### Toggle mechanism (independent, cached env read — no instrumentation)
| Candidate | Toggle env var | Gate location | What it skips |
|---|---|---|---|
| A | `EXO_DSV4_MTP_SKIP_UPSTREAM_SYNC=1` | `dsv4_mtp.py:2259` — and the gate AND an env read (read once at module load, cached) | both Step-1 `all_sum` (`:2269`) and Step-2 `all_max` (`:2306`) |
| B | `EXO_DSV4_BATCH_GEN_SKIP_AGREE=1` | `batch_generator.py:678` and `:719` | both `agree_on_tasks()` and `agree_on_cancellations_fast()` calls |

Proposed gates:
```diff
-        if coord_group is not None and coord_group.size() > 1 and len(gen_batch) >= 1:
+        if (
+            coord_group is not None
+            and coord_group.size() > 1
+            and len(gen_batch) >= 1
+            and not _SKIP_UPSTREAM_SYNC          # module const = int(os.environ.get("EXO_DSV4_MTP_SKIP_UPSTREAM_SYNC","0")=="1")
+        ):
```
```diff
-        with T("batch_gen.agree_on_tasks"):
+        if not _SKIP_BATCH_AGREE:                # module const = EXO_DSV4_BATCH_GEN_SKIP_AGREE
+          with T("batch_gen.agree_on_tasks"):
-            self.agree_on_tasks()
+              self.agree_on_tasks()
```
```diff
-        with T("batch_gen.agree_on_cancellations_fast"):
+        if not _SKIP_BATCH_AGREE:
+          with T("batch_gen.agree_on_cancellations_fast"):
-            self.agree_on_cancellations_fast()
+              self.agree_on_cancellations_fast()
```

### Depth strata (mandatory refinement #2) — and why these boundaries
Acceptance rate decays monotonically with depth (measured: 1.411 → 1.312 →
1.226), so a shallow cost does not generalize to deep decode. Rollback
interaction is the mechanism: the `upstream_sync`/`agree_*` blocks exist to keep
*rollback-driven* uid/`_num_tokens` divergence in check (see the block comment at
`dsv4_mtp.py:2230-2258`), so their *hit rate* (and thus cost) rises as deep
rollback frequency rises.

Stratify every sample by the *committed decode depth* = `gen_batch._num_tokens[0]`
(the count of tokens already accepted into the KV cache) at the instant the
collective fires:

| Stratum | Depth range | Why this boundary |
|---|---|---|
| S1 shallow | 0 – 4K | acceptance still near the measured 1.411 regime; rollback rare; coord round-trip hits a warm, short KV |
| S2 mid | 4K – 16K | acceptance slope 1.411→1.312 sits here; KV no longer fits warm GPU caches, first rollback pressure |
| S3 deep | 16K – 32K | acceptance ~1.312→1.226; sustained rollback; the 30K-token slow-decay regime cited in the cancellation fix (`:699-708`) |
| S4 very deep | 32K+ | acceptance floor ~1.226; maximum rollback and longest coord-sync latency; where any real production cost concentrates |

For A, depth = the value being max'ed (`_num_tokens` is itself the depth; the
all_max is exactly a depth consensus) — use `max(local _num_tokens)`. For B,
use `gen_batch._num_tokens[0]` (already snapshotted per step in
`_jaccl_dump_step`, `:626`).

### Safety verdict per arm (mandatory — the "unsafe ablation" hazard)
Reference: `decode-collective-cost-and-unsafe-ablation-2026-08-21.md`

| Arm | Safety | What breaks if it hangs/corrupts | Cluster-hang risk |
|---|---|---|---|
| **A-only-off** | **UNKNOWN / CONDITIONAL** | At **stable single-stream c=1 decode** both ranks agree on uid set and `_num_tokens`, so the Step1/Step2 collectives are logically no-ops — NOPing is *likely* safe there. At **BS-transitions (c=2→1), a mid-request finish/cancel**, or any rank drift, the comment (`:2237-2246`, `:2317-2345`) is explicit that skipping uid-intersection + num_tokens-max lets ranks issue **different subsequent TP forwards** → JACCL `LEN_ERR` cluster wedge. | **YES — arm can wedge the cluster** at/after BS-transitions |
| **B-only-off** (skip both) | **UNSAFE** | (a) `agree_on_tasks` is the *only* path that fills `_queue` (`:527`→`:184`); skipping it means **new submissions never get admitted → requests hang**, and any queued-but-unagreed task leaves ranks asymmetric → wedge. (b) `agree_on_cancellations_fast` is the *load-bearing* cancellation-observation point added 2026-08-09; skipping it **reintroduces the measured 133.7s/136.4s cancellation-latency bug** (`:699-718`) — the runner keeps serving a disconnected client. | **YES — hang risk on admission and on cancel** |
| **both-off** | **UNSAFE** (union of both above) | Combined failure modes; worst hang/corruption surface. | Highest |
| **baseline** | SAFE | — | none |

**Recommended safe ablation substitute** (if ablation is attempted at all):
skip *only the redundant fast-path `mx_any` gate* at `batch_generator.py:580`
(B2a) while keeping the load-bearing slow-path all_gather and the admission/
cancel paths — and only run it in **stable c=1 single-stream decode with no
mid-request cancels/finishes**. This isolates the marginal cost of the one
collective that fires every decode step without breaking any load-bearing
semantics. Even this must be paired with `EXO_RUNNER_HANG_TIMEOUT_SECONDS`
raised (the sync-profiling pitfall that can trip the watchdog) and
`EXO_PROFILER_SYNC_SPANS` left **unset** (it is not needed — direct timing
does not require it).

---

## 4. Pre-registered outcome bands (fixed thresholds)

Primary signal = mean per-call delta (µs) per collective, stratified by S1–S4,
compared against the decode-step wall (the `_next` inter-call period, ~16–40 ms
at 25–60 Hz) and against the ~1.2 ms/call prior reference.

| Outcome | Threshold (fixed in advance) | Attribution |
|---|---|---|
| **A costs** | `(A.all_sum + A.all_max) mean ≥ 600 µs/call` **AND** ≥ 10% of decode-step wall in any stratum S3–S4 | Candidate A (fenced upstream-sync all_sum+all_max) is a primary decoder-side collective cost → pursue coord-sync fusion or gate-skip only at stable c=1 |
| **B fast-path costs** | `(B1a + B2a) fast-gate mx_any mean ≥ 300 µs/call` **AND** ≥ 5% of decode-step wall | The every-step coord `mx_any` gate is the cost (not the rare all_gather) |
| **B slow-path costs** (rare) | `(B1b/B2c all_gather) mean ≥ 600 µs/call` **AND** its cumulative share of a full-request decode wall ≥ 10% | all_gather dominates only when cancels/task-admissions are frequent |
| **Neither costs** | `A_allsum+A_allmax+B_fast_mx_any combined < 500 µs/call` **AND** < 8% of decode-step wall | Neither coord candidate is the decode bottleneck → cost lives in the model forward's `moe.all_sum` (measured ~21% in the skill) — stop here |
| **Inconclusive** | strata S3–S4 means straddle a band boundary (±20% of the threshold) | do not claim attribution; re-run only the affected stratum |

Rules (pre-registered): decide on **S3–S4 means** (not S1 — shallow strata
underestimate per refinement #2). Any collective mean **< 200 µs** is treated as
noise regardless of band. No post-hoc threshold tuning after seeing S1.

---

## 5. Honest cluster-time estimate

| Option | Setup | Runs | Cluster-time estimate |
|---|---|---|---|
| **Direct timing (recommended)** | 1 env-gated instrumentation build + 1 deploy (`start_cluster.sh`, ~9 min) | 2–3 reps to reach S4 (each ~15–25 min at decode speed to 32K+ depth) | **~1.5–2 h** single session; zero run-failure risk (behavior unchanged when toggle off; when on only adds ~perf_counter) |
| **Ablation (fallback)** | 4 arms × (deploy ~9 min + run ~25 min) | 4 full runs | **~2.5–3.5 h** clean, **realistically 4–6 h** after the A-only-off / B-only-off / both-off wedge-recovery cycles (each recovery needs a full `start_cluster.sh` ~9 min), *plus* attribution ambiguity if any arm hangs instead of finishing |

### Recommendation
**Use direct per-collective timestamp deltas as the single instrument.** It is
strictly cheaper (1.5–2 h vs 4–6 h), zero-risk (no NOP of load-bearing
collectives), and — decisively for this pair — the ablation's arms are unsafe
(Section 3), so the ablation is not a viable *primary* even in principle. The
ablation is documented only as the fallback for the narrow case where direct
timing proves statistically indistinguishable, and only in its
**safe-substitute** form (skip only the redundant fast-path `mx_any`, stable c=1,
raised hang timeout, no `EXO_PROFILER_SYNC_SPANS`).

### Adjacent issues flagged (not in scope, triage-worthy)
- `batch_generator.py` carries **duplicated** `agree_on_tasks` /
  `agree_on_cancellations_fast` (SequentialGenerator `:159/:187/:208` vs
  BatchGenerator `:507/:536/:561`) with drifting docstrings (`:676` claims
  "utils_mlx.py:1102" for the mx_any short-circuit; it is actually at
  `utils_mlx.py:1757-1764`). Not the task's bug, but the stale line comment and
  the duplication will mislead the next instrumentation pass.
