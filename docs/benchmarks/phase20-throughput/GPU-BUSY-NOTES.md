# Phase 0d — GPU-busy tool: design, classification, and retune instructions

Tool: `bench/phase20_gpu_busy.py` (stdlib only; run with the shared venv python
`/Users/adam.durham/repos/exo/.venv/bin/python`).  Tests:
`bench/phase20_tests/test_phase20_gpu_busy.py` (offline, real idle fixtures +
synthetic decode fixtures).  This worker built and unit-tested everything
**offline**; the live `decode`/`idle` windows are run by the campaign PM.

## 1. What 0d measures

Three windows, on **both nodes simultaneously**: idle baseline; streaming
decode *benign*; streaming decode *agentic*.  The decode window starts only
**after the first SSE token** (content or reasoning_content) of a decode-only
(prefix-cached) request,
so prefill cannot contaminate it.  Two independent samplers run in that window:

* `sudo -n powermetrics --samplers gpu_power -i 500 -n <N>` → `GPU HW active
  residency` %; `GPU-busy = mean(decode residency) − mean(idle residency)`.
* `/usr/bin/sample <runner-pid> <secs+5> 1 -mayDie -file /tmp/…` (py-spy is NOT
  installed) → call-graph sample counts classified into
  **host-wait-on-GPU / host-Python-busy / comm-wait**, per node.

`GPU-busy` is reported per node; the frequency-bin and P-state distributions
are reported alongside (low P-states during decode = latency-bound signature).

## 2. powermetrics parsing design

* A capture is a sequence of `*** Sampled system activity (…) (NNN ms
  elapsed) ***` blocks.  **Each block is weighted by its `elapsed` ms** for
  every mean (a run has many blocks; no block is special).
* `GPU HW active residency: X% (338 MHz: .21% …, 1182 MHz: 0% 1182 MHz: 0% …)`
  is parsed as an **ORDERED LIST**, never a dict: the label `1182 MHz` occurs
  twice in the real M4 Max residency line, and percentages use a leading dot
  (`.21%`).  Bins are aligned across blocks by position, then elapsed-weighted.
* Also parsed: `GPU HW active frequency`, `GPU SW requested state` (P1..P15),
  `GPU SW state` (SW_P1..SW_P15), `GPU idle residency`, `GPU Power` (mW).
* The **first block of a run is additionally reported on its own**
  (`first_block_hw_active_residency_pct`) because it can be unrepresentative;
  it is still counted in the weighted mean.

Real idle fixtures (what the tool prints):

| node | n_blocks | mean hw-active residency | first block | mean freq | mean idle | mean power |
|---|---|---|---|---|---|---|
| m4-1 | 3 | **3.978 %** | 3.75 % | 768.68 MHz | 96.022 % | 12.0 mW |
| m4-2 | 3 | **3.713 %** | 3.48 % | 768.00 MHz | 96.287 % | 10.0 mW |

## 3. `sample` call-graph parsing + classification

Parser: the file is split at `Call graph:` → `Total number in stack` →
`Sort by top of stack` → `Binary Images:`.  Inside the call graph each thread
tree starts at a root line (indent, count, symbol beginning `Thread_<id>`);
children are indented with `+` (and `!` for a diverging sub-path).  Relative
depth = `(indent_len − base_indent) // 2`.  Each root owns the following lines
until the next root; the `Thread_<id>` id is the thread key.

Frame symbols are normalised: drop the `[0x…]` address (incl. multi-address
`[0x…,0x…]`), the `(in lib…)` clause, and the trailing byte-offset.  A frame is
a **leaf** when no deeper frame follows it in its thread tree.

### Classification (per leaf stack path)

For every leaf stack path we take the **deepest frame that matches**, walking
root → leaf.  The category config is a JSON of `regex → category`
(`DEFAULT_CLASSIFY_CONFIG`, overridable with `--classify-config`):

* **`comm`** — `jaccl`, `ibv_*`, `rdma|RDMA`, `libmlx5|librxe|libthunderboltrdma|…`,
  `AllReduce|all_sum|collective|send_recv|poll_cq|ib_qp` …
* **`gpu_wait`** — `mlx::core::eval|eval_impl`, `Scheduler::wait`,
  `metal::*`/`MTL*`/`AGX`/`IOGPU`, `StreamThread`, `mach_msg`,
  `command_buffer|waitUntilCompleted`.  Weak signals (`__psynch_cvwait`,
  `_pthread_cond_wait`, `semaphore`, `__semwait_signal`,
  `std::condition_variable`) count as `gpu_wait` **only when an anchor frame**
  (a real MLX/Metal frame) is also present on the path.
* **`python_busy`** — interpreter-ACTIVE frames only: `_PyEval_EvalFrameDefault`,
  `PyEval_EvalCode`, `PyObject_*`, `_PyObject*`, `PyDict_*`, `PyUnicode_*`,
  `take_gil`, `pymalloc`, `gc_collect` …
* **`other`** — anything else, including threads that end on a **blocking
  primitive** (`__psynch_cvwait`, `__workq_kernreturn`, `__semwait_signal`,
  `nanosleep`, `read`, `mach_msg`, `kevent`, `poll`).  A blocked thread is
  **not** Python-busy, so a leaf ending on a blocking primitive is
  short-circuited to `other` — this deliberately prevents an *ancestor*
  `_PyEval_EvalFrameDefault` frame from stealing a parked thread's samples.

Ties for a single frame are broken by `precedence` (`comm` > `gpu_wait` >
`python_busy`).  Fractions are computed on the **Python main thread** (the
thread whose root carries `Py_RunMain`) and on **all threads**; a per-thread
table lists each thread's sample total, category fractions and top-5
leaf/top-of-stack frames.  **Unclassified top frames are printed** in
`unclassified_top_frames` so nothing is silently dropped, and the
`Sort by top of stack` collapsed counts are surfaced verbatim.

Real idle fixtures (all-wait, as expected): 57 threads, ~127 k samples,
**main thread 100 % `other`**, all-threads ~94.7 % `other` + 5.3 % `gpu_wait`
(`StreamThread::thread_fn` parked), `python_busy = 0 %`.  Live smoke (idle
runner, 5 s) reproduces this on both nodes.

## 4. CLI

```
python bench/phase20_gpu_busy.py parse-pm     FILE [--idle FILE] [--md FILE]
python bench/phase20_gpu_busy.py parse-sample FILE [--classify-config F] [--md FILE]
python bench/phase20_gpu_busy.py idle         [--secs 50] [--out FILE] [--dry-run]
python bench/phase20_gpu_busy.py decode --workload {benign,agentic} \
        [--depth 100000] [--secs 50] [--max-tokens 2600] [--own-registry PATH] [--dry-run]
python bench/phase20_gpu_busy.py report       [--idle F] [--benign F] [--agentic F]
python bench/phase20_gpu_busy.py fallback-b   [--pid N] [--out NAME]
```

All four offline parsers/reporters are safe to run anywhere.  `idle` and
`decode` touch the cluster: `idle` only reads `powermetrics`; `decode` is the
PM-owned live path (guarded, needs `bench/phase20_guard.py`).  The decode
sequence (plan in `--dry-run`): canary ok → cold feed (prefix build, **not**
measured) → the **exact same prompt** re-fed (expect `turn reuse: … prefill=0`)
→ on the **first SSE token** (content **or** reasoning_content — this model
streams `delta.reasoning_content`) start powermetrics + `sample` on both
nodes → stream until the window ends → verify the window lay inside
`[first token, last token]` (else the run is marked INVALID) → scp back →
parse → `docs/benchmarks/phase20-throughput/raw/gpu_busy.<workload>.json` + md.

## 5. Decision gate (PREREG 0d, frozen — `decide()`)

* GPU-busy ≥ **90 %** both nodes AND host-Python < **5 %** → `GPU_SERIALIZED`
  (PROF Mode-2 matters most).
* GPU-busy < **85 %** on either node → `HOST_COMM_BOUND` (Phase-1 timer
  mandatory).
* node gap > **10 points** → `RANK_IMBALANCE` (one rank waiting on the other;
  jaccl / load imbalance).
* **falsifier**: residency within **5 points of idle** on either node →
  `FALSIFIER_STOP_FALLBACK_B` (wrong GPU/process; stop, switch to Fallback B).
* A sampling window that did not lie inside the stream interval →
  `INVALID_WINDOW`.
* Otherwise `INCONCLUSIVE`.  Primary verdict priority: falsifier > invalid
  window > gpu-serialized > host/comm > rank-imbalance > inconclusive.

`decode` GPU-busy uses the **mean of the decode arms' mean residency** minus
idle, per node (benign and agentic averaged when both are supplied).

## 6. Retuning after real decode samples

The idle fixtures are all-wait; the real decode `sample` shape will differ
(the Python main thread should show `python_busy` and the MLX/jaccl threads
should widen `gpu_wait`/`comm`).  Retune **without touching code** via
`--classify-config`:

1. Run `parse-sample` on a real decode capture and read
   `unclassified_top_frames` — anything large there is a frame the config
   doesn't yet recognise.
2. Either add its regex to the right category in a copy of
   `DEFAULT_CLASSIFY_CONFIG`, **or** narrow a category that over-fires.  If a
   real MLX path ends on a plain `__psynch_cvwait` with no anchor on the same
   path, add the missing MLX frame to `gpu_wait.anchors` (the weak
   cond-wait patterns then count on that path).
3. Re-run `parse-sample --classify-config tuned.json` and confirm
   `unclassified_top_frames` shrank while the per-thread table still sums to
   `total_samples`.  Commit the tuned JSON under `fixtures/gpu_busy/` for
   reproducibility.

## 7. Fallback B (falsifier path)

If residency is within 5 points of idle on either node the measurement is on
the wrong GPU/process; stop and capture a Metal System Trace instead.  The tool
only *constructs* (never runs) the command, per node:

```
ssh <node> "xcrun xctrace record --template 'Metal System Trace' \
  --attach <runner-pid> --time-limit 30s --output /tmp/<name>.<tag>.trace \
  --no-prompt && xcrun xctrace export --input /tmp/<name>.<tag>.trace --toc"
```

`xctrace` 27.0 exists on the nodes; if the template is unavailable, fall back
to the `sample`-only host-wait proxy (already implemented).

## 8. Verification status

* Offline: `24 passed` against the real idle fixtures + synthetic decode
  fixtures; two tests sabotage-proven (naive-mean vs elapsed-weighted mean;
  removing the blocking short-circuit reclassifies idle threads as
  `python_busy`).
* Live smoke (read-only, once): `powermetrics` 20 blocks on both nodes parsed
  (m4-1 3.93 %, m4-2 3.48 % mean residency); 5 s `sample` on both live runner
  pids parsed (57 threads, main thread 100 % `other`, both nodes).
* **NOT verified**: a real **decode-shaped** `sample` capture (host busy on the
  MLX path) was not available to this worker — the decode classification
  fractions are exercised only against the hand-made synthetic fixture.  The
  `decode` live path itself (SSE start trigger, window-validity, guard
  integration) is untested against the cluster by design; the PM runs it.

## FIX 1 — registry wiring + reasoning_content first-token (branch `p20/gpubusy`)

Two defects root-caused by the PM from live data; both fixed, offline-tested.

**FIX 1a — persistent own-request registry.** `cmd_decode` constructed
`ChunkGuard(label, max_wall_s, log_dir)` with **no** `own_requests` and **no**
`registry_path`.  The guard's entry idle-check refuses to start if any
generation POST on a node is newer than `min_idle_s` (600 s) unless it matches a
registered own request — so this tool's own prior POSTs (or a sibling phase20
tool's, in its own process) stalled every run ~10 min.  Now `cmd_decode`:
loads the registry with a local `load_own_requests()` (copy of the
`phase20_delta_ladder.py` helper — JSONL `{"label","t"}`, missing file → `[]`,
garbage/blank lines skipped) and passes **both** `own_requests=<epochs>` and
`registry_path=<file>` to `ChunkGuard`; `--own-registry PATH` overrides the
default `<outdir>/raw/own_requests.jsonl`.  Every request (`g.register_own_request()`
before the warmup feed **and** before the measured feed) keeps appending to it.
No recorded JSON keys changed.

**FIX 1b — `reasoning_content` first-token.** `stream_once` watched only
`delta.content`; this reasoning model streams its output in
`delta.reasoning_content`, so `ttft`/`first`/`last` stayed `None`, the samplers
never launched, and window-validity read INVALID.  Now it accumulates
`delta.content or delta.reasoning_content` and fires `on_first_token` on the
first non-empty delta of **either** kind (mirrors
`phase20_delta_ladder.py`/`phase20_common.py:stream_once`).  Returned key set
unchanged.

Tests added (5): reasoning-only stream fires + non-None `ttft`; content+
reasoning both accumulate (fires once); `cmd_decode` loads the registry and
passes non-empty `own_requests` + `registry_path`; `--own-registry` override;
None-ttft stream does not crash window-validity; loader tolerates
missing/garbage.  Suite: `30 passed` (was 24).  Pre-fix at HEAD, 5 of the new
tests fail/error.

**NOT verified**: no live smoke of the fixed decode path — `bench/phase20_guard.py`
is not present in this worktree (the live path needs it; the earlier branch commit
shipped without it), so `decode` exits 2 with `phase20_guard unavailable` before
any cluster contact.  The PM's worktree (which has the guard) must run the smoke.
The registry wiring is asserted only against a fake guard; the real
`ChunkGuard(own_requests=…, registry_path=…)` call path is exercised by the PM.

## FIX 2 — non-blocking samplers + first-token callback signature (branch `p20/gpubusy`)

Two defects root-caused by the PM from the first live `decode` run; both fixed,
offline-tested.

**FIX 2a — callback signature mismatch (live crash).** `stream_once()` invokes
its callback as `on_first_token(now, resp)` (two args), but `cmd_decode`'s
`start_samplers(first_epoch)` accepted **one** → `TypeError:
cmd_decode.<locals>.start_samplers() takes 1 positional argument but 2 were
given`, raised at the first SSE token (gpu_busy.py ≈992/1013).  The contract is
now uniform: **`on_first_token(first_epoch, resp)`** at the call site and in
every callback; `start_samplers` is defined `def start_samplers(first_epoch,
resp=None)` (the `resp` is accepted and unused — kept `None`-default so an
internal 1-arg call would still work, but `stream_once` always passes both).

**FIX 2b — samplers blocked the SSE read loop.** `start_samplers` ran the
blocking ssh `powermetrics`/`sample` captures **inline** (it started the per-node
threads and immediately `join()`ed them, waiting `secs + 60` s) — but it is
called from *inside* the SSE read loop, so it stalled token reading for the
whole sampler window and corrupted the decode timing / window-validity.  Now the
inline callback does only the cheap, non-blocking work: it records the
first-token epoch into `node_out` for both node tags and **spawns** one
background `_sampler_one` thread per node (daemon; captures + writes the raw
files).  The callback returns immediately so `stream_once` keeps reading at full
rate.  After the stream ends (still inside the guard, before parsing) the main
thread **joins the background threads with a bounded timeout**
(`secs * 3 + 120` s) and only then computes window-validity and parses.  A
capture that overran its bound is left `_pending` and simply contributes no
parsed entry — the old per-node results dict / recorded JSON schema are
unchanged (`node_out[tag]` keys: `first_token_epoch`, `_pending`, `pid`,
`powermetrics`, `sample`; `rec["decode"]` key set unchanged).

Window-validity is still `window_inside_stream(pm.start_epoch, pm.end_epoch,
first_token_epoch, last_token_epoch)` per node → `window_valid[tag]`, i.e. the
sampler window must lie inside `[first token, last token]`.

Tests added (3, all driving the real `cmd_decode` with a fake guard + fake ssh):
(a) `stream_once`-style callback called with **two args** does not raise, records
the epoch and returns promptly (`< 0.2 s` while the faked capture sleeps `0.5 s`),
with both nodes' captures spawned; (b) background sampler results are collected
after the stream ends and parsed into `rec["parsed"]`, and `rec["decode"]`'s key
set is unchanged; (c) window-validity end-to-end (inside → True, starts-in-prefill
→ False).  Suite: **33 passed** (was 30).  Pre-fix at HEAD the same three fail
(the signature one reproduces the exact live `TypeError`).

**NOT verified**: the same caveat as FIX 1 — the live window needs
`bench/phase20_guard.py`, which this worktree lacks, so the real cluster window
(SSE trigger against a live decode, real ssh capture, real window timing) is run
by the PM; the callback/threading/join logic is exercised only against fakes.
Note the join is best-effort: a capture that overruns its bound leaves that
node's entry `_pending` (no parsed data) rather than corrupting the record.

## FIX 3 — window-validity epoch UNITS: monotonic vs wall clock (branch `p20/gpubusy`)

**Symptom.** A live decode returned `window_valid={"m4-1": false, "m4-2": false}`
even though the 20.4 s sampler window plainly sat inside the 21.8 s decode.

**Root cause — unmatched clocks.** `stream_once` stamped
`first_token_epoch`/`last_token_epoch` with `time.perf_counter()` (a
**monotonic** clock, ~seconds since boot; the live value was ≈648044 s), while
`_ssh_capture` stamps the sampler capture's `start_epoch`/`end_epoch` with
`time.time()` (**wall clock**, ≈1.79e9 = 2026 epoch seconds).  `window_valid`
calls `window_inside_stream(pm.start_epoch, pm.end_epoch, first_token_epoch,
last_token_epoch)`, comparing a ~6.5e5 value against a ~1.79e9 value: the
decode end could never be ≥ the window start, so the check was structurally
incapable of returning `True`.

**Fix.** In `stream_once`, stamp the emitted epochs with `time.time()`:
`now_epoch = time.time()` is used for `first`/`last_token_epoch` and passed to
`on_first_token(now_epoch, resp)`; `now = time.perf_counter()` is retained
purely for the *durations* (`ttft = now - t0`, `wall = time.perf_counter() -
t0`), which are differences and unit-agnostic.  The sampler capture side
(`_ssh_capture`, wall clock) is unchanged.  Recorded JSON schema keys are
unchanged (`rec["decode"]` and `rec["window_valid"]` shapes identical).

**Tests added (3).** (a) `test_stream_once_epochs_are_wall_clock` — the units
test: `stream_once`'s emitted `first_token_epoch`/`last_token_epoch` must lie
within `[time.time() before, time.time() after]` (fails pre-fix with
`first_token_epoch 648044.3` vs wall `1791436444.6`); (b)
`test_window_inside_same_unit_wall_clock_magnitudes` — with real wall-clock
magnitudes, a window strictly inside `[A,B]` (A<a<b<B) → `True`, an end-overhang
→ `False`, a start-overhang → `False`; (c)
`test_decode_window_validity_wall_clock_end_to_end` — the real `cmd_decode`
with wall-clock epochs on both sides yields `{"m4-1": True, "m4-2": False}`.
Suite: **36 passed** (was 33); the units test fails pre-fix, the other two are
invariant under the unit swap (they pin the comparison semantics).

**NOT verified**: as FIX 1/2 — the live cluster window needs
`bench/phase20_guard.py` (absent in this worktree), so the real SSE-triggered
capture is run by the PM; the unit fix and the comparison are exercised offline
only.  The live run must be re-checked to confirm `window_valid` now reports
`true` for a genuinely-inside window.
