# ROUND 9 / TASK 1 — Code analysis: why RV 200→0 measured −480 ms, not −200 ms

**Scope:** read-only source analysis of `~/repos/exo` @ `main` (`e3dbb6f65`). No cluster
commands, no source edits, no commits. Every mechanism claim below carries a `file:line`
I actually opened and read.

**Bottom line up front:**

> **The code predicts the RV=200 → RV=0 short-prompt TTFT delta is 200 ms.**

The rendezvous window is paid **exactly once per request**, is **deadline-bounded** (so the
loop physically cannot double-tick past the deadline), and the two ranks pay it
**concurrently, not serially** (each rank's deadline starts from its own local task arrival;
they only meet later, at a collective). There is **no second consumer** of the constant
anywhere in the repo, and **nothing downstream is quantized** by a poll/tick that the window
could push a request past. The −480 ms measurement is therefore **NOT explained by the
rendezvous alone**; §7 names what else is in that path and shows, from the round-7 result
JSONs, that the excess is distributed across a term the rendezvous cannot touch.

---

## 1. Every consumer of `EXO_BATCHED_PREFILL_RENDEZVOUS_MS`

Repo-wide grep for both the env-var string and the Python identifier
(`grep -rn "BATCHED_PREFILL_RENDEZVOUS"`, excluding `tmp/` and `.pyc`):

| # | Site | Kind |
|---|---|---|
| 1 | `src/exo/shared/constants.py:138-140` | **definition** — `int(os.getenv("EXO_BATCHED_PREFILL_RENDEZVOUS_MS", "200"))` |
| 2 | `src/exo/worker/runner/runner.py:16` | import (into `runner.py`'s top-level namespace) |
| 3 | `src/exo/worker/runner/runner.py:580` | **read #1** — the `> 0` enable gate |
| 4 | `src/exo/worker/runner/runner.py:582` | **read #2** — deadline arithmetic (`/1000.0`) |
| 5 | `src/exo/worker/runner/runner.py:624` | **read #3** — log-message interpolation only, inside `if extras_seen > 0` |
| 6 | `start_cluster.sh:136` | default `:=200` |
| 7 | `start_cluster.sh:1697` | unconditional propagation into `EXO_ENV` |
| 8 | `bench/trusted_measurement/fingerprint.py:95` | measurement fingerprint registry (records the value; does not consume it) |
| 9 | `src/exo/worker/main.py:377` | **comment only**, no read |

Docs/plans hits (`docs/PERFORMANCE_HISTORY.md:1245`,
`docs/prefill-throughput-breakthrough-2026-06-24.md:171,303`,
`docs/b2-quality-handoff-2026-06-24.md:147`,
`docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md:9031`,
`.hermes/plans/*`) are prose, not code.

**Conclusion (1):** there are **three** functional reads, all inside one `if` block in one
method (`Runner.handle_generation_tasks`), and one of them is a log string. There is no
second, hidden consumer that could charge the window a second time. **Anchor verified.**

Both outer gates are live in the measured config:
* `EXO_DSV4_BATCHED_PREFILL` — default `"1"` at `constants.py:115`, and forced on by
  `start_cluster.sh:129` (`: "${EXO_DSV4_BATCHED_PREFILL:=1}"`), propagated at
  `start_cluster.sh:1696`. **Anchor verified** (brief said `:129`/`:1696-1697` — correct).
* `EXO_MAX_CONCURRENT_REQUESTS` — `constants.py:108`, default `8`. `start_cluster.sh:2539`
  only exports it when already set, and the PP-forcing branches at `:2531`/`:2536` are gated
  on `DSV4_SHARDING=Pipeline`, whereas the default is `Tensor` (`start_cluster.sh:392`). So
  under the measured TP config the value is **8**, and `runner.py:596`'s
  `len(self.active_tasks) < 8` is trivially true at c=1 — the loop condition does not
  short-circuit the wait.

---

## 2. How many times is the window paid per request at c=1?

**Once.** The window lives on the *first* task's admission path only.

The runner's outer loop is `Runner.main()` (`runner.py:331-355`): it blocks on
`self._work_queue.get()` (`:335`) and hands the item to `handle_first_task` (`:345`), which
for a `TextGeneration` in `RunnerReady` calls `handle_generation_tasks(starting_task=task)`
(`runner.py:430-433`). That method:

1. acks + submits the starting task (`runner.py:566-569`),
2. **enters the rendezvous block once** (`runner.py:580-625`),
3. then enters `while self.active_tasks:` and calls `self.generator.step()`
   (`runner.py:643-647`).

The `step()` loop's own re-entry point for *later* work is a **non-blocking**
`self._work_queue.get_nowait()` (`runner.py:794`) — no timeout, no window. So a request that
arrives while another is in flight never pays the rendezvous at all; the window is charged
strictly to the request that transitions the runner Ready→Running. At c=1 (one request at a
time, the measured regime) that is every request, exactly once each.

**There is no second payment at a "batched-prefill rendezvous".** The engine-side batched
gate is `batch_generator.py:757` (`if agreed_slots > 1 and agreed_queue_len >= 2:`), reached
from `step()`. It contains no sleep and no timeout — it is a pure branch on two collective
`mx_min_int` reductions (`:753-756`). It never waits for a straggler.

### 2a. Serial across ranks (400 ms) or concurrent (200 ms)?

**Concurrent → 200 ms, not 400 ms.**

Each rank runs its own `Runner` process with its own `_work_queue`, fed by its own
`task-reader` thread from its own node's worker (`runner.py:255-288`, `:263`). Task dispatch
to the two nodes is issued **non-blocking** on the worker side —
`self._tg.start_soon(self._start_runner_task, task)` (`worker/main.py:383`), with the comment
at `worker/main.py:373-382` stating explicitly that a blocking `await` here used to serialize
c≥2 dispatch and was removed for exactly this reason.

So both ranks arm `rendezvous_deadline = now + W` (`runner.py:581-583`) at their own local
arrival times `t0` and `t1`, and both wait until `t_i + W`. The first point where the ranks
actually meet is the collective inside `step()` → `agree_on_tasks()`
(`batch_generator.py:683` → `:521-525`), which is an `mx_any` fast path plus, if any rank has
work, an `mx.distributed.all_gather` + set intersection
(`utils_mlx.py:2284-2338`; `agreed_ids = set.intersection(...)` at `:2335`). That is a
**barrier**, so the joint start-of-prefill is `max(t0, t1) + W`, i.e.

```
TTFT(W) = max(t0, t1) + W + prefill + first-token-emit
TTFT(200) − TTFT(0) = 200 ms      (the max() term is identical in both arms)
```

Nothing sums the two ranks' windows. **`agree_on_tasks` is independent of the window**
(anchor verified): the window only determines *what is already in `_maybe_queue`* when the
gather runs; the gather/intersection cost itself does not scale with `W`.

---

## 3. Is it a sleep or a `queue.get(timeout=...)` loop? Can it tick twice?

It is a `queue.get(timeout=...)` **inside a deadline-bounded `while`**, quoted verbatim from
`runner.py:594-620`:

```python
            extras_seen = 0
            while time.monotonic() < rendezvous_deadline and (
                len(self.active_tasks) < EXO_MAX_CONCURRENT_REQUESTS
            ):
                remaining = rendezvous_deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._work_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                # Stash non-generation items (PrefillTask, _TaskStreamClosed,
                # other commands) back into the queue so the existing main
                # loop handles them — only batch GenerationTasks at this
                # rendezvous point.
                if isinstance(item, GenerationTask):
                    if item.task_id in self.seen:
                        continue
                    self.seen.add(item.task_id)
                    self.acknowledge_task(item)
                    self.submit_generation(item)
                    extras_seen += 1
                else:
                    # Re-enqueue and exit rendezvous early so the loop can
                    # handle this item promptly.
                    self._work_queue.put(item)
                    break
```

**It can iterate more than once, but it cannot cost more than the window**, because the
timeout is recomputed each pass as `remaining = rendezvous_deadline - time.monotonic()`
(`:598`) against a deadline fixed *once* at `:581-583`. Total blocked time is bounded above
by `W` by construction. The two `continue` paths (duplicate `task_id`, `:610-611`) and the
`break` paths (empty at deadline `:603-604`; non-generation item `:619-620`) all either
shorten or terminate the wait.

At c=1 the concrete trace is: one `get(timeout=0.200)` → nothing arrives → `queue.Empty` →
`break`, after exactly 200 ms. **One tick, full window, no more.** The gate at `:580` is
`EXO_BATCHED_PREFILL_RENDEZVOUS_MS > 0`, a plain skip — `0` cleanly disables the whole block,
it is not a sentinel meaning "infinite" or "default". **Anchors verified** (`:580`, `:596`).

---

## 4. Is anything else serialized behind the window / quantized by a downstream tick?

I looked for every sleep/poll/timeout constant that sits *downstream* of the window on the
first-token path. **None of them is quantized in a way the window could push a request past.**

| Constant | Value | Where | On the post-window TTFT path? |
|---|---|---|---|
| worker planner tick | **100 ms** | `worker/main.py:195` (`await anyio.sleep(0.1)` in `plan_step`) | **UPSTREAM only.** It selects the task (`plan.py:49`, `_pending_tasks:346-390`) and dispatches it *before* the runner ever arms the window. It cannot be re-entered between window-close and prefill. |
| master plan loop | 10 s | `master/main.py:520` | No — instance/topology GC only. |
| JIT idle reaper | 5 s | `master/main.py:537` | No — gated on `jit_enabled()` (`:538`). |
| supervisor watchdog | 5 s tick | `runner/supervisor.py:445` | No — liveness only; `_check_hang` (`:451-…`) never gates dispatch. |
| SSE keep-alive | 10 s | `api/keepalive.py:12` (`interval: float = 10.0`) | No — `move_on_after(interval)` only *adds* a comment line on timeout; a real chunk is forwarded immediately (`keepalive.py:24-33`). |
| API cancel poll | 50 ms / 5 s | `api/main.py:908`, `CANCEL_ACK_TIMEOUT_SECONDS=5.0` at `:256` | No — cancel path only. |
| JIT placement poll | 2.0 s | `api/main.py:246`, used at `:1429` | No — only when no instance exists. |
| JIT load poll | 250 ms | `api/main.py:1358` | No — same, load path only. |
| prefill-server pickup | 3 s | `runner.py:70` (`PREFILL_PICKUP_TIMEOUT_SECONDS`) | No — disaggregation only, `ENABLE_DISAGGREGATION` default `false` (`constants.py:106`). |
| runner heartbeat throttle | 15 s | `runner.py:767` | No — status re-emit, post-admission. |
| cache-evict timing log | 50 ms | `engines/mlx/cache.py:106` | Diagnostic threshold only, gated on `EXO_CACHE_EVICT_TIMING_LOG` (`:105`, default `0`). |

The event path from the runner to the client is **push, not poll**, end to end:
`send_chunk` → `event_sender.send(ChunkGenerated(...))` (`runner.py:878`) → API
`_apply_state`'s `async for` over the event stream (`api/main.py:2453-2460`) →
`_text_generation_queues[...].send(chunk)` (`:2476`) → `_token_chunk_stream`'s
`async for chunk in token_chunks` (`api/main.py:961-963`) → SSE. There is no timer between
prefill completion and the first token reaching the client that the window could desynchronize.

**Conclusion (4): nothing is serialized behind the window, and no downstream quantum exists
to be missed.** A 200 ms shift in dispatch time produces a 200 ms shift in TTFT, not a
200 ms + one-quantum shift.

---

## 5. Does decode/streaming add a per-request constant that scales with how late prefill started?

**No.** Every per-request timing anchor is captured *relative to the request's own progress*,
never against an absolute schedule:

* prefill wall starts at `start_time = time.perf_counter()` **inside** `prefill()`
  (`generator/generate.py:827`) and `elapsed`/`tokens_per_sec` are computed at `:975-976` —
  both after the window.
* decode's anchor is `generation_start_time=time.perf_counter()` captured at admission
  (`batch_generate.py:2776`) and `generation_time_at_start=_mlx_gen_elapsed_seconds(...)`
  (`:2782`), i.e. also after prefill.
* the cancellation counter `check_for_cancel_every` is a **token count**, not a clock
  (`batch_generator.py:342-352`), and the fast per-step cancel check
  (`agree_on_cancellations_fast`, `:567-595`) is one `mx_any` on the coord subgroup.
* the first token is emitted the moment `step()` returns it (`runner.py:715-724`).

Nothing reads wall-clock-since-admission and rounds it, sleeps to a boundary, or retries on a
fixed period. **A later prefill start does not amplify.**

---

## 6. THE NUMERIC PREDICTION

> ### The code predicts the RV=200 → RV=0 short-prompt TTFT delta is **200 ms**.

**Derivation.** The constant has exactly three functional reads, all in
`Runner.handle_generation_tasks` (`constants.py:138-140` → `runner.py:16`, `:580`, `:582`,
`:624`), and only two of those affect control flow. At c=1 with
`EXO_DSV4_BATCHED_PREFILL=1` (`start_cluster.sh:129`) and `EXO_MAX_CONCURRENT_REQUESTS=8`
(`constants.py:108`; the PP-forcing branches at `start_cluster.sh:2531/2536` don't fire
because `DSV4_SHARDING` defaults to `Tensor` at `:392`), the gate at `runner.py:580` is open
and the loop condition at `:596` is true. The loop arms a single deadline at `:581-583`
(`monotonic() + W/1000`) and blocks in `self._work_queue.get(timeout=remaining)` (`:602`)
with `remaining` recomputed from that *fixed* deadline each pass (`:598`); with no second
request in flight, the first `get` raises `queue.Empty` at the deadline and `break`s
(`:603-604`). Cost per request: exactly `W`, once, and structurally bounded by `W` even if
the loop iterates. Each rank arms its own deadline from its own non-blocking dispatch
(`worker/main.py:383`) and the ranks first meet at the `agree_on_tasks` all_gather barrier
(`batch_generator.py:683` → `:521-525` → `utils_mlx.py:2284-2338`), so the joint prefill
start is `max(t0,t1) + W` — the windows overlap rather than sum, giving **W, not 2W**
(200, not 400). Nothing downstream is quantized (§4) and no decode/stream constant scales
with prefill start time (§5). Therefore `TTFT(200) − TTFT(0) = 200 − 0 = **200 ms**`.

**I could not find any code path that yields 400 ms or 480 ms.** I specifically looked for
and ruled out: a second window read (§1), a second payment per request (§2), a serial
per-rank pay (§2a), a multi-tick loop (§3), a missed downstream quantum (§4), and a
late-start-amplifying decode constant (§5).

---

## 7. So what is the other ~280 ms? (labelled: EVIDENCE-BACKED DECOMPOSITION + SPECULATION)

Since the code says 200 and the measurement says 480, the difference must live in a term the
rendezvous does not control. I decomposed round 7's own §2.3 result files
(`tmp/perf-campaign-2/round7/results/{Z,P2}_short_r*.json`, n=10 each) using the
server-side stat that is timed *inside* the generator and therefore **excludes the window
entirely**: `prompt_tps = state.prefill_tps` (`batch_generate.py:4598`), whose numerator is
`num_tokens` and denominator is `perf_counter()` measured strictly inside `prefill()`
(`generate.py:827`, `:975`, `:967`). Reconstructing `prefill_elapsed ≈ (prompt_tokens−1) /
prompt_tps` and subtracting it from the client-side `prefill_s` (= TTFT; `t_first − t_start`,
`bench/long_decode_probe.py:147,160`) splits each rep into two buckets:

| bucket | Z (RV=0) median | P2 (RV=200) median | delta |
|---|---|---|---|
| **TTFT** (client) | 1510 ms | 1990 ms | **+480 ms** |
| **in-`prefill()` compute** (server-timed, window-excluded) | 1027 ms | 1296 ms | **+269 ms** |
| **residual** (dispatch + window + post-prefill + emit) | 441 ms | 726 ms | **+285 ms** |

(Excluding P2's 3980 ms outlier: TTFT +430, in-prefill +257, residual +279.)

Two things follow, and they matter:

1. **The residual bucket — the only bucket the window can be in — moved +285 ms**, which is
   the 200 ms prediction plus ~85 ms of ordinary run-to-run spread (Z's own residual spans
   412–577 ms across its 10 reps). **That is consistent with the 200 ms prediction.**
2. **Roughly 270 ms of the 480 sits in `prefill()` compute itself** — a code region the
   rendezvous window provably cannot enter, since the window closes before `step()` is ever
   called (`runner.py:580-625` precedes `:643-647`). Z's in-prefill median is 1027 ms for a
   ~226-token prompt; P2's is 1296 ms for the same prompt sizes (220–234 tokens both arms,
   `prefix_cache_hit = "none"` in all 20 reps). **Same code, same prompt, 26% more compute
   time.** No rendezvous mechanism explains that.

**SPECULATION (not established from code, offered as the ranked hypothesis list for a
follow-up measurement):** the +269 ms in-prefill gap is almost certainly a **boot/warm-state
confound, not a rendezvous effect**. Round 7's own report already flags the arms were on
different boots — §2.2 records arm P on a 4.5-h-old boot vs arms Z and P2 immediately after
relaunch, and observes arm Z's A1 width was ~5× arm P's on identical hardware
(`round7/REPORT.md:79-95`). Candidate contributors, all *outside* the rendezvous:
   * MLX allocator state: the runner calls `gc.collect()` + `mx.clear_cache()` on every
     idle transition (`runner.py:848-856`, `_RECLAIM_ON_IDLE` default on at `runner.py:84`),
     so first-request-after-idle prefill re-grows the pool — and the short-prompt reps are
     precisely the idle-gap-dominated regime.
   * KV-prefix-cache bookkeeping: these reps are non-bench (`long_decode_probe.py:122` posts
     to `/v1/chat/completions`, so `task_params.bench` is False), which enables
     `_save_prefix_cache` + `_evict_if_needed` on the post-prefill/pre-first-token path
     (`batch_generate.py:2586-2591`, `cache.py:1906` `_evict_if_needed`) — a real per-request
     cost in the residual bucket whose magnitude depends on accumulated trie state, i.e. on
     **how many requests that boot has already served**. Arm ordering, not the window.

**Recommendation for round 9's measurement design:** a paired-boot, interleaved A/B (both
arms on the same boot, alternating reps) with the decision statistic taken on the
**residual** bucket (`TTFT − (prompt_tokens−1)/prompt_tps`) rather than raw TTFT. That
isolates the term the code says the window lives in and cancels the in-prefill confound
that currently contributes ~270 of the unexplained 280 ms.

---

## 8. Honest limits of this analysis

* I did **not** run the cluster, relaunch anything, or edit any source file. The only file
  created is this document.
* §7's bucket split is arithmetic on **round 7's existing result JSONs**, not a new
  measurement; `prefill_elapsed` is *reconstructed* from `prompt_tokens / prompt_tps` and so
  inherits any imprecision in that ratio (the numerator convention — `num_tokens` passed to
  `prefill()` is `prompt_tokens[:-1]`, `batch_generate.py:2561` — is why I used
  `prompt_tokens − 1`; using the raw count shifts each bucket by <5 ms and changes no
  conclusion).
* The §7 causal attribution of the in-prefill gap is explicitly labelled **speculation**. What
  is *established* is the negative: that ~270 ms of the 480 lies in a code region the
  rendezvous window cannot reach.
* I did not audit the mlx / mlx-lm submodules below `prefill()`; the claim there is only that
  the window's value never crosses that boundary (it is read at three sites in `runner.py`
  and nowhere else — §1).
