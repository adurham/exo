# EXO_BATCHED_PREFILL_RENDEZVOUS_MS read-through (round7, taskA)

Scope: pure code reading, no execution. Repo: `/Users/adam.durham/repos/exo`, live code
under `src/exo/` (build/ mirrors ignored).

## (a) Is it a "wait for additional concurrent tasks" batching window?

Yes. Defined at `src/exo/shared/constants.py:138-140`:

```
EXO_BATCHED_PREFILL_RENDEZVOUS_MS = int(
    os.getenv("EXO_BATCHED_PREFILL_RENDEZVOUS_MS", "200")
)
```

Consumed in `src/exo/worker/runner/runner.py:559-625`, inside
`handle_generation_tasks`, immediately AFTER the first (`starting_task`) has been
acknowledged and submitted (`runner.py:566,569`). The loop/sleep that implements
the wait:

```python
# runner.py:580-604
if EXO_DSV4_BATCHED_PREFILL and EXO_BATCHED_PREFILL_RENDEZVOUS_MS > 0:
    rendezvous_deadline = (
        time.monotonic() + EXO_BATCHED_PREFILL_RENDEZVOUS_MS / 1000.0
    )
    ...
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
        if isinstance(item, GenerationTask):
            if item.task_id in self.seen:
                continue
            self.seen.add(item.task_id)
            self.acknowledge_task(item)
            self.submit_generation(item)
            extras_seen += 1
        else:
            self._work_queue.put(item)
            break
```

This is a blocking drain of `self._work_queue` with a timeout bounded by
`rendezvous_deadline`, i.e. exactly a rendezvous window that lets additional
concurrent `GenerationTask`s arrive and get submitted (`submit_generation`,
line 614) alongside the first one before the runner proceeds to
`self.generator.step()` (`runner.py:643,647`). The docstring comment at
`runner.py:571-579` confirms intent: <cite index="0-0">"Rendezvous window for batched prefill: when EXO_DSV4_BATCHED_PREFILL is on, drain the work queue briefly BEFORE the first step() call so any concurrent c=2+ requests can land in the engine's queue at the same step() iteration as starting_task."
Also mirrored at the dispatch site `src/exo/worker/main.py:373-382` (comment
only, not part of the wait implementation itself).

## (b) At concurrency=1, is the window PURE added latency with no other functional effect?

Yes — confirmed by reading the loop body and its surrounding scope.

- The loop body's only side effects are: pulling items off `self._work_queue`
  (`runner.py:602`), and for `GenerationTask` items, `acknowledge_task` +
  `submit_generation` (`runner.py:613-614`), plus incrementing a local counter
  `extras_seen` (`runner.py:615`) used only for a log line at
  `runner.py:621-625`. At c=1 there is nothing else in `self._work_queue`, so
  `self._work_queue.get(timeout=remaining)` blocks for the full window and then
  raises `queue.Empty`, hitting `break` (`runner.py:603-604`) — no task
  processing happens inside the loop at c=1.
- Nothing downstream reads elapsed rendezvous time. `handle_generation_tasks`
  proceeds directly from the loop (or the `if` block being skipped) into the
  main step loop `while self.active_tasks:` at `runner.py:643`, which calls
  `self.generator.step()` (`runner.py:647`) — no argument or state is
  conditioned on how long the rendezvous actually took, only on which tasks
  got submitted before it exited.
- Warmup/cache-priming is a *separate*, unconditional code path:
  `BatchGenerator.warmup()` (`batch_generator.py:134-148` and the duplicate at
  `:480-494`) calls `warmup_inference(...)`, `self.agree_on_tasks()`,
  `self.agree_on_cancellations()`, and `prewarm_coord_group(self.group)` at
  model load time — none of these are gated on or scheduled relative to
  `EXO_BATCHED_PREFILL_RENDEZVOUS_MS`; `warmup()` runs once before any
  `handle_generation_tasks` call and has no reference to the constant.
- Cross-rank sync (`agree_on_tasks`, see part d) is a separate, independently
  invoked *blocking collective* call inside `BatchGenerator.step()`
  (`batch_generator.py:682-683`), not something the rendezvous *sleep itself*
  feeds into — the rendezvous window only affects how many tasks are already
  sitting in `_maybe_queue` by the time that collective runs, not whether/how
  the collective executes.

So at c=1, the window's only observable effect is added wall-clock time before
`generator.step()` is first called for that task — pure TTFT latency, confirmed
by the comment at `runner.py:578-579`: <cite index="0-1">"Latency cost: EXO_BATCHED_PREFILL_RENDEZVOUS_MS added to c=1 first-token times when batched prefill is on."

## (c) What happens at exactly value 0?

The wait is skipped cleanly via an explicit guard, not a sentinel:

```python
# runner.py:580
if EXO_DSV4_BATCHED_PREFILL and EXO_BATCHED_PREFILL_RENDEZVOUS_MS > 0:
```

When `EXO_BATCHED_PREFILL_RENDEZVOUS_MS == 0`, this condition is `False`
regardless of `EXO_DSV4_BATCHED_PREFILL`, so the entire block
(`runner.py:580-625`, including `rendezvous_deadline` computation and the
`while` loop) is never entered. Control falls straight through to
`runner.py:627` onward (the `MLX_GPU_TIME` probe setup) and then the main
`while self.active_tasks:` loop at `runner.py:643`. There is no other branch
in `runner.py` or `constants.py` that treats 0 as a special/sentinel value
(e.g. "wait forever" or "disable batched prefill entirely") — `0` only fails
the `> 0` comparison. This matches the constants.py comment
(`constants.py:136-137`): <cite index="0-2">"Set to 0 to disable rendezvous (per-task path even with batched prefill enabled)."
— verified true by the `> 0` guard, not merely asserted by the comment.

Note `EXO_DSV4_BATCHED_PREFILL` itself is untouched by this value — batched
prefill can still fire later if multiple tasks happen to land in
`self.active_tasks`/`_maybe_queue` before the next `step()`'s
`agree_on_tasks()` collective; setting the window to 0 just removes the
runner's deliberate wait for that to happen, it does not disable the
downstream batching gate in `BatchGenerator.step()`.

## (d) agree_on_tasks — cross-rank agreement mechanism, and does window=0 risk disagreement at c=1?

Two near-identical implementations exist (PP-engine and TP/ExoBatchGenerator
variants) at `batch_generator.py:159-185` and `batch_generator.py:505-535`;
mechanism is the same in both.

**Mechanism** — deterministic collective over a *serialized task-id list*, not
a time-windowed snapshot and not a leader broadcast:

1. Fast-path gate: `coord = get_coord_group(self.group)`; `if not
   mx_any(len(self._maybe_queue) > 0, coord): return` (`batch_generator.py:175-177`,
   `521-523`). This is itself a blocking collective (`mx_any`) — every rank
   must call `agree_on_tasks()` in lockstep or the other rank deadlocks
   waiting inside the collective, per the caller-site comment at
   `batch_generator.py:676-681`: <cite index="0-3">"agree_on_tasks() is a collective (mx.distributed.all_gather). Both ranks must call it together — gating on per-rank self._queue lets one rank skip it while the other waits forever inside, deadlocking the cluster on the next iteration's all_reduce."
2. If any rank has pending items, `mx_all_gather_tasks(self._maybe_queue, coord)`
   is called (`batch_generator.py:178`, `525`), implemented at
   `utils_mlx.py:2249-2340`. This all-gathers each rank's local task-id list
   (UUID bytes, padded to `max_tasks`) via `mx.distributed.all_gather`
   (`utils_mlx.py:2284-2287,2324-2329`), decodes every rank's task-id set
   (`utils_mlx.py:2330-2333`), and computes
   `agreed_ids = set.intersection(*(set(tids) for tids in all_task_ids))`
   (`utils_mlx.py:2335`). Only task ids present in **every** rank's local
   queue at the moment of the call end up in `agreed`
   (`utils_mlx.py:2337-2338`); ids not yet seen by all ranks are returned in
   `different` (`utils_mlx.py:2339`) and stay in `self._maybe_queue`
   (`batch_generator.py:185`, `529`) to be retried on the *next* `step()`'s
   `agree_on_tasks()` call.
3. Because `all_gather` is a synchronous, blocking collective, both ranks
   compute the identical `agreed_ids` set from the identical gathered data —
   there is no local time-based decision embedded in the agreement logic
   itself; agreement is purely "was this task_id present in every rank's
   `_maybe_queue` argument to this specific synchronous call".

**Does the rendezvous window feed into this?** No direct dependency. The
window (`runner.py`) only controls how many tasks are already sitting in
`_maybe_queue` (via `submit`, `batch_generator.py:150-157`, `496-503`) *before*
the next `agree_on_tasks()` call happens inside `BatchGenerator.step()`
(`batch_generator.py:682-683`, called unconditionally every `step()`). The
window never appears as an input to `mx_any`, `mx_all_gather_tasks`, or the
intersection computation — those only look at `self._maybe_queue`'s *current
contents* at call time, not at how long ago the runner started waiting.

**At c=1, can window=0 cause rank 0 and rank 1 to see DIFFERENT task sets
(disagreement)?** No — for two separate reasons:

1. The collective is blocking/synchronous (`mx.distributed.all_gather` at
   `utils_mlx.py:2284-2287,2326`) — both ranks execute the *same* call with
   whatever each side currently has, and the intersection math
   (`utils_mlx.py:2335`) is deterministic given that gathered data. There is
   no code path where the two ranks compute a *different* `agreed_ids` from
   the same all-gather round; they physically cannot "disagree" — they either
   agree on the same (possibly smaller, possibly empty) intersection, or a
   given task id simply isn't agreed *yet* and is deferred, never *mis*-agreed.
2. At c=1 there is exactly one task id in play. If it hasn't reached both
   ranks' `_maybe_queue` yet at some particular `step()`'s `agree_on_tasks()`
   call, `agreed_ids` for that round is `{}` for that task (it lands in
   `different` on whichever rank(s) already had it, `utils_mlx.py:2339`) —
   this only *delays* when the task enters `self._queue`
   (`batch_generator.py:184`, `528`) on both ranks to a later `step()`
   iteration; it cannot cause the ranks to end up with *different, non-empty*
   final task sets, because eventual agreement only happens when the id is in
   both queues simultaneously, which is symmetric by construction.

**What the 50/100ms launcher-comment failure actually meant, and why 0 is not
in the same failure class:** The comment at `constants.py:125-134`
states: <cite index="0-4">"50ms:  m4-1 rendezvous never fired → per_req=23.4 (legacy serial); 100ms: m4-1 + m4-2 rendezvoused on DIFFERENT iterations → no batch"
and explains the mechanical reason at `constants.py:127-130`: <cite index="0-5">"For batched prefill to fire, BOTH ranks must catch BOTH tasks within their own windows — otherwise agree_on_tasks gates to the intersection (1 task), and the batched gate at BatchGenerator.step() (len(queue) >= 2) fails on both ranks collectively."
This is a **performance/throughput** failure mode, not a correctness/agreement
failure: at c=2, if one rank's rendezvous window closes before its 2nd task
arrives while the other rank's window catches both, `agree_on_tasks`'
intersection at that step is just the 1 task both ranks share — which is
still a *correct* (if smaller) agreement, just not a batch of 2, so
`BatchGenerator.step()`'s `len(queue) >= 2` batching gate doesn't fire and the
system falls back to serial per-task prefill (slower, not wrong). Window=0 at
c=1 is not in this failure class at all: there is only one task, so there is
no "which tasks land in the window" race to lose — `agree_on_tasks` will
eventually see that single id on both ranks (the very next `step()` after
each rank's runner submits it) and agree on `{that_id}` with 100% consistency
regardless of whether the runner waited 0ms or 200ms beforehand. Reducing the
window to 0 only removes the deliberate stall that gives a *second* c>=2 task
a chance to catch up to the first — it does not touch the intersection
mechanism's correctness at any concurrency, including c=1.

## VERDICT

**window=0 is SAFE at c=1 for rank agreement** — `agree_on_tasks`
(`batch_generator.py:159-185`/`505-535`) agrees via a synchronous
`mx.distributed.all_gather` + set-intersection over serialized task ids
(`utils_mlx.py:2284-2338`) that only reads each rank's *current* `_maybe_queue`
contents at call time; the rendezvous window (`runner.py:580-604`, guarded by
`> 0` at `runner.py:580`, skipped cleanly when 0) never feeds into that
computation, so at c=1 (single task id) the two ranks cannot disagree —
window=0 only removes the deliberate stall that helps a *second* concurrent
task join the same batch window at c>=2, a throughput concern, not a
correctness one.
