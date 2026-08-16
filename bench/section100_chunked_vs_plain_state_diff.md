# Section 100: What state CHUNKED prefill leaves behind that PLAIN prefill does not

Scope: read-only code investigation, no cluster relaunch. Answers Section 85's
open item #2 ("explain why the chunked path specifically triggers
`EventImpl::wait`") and Section 96's mechanism map, by tracing `mx.Stream`
identity across both prefill paths and the shared decode step.

## TL;DR

**CONFIRMED BY READING THE CODE**: there are **two distinct `mx.Stream`
objects** named `generation_stream` in this codebase, not one:

| Symbol | Definition | Kind |
|---|---|---|
| `mlx_lm.generate.generation_stream` | `mlx-lm/mlx_lm/generate.py:220` — `mx.new_thread_local_stream(mx.default_device())` | thread-local stream |
| `exo...generator.generate.generation_stream` | `src/exo/worker/engines/mlx/generator/generate.py:282` — `mx.new_stream(mx.default_device())` | plain (non-thread-local) stream |

These are **not aliases**. `exo/.../generate.py` never imports
`generation_stream` from `mlx_lm.generate` (its `from mlx_lm.generate import
(...)` at lines 13-16 pulls only `maybe_quantize_kv_cache, stream_generate` —
confirmed by reading the import block) — it defines its own separate global
of the same name at line 282.

**Decode and PLAIN prefill both run on `mlx_lm.generate.generation_stream`
(stream A). CHUNKED prefill runs on exo's own `generation_stream` (stream
B).** Every decode-step in-place KV-cache update (`cache.keys =
mx.concatenate([cache.keys, new_k], ...)`) therefore has to synchronize
against a producer op that was last enqueued on stream B whenever the cache
was built by chunked prefill — a genuine cross-stream dependency, on the GPU
scheduler's own terms, recreated by the concatenate-based in-place update
pattern every single token. This is exactly the mechanism Section 85 predicted
as its top candidate and exactly matches the observed symptom (idle GPU, CPU
parked in `EventImpl::wait`, flat cost regardless of chunk count/depth).

## The evidence, path by path

### Decode (both after plain AND after chunked prefill — same code)

- `src/exo/worker/engines/mlx/generator/batch_generate.py:9-16` imports
  `BatchGenerator as MlxBatchGenerator` and `generation_stream` **directly
  from `mlx_lm.generate`** (stream A).
- Decode dispatch (`ExoBatchGenerator.step()` → `self._mlx_gen.next()`,
  `batch_generate.py:4228`) calls into `mlx_lm.generate.BatchGenerator.next()`
  (`mlx-lm/mlx_lm/generate.py:2175`), which does `with
  mx.stream(self._stream): return self._next()`.
- `BatchGenerator.__init__` sets `self._stream = stream or generation_stream`
  (`mlx-lm/mlx_lm/generate.py:1854`) — the module-global stream A, unless a
  caller overrides it (nothing in exo does).
- So **every decode token's forward pass, and every decode-step
  `cache.update_and_fetch` in-place write, is enqueued on stream A.**

### PLAIN prefill (`prefill()`'s else-branch, `num_tokens < EXO_PREFILL_STEP_SIZE`)

- `generate.py:866-892` (exo's `prefill()`) calls `stream_generate(...)`,
  imported directly from `mlx_lm.generate` (`generate.py:14`).
- `stream_generate` (`mlx-lm/mlx_lm/generate.py:787`) drains `generate_step`
  (`mlx-lm/mlx_lm/generate.py:301`), whose `_step` closure explicitly does
  `with mx.stream(generation_stream):` at `mlx-lm/mlx_lm/generate.py:393`,
  and the outer prompt-processing loop is itself wrapped `with
  mx.stream(generation_stream):` at `mlx-lm/mlx_lm/generate.py:431`.
  Both refer to `generate_step`'s *own module's* global — **stream A**,
  the identical object decode uses.
- **Plain prefill's cache-populating ops and decode's cache-consuming ops
  are on the same stream.** No cross-stream event is required; same-stream
  ops execute in enqueue order with no synchronization primitive needed.
  This matches the fast (~16ms/token) observed cost.

### CHUNKED prefill (`pipeline_parallel_prefill` / `prefill_interruptible_start` → `ResumablePrefillSession.advance()`)

- `_pipeline_parallel_prefill_steps` (`generate.py:421`) wraps its ENTIRE
  chunk loop — leading dummies, every real chunk (including the
  `yield ("chunk", i, chunk_tokens)` at `generate.py:541`), and trailing
  dummies — inside `with mx.stream(generation_stream):` at `generate.py:524`.
  This `generation_stream` resolves via normal Python scoping to
  **`exo/.../generate.py`'s own module global at line 282 — stream B**,
  because that is the module `_pipeline_parallel_prefill_steps` is defined
  in.
- The post-loop single-token forward pass is *also* explicitly wrapped:
  `with mx.stream(generation_stream):` at `generate.py:630`, and the final
  cache-state eval at `generate.py:637` — both stream B.
- `mx.stream()` is a context-manager RAII push/pop onto MLX's thread-local
  default-stream stack — the push happens at `__enter__` and is NOT undone
  until `__exit__` actually runs. Because `_pipeline_parallel_prefill_steps`
  is a Python generator that `yield`s from *inside* that `with`-block, the
  push stays live (context never exits) across the entire suspension — i.e.
  while `ResumablePrefillSession.advance()` (`pp_prefill_session.py`) is
  driving the inner `_forward_steps` generator layer-by-layer via
  `next(gen)`/`self._ctx.run(...)`, on the SAME OS thread, stream B is still
  the ambient default stream. `_forward_steps` itself
  (`mlx-lm/mlx_lm/models/deepseek_v4.py:6238`) contains **no `mx.stream(...)`
  of its own** (confirmed by grep — zero matches) and no per-layer stream
  switch, so every attention layer's `cache.update_and_fetch` call during a
  chunked-prefill pause runs under whatever's ambient — stream B.
- `prefill_interruptible_advance()`'s cache trim/rollback tail
  (`generate.py:1198`, `with mx.stream(generation_stream):`) is likewise
  stream B — the object at line 282, not the mlx_lm one.
- **Net effect: every KV-cache array that chunked prefill creates or
  last-writes (`cache.keys`, `cache.values`, and their quantized/SSM
  counterparts) is last-produced on stream B.** The immediately following
  decode step then does `cache.keys = mx.concatenate([cache.keys, new_k],
  axis=-2)` **on stream A** (`mlx_lm/models/cache.py:511-512`,
  `KVCache.update_and_fetch`) — a cross-stream read of a stream-B-produced
  array, forcing MLX's scheduler to insert a synchronization event before
  the concatenate can run on stream A's queue.

## Thread / contextvars: ruled out as the mechanism

- Grepped the chunked-prefill glue path (`pp_batched_decode_glue.py`,
  `pp_prefill_session.py`) for `threading.Thread`, `Thread(`,
  `asyncio.to_thread`, `run_in_executor`, `ThreadPoolExecutor` — **zero
  matches.** `ResumablePrefillSession.advance()` runs synchronously, inline,
  on the same call stack / same OS thread as `Rank0BatchedDecodeGlue.tick()`,
  which is the same thread the rest of the runner (including decode) drives
  from. There is no cross-thread handoff.
- `ResumablePrefillSession._ctx: contextvars.Context`
  (`pp_prefill_session.py`, `__post_init__` calling `contextvars.copy_context()`)
  exists **only** to isolate `ForwardStepInfo` state
  (`set_forward_step_info`/`pp_metaframe.py`) between an interleaved decode
  step's own forward pass and this session's resumes — it has nothing to do
  with `mx.Stream` selection. MLX's default-stream stack is a C++-level
  thread-local, not a Python `contextvars` variable; `Context.run(...)`
  does not save/restore it. **This candidate is confirmed NOT to be the
  differentiator** — the same-thread, same-context execution proves streams
  (not threads or contextvars) are the axis that actually differs.

## `mx.eval` discipline: no asymmetry found

Both paths eval aggressively and by design (not a leak/laziness gap):
chunked prefill evals per-chunk (`mx.eval([c.state for c in _prompt_cache])`,
`generate.py:566`; per-layer pause eval in `advance()`,
`pp_prefill_session.py:293,322`); plain prefill evals per prompt-chunk inside
`generate_step`. This was already investigated and ruled out at the top of
the task (produces MORE compute, not an idle-GPU wait) — confirmed again here:
no path skips eval or defers it asymmetrically in a way that would explain an
idle-GPU wait. The differentiator is stream *identity*, not eval *timing*.

## Ranked hypothesis

1. **(Confirmed mechanism, top candidate) Cross-stream KV-cache dependency.**
   Chunked prefill's cache arrays are last-written on
   `exo.../generate.py:282`'s `generation_stream` (stream B); every decode
   step's in-place cache concatenate runs on `mlx_lm.generate.py:220`'s
   `generation_stream` (stream A). MLX must insert a stream-B→stream-A
   synchronization event for the first op on stream A that consumes the
   stream-B-produced cache array. Because `KVCache.update_and_fetch`
   reassigns `self.keys`/`self.values` via `mx.concatenate` **every single
   decode token** (`mlx_lm/models/cache.py:511-512`), and MLX's lazy engine
   schedules based on which stream last produced an array feeding the
   current op's dependency graph, this is consistent with a recurring
   per-token cross-stream wait if the scheduler's dependency tracking (or a
   quantized/SSM cache's dequant path touching the original stream-B buffer)
   keeps re-crossing the stream boundary rather than the "steady state"
   settling onto stream A after the first token. **Not yet proven that the
   wait literally recurs at 100% of tokens rather than only the first one**
   (see cheapest experiment below) — but it is the only difference between
   the two paths that (a) requires zero extra GPU compute, (b) is genuinely
   flat regardless of chunk count or context depth (a stream identity is a
   fixed property of the code path, not of size), and (c) matches the
   `EventImpl::wait` / idle-GPU signature exactly.
2. Distributed op baked into the cache graph (Section 85's "also plausible"
   candidate): not independently confirmed or refuted here — chunked
   prefill's pipeline recv/send/flush machinery (`flush_prefill_sends()`,
   `mx.distributed.recv_like` inside `_forward_steps`) does run on stream B
   too, so if any deferred/queued send op remains part of the cache's lazy
   graph ancestry, it would ALSO cross to stream A on the first decode read
   — this is not distinguishable from hypothesis 1 by static reading alone;
   both point at the same stream-B/stream-A boundary and could be the same
   underlying event.

## If streams turned out identical — they do not

Per the task's instruction to say so plainly if this were a negative result:
**it is not.** Two separately-constructed `mx.Stream` objects, both
module-globally named `generation_stream`, are used by the two prefill code
paths, and only one of the two (`mlx_lm.generate`'s) is shared with decode.
This is a confirmed, non-trivial divergence, not an artifact of naming
confusion — grep confirms exo's `generator/generate.py` never imports the
mlx_lm one, and defines its own at line 282.

## Cheapest experiment (Python-only, no MLX rebuild)

Make exo's chunked-prefill path use the **same stream object** decode uses,
and see if the 550ms step function collapses to the plain-path cost at
>=2048-token prompts:

```python
# src/exo/worker/engines/mlx/generator/generate.py, near line 282
from mlx_lm.generate import generation_stream  # instead of:
# generation_stream = mx.new_stream(mx.default_device())
```

This is a one-line import swap (delete the `mx.new_stream(...)` definition,
import the mlx_lm one instead) — every `with mx.stream(generation_stream)` in
`generate.py` (lines 524, 630, 637, 1198, 1378, 2317) then binds to stream A
automatically, with no other code change. Run the exact same sweep from
Section 85 (91-95, 746-751, 1923, 2365, 2925, 4412, 14273 prompt tokens) and
check whether the 2365-14273 range's `last_layer_eval`/decode tok/s collapses
from ~550ms/~0.47tok/s back down to the ~16ms/~20tok/s plain-path numbers.

- If it collapses: hypothesis 1 (cross-stream KV-cache dependency) is
  **confirmed** as the root cause, and the real fix is either this import
  unification (if it doesn't reintroduce whatever motivated giving chunked
  prefill its own stream in the first place — needs a `git blame`/history
  check on `generate.py:282` before shipping) or an explicit
  `mx.synchronize()` / stream-tag-normalizing copy at the chunked→decode
  handoff boundary (once per request, not once per token).
- If it does NOT collapse: hypothesis 1 is killed cleanly, and the search
  should move to hypothesis 2 (distributed op ancestry) or re-open the
  "count `EventImpl::wait` per token" `lldb` experiment Section 85 already
  scoped (item 1 in "Next, in order").

This experiment costs one Python edit, one redeploy, no MLX rebuild, and
directly targets the ranked top candidate.

## Files/lines referenced

- `mlx-lm/mlx_lm/generate.py:220` — `mlx_lm.generate.generation_stream` definition (thread-local stream, "stream A")
- `mlx-lm/mlx_lm/generate.py:393,431,684,733` — stream-A usage inside `generate_step`/prompt processing
- `mlx-lm/mlx_lm/generate.py:1854,2182,2192,2175` — `BatchGenerator` (decode) uses stream A by default
- `mlx-lm/mlx_lm/models/cache.py:489-514` — `KVCache.update_and_fetch`, the per-token in-place `mx.concatenate`
- `mlx-lm/mlx_lm/models/deepseek_v4.py:6238` — `_forward_steps`, no internal `mx.stream`
- `src/exo/worker/engines/mlx/generator/generate.py:13-16` — import block proving exo does NOT import mlx_lm's `generation_stream`
- `src/exo/worker/engines/mlx/generator/generate.py:282` — exo's own `generation_stream` definition ("stream B")
- `src/exo/worker/engines/mlx/generator/generate.py:421-524,541,630,637` — `_pipeline_parallel_prefill_steps`, stream-B usage, yield point inside the `with`-block
- `src/exo/worker/engines/mlx/generator/generate.py:866-892` — plain-prefill branch calling `stream_generate` (stream A)
- `src/exo/worker/engines/mlx/generator/generate.py:1198` — `prefill_interruptible_advance` cache trim, stream B
- `src/exo/worker/engines/mlx/pp_prefill_session.py` — `ResumablePrefillSession`, `_ctx: contextvars.Context`, zero `mx.stream`/threading references
- `src/exo/worker/engines/mlx/generator/batch_generate.py:9-16,4228` — decode dispatch imports `generation_stream` from `mlx_lm.generate` (stream A)
