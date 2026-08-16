# Section 100: Static hunt for a timeout/poll-interval constant behind the flat ~550ms PP decode wait

Scope: pure static analysis of `src/`, `mlx/`, `mlx-lm/` per the task's read-only constraint.
No cluster runs performed. Builds on doc Sections 85/86/89/96 (all read in full).

## TL;DR verdict

**Found a strong candidate: `MLX_JACCL_ACK_RETRANSMIT_US` (default 500,000us = 500ms),
`mlx/mlx/distributed/jaccl/lib/jaccl/mesh_impl.h:174-180`.** It is a flat 500ms constant,
independent of context depth, gated only on "did the collective/ack loop see zero forward
progress" — exactly the flat-across-depth signature required. It is documented in the
codebase's own comments (same file, lines 183-192) as having caused almost this exact
symptom before on the PP batched-decode path (~525ms drain + 500ms retransmit ≈ 1.0s/drop),
which is why `MLX_JACCL_P2P_DRAIN_QUIET_US` (100ms adaptive-floor, Section 71/76-78) was
split out from it as a *separate* knob for the p2p send/recv drain loops specifically.

**However — reachability is NOT proven, and there is a strong competing explanation already
recorded in the doc (Section 89) that the current investigation had NOT yet resolved as of
Section 96.** See "Reachability audit" below. My read: the 500ms ACK-retransmit timer remains
plausible only if the PP batched-decode last-layer path routes through the *collective* ACK
mechanism (`jaccl_ack_retransmit_us`) rather than the p2p send/recv drain path (which already
has its own tighter, adaptive ~100ms-ish `jaccl_p2p_drain_quiet_for`). The doc's own account of
`first_layer_recv`/`gather_send` at 0.1-0.3ms (Sections 85/86, and independently re-derived below)
argues those p2p ops are NOT stalling — so if a constant-timer explanation survives at all, it
has to be one on a path *not* directly instrumented by the `[LAYER_PHASE]` timers, i.e. something
inside the collective/ack layer nested underneath `mx.eval(output)` in the last-layer body, not
around the explicit recv/send calls that already read fast.

**Second finding, requested separately: the transport timings ARE trustworthy.** Every
`[LAYER_PHASE]` timer in `pp_batched_decode_layers.py` wraps a call immediately followed by a
forced `mx.eval()` on the exact tensor produced by that call, before the timer stops. There is
no lazy-enqueue gap between "op issued" and "timer stopped" for any of `first_layer_recv`,
`last_layer_send`, or `gather_send`/`gather_recv`. See full audit below — this closes the
question raised in the task and the doc never separately verified this.

---

## 1. Candidate constants, ranked

### Rank 1 (plausible, unresolved): `MLX_JACCL_ACK_RETRANSMIT_US` = 500,000us
`mlx/mlx/distributed/jaccl/lib/jaccl/mesh_impl.h:174-180`
```cpp
inline uint64_t jaccl_ack_retransmit_us() {
  static const uint64_t v = [] {
    const char* e = std::getenv("MLX_JACCL_ACK_RETRANSMIT_US");
    return e ? std::strtoull(e, nullptr, 10) : 500000ULL;
  }();
  return v;
}
```
Used at: collective ACK-barrier retransmit loop (line ~725, `quiet_us = jaccl_ack_retransmit_us()`),
`reliable_all_reduce`'s retransmit loop (line ~1251, reused via comment "reuse knob"), and any
site consuming this default before Section 71 split the p2p-specific
`jaccl_p2p_drain_quiet_us`/`jaccl_p2p_drain_quiet_for` out of it.

- **Value class:** exactly a flat, depth-independent 500ms timer — matches "flat across 2.9K→14.3K
  tokens" requirement precisely (it is a *wait-for-zero-progress* timer, not a per-byte one).
- **Reachability on PP decode last-layer path:** UNCERTAIN. The doc's own comment block
  (lines 183-200) states this exact mechanism previously produced "~525ms drain + 500ms
  retransmit ≈ 1.0s" on "the PP batched-decode path" for collective ACK traffic, and that the
  fix was to split out a *separate*, much tighter knob (`jaccl_p2p_drain_quiet_for`, ~100ms
  floor + 1ms/chunk, adaptive, capped 500ms) specifically for "the p2p send()/recv() DRAIN loops."
  If BatchedMetaFramedPipelineLastLayer's send/recv (measured 0.1-0.3ms per the doc and
  independently confirmed by me, see §2) goes through the p2p path and NOT the ACK/collective
  path, then `jaccl_ack_retransmit_us` (500ms) does not fire on this call at all, and only the
  100ms-ish adaptive p2p quiet period could. Whether last-layer decode invokes any *collective*
  op (all_sum/all_gather/barrier) nested under the hood — as opposed to pure point-to-point
  send/recv — is the open reachability question this static hunt could NOT settle from mesh_impl.h
  alone; it requires tracing which jaccl call `mx.distributed.recv_like`/`send` compile down to
  for THIS sharding mode (Pipeline, not Tensor). Section 89 already proved Pipeline mode does NOT
  do `all_sum` (that's TP-only) — but that only rules out the MoE `all_sum` fence specifically, it
  does not rule out an ACK/liveness sync elsewhere in the pipeline send/recv machinery.
- **Verdict:** rank #1 by plausibility (exact flat 500ms match, and the file's own comments
  describe this exact failure mode on this exact path historically) but NOT confirmed reachable.
  Flag loudly: this deserves the next dynamic check (grep runtime logs for
  `jaccl-v2] rank=%d call=%u serving retransmit` around a slow decode token) before being treated
  as proven.

### Rank 2 (plausible but smaller / already-tuned): `MLX_JACCL_P2P_DRAIN_QUIET_US` adaptive quiet period
`mesh_impl.h:270-300` (`jaccl_p2p_drain_quiet_for`)
- Floor 100,000us (100ms) + 1,000us/chunk, capped at 500,000us.
- **Reachability:** this IS on the p2p send/recv drain path that `first_layer_recv` /
  `last_layer_send` / `gather_send` most plausibly hit. But it only fires when a frame is
  genuinely lost (zero progress) — a rare/probabilistic event, not something that would produce
  a **flat, always-on** ~550ms on every single token across a whole benchmark run. A retransmit
  timer explains occasional spikes, not a steady-state floor. Doc Sections 85/86 already measured
  `first_layer_recv` at 0.1-0.3ms in the SAME runs that showed the 550ms last-layer stall — i.e.
  the drain quiet period was NOT engaging on those calls (no observed 100ms+ delay on
  first_layer_recv). This argues AGAINST it being the explanation for the flat 550ms, though it
  remains a candidate for occasional worse spikes layered on top.
- **Verdict:** demoted below rank 1 — doesn't fit "flat every-token" as well, and the very
  timers that would show it (first_layer_recv, gather_send) already read fast.

### Rank 3 (ruled out): `MLX_EVENT_WAIT_POLL_US` / `MLX_EVENT_WAIT_SPIN` (event.cpp)
`mlx/mlx/backend/metal/event.cpp:76` (default 50us poll), `:96` (default 2000 spin).
- **Reachability:** yes — this is literally where the stack traces park (Section 85's 89-92%
  `EventImpl::wait -> sleep_for`).
- **Verdict: RULED OUT by the doc's own live A/B (Section 86).** Setting
  `MLX_EVENT_WAIT_SPIN=50,000,000` (removing the sleep path entirely) made the slow case ~20%
  WORSE, not better — proving this is where the thread *parks*, not what it is *waiting for*.
  Included here only for completeness / to explicitly confirm I did not silently re-promote an
  already-refuted candidate.

### Rank 4 (ruled out): `MLX_EVENT_WAIT_TIMEOUT_MS` (self-abort deadline)
`event.cpp:89` (default 40,000ms = 40s).
- Two orders of magnitude too large to produce ~550ms; this is a liveness backstop, not a
  per-token cost. Not reachable as an explanation.

### Rank 5 (ruled out structurally, per doc Section 89): the MoE fence's `mx.eval(y)`
`mlx-lm/mlx_lm/models/deepseek_v4.py:2835-2893` (Phase H Lever 1 fence under
`sharding_group is not None`).
- Doc Section 88 promoted this, Section 89 retracted it with code proof:
  `sharding_group` is only set by `TensorParallelShardingStrategy` (auto_parallel.py:1065/1081/1121,
  called only at auto_parallel.py:849, the TP-only call site). This cluster runs
  `DSV4_SHARDING=Pipeline`, so `self.sharding_group is None` on every layer and the fence block
  never executes. I did not re-verify this myself beyond confirming the doc's citation is
  internally consistent (it also holds up arithmetically: 43 layers × ~1.1ms/layer ≈ 47ms, not
  550ms, and the fast 1,927-token case runs the identical 43 layers at 16ms total — a per-layer
  unconditional cost can't be present in one case and absent in the other with the same layer
  count). Not re-litigated; carried forward as settled.

### Rank 6 (ruled out): `JACCL_POLL_INSTRUMENT_THRESHOLD_US` (default 100,000us)
`mesh_impl.h:106` — this is a diagnostic-logging threshold ("log if a poll call exceeds
100ms"), not a wait/sleep duration. It cannot itself cause a delay; it only decides whether to
print. Not a candidate.

### Rank 7 (ruled out): `MLX_JACCL_STALL_TIMEOUT_US` / `MLX_JACCL_RELIABLE_IDLE_US` (15us)
`mesh_impl.h:155`, `:391`. Stall-detection deadline (independent large constant, not ~550ms) and
a 15us idle-poll sleep (far too small, and it's a per-iteration yield not a wait floor). Neither
matches the ~550ms magnitude or the flat-regardless-of-progress signature.

### Other numeric scan
Grepped both `mlx/` and `src/` trees broadly for 100-1000ms-range literals, `wait_for`,
`sleep_for`, `poll`, `timeout`, `interval`, `deadline`, `retransmit`, `quiet`, `drain`, `flush`.
No other candidate constant in the 400-700ms band was found outside the jaccl mesh file and the
already-addressed event.cpp knobs. (Did not exhaustively list every hit — see method note below —
but nothing else surfaced as a `getenv`-backed default in the relevant magnitude.)

**Method note on exhaustiveness:** I did not attempt to enumerate every literal number in three
large trees (per the task's own instruction not to dump a useless 200-constant list). I targeted
`mesh_impl.h`, `event.cpp`, `scheduler.h`/`metal.cpp`, and the exact call sites named in the task,
plus keyword greps across the trees for the listed terms. `scheduler.h` and `metal.cpp` were
checked and contain no timeout/poll constants in the relevant range (scheduler.h is queue/thread
management with no sleep constants; not reproduced here for brevity since nothing was found).

---

## 2. Transport-timing audit: completion vs lazy enqueue — DEFINITIVE

File: `src/exo/worker/engines/mlx/pp_batched_decode_layers.py`.

**Verdict: all three cited timers (`first_layer_recv`, `last_layer_send`, `gather_send`, and
also `gather_recv`) measure forced completion via an explicit `mx.eval()` inside the timed
window, not lazy graph construction. The 0.1-0.3ms numbers are trustworthy.**

Evidence, each timer's exact code:

- `first_layer_recv` (lines ~239-249):
  ```python
  _t_recv = time.perf_counter() if _DECODE_PHASE_TRACE else 0.0
  x_recv = mx.distributed.recv_like(x_bf16_template, self.r - 1, group=self.group)
  mx.eval(x_recv)          # <-- forced eval BEFORE the timer reads elapsed time
  if _DECODE_PHASE_TRACE:
      logger.info(f"[LAYER_PHASE] first_layer_recv=...")
  ```
  `mx.eval(x_recv)` sits between the `recv_like` call and the log line that reads
  `time.perf_counter() - _t_recv`. This is a completion measurement.

- `last_layer_send` (lines ~368-384): same shape — `mx.eval(sent_forward)` executes before the
  elapsed-time log line.

- `gather_send` (lines ~411-417) and `gather_recv` (lines ~432-440): same shape —
  `mx.eval(sent)` / `mx.eval(output_for_gather)` execute before their respective log lines.

- `last_layer_build` / `last_layer_eval` (lines ~332-345): this is the ONE place the code's own
  comment (lines 325-331) explicitly explains the split exists BECAUSE build is lazy and eval is
  forced — `build = self.original_layer(...)` (graph construction only, expected microseconds)
  vs `eval = mx.eval(output)` (forced, where "the real compute AND any cross-rank transfer nested
  in the graph is actually paid"). This is exactly the split the task asked me to verify exists;
  it does, and it's the mechanism that correctly attributes the 550ms to `eval` not `build`
  (doc: build=0.1ms, eval=550ms+, consistent with GPU-idle/CPU-blocked stacks).

**Conclusion:** there is no gap in this file between "operation issued" and "timer stopped" — MLX
laziness is explicitly defeated by an `mx.eval()` call inside every timed window I found. The
claim "transport is 0.1-0.3ms" is measuring real wall-clock completion of those specific
recv/send calls, not enqueue latency. This means transport (in the narrow sense of "the explicit
p2p send/recv calls this file makes") is legitimately ruled out as the site of the 550ms, exactly
as Sections 85/86 concluded — but it does NOT rule out a nested collective/ACK synchronization
happening *inside* `self.original_layer(x, ...)` and only surfacing at `mx.eval(output)` in the
`last_layer_build`/`last_layer_eval` split, which is unlabeled by any narrower timer. That's the
one gap in observability: `last_layer_eval` is a single lump covering "whatever mx.eval(output)
actually waits on," and if that includes an ACK-retransmit timer nested inside jaccl's collective
path, no Python-level timer in this file would separately attribute it — which is exactly why
Rank 1 above remains open rather than closed.

---

## 3. What would resolve Rank 1 (for the record, not executed — static hunt only)

1. Grep runner logs for `jaccl-v2] rank=%d call=%u serving retransmit` or
   `reliable_all_reduce exceeded max retransmit rounds` around a timestamp captured mid a slow
   (~550ms) `last_layer_eval` in `[LAYER_PHASE]` output. If present at ~500-550ms cadence, Rank 1
   is confirmed live-fire, not just theoretically reachable.
2. Statically trace what `self.original_layer(...)` (the wrapped last-decoder-layer forward,
   `original_layer` — the un-wrapped model layer) does internally for Pipeline sharding: does it
   invoke ANY jaccl call other than the explicit `mx.distributed.send`/`recv_like` already timed
   in this file? If yes, that untimed call is the leading suspect and should get its own
   `[LAYER_PHASE]` timer around a forced `mx.eval()`.

---

## Files read (for traceability)
- `docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md` Sections 85, 86, 87, 88, 89, 90, 96
  (full text, end of file)
- `mlx/mlx/backend/metal/event.cpp` (full file)
- `mlx/mlx/distributed/jaccl/lib/jaccl/mesh_impl.h` (targeted sections: ~40-450, 560-630, 700-730,
  1000-1090, 1250-1345, 1400-1500, 1660-1680)
- `src/exo/worker/engines/mlx/pp_batched_decode_layers.py` (full timer call sites)
