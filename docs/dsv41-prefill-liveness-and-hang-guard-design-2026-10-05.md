# DSv4.1 prefill liveness + hang-guard design (2026-10-05)

**Status:** implementing. Companion record: `docs/PERFORMANCE_HISTORY.md` entry
"2026-10-04 (soak 2 postmortem)" (commit `3e6435acf`).

## Incident in one paragraph

A 290K-row delta prefill at >750K context ran 5h07m, healthy, at ~137 GB
footprint on a 137.44 GB box (compressor holding 5-7 GB; swap total 1 GB).
The supervisor's hang watchdog saw 68 s with no runner event and a "flat"
footprint (+0.20 GB growth vs the 0.25 GB threshold at the memory ceiling,
where the process CANNOT grow) and SIGKILLed it as hung. Rank 1 died on the
peer EOF mid-JACCL-collective. Both runners re-placed automatically. The
kill was a false positive of a structurally-blind probe.

Two root causes:
1. **Progress-event granularity.** Events fire per prefill chunk; at deep
   context a chunk takes minutes (128-row chunks at <16 tok/s), so silence
   routinely exceeds the 45 s window while real compute progresses.
2. **The growth probe cannot discriminate at the ceiling.** A working process
   pinned at physical size shows "plateau"; the probe reads that as wedged.

Code doctrine (supervisor.py): *signal real progress rather than widen the
timeout.* This design follows it. Fable review (2026-10-05) blessed the shape
with the corrections inline below.

## Fix A — fence-point liveness hook (observation-only; arms immediately)

The mlx-lm `deepseek_v41` model commits each multi-row chunk every K layers
(`model._fence_every`, default 2) at a synchronous `mx.eval(h, pre_mix)`
fence. We key a liveness callback to that same point: every fence call is
backed by K layers of actually-committed compute.

**Contract — mlx-lm (`mlx_lm/models/deepseek_v41/model.py`):**
- After each fence `mx.eval`, call `hook = getattr(self, "_fence_hook", None)`
  (resolved ONCE per forward, before the loop) if set and n > 1.
- The call is wrapped: `try: hook() except Exception: <policy>`. Policy:
  log-once (module logger, warning + exc_info), set `self._fence_hook = None`
  (disable for subsequent forwards), latch `self._fence_hook_failed = True`.
  A hook bug must NEVER propagate into `Model.__call__` and must never repeat
  every fence.
- Hook contract for implementors: no locks, no model-state mutation, no
  `mx.*` calls, non-blocking emission only. Runs on the model thread.

**Contract — mlx-lm drivers (`prefill.py`):** `prefill()` and
`chunked_prefill()` accept `fence_hook=None`; set/restore
`model._fence_hook` alongside `_fence_every` (restore in `finally`).
`warmup()` accepts it for symmetry, default None.

**Contract — exo driver (`src/exo/worker/engines/mlx/dsv41/session.py`):**
`engine_prefill` accepts `fence_hook=None`; sets/restores it with the fence
(`fence_hook_prev = getattr(model, "_fence_hook", None)`; finally restores).
Warn once at prefill start if resolved `fence_every == 0` ("liveness hook
unavailable"). After the loop, if `getattr(model, "_fence_hook_failed",
False)`: log CRITICAL loudly and reset the flag. Extend the existing
"[DSV41] prefill controls" log line with `fence_hook=on|off`.

**Contract — exo engine (`engine.py`):**
- New method `Dsv41Engine._session_fence_heartbeat()` → calls
  `self.prefill_heartbeat()` (the existing 15 s-throttled status re-emit that
  resets the supervisor's silence clock). Add a spacing watchdog: warn when
  the interval between hook invocations exceeds 30 s (the invariant is
  event gap ≤ 15 s throttle + fence spacing; spacing > 30 s is the failure
  predictor — measure it, don't assert it).
- Wiring chain: `Dsv41Engine` → `Dsv41Sessions(fence_heartbeat=…)` →
  `SessionCache(fence_hook=…)` → `engine_prefill(fence_hook=…)`. BOTH
  `engine_prefill` call sites in `SessionCache` (conv path and non-conv path)
  pass it.
- Active on ALL ranks (each rank's own supervisor must see its own runner's
  events; rank 1's death by peer-EOF is exactly what this prevents).

**Invariant, corrected (Fable):** "silence ≤ 15 s" is NOT asserted — the true
bound is throttle + longest between-fence region. At `fence_every=2` the
observed per-fence spacing is ~8-16 s (margin ~3x under the 45 s window).
The spacing-warn (>30 s) makes violations loud; soak logs are the audit.

**Bit-identity:** the hook performs no mx ops; fenced runs must remain
bit-identical with and without the hook (test: seeded logits hash equality).

## Fix B — supervisor stack-class guard (SHADOW this deploy)

Replace "flat footprint ⇒ kill" with a classification when the probe
plateaus. Decision table (**armed** mode):

| Condition | Verdict |
|---|---|
| footprint grew ≥ 0.25 GB | extend (existing growth path) |
| flat + SPIN (CPU delta ≥ threshold) | **kill fast** (the c>=2 jaccl-spin wedge class; no extension regardless of stack) |
| flat + blocked + gpu-stack + at-ceiling | extend (bounded) |
| flat + blocked + native-setup (existing narrow signature, UNCHANGED) | extend (existing path) |
| flat + blocked + unknown | kill |
| classifier/sample failure | extend once + alert, then kill on repeat |

- **SPIN metric:** CPU-time delta via `ps` cputime between probe ticks ÷ wall
  interval; threshold env-tunable (`EXO_RUNNER_HANG_SPIN_FRACTION`, default
  0.5). The two incident classes separate cleanly on this axis: healthy
  at-ceiling prefill is *blocked* (main thread in
  `Scheduler::wait_for_one`, workers in `__psynch_cvwait`, low CPU delta);
  the wedge burns 100% CPU.
- **gpu-stack classifier:** symbols + image ranges (libmlx.dylib
  eval/scheduler/metal frames; AGXMetal*/IOGPU images). Must NOT widen the
  existing native-setup signature — the wedge must stay on the fast-kill path.
- **at-ceiling:** footprint within ~2 GB of `hw.memsize`
  (`EXO_RUNNER_HANG_CEILING_MARGIN_GB`, default 2.0). The incident plateau
  (137.20 vs 137.44 GB) qualifies. gpu-class extension REQUIRES at-ceiling
  (conservative first ship; monotone improvement over kill-everything-flat).
- **Budget:** ONE shared monotonic counter across all extend reasons
  (existing 20×20 s), reclassification does not reset it; a real event still
  resets everything (existing behavior).
- **Modes:** `EXO_RUNNER_HANG_STACK_MODE = off | shadow | arm`, read at
  module import; **default shadow**. Shadow = run the classifier on every
  plateau tick, log verdict + all input fields, kill path UNCHANGED. Arm
  only after a soak's shadow logs validate the classifier (do NOT co-arm
  with a precision change; if a soak regresses, cause must be attributable).
- **Asymmetry (state it):** a false kill costs a multi-hour run + a peer
  cascade; the worst case B-armed can cause is ≤ 7 min of extra wedge
  latency (bounded budget). Trade direction is right; the spin axis keeps
  the SIGKILL-is-for class on the fast path.

**Launcher wiring:** add the three envs to start_cluster.sh's EXO_ENV
allow-list (they default in code; audit for stale values first — these are
new, so none expected).

## Verification bar

**Laptop unit tests (mandatory):**
- mlx-lm: hook call-count formula vs `_fence_every` (multi-row forward);
  zero calls at n==1; zero when unset; exception-in-hook → forward
  unaffected + hook disabled + flag latched; driver set/restore including
  after exception; **bit-identity** with/without hook.
- exo driver/engine: set/restore around prefill; heartbeat throttle behavior
  (fake clock); rank!=0 emissions; spacing-warn fires (fake clock).
- supervisor: golden fixtures — the incident dump
  (`~/.hermes/cache/scratch/hang-dumps/incident_20261004_47545.txt`) MUST
  classify gpu/blocked/at-ceiling (extend-class); a locally-generated
  synthetic spin dump MUST classify kill-class; decision-table tests
  covering the full matrix incl. budget monotonicity across reasons and
  no-reset-on-reclassify; shadow mode changes no kill behavior; mode env
  parsing.

**Live gates:** the r1M re-proof soak (next deploy) is the live gate for A;
its shadow logs are the pre-arm gate for B. A wedge drill (synthetic spin)
measures post-change detection latency before arming B.

## Rejected alternatives

Blind runner-side timer emitting every 15 s regardless of progress (masks
genuine wedges — every beat must be backed by committed compute). Widening
HANG_TIMEOUT_SECONDS (delays genuine-wedge detection; against doctrine).
Unconditional extension on flat footprint (weakens the wedge class B exists
for). Generalizing the native-setup signature (same). A shared-memory
fence counter read by the supervisor (deterministic upgrade idea; deferred —
larger IPC surface, revisit if the sample-based classifier proves fragile).
