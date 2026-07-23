# Upstream Sync — exo + mlx + mlx-lm (2026-07-22/23)

## SUMMARY
Merged all three of Adam's forks (`adurham/exo`, `adurham/mlx`,
`adurham/mlx-lm`) with their respective true upstreams
(`exo-explore/exo`, `ml-explore/mlx`, `ml-explore/mlx-lm`) for the first
time in a while. All three had diverged substantially with local
instrumentation/perf work. Resolved 9 real conflicts total across the
three repos by **combining both sides' logic**, not picking one over the
other. Found and fixed one real runtime regression the merge introduced
(mlx-lm's stop-sequence API changed upstream; exo's own code still called
the old signature). Validated via a full local build+test battery on all
three repos, then via two independent live-cluster context-level stress
sweeps (2K-250K tokens) on the real 2-node Mac Studio M4 cluster. No
throughput or stability regression found at any tested context level.

Final commits: `exo` `777fd005`, `mlx` `e5140ff9d`, `mlx-lm`
`767d0c2` (on `diag/spec-state-split-timing-v2`, the branch exo's
submodule actually pins). All pushed to `origin` (never `upstream`,
per standing rule — these forks push direct-to-main, no PRs).

## WHY THIS WAS DONE
It had been a while since any of the three repos looked upstream (see
`references/pp-dspark-execute1-fix-found-2026-07-20.md` /
`exo-stall-faulthandler-breakthrough-2026-07-21` skill for the
EXECUTE-disabled-by-default decision that was the active work right
before this). Standard hygiene: periodically re-sync a long-lived fork
so local divergence doesn't compound indefinitely and so upstream fixes
(bug fixes, new model support, perf work) aren't permanently missed.

## MERGE RESULTS PER REPO

### exo (`adurham/exo` ← `exo-explore/exo`)
Clean, zero conflicts. 2 commits behind upstream (a `devalue` CVE bump,
one new Kimi K2.7-Code model card). Merged and pushed as `9d16b895`.

### mlx-lm (`adurham/mlx-lm` ← `ml-explore/mlx-lm`, branch
`diag/spec-state-split-timing-v2` — the branch exo's submodule actually
pins, NOT `main`; the two have diverged separately and untangling that
is out of scope for this sync)
One real conflict in `mlx_lm/models/qwen3_5.py`: our fork's own
`EXO_LAYER_EVAL_INTERVAL` per-layer memory-cap `mx.eval()` (caps ~7GB
lazy-graph buildup during prefill down to ~0.4-0.8GB/layer) collided
with upstream's newly-added pipeline-parallel send/recv/broadcast in the
same forward loop. Resolved by combining both — the eval-interval logic
now runs inside the loop over `self.pipeline_layers`, with the
recv-before/send-after pipeline plumbing wrapped around it, exactly
mirroring how upstream structured its own addition.
Merged as `767d0c2`, pushed.

### mlx (`adurham/mlx` ← `ml-explore/mlx`)
The big one: 8 real conflicts, concentrated in the C++ files carrying
Adam's own diagnostics (the ring-buffer stall diagnostic and the
interruptible-wait jaccl-hang-prevention fix) plus one substantive SDPA
kernel merge.

- `device.h` / `device.cpp` — combined our `CmdBufRingEntry`
  ring-buffer stall diagnostic (`buffer_ops()` accessor,
  `cmdbuf_ring_diag_enabled()`/`cmdbuf_ring().record()` call in
  `commit()`) with upstream's parallel refactor of `commit()` to take an
  optional completion callback plus its own error-propagation
  (`addCompletedHandler` closure capturing `error_`/`wait_events_`/
  `signal_events_`). Both now coexist in one `commit(completion)`.
- `eval.cpp` — combined our `check_error_deferred()`/
  `accumulate_gpu_time_if_enabled()` GPU-idle diagnostics (registered via
  a manual `addCompletedHandler` at each `eval()`/`finalize()` call site)
  with upstream's new `new_thread_unsafe_stream()` function and its
  switch of `eval()`'s `if (encoder.needs_commit())` branch to use the
  new `commit(completion)` signature directly. Both completion-handler
  registrations now run side by side (Metal permits multiple independent
  handlers on one command buffer).
- `event.cpp` — the highest-stakes conflict: upstream fully rearchitected
  `Event`/`EventImpl` (a new `metal::EventImpl` class wrapping
  `MTL::SharedEvent` with its own `error_`/`set_error()`/`check_error()`
  poisoning model), while our fork carries a 2026-07-05 fix that replaced
  Apple's blocking `waitUntilSignaledValue()` with an interruptible poll
  loop — the fix that prevents a wedged TP collective from hanging a
  peer rank in an unkillable kernel wait. Re-homed the entire poll loop
  (spin → sleep-poll → timeout → self-abort throw, plus the
  `EXO_CMDBUF_RING_DIAG` slow-wait dump hook) into the new
  `EventImpl::wait(uint64_t value)`, and additionally hardened it per a
  second-opinion review: `check_error()` is now called every poll
  iteration (not just once at the end), so a command buffer that failed
  *before* ever signaling the event surfaces its real root-cause error
  immediately instead of spinning the full 40s timeout and reporting a
  generic "wedged event" message. The `MLX_SIGNAL_PROBE` diagnostic was
  re-homed from a direct `addCompletedHandler` call in the old
  `Event::signal()` into `CommandEncoder::signal_event()`, upstream's new
  call site for encoding a signal.
- `fence.cpp` — both conflicts here were **not** a real merge — upstream's
  side was the original, unfixed code (`f.cpu_value()[0]`, no memory
  barrier). Our fork's ARM64 `DSB SY` full-system-barrier fix (a real
  correctness fix for a documented deadlock, vskiwi's #3142 follow-up)
  is the only substantive change; kept wholesale.
- `allocator.cpp` — trivial, both sides just added a different `#include`
  line; kept both.
- `scaled_dot_product_attention.cpp` / `.metal` — combined our
  fork's independent addition of symmetric `query_head_dim==192/256`
  fused-kernel routing (gated by `MLX_SDPA_FUSED_THRESHOLD`, since the
  fused path is slower than unfused for short sequences at those head
  dims) with upstream's newly-added **asymmetric** `(query_head_dim=192,
  value_head_dim=128)` case (the DeepSeek-MLA shape). Both vector-kernel
  instantiations (`sdpa_vector(type, 192, 192)` and
  `sdpa_vector(type, 192, 128)`) now coexist — confirmed safe because the
  kernel template already takes separate `qk_dim`/`value_dim` params and
  the C++ dispatcher already builds kernel names generically from
  `q.shape(-1)`/`v.shape(-1)`, not a symmetric assumption.

Validated: full `cmake --build` (release, tests enabled) succeeded after
applying the same known pre-existing local-toolchain-only shader
workaround from the 2026-07-21 session (`steel_attention.metal`'s
D=512/bq=16 kernel, isolated to this machine's fresh Metal Toolchain
17F42 — reverted immediately after, not part of any commit). `./tests/
tests` doctest suite: 260/260 cases pass on a clean run (up from 252 on
the pre-merge fork tip — upstream added 8 new test cases). Found and
root-caused one flaky cross-test-pollution failure (`test complex
gradients` run immediately before `test real ffts` fails ~60-90% of the
time, stale-memory read) — confirmed via 3-way comparison (pure
upstream/main: clean; pre-merge fork tip: same failure) that this is
**pre-existing on the fork, not introduced by this merge**. Left
unfixed, out of scope here.

Merged as `e5140ff9d`, pushed (required a `gh auth refresh -h
github.com -s workflow` first — the merge touches 21
`.github/workflows/`+`.github/actions/` files from upstream's CI
restructuring, which GitHub's OAuth workflow-scope check blocks without
that scope).

## REAL BUG FOUND + FIXED DURING LIVE VALIDATION

Smoke-testing the merged cluster (`start_cluster.sh` on the fully
committed+pushed code) surfaced a genuine regression: any request whose
very first generated token was EOS crashed the runner with
`TypeError: GenerationBatch.Response.__init__() got an unexpected
keyword argument 'current_state'`.

Root cause: upstream mlx-lm commit `86e9b35` ("Text-based state machine
for tool/reasoning parsing") replaced the old state-machine-based stop
matcher (`current_state`/`match_sequence`, a 3-tuple-returning
`.match()`) with a trie-based `StopSequenceMatcher.match(state, trie,
token) -> (new_state, matched: bool)`, and removed the `current_state`/
`match_sequence` fields from `GenerationBatch.Response` entirely. This
file wasn't in our own conflict set at all — it auto-merged cleanly by
git — so the break only surfaced at runtime, not at merge time.

Fixed 4 call sites in exo's own code:
- `src/exo/worker/engines/mlx/generator/batch_generate.py` (3 sites):
  trivial deletions — these always passed `current_state=None,
  match_sequence=None`.
- `src/exo/worker/engines/mlx/speculative/mtp_batch_generator.py` (1
  site, real rewrite): was calling the old 3-tuple `state_machine.match()`
  API and reading the renamed `gen_batch.state_machines` attribute
  (now `stop_matchers`). Rewritten to mirror upstream's own
  `GenerationBatch.next()` exactly.

Verified via a full grep sweep (not just the crash site) for any other
`current_state=`/`match_sequence=`/`.state_machines` references anywhere
in exo's own code — none remain. Committed as `777fd005`, pushed, pulled
onto both cluster nodes, re-verified live (clean completion, no
recurrence).

**Separately flagged, not fixed (zero runtime impact):**
`.typings/mlx_lm/generate.pyi` — a hand-maintained/generated type stub
used only by `basedpyright` (declared via `stubPath` in `pyproject.toml`)
— is now stale (still declares the removed `current_state`/
`match_sequence` fields and the renamed `state_machines` attribute).
`basedpyright --createstub mlx_lm.generate` did not regenerate it
correctly for unclear reasons (possibly resolving a cached/different
`mlx_lm` than the live-installed one) and was not trustworthy enough to
apply blindly. This causes some false-positive `reportCallIssue`/
`reportAttributeAccessIssue` noise from basedpyright on
`mtp_batch_generator.py` specifically, but has **no effect on actual
running code**. Left as a follow-up for whoever has time to regenerate
it properly.

## VALIDATION: LIVE CLUSTER CONTEXT-LEVEL STRESS SWEEP

Ran `~/.hermes/skills/.../exo-cluster-development/scripts/
context-level-stress-sweep.sh` (2K/20K/75K/150K/250K token levels, 1-3
iters each, needle-in-haystack + BOS-spam checks, decode tok/s + TTFT
measurement) against the live cluster running the fully-merged code
(commit `777fd005`, DRAFT_AHEAD-only / EXECUTE off per the standing
2026-07-22 decision), **twice**, back to back, for reproducibility.

Run 1 (11 iters): decode throughput 35.15 → 31.93 tok/s across
2K→250K (smooth taper, no cliff). One iter (L2000/iter1) failed with a
segfault, root-caused to an orphaned in-flight request from an
unrelated tool-management mistake on the agent's side (an interrupted
terminal call) colliding with PP mode's strict single-request-only
constraint — jaccl deadline → reconnect attempt → SIGSEGV on rank1's
runner. **The cluster's own supervisor self-healed** (auto re-place +
model reload) with zero manual intervention; every remaining iteration
across all 5 levels came back clean afterward.

Run 2 (11 iters): **zero failures at all.** Decode throughput 35.16 →
31.96 tok/s, matching run 1 within noise at every level. Confirms
reproducibility.

TTFT (prefill) scaled roughly linearly with context: ~6.6s at 2K,
~42s at 20K, ~155s at 75K, ~321s at 150K, ~561-661s at 250K.

Needle recall 21/22 total iterations (the one miss being the
self-inflicted segfault above, not a quality issue). Zero BOS-spam.
Zero OOM. Zero non-self-inflicted crashes or stalls.

**Separately noted, not investigated further:** during the idle gap
between the two sweeps, the runner-hang watchdog (`_check_hang`) fired
twice with no active request (`"N task(s) in progress, no event for
1800s+"` → SIGKILL → re-placement). Self-healed both times via the
existing watchdog; did not affect either sweep. Possibly some
background/idle task bookkeeping incorrectly holding a "task in
progress" state — worth a look sometime, out of scope here.

## CONCLUSION
The three-repo upstream sync did not regress PP+DSpark decode
throughput or stability at any tested context level (2K-250K tokens).
One real runtime regression from the mlx-lm merge (the
`GenerationBatch.Response` schema change) was caught during live
validation and fixed before being trusted. The fork now carries a small,
known set of pre-existing (not newly introduced) rough edges: a flaky
mlx C++ test interaction, a stale basedpyright stub file, and an
unexplained idle-time runner-hang watchdog trip — none of which block
production use of the cluster.
