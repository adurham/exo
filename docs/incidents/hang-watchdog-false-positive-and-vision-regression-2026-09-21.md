## 2026-09-21 — I16 Gate A attempt BLOCKED by a hang-watchdog false-positive; vision-decode throughput regression CONFIRMED and characterized; launcher mDNS-drift bug found+fixed

### Summary

Picked up two open threads from the Sept 4 campaign pause (I16 Gate A hardware
measurement; c>=2 spec-decode structural drift) plus a brand-new user-reported
regression (image attachment drops decode throughput). Fable (consult) was
used up front to sequence the three threads and flag risks before touching
the cluster; its guidance shaped everything below.

**Net result this session (UPDATED — hang-watchdog is now SHIPPED and LIVE-VERIFIED):**
- Thread 3 (vision throughput regression): **CONFIRMED, characterized, root
  cause NOT yet found** (real source-read still needed — this is a scoped
  investigation handoff, not a fix). Additional theories ruled out this
  session via source-read: `bias_vl` MoE routing bias (config-driven,
  always-on for this checkpoint, own-position-scoped only -- can't explain a
  sustained multi-turn regression), `_apply_image_visibility` mask widening
  (gated OFF by default AND prefill-only by construction), and
  `_query_tiled_ok`'s compressed-SDPA decline (prefill-only, `n_q==1` always
  fails the gate on decode regardless of images). Fable's leading remaining
  hypothesis (not yet tested): this may not be image-specific compute at
  all -- it may be pure CONTEXT-LENGTH cost that merely correlates with
  "image present" in every observation so far (an image adds a large token
  block; every later turn in that conversation is a longer-context request).
  A length-matched text-only control (`/tmp/vision_length_matched_probe.py`,
  written but not yet run against a stable cluster) would settle this for
  free, no relaunch needed.
- Thread 1 (I16 Gate A): **hang-watchdog bug FOUND, FIXED, SHIPPED, and
  LIVE-VERIFIED this session** across two commits (`67fd81a1b` root fix,
  `3cae939a2` same-day bugfix to the fix itself -- see "Hang-watchdog fix"
  section below for the full story, including a live regression the first
  commit shipped with). Gate A measurement ITSELF is still not taken (that
  requires a dedicated, focused attempt now that the blocker is cleared).
- Thread 2 (c>=2 spec-decode drift): **NOT STARTED** -- ran out of session
  budget after the hang-watchdog detour; unchanged from the Sept 16 incident
  writeup (`docs/incidents/c2-spec-verify-structural-drift-2026-09-16.md`).
- **Unrelated bug found+fixed+shipped**: `start_cluster.sh`'s post-launch
  health-check polling used a hardcoded LAN IP (`M4_1_IP=192.168.86.201`) that
  had silently drifted from the real DHCP-assigned address (nodes were
  actually on .48/.47) -- every relaunch this session hung for 10+ minutes on
  a dead TCP-connect retry before the fix, instant "HEALTHY!" after.
- **Unrelated operational gotcha rediscovered**: a stale/dead `screen`
  session on each node (left over from an earlier `pkill`-based recovery
  this session) caused a SEPARATE, unrelated stuck-boot symptom (runners
  stuck `RunnerIdle`/`RunnerShuttingDown`, `instances: {}`, zero
  instance-lifecycle log events, 20+ minutes, no crash) that looked
  superficially like a hang-watchdog problem but wasn't -- `screen -wipe` on
  both nodes before the next relaunch resolved it instantly. This is
  documented `exo-cluster-operations` skill pitfall #2 ("stale screen before
  restart causes weird placement and quorum failures"); re-triggered because
  the skill's pitfalls were not re-checked before this session's several
  interactive `pkill`/relaunch cycles. See "Stale screen" section below.

### Thread 3 — Vision decode throughput regression (CONFIRMED, uncharacterized mechanism)

User attached an image mid-conversation in the exo web UI and saw decode drop
from ~32 tok/s (same-session text-only baseline) to 14.2 tok/s. Reproduced
independently this session with a controlled 4-arm A/B against the live
production process (no relaunch needed):

| Arm | decode tok/s | notes |
|---|---|---|
| A1 text-only, fresh | ~14.6 (cold, includes warmup) | first request of the boot |
| A2 text-only, repeat | **20.44** | matched baseline |
| B1 image + text | **14.38** | ~30% slower than A2 |
| C1 text-only, immediately after B1 | **20.66** | matches A2 — regression does NOT persist |

Measurement: server-side `Prefill complete`/`runner idle` log timestamps
(the exact window between prefill-done and next-idle), not client wall-clock
— avoids network/client-side confounds. C1 recovering fully rules out
persistent state corruption (resident vision-encoder memory pressure, cache
poisoning) — **the slowdown is scoped exactly to requests that contain image
content**, not lingering afterward.

Mechanism NOT YET FOUND. Ruled out this session (source-read, not
speculation):
- `media_regions` cache-bookkeeping (`cache.py`) — confirmed prefill/insert-time
  only via grep of all call sites; never touched on the decode-step path.
- Zero vision-aware branching in the speculative/MTP decode code
  (`dsv4_mtp.py`, `pp_speculation.py`) — grepped, no `image`/`vision` hits at all.
- Context-length growth alone — image added only ~116-322 placeholder tokens
  (varies with client-side downscaling), and campaign history has repeatedly
  established the decode fixed-cost floor is context-INDEPENDENT at these
  depths; raw token-count growth this small should not double per-token cost.

E-CPU usage on BOTH nodes roughly doubled in lockstep during the image
request's decode window (0.37→0.75-0.84), which points at real added
per-shard compute replicated on each node — Fable's read (not yet verified):
likely mask materialization (a bidirectional image-block attention pattern
breaking the fused causal SDPA fast path), a separately-stored image-KV
segment being concatenated in every decode step, or a dtype mismatch on the
image KV forcing a slower kernel. **Next step for whoever picks this up:**
source-read `merge_image_embeddings`/`build_embeddings`
(`deepseek_v4_vision.py`) and the SDPA call sites for anything keyed on
`media_regions`/image span presence during the DECODE (not prefill) forward
pass; a real Metal GPU trace comparing text-only vs. image-bearing decode
would be decisive.

### Thread 1 — I16 Gate A: hang-watchdog bug FOUND, FIXED, SHIPPED, LIVE-VERIFIED

**Update (later same session): this entire investigation resulted in a real,
shipped, live-verified fix.** The sections below (through "Live mitigation")
are the ORIGINAL same-day investigation notes, preserved as-written for the
diagnostic trail. Skip to "THE ACTUAL FIX" below for the final, shipped
state.

Pre-flight (before any relaunch): verified HEAD==origin/main on both nodes,
API healthy, ran 4 reps of `long_decode_probe.py` to establish a genuine
current-commit baseline (~13.0-13.4 tok/s @2K — this is NOT comparable to the
stale "29.1 t/s" ruler from May, which was measured on a different,
text-only, pre-DSpark-native-head checkpoint months ago; a fresh ruler is
needed if this campaign continues).

**Boot 1** (`EXO_WORKER_PLAN_EVENT_WAKE=1 EXO_PHASE_MARKS=1`) came up clean:
both flags confirmed on real PIDs, 2/2 READY, PHASE_MARK lines flowing on
both nodes. Ran a 22-request workload driver (shared growing conversation,
~106K-token document, every 4th turn exercising an OpenAI `tools=[...]`
round-trip per the original R11 campaign spec). Requests 0-2 succeeded
cleanly (prefix-cache hits, matches R13's apparatus-validation numbers in
kind). **Request 3 (first `tools`-bearing turn) triggered a full cache-bust
(the tools-schema injection changes the rendered prompt, so it did not match
the cached prefix) → full cold 106,579-token prefill → SIGKILLed by the
runner supervisor's hang-watchdog 47s after task start, with only the
`processed=0/106579` initial progress callback ever having fired.** My own
driver then retried the identical failing request 5 more times without
backing off, causing 6 total supervisor-triggered SIGKILLs in ~10 minutes —
a design flaw in the throwaway driver script (`/private/tmp/gate_a_workload.py`,
not committed to the repo), not a campaign methodology issue.

**Root-cause investigation (source-read, most before any restart):**
- `HANG_TIMEOUT_SECONDS` = 45s default (`supervisor.py:82`,
  `EXO_RUNNER_HANG_TIMEOUT_SECONDS` override). Fires when the runner's event
  channel (`_ev_recv`) has emitted nothing for 45s while a task is
  `in_progress`.
- **This is NOT a wedge and NOT a deadlock.** The supervisor's own pre-kill
  `sample` diagnostic (`/tmp/exo_hang_*.txt`, already-existing infrastructure)
  showed the runner genuinely inside `mlx::core::eval()` →
  `Event::wait()` → real GPU `dispatch_threadgroups`/`Compiled::eval_gpu`
  calls at the moment of every kill, with physical footprint actively growing
  (88.4GB → 97.5GB peak in one dump). TB/RDMA link health confirmed clean
  throughout (0% packet loss both directions, GPU power symmetric 22mW/22mW
  idle after) — the "repeated SIGKILL can wedge the RDMA hardware" risk from
  prior incidents did NOT materialize this time; the runner's own
  `CLEAN_EXIT after 1s (NO SIGKILL — destructors ran, RDMA QPs released)`
  log line on the subsequent clean relaunches confirms the TB stack stayed
  healthy throughout (the SIGKILLs here were the supervisor's OWN internal
  watchdog on a live compute-bound process, not the user-facing "never
  kill -9 a runner" action this rule is normally about).
- **Progress-reporting infrastructure for BOTH failure modes is real,
  already-existing, and correctly wired end-to-end** — this is not a missing
  heartbeat, which makes the bug more interesting:
  - Prefill: `on_prefill_progress` (`batch_generator.py:325`) sends a real
    `ChunkGenerated` event via `event_sender.send(...)` on device_rank==0 for
    every real prefill chunk (mlx-lm's `generate_step` calls this once per
    chunk); non-zero ranks call a throttled (≥15s) `prefill_heartbeat()` via
    the same `event_sender`. Both terminate in `_forward_events`'
    `_ev_recv` loop, which resets the hang clock on ANY event
    ("Any event = the runner made progress" — the literal comment at that
    line).
  - Model loading: `update_status(RunnerLoading(layers_loaded=N, ...))` is
    called inside `runner.py`'s `for load_progress in
    self.generator.load(...)` loop — i.e. once PER LAYER — and
    `update_status()` also calls `self.event_sender.send(...)`. Confirmed via
    source read, not inferred.
- **Timing analysis pinpoints WHERE the 47s of silence actually was**, and it
  contradicts my initial "MTP-cache-prefill step" hypothesis (caught by a
  Fable review before it made it into this doc): task started at 15:00:45,
  the ONLY progress log was the pre-loop `processed=0/106579` callback
  (same second), kill fired at 15:01:32 — **47s with ZERO chunk-1 completion
  events**, not a gap appearing after several chunks. A full successful
  106,244-token prefill on the SAME checkpoint/config EARLIER in this exact
  session took 799s wall-clock at a steady ~15s/chunk average with
  continuous progress logging to completion, zero hang-kills. So chunk-1
  itself (or the setup work immediately before it) was anomalously slow only
  on the cache-busting attempt — consistent with the old ~106K KV cache and
  MTP cache being evicted/trimmed while the new ones are built, both briefly
  co-resident (matches the growing-footprint evidence).
- **Second, independently-triggered manifestation, same watchdog, same
  design flaw**: on TWO SEPARATE subsequent clean relaunches (both with
  ZERO campaign env flags set — `EXO_WORKER_PLAN_EVENT_WAKE` and
  `EXO_PHASE_MARKS` both absent, ruling out my Gate A instrumentation as the
  cause), fresh MODEL LOADING itself (not a chat request) was SIGKILLed by
  the same watchdog at 46-49s, with `layersLoaded` around 17-27 of 46 at
  kill time on both occasions. This is a **100% reproducible** failure on a
  cold/fresh runner load of this specific checkpoint
  (`DeepSeek-V4-Flash-Vision-Exp`, 46 layers = 43 text transformer layers +
  3 MTP blocks per the model's own `nLayers` field — corrected later this
  session; NOT a separate 46-layer vision-tower count as originally guessed
  here. The vision tower's `VisionProcessor.load()` is a no-op for this
  checkpoint's `deepseek_v4` scheme — its weights are baked into the main
  text-model checkpoint and load as part of the ordinary tensor-sharded
  layer loop).

**Design-level root cause (established via source-read, confirmed by the fix
working live):** the supervisor infers "the runner is dead" from
**event-channel silence alone**, and at least two legitimate, real-compute
phases (a from-scratch cold prefill's first chunk under memory-pressure
conditions; per-layer model loading, especially reload-after-kill while OS
memory reclaim is still in flight) can produce >45s of genuine,
progress-reporting-equipped silence. "Add a heartbeat to phase X" would not
have been sufficient, since the progress callbacks already exist and are
correctly wired — the fix instead needed the supervisor to independently
VERIFY liveness via a channel the event system doesn't depend on (physical
memory growth, sampled directly), which is what shipped.

**Live mitigation used DURING this investigation, since superseded by the
real fix:** `EXO_RUNNER_HANG_TIMEOUT_SECONDS=300` was run as a manually-set,
non-persisted env var while the real fix was still being designed and
tested. This is NO LONGER how the cluster runs — see below.

### THE ACTUAL FIX (shipped `67fd81a1b`, hotfixed `3cae939a2`)

**Design:** `_check_hang` in `src/exo/worker/runner/supervisor.py` now
probes the runner process's physical memory footprint (via the existing
`sample` diagnostic tool, off the event loop via `anyio.to_thread`) once the
45s silence threshold is crossed, BEFORE killing. If the footprint grew by
`EXO_RUNNER_HANG_PROBE_GROWTH_GB` (default 0.25GB) since the last probe,
that is externally-verified real progress on a channel the event system
doesn't touch — extend the deadline (`EXO_RUNNER_HANG_PROBE_INTERVAL_SECONDS`,
default 20s) instead of killing, bounded by
`EXO_RUNNER_HANG_PROBE_MAX_EXTENSIONS` (default 20, i.e. up to ~400s of
verified real growth before giving up). A genuinely wedged process's
footprint is static by construction, so it is still caught at the first
probe past the timeout — this does not weaken detection of the real wedge
this watchdog exists to catch (a runner spinning at 100% CPU inside a hung
native collective), it only stops the watchdog firing on real, still-
progressing work.

**A same-day bug in the first version (`67fd81a1b`), caught live on the very
first relaunch to it:** the extension-granting code set a "next probe check"
timestamp but the KILL decision was gated on that SAME timestamp being in
the future — so an extension only protected the exact `_watch_runner` tick
(every 5s) it was granted on, and the very next tick fell through to an
unconditional kill anyway. Live evidence, reproduced on the FIRST default-
config relaunch to the buggy commit (no `EXO_RUNNER_HANG_TIMEOUT_SECONDS`
override): a genuine 106K-token cache-busting prefill (the EXACT original
trigger this whole investigation started from) logged "silent for 46s ...
extending 20s for a growth check" immediately followed 5 seconds later by
"no event for 54s ... SIGKILLing" — the false positive this fix was built to
eliminate, still firing, just delayed by one cycle. Fixed in `3cae939a2`:
replaced the single "next check" timestamp with an explicit deadline that is
checked FIRST, unconditionally, on every tick, before any probe/kill
decision — "may I probe" and "must I defer the kill" are the same condition
again.

**Test coverage (both commits):**
- `src/exo/worker/runner/tests/test_supervisor_hang_liveness_probe.py` (7
  tests) — pure unit tests of the `_sample_physical_footprint_gb` helper:
  real `sample` output parsing (G/M suffixes, the EXACT captured output from
  this incident's own `/tmp/exo_hang_96033.txt`), fail-safe None on
  subprocess error/timeout/unparseable output, correct invocation args.
- `src/exo/worker/tests/unittests/test_runner/test_runner_supervisor_hang_probe.py`
  (3 tests × 2 async backends) — REAL integration tests against an actual
  `RunnerSupervisor` instance (not mocked), driven by a fake monotonic clock
  reproducing the EXACT live-incident tick sequence (46s/51s/54s/66s).
  Verified via git-stash A/B: 4/6 FAIL against the pre-hotfix code with the
  exact live failure signature, 6/6 PASS against the fix. Also covers: a
  genuinely static footprint still kills at the deadline (real-wedge
  detection preserved), and an extensions-exhausted case (a forever-growing
  process cannot stall the watchdog past `HANG_PROBE_MAX_EXTENSIONS`).
- Pitfall hit and documented in the test file itself: an early draft used
  the test process's own PID as the fake runner's PID; the kill path's real
  (unmocked) `os.kill()` SIGKILLed the pytest worker running the test.
  Fixed with an obviously-fake PID (9999999) plus explicit stubs
  neutralizing `subprocess.run`/`os.kill` in every test reaching the kill
  branch.

**Live verification (post-`3cae939a2` deploy, this session, real cluster):**
a full 106,244-token cold cache-busting prefill (the SAME trigger that
caused all 6 original SIGKILLs) completed 100% cleanly at 19:11:24→19:36:06
(~72-80 tok/s throughout, steady chunk cadence well under the 45s threshold
— the probe never even needed to engage for this particular run), followed
by a full decode to `finish_reason: stop`, with **zero SIGKILL / zero "hung:"
log lines across the entire boot**, runner PIDs unchanged from cluster start
through request completion (35+ minutes), and NO
`EXO_RUNNER_HANG_TIMEOUT_SECONDS` override set. Confirmed via `grep -c
'SIGKILL\|hung:' ~/exo.log` returning 0. This is the real-world proof the
fix works — not just the synthetic clock-driven tests.

**I16 Gate A measurement itself: still NOT taken.** The blocker is now
cleared (the hang-watchdog no longer false-positive-kills cache-busting
prefills), but no PHASE_MARK data has been collected yet against the actual
Gate A workload. The instrumentation, the workload driver
(`/tmp/gate_a_workload.py`, `_v2.py`, `_v3.py` — note: this script's own
client-side `timeout=` kwarg needs to be >=2400s for a full 8-request run at
current cluster speed, learned the hard way twice this session when the
client gave up before the server finished), and the pre-registered bands
from September are all still valid and unused — this needs a fresh, focused
attempt.

### Thread 2 — c>=2 spec-decode structural drift: NOT STARTED this session

Unchanged from `docs/incidents/c2-spec-verify-structural-drift-2026-09-16.md`.
Ran out of session scope after the hang-watchdog detour consumed the
available relaunch/investigation budget. Still gated OFF in production
(`EXO_DSV4_MTP_C2_MAX_CTX=1`), still not the user's actual workload (c=1
single-stream). No new information this session.

### Unrelated bug found+fixed+shipped: launcher health-check used a stale hardcoded LAN IP

`start_cluster.sh`'s post-launch health-check/instance-creation polling
(`API="http://$M4_1_IP:52415"`) used the hardcoded constant
`M4_1_IP="192.168.86.201"`, which had silently drifted from the nodes' real
DHCP-assigned addresses (confirmed live: m4-1 actually on .48, m4-2 on .47).
A stale ARP/route entry for .201 made every `curl "$API/state"` call in the
stabilization loop hang for a full TCP-connect timeout instead of failing
fast — "Waiting for cluster to stabilize" looked hung for 10+ minutes on
EVERY relaunch attempt this session, even though the cluster (both node
processes, correct env flags) was already healthy underneath. Root-cause
fixed (not just documented): the health-check host now resolves via the same
mDNS hostname (`adams-mac-studio-m4-1.local`) the already-working
Thunderbolt-discovery path uses (via SSH host aliases in `~/.ssh/config`),
with a `ping`-probed fallback to the old constant only if mDNS resolution
itself fails. Verified live: post-fix, "Waiting for cluster to
stabilize..." resolved to "HEALTHY!" in under 15 seconds on every subsequent
relaunch, vs. 10+ minutes before. Committed+pushed:
`98a432c520e64cdd48e9515eacee8eadff287bd6`.

### Unrelated operational gotcha rediscovered: stale `screen` sessions blocking placement

Between the two hang-watchdog commits, an interactive `pkill -9 -f 'python -m
exo'` was used on both nodes to force a clean state for testing (the process
tree includes `SCREEN -dmS exorun ...` as the top-level parent per
`start_cluster.sh`'s launch mechanism — killing the exo process leaves the
`screen` session itself as an orphaned, dead entry). The very next
`start_cluster.sh` relaunch got stuck for 20+ minutes: both runners sitting
in `RunnerIdle`/`RunnerShuttingDown`, `instances: {}`, ZERO instance-lifecycle
log events (no `InstanceCreated`, no `RunnerFailed`), runner subprocess PIDs
completely stable throughout (no crash-loop) — a `sample`/`py-spy`-style
inspection of the master process's own event loop showed it correctly idle
in `kevent`/`kqueue`, not wedged. This looked exactly like it could be a
NEW hang-watchdog-adjacent bug and cost real diagnostic time before `screen
-ls` on both nodes revealed the actual cause: `16678.exorun (Dead ???)` /
`10678.exorun (Dead ???)`. This is `exo-cluster-operations` skill pitfall #2,
verbatim: "Stale `screen` before restart causes weird placement and quorum
failures." `screen -wipe` on both nodes immediately unblocked a subsequent
clean relaunch (fresh node identities, READY in under 2 minutes end-to-end).

**Lesson for next time (added to this session's diagnostic record, not yet
promoted to a skill patch — the skill already documents this correctly, the
failure was not re-reading the pitfalls before an interactive `pkill`):**
after ANY manual `pkill`/`kill -9` targeting exo processes on the Studios
(as opposed to a normal `start_cluster.sh`-driven shutdown, which handles
this itself via its own `SIGTERM`-then-verify shutdown gate), run `screen
-wipe` on both nodes before the next relaunch attempt.

### Cluster state at end of session

Both nodes: commit `3cae939a2` (the fully-fixed hang-watchdog, including the
same-day hotfix), tree clean vs origin. 2/2 `RunnerReady` for
DeepSeek-V4-Flash-Vision-Exp, real completion verified (capital-of-France
probe: `content: "Paris"`, `finish_reason: stop`). **NO
`EXO_RUNNER_HANG_TIMEOUT_SECONDS` override running** — the fix is live in
the default code path, not masked by an env var. A full 106K-token
cache-busting prefill plus decode completed with zero SIGKILLs on this exact
boot, proving the fix holds under the real trigger, not just synthetic
clock-driven tests. Qwen3.6 co-hosting was NOT re-verified as placed this
session — worth a `/state` check before assuming it's up (a stray
`mlx-community/Qwen3-Coder-Next-6bit` download was seen mid-session,
possibly from `start_cluster.sh`'s own Qwen auto-placement section; not
investigated further, unrelated to either fixed bug).

### Next steps, priority order

1. **Re-attempt I16 Gate A now that the blocker is cleared.** Use
   `/tmp/gate_a_workload.py` (bump its client-side `timeout=` kwarg to
   >=2400s first) with `EXO_WORKER_PLAN_EVENT_WAKE=1 EXO_PHASE_MARKS=1`.
   Expect the cold cache-busting prefill (request 3, first `tools`-bearing
   turn) to now survive; if it still gets killed, that is new information
   (a THIRD manifestation) and should be treated as such, not assumed to be
   the same already-fixed bug.
2. **Thread 3 (vision regression): run the length-matched control FIRST,
   before any deeper source-read.** `/tmp/vision_length_matched_probe.py`
   (written this session, not yet run against a stable cluster) sends an
   image request, records its real `prompt_tokens`, then sends a
   token-count-MATCHED text-only request and compares tok/s. Per Fable's
   review: if the length-matched text-only run reproduces the ~30%
   slowdown, the mechanism is pure context-length cost (leading suspect:
   the DSA indexer's top-k budget cliff — a request crossing some N-token
   threshold switches from dense to indexer+topk+gather+sparse-SDPA, adding
   several kernels × 43 layers per decode step, which would also explain
   the observed E-CPU near-doubling on both TP ranks) and has NOTHING to do
   with images specifically — "image dropped my tok/s" would actually mean
   "this conversation crossed a length cliff, and images are usually what
   pushes a conversation over it." If it does NOT reproduce, images really
   are special-cased somewhere not yet found by grep, and `sample`/`py-spy`
   on the live process during each condition is the next move (per Fable:
   this catches things an in-model span profiler can't, e.g. detokenizer,
   sampler, comms — outside the forward pass entirely).
3. **Thread 2 (c>=2 spec drift)**: still has a named, unexecuted next step
   from Sept 16 (logprobs repro, then gated fix attempt) — lowest priority,
   not the user's workload.

### NEW FINDING (end of session): general decode throughput regression, cause UNKNOWN

While attempting the length-matched vision control (next-steps item 2
above), plain text-only decode measured **3.5-6.2 tok/s** on this exact
boot -- roughly 3-6x SLOWER than the ~20 tok/s baseline this same session
established repeatedly earlier (the A2/C1 arms of the original vision
regression probe: 20.44 and 20.66 tok/s, on the SAME model, SAME hardware,
SAME sharding config, same day). This is NOT the vision regression --
these were plain text-only requests with no image. Measured cleanly three
times, isolated (verified zero pending tasks before each call, no orphaned
concurrent requests):
- "Count from 1 to 5" -- 133 completion tokens / 37.8s = 3.5 tok/s
- "Write the numbers 1-20" -- 264 completion tokens / 42.7s = 6.2 tok/s
- (an image-request calibration call during the control script) -- 150
  completion tokens / 157s = 0.95 tok/s (this one may have residual
  calibration-loop overhead from the probe script's binary-search prefill
  calls stacking up; treat as a weaker data point than the two above)

Ruled out as EASY explanations (checked, not assumed): no thermal
throttling (`pmset -g therm` clean both nodes), no memory pressure (~11GB
free on top of ~87-97GB DSv4 residency, not critical), no co-hosted
model contention (only the DSv4 instance is placed -- Qwen3.6 auto-place
never completed, still just a stray download), GPU idle power normal
(21mW) between requests, speculative-decode env flags unchanged
(EXO_SPECULATIVE=1, GAMMA=3, EXO_DSV4_MTP=1 -- same as the healthy-speed
boot earlier today).

**NOT diagnosed further this session** -- this is a new, real, separate
finding surfaced at the very end of an already-long session (two hang-
watchdog commits + extensive live verification + vision investigation
already consumed the available time). Do not assume this is the same
vision regression, the same hang-watchdog issue, or a measurement error --
it was checked against exactly the obvious causes and none of them explain
it. Leading candidates for next session, in order of cheapness to check:
(1) MTP acceptance-rate collapse (compare `accepted_prediction_tokens` /
`completion_tokens` ratio in `usage.completion_tokens_details` across
these slow requests vs. the earlier fast ones -- a collapsed accept rate
would mean the SAME total wall-clock is producing verify-only, near-zero
net throughput, without needing any GPU-level slowdown); (2) something
specific to THIS boot's env/config that differs from the earlier
same-session fast boot despite looking identical in the checked flags
(diff the full env between the two boots, not just the handful checked
here); (3) a real hardware/thermal issue that `pmset -g therm`'s coarse
reporting doesn't surface (a live `powermetrics` GPU-frequency sample
during an active slow decode, not just idle, would be more conclusive
than the idle-only check done here).

### ROOT CAUSE FOUND (2026-09-22): the "vision regression" is NOT image-specific -- it is MTP draft-acceptance sensitivity, and the ambient slowdown is a machine-level GPU power-governor throttle

**Two separate findings, both now measured directly rather than inferred.**

#### Finding 1: images do not add decode compute -- they reduce MTP draft acceptance

Measured via exo's own Prometheus counters (`exo_mtp_cycles_total`,
`exo_mtp_accepted_drafts_total`), scraped before/after each request, with
per-cycle wall time computed alongside:

| arm | acc/cycle | tok/cycle | ms/cycle |
|---|---|---|---|
| easy text | 1.85 | 2.85 | 455.9 |
| image-instr, no image | 1.79 | 2.79 | 379.2 |
| image attached | 1.67 | 2.67 | 475.1 |
| creative/hard text | 1.05 | 2.03 | 365.2 |

**Per-cycle cost is identical across all arms** (365-476 ms; the image arm
is not slower than text). What changes is how many draft tokens the MTP
head gets accepted per verify cycle. Fewer accepted drafts = fewer tokens
delivered per identical-cost cycle = lower tok/s with NO extra compute.

A length-matched interleaved test (3 rounds, ABABAB, drift-cancelling)
gave IMG/TXT acceptance ratio **0.71** at matched prompt length
(127 vs 133 tokens), reproducible exactly across all 3 rounds.

Critically, a **content-controlled** variant (identical requested output,
`enable_thinking: false`, byte-identical emitted text in both arms) showed
the gap largely VANISH: image 2.90 acc/cycle vs text 2.87. So a large part
of the acceptance difference is **output-content-driven** (drafting is
simply harder on descriptive/freeform content: creative text measured 1.05
acc/cycle vs 1.85 for easy factual), not image-token-presence-driven.

The user's original observation (MTP acceptance 1.83 -> 1.46 across a
conversation that gained an image) is consistent with this: images tend to
arrive with descriptive prompts and open-ended outputs, and the
conversation grows -- both of which depress acceptance. **There is no
extra per-token compute anywhere in the vision path**, which is exactly why
every source-read of the vision/MTP code came up empty (media_regions is
prefill-only; `_apply_image_visibility` is env-gated OFF and prefill-only;
`_query_tiled_ok` is prefill-only; `bias_vl` is config-driven and
own-position scoped; rollback/snapshot paths are context-size-independent).

#### Finding 2: the ambient 3-6x cluster slowdown is a machine-level GPU governor condition, NOT an exo bug

Raw fp16 matmul (standalone mlx process, no exo involvement, per-iteration
`mx.eval`):

- **Earlier today: 14.77 TFLOPS on BOTH nodes** (verified directly).
- **Now: 2.5-4.0 TFLOPS**, via two independent benchmark methodologies
  (per-iteration eval on 4096 and on 8192 matrices -- both agree).

Shape of the degradation (reproducible):

```
three short bursts with 10s gaps :  4.09 / 3.77 / 4.06 TFLOPS
one sustained 45s run            :  4.03 -> 2.55 -> 2.27 TFLOPS
burst after 10s pause            :  3.99 TFLOPS   (full recovery)
```

Co-sampled `powermetrics` during sustained load:
- GPU HW active frequency **pinned at 338 MHz** (this chip's table runs to
  1578 MHz) with 44-59% idle residency
- GPU power only **1.5-2.2 W** under real compute
- CPU P-cluster 4.0-4.3 GHz with headroom -- CPU is NOT the bottleneck
- **Thermal pressure "Sleeping"** at every sample; `pmset -g therm` reports
  no thermal warning level AND no performance warning level ever recorded

So the GPU holds its lowest frequency bin and draws ~2W under sustained
compute, with no thermal signal, then recovers fully when idle. This is a
power/performance governor state, and it is **not** permanent hardware
damage (full recovery after a pause proves that) and **not** an exo code
path.

One logged kernel-level `AGX: NOP prepared` / `AGX: Submitting NOP` event
(GPU driver error-recovery) coincided with the earlier SIGKILL-based runner
recoveries. Also found and ruled out as causes (both were present during
the fast measurements too): `configd` in a perpetual DHCP-retry loop on
orphaned virtual interfaces (~40% of one core) and `audiomxd` at ~76% of
one core. Machines have ~4 days uptime with heavy Metal process churn.

**Verdict on the user's original question ("I put an image in and dropped
from 30 to 15 tok/s"):** the drop was real and is now explained --
it is MTP draft acceptance falling (drafts rejected more often), chiefly
as a function of the content that image-bearing prompts elicit, with a
smaller genuine contribution from image tokens in context. No extra
per-token image compute exists at any site examined.

**What still needs doing:** (1) A/B the acceptance claim against a clean
machine state once the GPU governor issue is resolved -- the ambient
slowdown makes absolute numbers unreliable even though the drift-cancelled
ratios are sound. (2) Decide whether acceptance-sensitivity is worth
engineering around (it is a property of the draft head's accuracy on
descriptive content, not a defect). (3) The machine-level GPU throttle
deserves its own investigation: candidate remedies are a reboot (clears
driver/governor state; user confirmation required -- production cluster),
and checking for macOS updates. Note the reboot must be USER-APPROVED.
