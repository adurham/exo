## 2026-09-21 — I16 Gate A attempt BLOCKED by a hang-watchdog false-positive; vision-decode throughput regression CONFIRMED and characterized; launcher mDNS-drift bug found+fixed

### Summary

Picked up two open threads from the Sept 4 campaign pause (I16 Gate A hardware
measurement; c>=2 spec-decode structural drift) plus a brand-new user-reported
regression (image attachment drops decode throughput). Fable (consult) was
used up front to sequence the three threads and flag risks before touching
the cluster; its guidance shaped everything below.

**Net result this session:**
- Thread 3 (vision throughput regression): **CONFIRMED, characterized, root
  cause NOT yet found** (real source-read still needed — this is a scoped
  investigation handoff, not a fix).
- Thread 1 (I16 Gate A): **BLOCKED again** — not by the tool-permission wall
  that blocked it in September, but by a **real, independently-discovered
  hang-watchdog bug** that this session traced to two concrete manifestations
  and partially root-caused. A live mitigation (raised timeout) is running
  right now; the actual fix is NOT shipped.
- Thread 2 (c>=2 spec-decode drift): **NOT STARTED** — ran out of session
  budget after the hang-watchdog detour; unchanged from the Sept 16 incident
  writeup (`docs/incidents/c2-spec-verify-structural-drift-2026-09-16.md`).
- **New, unrelated bug found+fixed+shipped**: `start_cluster.sh`'s post-launch
  health-check polling used a hardcoded LAN IP (`M4_1_IP=192.168.86.201`) that
  had silently drifted from the real DHCP-assigned address (nodes were
  actually on .48/.47) — every relaunch this session hung for 10+ minutes on
  a dead TCP-connect retry before the fix, instant "HEALTHY!" after.

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

### Thread 1 — I16 Gate A: BLOCKED by hang-watchdog false positives (2 confirmed manifestations)

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
  co-resident (matches the growing-footprint evidence). **NOT yet confirmed
  which specific operation inside chunk 1 was slow — this is the honest
  remaining gap**, not a closed root cause.
- **Second, independently-triggered manifestation, same watchdog, same
  design flaw**: on TWO SEPARATE subsequent clean relaunches (both with
  ZERO campaign env flags set — `EXO_WORKER_PLAN_EVENT_WAKE` and
  `EXO_PHASE_MARKS` both absent, ruling out my Gate A instrumentation as the
  cause), fresh MODEL LOADING itself (not a chat request) was SIGKILLed by
  the same watchdog at 46-49s, with `layersLoaded` around 17-27 of 46 at
  kill time on both occasions. This is a **100% reproducible** failure on a
  cold/fresh runner load of this specific checkpoint
  (`DeepSeek-V4-Flash-Vision-Exp`, 46 layers including an unquantized vision
  tower — larger than the historical DSv4 text-only 43-layer count this 45s
  default was presumably tuned against). One open, undiagnosed asymmetry:
  the SAME checkpoint loaded successfully under the 45s default earlier in
  this exact session (that's how the 799s successful prefill above was even
  possible) — all 3 loading-phase failures were specifically on
  RELOADS-after-a-prior-SIGKILL, not a cold first boot. Not yet established
  whether reload-after-SIGKILL is genuinely slower (OS still reclaiming ~97GB
  of wired memory from the just-killed process) or whether first-load vs.
  reload arms the watchdog differently.

**Design-level root cause (this part IS established, not speculative):** the
supervisor infers "the runner is dead" from **event-channel silence**, and at
least two legitimate, real-compute phases (a from-scratch cold prefill's
first chunk under some not-yet-isolated condition; and per-layer model
loading under some not-yet-isolated condition) can produce >45s of genuine,
progress-reporting-equipped silence. The fact that prefill's progress
callback IS wired and STILL produced a >45s gap on the cache-bust case means
"add a heartbeat to phase X" is not automatically sufficient — the real fix
needs either the supervisor consuming the diagnostic it already collects
(the `sample` stack dump — same-stack-twice-in-a-row + non-growing footprint
= genuine hang; growing footprint / different stack = extend) rather than a
fixed wall-clock, or finding and fixing whatever specific sub-operation
within "chunk 1 of a cache-busting prefill" and "loading a reload-after-kill
model" is failing to emit its already-existing progress event in those two
specific circumstances.

**Live mitigation, explicitly NOT the fix:** `EXO_RUNNER_HANG_TIMEOUT_SECONDS=300`
passed as a manually-set env var on the current live launch (not committed
to `start_cluster.sh`, not persisted). The cluster IS healthy right now
(2/2 RunnerReady, real completion verified, both nodes on
`98a432c520e64cdd48e9515eacee8eadff287bd6`) BUT **a future bare relaunch
without this override will very likely reproduce the same loading-phase
failure** — this is the single most important thing for whoever picks this
up next to know. The codebase already has a precedent for this exact
timeout-widening pattern (`start_cluster.sh` auto-sets
`EXO_RUNNER_HANG_TIMEOUT_SECONDS=1800` for `DSV4_SHARDING=Pipeline`, this
cluster uses the default `Tensor` sharding which gets no such override) —
extending that precedent to Tensor sharding was considered and explicitly
REJECTED as the fix here (per the user's root-cause-only standing rule and a
second consult review): it would hide, not fix, both manifestations, is
keyed on a speculative heuristic (layer count / vision-tower size) rather
than the actual variable (bytes-to-materialize ÷ real bandwidth), and would
degrade detection of the exact class of GENUINE hang (a truly deadlocked
collective) this watchdog exists to catch, for the runner's entire lifetime,
just to buy headroom in one phase.

**I16 Gate A measurement itself: still NOT taken.** Zero PHASE_MARK data was
collected against real request-path traffic before the boot got consumed by
this investigation. The instrumentation, the workload driver, and the
pre-registered bands from September are all still valid and unused — this
needs a fresh, focused attempt (ideally AFTER the hang-watchdog's actual
per-request stall mechanism for cache-busting prefills is understood, since
otherwise any tools-bearing or cache-miss request in the Gate A workload
risks repeating this exact block).

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

### Cluster state at end of session

Both nodes: commit `98a432c520e64cdd48e9515eacee8eadff287bd6`, tree clean vs
origin (pre-existing local-only `dashboard/package-lock.json` diff,
unrelated, present since before this session). 2/2 RunnerReady for
DeepSeek-V4-Flash-Vision-Exp, real completion verified. **Running with
`EXO_RUNNER_HANG_TIMEOUT_SECONDS=300` as a live, non-persisted env override**
— the next relaunch (by anyone, including an automated PM) that does NOT set
this will very likely re-trigger the model-loading hang-kill loop documented
above. Qwen3.6 co-hosting was NOT re-verified as placed this session (the
launcher's own script that handles that step was killed mid-run during
recovery); worth a `/state` check before assuming it's up.

### Next steps, priority order

1. **Fix the hang-watchdog properly** (not the timeout bump): make `_check_hang`
   consult its own `sample` diagnostic (stack-identical + non-growing
   footprint across 2+ samples = genuine hang; anything else = extend) rather
   than a fixed wall-clock, OR find and fix the specific missing/delayed
   progress-event emission in (a) cache-busting prefill's first chunk and
   (b) reload-after-kill model loading. This blocks reliable Gate A
   measurement AND is a real production reliability bug independent of the
   campaign (any real user request with tools + a busted cache would hit
   this today, live, with the shipped 45s default).
2. **Re-attempt I16 Gate A** once (1) is understood well enough to be
   confident the workload driver won't get killed mid-run. Instrumentation
   and workload driver both still valid.
3. **Thread 3 (vision regression) source read**: `deepseek_v4_vision.py`'s
   `merge_image_embeddings`/`build_embeddings` and the SDPA call sites,
   looking for anything keyed on media-region/image-span presence during
   DECODE (not prefill). A real Metal GPU trace comparing matched-length
   text-only vs. image-bearing decode would be decisive and is probably the
   single highest-value next measurement.
4. **Thread 2 (c>=2 spec drift)**: still has a named, unexecuted next step
   from Sept 16 (logprobs repro, then gated fix attempt) — lowest priority,
   not the user's workload.
