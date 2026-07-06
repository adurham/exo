# DeepSeek-V4-Flash c=2 serving hardening — handoff (2026-07-06)

> **RESOLVED (2026-07-06, later session) — task #25 root-caused and fixed; Part B below is
> OBSOLETE.** The "admission wedge" was never a generator stall or rank divergence: it was
> **rank-1 event starvation**. `send_chunk` drops ChunkGenerated on rank != 0 (the c=2 dedup
> guard) and the liveness heartbeat suppressed itself whenever step() returned results, so a
> *healthily streaming* rank-1 runner emitted ZERO supervisor events for the whole stream;
> after `EXO_RUNNER_HANG_TIMEOUT_SECONDS` its own supervisor SIGKILLed it, the peer died on
> ECONNRESET mid-collective, and the SIGKILL leaked QPs. Evidence: two kills on 2026-07-06
> ("2 task(s) in progress, no event for 46s/121s"), PROF lines showing steps returning at
> B=2 decode rate (~96 ms) through the whole "silent" window, stall sampler confirming NO
> in-step stall, zero uid-drop warnings, and both kills landing on whichever node held
> rank 1 that placement. Verified fixed: 3/3 battery pairs (mid-decode admissions,
> asymmetric stop-finish, drain) fully clean — all streams complete, zero kills/deadlines.
> See "Session 2 addendum" at the bottom for the fix stack, the warmup QP-flake findings,
> and current open items.

Branch: `fix/c2-serving-hardening` · exo `a8884075` · mlx `26ac74b7` (adurham/main) · mlx-lm `d46ac14`

## TL;DR

Goal was to make **c≥2 (concurrent) tensor-parallel serving** of DSv4-Flash on the
2-node Mac Studio cluster (MLX / jaccl over Thunderbolt RDMA) fully stable and correct.

- **SOLVED — the jaccl transport-level c=2 wedges.** A reliable ARQ `all_reduce`
  over UC + a framed coordinator barrier eliminate the `all_reduce STALLED` data
  wedge and the coordinator-barrier desync. Output is correct; ~20 tok/s decode,
  prefill 609 tok in ~7 s. (tasks #23, #24)
- **OPEN — a residual c=2 instability that is NOT jaccl.** It is in the
  **exo/mlx-lm batched-generation *admission* path**: admitting a request into an
  already-running batch diverges the two TP ranks → a false-positive hang +
  reliable-deadline re-place. **c=2-from-start (streams that start together) is
  stable.** Root cause and the remaining fix are in Part B / task #25.

## How to run c=2 today (recommended config)

```
MLX_JACCL_RELIABLE_DATA=1 \
MLX_JACCL_RELIABLE_MAX_SZ=2 \
MLX_JACCL_RELIABLE_INFLIGHT=2 \
EXO_TARGET_BRANCH=fix/c2-serving-hardening ./start_cluster.sh
```

This is what the cluster is currently running (serving, verified: "3 plus 4" → "7").
Interim stability rule: **avoid admitting a new request into a running batch
mid-decode** (batch requests that arrive together, or let a batch drain first).
c=1 and c=2-from-start are stable.

---

## Part A — jaccl transport (SOLVED, correct)

### The hardware truth (measured, load-bearing)

Apple's Thunderbolt RDMA is **UC-only** (no reliable connection; confirmed, task #20).
UC send size limits, measured by bisection:

| payload            | behavior                                                            |
|--------------------|--------------------------------------------------------------------|
| ≤16 KB (sz≤2)      | **clean** — completes, correct data                                |
| ≥64 KB (sz≥4)      | **sticks** — `post_send` returns OK but never completes; peer gets nothing; a 2nd *concurrent* large send returns ENOMEM(-12) |

The ≥64 KB stick is exactly what the original UC `all_reduce` wedged on
(`all_reduce STALLED`). Concurrent *small* (≤16 KB) sends are fine.

### The reliable ARQ all_reduce

`mlx/distributed/jaccl/lib/jaccl/mesh_impl.h :: reliable_all_reduce<T>()`
(gate `MLX_JACCL_RELIABLE_DATA=1`, 2-rank only).

- Chunk = `[4-byte seq header][data]`; receiver assembles the peer's full message
  by seq, dedups (`got[]`), **defers the reduce so it is idempotent**, retransmits
  missing chunks until both-have-all, then reduces **once**.
- Knobs: `MLX_JACCL_RELIABLE_MAX_SZ` (chunk size class; **use 2 = 16 KB** — 4× fewer
  chunks than sz=0, still completes reliably), `MLX_JACCL_RELIABLE_INFLIGHT`
  (pipeline depth; use 2 — a sliding-window recv keeps `posted_recvs ≤
  num_chunks - all_recv`, so **zero leftover recv WRs**), `MLX_JACCL_RELIABLE_IDLE_US`
  (idle-poll sleep, default 15 µs, anti CPU-spin).
- A **15 s drain deadline** converts any stuck reliable collective into a clean
  logged throw → clean re-place (never a silent hang).

Bugs fixed along the way (mlx commits, newest first `26ac74b7 → f1850b4e`):
pipelining + sliding-window recv, idle-sleep, **framed barrier**, chunk-size cap,
drain deadline, progress-based drain exit, zero-leftover recvs, 1-outstanding-send
(the -12).

### The framed coordinator barrier (task #23)

`mlx/distributed/jaccl/lib/jaccl/rdma.h :: SideChannel::reliable_barrier()`.

The generic `SideChannel::all_gather` is **raw TCP with no framing** — any byte-count
mismatch cascades into a permanent silent hang (partial reads keep resetting
`SO_RCVTIMEO`). `reliable_barrier` prefixes a fixed 16-byte header
`{MAGIC, call_id, round, count}`; a desync now **throws** (clean re-place) and
correct framing prevents the misalignment. `tcp.cpp` sets both `SO_RCVTIMEO` and
`SO_SNDTIMEO` on coordinator sockets.

### Correctness note (important)

Validate coherence with **`max_tokens ≥ 200`** and read `content`. DSv4-Flash is a
**reasoning model** — it emits `reasoning_content` first, so short-`max_tokens`
checks show an empty/terse `content` during the reasoning phase. Several iterations
were lost to false "corruption" conclusions (16 KB chunks, pipelining) that were
actually just this testing artifact — both are correct.

---

## Part B — residual c=2 instability (OPEN, task #25) — batched-generation admission

**Not jaccl.** Reproduces only when a request is **admitted into an already-running
batch** (batch-size / uneven-context transition). c=2-from-start is stable (control:
two simultaneous requests both complete cleanly, 1500 tok each, ~7.8 tok/s/stream).

### What happens at admission

1. `self._mlx_gen.next_generated()` (the DSv4MTP batch generator, in
   `batch_generate.py:2026`) enters a long **internal forward loop / catch-up** —
   the model forward counter climbs ~13 forwards/s (B=2, L=1) then **freezes** —
   **without yielding responses**. `step()` doesn't return → no `ChunkGenerated` →
   the supervisor hang watchdog SIGKILLs a *healthy, decoding* runner (false positive).
2. The two TP ranks **diverge** (one catches up, the other reaches the next
   collective) → the waiting rank's `reliable_all_reduce` hits its 15 s deadline →
   **both re-place**.

### Evidence chain (all ruled out with hard data)

- Not jaccl (reliable transport correct, 0 retransmit rounds).
- Not memory (free RAM steady ~40 GB / 128 GB through the stall).
- Not MTP/spec (persists with `EXO_DSV4_MTP=0 EXO_SPECULATIVE=0`).
- Not corruption (outputs correct with proper `max_tokens`).
- Not a re-prefill (the stuck step is L=1 decode; the only L>1 forwards at admission
  are the joiner's small prefills, L=46/49).
- GPU util holds **66–67 %** the whole stall (computing, not idle) — a `DSV4_SHAPE`
  forward counter proved the runner keeps decoding while emitting nothing.

### Why the landed heartbeat is insufficient

A liveness heartbeat was added — `src/exo/worker/runner/runner.py` (in the
`step()` loop, ~line 508): when `step()` returns empty but tasks are active, re-emit
the current status (throttled 15 s) to reset the supervisor's hang clock
(`_last_event_monotonic`, `supervisor.py:362`). Rationale: a returned-empty `step()`
proves liveness; a genuine hang stuck *inside* `step()` never returns, so it's still
caught. **It helps some pairs but is insufficient**: during the admission transition
the loop is *inside* `next_generated()` (mlx-lm), so `step()` never returns and the
post-step heartbeat can't fire.

### The proper fix (well-scoped next effort)

1. Keep the two TP ranks in **lockstep across the admission transition** — no
   divergent per-rank catch-up (this is what desyncs the collective → reliable
   deadline). See the fence/lockstep machinery in
   `mlx-lm/mlx_lm/models/deepseek_v4.py` (`_FENCE_*`, `FENCE_EVERY_N_LAYERS`) and the
   per-stream ring bootstrap `mlx-lm/mlx_lm/models/cache.py:2640
   _bootstrap_per_stream_ring` (BS-transition path; cheap tensor op, not the loop —
   but it's where the uneven-context reconciliation lives).
2. Make `next_generated()` **yield / bound its work** at the transition so `step()`
   returns and streams progress incrementally, instead of decoding many tokens
   silently. Entry point: `batch_generate.py step()` (1955) →
   `next_generated()` in `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py`.

**Mitigation until then:** admission control — don't admit into a running batch
mid-decode.

---

## Env var reference

| var | meaning | recommended |
|-----|---------|-------------|
| `MLX_JACCL_RELIABLE_DATA` | enable reliable ARQ all_reduce | `1` for c≥2 |
| `MLX_JACCL_RELIABLE_MAX_SZ` | chunk size class (0=4 KB … 7=512 KB) | `2` (16 KB) — MUST stay ≤2 (≥sz4 sticks) |
| `MLX_JACCL_RELIABLE_INFLIGHT` | pipeline depth (send+recv) | `2` |
| `MLX_JACCL_RELIABLE_IDLE_US` | idle drain-poll sleep | default `15` |
| `EXO_RUNNER_HANG_TIMEOUT_SECONDS` | supervisor hang watchdog | default 45; raising it does NOT fix task-#25 (generator never yields during the stall) |
| `EXO_DSV4_MTP` / `EXO_SPECULATIVE` | speculative decode | default on; independent of the task-#25 wedge |
| `EXO_DECODE_PROBE` / `_EVERY` | per-step wall/gpu timing to stderr | diag only |

## Diagnostics & repro

- **Repro:** `~/scratch/specoff_battery.py N` on m4-1 — submits overlapping pairs
  (TOKENS=4000, stream), which produces mid-decode admissions → the task-#25 wedge
  within ~2–3 pairs. c=2-from-start (2 simultaneous single requests) does NOT wedge.
- **Forward-progress probe** (loop detector): a throttled `[DSV4_SHAPE] fwd# B L`
  print in `DeepseekV4Model.__call__` — parked on `adurham/mlx-lm` branch
  **`diag-c2-admission`** (`5848648`). `main` is clean at `d46ac14`. **Do NOT deploy
  the diag branch**: `flush=True` on every prefill forward stalls the forward → peer
  `reliable_all_reduce` deadline → re-place loop. This exact trap cost a debug cycle.
- GPU util: `ioreg -r -d 1 -w 0 -c IOAccelerator | grep "Device Utilization %"`.
- Live stack of a stuck runner: `sample <pid> 4 -file /tmp/x.txt` (find pid via
  `pgrep -f multiprocessing.spawn`).

## Deploy mechanics (gotchas)

- WAN HTTPS to GitHub is broken from these hosts; push over **SSH**
  (`git push git@github.com:adurham/mlx-lm.git …`). mlx uses the `adurham` SSH remote.
- **mlx-lm is a git submodule of exo.** To ship an mlx-lm change: commit+push mlx-lm
  (SSH), then `git add mlx-lm && commit` in exo to bump the submodule pointer, then
  redeploy (`start_cluster.sh` does `git submodule update` + `uv pip install
  --force-reinstall ./mlx-lm`).
- mlx is pinned in `uv.lock`; bump with `uv lock --upgrade-package mlx` after pushing
  to `adurham/main`.

## Known leftovers / cleanup TODO

- **mlx reliable path retains 5 gated diagnostic `fprintf`s** in
  `mlx/distributed/jaccl/lib/jaccl/mesh_impl.h` (compiled into the deployed `26ac74b7`):
  `ENTER` (L351), `post_send FAILED` (L379), `RECV` (L403), `DEADLINE` (L467),
  `BARRIER` (L536). All gated (first-N calls or `jaccl_progress_enabled()`), so
  harmless, but trim for a clean prod mlx. **Keep `DEADLINE`** — it's the clean-fault
  log when a reliable collective is abandoned.
- **exo runner heartbeat** (`runner.py` ~L508) is committed and kept (correct
  direction, harmless) but only **partially** fixes task #25 — see Part B.
- **`adurham/mlx-lm main` is clean (`d46ac14`)**; diag prints live only on
  `diag-c2-admission` — never deploy that branch.
- **Task #21 (in-place collective recovery) — parked, not viable on this hardware.**
  The peer parks in an *uninterruptible* Metal/GPU wait, so an in-place jaccl
  reconnect cannot reliably resume both ranks. The accepted recovery model is: the
  reliable path's 15 s drain deadline (or the coordinator/`Event::wait` timeouts)
  surfaces a clean fault → runner catches jaccl-transport faults and attempts
  `group.reconnect()` (model stays resident), else propagates → full re-place (~90 s).

## File & commit index

**mlx** (`26ac74b7`, adurham/main):
- `mlx/distributed/jaccl/lib/jaccl/mesh_impl.h` — `reliable_all_reduce`, config gates
- `mlx/distributed/jaccl/lib/jaccl/rdma.h` — `SideChannel::reliable_barrier`
- `mlx/distributed/jaccl/lib/jaccl/tcp.cpp` — `set_recv_timeout_secs` (RCV+SND)

**exo** (`e3a8250f`, `fix/c2-serving-hardening`):
- `src/exo/worker/runner/runner.py` — step→emit loop + **liveness heartbeat** (~508)
- `src/exo/worker/runner/supervisor.py` — `_check_hang` (407), hang clock (362)
- `src/exo/worker/engines/mlx/generator/batch_generate.py` — `step()` (1955),
  `next_generated()` call (2026), `_submit_batched_eligible` (1529)
- `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py` — DSv4MTP batch generator (task-#25 fix site)
- `start_cluster.sh` — env forwarding for the `MLX_JACCL_RELIABLE_*` + hang-timeout knobs

**mlx-lm** (`d46ac14`, clean):
- `mlx_lm/models/deepseek_v4.py` — `DeepseekV4Model.__call__` (3152), fence machinery (`_FENCE_*`)
- `mlx_lm/models/cache.py` — `_bootstrap_per_stream_ring` (2640)

Related memory / prior handoffs: `exo_jaccl_uc_send_limits`,
`exo_dsv4_c2_mtp_corruption`, `exo_dsv4_gemv_gemm_batchdrift`,
`docs/deepseek-v4-c2-mtp-verify-fixes.md`.

---

# Session 2 addendum (2026-07-06) — task #25 RESOLVED + serving self-heal hardening

## Root cause of task #25 (the real one)

**Rank-1 event starvation → false-positive watchdog SIGKILL → cascade.**

- `runner.py send_chunk` drops `ChunkGenerated` on `device_rank != 0` (correct c=2 dedup).
- The liveness heartbeat (`handle_generation_tasks`) skipped emitting whenever
  `step()` returned results — valid only on rank 0 where chunks double as events.
- `supervisor._check_hang` is rank-blind.
- Net effect: any stream longer than the hang timeout got its **rank-1 runner
  SIGKILLed while perfectly healthy** (GPU decoding at ~96 ms/step B=2 the whole
  time). The peer then hit ECONNRESET / reliable-deadline mid-collective → re-place;
  the SIGKILL leaked QPs (no destructors). The handoff's Part B "internal forward
  loop / catch-up + rank divergence" was a misreading of this cascade's symptoms.

**Fix:** non-zero ranks re-emit their runner status every 15 s while tasks are
active, unconditionally (`runner.py` heartbeat; commit message has full detail).

**Verified:** `specoff_battery.py 3` — 3/3 pairs clean with full per-stream stats
(4000/4000 length, 2827-stop/4000 asymmetric finish, 4000/4000), zero SIGKILLs,
zero deadlines, zero re-places across the run; c=1 smoke clean before and after.

## Second finding: warmup/connect QP flake (NEW, partially mitigated)

Fresh jaccl QP pairs intermittently come up with the **UC data path silently dead
in BOTH directions** (`all_recv=0` for the full 15 s reliable window on both ranks,
`outstanding_sends` stuck; TCP side-channel fine). Reproduced on a **freshly
rebooted OS** → NOT accumulated driver state, NOT settle-time, NOT the `!` route
flag (present on healthy links too). In-process `group.reconnect()` retries did
NOT heal it (3/3 attempts failed within a process); a fresh runner process
sometimes does (~1 in 2-4). Previously each roll cost a full model load because
the failure only surfaced in warmup.

Mitigations landed:
- `_warmup_with_reconnect` (runner.py): warmup retries through jaccl faults.
- **Connect-time data-path probe** (`mlx/builder.py _probe_data_path`): two
  all_sums (1-chunk + ~128 KB sz=2) right after distributed init, reconnect-and-
  retry up to `EXO_JACCL_CONNECT_PROBE_ATTEMPTS` (default 5) — moves the dice
  roll BEFORE the load. Deployed but not yet observed in action (next placement).
- Root cause in jaccl/librdma QP establishment still OPEN (likely recv-not-ready /
  activation race; needs mesh-level investigation).

## Serving self-heal fixes (all deployed)

- `supervisor._check_runner`: a runner that reports a critical exception and exits
  rc=0 during NON-generation work (warmup/load) now emits `RunnerFailed` — kills
  the "WARMING UP zombie with zero runner processes" mode (hit twice today).
- Bounded `step()` (one `_next()` pass per call) — next_generated()'s internal
  loop can no longer hold step() open.
- Stall sampler: `EXO_STALL_SAMPLER_SECONDS=N` dumps all-thread stacks to
  `~/exo_stall_dumps/` (reboot-durable) when step() stops returning. Proved the
  "stall" wasn't one.

## Commit index (session 2, exo, newest first)

- `a8884075` feat(builder): jaccl data-path probe at connect
- `6a4…/…`   fix(runner): non-zero ranks heartbeat unconditionally  ← the task #25 fix
- `65dd052d` diag(runner): stall dumps to ~/exo_stall_dumps
- `10e9d471` fix(supervisor): RunnerFailed for clean-exit crashes in non-generation work
- `a6e1c589` fix(runner): warmup retries through jaccl faults via reconnect
- `c44a76c0` fix(engine): bounded step() + stall stack sampler

## Current cluster state / gotchas

- Both Studios were OS-rebooted today (ruled out driver-state theories).
  `iogpu.wired_limit_mb` reverted to default (0) — exo sets its own per-process
  wired limit (107.5 GiB) so serving is fine, but re-apply 115000 via sudo when
  Qwen co-hosting returns (see start_cluster.sh rationale).
- Launch scripts on the nodes (`~/relaunch_exo.sh`) carry two diagnostic envs:
  `EXO_STALL_SAMPLER_SECONDS=10` (keep, cheap) and
  `EXO_RUNNER_HANG_TIMEOUT_SECONDS=120` (was for diagnosis; with the heartbeat
  fix the default 45 is fine again — drop on next restart).
- Known minor open item: rank-0 segfaulted once inside `group.reconnect()` when
  its peer had just been SIGKILLed mid-collective (09:51). Rare now that the
  false-positive kills are gone; jaccl reconnect-vs-dead-peer robustness is a
  future item.
- The mlx gated-diag-fprintf cleanup and task #21 notes from Part A/leftovers
  still stand.

## Session 2 addendum, part 2 — warmup QP-flake investigation (deep dive)

**Where the wedge lives (narrowed, not fully root-caused):** the dead UC data
path is a *device-context-level* state in Apple librdma. Evidence: a stuck
send (post_send OK, CQE never arrives, nothing on wire) survives
`MeshGroup::reconnect()` — which by design resets ONLY QPs and preserves
PD/CQ/MRs/ibv-context (3/3 in-process reconnects failed at 09:27) — but a
fresh process (fresh `ibv_open_device`) rolls new dice. The prod failing
fingerprint: reliable call_ids 1-3 (tiny) pass, the first sz=2 multi-chunk
collective (call_id 4) deadlines, sometimes asymmetrically (one direction
clean, the other's first send stuck).

**Repro attempts (all NEGATIVE — 56 standalone harness rolls + 3 fresh real
placement cycles, all clean):** minimal init+probe; 45 s idle QPs; SIGKILL-
leaked QPs from a prior pair; 60 GB wired-memory pressure; init during a
60 GB wired teardown; two-Mesh coord-subgroup interleave replicating
warmup's exact op sequence + full prod env. The morning's 5-of-8 failure
rate could not be reproduced in the afternoon — the dice are not
stationary and the elevated-failure trigger remains unidentified.
Harness: `~/scratch/jaccl_probe_harness.py` on both Studios (rank,
coordinator, [delay_s], [wire_gb]; `PROBE_TWO_MESH=1` for the warmup-like
sequence) — 2-second rolls, no model load; keep for future hunts.

**Recovery posture (what makes the flake operationally moot):** connect-time
data-path probe (pre-load, ≤30 s, default ONE reconnect then fail-fast) →
RunnerFailed → supervisor re-place spawns a fresh process = fresh device
context. Warmup retry + zombie fix backstop the late cases. Every stage
self-heals; worst case is added startup latency, never a stuck cluster.

**True fix (future):** jaccl "hard reconnect" — rebuild device context, PD,
CQ, and re-register all buffers in place. Significant C++ surgery in
`mesh.cpp`/`rdma.cpp`; do not attempt without a repro to validate against.

## Session 2 addendum, part 3 — jaccl reconnect_fresh (the QP-flake fix, LANDED)

mlx `e399ecfb` (adurham/main; exo uv.lock bumped): `MeshGroup::reconnect_fresh()`
closes and reopens the ibv device contexts and rebuilds everything on top
(PD/CQ/QP, buffer MRs — registered BEFORE the INIT transition, macOS librdma
locks the QP MR table there — exchange/RTR/RTS over the surviving TCP side
channel, MeshImpl/RingImpl span views, ACK recv pool, bootstrap barrier). This
is the in-process equivalent of the runner respawn — the only recovery that
ever cleared the dead-UC-path wedge — at ~0.15 s instead of a re-place.

- Gate: `MLX_JACCL_RECONNECT_FRESH=1` (now in both nodes' `~/relaunch_exo.sh`);
  `reconnect()` takes the fresh path pre-split and falls back to QP-only reset
  once subgroups exist (they borrow the parent's contexts; `has_split_`).
- exo connect probe defaults to 3 attempts under fresh mode (each retry is a
  real fresh-context roll) vs 1 (fail fast to respawn) without.
- Validated: 10/10 fresh rebuild loops on both ranks, ~0.15 s each, collectives
  bit-correct after every rebuild (exact small + 106 KB bf16 all-elements);
  clean serving placement + smoke on the new build.
- Deploy note: mlx now installed on the nodes from `~/repos/mlx` (branch merged
  to adurham/main); exo `uv.lock` pins `e399ecfb`, so `start_cluster.sh`'s
  `uv sync` keeps it.
- Future: cascade fresh reconnect to subgroups (child registry + rebuild) so
  SERVING-time faults can also recover in-process with the model resident.
