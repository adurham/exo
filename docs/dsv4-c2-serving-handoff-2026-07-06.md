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

## Session 2 addendum, part 4 — MTP RE-ENABLED AT c>=2 (pooling B-invariance fix)

**The "c=2 verify corruption" root cause is REWRITTEN — structural, not kernel
numerics.** `PoolingCache` (c=1) runs the W4 path-1 DEFERRED pool update by
default (just-pooled entry attendable NEXT step); `BatchPoolingCache.
update_and_fetch_deferred` was a documented synchronous fallback ("c=2
frozen"), so at B>=2 the entry was attendable IMMEDIATELY — different pool
visibility for identical content in every pooled attention layer, worst in
L>1 MTP-verify chunks (measured P=13 vs P=14 at the r4 indexer). The
gemv/gemm kernel-drift theory is demoted: an M-sweep probe showed ALL
quantized matmuls (affine8+mxfp4, every DSv4 shape incl. lm_head) already
batch-invariant to M=8; only plain bf16 mm K>=1024 flips at M=2, a ~0.04
residual, sub-threshold.

Fix (mlx-lm `a7db9e2` + sizing fix, adurham/mlx-lm main; submodule bumped):
true per-stream deferral in BatchPoolingCache — `_pending_bumps` staged in
update_and_fetch_deferred (pre-write tensor returned), applied in
commit_pending; `_visible_width` keeps make_mask sized to the consumed
tensor; save/restore_meta carry bumps through spec rollback; structural ops
(filter/extend/extract/merge/meta_state) commit eagerly AND resize the
bumps list (missing resize in extend() was an IndexError at the first
post-admission deferred update — fixed).

Validated:
- dsv4_bdiff_model.py: B=2-vs-B=1 max drift 0.318 -> 0.039, argmax 1.000,
  [OK] plainBatch + PerStream; pool lengths/SDPA shapes now identical.
- mlx-lm test_prompt_cache: 21 passed, 2 pre-existing failures (identical
  on unpatched venv).
- **specoff_battery 3 pairs with `EXO_DSV4_MTP_C2_MAX_CTX=0` (MTP ON at
  c=2): 3/3 clean** — mid-decode admissions, temp=1.0, mixed stop/length
  finishes, zero degeneration/kills/deadlines.
- Throughput: c=2 MTP-on 14.3 tok/s/stream (28.6 agg, incl. prefill) vs
  ~8-10 spec-off.

Config: `EXO_DSV4_MTP_C2_MAX_CTX=0` now in both relaunch scripts (spec ON
at all batch sizes). Follow-ups: retest `EXO_DSV4_FENCE_ASYNC_C2=2` on top
(gave 20.4 t/s/stream pre-MTP), and a high-context needle pass (200K+)
before calling long-context quality fully validated. Diagnostic tools:
`~/scratch/bdiff_r128_bisect.py`, `qmm_invariance_sweep.py`,
`dsv4_bdiff_model.py` (fixed-seed 4-layer repro, ~60 s).

---

# Session 3 addendum (2026-07-06) — long-context speed: 500K ladder baseline + prefill bottleneck

Goal: 300 tok/s prefill and 30 tok/s decode at 500K context (c=1).

## Bug fixed first: rank-1 event starvation during long prefill (exo `0cfb16b7`)

Any prefill longer than EXO_RUNNER_HANG_TIMEOUT_SECONDS SIGKILLed the rank-1
runner while healthy: `on_prefill_progress` emits PrefillProgressChunk only on
rank 0, and the task-#25 decode heartbeat lives in the step() loop — never
reached while submit() is inside a long synchronous prefill. Same starvation
class as task #25, prefill edition. Fix: `Engine.heartbeat` hook (installed by
Runner after build()); non-zero ranks fire it throttled (15 s) from all three
prefill-progress callback sites. Verified: 59-minute 500K prefill, zero kills.

## 500K ladder baseline (c=1, MTP on, prod config, fresh salted prompts)

| context | prefill tok/s | decode tok/s |
|---------|---------------|--------------|
| 4.7K    | 152           | 27.6         |
| 32K     | 151           | 28.0         |
| 64K     | 152           | 27.4         |
| 128K    | 152           | 24.2         |
| 256K    | 150           | 21.3         |
| 500K    | 141.6         | 17.2         |

- **Prefill is FLAT in context** — purely per-token constant cost, not a
  long-context cliff. Gap to target: 2.1x.
- **Decode droops linearly** (~44 µs/token added per 100K ctx; 500K measured
  17.2 exactly on the extrapolation). Gap at 500K: 1.74x.
- Peak wired 83.7 GB / 128 GB at 500K (KV headroom fine; 1M would squeeze).
- GPU util during 256K prefill: **75% on BOTH nodes** → NOT compute-saturated.

## Prefill bottleneck: reliable ARQ is stop-and-wait at sz=2

`mesh_impl.h reliable_all_reduce`: `SEND_INFLIGHT = (sz==0) ? inflight : 1` —
at the production chunk class (sz=2, 16KB) exactly ONE send is in flight;
MLX_JACCL_RELIABLE_INFLIGHT does nothing. Measured with a standalone bench
(`~/scratch/jaccl_bw_bench.py`, both nodes, cluster idle):

| payload | per-op | effective | note |
|---------|--------|-----------|------|
| 8 KB    | 0.19 ms | 44 MB/s  | decode-shaped (1 chunk + TCP barrier) |
| 2 MB    | 4.4 ms  | 475 MB/s | prefill-shaped; **identical at inflight=1 and =2** |
| 8 MB    | 16.9 ms | 497 MB/s | |

Consistency check: 688 KB collective traffic per prefill token ÷ 475 MB/s ≈
1.45 ms of the 6.6 ms/token budget ≈ 22% — matches the 25% GPU idle.

Patch (mlx `452fbebf`, branch `perf/reliable-send-pipelining`, pushed to
adurham + both nodes): honor RELIABLE_INFLIGHT for sz<=2 (concurrent <=16KB
UC sends measured clean 2026-07-05; ENOMEM only >=64KB) + NUM_BUFFERS 2->8.

## Decode target notes (next after prefill)

- EVERY reliable collective pays a TCP coordinator barrier round-trip (bitmask
  exchange) even for 1-chunk decode payloads — the 0.19 ms 8KB floor.
- Decode context-slope (~44 µs/100K) is separate (indexer scan?); profile with
  EXO_PROFILER=spans + EXO_DECODE_PROBE at 256K+.
- Follow-up from session 2 still open: retest EXO_DSV4_FENCE_ASYNC_C2=2.

## Session 3, part 2 — ARQ send pipelining LANDED (mlx `452fbebf`)

Patch validated and deployed (adurham/main; exo uv.lock bumped `e463883e`;
MLX_JACCL_RELIABLE_INFLIGHT=8 in both relaunch scripts):

- Standalone bench: 2MB collectives 475 -> 3711 MB/s (if=8), 8MB -> 4287 MB/s.
  8KB decode-shaped unchanged ~0.17ms (TCP-barrier floor, expected).
- Bitexact: 80/80 pipelined collectives exact. specoff_battery 3/3 clean
  (c=2 mid-decode admissions, MTP on).
- One warmup QP-flake DEADLINE on first placement post-restart —
  reconnect_fresh healed in-process (~0.15s), warmup completed. Watch whether
  depth-8 raises flake frequency (faster sends after QP establishment).

Ladder, baseline -> pipelined:

| context | prefill tok/s      | decode tok/s     |
|---------|--------------------|------------------|
| 64K     | 152 -> **316**     | 27.4 -> **32.1** |
| ~300K   | 150 -> **302**     | 21.3 -> 21.5     |
| 500K    | 141.6 -> **249**   | 17.2 -> 18.6     |

Targets (300 prefill / 30 decode @500K): prefill 83% there (droop 316->249
past 300K = a context-dependent prefill cost emerging — indexer/attention
share?); decode 62% there (context-linear ~44µs/100K term dominates; comms
pipelining barely moves it, as predicted).

Next: profiled 500K run (EXO_PROFILER=spans + EXO_DECODE_PROBE=1, in both
relaunch scripts as of 17:05) to attribute (a) the 500K prefill droop and
(b) the decode linear term (suspect: indexer top-k scan over full ctx).
Decode-phase spans flush at the NEXT request's prefill start — the profiled
run submits 500K then a 4K chaser for exactly this.

## Session 3, part 3 — profiling attribution + prefill A/Bs

**Span attribution** (EXO_PROFILER=spans, ratios only — spans under lazy eval
are unreliable for absolutes; ~2-2.6x overhead while armed):
- Prefill 500K: moe.switch_mlp ~45% (flat vs 4K), attn grows 33.8→37.9%,
  driven by attn.sdpa 3.8→7.0% and attn.indexer 1.4→4.2% — that pair is the
  >300K prefill droop (316@64K → 249@500K). Comms ~19% post-pipelining.
- Decode 500K vs 4K: context-linear cost is ALL attention-path —
  attn.sdpa 9.5→18.8% (avg 369→917µs/layer-step), indexer 7.0%,
  compressor 6.5%. MoE + comms shares SHRINK. Decode@500K = indexer/sparse
  kernel problem, not comms. (Task: optimize the O(ctx) indexer scan.)

**A/B results (64K probe, unprofiled):**
| config change | 64K prefill | verdict |
|---|---|---|
| tracing off (EXO_TRACING_ENABLED=false, LOG_LEVEL=INFO) | 317 (=) | keep off anyway; decode deltas within MTP-acceptance noise (±20% across prompts — don't A/B decode via single 256-tok gens) |
| EXO_DSV4_FUSED_MOE=1 | 317 (=) | DEAD for prefill; reverted to 0 |
| EXO_PREFILL_STEP_SIZE 256→1024 | **393 (+24%)** | ADOPTED (both relaunch scripts) |
| EXO_PREFILL_STEP_SIZE 2048 | 440-467 mid-run, then **hard GPU wedge** | REJECTED |

**2048 wedge detail:** froze at 12K/63K tokens; stall sampler caught main
thread in `mx.eval(y)` at the per-layer MoE all_sum fence (deepseek_v4.py
~1647) with NO jaccl DEADLINE in 112s → the pre-existing uninterruptible
Metal/GPU wedge class (tasks #15-19), likelier with 16MB collectives /
L=2048 command buffers. Hang watchdog killed + re-placed cleanly.

specoff_battery 3/3 clean at step 1024. Final validation ladder running.

## Session 3, part 4 — seq-split gather fix + FINAL 500K numbers

**Step-1024 wedge root-caused and fixed.** The 256K failure at step 1024 was
the seq-split band all_gather: it runs on a jaccl SUBGROUP (no TCP
coordinator → reliable ARQ can't arm → raw UC), and ~4MB bands
intermittently hit the UC stuck-send wedge; subgroup reconnect can't heal
(reconnect_fresh is top-level only) → full re-place. Fix (mlx-lm `7de756d`,
submodule bump exo `0c2bee10`): reconstruct bands via ZERO-PADDED all_sum on
the top-level group — bit-exact, rides the pipelined reliable path, kills
the wedge class. Gate: EXO_DSV4_SEQSPLIT_GATHER_VIA_ALLSUM (default 1).

**FINAL ladder (pipelined ARQ if=8 + step 1024 + reliable gather + tracing off):**

| context | prefill tok/s (was, 09:00 today) | decode tok/s |
|---------|----------------------------------|--------------|
| 4.7K    | 363.5 (152)                      | 29.5 (27.6)  |
| 256K    | 367.4 (150)                      | 21.1 (21.3)  |
| 500K    | **342.4 (141.6) — TARGET 300 MET** | 17.5 (17.2)  |

GPU util during 500K prefill: ~94-95% (was 75%) — comms overhead gone.
Zero kills/wedges across the final ladder.

**Decode @500K remains the open target (17.5 vs 30).** Attribution says the
gap is context-linear attention-path cost (indexer O(ctx) scan + sparse
gather/sdpa + compressor — sdpa avg 369→917µs/layer-step from 4K→500K
profiled) plus the per-collective TCP-barrier floor (~0.17ms × ~dozens of
collectives/token). Next levers, in order:
1. Indexer scan kernel at long ctx (the ~44µs/100K/token slope) — kernel
   work; see gather_qmv experience in memory `exo_gather_qmv_rhs_kernel`.
2. TCP-barrier elimination for 1-chunk reliable collectives (piggybacked
   UC ack or barrier-every-N) — mesh_impl.h.
3. Retest EXO_DSV4_FENCE_ASYNC_C2=2 on the new transport (old +36% c=2).

**Config deltas today (both relaunch scripts):** MLX_JACCL_RELIABLE_INFLIGHT
2→8, EXO_PREFILL_STEP_SIZE 256→1024, EXO_TRACING_ENABLED true→false,
LOG_LEVEL DEBUG→INFO, EXO_DSV4_FUSED_MOE stays 0 (A/B'd: dead for prefill),
step 2048 REJECTED (GPU wedge at 16MB command buffers).

**Commits:** exo `0cfb16b7` (prefill heartbeat), `e463883e` (mlx lock bump),
`0c2bee10` (mlx-lm bump); mlx `452fbebf` (ARQ pipelining, adurham/main);
mlx-lm `7de756d` (reliable gather). All pushed; node repos + venvs synced.

---

# Session 4 addendum (2026-07-06 evening) — decode @500K: attribution + optimistic ARQ (v2)

Goal: decode 30 tok/s @500K c=1 (from 17.5). Prefill target already MET (342).

## Attribution (NOP-toggle method — the ground truth)

Method: `/tmp/dsv4_nop_targets` live block-disable + fixed unsalted prompt
(prefix cache pays the 500K prefill ONCE) + MTP/spec OFF so median
inter-delta gap == per-forward ms. Driver: laptop `nop_attrib.py` (in
scratchpad; salt-free variant of longctx_bench). Toggles flip BETWEEN
requests on both nodes (never mid-flight — and never NOP `sparse_attn` at
500K: garbage output crashed a runner → ECONNRESET cascade → re-place).

**Decode per-forward composition (p50 ms, spec-off, allsum-probe config):**

| component (NOP delta) | 4K    | 500K  |
|-----------------------|-------|-------|
| baseline forward      | 65.1  | 67.5  |
| collectives (all_sum) | 23.0  | 19.3  |
| MoE compute           | ~16.7 | —     |
| sparse sdpa+gather    | 10.2  | (crashed) |
| compressor            | 6.5   | —     |
| indexer (score+topk)  | 3.9   | 7.1   |
| topk_fused live-on    | ±0    | ±0    |

**Two decisive findings:**
1. **The spec-off forward is nearly FLAT in context** (65→67.5ms; indexer
   +3.2ms is the only slope). The synthetic op microbench agrees (per-op
   O(ctx) terms are ~100-285us/layer at 500K — small).
2. **The 500K decode droop is an MTP-effectiveness collapse**: MTP-on/spec-off
   yield = 1.96x @4K → 1.27x @500K. Fixing the forward path can't close the
   500K gap; the MTP cycle (acceptance and/or per-cycle cost) at long ctx is
   the target. Hypothesis: draft/verify top-k selection agreement degrades as
   k=512 covers 0.4% of the 125K pool (vs 43% @4K) — the code already
   documents acceptance sensitivity to tiny top-k perturbations (bf16 note in
   `_indexer_score`). First lever: EXO_DSV4_INDEX_TOPK at long ctx.
   Measure first: `EXO_DSV4_MTP_LOG_INTERVAL=50` (NOT `EXO_DSV4_MTP_LOG` —
   stale comment) prints mean_accept + histogram per 50 cycles.
3. Comms is ~19-23ms/forward FLAT — the #1 decode cost at every context.

## Optimistic reliable ARQ (v2) — LANDED, +29% decode @4K

mlx branch `perf/reliable-optimistic` (572e29e1a + da03b622, pushed to
adurham): `reliable_all_reduce_v2`, gate `MLX_JACCL_RELIABLE_OPTIMISTIC=1`.
Kills the per-collective TCP coordinator barrier for SMALL collectives
(num_chunks<=3 — all decode all_sums):

- 12-byte in-band header {call_id, seq, len}; **uniform cap size class for
  ALL v2 messages** (Apple librdma silently kills size-class-mismatched
  send/recv pairs — the documented LOC_LEN_ERR FIFO class; first build
  deadlined on this).
- Standing 8-recv pool per peer (cap class, repost-on-consume, re-armed via
  reset_ack_state on reconnect) — peer recv queue can never be "not ready".
- Optimistic exit: leave on (all peer chunks + own send CQEs). Send buffers
  parity-partitioned (call_id&1 → slots 0-3/4-7) and retained one collective;
  a stuck peer's quiet-timeout STATUS (got-bitmask) is answered verbatim from
  retained buffers by the NEXT collective's poll loop. Skew is provably <=1.
- One-call lookahead stash; skew>1 or malformed header throws (clean
  re-place); 15s deadline unchanged. LARGE collectives (prefill) keep the TCP
  rendezvous exit but ride the same pool/header; their send pipeline is now 4
  parity slots (was 8) — measured -5% on 2MB ops (3623→3429MB/s), prefill
  unchanged e2e (345-362 t/s @4K rung).
- exo side (`fix/c2-serving-hardening` f342996e7): warmup + cache-pressure
  all_gathers rerouted to the coord subgroup — the model group's data QP must
  be ALL-all_sum for the pool. **PP placements must keep the gate OFF**
  (PipelineLastLayer send/recv/all_gather ride the model group).

Validation: `~/scratch/jaccl_v2_validate.py` (bitexact soak, mixed
small/large transitions, skew stress, latency ladder) — OK on both ranks;
decode-shaped all_sum 147.5→91.6us idle-link. Serving: smoke correct,
specoff_battery 3/3 clean (c=2 mid-decode admissions, MTP on), **4K decode
29.5 → 38.3/37.4 t/s (+29%)** — matches 86 collectives/forward × ~56us saved.

Deploy state: nodes' venvs run mlx da03b622 from ~/repos/mlx (NOT in
uv.lock — a `uv sync` REVERTS to 452fbebf; bump the lock after merging
perf/reliable-optimistic to adurham/main). Cluster runs
`~/relaunch_exo_v2.sh` (prod + OPTIMISTIC=1 + MTP_LOG_INTERVAL=50).

## Diagnostic gotchas learned this session

- `EXO_DSV4_SECTION_TIME` dump gate was dead (layer_count increment removed
  in a refactor) — fixed in mlx-lm `9a1e21f`.
- `EXO_DSV4_ALLSUM_PROBE=1` forces per-layer blocking eval (replaces the
  async fence) — decode ~1.7x slower, prefill ~40% slower. Attribution only.
- DSv4 streaming: count BOTH `content` and `reasoning_content` deltas.
- m4-2 ~/repos/mlx origin was HTTPS (WAN-broken) — now SSH like m4-1.

## Session 4, part 2 (2026-07-07 early) — MTP verify slope: root-cause hunt + partial fixes

**MTP-PROF phase timing** (`EXO_DSV4_MTP_PROFILE=N`, per-cycle draft/verify/
accept/rollback means every N cycles — the tool that cracked it): draft
4.9ms / accept 1.1 / rollback 0.3 are ALL context-flat; **verify (the L=3
main forward) is the entire slope: 45ms @4K → ~67ms @256K → ~89ms @500K.**
Acceptance is context-flat too (0.86-1.0 at every ctx — MTP_LOG windows), so
the yield collapse is pure cycle-time growth.

**Verify-slope decomposition at 256K** (NOP windows, cached-prefix sessions;
use VERIFY MEAN not decode_tps — garbage-quality toggles inflate acceptance
up to 1.66/2):
- indexer (score+topk over P): ~6.3 ms/token
- pool write+compress: ~3.7 ms/token (compressor_compress window — NB it
  also skips the write via empty px)
- `compressor_pool` NOP (pool frozen at 1 entry → ALL pool-path work gone,
  acceptance-normal window): ~19.7 ms/token → a ~10 ms/token remainder that
  is NOT indexer-score, NOT k-proportional (EXO_DSV4_INDEX_TOPK=192 A/B:
  24.08 vs 24.0 t/s = ZERO effect at 256K — k-gather/sdpa is cheap), NOT
  the write. Mechanism unresolved — next tool: mx.metal.start_capture GPU
  trace of one verify at 4K vs 256K.

**Pool-write donation fixes (landed, measured NEUTRAL so far):** mlx-lm
276869c + f0ll0wup — the W4 deferred pool update's pre-write view blocks MLX
slice-update donation → O(P·D) copy per flush (microbenched: 0.85ms/flush
at 500K shapes vs ~0 donated; `EXO_DSV4_POOL_DEFER_COPY_MAX_BYTES`
threshold, default 32MB, 8MB in prod config; + mx.async_eval enqueue for
ordering). e2e effect at 256K: none measurable — either donation still
loses to in-graph aliasing or the copies were already amortized. The
sync'd section-time compressor-span growth (+88% at 256K) says SOMETHING
in that span scales; capture will tell. Semantics unchanged (post-write
prefix view = same values), so the code is safe to keep.

**Dead/neutral levers this session:** EXO_DSV4_INDEX_TOPK=192 (no speed
effect at 256K, k=512 kept for quality margin), topk_fused live-enable (no
e2e effect), pool donation threshold/async_eval (neutral so far).

**Incidents:** one coord-subgroup `drain_acks STALLED` (lost UC completion,
call_id 27027) mid-256K-prefill → clean re-place, self-healed. Same flaky-UC
class as the warmup QP flake; NOT the v2 data path (zero v2
DEADLINE/PROTOCOL/WC_ERR all session). Also: NOPing `sparse_attn` at long
ctx crashes a runner (garbage → engine cascade) — bench-only toggles.

## FINAL STATE (2026-07-07 ~00:30)

| metric | before session | after |
|--------|----------------|-------|
| 4K decode | 29.5 t/s | **37-38 t/s** (+29%) |
| 256K decode | 21.1-24 | ~24-27 |
| 500K decode | 17.5 | **~19.7-20.2 t/s** (target 30 — OPEN) |
| 500K prefill | 342 | 332-363 (target 300 — HOLDING) |
| stability | — | battery 3/3 clean ×3 runs; smoke correct |

Prod config: `~/relaunch_exo.sh` on both nodes (v2 ON, pool threshold 8MB,
k=512, no diag envs; previous prod backed up as relaunch_exo.sh.bak-pre-v2).
Variants kept: `relaunch_exo_v2.sh` (+MTP_PROFILE/MTP_LOG diag),
`relaunch_exo_attrib.sh` (MTP off + allsum probe), `relaunch_exo_k192.sh`.
mlx main = 57ffb39a (v2 merged; uv.lock pinned); mlx-lm main @ pool-donation
commits (submodule bumped).

**Path to 30 t/s @500K (residual ~-17ms/token needed), in order:**
1. mx.metal.start_capture kernel diff of ONE verify forward 4K vs 256K —
   pin the unexplained ~10ms/token pool-path term. (The answer is IN the
   pool path: freezing it recovers everything.)
2. Pool-write batching (accumulate K=16 entries, write every 16th flush —
   staleness ≤64 tokens, covered by the 128-token local window; ~30-line
   PoolingCache change sketched in session notes) — recovers the write term
   robustly regardless of donation mechanics.
3. Indexer score GEMV: fp8/int8 indexer pool cache (halves the 32MB/layer
   scan bandwidth) — kernel work, see exo_gather_qmv_rhs_kernel lessons.
4. Draft-side: 4.9ms × ~1 draft-phase per cycle is 10% of cycle — EAGLE_K/
   gamma tuning interplay only after the verify slope is gone.

---

# Session 5 addendum (2026-07-07 overnight) — decode @500K: verify-slope root cause + SDPA dispatch fixes

Goal: decode 30 t/s @500K c=1 (from ~19.7-20.2). Constraint: pure compute
optimizations only — no stability or quality changes.

## Root cause of the verify slope (single-node harness decomposition)

New tool: `~/scratch/verify_slope_ladder.py` (m4-1) — 4-layer random-weight
model, prod quantization, real verify shape (B=1 L=3 trim 2), context ladder
4K/64K/256K, sub-op ablation via /tmp/dsv4_nop_targets + in-process patches
(gates OFF via EXO_DSV4_CATTN_LSPLIT_MAX_L=0 / EXO_DSV4_SPARSE_VERIFY_BATCHED=0).

Findings (p50 per cycle, 4 layers; ×5 ≈ prod's 20 layers per ratio class):

- **Per-layer pool compute is nearly FLAT in context** (ratio-4 base
  11.45→11.90ms from 4K→256K). The prod "indexer 6.3ms / write 3.7ms"
  session-4 numbers reproduce (score+topk 1.15ms/4L ×5 ≈ 5.8 ✓).
- **The session-4 "unexplained ~10ms/token" = the sparse verify per-row
  block** (gather + mask build + per-row L=1 SDPA): sparse0 delta
  1.52-1.86ms/4L ×5 ≈ 9.3ms — k-fixed (why k=192 didn't move it), mostly
  context-flat kernel-count overhead.
- **The real context slope is the ratio-128 (CompressedAttention) dense
  SDPA at L=3**: csdpa0 delta 0.73ms/4L @4K → 3.39ms/4L @256K (L=1 same
  ctx: only 0.65). ×5 ≈ 17ms/verify @256K, extrapolating ~29ms @500K.
  Cause is kernel DISPATCH, not architecture: mx.fast.sdpa at B=1,
  1<L<=8 over a long local+pooled KV falls off the single-query fast
  path and costs ~5x the same work as L separate L=1 calls.

## Fixes landed (mlx-lm `9da3cd4`, adurham main; exo submodule bump `1f4f36289`)

1. **CompressedAttention L-split** (`8162930`): at 1<L<=8, B=1, array
   mask with per-row rows — issue L separate L=1 fused SDPA calls.
   Gate `EXO_DSV4_CATTN_LSPLIT_MAX_L` (default 8; 0 disables).
   Harness: ratio-128 @256K 13.26 → 10.99ms/4L (−2.27ms, −11ms/verify at
   prod scale, grows with ctx). Equivalence: 20 paired forwards @256K,
   worst max|dlogit| 0.024 (1-2 bf16 ulp — same class as landed variant_d
   / pooling fixes), argmax 60/60 identical; L=1 path is the MORE accurate
   fp32-fused kernel (same argument as the sparse per-position split).
2. **Sparse verify batched prep** (`33bb37a`): the small-L sparse path ran
   gather + KV concat + mask fill/broadcast/concat PER ROW (~30 small
   kernels/layer); now built once at (B,·,L,·) with per-row slice views.
   **Bit-exact** (paired forwards: worst diff 0.00000), SDPA calls
   unchanged. Gate `EXO_DSV4_SPARSE_VERIFY_BATCHED` (default 1).
   Harness: −0.21ms/4L @4K (context-flat win).

Dead ends ruled out tonight: fused top-K at k=512 is NOT quality-safe
(threadgroup kernel keeps 4 candidates/thread = 1024 total; at k=512 the
Poisson tail loses ~3% of true top-k — fine at the k<=160 it was built
for, wrong at 512). argpartition/argsort at L=3 are already pipelinable —
not the slope. Isolated-op microbenches UNDERSTATE in-graph costs ~60x
(dependency-chain latency vs throughput) — ablate in-model instead.

## Deploy incident (self-inflicted, resolved) — venv restoration

A plain `uv sync` (and a `--force-reinstall ./mlx-lm` without `--no-deps`)
on the nodes broke the venvs: bare sync uninstalls the whole `mlx` extra
(mlx/mlx-lm/mlx-vlm/mflux/torch — WAN git+https unreachable aborts it),
the no-`--no-deps` install bumped transformers 5.9→5.13 breaking imports.
Restored via the canonical start_cluster.sh sequence (sync --extra mlx
--all-packages + vendored mlx-lm pin + maturin exo_rs rebuild) after
setting `git config --global url."ssh://git@github.com/".insteadOf
"https://github.com/"` on both nodes (makes uv's git fetch work over SSH).
The serving cluster rode through it (imports already resident). Recipe now
in CC memory `exo-node-deploy-gotchas`.

## Session 5, part 2 — pre-existing L==1 prefill-remainder crash found + fixed

The first 500K validation rung died with a runner crash at the sparse
L==1 fast path: `mx.concatenate([lm, pm])` → "dimensions 2 and 4". NOT a
session-5 regression — the identical crash is in the log at 2026-07-06
23:07 on pre-patch code. Trigger: a prefill remainder chunk of exactly 1
token reaches the L==1 path carrying the model-level 2-D (L,S) causal
mask while the gathered sparse_mask is 4-D; prompt-length dependent
(needs an unlucky length ≡ tail-1 in the chunking), which is why the
ladder only hit it sometimes. Fix (mlx-lm `85531a3`, main; submodule
bump in exo): the same 2D→4D normalization the 1<L<=16 branch already
applies. Unit-verified: crash shape now returns output bitwise-identical
to a pre-normalized call (diff 0.0).

## Session 5 FINAL STATE (2026-07-07 ~04:00)

Validation on the deployed build (mlx-lm `85531a3` = L-split + batched
sparse prep + L==1 mask fix; exo `fix/c2-serving-hardening` submodule-
bumped; mlx unchanged `57ffb39a`):

| metric | session-4 end | session-5 end |
|--------|---------------|---------------|
| 4K decode | 33-38 t/s (±20% prompt noise) | 34-35 t/s (same band; harness shows the patch is ≥neutral at 4K) |
| ~300K decode | ~21.3-21.5 t/s | **28.3 t/s** |
| deep rung decode | 17.5-20.2 @500K | **25.6 @586K** (~26.4 interpolated @500K) |
| prefill @586K | 342 @500K | 328-362 (target 300 HOLDING at 586K) |
| verify forward | ~89ms @500K | ~65ms @586K (from cycle math) |
| MTP acceptance @586K | 0.86-1.0 band | 1.07-1.18/2 first windows (≥ band) |
| c=2 battery | 3/3 clean | 3/3 clean (mid-decode admissions, temp 1.0) |
| temp=0 smoke | ✓ | ✓ ('7') |

**Decode @500K: ~26.4 t/s vs target 30 — OPEN, gap ~12%.** (Was 41-52%.)

Cluster state: both nodes on `~/relaunch_exo.sh` (prod, no diag), model
warm, smoke verified. Both venvs restored canonically (see part 2).
Cleanup note: the staged gamma-3 A/B scripts were deleted (never used —
acceptance math says gamma 3 is a wash: per-depth decay too steep; MTP
tokens/cycle is 1.84, so a third draft adds ~0.1 tok for ~7ms of cycle).

**Path to 30 t/s @500K (residual ~-8ms/verify needed), in order:**
1. Remaining O(P) verify terms measured in the harness at 256K, scale ~2x
   to 500K, per 20 layers: indexer score GEMM ~5ms, topk block
   (idxnop-score0 = argpartition + `-scores` + idx-side compressor)
   ~6ms, CompressedAttention L-split residual (q-chain + concat + 3x L=1
   sdpa) ~5.6ms. An EXACT fused top-k needs a histogram/threshold
   two-pass design (the existing threadgroup kernel LOSES ~3% of true
   top-512 — candidates/thread too few; do NOT just raise the gate).
2. Indexer-pool fp8 scan (halves the 32MB/layer score read) — QUALITY-
   SENSITIVE (bf16-perturbation acceptance lesson); needs needle +
   acceptance validation before prod.
3. CompressedAttention: reserved-tail pool layout to kill the per-step
   local+pool concat (~0.4ms, bit-exact, small).
4. Re-run the verify_slope_ladder decomposition at PROMPT_L=500K to
   re-rank 1-3 with the L-split already landed (numbers above are 256K
   extrapolations).

Tooling added this session (m4-1 ~/scratch): verify_slope_ladder.py
(slope decomposition w/ gates), lsplit_equiv.py + sparse_batched_equiv.py
(paired-forward equivalence gates), sdpa_rowsplit_microbench.py,
validate2.sh (smoke/rung/battery driver). Lesson recorded: isolated-op
microbenches understate in-graph costs ~60x — always ablate in-model.

---

# Session 6 addendum (2026-07-07 overnight) — decode @500K: attribution CORRECTED, fused-kernel post-mortem, bitwise node diet

Goal: decode 30 t/s @500K c=1 (from ~25.6 @586K). Constraint: pure compute
optimizations, zero stability/quality changes.

## The load-bearing negative result: NOP attribution deltas were inflated by lazy-graph dead-code pruning

The session-4/5 "unexplained in-graph overhead" (isolated ops ~0.14ms vs
2.3ms NOP delta per 4 sparse layers) was largely an artifact: NOPing a
block (e.g. `sparse_attn` → zeros) makes everything that ONLY feeds that
block dead code in the lazy graph — mx.eval prunes it. So `sparse0`'s
delta included the ENTIRE indexer chain (score GEMM + topk), not just the
gather+SDPA it nominally ablates. Corrected @500K r4 L=3 verify budget
(per 4L → per 21 prod layers, full-width single-node):

| block | corrected delta | per verify |
|---|---|---|
| indexer chain total (idxnop) | 1.60ms/4L | ~8.4ms (score GEMM ~4.0, topk block ~4.4) |
| sparse gather+SDPA TRUE (sparse0 − idxnop) | 0.70ms/4L | ~3.7ms |
| CATTN sdpa + its pruned q-chain (csdpa0, r128) | 1.47ms/4L | ~7.4ms/20L |
| pool write (poolw0) | 0.11ms/4L | ~0.6ms |

Implication: the remaining decode budget is mostly REQUIRED compute
(argpartition sort over P=125K, score GEMM bandwidth, CATTN q-chain), not
removable dispatch overhead. When ablating with /tmp/dsv4_nop_targets,
always difference against a NOP that keeps the upstream chain alive.

## Fused sparse gather-SDPA kernel — built, validated, REJECTED (kept default-OFF)

mlx-lm branch work now on main, gate `EXO_DSV4_SPARSE_FUSED_SDPA=0`
(default). One mx.fast.metal_kernel replacing gather+concat+mask+SDPA loop.
Findings that killed it (all measured):
- These shapes NEVER hit MLX's fused SDPA kernels: `use_fallback` requires
  qsl*gqa <= 32; DSv4 attention is 64:1 MQA (gqa=64) with D=512 (not in
  the vector kernel's head-dim list either). Every decode/verify SDPA call
  is the COMPOSED fallback (bf16 score matmul → where → precise softmax →
  bf16 probs matmul). The mlx-lm "fused fp32 kernel" comments are wrong
  for this model; the CATTN L-split win is actually gemv(M=1) vs gemm(M=3)
  dispatch, consistent with exo_dsv4_gemv_gemm_batchdrift.
- Rounding-matched kernel (bf16-rounded scaled-q/scores/probs, fp32
  fast::exp softmax, reciprocal-multiply) gets to ≤1 bf16 ulp per call
  ("no masks" case bit-exact) but model-level compounding measured worst
  |dlogit| 0.141 with 5/60 argmax flips at 256K — 6x the landed L-split
  bar (0.024, 60/60). Quality gate FAILED.
- GPU-time neutral at D=512: one threadgroup per (b,l,h) row re-reads the
  640-row KV per head (the composed matmul shares K reads across all 64
  heads). A win needs a head-shared flash-style restructure AND bitwise
  gemv/softmax accumulation-order matching. Post-mortem in the kernel
  comment block in deepseek_v4.py.
- sdpa_vector kernel contract documented: masks with batch>1 must be
  (B,H)-dense — the kernel indexes (b*H+h)*head_stride with ONE stride;
  an H=1 mask at B>1 silently reads out of bounds. (Found while chasing a
  fold-equivalence failure; the mask H-broadcasts in the sparse paths are
  load-bearing, now commented as KERNEL CONTRACT.)

Also rejected on measurement: batch-fold of the verify SDPA rows
(EXO_DSV4_SPARSE_VERIFY_FOLD=0 default; neutral-to-slower in-model),
MLX_MAX_OPS_PER_BUFFER raise (50 beat 200/800 in-harness; prod stays 200),
indexer score 3x-gemv split (41% faster isolated but perturbs top-k tie
selection — acceptance risk not worth it), argpartition negation
avoidance (~2%, same tie-perturbation class), reserved-tail pool layout
(BatchPoolingCache surgery for ~0.2ms — stability risk), CATTN
_extend_mask caching (mask/kv-order semantics not fully verifiable
tonight — revisit with fresh eyes on _extend_mask's clamp alignment).

## Landed: bitwise-exact decode node diet (mlx-lm `abdba1e`, main)

`EXO_DSV4_DECODE_NODE_DIET=1` (default): cached verify combined mask
((L,sw)-structural; pool mask is None at L<=8 per PoolingCache.make_mask),
B==1 gather offset-chain skip, per-module attn_sink cast cache, cached
zero-width update_and_fetch values arg. ~200 graph nodes/step removed.
Gate: 20 paired 256K verify forwards worst |dlogit| 0.00000, argmax 60/60
— bitwise-exact by construction (first build runs the legacy op chain).
Harness timing: neutral (the removed nodes were off the critical path —
consistent with the corrected attribution); kept because it is free CPU
work reduction on the 2-rank lockstep path and strictly exact.

## Where the remaining ~8ms/verify must come from (ranked, for next session)

1. Exact fused top-k (histogram/threshold two-pass) for the indexer topk
   block (~4.4ms/verify): NOTE ties at the threshold are selected
   arbitrarily — set-equal only for distinct scores; bf16 ties at P=125K
   are common, so this still needs an acceptance gate, not just a
   bitexact one. argpartition itself is ~2.4ms of real GPU sort time.
2. Head-shared flash-style fused sparse SDPA with bitwise gemv-order
   matching (the full sparse block ~3.7ms + CATTN sdpa share) — hard; see
   post-mortem.
3. Indexer score GEMM (~4.0ms): pure bandwidth (32MB/layer @500K);
   fp8 pool scan halves it but is QUALITY-SENSITIVE (session-4 lesson).
4. MTP cycle shape (draft 4.9ms flat, gamma-3 a wash per session-5) — the
   yield side is already near its ceiling (acceptance 0.86-1.18 flat in
   ctx), only cycle-time cuts remain.

## Session 6 FINAL STATE (2026-07-07 ~08:00) — deployed + validated

Build: mlx-lm `baa3b59` (node diet ON, fused kernel OFF), exo
`fc4218ced` (fix/c2-serving-hardening), mlx unchanged `57ffb39a`.
Both nodes pinned (`uv pip install --no-deps --force-reinstall ./mlx-lm`),
cluster restarted on `~/relaunch_exo.sh` (prod config unchanged).

| check | result |
|---|---|
| temp=0 smoke | ✓ ('7') |
| 586K rung | prefill 331 t/s (target 300 HOLDING), **decode 25.94 t/s** (was 25.6 same depth) |
| c=2 battery | 3/3 pairs clean, 0 degenerated (mid-decode admissions, MTP on) |
| 4K rung | prefill 349 t/s, decode 34.8 t/s (in the 34-38 band) |
| new-build log | zero DEADLINE / SIGKILL / RunnerFailed |

**Decode @500K: ~26.5 t/s interpolated (25.94 @586K) vs target 30 — OPEN.**
Session 6's contribution is knowledge, not tokens/s: the corrected
attribution shows the remaining ~8ms/verify is REQUIRED compute
(argpartition sort ~2.4ms GPU + score GEMM bandwidth + CATTN q-chain),
so the next session should start from the ranked list above (exact fused
top-k with an acceptance gate is the best-value target), not from
overhead hunting.

## Session 6, part 2 — EXACT fused top-k LANDED (mlx-lm `4d87751`)

The ranked-list #1 lever, done this session after all. Histogram/threshold
two-pass Metal kernel replaces `-scores` + argsort/argpartition in
Indexer.__call__ at L<=16 (decode + verify): monotonic 16-bit key over
bf16 scores → exact threshold via two 256-bin histogram passes → one
deterministic index-ordered compaction (all > threshold, then
lowest-index ties to fill k). Gate `EXO_DSV4_EXACT_TOPK=1` (default);
`exact_topk_off` in /tmp/dsv4_nop_targets disables live.

- Exactness: selected score MULTISET always == argpartition's (unit gate:
  heavy-ties/all-const/99%-masked/P=k+1/B=2/L=16 all OK). Tie IDENTITIES
  at the boundary differ (deterministic lowest-index vs argpartition's
  implementation-defined pick) — the same documented arbitrary-ties class.
  Only fires at P > k (ctx >~2K): short-context outputs bitwise unchanged.
- Isolated: 0.145 → 0.045ms at (1,3,125K) k=512 (3.2x). In-model 4-layer
  harness @256K: −0.14ms/4L reproducible (first real in-model timing win
  of the night). Model gate: worst |dlogit| 0.125, 3/48 argmax flips —
  pure tie-set identity effect (etopk-vs-etopk control 0.0e+00).
- Draft-phase note: the MTP module is LocalAttention+MoE+shared-lm_head —
  no indexer/topk; its 4.9ms is dominated by the ~925MB quantized lm_head
  gemv read (bandwidth). Vocab-pruning the draft lm_head would be
  quality-SAFE for outputs (rejection sampling preserves the target
  distribution) but acceptance-risky — future lever, needs measurement.
### Exact top-k e2e validation (2026-07-07 ~10:00) — ALL GREEN, kept ON

| check | result |
|---|---|
| temp=0 smoke | ✓ ('7' — bitwise-unchanged path at short ctx) |
| 4K rung pre/post-battery | 34.99 / 34.88 t/s (band 34-38 — acceptance HELD) |
| 586K rung | prefill 331.1, **decode 27.04 t/s (was 25.94 same depth, +4.2%)** |
| c=2 battery | 3/3 pairs clean, 0 degenerated |

**Decode @500K ≈ 27.6 t/s interpolated (27.04 @586K) vs target 30 — gap
now ~9% (was ~17% at session-5 end).** Cluster state: both nodes on
mlx-lm `4d87751` (exact-topk ON + node diet ON + fused-sdpa OFF), exo
`fix/c2-serving-hardening`, prod `~/relaunch_exo.sh` unchanged, model
warm, all watchers clean. Remaining ranked levers: indexer score GEMM
bandwidth (fp8 pool scan — quality-gate required), head-shared fused
sparse SDPA (post-mortem above), draft lm_head bandwidth (~925MB/step —
vocab pruning is output-quality-safe but acceptance-risky).
