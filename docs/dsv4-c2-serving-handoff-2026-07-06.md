# DeepSeek-V4-Flash c=2 serving hardening — handoff (2026-07-06)

Branch: `fix/c2-serving-hardening` · exo `e3a8250f` · mlx `26ac74b7` (adurham/main) · mlx-lm `d46ac14`

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
  print in `DeepseekV4Model.__call__` — pushed to `adurham/mlx-lm` (commit `5848648`)
  but **reverted from the deploy** (`d46ac14`) because `flush=True` on every prefill
  forward stalls the forward → peer `reliable_all_reduce` deadline → re-place loop.
  Re-enable only via a separate branch if needed; don't ship it.
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
