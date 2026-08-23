# P3 worker D — crash forensics: the `[METAL] GPU Timeout Error` that killed the DSv4 instance during C2's xctrace capture — 2026-08-23

**One-line finding**: rank1's runner (**PID 46718 on `adams-mac-studio-m4-1.local`**, *not* m4-2
— C2's doc has the host↔rank labels swapped) died at **13:51:43.416 CDT** in
`mx.async_eval(y)` inside the DSv4 MoE ffn; the macOS kernel independently logged
**2 GPURestarts in 398 submissions** at 13:51:48, and the 31 s preceding the crash contain a
**jetsam idle-reap cascade + 26 swapfiles created in 18 s** on that node — with **zero thermal
events** and a **7-day-clean GPU-error history**. Verdict: **memory-pressure/paging induced by
xctrace's ~10 GB trace buffer landing on a ~90 GB-resident node**, i.e. a **tracing-procedure
risk, not a demonstrated production depth-scaling failure mode** — but it exposes a real
headroom number (90.3 GB resident / **115.3 GB peak** of 137 GB) that *is* P3-relevant.

**Status: NEW OPEN ITEM.** The model instance is down and has **not** been restored (see §7).

---

## 0. Correction to C2's account (read this first)

C2's doc and the task brief both carry a host↔rank mix-up. Resolved from each node's own
`exo.worker.engines.mlx.utils_mlx:mlx_distributed_init` line:

```
m4-1 exo.log 11:11:00.778  INFO  ... mlx_distributed_init:143 ] Starting initialization for rank 1
m4-2 exo.log 11:11:00.759  INFO  ... mlx_distributed_init:143 ] Starting initialization for rank 0
```

Corroborated three independent ways: the jaccl coordinator role (`rank 0` binds `0.0.0.0:57547`
= listener on m4-2; `rank 1` dials `192.168.200.2:57547` from m4-1), the `[jaccl] tcp coord group
rank=N` markers in each node's runner stderr, and the PID carrying the Metal error.

| | host | node id | rank | runner id | runner PID | outcome |
|---|---|---|---|---|---|---|
| **crash node** | `adams-mac-studio-m4-1.local` (API node) | `d850db36…` | **rank 1** | `6ac91846-…` | **46718** | **GPU Timeout → died 13:51:43** |
| peer | `adams-mac-studio-m4-2.local` (master) | `b13606a3…` | **rank 0** | `f85456ee-…` | 45206 | survived crash; SIGKILLed by hang-watchdog 13:52:06 |

So **"rank1 died" is correct**; "on m4-2 / PID 45206" is not. Consequence for C2's numbers: the
occupancy blocks labelled `100k_rank0` (pid 46718) and `100k_rank1` (pid 45206) are
**swapped** — 82.98% is rank1/m4-1 and 83.06% is rank0/m4-2. Since the two agree to 0.08 pp,
**C2's substantive conclusions are unaffected**; only the labels need flipping.

---

## 1. Full traceback (verbatim, from m4-1 `~/.exo/exo_log/exo.log`)

```
[ 2026-08-23 13:51:43.489 | WARNING | exo.worker.runner.bootstrap:entrypoint:368 ]
Runner 6ac91846-50fe-40fb-a343-5dc4b91ad936 crashed with critical exception
[METAL] Command buffer execution failed: Caused GPU Timeout Error (00000002:***)
Traceback (most recent call last):
  File ".../src/exo/worker/runner/bootstrap.py", line 337, in entrypoint
    runner.main()
  File ".../src/exo/worker/runner/runner.py", line 345, in main
    self.handle_first_task(item)
  File ".../src/exo/worker/runner/runner.py", line 433, in handle_first_task
    return_code = self.handle_generation_tasks(starting_task=task)
  File ".../src/exo/worker/runner/runner.py", line 646, in handle_generation_tasks
    results = self.generator.step()
  File ".../src/exo/worker/runner/llm_inference/batch_generator.py", line 849, in step
    results = self._gen.step()
  File ".../src/exo/worker/engines/mlx/generator/batch_generate.py", line 4228, in step
    _prompt_responses, responses = self._mlx_gen.next()
  File ".../mlx_lm/generate.py", line 2183, in next
    return self._next()
  File ".../mlx_lm/generate.py", line 2103, in _next
    generation_responses = self._generation_batch.next()
  File ".../mlx_lm/generate.py", line 1749, in next
    tokens, logprobs = self._step()
  File ".../mlx_lm/generate.py", line 1591, in _step
    logits = self.model(inputs[:, None], cache=self.prompt_cache)
  File ".../src/exo/worker/engines/mlx/auto_parallel.py", line 686, in patched_call
    logits: mx.array = original_call(self, *args, **kwargs)
  File ".../mlx_lm/models/deepseek_v4.py", line 6894, in __call__
    h = self.model(inputs, cache)
  File ".../mlx_lm/models/deepseek_v4.py", line 6877, in __call__
    *_, (_kind, _idx, out) = self._forward_steps(inputs, cache)
  File ".../mlx_lm/models/deepseek_v4.py", line 6721, in _forward_steps
    h = layer(h, mask, layer_cache, inputs)
  File ".../mlx_lm/models/deepseek_v4.py", line 5156, in __call__
    x = self.ffn(normed, input_ids)
  File ".../mlx_lm/models/deepseek_v4.py", line 3061, in __call__
    mx.async_eval(y)
RuntimeError: [METAL] Command buffer execution failed: Caused GPU Timeout Error (00000002:***)
```

**The crash site is a normal decode step**, not a tracer callback: `_step()` → `model(inputs[:, None])`
→ layer → **MoE `ffn` → `mx.async_eval(y)`**. `inputs[:, None]` confirms **L_q = 1**, i.e. a
single-token decode, so this was a routine decode step at ~100K context, not prefill and not a
batched verify.

Two follow-on facts worth noting:

```
[ 13:51:44.512 | WARNING | ...bootstrap:_release_gpu_memory_before_exit:195 ]
Best-effort MLX buffer release on exit failed:
RuntimeError('[METAL] Command buffer execution failed: Caused GPU Timeout Error (00000002:***)')
```
— the GPU context was *already* unusable, so even teardown couldn't submit. And the error
repeats **269 times** in m4-1's `exo.log` (every in-flight command buffer failed at once),
which is why the supervisor needed 7 attempts to reap the process.

---

## 2. Absolute timeline (all times CDT, 2026-08-23)

| time | node | event | source |
|---|---|---|---|
| 11:10:58.6 | both | runner processes launch | `sample` header / exo.log |
| 11:11:00.78 | m4-1 | `mlx_distributed_init: … rank 1`; m4-2 gets rank 0 | exo.log |
| 13:45:03.777 | client | PROMPT_READY, 544,306 chars, **~100,021 tokens** predicted | C2 doc |
| 13:50:07.701 | client | **DECODE_START**, ttft = 303.92 s | C2 doc |
| 13:50:14.978 | m4-1 | `xctrace[57262]` starts (LaunchServices/eGPUOverrides) | unified log |
| 13:50:15.15 | m4-1 | DTServiceHub attaches GPU/Metal instrument services | unified log |
| 13:50:16.695 | m4-1 | `runningboardd`: xctrace **"is not RunningBoard jetsam managed"** / "not memory-managed" | unified log |
| 13:51:03.72 | both | 50 s `--time-limit` expires, **tracer detaches**; finalize begins | C2 doc |
| 13:51:02–:09 | m4-2 | decode healthy: `step overhead … total=37.65–38.49ms` | m4-2 exo.log |
| **13:51:12.35–.37** | **m4-1** | **jetsam cascade**: ≥25 idle daemons reaped, `JETSAM_REASON_MEMORY_LONGIDLE_EXIT`; `memorystatus_available_pages 1,539,245 → 1,534,024`, `compressor_size 1,540,655 → 1,545,210` (**≈23.6 GB compressed**) | kernel/xnu memorystatus |
| 13:51:13.975 | m4-2 | decode degrades: `total=85.10ms` (2.2× the healthy step) | m4-2 exo.log |
| 13:51:15 | m4-1 | first swapfile created | apfs kernel |
| **13:51:25→13:51:43** | **m4-1** | **swapfiles 1→26 created in 18 s** (~1.4/s) | apfs kernel |
| 13:51:25→13:52:16 | m4-2 | m4-2 also swaps hard (**73** swapfile events) | apfs kernel |
| 13:51:36.14 / :41.38 | m4-1 | `[wait_for_one] slow: elapsed=3.0–3.1s n_active=6` | runner stderr |
| **13:51:43.4158** | **m4-1** | **first `IOGPUMetalError` + `(Metal) Execution of the command buffer was aborted … Caused GPU Timeout Error (00000002:***)` on PID 46718** | unified log |
| 13:51:43.489 | m4-1 | **runner 6ac91846 crashes**, traceback above | exo.log |
| 13:51:44.512 | m4-1 | teardown buffer-release also fails (context dead) | exo.log |
| **13:51:48.043** | **m4-1** | **kernel `IOGPUFamily`: `Deny submissions/ignore app[] with 2 GPURestarts in 398 submissions`** | kernel |
| 13:52:04.34 | m4-1 | runner exits signal 15 after "**7 attempts :)**" | exo.log |
| 13:52:04.50 | m4-1 | `Instance 6df9afc1 exceeded 5 retries, requesting deletion` → `DeleteInstance` | exo.log |
| 13:52:06.17 | m4-2 | hang-watchdog: rank0 runner `no event for 47s (>45s)`, **SIGKILL** | exo.log |
| 13:52:07.13 | m4-2 | thread dump `/tmp/exo_hang_45206.txt`: **footprint 90.3 GB, peak 115.3 GB**, blocked in `async_eval → wait_for_one → __psynch_cvwait` | sample |
| 13:52:20.65 | m4-2 | rank0 runner terminated `-9` | exo.log |

**Context depth**: ~100,021 predicted / 100,026 real (C2's inference; no `usage` block was
returned because the stream died). **Memory-hold duration**: the runners came up at
**11:10:58**, so the node had been holding the model resident for **2 h 41 min** at crash time.

---

## 3. Telemetry findings

### 3.1 GPU driver — POSITIVE, and it is kernel-side
```
13:51:43.4158  python3.13[46718]: (IOGPU)  IOGPUMetalError: <private>
13:51:43.4158  python3.13[46718]: (Metal)  Execution of the command buffer was aborted due to
                                            an error during execution. Caused GPU Timeout Error (00000002:***)
13:51:48.0435  kernel[0]: (IOGPUFamily) void IOGPUCommandQueue::retireCommandBuffer(IOGPUEventFence *):
                          Deny submissions/ignore app[] with 2 GPURestarts in 398 submissions.
```
This is the single most important line in the whole investigation. **The GPU was actually
restarted twice by the driver**, and the driver then blacklisted the process's submissions. This
is not an MLX-level bookkeeping error or a spurious Python exception — the hardware/driver
genuinely wedged and recovered. `398 submissions` also brackets the damage: only ~398 command
buffers were in flight/attempted across the failure.

Note the launch env sets `MTL_DISABLE_TIMEOUT=1`, `MTL_COMMAND_BUFFER_TIMEOUT=0`,
`EXO_DISABLE_METAL_TIMEOUT=1`, `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1`. **A timeout fired anyway** —
those knobs evidently do not cover the `IOGPUFamily` kernel-side watchdog. Worth knowing before
anyone concludes "we already disabled Metal timeouts, so this can't be a timeout."

### 3.2 Thermal — NEGATIVE, unambiguously
Zero thermal-pressure, throttle, or `thermalmonitord` events on either node in 13:44–13:56.
The only `thermalLevel` hits in a 3-day sweep are `locationd` statedumps reporting
`"thermalLevel":-1` (= not applicable on a desktop Mac). Mac Studios are actively cooled and
were on a mains-powered desk, not a thermally constrained enclosure. **Thermal is ruled out.**

### 3.3 Memory pressure / paging — POSITIVE and tightly time-aligned
Two independent kernel signals on the crash node, both inside the 31 s preceding the crash:

1. **Jetsam idle-reap cascade** at 13:51:12.347–13:51:12.372 — ≥25 daemons killed in 25 ms with
   `JETSAM_REASON_MEMORY_LONGIDLE_EXIT` (coreduetd, modelcatalogd, findmybeaconingd,
   containermanagerd_system, fskitd, rtcreportingd, securityd_system, backupd, osanalyticshelper,
   countryd, online-auth-agent, seputil, endpointsecurityd, …). Compressor was at
   `compressor_size ≈ 1.545 M pages × 16 KB ≈ 23.6 GB`.
2. **Swap explosion**: 26 swapfiles created on m4-1 between 13:51:15 and **13:51:43** — the last
   one in **the same second as the GPU timeout**. m4-2 saw 73 swapfile events over a slightly
   longer window.

Baseline check — is that swap burst normal for this box? Swapfile events on m4-1 for the whole
day, bucketed by minute:
```
02:58 ×1 | 03:16 ×24 | 03:17 ×27 | 03:18 ×8 | 03:24 ×6
10:55 ×6 | 10:56 ×21 | 11:09 ×13 | 11:10 ×11 | 13:51 ×27
```
Every single burst coincides with an xctrace/P2/P3 capture window (03:14–03:27 = P2 v2;
10:49–11:10 = P2 prefill-wedge incidents; 13:51 = this crash). **The machine does not swap during
ordinary operation — it swaps when a tracer is running.** That is a strong, independent
corroboration of the mechanism.

### 3.4 `.ips` / DiagnosticReports — no runner crash report
`/Library/Logs/DiagnosticReports/` on m4-1 has no `.ips` for python3.13 around 13:51 — expected,
since the runner died by a caught Python exception then SIGTERM, not by an unhandled signal.
The nearby entries are all Instruments' own (`xctrace_2026-08-23-142417*.diag`,
`com.apple.dt.instruments.dtsecurity_2026-08-23-135026*.diag` — that last one at **13:50:26** is
the tracer attaching for this very capture). No GPU-restart panic report was generated; the
driver recovered in place rather than panicking.

### 3.5 Peer (rank0 / m4-2) view — no jaccl/RDMA fault
m4-2's log across 13:50–13:56 contains **only** three `[wait_for_one] slow: elapsed=3.0–3.1s
n_active=6` lines and no jaccl/RDMA error, no peer-loss, no reconnect, no QP fault. Its decode
was healthy (37.5–38.5 ms/token) until 13:51:13.975, when one step jumped to **85.10 ms** — that
is m4-2 *waiting on its dying peer*, not m4-2 failing. It then sat blocked in
`async_eval → wait_for_one → __psynch_cvwait` until the 45 s hang-watchdog SIGKILLed it at
13:52:06. **rank0 is a collateral casualty; the fault is entirely rank1/m4-1's.**

### 3.6 Historical baseline — the machine was clean before today
| query | m4-1 (crash node) | m4-2 |
|---|---|---|
| `GPU Timeout Error` / `GPURestart` in unified log, **last 7 days** | **0 before today**; only the 13:51:43 burst + the 13:51:48 GPURestart line | **0, ever** |
| `GPU Timeout Error` in the 5 rotated `exo.*.log.zst` (2026-08-22 → today) | **0 in all 5** | **0 in all 5** |
| current `exo.log` | 269 (all from this one crash) | 0 |

**No pre-existing Metal instability.** T2/T5 ran xctrace on this same cluster on 2026-08-22 and
m4-1's own P2 captures ran at 03:14–03:27 and 10:49–11:10 today — none produced a GPU timeout or
a GPURestart. This failure is new and singular.

---

## 4. Verdict — candidate mechanisms ranked by evidence

### 1. **xctrace stop/finalize memory pressure on a ~90 GB-resident node** — *best supported*
Direct, time-aligned kernel evidence on the crash node: jetsam cascade at T−31 s, 23.6 GB
compressor, 26 swapfiles in the 18 s ending at T−0. The trace was **10 GB while recording**
(C2), and `runningboardd` explicitly logged that **xctrace is neither jetsam-managed nor
memory-managed** — so its buffer competes with the model for RAM with no OS backstop. Node
footprint was **90.3 GB resident / 115.3 GB peak of 137 GB**; +10 GB of trace buffer is enough
to cross into the swap regime, and the swapfile baseline (§3.3) shows this box *only* swaps when
a tracer runs. Once pages are being compressed/swapped, a GPU command buffer touching those
pages stalls long enough to trip the driver watchdog → 2 GPURestarts → the exception.
This refines C2's hypothesis: **the load axis is memory, not disk I/O** (see #5).

### 2. **Latent Metal/MLX kernel fragility exposed by that pressure** — *plausible contributor,
not separable*
The failing op is `mx.async_eval(y)` in the **MoE ffn** — the largest per-step allocation and
dispatch site in DSv4. Whether the command buffer "genuinely took >watchdog because its pages
were being paged" or "a specific MoE kernel wedged under allocator stress" is **not
distinguishable from the telemetry I have** — the trace has 99.98% generic "Compute" channel
names (C2 §8), so there is no per-kernel attribution. I am not going to pick between these.

### 3. **Coincidence / pre-existing instability** — *refuted*
7 days clean on m4-1, 5 rotated exo logs clean, m4-2 clean ever, and the crash sits inside a
31 s window of extreme memory events that themselves only occur when a tracer runs. The prior
probability of an unrelated first-ever GPU timeout landing in that specific window is negligible.

### 4. **Trace-finalize *disk* I/O** — *weak / not the driver*
C2 suspected the 10 GB→1.7 GB finalize's I/O. Disk had 137 Gi free; the only NVMe log lines are
routine `systemPowerChange`; no I/O errors or queue stalls. The apfs lines in the window are
**swapfile truncations** — i.e. the disk activity that shows up is *caused by* memory pressure,
not an independent I/O cause. Demoted below memory.

### 5. **Thermal** — *ruled out* (§3.2, zero events).

### Honest residual
I can attribute the crash to the **memory-pressure regime** with high confidence, and I can rule
out thermal and pre-existing instability outright. What I **cannot** do from retrospective logs
is separate mechanism #1's final step (paging stalls a command buffer past the watchdog) from
mechanism #2 (a specific MoE kernel wedges under allocator stress) — both predict exactly the
observed `IOGPUFamily` GPURestart at exactly this call site. **Insufficient telemetry to
distinguish 1-final-step vs 2**; §6 says what would separate them next time.

---

## 5. Production risk or tracing risk? — **primarily a TRACING-PROCEDURE risk**

This distinction matters for P3's conclusions, so stating it plainly.

**It is a tracing risk, on this evidence:**
- The trigger chain (jetsam → compressor → 26 swapfiles) is attributable to the **10 GB trace
  buffer**, which does not exist in untraced operation. `runningboardd` confirms xctrace is
  exempt from jetsam/memory management, so it takes RAM without the OS protecting the model.
- The swapfile-per-minute baseline shows **this node does not swap except when a tracer runs**.
- **7 days of untraced deep-context decode on this cluster produced zero GPU timeouts and zero
  GPURestarts** — including worker C's untraced 352.6K and 500K attention benches and B1's
  untraced depth anchors, all completed on this hardware without this failure mode.
- The peer node, which was *not* running a tracer against a full model + 10 GB buffer, never
  produced a Metal error at all.

**But there is one genuine production-relevant caveat, and it should not be waved away:**
the sampled runner footprint was **90.3 GB resident with a 115.3 GB peak, out of 137 GB**. The
tracer consumed the remaining headroom — but *so would a larger KV cache*. At 100K the model +
KV already sits at ~90 GB; the peak already touched 115 GB. **Nothing in this incident proves a
500K-context untraced decode can't reach the same jetsam/swap regime on its own.** The crash
does not demonstrate a depth-scaling production failure, but it does hand P3 a hard number for
how little headroom is left at depth, and that headroom shrinks as context grows.

**Bottom line for P3**: do not report this as "deep-context decode is unstable." Report it as
"tracing at depth is unsafe above a small window, **and** the node runs at ~90/137 GB at 100K,
so memory headroom is a depth-scaling risk to measure directly rather than infer from this
crash."

---

## 6. What to instrument next time

1. **Background `powermetrics` logger during every capture.** C2 correctly skipped it (no
   passwordless sudo). With `sudo powermetrics --samplers gpu_power,thermal -i 1000` writing to a
   file for the whole capture+finalize window, mechanism #1 vs #2 becomes separable: GPU
   residency/clock collapsing *before* the swap burst points at the kernel; collapsing *with* it
   points at paging. **This is the single highest-value addition.**
2. **Sample `memory_pressure` + `vm_stat` + `sysctl vm.swapusage` once/second** on the traced
   node from capture start through finalize end. Cheap, no sudo, and would have made the
   90 GB→swap crossing a measured curve instead of an inference from swapfile-creation timestamps.
3. **Log the runner's `mx.get_active_memory()`/`get_peak_memory()` per N steps** during any traced
   run, so model-side headroom is on the same timeline as the OS-side pressure.
4. **Watch for `IOGPUFamily … GPURestarts` explicitly** — `log stream --predicate 'eventMessage
   CONTAINS "GPURestart"'` — as a live abort signal. It fired 4.6 s *after* the Python exception
   here; as a live tripwire during a capture it is the earliest unambiguous "the GPU is in
   trouble" indicator available without sudo.
5. **Cap the trace budget against measured free RAM**, not against wall-clock alone. The real
   constraint is `trace_buffer_GB + resident_GB < RAM − margin`; C2's 12–15 s recommendation is a
   good proxy but the direct quantity is memory.
6. If a deep+long capture is genuinely required, **run it against a synthetic/idle-model process**
   or on a node not holding the full 90 GB working set, per §12's existing guidance.

---

## 7. Current cluster state snapshot — what needs restoring (NOT restored by me)

Read-only `curl http://adams-mac-studio-m4-1.local:52415/state` at 14:58 CDT, plus `ps`/`vm_stat`
on both studios.

**Cluster control plane: UP. Model instance: DOWN.**

```
instances: {}                                     <-- nothing serving
runners:
  6ac91846-…  (m4-1 / rank1)  RunnerFailed        <-- carries the GPU-timeout traceback
  f85456ee-…  (m4-2 / rank0)  RunnerRunning       <-- STALE: the process is gone (SIGKILL -9 at 13:52:20)
tasks: CreateRunner a653c68b-… Complete, instanceId 6df9afc1-c2c3-4ef8-9e12-36d9f6f7f2a7,
       model deepseek-ai/DeepSeek-V4-Flash (MlxJacclInstance, 2-way shard)
lastSeen: both nodes seen <1s ago — zenoh/election healthy, both nodes present
```

| check | m4-1 (rank1) | m4-2 (rank0) |
|---|---|---|
| `python -m exo` supervisor alive | **yes**, PID 43509, up 4 h 11 m | **yes**, PID 42153, up 4 h 11 m |
| runner child process | **gone** | **gone** |
| RAM free (`/state` nodeMemory) | **133.5 GB of 137.4 GB free** | **133.4 GB of 137.4 GB free** |
| memory released? | **yes** — 97% free, compressor drained to 9,695 pages | **yes** — 97% free, compressor 12,554 pages |
| swap | 146 MB / 1024 MB used (drained) | 129 MB / 1024 MB used (drained) |
| xctrace / Instruments still running | **none** | **none** |
| GPU state | recovered — no errors since 13:51:48 | never faulted |

**No leaked memory, no zombie runner, no stuck tracer, GPU healthy on both nodes.**

**To restore** (a human or an appropriately-scoped worker — *not* me, per this task's hard
read-only rule): re-place the DSv4-Flash instance the usual way. Both `python -m exo` supervisors
are still running with the full env from the 10:27 launch, so a fresh instance placement should
suffice without restarting the cluster processes. The stale `RunnerRunning` for `f85456ee` should
clear on re-placement; if it does not, that is itself worth noting as a supervisor-state bug.

**Also present (not mine, do not disturb)**: m4-2 is currently running
`/tmp/p3_donation_insitu_harness.py --depths 100026 --steps 128` (PID 60996, 477 MB) — a sibling
P3 worker's bench.

---

## 8. Limitations

- **Retrospective only.** No live instrumentation existed during the crash; everything here is
  reconstructed from `log show`, exo logs, and one `sample` dump.
- **Mechanism #1's last step vs #2 is unresolved** (§4). The trace's channel names are 99.98%
  generic "Compute", so no per-kernel attribution is possible even from the surviving trace.
- **n = 1.** One crash, one node, one depth. The mechanism ranking rests on time-alignment and a
  clean 7-day baseline, not on a reproduction. **I did not attempt to reproduce it** (would
  require running a GPU workload — forbidden here, and it would risk the cluster again).
- **Depth is inferred**, not returned (~100,021 predicted / 100,026 real), inherited from C2 —
  the stream died before the `usage` block.
- **Decode step index not determinable.** The logs give per-step timings but no absolute step
  counter for the failing step; from the stream, ~1330 events elapsed in the traced window and
  the crash came ~40 s after decode start.
- The **90.3 GB / 115.3 GB footprint figures are m4-2's** (`sample` was taken on the peer). m4-1's
  own footprint was not sampled before it died, but the two ranks hold symmetric shards, so
  m4-1's was materially the same.
- **Nothing was committed to git**; nothing on either studio was modified, killed, or restarted;
  all scratch files I created under `/tmp` on both studios were removed.

---

## 9. Files

Local (this Mac, scratch — `/tmp/p3d/`): `traceback.txt`, `m41_crash_pre.txt`,
`m41_crash_post.txt`, `m41_hist.txt`, `m41_swap.txt`, `m41_fin.txt`, `m42_peer.txt`,
`m41/{gpu,thermal,mem,wd}.txt`, `m42/{gpu,thermal,mem,wd}.txt`, `state.json`.

On the studios: **nothing left behind** — all `/tmp/p3d_*` scratch removed from both nodes.

Related: `docs/p3-worker-c2-depth-busy-idle-capture-2026-08-23.md` (the capture that triggered
this), `docs/p2-xctrace-prefill-collective-wedge-2026-08-23.md` (§12 HAZARD, prior incidents).
