# I1 — Collective latency: is the per-layer fence waiting on the PEER or on the LOCAL GPU?

Campaign 2 / round 1 / I1. Written 2026-09-03.

**VERDICT: (b) LOCAL-COMPUTE-WAIT.** The per-layer `mx.eval` fence is
overwhelmingly waiting on the local rank's own GPU to drain its preceding
work, not on the peer rank arriving at the collective. The collective is
merely *where the sync lands*.

**NO RELAUNCH WAS SPENT ON THE MEASUREMENT.** The prior record already
answers the question decisively, from four independent instruments, one of
which is a direct control experiment with no peer involved at all. Per the
supervisor's explicit STEP-0 authorization, the discriminating instrumentation
was not built and no profiled run was performed. A relaunch *was* performed,
but only to execute the mandatory STEP-4 revert-and-restore — and it turned
out to be independently necessary, because node2's runner was found dead
(see CLUSTER HEALTH VERIFICATION).

---

## RECONCILIATION WITH PRIOR RECORD

### What was already established (cites)

The brief listed six things as already-known. All six are confirmed present in
the record, and I read the primary sources rather than relying on the
`PERFORMANCE_HISTORY.md` summary alone.

| # | Finding | Source |
|---|---|---|
| P1 | Per-layer all_sum drain ~1.4ms × 43 layers, UNIFORM. The apparent 35% spread was iter-0 cold-compile-cache contamination. | `.hermes/plans/2026-05-19_allsum_tail_findings.md` §Results |
| P2 | Chained-collective peer-CQE-arrival-tail hypothesis FALSIFIED: FENCE=1 made verify WORSE (63.16ms vs 57.10ms, +10.6%) — the opposite of its prediction. | same doc, §"What FENCE=1 vs FENCE=43 tells us" |
| P3 | jaccl/RDMA is not the lever: poll wall 1.9% of verify cost (mean 8.15µs, median 5.00µs) vs ~98% attributable to the `mx.eval` fence (median p50=37.4ms). | `.hermes/plans/2026-05-19_phase_f_findings.md` §Data / §Decomposition |
| P4 | 96–97% GPU "utilization" coexists with `moe.all_sum` eating 61–64% of wall, because a blocked-but-scheduled submission thread reads as "busy". | `PERFORMANCE_HISTORY.md` §2.6 (GPU utilization reconciliation), §12.5 ¶2 |
| P5 | Three independent dispatch-fusion / fence-reordering attempts to capture a GPU-idle envelope during `moe.switch_mlp` all FAILED. | `PERFORMANCE_HISTORY.md` §4.4, §12.5 ¶2 |
| P6 | CPU-to-GPU dispatch latency = 96.8µs ± 4.6µs (Instruments Metal System Trace). | `PERFORMANCE_HISTORY.md` §2.7 |

**Correction to the brief:** it states
`.hermes/plans/2026-05-19_phase_f_findings.md` is "referenced but MISSING from
disk". It is **not** missing — it is present (5157 bytes, 140 lines) and I read
it directly. P3 above is cited from the primary source, not from the summary.

### What the record ALSO contains that the brief did not list — and which settles the question

Four further items, each an independent instrument, and none of them a repeat
of P1–P6:

**R1 — jaccl-INTERNAL C++ transport timing (2026-08-21/22).**
`docs/jaccl-internal-timing-allsum-transport-fast-2026-08-21.md`. Real
`std::chrono::steady_clock` timestamps placed *inside* jaccl's C++
`all_reduce<T>()` — explicitly immune to MLX/Python graph laziness (the doc
records that a `perf_counter`-around-call-site design was reviewed and
rejected as methodologically unsound). 45,666 real decode-time 8192-byte
`moe.all_sum` calls per rank:

| | rank0 | rank1 |
|---|---|---|
| median | 36.1µs | 36.0µs |
| mean | 66.3µs | 58.9µs |
| p75 | 63.0µs | 62.5µs |
| p95 | 165.4µs | 142.8µs |
| p99 | 252.6µs | 266.1µs |

Only 18/45,666 calls (0.04%) exceed 4094µs.

**R2 — the GPU→CPU stream-boundary control experiment (2026-08-20).**
`PERFORMANCE_HISTORY.md` §2.3, from
`docs/phase0a-allsum-boundary-decomposition-2026-08-20.md`. The stream-boundary
coherency cost (required because MLX collectives are CPU-stream-only) is
payload-proportional and **"NOT collective-specific — a plain non-collective
CPU-stream op reproduces the same 2.66x penalty."**

**R3 — arithmetic reconciliation of the sync-span artifact (2026-08-22).**
`docs/allsum-sync-span-artifact-arithmetic-check-2026-08-22.md`. The old
"4094µs/call" was shown impossible (43 × 4094µs = 176ms/token predicted vs
53.48ms/token real measured — 3.29×). The stated mechanism is precisely the
peer-vs-local discriminator: `mx.synchronize()` at a span boundary *"drains
MLX's ENTIRE pending lazy graph since the last sync point (not just the spanned
op), so a span ending right after `all_sum` misattributes real upstream
GPU-compute time from prior layers to the collective specifically."*

**R4 — the overlap gate (2026-08-20).** `PERFORMANCE_HISTORY.md` §2.6 Phase 0b.
`all_sum` runs on a CPU stream in MLX; overlap is destroyed by *"ordinary
same-GPU-stream FIFO ordering, **not** a device-wide barrier"*, and ~0% overlap
*"LOOKS EXACTLY LIKE (but isn't) a hardware drain — likely why prior sessions
wrongly concluded collectives 'block everything.'"*

### What I added

- Read the two `.hermes/plans/` primaries directly (one of which the brief
  believed was missing) rather than trusting the summary.
- Assembled R1–R4, which the brief's "already established" list omitted, into
  the explicit peer-vs-local discrimination below.
- **Live confirmation of the collective count = 43** (secondary objective).
- Independent review of the reasoning by a second model (see caveat C3).
- Found and fixed a genuinely down node2 runner.

---

## DISCRIMINATING MEASUREMENT

### Design decision: the discriminating measurement already exists

The brief proposed building two brackets: (i) a separately-`mx.eval`'d
pre-collective MoE bracket, and (ii) per-rank collective-entry timestamps
logged by call index. Both were reviewed against the record before building:

**(i) is a strictly weaker version of R2.** The proposed bracket would attribute
per-layer time between "local MoE compute" and "the collective". R2 already
performed the *stronger* form of that experiment — it ran the same
stream-boundary op with **no collective and no peer at all** and reproduced the
same 2.66× penalty. An experiment that removes the peer entirely dominates one
that merely times around it.

**(ii) is bounded to insignificance by R1, and its known failure mode is
already documented.** Per-rank entry timestamps are designed to expose skew.
But a blocking `all_reduce` whose peer has not yet arrived blocks *inside* the
transport call — which is exactly the region R1's C++ timestamps wrap. R1
measures 45,666 real calls per rank at median 36µs with near-identical
cross-rank distributions. Against a ~1400µs per-layer cost, peer-wait is
bounded at ~2.6% by the median and ~4.7% by the mean. Separately, the brief
itself notes the two nodes' clocks may not align well enough for cross-rank
correlation; the record confirms this concretely — the live two-rank
Instruments trace (2026-08-21) states the two captures *"used independent,
non-synchronized system clocks, so true cross-rank wall-clock gap correlation
was not attempted."* The fallback the brief proposes (compare rank0-vs-rank1
wait *distributions* for asymmetry) is exactly what R1 already provides, at
n=45,666 per rank, and it comes out **symmetric**.

Building either bracket would therefore have produced a weaker restatement of
data already in hand, at the cost of a relaunch. Per STEP 0, I skipped it.

### The numbers that discriminate

Per-layer budget = ~1400µs (P1).

| Component | Measured | Share of 1400µs | Peer-dependent? | Source |
|---|---|---|---|---|
| jaccl transport, median | 36.1 / 36.0 µs | **2.6%** | YES | R1 |
| jaccl transport, mean | 66.3 / 58.9 µs | **4.7%** | YES | R1 |
| jaccl transport, p99 | 252.6 / 266.1 µs | 18.0% | YES | R1 |
| RDMA poll loop, median | 5.00 µs | **0.36%** | YES | P3 |
| RDMA poll loop, mean | 8.15 µs | 0.58% | YES | P3 |
| CPU→GPU dispatch | 96.8 ± 4.6 µs | 6.9% | NO | P6 |
| **Residual (local GPU drain + dispatch/sync)** | **~1330 µs** | **~95%** | **NO** | by difference |

Cross-rank symmetry (the asymmetry test the brief asked for, at n=45,666/rank):

| Statistic | rank0 | rank1 | Asymmetry |
|---|---|---|---|
| median | 36.1µs | 36.0µs | 0.3% |
| mean | 66.3µs | 58.9µs | 12% (rank0 higher) |
| p95 | 165.4µs | 142.8µs | 16% (rank0 higher) |
| p99 | 252.6µs | 266.1µs | 5% (rank1 higher) |

There is no chronically-late rank. The p95/p99 asymmetries point in *opposite
directions*, and the absolute magnitudes (tens to low hundreds of µs) are
negligible against 1400µs regardless of sign.

### The three structural arguments

**S1 — the no-peer control (strongest).** R2 reproduces the same 2.66×
stream-boundary penalty with a plain non-collective CPU-stream op. There is no
peer in that experiment. A cost that survives the removal of the peer cannot be
peer-wait.

**S2 — the falsified prediction, with a signed fingerprint.** P2: fencing every
layer made verify *worse* (+10.6%), the opposite of what a peer-arrival tail
predicts. Steady-state per-layer p99/p50 is only ~1.05–1.15 — no straggler tail
exists to clip. Decisively, the per-layer cost **alternates deterministically
with each layer's own `compress_ratio` config** (compress=4 layers ~1.50ms vs
compress=128 layers ~1.28ms;
`.hermes/plans/2026-05-19_allsum_tail_findings.md` §Per-layer pattern). Peer
skew has no reason to track a *local* per-layer architectural constant. Local
compute shape does, by construction.

**S3 — the misattribution mechanism is documented, not inferred.** R3 states
outright that a span ending after `all_sum` bills upstream local GPU compute to
the collective. R4 supplies the matching mechanism (same-GPU-stream FIFO
ordering, explicitly *not* a device-wide barrier) and even predicts this exact
misreading.

---

## THE ANSWER: PEER-WAIT OR LOCAL-COMPUTE-WAIT

### **(b) LOCAL-COMPUTE-WAIT.**

The local rank is waiting on **its own GPU** to finish the preceding MoE
compute. The collective is merely where the synchronization lands. All
peer-dependent components together account for **~3% (median) to ~5% (mean)**
of the ~1400µs per-layer cost; the remaining **~95%** is local GPU drain plus
CPU→GPU dispatch/sync overhead.

**The lever is overlap/pipelining, NOT comm or scheduling.**

Corollaries, already independently supported:

- The comm lever is already closed. P3 (jaccl 1.9%) and R1 (transport 36µs)
  both bound it; `PERFORMANCE_HISTORY.md` §2.7 computes the perfect-overlap
  ceiling at only 2.9–5.3% of wall time.
- The overlap lever is already largely captured. `EXO_DSV4_FENCE_ASYNC=1` is
  the comm/compute overlap design, live in production, measured at +1.04% on a
  clean re-A/B (§2.6) — coherent with a ~3–5% ceiling. The historic "+28%"
  claim was traced to the same forced-sync artifact class (§2.6, RESOLVED
  2026-08-22).
- P5's three failed dispatch-fusion / fence-reordering attempts are consistent
  with (b): the time is real local GPU work, so reordering fences cannot
  recover it — only genuinely overlapping or reducing the compute can.

This is **not** ambiguous. Four independent instruments (C++ transport
timestamps, RDMA poll instrumentation, a no-peer control experiment, and
end-to-end wall-clock arithmetic) converge, and the one hypothesis that
predicted (a) was falsified by a signed, opposite-direction result.

### Caveats

**C1 — MANDATORY CAVEAT, stated as required.** Production runs
`EXO_DSV4_FENCE_ASYNC=1` (non-blocking `mx.async_eval`). Every bracketed
measurement discussed here forces blocking evals, which destroys overlap that
production genuinely has. **Any "share" figure is an UPPER BOUND, and no
profiled forward time in this document or its sources may EVER be quoted as a
decode-latency number.** This is the same artifact class that produced the
"4094µs/call", "21.4% of wall", and "+28% decode" claims, all three of which
the record has since retracted.

**C2 — staleness.** R1/R2/R3/R4/P4/P6 are ~2 weeks old; P2/P3 ~3.5 months. The
repo's own §12.5 ¶4 warns against trusting old results across a code gap.
I judged this acceptable because the question is a **structural attribution**
(*where* the wait lives) resting on ratios of ~36µs to ~1400µs — a ~39×
separation. Nothing in the intervening period changed the TP topology, the
transport, or MLX's CPU-stream collective model. A throughput *number* would
not survive this staleness argument; this attribution does.

**C3 — independent review.** The reasoning was reviewed by a second model,
which returned verdict (b), independently nominated S1 (the no-peer control) as
the strongest discriminator with S2 as backup, and flagged one wording issue,
adopted here: it is imprecise to say the *only* thing that can delay a rank
from reaching the transport call is materializing its own local input. Other
pre-call delays exist (MLX scheduler/dispatch latency, Python/GIL, thread
wakeup, allocator, prior-collective completion callback). All of these are
nonetheless **local, non-peer** costs, so they fall inside bucket (b) and the
verdict is unaffected. It also correctly cautioned that R1's cross-rank
symmetry alone is weaker than it looks (two ranks with *symmetric* skew would
also appear identical) — which is why S1 and S2, not R1, carry the argument.

### Secondary objective: live collective count = 43, CONFIRMED

Confirmed live on both nodes' production runners (post-restore):

```
--- macstudio-m4-1 ---        --- macstudio-m4-2 ---
EXO_DSV4_ATTN_ALLSUM=0        EXO_DSV4_ATTN_ALLSUM=0
```

With `_ATTN_ALLSUM` false, all four attention-path `all_sum`/`all_gather` sites
in `deepseek_v4.py` (`:4493`, `:4798/:4807/:4817`, `:5256/:5265/:5275`) are
gated off — each guarded by `self.sharding_group is not None and _ATTN_ALLSUM`
or the `elif` of the same. That leaves exactly one collective per layer, the
MoE `all_sum` at `:3249`, over 43 layers.

**Count = 43 per forward, not 86.** The code-read claim is confirmed live. This
also matches R1's independent byte-histogram evidence (45,666 calls at exactly
8192 bytes = `hidden_size=4096 × 2 bytes/bf16`, a single per-layer MoE payload).

---

## CLUSTER HEALTH VERIFICATION

### Incident found and fixed: node2's runner was DEAD on arrival

Before touching anything, read-only inspection found the cluster **half-down**:

```
=== node1 screen ===              === node2 screen ===
There is a screen on:             No Sockets found in
  15379.exorun (Detached)         /var/folders/.../T/.screen.

=== node2 exo/python procs ===
(empty)

=== node2 log mtime vs now ===
Thu Sep  3 14:36:24 CDT 2026      <- now
-rw-r--r-- 895976 Sep  3 14:27    <- exo.log frozen 9 min earlier
```

node1's `/state` corroborated: `"instances":{}` with one runner in
`RunnerShuttingDown`. Node2's log tail ends in `anyio` `TimeoutError`s. The
STEP-4 production relaunch was therefore independently necessary, not merely
procedural. Also note: raw-IP SSH to `.202` timed out during this check while
the `macstudio-m4-2` mDNS alias worked — the known Private-Wi-Fi-Address/stale-ARP
issue (`exo-cluster-network-flakiness` skill); all subsequent SSH used the aliases.

### 1. Revert — exactly three files, targeted checkout

```
git checkout -- src/exo/worker/engines/mlx/speculative/dsv4_mtp.py
git checkout -- start_cluster.sh
git -C mlx-lm checkout -- mlx_lm/models/deepseek_v4.py
```

No `git stash`, no `git checkout .`, no `git add -A`, no commits. Verified
beforehand that these were the *only* modified tracked files, so no other
agent's work was at risk.

```
=== git status (tracked only) ===
(empty = the 3 files clean)
=== submodule ===
(empty = clean)
=== tmp deliverables intact ===
      28
```

Laptop-side profiler symbol counts after revert — all zero:

```
mlx-lm/mlx_lm/models/deepseek_v4.py:0
src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:0
start_cluster.sh: 0
```

### 2. Relaunch on shipped production config

`env -u EXO_DSV4_COLL_PROFILE ./start_cluster.sh` (no profile var set):

```
Verifying commit consistency between nodes...
Nodes synchronized on commit 50ed3b295.
Starting Exo on macstudio-m4-1...
Starting Exo on macstudio-m4-2...
Waiting for cluster to stabilize...... HEALTHY! (Nodes: 2, Identities: 2)
Auto-placing DeepSeek V4 Flash (deepseek-ai/DeepSeek-V4-Flash-0731) across both Studios via RDMA...
Waiting for 2 DeepSeek V4 runner(s) to become Ready......................... READY (2/2)
```

### 3. Health evidence

**API responds:**
```
$ curl -s -m 10 http://192.168.86.201:52415/v1/models
{"object":"list","data":[{"id":"mlx-community/MiniMax-M2.7-4bit","object":"model",...
```

**Both nodes have a live runner:**
```
--- macstudio-m4-1 ---     --- macstudio-m4-2 ---
31240                      35647
31241                      35648
31242                      35649
31252                      35658
```

**Both READY in /state, model instance placed:**
```
runners: {
 "bb9deb10-4629-4535-b9fd-2cd570e9f9d9": { "RunnerReady": { "prefillServerPort": null } },
 "c505ce3c-3d25-47c3-ba94-abcee85ea741": { "RunnerReady": { "prefillServerPort": null } }
}
instances: ['1779c1d3-312d-4cf4-8cbf-1bbb205f4dbb']
```

**Real chat completion returns sane content** (venv python + urllib, no shell
pipe — `tmp/perf-campaign-2/round1/i1_health_probe.py`):
```
wall_s=3.14 ttft_s=1.74
delta_keys_seen=['content', 'reasoning_content', 'role']
finish_reason=stop
usage={'prompt_tokens': 17, 'completion_tokens': 50, 'total_tokens': 67, ...}
reasoning_len=164 content_len=31
--- reasoning head ---
1.  The user asks for the capital of France and specifies to answer in one short sentence.
2.  The capital of France is Paris.
3.  Compose a single, short sentence.
--- content ---
The capital of France is Paris.
VERDICT: SANE CONTENT RETURNED
```

**Installed site-packages on BOTH Studios no longer contain the profiler
symbol** (`grep -c` → 0 across all three files on both nodes):
```
--- macstudio-m4-1 ---     --- macstudio-m4-2 ---
0   (site-packages/mlx_lm/models/deepseek_v4.py: _COLL_PROFILE)
0   (src/exo/.../dsv4_mtp.py: EXO_DSV4_COLL_PROFILE)
0   (start_cluster.sh: EXO_DSV4_COLL_PROFILE)
```

**Production env confirmed on the live runners — profile var absent:**
```
--- macstudio-m4-1 ---        --- macstudio-m4-2 ---
EXO_DSV4_ATTN_ALLSUM=0        EXO_DSV4_ATTN_ALLSUM=0
EXO_DSV4_FENCE_ASYNC=1        EXO_DSV4_FENCE_ASYNC=1
```
(`EXO_DSV4_COLL_PROFILE` returns no match on either node.)

**CLUSTER IS HEALTHY ON SHIPPED PRODUCTION CONFIG.**

### Note on a harness bug found (not a cluster fault)

`tmp/perf-campaign-2/round1/i1_workload.py` crashes with
`TypeError: unsupported format string passed to NoneType.__format__` at line
204 when no `delta.content` arrives. Root cause: it only accumulates
`delta.content`, but DSv4-Flash is a reasoning model that emits
`reasoning_content` deltas first — at small `--max-tokens` the entire budget
can be consumed by reasoning, leaving `ttft_s = None`. This is a **harness
bug, not a serving failure**: the same request path returns
`finish_reason=stop` with correct content under a probe that reads both delta
keys. Left unfixed (out of scope, and the file is another agent's deliverable);
`i1_health_probe.py` is added alongside it and handles both keys.

---

## FILES

- `tmp/perf-campaign-2/round1/I1-COLLECTIVE-LATENCY.md` — this document
- `tmp/perf-campaign-2/round1/i1_health_probe.py` — NEW; stdlib post-restore
  health probe that reads both `content` and `reasoning_content`
- Reverted to HEAD: `mlx-lm/mlx_lm/models/deepseek_v4.py`,
  `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py`, `start_cluster.sh`
- No commits made (the PM commits).
