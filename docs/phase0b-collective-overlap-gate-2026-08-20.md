# PHASE 0b GATE — can GPU compute overlap an in-flight `mx.distributed.all_sum`?

Date: 2026-08-20
Hardware: LAPTOP MacBook Pro M4 Max. 2 ranks via
`.venv/bin/mlx.launch -n 2 --backend ring .venv/bin/python <probe>` (loopback,
**both ranks share one GPU**).
exo `fb426f395`, mlx submodule `ac73d0c9` (mlx 0.32.0.dev20260804).

## VERDICT: **GATE PASSES — with a mandatory implementation constraint.**

There is **no device-wide drain**. A ring-backend `all_sum` runs on a **CPU
stream** (`RingGroup::communication_stream` → `to_stream(s, Device::cpu)`;
`AllReduce` has no `eval_gpu` on Metal at all — it throws). Independent GPU
work overlaps it essentially perfectly.

**BUT**: overlap is destroyed if the collective's input is produced by a GPU op
issued on the *same* GPU stream as the compute you want to overlap. That is not
a device drain — it is single-stream FIFO ordering, and it is escapable by
issuing the overlapping compute on a **second GPU stream**.

## Evidence

### Probe 1 — pre-evaluated collective input (`bench/phase0b_collective_overlap_probe.py`)

| metric | rank 0 | rank 1 |
|---|---|---|
| COMPUTE_ONLY (2048² matmul ×24) | 77.75 ms | 77.51 ms |
| COMM_ONLY (64 MB all_sum) | 36.93 ms | 36.87 ms |
| BOTH | 77.34 ms | 77.56 ms |
| **overlap_ratio (BOTH / max)** | **0.995** | **1.001** |
| serial_ratio (BOTH / sum) | 0.674 | 0.678 |

Perfect overlap. 37 ms of a 115 ms serial budget recovered (~33%).

### Probe 3 — interleaved A/B, one process, isolating the single variable (`..._probe3.py`)

`PREEVAL` = `all_sum(src)` with `src` already eval'd.
`GPUPROD` = `all_sum(src * 1.0)` — a GPU elementwise feeds the collective.
`NULLDEP` = `all_sum(src)` **plus** the same `src * 1.0` eval'd in the same
graph but *not* feeding the collective (controls for the elementwise's own cost).

rank 0, 64 MB payload, compute = 103.4 ms:

| arm | comm | BOTH | b/max | b/sum |
|---|---|---|---|---|
| PREEVAL | 43.9 | 101.1 | **0.977** | 0.686 |
| GPUPROD | 41.1 | 146.4 | 1.416 | **1.014** |
| NULLDEP | 44.2 | 98.2 | **0.949** | 0.665 |

16 MB shows the same shape (PREEVAL 0.999, NULLDEP 1.004, GPUPROD 1.211).
Both ranks agree. `NULLDEP` is the key control: running the identical
elementwise op costs nothing — only *feeding it into the collective* serializes.

### Probe 4 — is it a drain, or stream FIFO? (`..._probe4.py`)

Same `GPUPROD` shape, but the matmul chain issued on `mx.new_stream(mx.gpu)`:

| MB | arm | comm | compute | BOTH | b/max | b/sum |
|---|---|---|---|---|---|---|
| 16 | GPUPROD_SAME | 16.6 | 77.8 | 93.7 | 1.204 | 0.993 |
| 16 | GPUPROD_ALT | 16.6 | 77.8 | **78.0** | **1.002** | 0.827 |
| 64 | GPUPROD_SAME | 34.7 | 80.5 | 114.8 | 1.427 | 0.997 |
| 64 | GPUPROD_ALT | 34.7 | 80.5 | **81.1** | **1.008** | 0.704 |

A second GPU stream fully restores overlap. **Decisive**: a device-wide drain
could not be escaped by changing streams.

## Mechanism (code-grounded)

- `mlx/distributed/ops.cpp:34` — every collective is built on
  `detail::communication_stream(group, s)`.
- `mlx/distributed/ring/ring.cpp:465` — ring returns `to_stream(s, Device::cpu)`.
  (jaccl, `jaccl.cpp:88-95`, pins its own single CPU stream — same class.)
- `mlx/backend/metal/distributed.cpp:17` — `AllReduce::eval_gpu` **throws**;
  there is no Metal collective. Only `eval_cpu` exists.
- `mlx/transforms.cpp:264-307` — when a producer and consumer sit on different
  streams, `eval` inserts a `Fence`. For a GPU-produced collective input, the
  GPU-side `Fence::update` (`backend/metal/fence.cpp:127-158`) is encoded into
  the **same GPU command encoder** as everything else on that stream, behind an
  `input_coherent` kernel + `compute_encoder.barrier()`. The CPU collective
  therefore cannot start until that stream's already-queued matmuls retire.

That is ordinary FIFO, not a device barrier. Note also `Fence::wait`'s GPU
`fence_wait` kernel (`kernels/fence.metal:40`) spins on-GPU — if a *GPU*
consumer waits on a CPU collective it will burn a GPU slot, another reason to
keep the waiting consumer off the stream doing the overlapping work.

## Implementation constraint for downstream phases

To overlap compute with an in-flight `all_sum`:

1. Either **pre-evaluate the collective's input** before issuing overlapping
   compute, **or**
2. issue the overlapping compute on a **dedicated `mx.new_stream(mx.gpu)`**.

Doing neither yields ~0% overlap (b/sum ≈ 1.00) and looks exactly like a drain.
This is very likely why prior sessions concluded collectives "block everything".

## NOT verified

- Only the **ring** backend was measured. jaccl/RDMA (the real cluster
  transport) uses a pinned CPU stream of the same class, so the conclusion
  should carry — but it is **not measured here**.
- Both ranks shared ONE GPU (loopback). GPU-contention effects on a real
  2-node Mac Studio cluster are untested; the `compute_only` baseline drifted
  77→103 ms between runs, consistent with laptop thermal/shared-GPU noise.
  Ratios were stable regardless.
- No real model/MoE workload was run. Payloads 1–64 MB, synthetic matmul chain.
- `nix fmt` not run (nix unavailable on this machine); ruff clean.
