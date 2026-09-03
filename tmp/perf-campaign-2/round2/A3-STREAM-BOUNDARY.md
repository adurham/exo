# A3 — Stream-boundary mechanism in the MLX Metal backend, and the 2.66x figure

Repo: `/Users/adam.durham/repos/exo/mlx` @ `e40a416b2` (submodule, deployed build).

## Direct answer

A stream boundary in MLX's Metal backend does **not** trip any max-ops-per-buffer
threshold — that threshold (`needs_commit()`, ~40-50 ops or ~40-50 MB, per-arch)
is a batching limit *within* a stream, unrelated to cross-stream transitions.
The real forced-commit mechanism is structural and happens on **every top-level
`eval()`/`async_eval()` call, synchronous or not**: `eval_impl()` walks its tape,
and after the loop it unconditionally calls `gpu::finalize(s)` on every stream
that was touched (`open_streams`), which does `end_encoding()` + `commit()` on
that stream's live command buffer (transforms.cpp:325-332). So if production
code issues one `mx.eval()`/`mx.async_eval()` per layer (as the Phase-H Lever-1
fence shape at `deepseek_v4.py:2836-2894` referenced in round-1 material does),
**both** the GPU stream's and the CPU/communication stream's command buffers get
committed every layer regardless of whether a MAX_OPS threshold was ever
reached — this is what "crossing a stream boundary" actually costs, not encoder
churn. Separately, when a *data dependency* crosses streams mid-tape (not at
eval-call granularity), `transforms.cpp:159-168` records a `needs_fence` entry
tagged `device_switch=true` when `a.stream().device != in.stream().device`;
`Fence::update` (`metal/fence.cpp:100-157`) on a GPU→CPU transition additionally
launches an `input_coherent` kernel over the **whole payload** before signaling,
and `Fence::wait` (`metal/fence.cpp:50-98`) on the CPU-stream side spin/sleep-polls
until that fence value appears. Root cause per source-code structure: crossing
streams forces (a) a real payload-sized coherency kernel pass and (b) an extra
command-buffer commit+drain that same-stream ops never pay, because same-stream
ops share one open encoder/buffer until the ops-count or MB threshold fires,
while cross-stream ops always pay the fence + (in the eval-per-layer pattern)
the eval-boundary finalize on top of that. This is corroborated independently
by `docs/phase0a-allsum-boundary-decomposition-2026-08-20.md`, whose own
measured boundary cost is **linear in payload bytes (~7 GB/s)**, i.e. dominated
by the coherency kernel + memory traffic, not by a fixed per-crossing commit
latency — the source gives no fixed-cost constant for a bare commit, so the
"86 commits x fixed cost" framing in the task brief is not directly supported;
the payload-proportional framing is.

## Evidence

1. `mlx/backend/metal/device.cpp:662-665` — `CommandEncoder::needs_commit()`: `(buffer_ops_ > max_ops) || ((buffer_sizes_ >> 20) > max_mb)`. This is the *only* place a threshold triggers a commit inside `gpu::eval()`. It is an intra-stream batching cap, not a stream-boundary trigger.
2. `mlx/backend/metal/device.cpp:757-781` — arch-dependent constants: `max_ops_per_buffer_` = 20 (phone) / 40 (base/pro) / 50 (max/ultra), `max_mb_per_buffer_` = 40 or 50; both overridable via `env::max_ops_per_buffer`/`env::max_mb_per_buffer` (i.e. `MLX_MAX_OPS_PER_BUFFER`/`MLX_MAX_MB_PER_BUFFER`-style env vars). On an M4 Max ("max"/'s') this is 50 ops / 50 MB.
3. `mlx/backend/metal/eval.cpp:119-175` — `gpu::eval(array&)`: gets `get_command_encoder(s)`, appends the op via `arr.primitive().eval_gpu(...)`, and only calls `encoder.end_encoding()` + `encoder.commit()` **if `encoder.needs_commit()`** (line 152); otherwise it just registers a completion handler and keeps appending to the same open encoder/buffer (line 162-174, explicit comment about avoiding double-counting GPU time on the *same* uncommitted buffer). This is the "cheap, reused encoder" path for same-stream ops.
4. `mlx/backend/metal/eval.cpp:177-187` — `gpu::finalize(Stream s)`: unconditionally does `encoder.end_encoding()` + `encoder.commit()` on that stream's buffer, no threshold check.
5. `mlx/transforms.cpp:242-332` — `eval_impl`'s main loop: each `array` in the tape is dispatched via `gpu::eval(arr)` (line 279-280) onto its own stream, streams touched are collected in `open_streams` (line 248). **After the tape finishes**, lines 325-332 iterate `open_streams` and call `gpu::finalize(s)` on every GPU stream that was used — i.e. every eval()/async_eval() call forces a commit+end_encoding on *all* streams it touched, independent of `needs_commit()`/thresholds. `eval()` additionally blocks via `.wait()` (transforms.cpp:368); `async_eval()` (line 337-349) does not block but still runs the identical `eval_impl` with the same per-call finalize loop — so the forced commit-per-stream-per-call happens in both sync and async modes.
6. `mlx/transforms.cpp:159-168` — cross-stream data-dependency detection: when a consumer array's stream differs from an input's stream, `needs_fence` records `(stream_index, device_switch)` where `device_switch = (a.stream().device != in.stream().device)`. This is the exact code path a GPU-stream→CPU/communication-stream (or reverse) data dependency hits.
7. `mlx/transforms.cpp:263-277` — the fence is actually applied per node: if an input needs a fence, `fences[...].wait(stream, in)` is called on the *consuming* stream before dispatch; this is separate from (and in addition to) the eval-boundary finalize in item 5.
8. `mlx/backend/metal/fence.cpp:100-157` — `Fence::update`: on `cross_device=true` (i.e. GPU→CPU transition), launches an `input_coherent` kernel sized to `x.data_size() * x.itemsize()` (line 131-140) — a full pass over the payload — before the barrier + `fence_update` kernel. This is a *payload-proportional* cost, not fixed.
9. `mlx/backend/metal/fence.cpp:50-98` — `Fence::wait`: on a CPU-stream consumer with the default (non-fast) path, calls `f.event->wait(stream)`, which (per `event.cpp`, not separately re-cited here) sleep/spin-polls a shared value — this is the consumer-side stall that makes the boundary visible as wall-clock cost.
10. `mlx/backend/metal/device.cpp:575-644` — `CommandEncoder::end_encoding()`: per-encoder fence bookkeeping (`waitForFence`/`updateFence`) that lets *same-stream* ops chain cheaply without a full commit; this machinery is what gets bypassed/reset at every commit, i.e. it is the concrete implementation of "ops reused within one stream are cheap."
11. `docs/phase0a-allsum-boundary-decomposition-2026-08-20.md:44-52` — direct source citations backing this same trace, written independently in round 1: `AllReduce::eval_gpu` throws (collectives are CPU-stream `eval_cpu`-only), `needs_fence`/`device_switch` in transforms.cpp:159-164, and `Fence::update`'s `input_coherent` kernel over the whole payload at fence.cpp:129-140 (matches item 8's current line numbers, ~100-141, after later edits).
12. `docs/phase0a-allsum-boundary-decomposition-2026-08-20.md:55-65` — **the actual measurement**: single-rank, 12 chained layers, production payload (16.8 MB bf16), `layered_gpuop = 23.755 ms` total vs `layered_cpuop = 63.232 ms` total → `63.232/23.755 = 2.66x`. This is a **whole-loop total-time ratio over 12 layers**, not a per-op ratio and not derived by averaging 12 separate per-layer ratios; it reduces to the same ratio per layer only if the per-layer boundary cost is uniform, which RESULT 2 (below) shows it mostly is at fixed payload.
13. `docs/phase0a-allsum-boundary-decomposition-2026-08-20.md:66-100` (RESULT 2) — sweeping payload 1.05 MB → 67.11 MB shows the boundary term scaling **linearly, ~7 GB/s** (`0.242 → 0.563 → 2.447 → 8.891` ms as payload goes 1.05→4.19→16.78→67.11 MB — 64x payload gives ~37x boundary cost). This is the load-bearing fact against the brief's "fixed per-commit cost" framing: the source shows no fixed per-crossing commit-cost constant, and the *empirical* boundary cost measured in round 1 is explicitly payload-proportional, not flat.
14. `docs/phase0a-allsum-boundary-decomposition-2026-08-20.md:82-90` (RESULT 3) — sweeping `MLX_EVENT_WAIT_POLL_US` 1→200 µs changes per-layer boundary by only ~3% (3.230→3.341 ms), ruling out poll-loop granularity (i.e. ruling out "commit overhead" in the sense of scheduling/poll latency) as the driver — consistent with the mechanism being the payload coherency kernel (item 8), not a fixed submission fee.

## Cost-model quantification (structural only)

- The source (`device.cpp`, `eval.cpp`, `transforms.cpp`, `fence.cpp`) contains **no comment or constant giving a microsecond/fixed cost for a command-buffer commit or submission**. `needs_commit()`'s thresholds (item 2) bound *how many ops* can share a buffer, not how expensive a commit is.
- What the source *does* establish structurally: (a) same-stream ops share one open encoder/command buffer until an ops/MB cap fires (cheap, amortized — item 3), and (b) every stream that is touched inside a given `eval()`/`async_eval()` call is force-committed at the end of that call regardless of the cap (item 5), and (c) a cross-device data dependency additionally pays a payload-sized coherency kernel (item 8) plus a spin/poll wait (item 9) on top of whatever commit already happens. If production wraps each of the 43 layers in its own `mx.eval()`/`mx.async_eval()` around the layer's `all_sum` (matching the round-1-cited `deepseek_v4.py:2836-2894` fence shape), then **43 layers x 2 streams touched = up to 86 forced end_encoding+commit pairs per forward**, purely from mechanism (b), independent of whether any op count/MB threshold was ever reached. That is a real, source-grounded multiplication of commit events — but its magnitude cannot be quantified as "86 x fixed_cost" because no fixed cost exists in source; the round-1 measurement (item 12-13) instead shows the dominant term is bytes-proportional (~7 GB/s), so the correct structural framing is "86 crossings x per-crossing payload-coherency cost," not "86 crossings x fixed submission fee."
- Net: the **asymmetry the brief asks about is real** — same-stream ops are cheap/batched (mechanism a), cross-stream ops are not (mechanisms b+c) — but the round-1 data itself already falsifies the *fixed-cost* sub-hypothesis in favor of a *payload-proportional* one (item 13, and explicitly RESULT 3's poll-granularity null in item 14 rules out fixed scheduling/submission latency as the driver).

## Verification of the round-1 2.66x figure

- **What was measured**: `bench/phase0a_allsum_boundary_decompose.py` (not present in this repo checkout under `tmp/perf-campaign-2/round1/` — round 1's files there, e.g. `i3_microbench_chained.py`, are a *different*, later, unrelated MoE-kernel-bandwidth microbenchmark and do **not** implement the no-peer control; `grep -ri "no-peer\|no_peer\|2.66\|stream boundary"` on `round1/*.py` and `round1/*.md` returns no matches — those files only echo/cite the 2.66x number, e.g. `REPORT.md:169` and `I1-COLLECTIVE-LATENCY.md:71/112/162`, they do not implement or re-derive it). The actual harness and number live in `docs/phase0a-allsum-boundary-decomposition-2026-08-20.md`, a doc round 1's `I1-COLLECTIVE-LATENCY.md` cites as prior-record source "R2."
- **Operation measured**: a plain non-collective op (`mx.abs`, per the doc's summary line "a plain `mx.abs` on a CPU stream reproduces 2.66x") run 12 times in a chained loop, each iteration doing `GPU chain -> op -> blocking mx.eval()`, at production payload shape (2048x4096 bf16, 16.8 MB), single rank, no peer.
- **Streams**: `layered_gpuop` = the op placed on the GPU compute stream (no boundary); `layered_cpuop` = the identical op placed on the CPU stream (the same stream MLX pins all collectives to, since `AllReduce::eval_gpu` throws — collectives are `eval_cpu`-only).
- **Ratio type**: **(ii) a whole-loop ratio** — `layered_cpuop` (63.232 ms total over 12 layers) / `layered_gpuop` (23.755 ms total over 12 layers) = 2.66x (doc line ~62-65). It is reported as one number over the full 12-iteration chain, not averaged per-op from 12 individual ratios, though the doc's own payload sweep (RESULT 2) shows the underlying per-layer boundary cost is consistent across repeats at fixed payload — so treating it as an implied per-layer ratio is reasonable but not literally how it was computed.
- **Methodology check — no `mx.eval`-in-timing-loop artifact found**: the doc explicitly uses `mx.eval()` *inside* each layered iteration by design (`"GPU chain -> op -> blocking mx.eval cycles"`), but this is not the same artifact class as the campaign's known bad measurement (charging host-dispatch overhead to a kernel by timing individual back-to-back `mx.eval()` calls in a tight loop and misattributing per-call fixed overhead as per-op compute). Here the eval is the intentional force-point that drains the collective/CPU-stream machinery under test — timing the whole 12-iteration loop wall-clock, not summing N per-call deltas — and the doc separately validates the boundary is payload-proportional (RESULT 2) rather than reporting a suspiciously round fixed number, which is the shape the retracted 172µs/call artifact had. `i3_microbench_chained.py`'s own header (lines 1-24) independently documents awareness of that artifact class ("serial-sync control ... confirm the chained number is faster per call, and that this reproduces roughly the same ~172us/call gap the retraction doc found") but that file measures MoE kernel bandwidth, not the stream boundary — it is unrelated to the 2.66x figure and should not be conflated with it.
- **Consistency with source-level mechanism**: yes. The measured op is CPU-stream-only, matching `AllReduce::eval_gpu`'s throw (collectives forced onto the CPU stream) — so the no-peer control is a faithful proxy for the collective's boundary cost. The doc's own RESULT 2 (payload-linear, ~7 GB/s) and RESULT 3 (poll-granularity null) are *consistent with, and in fact sharpen*, the source-derived mechanism in this deliverable: the cost is dominated by `Fence::update`'s `input_coherent` kernel pass over the payload (item 8) plus consumer-side event wait (item 9), not by command-buffer submission overhead per se. I found no methodological red flag (no mx.eval-in-a-hot-timing-loop misattribution) in the phase0a doc as summarized; I was not able to read the underlying `bench/phase0a_allsum_boundary_decompose.py` script itself (not present in this checkout) to verify the timing code line-by-line, which is a real limitation — see below.

## CONFIDENCE

High on the source-code mechanism (items 1-11, all directly read from the deployed `e40a416b2` submodule). Medium-high on the verification: I could not locate or read `bench/phase0a_allsum_boundary_decompose.py` itself (not present under `round1/` or elsewhere I searched in this repo) — my methodology assessment rests on the doc's prose description and self-reported results, not the harness source, so I cannot personally rule out a timing bug in that script the way I could for a script I could read directly.

## What I could not determine from source

- The exact production call pattern (does every layer really wrap its `all_sum` in its own blocking `mx.eval()`/`mx.async_eval()`, triggering the `eval_impl` open_streams finalize loop 43 times, or is there broader batching across layers?) — I did not read `deepseek_v4.py` in this task; I'm relying on round-1's citation of `deepseek_v4.py:2836-2894` as "the Phase-H Lever 1 fence shape."
- No fixed microsecond cost for a bare Metal command-buffer commit/submission exists anywhere in the C++ source I read; I did not invent one, per instructions.
- Could not read `bench/phase0a_allsum_boundary_decompose.py` (the actual round-1 no-peer-control harness) — it is not present in this checkout, only the resulting doc is. This means the "no mx.eval-in-timing-loop artifact" verdict above is based on the doc's description of its own method, not a line-by-line code read.
- Did not independently re-derive or re-run any benchmark (task is read-only; no benchmarks were executed).
