#!/Users/adam.durham/repos/exo/.venv/bin/python3
#!/usr/bin/env python3
"""PHASE 0a: decompose ``moe.all_sum``'s real per-call cost into components.

Question
--------
``docs/moe-all-sum-178ms-artifact-real-bottleneck-2026-08-20.md`` established
that the collective's OWN wall time is ~5-12 ms/call at the production
16.8 MB payload, and attributed the remaining (NOP-ablation-implied) cost to
"collateral serialization the collective imposes on the surrounding graph".
That attribution was source-grounded but **never isolated by measurement**.

This probe isolates it, using a control the repo did not previously have:

    a NON-COLLECTIVE op pinned to a CPU stream, at the identical shape.

Such an op has ZERO wire traffic and ZERO cross-rank lockstep, but pays the
EXACT same MLX machinery as ``all_sum``:

  * ``mlx/backend/metal/distributed.cpp:17`` -- ``AllReduce::eval_gpu`` is a
    hard ``throw``; every MLX collective is ``eval_cpu``-only.
  * ``mlx/distributed/jaccl/jaccl.cpp:88`` -- ``JACCLGroup`` pins every
    collective to one owned ``new_stream(Device::cpu)``, ignoring the
    caller's stream.
  * ``mlx/transforms.cpp:159-164`` -- when a consumer's stream differs from
    its input's stream, MLX records ``needs_fence`` with
    ``device_switch = (a.stream().device != in.stream().device)``.
  * ``mlx/transforms.cpp:272-290`` / ``mlx/backend/metal/fence.cpp`` --
    a CPU-stream consumer of a GPU-produced array does
    ``scheduler::enqueue(cpu_stream, [] { ...spin/sleep-poll until the GPU
    signals... })``.  On the non-fast path that is
    ``EventImpl::wait`` (``mlx/backend/metal/event.cpp:56``): 2000 yield
    spins then ``sleep_for(MLX_EVENT_WAIT_POLL_US)`` (default **50 us**).
  * ``fence.cpp:129`` -- a cross-device ``Fence::update`` additionally
    launches the ``input_coherent`` kernel over the WHOLE payload.

So: **anything** on a CPU stream downstream of GPU work pays the drain; the
collective is merely one instance of it.  Differencing the two isolates
wire+collective from boundary.

Conditions (all at the production shape (1, 2048, 4096) bf16 = 16.8 MB)
------------------------------------------------------------------------
  gpu_compute      GPU matmul chain -> eval                 (t_A)
  cpuop_ready      CPU-stream op on a PRE-EVALUATED input   (t_F)
  gpu_then_cpuop   GPU chain -> CPU-stream op, one graph    (t_C)
  allsum_ready     all_sum on a PRE-EVALUATED input         (t_R)
  gpu_then_allsum  GPU chain -> all_sum, one graph          (t_D)
  layered_*        the real shape: N layers of chain->op    (accumulation)

Derived
-------
  boundary       = t_C - t_A - t_F      GPU->CPU->GPU drain/refill, NO wire
  wire_plus_coll = t_R - t_F            the collective's own marginal cost
  induced        = t_D - t_A - t_R      serialization the collective imposes

If ``boundary`` is large and ~equal to ``induced``, the doc's redirect is
CONFIRMED by measurement: the lever is the stream boundary, not the wire.
If ``boundary`` ~ 0 while ``induced`` is large, the cost is collective-
specific (cross-rank lockstep) and the redirect is WRONG.

Run
---
  # single rank -- boundary terms only, all_sum degenerates to identity
  .venv/bin/python bench/phase0a_allsum_boundary_decompose.py

  # two ranks -- adds the real collective terms
  .venv/bin/mlx.launch -n 2 --backend ring \
      bench/phase0a_allsum_boundary_decompose.py

  # poll-granularity sweep (env is read ONCE per process, so re-exec)
  for p in 1 10 50 200; do MLX_EVENT_WAIT_POLL_US=$p \
      .venv/bin/python bench/phase0a_allsum_boundary_decompose.py --tag poll$p
  done

Laptop note: this box is NOT a dedicated test rig.  Absolute ms are noisy;
every number is a MEDIAN over --reps with --warmup discarded, and the
verdict is driven by RATIOS between conditions measured back-to-back in the
same process, not by absolute latency.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from typing import Callable

import mlx.core as mx

# Production payload: EXO_PREFILL_STEP_SIZE=2048 x hidden 4096 x bf16.
DEFAULT_TOKENS = 2048
DEFAULT_HIDDEN = 4096
DEFAULT_LAYERS = 43  # DSv4-Flash layer count -> all_sums per prefill chunk


def _median_ms(fn: Callable[[], None], reps: int, warmup: int) -> tuple[float, float]:
    """Return (median_ms, iqr_ms) over ``reps`` timed calls."""
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    med = statistics.median(samples)
    if len(samples) >= 4:
        q1 = samples[len(samples) // 4]
        q3 = samples[(3 * len(samples)) // 4]
        iqr = q3 - q1
    else:
        iqr = samples[-1] - samples[0]
    return med, iqr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    ap.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    ap.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    ap.add_argument("--matmul-dim", type=int, default=1024)
    ap.add_argument("--matmul-depth", type=int, default=8)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--json-out", type=str, default="")
    args = ap.parse_args()

    group = mx.distributed.init()
    rank, size = group.rank(), group.size()

    payload_bytes = args.tokens * args.hidden * 2
    cpu_stream = mx.new_stream(mx.cpu)
    gpu_stream = mx.default_stream(mx.gpu)

    # --- inputs -----------------------------------------------------------
    # A GPU matmul chain with a serial dependency, so it cannot be reordered
    # or elided, feeding a broadcast into the production payload shape.
    mm = mx.random.normal((args.matmul_dim, args.matmul_dim)).astype(mx.bfloat16)
    mm = mm / (args.matmul_dim**0.5)
    ready = mx.zeros((1, args.tokens, args.hidden), dtype=mx.bfloat16) + 1.0
    mx.eval(mm, ready)
    mx.synchronize()

    def gpu_chain() -> mx.array:
        """GPU work whose output IS the 16.8 MB payload (like the MoE tail)."""
        with mx.stream(gpu_stream):
            a = mm
            for _ in range(args.matmul_depth):
                a = (a @ mm) * 1.0001
            # Fold the chain into the payload so the payload genuinely
            # DEPENDS on GPU work (this is what forces the fence).
            scale = a[0, 0].astype(mx.bfloat16)
            return ready * (1.0 + scale * 0.0)

    def cpu_op(x: mx.array) -> mx.array:
        """Non-collective CPU-stream op: same shape, same stream device.

        ``mx.abs`` is elementwise, cheap, and NOT elidable on bf16.
        """
        with mx.stream(cpu_stream):
            return mx.abs(x)

    def gpu_op(x: mx.array) -> mx.array:
        with mx.stream(gpu_stream):
            return mx.abs(x)

    def all_sum(x: mx.array) -> mx.array:
        return mx.distributed.all_sum(x, group=group)

    # --- conditions -------------------------------------------------------
    results: dict[str, tuple[float, float]] = {}

    def bench(name: str, fn: Callable[[], None]) -> None:
        results[name] = _median_ms(fn, args.reps, args.warmup)

    bench("gpu_compute", lambda: mx.eval(gpu_chain()))
    bench("gpuop_ready", lambda: mx.eval(gpu_op(ready)))
    bench("cpuop_ready", lambda: mx.eval(cpu_op(ready)))
    bench("allsum_ready", lambda: mx.eval(all_sum(ready)))
    bench("gpu_then_gpuop", lambda: mx.eval(gpu_op(gpu_chain())))
    bench("gpu_then_cpuop", lambda: mx.eval(cpu_op(gpu_chain())))
    bench("gpu_then_allsum", lambda: mx.eval(all_sum(gpu_chain())))

    # Layered: the REAL per-chunk shape -- N sequential (GPU chain -> op ->
    # blocking eval) cycles, exactly matching the Phase-H Lever 1 fence at
    # deepseek_v4.py:2836-2894. This is where drain/refill ACCUMULATES.
    def layered(op: Callable[[mx.array], mx.array] | None) -> None:
        for _ in range(args.layers):
            y = gpu_chain()
            if op is not None:
                y = op(y)
            mx.eval(y)

    bench("layered_none", lambda: layered(None))
    bench("layered_gpuop", lambda: layered(gpu_op))
    bench("layered_cpuop", lambda: layered(cpu_op))
    bench("layered_allsum", lambda: layered(all_sum))

    if rank != 0:
        return

    # --- derivation -------------------------------------------------------
    def m(k: str) -> float:
        return results[k][0]

    boundary = m("gpu_then_cpuop") - m("gpu_compute") - m("cpuop_ready")
    wire_plus_coll = m("allsum_ready") - m("cpuop_ready")
    induced = m("gpu_then_allsum") - m("gpu_compute") - m("allsum_ready")
    gpu_side_control = m("gpu_then_gpuop") - m("gpu_compute") - m("gpuop_ready")

    per_layer_boundary = (m("layered_cpuop") - m("layered_gpuop")) / args.layers
    per_layer_allsum = (m("layered_allsum") - m("layered_gpuop")) / args.layers

    poll_us = os.environ.get("MLX_EVENT_WAIT_POLL_US", "50 (default)")
    fast_synch = os.environ.get("MLX_METAL_FAST_SYNCH", "0 (default)")

    print()
    print("=" * 78)
    print(f"PHASE 0a  moe.all_sum boundary decomposition   tag={args.tag or '-'}")
    print(f"  world_size={size}  payload={payload_bytes / 1e6:.2f} MB "
          f"({args.tokens}x{args.hidden} bf16)  layers={args.layers}")
    print(f"  MLX_EVENT_WAIT_POLL_US={poll_us}  MLX_METAL_FAST_SYNCH={fast_synch}")
    print(f"  reps={args.reps} warmup={args.warmup}  (medians; laptop = noisy)")
    print("=" * 78)
    print(f"  {'condition':<22s} {'median_ms':>11s} {'iqr_ms':>9s}")
    for k, (med, iqr) in results.items():
        print(f"  {k:<22s} {med:>11.3f} {iqr:>9.3f}")

    print()
    print("  DERIVED (single-graph, one op)")
    print(f"    boundary  GPU->CPU->GPU drain, NO wire   = {boundary:>9.3f} ms")
    print(f"    wire+coll collective's own marginal cost = {wire_plus_coll:>9.3f} ms")
    print(f"    induced   serialization from all_sum     = {induced:>9.3f} ms")
    print(f"    [control] same-device (GPU->GPU) overhead= {gpu_side_control:>9.3f} ms")
    print()
    print(f"  DERIVED (layered x{args.layers}, the real per-chunk shape)")
    print(f"    per-layer boundary (cpuop - gpuop)       = {per_layer_boundary:>9.3f} ms")
    print(f"    per-layer all_sum  (allsum - gpuop)      = {per_layer_allsum:>9.3f} ms")
    if per_layer_allsum > 0:
        share = 100.0 * per_layer_boundary / per_layer_allsum
        print(f"    boundary share of all_sum's per-call cost= {share:>8.1f} %")

    print()
    print("  VERDICT")
    if size < 2:
        print("    size=1: all_sum is an IDENTITY (ops.cpp:30 early-return), so")
        print("    the allsum_* rows measure NO collective. Only the boundary")
        print("    terms are meaningful here. Re-run under mlx.launch -n 2.")
    if boundary > 3.0 * max(gpu_side_control, 0.05):
        print("    CONFIRMED: crossing to a CPU stream costs far more than the")
        print("    same op on the GPU stream, at zero wire bytes. The drain is")
        print("    real and is NOT collective-specific.")
    else:
        print("    NOT CONFIRMED: the CPU-stream crossing is not measurably")
        print("    more expensive than the same-device control.")
    print("=" * 78)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(
                {
                    "tag": args.tag,
                    "world_size": size,
                    "payload_bytes": payload_bytes,
                    "layers": args.layers,
                    "poll_us": poll_us,
                    "conditions": {k: {"median_ms": v[0], "iqr_ms": v[1]}
                                   for k, v in results.items()},
                    "derived": {
                        "boundary_ms": boundary,
                        "wire_plus_collective_ms": wire_plus_coll,
                        "induced_ms": induced,
                        "gpu_side_control_ms": gpu_side_control,
                        "per_layer_boundary_ms": per_layer_boundary,
                        "per_layer_allsum_ms": per_layer_allsum,
                    },
                },
                f,
                indent=2,
            )
        print(f"  wrote {args.json_out}")


if __name__ == "__main__":
    main()
