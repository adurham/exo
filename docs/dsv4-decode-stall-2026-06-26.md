# DSv4 Decode Stall — Findings & Failed Optimization Attempts

**Date:** 2026-06-26
**Status:** Decode-stall root cause IDENTIFIED but not profitably actionable
via read-then-implement approaches. Decode baseline **37.2 t/s aggregate
at c=2** — already beats the 30 t/s target by 24%. Two structural
optimization attempts A/B'd and reverted with documented reasons.
This doc records the findings and the traps so the next attempt (by
someone with deeper MLX stream/fence understanding, or a fresh session)
doesn't re-tread them.

---

## TL;DR

- Decode is **83% MoE** (`moe.switch_mlp` alone is 73%). Attention is 17%.
- The 73% `switch_mlp` time is **>99% GPU idle/stall, not compute** — the
  three `gather_qmm` matmuls take ~2.3ms each on the GPU; the whole MoE
  compute is ~7ms, but the `switch_mlp` span envelope measures ~2935ms.
  The GPU sits idle during the cross-rank `all_sum` collective and
  between async kernels.
- **Two attempts to capture the stall headroom failed:**
  1. MoE gate+up fusion (`FusedSwitchGLU`, 3→2 dispatches): clean A/B,
     **−3.8% slower** (37.2 → 35.8 t/s). Fusion doesn't reduce compute
     (already ~7ms) and the concatenated buffer worsened the stall pattern.
  2. all_sum / shared_experts stream-overlap: **broke quality hard**
     (B=2 200K needle `all_needles=False`, near-zero output). The fence
     position is load-bearing for cross-rank bit-equiv; reordering it
     (even keeping a fence) let the next layer's lazy graph build against
     an unresolved collective.
- A prior attempt (OPT-7, gating the fence on `_fence_every_n`) was
  tested and reverted in 2026-06-18 for **−23% prefill** (graph-
  accumulation cost). Same trap class.
- **Three data points say this neighborhood is a trap for read-then-
  implement approaches.** The stall is real; capturing it correctly
  requires understanding MLX's lazy-graph + comm/GPU-stream + fence
  interaction at a level reading alone doesn't give.

---

## The profile (real GPU time, sync spans)

`EXO_PROFILER=spans EXO_PROFILER_SYNC_SPANS=1`, c=2, 256-token decode:

```
ffn                  84.4%   (moe.switch_mlp 73.3%, all_sum 7.5%, gate 1.3%, post_combine 1.9%)
attn                 ~15%    (o_proj 4.4%, proj_qkv 3.6%, all_sum 2.3%, indexer 1.3%, sdpa 0.9%, ...)

moe.switch_mlp envelope: 2935.64 ms  (73.3%)   ← the "73%" lump
  switch.up_proj:           2.44 ms  (0.1%)   ← real GPU compute
  switch.gate_proj:         2.32 ms  (0.1%)
  switch.down_proj:         2.27 ms  (0.1%)
  switch.activation:        2.43 ms  (0.1%)
  switch.gather_sort:       1.00 ms
  switch.scatter_unsort:    0.75 ms
  ───────────────────────────────
  sub-ops total:          ~11.2 ms
```

**The three `gather_qmm` matmuls are ~2.3ms each. The MoE compute is
~7ms. The envelope is 2935ms.** >99% is GPU idle.

### Why the non-sync profiler was misleading

`EXO_PROFILER=spans` *without* `SYNC_SPANS=1` measures Python-level wall
time (op queueing), not GPU execution — the sub-spans totaled ~1.6ms vs
2703ms envelope because the GPU kernels run async after the Python block
returns. **Must use `EXO_PROFILER_SYNC_SPANS=1` for real per-op GPU
timing** (it `mx.synchronize()`s before/after each span; slow, but
proportions are real).

---

## The MoE flow (deepseek_v4.py:1386-1474)

```
sum_gradients(x)          # comm stream (input collective)
gate(x, input_ids)        # GPU
switch_mlp(x, inds)       # GPU — 3 gather_qmm (~7ms compute)
post_combine:             # GPU
  shared_experts(x)       #   independent of all_sum(y) — reads x
  _moe_post_combine(y, scores, shared_out)
all_sum(y, group)         # comm stream (output collective)
mx.eval(y)                # FENCE — load-bearing for cross-rank bit-equiv
```

Key facts:
- `all_sum` already runs on MLX's `communication_stream` (ops.cpp:34
  `detail::communication_stream`) — separate from the GPU compute stream.
  The overlap *primitive* exists.
- The `mx.eval(y)` fence immediately after `all_sum` is **required** for
  cross-rank bit-equiv at c=2 temp=0 (comment at 1464-1473): without it,
  GPU stragglers let the two ranks dispatch the next MoE layer with
  subtly different inputs.
- `shared_experts(x)` is independent of `all_sum(y)` (reads `x`, not `y`).
  In principle overlap-able with the collective.

---

## Failed attempt 1: MoE gate+up fusion (`FusedSwitchGLU`)

**Idea:** `SwitchGLU` runs 3 `gather_qmm` (gate, up, down). mlx-lm's
`FusedSwitchGLU` / `_install_fused_gate_up` (auto_parallel.py:1107)
concatenates gate+up weights → 2 `gather_qmm`. Sharding-aware (accesses
`gp["weight"]` on the sharded wrapper). Qwen3.5 uses it.

**Implementation:** `_install_fused_gate_up(layer.ffn.switch_mlp)` at the
DSv4 sharding sites (auto_parallel.py:916-918 / 939-941), env-gated
`EXO_DSV4_MOE_FUSED_GATE_UP=1`.

**History note:** a DSv4-*specific* fusion (`FusedDeepseekV4SwitchGLU`)
was removed 2026-06-18 for B>1 batch-mis-specialization degeneration. The
generic `FusedSwitchGLU` (Qwen's path) does NOT have that bug — confirmed
quality-safe (B=2 200K needle passed with fusion on).

**A/B (clean, back-to-back, post-reboot, both ranks confirmed
`FusedSwitchGLU`, 5 iters each):**
```
baseline (fusion off):  37.2 t/s  (med 37.0, 36.7–38.0)
fused (fusion on):      35.8 t/s  (med 35.0, 34.8–37.8)
                        −3.8%
```
**Result: ~4% slower.** Fusion doesn't reduce compute (the matmuls are
already ~2.3ms — tiny) and the concatenated `(num_experts, 2*out_local,
in_packed)` buffer + the `gu[...,:n]`/`gu[...,n:]` split gave a worse
memory/SwiGLU read pattern that outweighed the 3→2 dispatch win.

**Conclusion:** dead end. Left gated off (`EXO_DSV4_MOE_FUSED_GATE_UP=0`).
Commits: mlx-lm (sharding sites wired), exo `7bdda395` / `9ccd59b73`.

---

## Failed attempt 2: all_sum / shared_experts stream-overlap

**Idea:** `all_sum(y)` (comm stream) and `shared_experts(x)` (GPU stream)
are independent. Queue both before the fence → GPU does the shared-expert
matmul concurrently with the cross-rank collective → turns the stall into
useful compute. Different from OPT-7 (which dropped the fence); this
keeps the fence, just reorders independent work before it.

**Implementation:** `EXO_DSV4_MOE_OVERLAP_SHARED=1` reorders
`DeepseekV4MoE.__call__` to queue `all_sum(y)` before `shared_experts(x)`
+ combine, fence after combine (mlx-lm `9bc2206`).

**A/B (B=2 200K needle, quality gate first):**
```
all_needles=False  agg=0.42 t/s  (both streams ~0.3/0.1 t/s, near-zero output)
runner UP (no crash, no DEGEN loop)
```
**Result: broke quality hard.** The fence I kept (after the combine) did
NOT preserve the cross-rank bit-equiv. Root cause: the original fence
(immediately after `all_sum`, BEFORE `shared_experts`/combine or any
downstream read) fences the *collective output* specifically. Moving the
fence to after the combine (which reads `y_reduced`) lets the next
layer's lazy graph build against an unresolved collective.

**Conclusion:** reverted (mlx-lm `bf95d1a`, exo `4c098c75`). Same trap
class as OPT-7 — fence *position* is load-bearing, MLX lazy + stream
split doesn't overlap the way the textbook model assumes.

---

## The trap (3 data points)

| Attempt | Approach | Result |
|---------|----------|--------|
| OPT-7 (2026-06-18) | gate `mx.eval` fence on `_fence_every_n` | −23% prefill (graph accumulation) → reverted |
| gate+up fusion | 3→2 `gather_qmm` dispatches | −3.8% decode (compute already tiny) → reverted |
| stream-overlap | reorder independent work before fence | broke quality (cross-rank equiv) → reverted |

All three hit the same wall: **MLX's lazy-graph + comm/GPU-stream +
fence interaction doesn't admit the textbook collective-overlap /
dispatch-reduction optimizations, and the fence position is load-bearing
for cross-rank bit-equiv in ways that aren't visible from reading.**

---

## What WOULD be needed (not attempted)

The decode-stall headroom is real (GPU idle >99% of the MoE envelope).
Capturing it correctly likely requires:
- Deep understanding of MLX's lazy graph scheduling across streams —
  when does an `mx.eval` on one stream drain another? Does
  `communication_stream` actually run concurrently with GPU compute on
  the JACCL/Thunderbolt backend, or is it serialized at the device level?
- Possibly MLX-level changes (explicit async streams, graph-reorder
  hints, a fence primitive that fences one stream without blocking the
  other) — not just exo/mlx-lm code.
- A correct overlap that fences the collective output *before any
  downstream read of `y`* while still letting independent GPU work run
  concurrently — the exact thing my attempt got wrong.

This is research-grade work, not a tuning pass. Reading alone wasn't
enough; the three failed attempts prove it.

---

## Instrumentation added (kept, zero-cost)

- `span("indexer.score")`, `span("indexer.topk")`, `span("attn.gather")`
  in deepseek_v4.py (mlx-lm `cbe3e4d`) — split the indexer/sdpa spans.
- `span("switch.up_proj"/"gate_proj"/"down_proj"/"activation"/"gather_sort"/"scatter_unsort")`
  in switch_layers.py (mlx-lm `e4761e8`) — split `switch_mlp`.
- All `span()` are no-ops when no profiler hook is registered → zero
  hot-path cost. Enabled via `EXO_PROFILER=spans` (+ `SYNC_SPANS=1` for
  real GPU timing).

## Also fixed this session (independently good)

- `maybe_apply_patches` was never called on the distributed (TP) load
  path — only single-device. Added to `shard_and_load` post-shard
  (utils_mlx.py, exo `edde3482`). This means Qwen3.5 MoE batched fused
  patches also never applied on TP clusters before this fix.

---

## Verified final state

- Both bugs fixed (seq-split `8a9cdee`, MTP bootstrap `48a4a3c`) — see
  `docs/b2-mtp-resolution-2026-06-24.md`.
- Decode baseline **37.2 t/s aggregate at c=2** (target 30, met +24%).
- Prefill knobs settled: `EXO_PREFILL_STEP_SIZE=128`, `MLX_MAX_MB_PER_BUFFER=200`
  (see resolution doc §"Post-seq-split tuning sweep").
- MoE fusion + overlap attempted, A/B'd, reverted, documented here.
- Cluster left on the clean verified config.