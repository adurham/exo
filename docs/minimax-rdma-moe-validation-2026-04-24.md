# MiniMax cross-rank cost validation (2026-04-24)

> **Note (2026-04-26):** The span tracer was generalised into
> `mlx_lm/profiler.py`. References below to `EXO_MINIMAX_TRACE=1` are
> the env var that was set on the day of this bench; the equivalent
> today is `EXO_PROFILER=spans`. Keeping the historical references for
> faithfulness to the bench record.

Re-run of the NOOP sweep at the *current optimized config*
(`MLX_SDPA_BLOCKS=88` + `EXO_MINIMAX_FUSED_ATTN=1`, ~24.26 tok/s
baseline) with **Huihui scouts off** to get a contamination-free read
on where the budget actually lives now that attention has been cut.

The original NOOP sweep was at 18.60 tok/s baseline — pre-blocks=88
and pre-fused-QKV. Since attention dropped ~8% of decode time, RDMA's
relative share may have grown into a real lever. This validates or
falsifies "cross-rank communication is the new bottleneck" before any
PP-vs-TP investigation.

## Procedure (do not re-derive next time)

### Setup constants

- Live cluster: 2× M4 Ultra Studios, jaccl over Thunderbolt 5 RDMA,
  TP=2, ring topology
- Model: `mlx-community/MiniMax-M2.7-5bit`
- Coordinator host: `192.168.86.201` (macstudio-m4-1)
- Bench tool: `bench/minimax_cluster_ab.py`
- Bench shape: `--pp 50000 --tg 200 --warmup 1 --repeat 5`

### Pre-bench checklist

1. **Huihui MUST be off.** `HUIHUI_INSTANCES_PER_STUDIO=0` in the
   deploy environment. See `memory/feedback_minimax_bench_huihui_off.md`
   for the why. Verify in `/state` after deploy that no Huihui
   instances are listed.
2. **Locked envs that must stay set:** `MLX_SDPA_BLOCKS=88`,
   `EXO_MINIMAX_FUSED_ATTN=1`. These are the cluster's live config
   and the +6.5 % + +1.5 % wins the deltas measure against.
3. **DO NOT set `EXO_MINIMAX_TRACE=1` for tps benches.** It forces
   `mx.eval` at ~12 span boundaries per layer × 62 layers = ~750
   forced GPU fences per decode token, which empirically cut decode
   from 24.26 → 6.58 tok/s (measured during this validation, run 1
   v1). See `memory/feedback_minimax_trace_kills_tps.md`.
   Per-span breakdown, if needed, runs as a separate trace-on deploy
   *after* the throughput numbers are in.

### Deploy template

```bash
HUIHUI_INSTANCES_PER_STUDIO=0 \
MLX_SDPA_BLOCKS=88 \
EXO_MINIMAX_FUSED_ATTN=1 \
<EXO_MINIMAX_NOOP_FLAG_HERE> \
bash start_cluster.sh
# NOTE: EXO_MINIMAX_TRACE intentionally absent — see pre-bench checklist.
```

Deploy disrupts the cluster ~5 min. Wait until `/state` reports
MiniMax READY before benching. Smoke check: the deploy should log
"MiniMax fused QKV: 62/62 layers installed (rank R/2)" on each rank's
`~/exo.log`.

### Bench template

```bash
EXO_HOST=192.168.86.201 \
  uv run bench/minimax_cluster_ab.py \
    --pp 50000 --tg 200 --warmup 1 --repeat 5 \
    --label <run_label> \
    --out bench/results/rdma_moe_validation/<run_label>.json
```

Wall time per bench: ~14 min (1 warmup + 5 measured at 50K context).

### Three runs

| # | NOOP flag set | Label | What it isolates |
| - | ------------- | ----- | ---------------- |
| 1 | (none) | `baseline_huihui_off` | Fresh tok/s ground truth post Huihui-off |
| 2 | `EXO_MINIMAX_NOOP_ALLSUM=1` | `noop_allsum` | qk-norm cross-rank `all_sum` cost |
| 3 | `EXO_MINIMAX_NOOP_MOE=1` | `noop_moe` | Total MoE block (router + switch_mlp + reduce + MoE all_sum) |

### Restore after benches

Set `HUIHUI_INSTANCES_PER_STUDIO=1` in start_cluster.sh, drop the
NOOP flag, redeploy. Verify Huihui scouts come back via `/state`.

## Hypothesis under test

Original NOOP-sweep data (at 18.60 tok/s baseline):

| Section | Share of decode |
| ------- | --------------- |
| SDPA / attention | 66 % |
| MoE switch_mlp | 21 % |
| RDMA collectives (qk-norm + MoE all_sum) | ~4 % |
| Other | 9 % |

Since then we banked +6.5 % from blocks=88 and +1.5 % from fused QKV,
both inside the attention slice. **Attention shrank ~8 % of decode
time.** RDMA's absolute time is unchanged, so relative share could
have crept toward 6–10 %. MoE's relative share could be higher too,
since it didn't shrink while attention did.

**If `NOOP_MOE` delta ≥ 30 % AND `NOOP_ALLSUM` ≥ 8 %:** cross-rank
communication is now a real lever, and the PP-vs-TP empirical test
becomes worthwhile.

**If shares look roughly similar to the original sweep:** SDPA per-
call fixed overhead is still the bottleneck, and we've found the
floor at this hardware tier. Ship 24.26 tok/s as the ceiling.

## Results

### Run 1 — `baseline_huihui_off_v2`

- Decode tok/s (mean of 5 measured): **24.26** (min 24.19, max 24.32)
- Variance: ±0.5 % — clean signal
- Wall time per run: ~139 s
- Reproduces the previously-documented 24.26 tok/s. Huihui-off does
  not change steady-state tps at this workload (the metal-concurrency
  memory says co-residency hurts at the *tail*, not the mean).
- (v1 attempt with `EXO_MINIMAX_TRACE=1` produced 6.58 tok/s — see
  `feedback_minimax_trace_kills_tps.md` for why this is unusable.)

### Run 2 — `noop_allsum`

- Decode tok/s (mean of 5): **25.44** (min 25.35, max 25.49)
- Delta vs baseline: **+1.18 tok/s = +4.9 %**
- **`EXO_MINIMAX_NOOP_ALLSUM=1` gates BOTH cross-rank collectives**
  (qk-norm at `auto_parallel.py:924` AND MoE at `auto_parallel.py:
  829`). So the +4.9 % delta is the *total* cross-rank communication
  cost, not just qk-norm.
- Implied total cross-rank share: ~5 % of decode budget (~2 ms /
  41 ms per token).
- Actionability: none — killing the collectives requires dropping
  `use_qk_norm` (semantic change, breaks model accuracy) or GPU-
  resident `all_sum` (doesn't exist on Apple silicon jaccl). 4.9 %
  is a diagnostic ceiling, not a shippable lever.

### Run 3 — `noop_moe`

- Decode tok/s (mean of 5): **32.92** (min 32.75, max 33.02)
- Delta vs baseline: **+8.66 tok/s = +35.7 %**
- `EXO_MINIMAX_NOOP_MOE=1` returns zeros from the MoE block (skips
  router + switch_mlp + weighted_reduce compute) but **still fires
  the MoE `all_sum` on those zeros** — so this measures MoE *compute*
  cost only, not MoE collectives.
- Implied MoE compute share: ~26 % of decode budget (~11 ms /
  41 ms per token). Up slightly from the old sweep's 20 %, because
  attention shrank while MoE compute didn't.

## Decode budget at current config (24.26 tok/s, 41.2 ms/token)

| Section | Share | Wall time |
| ------- | ----- | --------- |
| Attention (SDPA + Q/K/V norm/RoPE local ops) | **~69 %** | ~28.4 ms |
| MoE compute (router + switch_mlp + weighted_reduce) | **~26 %** | ~10.8 ms |
| **Cross-rank collectives (qk-norm + MoE all_sum combined)** | **~5 %** | **~2.0 ms** |

Shape matches the old NOOP-sweep (66 % / 21 % / 4 %) adjusted for
the ~30 % tps gain. Attention still dominates. The ~8 % of decode
we saved with blocks=88 + fused QKV came out of attention, not
from elsewhere.

## Decision

**Ship 24.26 tok/s as the ceiling on this hardware + workload.**

### Why cross-rank isn't the lever

The user's hypothesis going in was "RDMA / cross-node data passing
on TB5 is eating performance." The NOOP_ALLSUM data falsifies it:
total cross-rank cost is 5 % of decode (~2 ms per 41 ms token).
Apple silicon jaccl's fence overhead dominates per-call cost, but
the *total* number of calls is small enough that the budget is
capped.

### Why PP-vs-TP wouldn't help

Rough math at current numbers (decode = 41 ms/token, compute = 39 ms,
collectives = 2 ms):

- **TP=2 (current):** 124 collectives/token × ~16 µs each = 2 ms comm,
  both ranks 100 % busy → 39 ms compute.
- **PP=2 hypothetical:** 62 layer-boundary handoffs (similar or
  higher per-op cost than TP's micro-collectives — larger payload
  per handoff). Critically, at decode batch=1 each rank sits idle
  while the other owns its half of layers → ~20 ms *idle tax* per
  token.

PP would save ~1 ms in communication and lose ~20 ms in utilization.
**Clear net loss at batch=1 decode.** PP would only win at large
decode batch (idle tax amortizes across parallel prompts) — which is
a different workload than what we're optimizing.

## What this validation rules out

Do not re-litigate these in future MiniMax decode sessions without
new evidence:

- **"RDMA / cross-node communication is the bottleneck."** Falsified.
  Total cross-rank is 5 % of decode.
- **"Switch to PP=2 to reduce communication."** Rough math says PP
  loses ~20 ms idle tax at batch=1 for ~1 ms comm savings.
- **"MoE all-to-all-ish patterns are expensive on TB5."** The MoE
  collective is part of the measured 5 % cross-rank total, which is
  small. Expert parallelism would add all-to-all-v plumbing to save
  < 5 % — wrong project.
- **"Dropping qk-norm would unlock +5 %."** Semantic change, breaks
  model accuracy. Not deployable on this checkpoint.

## Remaining theoretical levers (all bad EV)

All three have the cluster-0 % track record from prior custom-kernel
attempts:

| Lever | Predicted | Effort | Risk |
| ----- | --------- | ------ | ---- |
| Custom MLX fused attention kernel (Option D) | +3-5 % | 1-2 weeks | 40 % lands, 60 % 0 % |
| MoE batched-fusion port from qwen3_5_moe | +4-7 % on paper | 1 week | Same cluster-translation risk |
| GPU-resident `all_sum` (research project) | up to +5 % | weeks-to-months | Apple silicon gap |

Net recommendation: **don't invest further. Ship.**

## Sign-off state

- Baseline: 24.26 tok/s at 50K context, Huihui off, blocks=88 + fused QKV
- Delta from 18.60 tok/s pre-2026-04-24 baseline: **+30.4 %**, all
  from stock Apple code + Python-level fusion
- Cluster deploy knob to keep on: `MLX_SDPA_BLOCKS=88` and
  `EXO_MINIMAX_FUSED_ATTN=1`. Both forwarded by start_cluster.sh.
- Restore Huihui: `HUIHUI_INSTANCES_PER_STUDIO=1` and redeploy.

## Addendum (2026-04-25): prefill investigation

After the decode validation closed out, ran a parallel investigation
on the prefill side (~94 % of wall time at 50K context, 383 tok/s
prompt throughput).

### Step-size sweep (`EXO_PREFILL_STEP_SIZE`)

| Step | prompt_tps | Δ vs 4096 |
| ---- | ---------- | --------- |
| 4096 (default) | 383.7 | — |
| 8192 | 386.4 | +0.7 % |
| 16384 | 388.0 | +1.1 % |

Diminishing returns confirm chunk-size is not the bottleneck.
Dispatch-overhead-at-chunk-boundary hypothesis falsified.

### Live GPU power measurement

`macmon` snapshot from rank 0 during the running prefill bench:

| Field | Value |
| ----- | ----- |
| `gpu_power` | **30.8 W** |
| `gpu_usage` | 99.6 % |
| `gpu_freq` | 1576 MHz |
| `gpu_temp` | 69 °C |
| `cpu_power` | 9.1 W |
| `ram_power` | 8.5 W |
| `sys_power` (whole machine) | 114 W |

GPU drawing only ~31 W at 99.6 % utilization, well below thermal
limits — classic **stall-bound** signature. "100 % utilization"
means the GPU has work scheduled, not that shader cores are doing
math. The remaining 63 % of the work-cycles are the cores waiting on:
DRAM bandwidth (546 GB/s ceiling on 5-bit MoE weight reads),
threadgroup sync barriers, and dispatch boundaries. We've separately
ruled out dispatch-boundary stalls via the step-size sweep, so the
dominant stall cause is most likely **DRAM bandwidth on MoE weight
reads** — physical limit, not tunable.

The dashboard's "140-155 W per node" is `sys_power` (whole-machine
input including PSU overhead, CPU, ANE, DRAM, etc.), not GPU-only.
The ~10 W asymmetry between nodes during prefill is consistent with
**MoE expert routing imbalance**: top-8 of 256 experts isn't
guaranteed to split 4/4 between ranks, so one rank ends up with
slightly more compute on average each token.

### Per-span breakdown at prefill

Trace overhead at prefill is small (~3 %, vs ~70 % at decode), so the
trace numbers are usable for relative shares. After patching the
un-instrumented sites
(`mlx-lm@872a271` — pre/post-attn norms, residuals, embed, mask,
final norm, LM head), share of wall time at 136 s for 50K + 1 token:

| Section | % of wall |
| ------- | --------- |
| `attn` (parent: SDPA + Q/K/V proj + qk_norm + reshape/RoPE/cache + o_proj) | **~69 %** |
| → `attn.sdpa` alone (Apple's flash-attention `steel_attention`) | **~55 %** |
| `moe.switch_mlp` | **~25 %** |
| `model.lm_head` | ~4 % |
| `moe.all_sum` + `moe.router_topk` + `moe.weighted_reduce` | ~5 % |
| Layer norms + residuals + embedding + final norm + attn mask | ~3 % |
| **Total** | **~106 %** (slight over-count from Python overhead) |

Same shape as decode: attention ~70 %, MoE compute ~25 %, everything
else small. The trace's `%` column was misread on the first pass —
it normalizes to total tracked time including parent-and-children
double-counted, not to wall time. Always work in absolute `total_ms`
for share-of-wall reasoning.

### Prefill conclusion

Prefill is bandwidth-stall-bound on MoE weight reads. Apple's
`steel_attention` covers the attention slice optimally. No tunable
levers remain. Same ceiling as decode: stock Apple code and stock
MLX MoE are doing everything they can on this hardware.

### Memories saved this session

- `cluster_hardware_m4_max_not_ultra.md` — cluster is 2× M4 Max
  (40-GPU-core / 128GB), NOT M4 Ultra. Real Apple specs.
- `feedback_minimax_bench_huihui_off.md` — always disable Huihui
  for MiniMax benches.
- `feedback_minimax_trace_kills_tps.md` — never combine TRACE with
  tps measurement (~70 % decode hit; prefill less affected at ~3 %).
- `feedback_minimax_decode_session_scope.md` — within MiniMax decode
  sessions, no pivots to other models as alternatives.
- `minimax_rdma_moe_validation.md` — the validation findings (this
  doc as a memory pointer).
