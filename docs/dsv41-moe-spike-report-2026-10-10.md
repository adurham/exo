# DSv4.1 EXL3 MoE kernel "occupancy" project (lever #2) — bounded SPIKE report

Author: Phase-20 PM (delegation). Date: 2026-10-10/11 CDT. Worktree `/private/tmp/p20-moe-spike`,
branch `p20/dsv41-moe-spike` stacked on `origin/p20/dsv41-moe-partition` (`9de67914d`). **OFFLINE,
studio2 bench only**: 0 boots/relaunches/kills, 0 serving/generation requests, 0 writes to the exo
server or its files; production md5 `649d2d6a0a68c8d6eab2a8a5394b7ab9` verified unchanged before/after
(node src-tree **and** site-packages); the live exo server (screen `exorun`, PID 53429) untouched,
never restarted. Run-book: `origin/main:docs/dsv41-moe-occupancy-scope-2026-10-10.md`.
Kernel measured: `mlx_lm/models/exl3/gemv_metal.py` base `6cc9c1e`; node scratch pkg
`/tmp/p20moe_partition_pkg` (v19c/v19d/v19e/v20), mlx `0.32.3.dev20260918+603f16eb7`, real 196 GB ckpt.

---

## 0. KILL-GATE VERDICT (pre-registered, applied verbatim)

> Gate 1 (§4): *P1: counters show occupancy is not the limiter (ALU utilization already high) → PARK.*
> Gate 2 (§4): *P2/P4: no variant beats v19d by ≥5% MoE throughput in the PIPELINED chain (gain > 3σ of P0's
> noise band) with bit-exact live rows → PARK permanently.*
> Gate 3 (§4): *P2: decode-side headroom (no_tload) < ~10% → skip double-buffering (Dir A), keep v19e only.*

| gate | verdict | the numbers |
|---|---|---|
| **#1 occupancy is the limiter?** | **CLEARED** — NOT the limiter is *not* the case; the limiter IS an operand-supply stall/occupancy | no_mma collapses the kernel to **−98.2%** (MMA = the whole wall); no_tload (remove trellis weight supply) = **−23.4% real / −11.4% uniform** ⇒ the MMA is stalled on weight-operand availability, ALU-issue is *not* saturated. `cycle_busy_pct` 98.7–99.1% is a single-dispatch bubble metric (non-sequitur per the memo), not a stall counter. |
| **#3 decode headroom < 10% → skip Dir A?** | **NOT TRIGGERED** — decode headroom is 23.4% real ⇒ double-buffering WAS in scope (P4 built) | no_tload real = **−23.4%** (gu), uniform = −11.4%; both ≥ 10%. |
| **#2 any variant ≥5% in-chain bit-exact?** | **✖ FIRES → PARK PERMANENTLY** | best in-chain bit-exact variant = **v20 +2.77% real**; v19e +0.12%; **v19p ping-pong +1.74%**; P3 gather_mm 0.86× (0.29× decode-incl). **None ≥ 5%.** |

**Bottom line: no measured variant — including the two never-before-measured candidates (v19e hoisted-A
and the v19p ping-pong) and the Dir-D gather_mm arm — beats v19d by ≥5% in the chained prefill while
staying bit-exact. Gate #2 fires → the full multi-week MoE kernel project is NOT justified. PARK.**

Every claimed variant below is **bit-exact by construction and verified 0-diff** (fp16-as-uint16) on the
real checkpoint: **36/36** kernel cases (v19e, v20 vs v19d) + **9/9** for v19p (incl. the full chained
`_prefill`, 63 M elems). Nothing here is a tolerance approximation.

---

## 1. P0 — measurement noise band (the significance bar)

v19d ×5 on the real chunk (E=384, layer-0 trace, 2048 tok × top-6 = 12288 pairs), alternating-round
interleave, chained `_full_prefill_ms`:

| run | n | mean (ms) | **σ (ms)** | **σ (%)** | **3σ (ms)** | **3σ (%)** |
|---|---|---|---|---|---|---|
| bench | 5 | 65.838 | **0.0468** | **0.071** | **0.1403** | **0.213** |
| P4 harness | 5 | 65.676 | **0.0302** | **0.046** | **0.0906** | **0.138** |

σ ≈ 0.03–0.05 ms (≈0.05–0.07%). The memo's fear that "node thermals can eat ~5%" is unfounded on an
idle-guarded studio2: the noise band is **~0.2% (3σ)**, so any delta ≥ 0.14 ms is real. The 5% bar is
thus ~24–35× the noise floor — significance is easy; the bar is the binding constraint, not noise.
**Significance gate, pre-registered: gain must be ≥5% AND > 0.14 ms (>3σ).**

## 2. P1 — limiter identification (`MLX_GPU_TIME=1` attribution + ablations)

- Per-command-buffer GPU-time attribution (`mx.metal.gpu_time_ns()`, gated on `MLX_GPU_TIME=1`):
  real gu_gate/gu_up/dn = 19.08/19.08/18.39 ms GPU vs 19.34/19.25/18.55 ms wall → **98.7/99.1/99.1%**
  busy; uniform 98.6–98.8%. Host wall ≈ GPU busy (no host-side bubbles).
- **The limiter is not ALU throughput.** no_mma → 0.34 ms (*−98.2%*): the tensor-core MMA loop is
  effectively the entire wall. no_tload (trellis weight loads replaced) → **−23.4% real / −11.4%
  uniform**: removing the *weight supply* recovers a fifth of the wall ⇒ the MMA is **stalled waiting
  on operands** (serialized decode→barrier→MMA), i.e. latency/occupancy, not issue-rate.
- **Gate #1 CLEARED** (occupancy/latency implicated). *Caveat:* true Metal occupancy/ALU-util/barrier-stall
  counters need Instruments (see §5 NEEDS-LIVE); the `.gputrace` capture failed headless
  (`Capture layer is not inserted`).

## 3. P2 — variant + ablation sweep (in-chain, bit-exact)

Chained `_full_prefill_ms` (the gate-#2 number), 5 rounds × 3 mixes, real checkpoint:

| mix | v19d (TF / %mix) | **v19e** vs v19d | **v20** vs v19d |
|---|---|---|---|
| **real** E=384 | 65.834 ms / **6.605 TF / 44.09%** | 65.756 / 6.613 / **+0.12%** | 64.011 / 6.794 / **+2.77%** |
| **uniform** E=192 | 49.201 / 8.839 / 58.26% | 49.060 / 8.864 / **+0.29%** | 48.134 / 9.035 / **+2.17%** |
| **skew** E=192 | 57.683 / 7.539 / 49.86% | 57.647 / 7.544 / **+0.06%** | 56.226 / 7.734 / **+2.53%** |

- **v19e (the in-tree hoisted-A variant, never measured before) is a WASH — +0.06…+0.29%, well under
  3σ on some mixes.** Hypothesis **H2 (unhidden A-loads) is DEAD**: hoisting the A-fragment loads buys
  nothing in-chain even though the `no_aload` ablation shows a 16–19% isolated share (below).
- **v20** (partition, lever #1) = +2.17…+2.77%, bit-exact — reconfirms the partition buildnote; it
  **clears 3σ but fails the 5% bar**.

Ablations (isolated single-launch, synthetic trellis, real + uniform routing mix, median-of-9):

| version | mix | full (ms) | no_tload | no_aload | no_mma |
|---|---|---|---|---|---|
| v19d | real (gu) | 18.757 | **−23.4%** | −18.7% | −98.2% |
| v19d | uniform (gu) | 13.296 | −11.4% | −16.2% | −98.0% |
| v19e | real / uniform | 18.784 / 13.260 | −22.7% / −11.2% | −18.9% / −15.9% | −98.2% |
| v20 | real / uniform | 18.166 / 12.825 | −23.4% / −10.7% | −16.9% / −15.7% | −98.1% |

- **Gate #3: no_tload real = 23.4% ≥ 10% ⇒ do NOT skip double-buffering** (Dir A / P4 kept in scope).
- The `no_aload` bound (A-load share) is 16–19% — *larger than* v19e's realised gain (≈0), confirming that
  simply issuing the A-loads earlier does not hide their latency; the register pressure of hoisting
  (`af[4]`) offsets it.

**Bit-exactness (0-diff uint16):** 36/36 — `v19e==v19d` and `v20==v19d` over synthetic trellis **and the
real dsv41 layer-0 checkpoint** weights, real/uniform/skew mixes, gu gate/up + dn (14.2 M + 62.9 M elems
each). Raw: `p20_moe_spike_bitexact.json`.

## 4. P3 (Dir D price) — decode-once→fp16 + MLX steel `gather_mm`

- Primitive **exists**: `mx.gather_mm(a,b,lhs_indices,rhs_indices,*,sorted_indices=False)`. The sorted
  fast path (`gather_mm_rhs`/`_nax`) is **M==1-only** (the fork's existing per-pair GEMV fallback); the
  generic batched gather runs full bm=64 tiles → **2.07× row padding** on the real ragged mix.
- Real ckpt: seg_pipe 57.97 ms (**7.50 TF**); steel Mt=64 pipe 67.48 ms (**6.44 TF**) = **0.86× (LOSS)**;
  interleaved ×5 confirm 0.86× (medians) / 1.17× (median-of-rounds) — **parity at best, ±2σ noisy**.
- **decode-once alone = 132.9 ms** (13.59 GB fp16 @ 102 GB/s) = **197% of the steel pipe** ⇒
  **decode-included 0.29×**. (Same structural wall as the autopsy Step-3 "pre-converted dense = 0.87×".)
- **P3 = LOSS** — Dir D is not viable, no ≥5% win. Raw: `p20_moe_spike_gathermm.json`.

## 4b. P4 (wblk ping-pong double-buffer) — built, bit-exact, priced in-chain

Triggered because P1 cleared (barrier/latency stall) and no_tload ≥ 10%. v19p = decode stage ks+1 into a
second `wblk` buffer while the guarded v19d MMA block consumes buffer ks (one barrier/stage; accumulation
order unchanged ⇒ bit-exact).

| mix | v19d | **v19p (ping-pong)** | v19d-NODECODE (decode removed) |
|---|---|---|---|
| **real** | 65.67 ms / 6.62 TF | 64.53 / 6.74 / **+1.74%** (seg3 +2.16%) | 63.63 / 6.83 / +3.11% |
| **uniform** | 48.96 / 8.88 | 47.92 / 9.08 / **+2.13%** (seg3 +3.12%) | 47.98 / 9.06 / +2.00% |

- **v19p = +1.74% real / +2.13% uniform in-chain, bit-exact (9/9 0-diff)** — clears 3σ but **fails 5%**.
- **Structural kill:** the NODECODE ablation removes the *entire* decode phase and recovers only
  **+3.11%** in-chain ⇒ the decode↔MMA serialization simply is not the dominant recoverable slack.
  Doubling `wblk` (5.6→10.2 KB/threadgroup) also halves resident threadgroups/core — a real occupancy
  cost that caps the upside. **Dir A cannot clear 5%.** Raw: `p20_moe_spike_p4.json`, emitted kernel
  `p20_moe_spike_p4_v19p.metal`.

---

## 5. Is the full project justified? — **NO (PARK)**

Gate #2 fires. Across the two previously-unmeasured candidates (v19e, v19p) and the Dir-D arm:

| candidate | in-chain, bit-exact | ≥5%? |
|---|---|---|
| v19e hoisted-A | +0.06…+0.29% | ✖ |
| v20 partition (lever #1) | +2.17…+2.77% | ✖ |
| v19p ping-pong | +1.74…+2.13% | ✖ |
| Dir D gather_mm | 0.86× (0.29× decode-incl) | ✖ |

Measured capture is **~1/8 of even the memo's own realistic bound** (memo: ⅓–½ of the +23% decode slack
⇒ +8–12% MoE ≈ +3% chunk; measured best = +2.77% MoE ≈ **+0.85% e2e**, and that is v20, lever #1, not
this project). The 58.5% zero-waste gap is therefore **structural to the hand-written fused kernel at
this decomposition** — recoverable neither by hoisting, by software-pipelining the decode, nor by
handing the GEMM to MLX's own steel/gather path. **Do not fund the multi-week kernel project.** The
kernel is at its practical ceiling for this design; further e2e is better spent on the still-unpriced
**dual-microbatch drain comm-hiding** TP lever (autopsy secondary), which is orthogonal.

**Expected e2e if a variant had won:** none won. For scale, v20 (+2.77% MoE) ≈ +0.9% e2e; a hypothetical
5%-clearing variant would be ≈ +1.5% e2e — the same order as lever #1's realized +0.9%. No joint win.

## 6. NEEDS-LIVE (user-gated; NOT executed — never touched the cluster)

1. **Metal hardware counters** (occupancy, ALU util, threadgroup-memory/barrier stall) via Instruments /
   Xcode on studio2. The headless `mx.metal.start_capture()` failed (`Capture layer is not inserted`).
   This is the only way to close Gate #1 with hardware counters — the *software* evidence (no_mma −98%,
   no_tload −23%) already cleared it.
2. **True TP2 pipelined e2e** — the §4.6 "in the pipelined chain" fully realised = the real 2-node
   prefill with all_sum/attention overlap. Here "in-chain" = the single-node chained `_prefill` + glue
   (`_full_prefill_ms`), the faithful offline proxy; a live TP2 chunk run is live-gated.
3. (moot) confirming mlx 0.32.3.dev's int32 `gather_mm` output cap matches the laptop — P3 lost anyway.

## 7. Method / safety / no-engine-contact

- Guards: `phase20_guard.py idle` ok (rc=0, reasons=[]) **and** canary `healthy` on both nodes
  (studio1 14.85, studio2 14.86 t/s) **before and after** every timed region; `state_active_tasks=0`
  throughout. Identical to the pre-spike baseline.
- **No engine contact:** 0 requests to the exo API/server, 0 generation POSTs, 0 boots/relaunches/kills;
  the `exo -v` server (screen `exorun`, PID 53429) and its files were never written to, only read
  (site-packages + src-tree `gemv_metal.py` md5 `649d2d6a…` before and after). Scratch-only writes on
  studio2 (`/tmp/p20moe_spike/`, `/tmp/p20moe_partition_pkg` unchanged md5 `479734e9…`). No studio1
  writes. Other worktrees and the shared checkout untouched. GPU budget: well under the 3 GPU-h cap
  (~15 min of timed regions total).

## 8. Artifacts (this branch)

- Harnesses: `bench/p20_moe_spike_{bench,ablate,bitexact,limiter,gathermm,p4}.py` (P0/P2 runner,
  ablation sweep incl. `no_aload`, bit-exact compare, P1 limiter, P3 price, P4 ping-pong).
- Raw: `docs/benchmarks/phase20-throughput/raw/p20-moe-spike/{p20_moe_spike_bench,..ablate,..bitexact,
  ..limiter,..gathermm,..p4}.json`, `{bench,ablate,limiter,gathermm,p4}.log`,
  `p20_moe_spike_p4_v19p.metal`, `guard_{pre,post}_*.json`.
