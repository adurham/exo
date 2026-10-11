# DSv4.1 EXL3 MoE kernel "occupancy" project — scoping memo (decision package)

Author: Phase-20 PM (delegation). Date: 2026-10-10 21:19 CDT. Worktree `/private/tmp/p20-moe-occupancy-scope`,
branch `p20/dsv41-moe-occupancy-scope` off `origin/main` `94daf8c18`. **OFFLINE, design-only**: 0 boots, 0
serving requests, 0 GPU microbenches, production untouched. All source reads on the pristine kernel
`mlx-lm/mlx_lm/models/exl3/gemv_metal.py` @ base commit `6cc9c1e` (md5 `649d2d6a0a68c8d6eab2a8a5394b7ab9`,
1956 lines). **Line-number note:** the autopsy cited `:1963/:2011/:2059` from its *instrumented* copy
(base **+75 lines**); all `:line` below are the pristine base. Inputs: autopsy report + secondary (branch
`p20/dsv41-moe-autopsy`), partition buildnote + raw (branch `p20/dsv41-moe-partition`),
`docs/PERFORMANCE_HISTORY.md`.

---

## 0. Framing (anchors this memo is judged against)

- Real production MoE chunk = **6.63 TF = 44.3%** of a ~14.96 TF mix-roofline; compute-bound (3-bit EXL3;
  crossover M*≈6.6, production M≈31). Achieved BW 7.3–10.7% of peak → **not bandwidth-bound**.
  (`autopsy-report:21-28`)
- **Uniform zero-waste control = 58.5%** (`autopsy-report:22`). The real-vs-uniform −15% is *block-packing
  waste*, which lever #1 (v20) attacked and realized only **+2.9% MoE / ~+0.9% e2e** (`buildnote:6,108`).
  ⇒ **the dominant gap is NOT packing**; it is intrinsic to the kernel at full work (58.5%).
- **This memo scopes the 58.5% → 100% residual** (lever #2). It is a *kernel-engineering* project.

---

## 1. Where the missing cycles go — ranked hypotheses + strongest evidence + cheap FALSIFIER

**Kernel shape (verified).** One TG = 128 threads = 4 simdgroups in a 2×2 grid (`:1770-1771`) owning a
**64-row × 64-col tile of ONE expert block** (`:1552-1554`); **16 fp32 accumulator fragments/simdgroup**
(`simdgroup_float8x8 C[4][4]` `:1805`, explicitly "the register budget" `:1554`). One Metal dispatch,
`grid=((out_e//64)*128, nb_max, 1)` (`:1951-1952`). Per **32-deep k-stage** (`stages=in_tiles>>1` `:1813`):
all 128 threads SWAR-decode 3-bit EXL3 weights into a **SINGLE** threadgroup buffer `wblk[32*72]`
(`:1802`, decode index `:1786-1788`, decode→store `:1816-1827`) → `threadgroup_barrier` (`:1815`) →
`threadgroup_barrier` (`:1831`) → MMA loop loads B from `wblk` (`simdgroup_load :1590-1593`) and runs
`simdgroup_multiply_accumulate` with A loaded device-direct (`:1596-1602`). **No double-buffering /
software pipelining.**

**Ablations (real mix, gu; `raw/partition/laptop_probe_and_ablation.txt:31-42`).** full **23.19 ms**;
`no_mma` **0.49 ms**; `no_tload` (remove trellis weight loads) **17.82 ms = −23%**. Uniform: full 16.31,
no_mma 0.34, no_tload 14.58 **= −11%**.
⇒ The **MMA path is ~98% of wall** (removing MMA collapses the kernel), and removing the *weight supply*
recovers **23% (real) / 11% (uniform)** → the MMA issue is **stalling on weight-operand availability**,
not on raw ALU.

| # | Hypothesis | Strongest evidence | Cheap falsifier |
|---|---|---|---|
| **H1** | **Serialized decode↔MMA, no pipelining** — single `wblk`, 2 barriers/stage; tensor cores idle during decode and stall on B-operand readiness. | `wblk` single buffer `:1802`; 2 barriers/stage `:1815,:1831`; `no_tload` = **−23% real / −11% uniform** (`:31-42`); design note says double-buffering was tried and **dropped as "noise" because decode measured ~1% at large-M S=4096** (`:1561-1562`) — that is the wrong regime (real M≈31, ragged). | `no_tload` vs `no_tload+double-buffer` variant on the real chunk; per-kernel/occupancy counters; decode ablation `EXL3_MM_NODECODE=1` (`:1755`). |
| **H2** | **A-fragment load latency, unhidden** — A loaded device-direct inside the k2/r MMA loop, not hoisted; the in-tree `v19e` hoists them (`:1741-1749`) but is **not the default** (`:1578` `v19d`) and appears **never measured on the real chunk**. | `:1596` device-direct A load in the MMA loop; `_mm_seg_hoisted` exists `:1741-1749`, selectable `EXL3_MM_SEG=v19e` `:1926`. | Measure **v19e vs v19d**; add a `no_aload` ablation to get *its own* ceiling (note: `no_tload` does **not** bound A-loads). |
| **H3** | **Occupancy / register budget** — only 4 simdgroups/TG; register pressure caps latency-hiding. | "16 fp32 accumulator fragments each — the register budget" `:1554`; **dynamic r-loop 2.4× slower** (static register alloc) `:1620-1625`; **v21 (256-thread/128-row) regressed 6.27→4.36 TF**, attributed to register pressure (`buildnote:162-166`). | Metal **occupancy + limiter** counters; an isolated sweep of threads/TG (64 vs 128) and accumulator tiling. |
| **H4** | **Grid shape vs tiny live work** — `grid=(out_e/64·128, nb_max, 1)`; DSv4.1 puts ~8 live rows per 64-row block, so many fragments/TGs are dead / early-return. | `:1951`; comment `:1571-1576` ("7/8 of the token-row fragments … dead work"); real `nb_real=397`, `blocks_full/row=2.068` (`autopsy-report:47-49`). | **Already down-ranked**: lever #1 removed this and got only +2.9%; the zero-waste uniform control is still 58.5%. |
| **H5** | **In-kernel decode ALU issue-rate (raw cost)** | decode ~1% at large-M `:1561`; `no_mma`=0.49 ms shows decode+everything-non-MMA is tiny. | Down-ranked as *raw ALU cost*; survives only as *latency* (folded into H1). |

**Top two: H1 (serialized decode↔MMA, ~23%/−11% slack) and H2 (unhidden A-loads, separate ceiling).**
H3 (register occupancy) is the reason *up-sizing* failed, not (yet) evidence of an unexploited lever.

---

## 2. Directions + realistic recovery bounds (anchored to v21 FAIL and the 58.5% control)

- **Dir A — software-pipeline decode↔MMA** (`wblk` ping-pong / prefetch next stage while MMA runs).
  Bound = the weight-supply dependency, ceiling **+23% real / +11% uniform**; realistic capture **⅓–½ ⇒
  +8–12% MoE ≈ +3% chunk**. *Why it beats the anchors:* it targets the one slack the ablation measured
  directly; the design note that dropped double-buffering measured the **wrong regime** (`:1561-1562`, S=4096,
  decode ~1%) — at production M the same stage is −23%.
- **Dir A′ — hoist A-fragment loads** (`v19e`, already in-tree, unmeasured). Bound = its own `no_aload`
  slack; cheap. *Kill-if* `no_aload` < `no_tload` (i.e. A-loads are not a material share).
- **Dir B — threadgroup up-sizing (256 threads / 128-row, decode sharing).** **KILLED**: v21 regressed
  hard (real 6.27→4.36, uniform 8.91→6.75) on occupancy/register pressure (`buildnote:162-166`;
  `laptop_probe_and_ablation.txt:22-27`). Up-sizing has no theory that beats a measured −30%.
- **Dir C — M-major / work-stealing partition.** This is **lever #1, a different mechanism**; its own bound
  is ~uniform level (59%) and v20 realized only +2.9%. **Not the occupancy project** (see §5 on bundling).
- **Dir D — decode-active-experts-once to fp16, then MLX steel (`gather_mm`) does the GEMM.** Near-neighbour
  already killed (*pre-converted dense* path = 0.87×, i.e. **slower**, `autopsy-report:72-77`) — but that
  arm was a per-expert *dense* GEMM, not MLX's batched steel primitive on the real mix. No cheap theory it
  wins, but a **half-day price** is worth it: if steel runs the mix near 14–15 TF minus amortized dequant,
  the whole occupancy question is moot.
- **Dir E — warp-specialized producer/consumer (decode warps + MMA warps).** Needs more threads per TG
  (the v21 direction, measured dead) or a different decomposition. **KILL** — no cheap theory.
- **Dir F — split decode kernel + MMA kernel pipelined on two streams.** Adds ~5.3× weight HBM traffic +
  a second launch; §4.6 says per-call-isolation wins vanish once pipelined. **Backlog**, not first.

**Killed with no theory of beating the anchors:** B, E, and C-as-occupancy (the zero-waste 58.5% control).

---

## 3. First probes (cheapest falsifiable) — **DESIGN-ONLY, not run** (offline; no node/GPU contacted)

> Reuse `bench/p20_moe_partition_{bench,bitexact,ablate}.py` + `bench/p20_moe_autopsy.py` (shapes, ceilings,
> `time_launches`). **Measure in the PIPELINED chain AND e2e, never per-call-in-isolation** (§4.6 trap
> `PH:1267-1302`). Studio2-only, idle-guard + canary, ≤2 GPU-h, never touch the exo server, no other worktrees.

- **P0 — noise band (½ h).** Run v19d ≥5× on the real chunk; establish σ. Gate must exceed **~3σ**
  (laptop/node thermals can eat ~5%).
- **P1 — attribution + occupancy counters (½–1 day).** Metal GPU capture / Instruments (occupancy,
  ALU utilization, limiter, threadgroup-memory/barrier stall) on the seg kernel; cross-check with
  `MLX_GPU_TIME=1` per-command-buffer `mx.metal.gpu_time_ns()` (`PH:11708-11717`; ≈per-kernel with
  `MLX_MAX_OPS_PER_BUFFER=1`). **Falsifier:** if occupancy is *not* the limiter and ALU utilization is
  already high → the gap is structural → **PARK**. (Note: GPU-busy-vs-host-wall alone is a **non-sequitur**
  for occupancy — it measures bubbles, not stalls; use the counters.)
- **P2 — variant + ablation sweep in isolation (1 day).** v19d vs **v19e** (hoisted, `:1926`); add a
  **`no_aload`** ablation (bound for v19e) alongside `no_tload`/`no_mma`/`EXL3_MM_NODECODE=1`; report TF
  on real+uniform, **bit-exact-checked** (`bench/p20_moe_partition_bitexact.py`, 153/153 convention).
- **P3 — (parallel, ½ day, Dir D price).** decode-once→fp16 + MLX `gather_mm`/steel on the real mix,
  pipelined-chain timing. If it wins, occupancy is moot.
- **P4 — only if P1 counters point at a barrier/latency stall:** build the `wblk` ping-pong and re-run P2.

---

## 4. Effort / risk / gate

- **Effort.** Spike (P0–P3): **2–3 agent-days**. Full project (only if the gate clears): **3–6 weeks**
  (kernel variants + bit-exact harness + TP2 pipelined e2e + battery/quality gate). High variance.
- **Risks.** (1) **Numerics** — any restructuring must keep live rows **bit-exact** (ping-pong/hoist don't
  change accumulation order → a mismatch is a *bug*, not a tolerance). (2) **Pipelined-vs-isolated trap**
  (§4.6 `PH:1267-1302`) — an isolated win may evaporate in-chain; gate in-chain + e2e. (3) **MMA-tiling
  wall** — hand-written Metal historically ~4.4–6.5 TF vs MLX steel 14–15 TF (`PH:6370`); the fused kernel's
  achievable ceiling is *below* 15.14, so **the full 44→100% is not recoverable** — only the measured slack
  is. (4) **Register pressure** (v21) caps TG up-sizing. (5) **Maintenance** — another env-gated kernel
  variant in a hot submodule.
- **KILL gate (pre-written, falsifiable, pre-registered):**
  1. **P1:** counters show occupancy is **not** the limiter (ALU utilization already high) → **PARK**.
  2. **P2/P4:** no variant beats **v19d by ≥5% MoE throughput in the PIPELINED chain** (gain > 3σ of P0's
     noise band) with **bit-exact** live rows → **PARK permanently**.
  3. **P2:** decode-side headroom (`no_tload`) **< ~10%** → skip double-buffering (Dir A), keep v19e only.

---

## 5. Recommendation — **CONDITIONAL GO** (fund a bounded spike, not a project)

**Fund a 2–3 agent-day spike (P0–P3) + a ½-day `gather_mm` price (P3).** The lever is real (58.5% at zero
waste; a measured −23% weight-supply slack) and one free variant (`v19e`) is unmeasured — but the two prior
attempts regressed and the realistic ceiling (**+8–12% MoE ≈ +3% chunk**) is the *same order* as lever #1's
realized +0.9% e2e. Do **not** fund a multi-week kernel project up front.

**The user's decision is therefore:** (1) **fund Y/N** the 2–3 agent-day spike; (2) then **fund the full
project only if** the pre-registered kill gate clears. If the user wants a single upfront answer:
**fund the spike, hold the project.**

**Lever #1 (v20) does NOT naturally bundle.** Different mechanism (block partitioning, not kernel
pipelining); v20's +2.9% MoE / +0.9% e2e is orthogonal and already *below its own deploy bar*. It is a
numerics-neutral, env-gated block (`EXL3_MM_SEG=v20`) that can ride along later if this project's measurement
harness is reused, but there is no joint win to bank.

**Comparative note (prefill priority).** This project's ceiling (~+1–3% e2e) is comparable to lever #1 and
to v20's realized gain. The **dual-microbatch drain comm-hiding** lever (autopsy secondary:29-40; `PH:11705`)
is **absent** and **unpriced** — a genuine TP prefill lever that may out-rank a bounded MoE kernel
micro-optimization, and should be priced in parallel.

---

## Appendix — evidence index
- Kernel (pristine `6cc9c1e`): grid `:1951`; TG `:1893-1956`; layout `:1552-1554,:1770-1771`; acc `:1805`;
  `wblk` `:1802`; barriers `:1815,:1831`; decode `:1786-1788,:1816-1827`; MMA `:1583-1602`; stages `:1813`;
  guard/tail `:1610-1613,:1713`; v19d default `:1578`; v19e hoist `:1741-1749,:1926`; design notes
  `:1561-1564,:1571-1576,:1620-1625`.
- Ablations: `raw/partition/laptop_probe_and_ablation.txt:31-42`; v21 `:22-27`.
- Autopsy: `autopsy-report:21-35,47-56,72-91`; secondary `:21-40`.
- Partition: `buildnote:6,106-129,159-170`.
- History: fusion trap `PH:1258-1302`; GPU-time Q5 `PH:11708-11717`; unsync caution `PH:2138-2157`; MMA wall `PH:6370`.
- Probes not run (design-only); no node/GPU contacted; production md5 `649d2d6a…` untouched.
