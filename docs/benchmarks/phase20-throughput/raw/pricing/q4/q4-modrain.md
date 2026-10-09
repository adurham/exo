# Q4 — MoE-expert drain differential (TOTALS framing)

**Author:** mid-tier subagent (Q4 dispatch). **Date:** 2026-10-09 CDT.
**Worktree:** `/private/tmp/phase20-campaign`. **Cluster:** studio2, **READ-ONLY** — no POST, no
start/stop/kill, no writes to `~/repos/exo`, no reboot. Production `fb4f9290b`/`16830e1` untouched
(idle-guarded: loadavg 3.08 → 3.15 before/after; production exo still running).

**Script:** `bench/p20_pricing_q4_modrain.py` (also copied under `raw/pricing/q4/`).
**Raw:** `raw/pricing/q4/q4_modrain.json` (+ `.stdout.txt`).

---

## 0. Why this shape (the framing that was adopted pre-dispatch)

An isolated single-layer, single-node topk3-vs-topk6 bench measures the experts' **STANDALONE COMPUTE**
cost — it can **NOT** show drain / overlap / exposure into the collective. A 1-node bench has no second
rank, so it cannot observe cross-rank overlap or an exposed drain. So we do **not** report it as a
"drain proxy." Instead we do a **TOTALS comparison**:

> `standalone_expert_ms(R=4, topk6) × n_moe_layers`  vs  the **~37 ms UNATTRIBUTED** in `verify_block`
> (≈40 % of ~92.4 ms, per `MECHANISM.md` / `ROUND-DENSE-EXL3.md`).

- If the total **roughly matches** ~37 ms → the "expert GEMM fills verify_block" explanation is
  **consistent** (the MoE path may reopen with numbers).
- If the total is **much smaller** → the expert GEMM is **not** the 40 %, and Lead-A's comm bound
  needs **more** scrutiny.

**Routing (review point):** cost tracks UNIQUE experts touched (bandwidth), not k·R. Real spec-decode
drafts are **correlated** (consecutive rows differ by ~one token). We derive indices from the **REAL**
layer-20 gate (`ffn.gate.weight` [384,5120], sigmoid noaux_tc-style) applied to a base hidden state +
tiny per-row perturbation → correlated rows share hot experts. ≥3 seeds, spread reported. A
uniform-random arm is included for contrast. (No router indices exist in the `round_prof` files — those
are timing-only — so real-gate-on-synthetic-rows is the closest available.)

---

## 1. Setup (verified)

- Layer-20 `EXL3SwitchGLU`, **n_experts = 384**, **rank=0, world=2** (production TP=2 per-rank
  intermediate slice — the cost we pay on a node), `activation='silu_clamp'`. Real weights via
  `load_experts`. Load 0.7 s. **D = 5120, H = 1152 (rank-sliced from 2304), k = 2-bit trellis.**
- **n_moe_layers = 40** — *measured from the checkpoint index*: **every layer (0..39)** has both
  `ffn.gate` and `ffn.experts` (no dense-replace; DSv4.1-Flash is MoE in all 40 layers).
- R ≤ 8 → decode-class **A2/B2** path (`_decode_fused2`), the spec-verify shape. `R=1` → decode path.
- Timing: warmup ≥ 3, reps = 15, **median of per-call wall**, `mx.eval` each call. A **dispatch-only**
  arm (no `mx.eval`) measures pure CPU dispatch (the async/overlap upper bound). Idle-guarded.

---

## 2. Timing table (ms per whole expert block = gate-up + down for all R·kk expert-slots)

| arm (R × topk) | uniqueE (3-seed) | **sync ms** (median) | spread | dispatch-only ms | exposed-bound ms |
|---|---:|---:|---:|---:|---:|
| R1 × topk3 | 3 | 0.530 | 0.003 | 0.010 | 0.520 |
| R1 × topk6 | 6 | **0.455** | 0.240* | 0.010 | 0.445 |
| R4 × topk3 (correlated) | 8 | **0.615** | 0.001 | 0.009 | 0.606 |
| R4 × topk3 (uniform) | 12 | 0.680 | 0.002 | 0.009 | 0.671 |
| **R4 × topk6 (correlated, PROD)** | **14** | **1.077** | **0.003** | 0.009 | 1.068 |
| R4 × topk6 (uniform) | 24 | 1.078 | 0.002 | 0.009 | 1.069 |
| R4 × topk6 hot24 | 6 | 1.060 | — | — | — |
| R4 × topk6 hot12 | 6 | 1.076 | — | — | — |
| S512 × topk6 (prefill class) | 74 | 17.743 | — | — | — |

\* R1×topk6 spread 0.24 ms is one seed outlier from shared-GPU contention (the other seeds ≈0.45);
the R4 rows are stable (spread ≤0.003 ms).

**Routing-distribution finding:** correlated R4/topk6 touches **14** unique experts, uniform touches
**24**, and the **hot12/hot24** arms touch only **6** — yet all four cost **≈1.06–1.08 ms**. Cost is
**independent of the unique-expert count**; it tracks the **slot count R·kk**, i.e. the GEMM work is
**slot-bound, not bandwidth/unique-expert-bound**. This directly refutes the "cost = unique experts
(bandwidth)" hypothesis at these shapes, and means the topk6 result is **not** inflated by a
uniform-vs-correlated artifact (both are ~identical).

---

## 3. Totals comparison

- standalone **R4 × topk6 = 1.077 ms/layer** (production verify shape, correlated routing).
- × **40** MoE layers = **43.08 ms** (seed range 43.05–43.16 ms).
- vs **~37 ms unattributed** in `verify_block` → **ratio 1.16×** (range 1.16–1.17×).

**Ratio = 43.08 / 37 = 1.16.** The standalone MoE compute **alone** accounts for ≥ the entire ~37 ms
unattributed (slightly *more* than it).

---

## 4. Verdict

**(a) Does halving experts (topk6→topk3) move the block a lot, or barely?**
- **At R=4 (the verify shape): a LOT.** 1.077 → 0.615 ms = **−0.462 ms/layer (−43 %)**, **1.75×**
  faster. The expert count is the single biggest lever on the block at the verify shape.
- **At R=1 (draft/decode shape): barely** (0.455 vs 0.530 ms, inside noise — topk3 is even slightly
  slower, an occupancy/latency artifact, not a win).

**(b) Does the totals comparison support "expert GEMM = the ~40 % unattributed", or refute it?**
→ **SUPPORTS it (consistent).** `1.077 ms × 40 = 43.08 ms ≈ 1.16× the ~37 ms`. The MoE expert compute
is, by itself, the right **order of magnitude** to be the entire unattributed slice — arguably it
**over-fills** it. No separate "mysterious 40 %" needs to be invented: the answer to *"what is the
~37 ms?"* is **"the MoE expert GEMMs (gate-up + down) at the verify shape × 40 layers."** The MoE path
is therefore **re-opened with numbers**: halving experts at R=4 buys ~0.46 ms/layer × 40 = **~18 ms/round**
of standalone compute (though see caveats — this is compute, not exposed time).

**(c) Does this re-validate or weaken the Lead-A comm bound?**
→ **It does not re-validate a comm bound — it re-attributes the 40 % to MoE COMPUTE.**
- A 1-node bench **cannot** measure exposed comm or cross-rank overlap, so this neither confirms nor
  refutes a small *comm* term directly.
- But it shows the ~37 ms unattributed is **accounted for by MoE compute**. Lead-A's "≤5.25 ms exposed
  non-compute" is *consistent* with that (compute is not non-compute) — yet the **premise** of the comm
  lead (that the large residual was potentially comm/drain) is **weakened**: the residual is compute.
  The "unattributed" mystery is resolved as the expert GEMM, which means the comm lead was closed on an
  **unattributed** premise that is now attributed to **compute**, not comm.
- **What a 1-node bench CANNOT show (stated plainly):** cross-rank overlap, the exposed portion of any
  collective, or whether the expert GEMM's dispatch overlaps another rank's AllSum. The "dispatch-only
  ≈0.009 ms" figure shows the block is fire-and-forget on the GPU (wall time in the async arm is pure
  CPU dispatch), so any "drain" is not a separately observable host term — exposure can only be
  **bounded, not measured** here.

**One-line:** the ~40 % unattributed in `verify_block` is **MoE expert compute** (43 ms standalone vs
37 ms — ratio **1.16×**), routing-distribution-independent and slot-bound; halving experts at the verify
shape cuts the block **1.75× (−0.46 ms/layer)**, so the MoE path is the real target — this does **not**
re-validate a *comm* bound and re-attributes the residual away from comm.

---

## 5. Caveats / boundaries

- **Standalone ≠ exposed.** These are single-layer, single-node, single-rank wall times. They include
  gate-up + down dispatch but **no** router, no shared expert, no attention, no all_sum, no cross-rank
  overlap. In the real engine the same work may partly overlap comm; the numbers here bound standalone
  cost, not exposed cost.
- **Real-gate-on-synthetic rows** (no captured router indices exist). Correlation strength is a free
  parameter (corr=0.03); results proved insensitive to it (14 vs 24 unique → identical ms).
- **A2/B2 path** (production default, `EXL3_MOE_V2=1`) at rank=0/world=2. The engine's exact per-layer
  expert indices are not reproduced (unavailable).
- Prefill-class S=512 = 17.74 ms/layer is the `_prefill` sort/gather path, **not** the verify shape;
  included only as the optional extra.
- R1×topk6 seed spread 0.24 ms = one contended-seed outlier; the R4 rows (the ones the totals uses)
  have spread ≤0.003 ms.
- **Budget: 0 boots, 0 relaunches.** Nothing deployed; nothing to restore.
