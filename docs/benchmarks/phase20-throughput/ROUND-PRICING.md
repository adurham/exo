# ROUND-PRICING.md — Fable-contested lever pricing round (Q1–Q5)

Author: Phase-20 PM (delegation). Date: 2026-10-09 (CDT). Worktree `/private/tmp/phase20-campaign`.
Branch `deploy/phase20-campaign`.

This round **PRICES** the five levers a Fable review named against prior closures. It is **NOT a ship
round**: nothing is re-quantized, no static default is changed, no code is deployed. Outputs are price
tables, attribution verdicts, and — for anything that would be a quality-gated config change — an explicit
**OWNER-DECISION memo**.

---

## 0. DECLARED BUDGET (fixed BEFORE spending, per the brief)

- **Boots: ≤2 TOTAL.**
  - **Q2 (gamma re-pricing) = 1 boot.** VERIFIED: per-request `spec_gamma` is **NOT** in the deployed build
    (`fb4f9290b`; it lives on the divergent `deploy/next14-gamma` branch, `git merge-base --is-ancestor
    01c416b10 fb4f9290b` = FALSE), and `EXO_SPECULATIVE_GAMMA` is read in generator `__post_init__`
    (`batch_generate.py:845`, `dsv4_mtp.py:3969`) → **one γ per boot**. So Q2 = 1 boot with arms
    γ2 + γ3 (control) interleaved × {benign 20K, agentic 91K}; a 2nd boot (reserve) only if a γ4 add is
    worth an extra boot.
  - **Q4 (MoE-drain differential) = 0 boots.** VERIFIED runnable as a **bench-only** process: `load_experts`
    loads the layer-20 `EXL3SwitchGLU` in the node venv on studio2; decode-class forward smoke: E=16,
    D=5120, H=1152, R=1 topk3 0.510 ms / topk6 0.682 ms, R=4 0.671/1.065 ms (this PM smoke, real weights).
    No engine, no deploy needed.
- **Everything else OFF (0 boots):** Q1 (offline microbench), Q3 (offline analysis), Q5 (offline read +
  prototype).
- **No ship of anything.** Cluster stays production `fb4f9290b` + mlx-lm `16830e1`, gates unset.

### Entry state (verified ~10:57 CDT)
Production `fb4f9290b` (both nodes; `git rev-parse HEAD` on studio2 = `fb4f9290b4e0b…`), mlx-lm `16830e1`;
2 runners Ready, loadavg 1.33/1.54/2.79 (idle); model
`~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`; node mlx `0.32.3.dev20260918+603f16eb7`
(`mx.quantize`/`mx.quantized_matmul` present). Local mlx source for Q5: `~/repos/mlx` HEAD
`ac73d0c9eeb2240725fb203d3e6745c516a3dd8e`.

### 2nd-opinion fold-in (auxiliary.consult, pre-dispatch)
Adopted: (Q1) **graph-chained timing** (not per-call at m=1 — per-call is launch-overhead-dominated at
tiny shapes; chain the whole layer's dense roster, one eval, /n_reps); **TP-sharded shapes**; **quality is
UNMEASURED** (re-quant of an already-2.9bpw-lossy tensor ≠ model-quality proxy) and **capacity check**
before the expert-side memo. (Q4) the isolated topk differential is a **compute, not a drain/overlap**
proxy → frame it as a **totals comparison** (standalone expert-block ms × n_moe_layers vs the ~37 ms
unattributed in verify_block); **cost tracks unique experts touched, not k·R** → use real-ish routing, not
uniform random; optional **forced-sync arm** to bound exposure. (Q3) confirm the 101.06 client number's
build before attributing 7.26 ms; the engine-loop read is a correctness answer → **sr-coder**. (Ordering)
run Q1/Q3/Q4/Q5 before the Q2 boot; fold any live claim into it. (Contention) benches share studio2's GPU
with production → **idle-guard before and after, bounded reps**.

---

## Q1 — Quant-format pricing (EXL3-fused vs native quantized `mx.quantized_matmul`)  [STATUS: dispatched]

## Q2 — Gamma re-pricing on the current build  [STATUS: pending, 1 boot]

## Q3 — The 7.26 ms client↔server boundary  [STATUS: dispatched]

## Q4 — MoE-expert drain differential  [STATUS: dispatched]

## Q5 — GPU-time instrument feasibility  [STATUS: dispatched]

