# Phase 8 — Build-path fork: consult verdict + its two verification gates

Date: 2026-09-28
Status: **verifications complete** — verdict stands: 2-3 day kernel spike
first, then commit
Consult: GLM-5.3 on ollama-cloud (the sanctioned Fable substitute), run on the
real acceptance number (phase 7)

## Bottom line

The fork between the two unbuilt paths — (A) EXL3 kernels ported into the
runtime, resident at ~98 GB/rank; (B) native MXFP4 experts streamed from SSD —
went to the reference model with the measured acceptance. Verdict:

> **"Call: build B first — but only after a 2–3 day spike that could flip it to
> A."** … "If 2.65× can't be moved toward ~1.5×, the gap is format-inherent and
> you commit to B. If it can, A wins outright — resident, no I/O cliff … and
> the streamer is never built."

It flagged two things to verify before spending anything. Both are now
verified against the live artifacts, not by argument:

1. **Is the 9.84 tok/s projection an EXL3 build?** — **No.** It is the port's
   own **affine 4-bit gs=64** kernels: the converted safetensors carry U32
   packed weights + BF16 scales/biases in exactly the affine shapes, and the
   port source contains zero EXL3/trellis references. So the projection
   carries **no EXL3 penalty**, the kernel spike really does decide path A,
   and the consult's "A is dead on arrival" branch does not apply.
2. **Does chunk verify's 5-row fan-out collapse Plan B's SSD ceiling?** —
   **No.** Simulated on the real gate trace: cold experts/token is essentially
   unchanged across plain and speculative modes (6.72-6.91 vs 6.96 at a
   256-expert cap). The union of a verify window's picks is the same expert set
   plain decode reads over the same positions — the window only advances by
   the tokens it commits.

## Gate 1 — kernel-currency check (why 9.84 is not an EXL3 number)

Evidence from the converted build headers
(`~/v41-mtp-converted/model-layers-00.safetensors`):

- `layers.0.ffn.experts.gate_proj.weight` `U32 [384,2304,640]` +
  `.scales`/`.biases` `BF16 [384,2304,80]` ⇒ 4-bit packed, group size 64,
  affine — exactly what the port's `convert.py` (`mx.quantize(...)`) emits.
- `grep -rn -i 'exl3|trellis' deepseek_v41_mlx/` → no matches.

Cross-check from phase 2: the affine control built at the same shape
(4-bit/gs64) timed **1.10 ms vs 1.025 ms** for production MXFP4 at R=4 — the
port's kernel class is production-class. EXL3 v3 kernels are **2.65-3.1x**
that class at decode/verify.

So today's A-plain ≈ 9.84 tok/s with the MoE share scaled by ~2.9x; the
spike's target is to move the 2.65x toward ~1.5x. That number decides A vs B.

## Gate 2 — verify fan-out vs Plan B's SSD ceiling

Simulator: `scripts/p11_verify_fanout_sim.py` (real trace `p8c_trace.json`,
4 layers, 746 positions; per-layer LRU; union-over-window reads; advance =
tokens committed; steady-state skip 85 positions).

Key rows, cap = 256 experts/layer (rank budget: native 96.3 GB / EXL3 68.2 GB):

| mode | cold experts/tok | native MB/tok | native ceiling tok/s |
|---|---:|---:|---:|
| plain | 6.96 | 130.8 | 49.7 |
| spec a=0% | 6.72 | 126.3 | 51.5 |
| spec a=25% | 6.71 | 126.1 | 51.5 |
| spec a=50% | 6.71 | 126.2 | 51.5 |
| spec a=100% | 6.91 | 129.9 | 50.0 |

The ceiling holds **through the MTP shape**, not just plain decode. Calibration
note: this sim counts distinct misses directly (6.96); the phase-5 sim applied
a hit-rate to picks (8.70). Same picture, ~20% method spread.

**Caveats (optimistic by construction):** 4-layer trace (a full 40-layer,
multi-domain workload raises distinct counts); per-expert fetch granularity
assumed; no prefetch overlap modeled (consult: real overlap 60-80% → plain
~25-32 tok/s); pipeline stragglers and the engram stream are outside this
model.

## Still open (per the consult, unchanged)

- **B**: prefetch overlap must be engineered (60-80% assumed); full-model
  routing concentration unverified; the streamer is the bigger build.
- **A**: entirely spike-dependent.
- **Both**: the real end-to-end V4.1 numbers (including acceptance) materialize
  only when a full-body runtime exists — that is the shared first deliverable.

## Next action (recorded plan)

Run the **EXL3 MoE kernel spike** — timeboxed 2-3 days, node 1, scratch venv
only:

- shapes R=1/4/8 (+ R=512 prefill), E=384, D=5120, H=2304, real weights
- entry points: `~/repos/ref/PonyExl3/ponyexl3/mlx/exl3_moe.py` (v3 chunked-
  prologue patch: `~/exl3-moe-v3-chunked-prologue.patch`); bench methodology
  from phase 2 (mx.eval per call, median over reps, the affine control at the
  same shape)
- **pass**: ratio moves meaningfully toward 1.5x ⇒ A is the build
- **fail**: stuck near 2.65-3x ⇒ format-inherent ⇒ commit to B (streamed
  experts)

## Artifacts

- `scripts/p11_verify_fanout_sim.py`
- `raw/p11_fanout_sim.out` — full table
- `raw/consult_verdict.txt` — the consult answer, verbatim
