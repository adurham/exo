# Phase 2 — EXL3 trellis kernels on the real V4.1 MoE (measured 2026-09-27)

**Verdict: the plan's Phase 2 gates are decisively MISSED against the real
production baseline. EXL3 is 3.3x slower than the current MXFP4 path at decode
and 9.5–12x slower at the speculative-verify shape that MTP/DSpark depends on.
The kernel is *correct* (bit-exact vs fp16 reference) — this is purely a
performance result, and it is a genuine fork in the road for the V4.1 project.**

This is **not** a Plan B declaration and **not** a plan to abandon EXL3 — see
"What this does and does not mean" at the end. Read that section before acting.

## What was measured

Straight from the **already-converted V4.1 EXL3 checkpoint** (no conversion, no
exo integration, no production relaunch), in the isolated `~/phase1-exl3` venv:

- **Shapes:** V4.1 real dims — hidden 5120, `moe_intermediate_size` 2304,
  **384 routed experts**, top-6 routing. `layers.1` (k=3) and `layers.18` (k=2)
  measured separately; an E=384 run checks expert-count scaling.
- **EXL3 module:** `EXL3SwitchGLU` built from real `w1/w3` (gate/up) and `w2`
  (down) trellis tensors, `codebook=mul1` per the checkpoint's config.
- **Baseline:** the **production** expert format. V4-Flash's experts are FP8 in
  the checkpoint but `mlx-lm`'s loader converts them to **MXFP4 4-bit,
  group_size 32** at load (`utils.py`: "packing (mxfp4 experts / mxfp8
  attention+shared+MTP)"). The baseline was built by dequantizing EXL3 → fp16 →
  `mx.quantize(mode="mxfp4", gs=32, bits=4)`, so both sides run the *same
  weights*; this is format-vs-format, not weights-vs-weights.
- **Timing method:** `mx.eval()` around **every** measured call. MLX is lazy, so
  building graphs in a loop and dropping results measures almost nothing — an
  early "amortized" run of this bench was **100x off** (77 µs vs the true
  12.5 ms for the same op) and was discarded. Do not use loops-per-sync timing
  on MLX without forcing evaluation.

## Results (ms per MoE block; ratio vs production MXFP4)

E=128, layer 1 (k=3), median of 250/12/10/6/4 reps:

| rows | EXL3 ms | prod MXFP4 ms | ratio | gate | verdict |
|---|---|---|---|---|---|
| **1** (decode) | 1.305 | 0.386 | **3.38x** | 1.25 | **MISS** |
| **4** (DSpark verify) | 12.251 | 1.025 | **11.95x** | 1.25 | **MISS** |
| **8** (verify) | 19.079 | 1.951 | **9.78x** | 1.25 | **MISS** |
| 256 | 52.539 | 19.374 | 2.71x | 1.7 | MISS |
| **512** (prefill) | 54.140 | 27.356 | **1.98x** | 1.7 | **MISS** |
| 1024 | 59.358 | 43.965 | 1.35x | 1.7 | PASS |
| 2048 | 112.348 | 76.915 | 1.46x | 1.7 | PASS |

`raw/p2-final.log`. Other runs, same shape:
- **E=384** (the real expert count) makes prefill **worse**: R=512 → **3.16x**,
  R=2048 → 1.67x; decode unchanged at 3.36x. `raw/p2-e384.log`
- **k=2 layers** (18–22): decode 3.30x, verify-R4 11.44x — essentially identical
  to k=3. Lower bit width does **not** rescue the ratio, because at E=128 the
  trellis is only ~1.6 GiB and the cost is not byte-bound. `raw/p2-k2c.log`

An affine (non-EXL3) control at the same bit width and group size was also
measured to isolate format from kernel: affine 4-bit/gs64 is **10.5x faster
than EXL3 at R=4** (1.10 ms vs 11.98 ms). So the gap is the trellis kernel and
its code path, not the quantization scheme in the abstract.

## Two distinct root causes

**(a) R=2..8 takes a pathological code path.** `EXL3SwitchGLU._decode_fused2`
(the fused multi-row decode) requires `hidden_dims <= 512` — a hard
threadgroup-memory limit (`tg_g[1024]` at 4 bytes × 2 projections). V4.1's
hidden is **2304**, so `_v2_ok()` is `False` and **every R=2..8 call silently
falls through to `_prefill`**, which decodes the *entire stacked trellis* to a
transient fp16 buffer. That is the 9.5–12x.

Measured workaround (`raw`: `p2-valid.log`): loop the R=1 fused path once per
row — **4.29 ms @ R=4** and **7.56 ms @ R=8** (vs 12.25 / 19.08 through
`_prefill`). Better, but still ~4x the production verify cost. Porting the
`tg_g[1024]` prologue to chunk the hidden dimension is the real fix and is a
bounded engineering task.

**(b) The non-segmented prefill fallback cannot run at scale at all.** With
`EXL3_MOE_MM=0`, the fallback materializes `E * out * in * 2` bytes:
**3.02 GB at E=128 → int32 overflow crash** (`3019898880 > INT32_MAX`, observed);
**9.06 GB at E=384**. So the segmented-GEMM path is mandatory, and the prefill
numbers above are already the *good* path.

## Cross-check: is the 3.3x a hardware floor or a tuning problem?

The consult I took on this flagged that a flat ratio across R would indicate an
ALU-throughput wall (untunable) while a *collapsing* ratio indicates
occupancy/latency-binding (tunable). The data collapses hard: **3.38 → 11.95 →
9.78 → 2.71 → 1.98 → 1.35 → 1.46**. The R=1 number is therefore very likely
retunable — per-expert parallelism is currently `_GEM_THREADS = 128` with
`grid = E_sel * (gu_tiles//8) * 128`, and the plan itself says "profile
`_decode_fused2` tile sizes on M4 Max before deciding; PonyExl3's numbers are
M5."

Corroborating: PonyExl3's own README claims 2.28x prefill scaling from
"Advanced" tuning on M5 Max, and the plan's risk register already lists "M4 Max
slower than PonyExl3's M5 Max numbers → accept or retune tiles."

**This retuning HAS since been attempted — see the correction at the end of
this file.** The gate result below is the pre-retune number.

## What this does and does not mean

**Does:** Phase 2's stated purpose ("de-risk the two unknowns nobody has
measured") is now *answered* — the unknowns are no longer unknown. The answer is
unfavourable to EXL3 as currently implemented, and favourable to the concern
that trellis decode is the wrong bet for this silicon.

**Does not:** justify switching to Plan B today. Plan B ("keep the base
checkpoint's MXFP4 experts and MXFP8 dense, hold ~60% of experts per rank
resident, stream the rest") uses **~4.25 bpw** (4-bit + E8M0 scales) against
EXL3's measured **3.01 bpw** — i.e. Plan B *increases* per-rank bytes by ~40%
in a situation where fitting the 115 GB wired limit is already the binding
constraint and already requires streaming. Plan B trades a *diagnosed, bounded*
kernel-performance problem for a *larger, unmeasured* memory problem.
Sustained SSD read measured on node 1: **6.5 GB/s cold, 21 GB/s warm** — that
number is what sizes Plan B's feasibility and has not yet been plugged in.

## Recommended next steps (in order)

1. **Size Plan B before choosing it.** Instrument per-token top-6 expert
   indices on V4-Flash over a realistic prompt mix; compute LRU hit rate and
   **cold-miss bytes/token** at both budgets (Plan A ~95 GB of 112 resident;
   Plan B ~95 GB of 150+). Cold-miss bytes/token ÷ 6.5 GB/s gives Plan B's
   decode ceiling. If that ceiling is below ~25 tok/s, **Plan B is dead by
   construction** and EXL3 retuning is the only path — proven, not assumed.
2. **Bounded EXL3 retune (the plan's own instruction on a decode miss).**
   Port the `_decode_fused2` prologue to chunk hidden > 512 (kills the 9.5–12x
   verify hole), then sweep per-expert parallelism / tile width on the R=1
   GEMV. Re-measure against these same gates.
3. **Only if (2) stalls and (1) shows a viable ceiling**, switch — and prefer a
   **hot/cold format hybrid** (resident experts MXFP4, streamed experts EXL3)
   over all-MXFP4, since the streamed bytes' kernel cost is hidden behind the
   I/O that fetches them anyway, and this preserves most of the 3.01 bpw memory
   win.
4. **Do not drop MTP to make the budget work.** The verify-shaped numbers here
   make MTP look expensive, but the looped-R=1 proxy (1.07 ms/token at R=4) shows
   the per-token MTP *shape* is competitive once the `_prefill` fallthrough is
   fixed. The user's constraint stands on its own: MTP stays.

## Reproduction

Scripts (all in `raw/`): `p2_final_check.py` (main, corrected baseline),
`p2_valid_check.py` (row-loop workaround + mxfp8 control),
`p2_prefill_check.py` (prefill rows, `EXL3_MM_MAX_ROWS` set before import),
`p2_amort_check.py` (**retained as a negative example** — its timing method is
invalid on lazy MLX), `p2_weight_error_probe.py` (weight-level error).

Run as: `BENCH_LAYER=<n> BENCH_EXPERTS=<n> ~/phase1-exl3/.venv/bin/python <script>`
on node 1. Environment needed: `~/phase1-exl3` venv (PonyExl3 0.3.0 from
source, mlx 0.32.2).

## Chip note

The cluster nodes are **Apple M4 Max** Mac Studios (16 cores, 128 GB unified
each — verified on both machines 2026-09-27 via
`sysctl machdep.cpu.brand_string`). Older docs in this repo say "M4 Ultra";
that part does not exist in the M4 generation, and those docs are wrong.

---

## CORRECTION (2026-09-28) — the retune was done, and it worked

The section "Cross-check: is the 3.3x a hardware floor or a tuning problem?"
closed with *"This retuning has not been attempted."* That was accurate when
written; it is no longer. Later the same day the **v3 chunked-prologue** kernel
was written and measured at the real shape (E=384, D=5120, H=2304,
`_v2_ok()=True` — the pathological `_prefill` fallthrough is gone).

Patch: `~/exl3-moe-v3-chunked-prologue.patch` on node 1, applied to
`~/repos/ref/PonyExl3/ponyexl3/mlx/exl3_moe.py`.

| shape | v2 ms | v2 ratio | v3 ms | v3 ratio | speedup |
|---|---|---|---|---|---|
| decode R=1 | 1.305 | 3.38 | 1.029 | **2.65** | 1.27x |
| verify R=4 (DSpark) | 12.251 | 11.95 | 3.171 | **3.04** | **3.86x** |
| verify R=8 | 19.079 | 9.78 | 6.050 | **3.11** | 3.15x |

Evidence: `~/p2b-v3.log` (E=128) and `~/p2c-v3-e384.log` (E=384) on node 1;
also `~/p2-layer20.log` for the tile/threads sweep (R=1 1.019 ms best).

**The 9.5-12x verify hole that dominated this document is CLOSED to ~3.0x.**
What remains is a ~2.65-3.1x gap against production MXFP4 at decode/verify,
plus prefill R=512 now at 3.10x at E=384 (worse than the 1.98x measured at
E=128, but E=384 is the real expert count and costs more at prefill in *both*
formats — not a like-for-like comparison).

Consequence for the choice this document frames: with verify at ~3.0x and the
experts being the only format that fits resident (see
`../phase5-planb-sizing-2026-09-28/`), EXL3 is no longer a memory-only bet with
a broken speed story. The remaining decision is between two *unbuilt* paths —
streaming Plan B vs porting EXL3 kernels into exo — not one working path against
one broken one.
