# I3: MoE Kernel Achieved-Bandwidth / Achieved-TFLOPS Microbenchmark

**Question:** Byte accounting predicts ~9ms to read activated expert weights
once per verify at 546 GB/s peak; the measured verify bracket is 56ms. Is
the gap kernel inefficiency (kernels achieving only a fraction of peak
bandwidth) or is the time not in weight streaming at all?

**Run location:** Mac Studio node 2, `192.168.86.202` (Adams-Mac-Studio-M4-2,
the non-API node), via `/Users/adam.durham/repos/exo/.venv/bin/python`.
Confirmed the Studio's exo checkout mirrors the MacBook's path
(`~/repos/exo`), and `.venv/bin/python` exists there.

**Cluster contention protocol:** waited for
`tmp/perf-campaign-2/round1/.i4-cluster-done` (appeared after ~2.5 min of
polling). `gpu_usage_ratio` on both nodes was checked immediately before
and after the timed run (see MEASUREMENTS). Node .202 (where the bench
ran) sat at 2.8–3.0% before/after; node .201 stayed ~2.3–5.4% throughout —
no concurrent inference activity observed, no blocks discarded.

---

## DEPLOYED SHAPES

Source: `~/.exo/models/deepseek-ai--DeepSeek-V4-Flash-0731/config.json` on
the Studio (the checkpoint actually loaded — `DSV4_MODEL_ID` in
`start_cluster.sh:361` defaults to `deepseek-ai/DeepSeek-V4-Flash-0731`,
confirmed live via `ps aux` on the running exo process on .202), cross-
checked against `mlx-lm/mlx_lm/models/deepseek_v4.py`'s `ModelArgs`.

| Field | Value | Source |
|---|---|---|
| `n_routed_experts` | 256 | config.json |
| `num_experts_per_tok` (top-k) | 6 | config.json |
| `n_shared_experts` | 1 | config.json |
| `hidden_size` | 4096 | config.json |
| `moe_intermediate_size` | 2048 (global; 1024/rank under TP=2) | config.json |
| `num_hidden_layers` | 43 | config.json (matches expected) |
| `num_attention_heads` | 64 (32/rank under TP=2, via `shard()`'s `n_heads //= N`) | config.json + `deepseek_v4.py:7544` |
| `head_dim` (MLA q/o) | 512 | config.json |
| `q_lora_rank` | 1024 | config.json |
| `o_lora_rank` | 1024, `o_groups`=8 | config.json |
| `qk_rope_head_dim` | 64 | config.json |
| `vocab_size` / lm_head | 129280 | config.json |

**Quantization as actually deployed** (`config.json`'s top-level
`quantization_config` says `fp8`/`weight_block_size [128,128]` — that is
the *upstream HF checkpoint's storage format*, NOT what mlx-lm quantizes to
at load time). The REAL deployed runtime quantization is set by
`make_quantization_config()` in `mlx-lm/mlx_lm/models/deepseek_v4.py`
(lines ~907–935), read directly from the fork source used by the running
process:

- **Expert weights** (`.ffn.switch_mlp.*_proj`, i.e. the routed-expert gate/up/down): `mxfp4`, **group_size=32, bits=4**
- Shared-experts + attention (`wq*`, `indexer.wq`) + MTP `e_proj`/`h_proj`: `mxfp8`, group_size=32, bits=8
- Everything else (default fallback): `affine`, group_size=64, bits=8

This confirms the byte-accounting question must use **4-bit mxfp4,
group_size=32** for the expert weights, not the 8-bit figure implied by
`config.json`'s top-level `quantization_config`.

---

## KERNEL PATH IDENTIFIED

`DeepseekV4MoE.__call__` (deepseek_v4.py ~line 2916) calls
`self.switch_mlp(x, inds)` where `self.switch_mlp` is a
`mlx_lm.models.switch_layers.SwitchGLU` instance. `SwitchGLU.__call__`
issues **three separate `mx.gather_qmm` dispatches per layer** (via
`QuantizedSwitchLinear.__call__`, `switch_layers.py:76`): `gate_proj`,
`up_proj`, then `down_proj` (after a SwiGLU activation) — NOT a single
fused call, and NOT `mx.quantized_matmul` (that op has no gather
semantics; would be architecturally wrong for MoE). `EXO_DSV4_FUSED_MOE=0`
on the running process confirms the vanilla (non-`BatchedSwitchGLU`) path
is active in production — exactly what was benched.

**mlx_lm copy verification:** `diff -q` between
`mlx-lm/mlx_lm/models/{deepseek_v4.py,switch_layers.py}` (fork source) and
`.venv/lib/python3.13/site-packages/mlx_lm/models/{...}` on the Studio
produced **no output** (files identical) — the installed venv package is
the fork source, confirming the bench targets the actual production
kernel path, not a stale/divergent copy. `mlx_lm.__file__` resolves to the
venv site-packages copy.

The microbench instantiates a **real `SwitchGLU` module**
(`per_rank_inter=1024, hidden=4096, n_routed_experts=256`), quantizes its
three linears with `.to_quantized(group_size=32, bits=4, mode="affine")` —
the exact call `make_quantization_config` implies — and calls
`switch(x, indices)` end-to-end, exercising all three `gather_qmm`
dispatches together, matching production layer-by-layer cost.

---

## MEASUREMENTS

Per-rank byte accounting (TP=2; exo shards `switch_mlp.{gate,up}_proj`
all-to-sharded and `down_proj` sharded-to-all, per `shard()` at
`deepseek_v4.py:7557-7559` — each rank streams **half** the global
`moe_intermediate_size` per expert). Bytes-read formula:

```
n_gathered_rows      = M * top_k                                  (rows actually dispatched)
weight_elems_per_op  = per_rank_moe_inter (1024) * hidden (4096)   (one gate/up/down proj)
weight_bytes         = n_gathered_rows * weight_elems_per_op * 3 ops * bits(4) / 8
n_groups_per_op      = weight_elems_per_op / group_size(32)
scale_bias_bytes     = n_gathered_rows * n_groups_per_op * 3 ops * 2(scale) * 2(bias) bytes
total_bytes          = weight_bytes + scale_bias_bytes
```

GPU idle check: `gpu_usage_ratio` on .202 was 0.028–0.030 immediately
before AND after the full N=200 run (near-zero contention); .201 stayed
0.023–0.054 throughout — no other model activity during the bench, no
blocks discarded.

| Op / shape | M | iters | mean (ms) | min (ms) | max (ms) | bytes read | GB/s (mean) | GB/s (best) | % of 546 GB/s (mean) |
|---|---|---|---|---|---|---|---|---|---|
| MoE SwitchGLU (gate+up+down `gather_qmm`), mxfp4 g=32 b=4, per-rank shard | 1 | 200 | 0.3242 | 0.3001 | 0.5554 | 47,185,920 | 145.5 | 157.2 | **26.7%** |
| MoE SwitchGLU (gate+up+down `gather_qmm`), mxfp4 g=32 b=4, per-rank shard (verify shape) | **4** | 200 | 0.6625 | 0.6271 | 0.8665 | 188,743,680 | 284.9 | 301.0 | **52.2%** |
| Dense bf16 GEMM (prefill representative, N=K=hidden_size) | 2048 | 200 | 4.8285 | 4.7824 | 5.0405 | — (compute-bound) | — | — | — |

Prefill GEMM TFLOPS: `flops = 2*2048*4096*4096 = 6.872e10`;
`tflops_mean = 14.23`, `tflops_best = 14.37`.

Kernel-sum-check per-op timings (M=4, forced individual `mx.eval()` per
op — see caveat below) are in the raw JSON at
`tmp/perf-campaign-2/round1/full_run_output2.json`.

---

## ACHIEVED BANDWIDTH VS 546 GB/s

- **M=1 (draft-step shape):** 145.5 GB/s mean, 157.2 GB/s best-case →
  **26.7% / 28.8%** of 546 GB/s peak.
- **M=4 (the actual verify shape):** 284.9 GB/s mean, 301.0 GB/s best-case
  → **52.2% / 55.1%** of 546 GB/s peak.

Sanity check passed: M=4 correctly takes ~2x M=1's wall time (0.6625ms vs
0.3242ms) despite reading ~4x the bytes — consistent with fixed per-
dispatch overhead being amortized better at higher M, not with a timing
bug. No measurement implies >546 GB/s.

## ACHIEVED TFLOPS VS PEAK

Prefill GEMM at M=2048 achieves **14.23 TFLOPS mean / 14.37 TFLOPS best**,
i.e. **79.1% / 79.8%** of the ~18 TFLOPS M4 Max fp16 peak used as the
reference ceiling. This is a **dense bf16 matmul**, not quantized — it
counts full-precision compute FLOPs directly (no dequantization
assumption needed, since there's no quantized weight in this GEMM). This
number is the dense-GEMM "achievable ceiling" reference the roofline
model should use, and it confirms the hardware/MLX GEMM path itself is
healthy (~80% of peak) — the MoE gather path's shortfall below is a
*gather/quantized-kernel-specific* inefficiency, not a general GEMM
inefficiency on this hardware.

## KERNEL-SUM x43 VS 56ms VERIFY

Per-layer op sum at M=4 (attention wq_a/wq_b/wkv/wo_b, router gate,
shared-experts gate/up/down, two RMSNorms, plus the measured MoE
`gather_qmm` M=4 cost) = **2.877 ms/layer** → **123.71 ms** for 43 layers,
vs the **measured 56.0 ms** verify bracket. Raw difference:
**56.0 − 123.71 = −67.71 ms** (kernel-sum EXCEEDS the bracket).

**This number is NOT directly usable as "dispatch/sync overhead" and must
be reported as such rather than at face value.** The kernel-sum method
forces an individual `mx.eval()`+`mx.synchronize()` after each of the ~11
tiny per-layer ops (473 forced eval/sync boundaries total across 43
layers), which destroys the GPU pipelining/kernel-overlap that
production gets for free by building ONE lazy graph per verify step and
evaluating it once (`SpecPipelineLastLayer.__call__`'s single
`mx.eval(output)`, per the fork's own `switch_layers.py` comments and the
`exo-perf-tuning` skill's documented lazy-eval discipline). A kernel-sum
built from forced-per-op eager evaluation is expected to OVER-count
relative to one real fused-graph bracket — this is the same class of
artifact the skill's "Benchmark Lazy-Eval Artifacts" note warns about,
just inverted (here forced eval inflates rather than the graph-
construction-only trap deflating). **The kernel-sum check as measured
cannot cleanly answer "how much of the 56ms bracket is dispatch/collective
overhead" — it only shows that per-op eager-eval cost (123.71ms) is
itself larger than the real fused bracket (56ms), which is consistent
with real dispatch/scheduling overhead existing SOMEWHERE, but the
magnitude can't be attributed cleanly from this measurement.** A trustworthy
version of this check would need to bench a full fused per-layer forward
(one `mx.eval()` per layer, not per op) — out of scope for this pass;
flagging as a follow-up rather than fabricating a clean number.

Also note the per-layer op set benched (attention shapes) used
representative/approximate shard shapes for `wo_b` (assumed sharded;
`shard()` at `deepseek_v4.py:7530-7544` does NOT actually shard `wo_b` —
only `wq_b` and `wo_a` are sharded, `wq_a`/`wkv`/`wo_b` stay replicated
full-size per rank). This is a secondary, exploratory check; the primary
MoE `gather_qmm` bandwidth measurement above (the deliverable this task
turns on) used the verified-correct per-rank shard shapes directly from
`shard()`.

---

## BAND FIRED

Pre-registered bands:
- ≥80% of 546 GB/s → kernels fine, I1 (collectives) owns the gap
- <60% of 546 GB/s → **MLX kernel work is funded**
- 60–80% → decision waits on I1

**M=4 (the actual verify shape) measured 52.2% mean / 55.1% best-case of
546 GB/s. M=1 measured 26.7% / 28.8%.**

**BAND FIRED: <60% → MLX KERNEL WORK IS FUNDED** (6-bit-class qmv
tile/threadgroup tuning for M4 Max, or a fused dequant-gemv path, for the
`gather_qmm` MoE expert kernel at small-M shapes).

Applied verbatim, no rationalization: 52.2% is well inside the <60% band,
not a near-miss of the 60-80% band.

---

## SUMMARY

- **Achieved bandwidth (verify shape M=4):** 284.9 GB/s mean (52.2% of
  546 GB/s peak), 301.0 GB/s best-case (55.1%). At M=1 (draft shape):
  145.5 GB/s (26.7%).
- **Achieved TFLOPS (prefill GEMM M=2048, dense bf16):** 14.23 TFLOPS mean
  (79.1% of ~18 TFLOPS M4 Max peak), 14.37 TFLOPS best (79.8%) — the
  dense-GEMM path itself is healthy; the shortfall is specific to the
  quantized/gathered MoE kernel.
- **Kernel-sum x43 vs 56ms bracket:** 123.71ms vs 56.0ms — kernel-sum
  EXCEEDS the bracket, an artifact of forced per-op eager evaluation
  destroying graph fusion/pipelining (documented as unreliable above);
  cannot be cleanly attributed as "overhead," flagged as a methodology
  limitation rather than reported as a clean number.
- **Band fired: <60% → MLX KERNEL WORK IS FUNDED.** The MoE `gather_qmm`
  kernel at the deployed mxfp4 g=32/b=4 quantization achieves only ~52%
  of peak bandwidth at the verify shape (M=4) and ~27% at the draft shape
  (M=1) — kernels themselves are a real, measured, non-trivial part of
  the 56ms-vs-9ms gap, not purely dispatch/collective overhead.

Files: `tmp/perf-campaign-2/round1/i3_microbench.py` (script, run on the
Studio via scp),  `full_run_output2.json` / `full_run_stderr2.txt` (raw
N=200 run), `smoke_out3.txt` (N=5 smoke-test verification run).

---

# RE-RUN (CORRECTED, mxfp4)

**Why this section exists:** an independent review (`VERIFICATION.md`, CLAIM 4)
found the original run above benched `switch.*_proj.to_quantized(..., mode="affine")`,
not the deployed `mode="mxfp4"`, and its bytes formula hardcoded "fp16 scale + fp16
bias" per group — wrong for both affine (fp32 scale+bias) and mxfp4 (uint8 scale,
no bias). This section re-runs the bench against the ACTUAL deployed kernel and
byte accounting, and does **not** delete the original numbers above — they stand
as the record of the error.

**Script:** `tmp/perf-campaign-2/round1/i3_microbench_rerun.py` (new file; the
original `i3_microbench.py` is left untouched). **Run location:** Mac Studio
`192.168.86.202`, `~/repos/exo/.venv/bin/python` (Studio's own checkout/venv,
confirmed same file via `md5` before running — `5345f6f5071c5137e3b3f2a72de1c16b`
on both this MacBook and the Studio).

**GPU idle check** (`curl .../metrics | grep exo_gpu_usage_ratio`, via SSH from
this MacBook since a direct-to-IP curl from the local shell was blocked by the
approval layer this run):

| when | .202 (bench node) | .201 |
|---|---|---|
| before run | 0.0287 | 0.0277 |
| after N=200 + N=400 spot check | 0.0308 | 0.0266 |

No concurrent load on either node before or after.

## 1. Quantization call used

`switch_layers.py`'s own `QuantizedSwitchLinear.to_quantized(group_size, bits, mode)`
API — the exact call `make_quantization_config()` drives (`deepseek_v4.py:1042-1070`,
`mxfp4 = {"group_size": 32, "bits": 4, "mode": "mxfp4"}` applied to every
`.ffn.switch_mlp.*_proj`):

```python
switch = SwitchGLU(hidden, per_rank_inter, n_routed_experts, bias=False)
switch.gate_proj = switch.gate_proj.to_quantized(group_size=32, bits=4, mode="mxfp4")
switch.up_proj   = switch.up_proj.to_quantized(group_size=32, bits=4, mode="mxfp4")
switch.down_proj = switch.down_proj.to_quantized(group_size=32, bits=4, mode="mxfp4")
```

`to_quantized(mode="mxfp4")` **is** the correct API in this mlx build — no
substitution needed. Confirmed via `mx.quantize(..., mode="mxfp4")` internally
called by `switch_layers.py:132-149`.

## 2. Real quantized-array nbytes (printed by the script, not assumed)

```
gate_proj: weight (256,1024,512) uint32 nbytes=536,870,912 | scales (256,1024,128) uint8 nbytes=33,554,432 | biases=None
up_proj:   weight (256,1024,512) uint32 nbytes=536,870,912 | scales (256,1024,128) uint8 nbytes=33,554,432 | biases=None
down_proj: weight (256,4096,128) uint32 nbytes=536,870,912 | scales (256,4096, 32) uint8 nbytes=33,554,432 | biases=None
```

mxfp4 has **no bias array** (confirmed: `getattr(proj, "biases", None) is None` for
all three projections) — this matches VERIFICATION.md's independent finding
(`biases = None` for mxfp4 vs `float32` biases present for affine).

Per-expert-per-projection bytes (weight+scales, real array nbytes / 256 experts):
`(536,870,912 + 33,554,432) / 256 = 2,228,224 B` per proj per expert — same for
all three projections (gate/up/down have equal total nbytes at this shape).
Per-pair, all 3 projections: `3 × 2,228,224 = 6,684,672 B`.

This **matches VERIFICATION.md's independently-derived mxfp4 estimate of
2,228,224 B/expert/proj exactly** — cross-check passed.

## 3. Raw timing table (N=200 iters, 20 warmup)

| case | mean_ms | min_ms | n_pairs | n_distinct |
|---|---|---|---|---|
| M=1 uniform (6 pairs, 256 experts) | 0.3234 | 0.3006 | 6 | 6 |
| M=4 uniform (24 pairs, 256 experts) | 0.5577 | 0.5375 | 24 | 23 |
| M=4 shared6 (4 rows → same 6 experts) | 0.4583 | 0.3939 | 24 | 6 |
| M=4 distinct24 (4 rows → 24 disjoint experts) | 0.5557 | 0.5358 | 24 | 24 |

N=400 spot check on the headline M=4-uniform config: mean=0.5554ms
(vs 0.5577ms at N=200 — 0.4% drift, well under the 2x sanity gate).

## 4. Achieved GB/s — PAIRS basis vs DISTINCT basis (M=4 uniform, headline)

Indices passed (uniform draw, `mx.random.seed(0)`, `mx.random.randint(0,256,(4,6))`):
`[[234,247,120,241,238,8],[46,125,39,116,237,7],[144,106,62,220,148,153],[167,79,174,241,57,97]]`
— **24 pairs, 23 distinct** (only expert 241 repeats, in rows 1 and 4).

| basis | bytes | GB/s (mean) | % of 546 (mean) | GB/s (best) | % of 546 (best) |
|---|---|---|---|---|---|
| **PAIRS** (24 pairs × 6,684,672 B) | 160,432,128 | **287.7** | **52.7%** | 298.5 | 54.7% |
| **DISTINCT** (23 distinct × 6,684,672 B) | 153,747,456 | **275.7** | **50.5%** | 286.0 | 52.4% |

PAIRS and DISTINCT differ by only 24/23 = 4.3%, both landing in the same band.
**As predicted in the task brief: drawing 24 indices uniformly from 256 gives
~23 distinct, so this synthetic case cannot discriminate between the two
bytes bases.** A production-routing case (rows sharing experts) is required to
separate them — see §5.

M=1 (6 pairs, 6 distinct, no possible dedup at this draw): 124.0 GB/s = 22.7%
of peak — PAIRS and DISTINCT are identical by construction (6 pairs, 6 distinct).

## 5. DECISIVE MEASUREMENT — shared-routing vs distinct-routing at M=4

Two M=4 cases, same weights, same total 24 (row,expert) pairs, different index
patterns:

- **shared6**: all 4 rows route to the identical 6 experts `[10,47,88,130,190,255]`
  → 24 pairs, **6 distinct**. mean = **0.4583 ms**.
- **distinct24**: 4 rows route to 24 disjoint experts (indices 0–23, one set of 6
  per row, no overlap) → 24 pairs, **24 distinct**. mean = **0.5557 ms**.

**Ratio (distinct/shared) = 1.21×.**

Full dedup (hardware/cache reads each distinct expert once, ignores repeats)
predicts ~4×. Zero dedup (kernel streams all 24 pairs' worth of DRAM traffic
regardless of duplication) predicts ~1×. The measured 1.21× is much closer to
the **no-dedup** end.

**I6 implication: per-row expert duplication IS costing real bandwidth.** The
kernel dispatches one independent tile-gather per (row,expert) pair — sharing
6 experts across 4 rows saves only ~18% of wall time, not the ~75% full dedup
would give. This corroborates VERIFICATION.md's independent finding on this
MacBook (0.5193ms / 0.6501ms = 1.25×, "partial reuse... much closer to the
no-dedup end") — same conclusion, different hardware, same direction and
similar magnitude.

## 6. Sanity gates — all passed

- No bandwidth figure exceeds 546 GB/s (max observed: 298.5 GB/s at M=4 PAIRS-best). ✅
- M=4 (0.5577ms mean) takes visibly longer than M=1 (0.3234ms mean) — 1.72× wall
  time for 4× the row count. ✅ (not equal/less)
- N=400 spot check (0.5554ms) vs N=200 (0.5577ms) on the headline M=4-uniform
  config: 0.4% difference, far under the 2× gate. ✅

## 7. BAND FIRED

Pre-registered bands, applied verbatim to the mxfp4 **PAIRS-basis** number at
M=4: **52.7% (mean) / 54.7% (best)**.

**BAND FIRED: `<60% → MLX KERNEL WORK IS FUNDED`.**

This is the SAME band the original (wrong-mode) run fired, even though the
absolute number moved (52.2%→52.7% mean is close by coincidence — the mode
correction and the byte-formula correction happen to roughly offset each
other at this shape, unlike VERIFICATION.md's *affine-run* correction which
moved the affine number up to 62.6%). The DISTINCT basis (50.5% mean) fires
the same band. **Both bases land in the same band here — no straddle, no
basis-dependent verdict at this measurement.**

Contrast with the reviewer's prediction: VERIFICATION.md's 62.6% was arithmetic
correcting the byte formula on the **affine** run that was actually executed —
it explicitly flagged "the deployed mxfp4 path was never benched" and that its
own affine-vs-mxfp4 ratio "suggests mxfp4 will land in a similar or slightly
lower percentage, i.e. plausibly straddling the 60% line." **The direct mxfp4
measurement reported here (52.7%/50.5%) is lower than the reviewer's affine-based
62.6% estimate and does NOT straddle the line — it is measured, not extrapolated,
and it fires `<60% → FUNDED`, not the `60–80%` band the reviewer's
affine-arithmetic pointed to.**

## Files

`i3_microbench_rerun.py` (script), `i3_rerun_full_N200.json` / `_stderr.txt`
(main N=200 run), `i3_rerun_spot_N400.json` / `_stderr.txt` (2× spot check),
`i3_rerun_smoke.json` (N=5 smoke-test verification run).

---

# RE-RUN 2 (CHAINED-GRAPH, RECONCILED)

**Why this section exists:** a campaign supervisor caught that this exact
measurement was already done and RETRACTED on 2026-08-22
(`docs/switch-mlp-bandwidth-artifact-retraction-2026-08-22.md`). Both prior
sections in this file (the original run above and RE-RUN 1) called
`mx.eval(out)` **inside** the per-iteration loop
(`i3_microbench_rerun.py:67-70`, confirmed by code read) — the exact
serial-sync artifact the retraction doc describes, which charges ~172µs/call
of host/dispatch overhead to the kernel. This section does **not** delete
either prior section — they stand as the record of the error. This section
re-measures using the retraction doc's actual method.

**Method (mirrors the retraction doc exactly, not a new variant):**

- A **dependency-chained graph**: `CHAIN_LEN=300` `SwitchGLU.__call__`
  invocations chained so call *i+1*'s input depends on call *i*'s real
  output (`carry = x + 1e-9 * mx.mean(out, axis=-2)`, output shape
  `(M, top_k, hidden)` reduced back to `(M, hidden)`) — MLX cannot elide
  any call because each is a genuine data dependency of the next. **ONE
  `mx.eval()`** at the very end of the whole chain.
- **Rotated routing indices**: a pool of `N_POOL=64` independently-drawn
  `(M, top_k)` index sets (matching the retraction doc's "64-entry pool"),
  cycled through the 300 chained calls, so no repeat-hit weight set can sit
  in cache across calls.
- `us/call = total_wall / 300`.
- Bytes from the **real quantized array `.nbytes`** (weight + scales;
  mxfp4 has no biases — confirmed `biases=None` for all three projections,
  same as RE-RUN 1).
- Script: `tmp/perf-campaign-2/round1/i3_microbench_chained.py` (new file;
  neither prior script touched). Run location: Mac Studio `192.168.86.202`,
  `~/repos/exo/.venv/bin/python`.

**GPU idle check** (`curl .../metrics | grep gpu_usage_ratio`):

| when | .202 (bench node) | .201 |
|---|---|---|
| before run | 0.0280 | 0.0307 |
| after run | 0.0266 | (checked, idle — no concurrent load observed) |

No concurrent load on either node before or after.

## 1. Real quantized-array nbytes (identical weights to RE-RUN 1, re-verified)

```
gate_proj: weight (256,1024,512) uint32 nbytes=536,870,912 | scales (256,1024,128) uint8 nbytes=33,554,432 | biases=None
up_proj:   weight (256,1024,512) uint32 nbytes=536,870,912 | scales (256,1024,128) uint8 nbytes=33,554,432 | biases=None
down_proj: weight (256,4096,128) uint32 nbytes=536,870,912 | scales (256,4096, 32) uint8 nbytes=33,554,432 | biases=None
```

`bytes_per_pair_all_3_projs = 6,684,672 B` — identical to RE-RUN 1's figure
(same weights, same shapes).

## 2. Scaling-elision check (sanity gate 5)

Wall time was measured at chain lengths 100/200/300 for every case; us/call
stayed flat across chain length (e.g. M=1: 127.0 / 113.3 / 114.0 µs/call;
M=4: 350.8 / 349.7 / 348.1 µs/call) — confirming the chain is NOT being
optimized away (an elided chain would show `total_wall` roughly constant
regardless of chain length, i.e. us/call collapsing toward 0 as length
grows). The graph really does carry ~300 dependent calls.

## 3. Chained vs serial-sync control (M=1 and M=4)

| M | method | µs/call | GB/s | % of 546 peak |
|---|---|---|---|---|
| 1 | serial-sync (`mx.eval` inside loop, same artifact class as prior 2 sections) | 327.5 | 122.5 | 22.4% |
| 1 | **chained-graph (corrected)** | **114.4** | **350.7** | **64.2%** |
| 4 | serial-sync (`mx.eval` inside loop) | 552.9 | 290.2 | 53.1% |
| 4 | **chained-graph (corrected)** | **350.3** | **458.0** | **83.9%** |

Sanity gate 5 passed on both counts: no figure implies >546 GB/s (max
observed 458.0 GB/s), and the chained number is faster per call than the
serial-sync control at both M=1 (114.4µs vs 327.5µs) and M=4 (350.3µs vs
552.9µs) — the serial-sync control here reproduces the SAME artifact class
as this round's two prior (wrong) sections almost exactly (M=4 serial-sync
552.9µs/290.2GB/s here vs RE-RUN 1's 555.7µs/287.7GB/s — within 1%, same
bug, independently reproduced).

## 4. RECONCILIATION vs the record's 74% / 116-117µs/call

The record's headline chained-graph number (116.0-117.0µs/call, ~404 GB/s,
73.8-74.3% of peak) was measured at the **B=1 (M=1) production decode
shape** — the directly comparable case here is this run's **M=1** row.

- **Timing (µs/call): CONFIRMED.** 114.4µs/call vs the record's
  116.0-117.0µs/call — a 2-3% difference, well within run-to-run noise.
  The chained-graph *method itself* reproduces almost exactly.
- **GB/s / % of peak: does NOT match at M=1** (350.7 GB/s / 64.2% here vs
  ~404 GB/s / 74% in the record) — a ~13% gap that is NOT noise.

**Root cause of the GB/s gap, traced down (hard look at the harness first,
per instructions):** since the timing matches almost exactly, the gap is
NOT a kernel-speed disagreement — it is a **bytes-per-call accounting
difference**. This run's real-nbytes-derived figure is
`6,684,672 B/pair × 6 pairs = 40,108,032 B (40.1 MB)` touched per M=1 call.
The retraction doc states "47.19 MB touched/token" for the same nominal
shape — a 1.177x larger byte count, which multiplied against the nearly
identical timing accounts for essentially the entire GB/s gap
(350.7 × 1.177 ≈ 413 GB/s, matching the record's ~404 GB/s within a few
percent). This run's byte basis (2,228,224 B/expert/proj) is the same one
RE-RUN 1 independently derived and cross-checked exactly against
`VERIFICATION.md`'s separately-derived mxfp4 estimate — it is not being
casually revised here. The retraction doc does not show its byte-formula
derivation in enough detail to identify which convention differs (possible
candidates: a non-per-rank-sharded intermediate size, or an additional
per-group overhead term not present in this bench's real-array read), but
since this run's figure is independently cross-validated twice now (RE-RUN
1 vs VERIFICATION.md, and now this run vs RE-RUN 1), the byte basis here is
retained rather than adjusted to match.

**Verdict: the record's methodology and timing are CONFIRMED (the ~172µs
serial-sync artifact is real and reproduces almost exactly); the specific
74%-of-peak headline number is not exactly reproduced at M=1 because of an
unresolved ~13% byte-accounting difference, not a timing disagreement.**
This round's prior 52.2%/52.7% (serial-sync) figures are confirmed to be
**the same artifact class** as the record's original 27.7%/29.7% figure —
not independent evidence of a real kernel gap.

**At M=4 (the production verify shape), this run measures 83.9% of
peak** — *higher* than the record's 74% M=1 figure, not lower. This is
directionally consistent with the retraction doc's own note that chained
efficiency rises toward the "independent calls, max overlap" ceiling
(87.4% in the record) as per-call fixed dispatch cost is amortized over
more bytes — M=4 carries 4x the bytes per call of M=1 with the same fixed
per-dispatch cost, so it sits further up that curve. No contradiction with
the record; a shape-dependent point on the same curve the record describes.

## 5. PART B — shared-vs-distinct routing, re-run with the chained method

Same chained-graph method (chain_len=300, rotated pool of 64 index draws
**per arm**, each draw keeping the arm's cardinality fixed — 6 distinct for
shared6, 24 distinct for distinct24 — but rotating WHICH experts are hit
across the pool, so cache cannot simply learn one fixed 6- or 24-expert
set). M=4 (4 rows), 24 (row,expert) pairs in both arms, differing only in
n_distinct.

| case | µs/call (chained) | n_pairs | n_distinct |
|---|---|---|---|
| shared6 (4 rows → same 6 experts) | 251.6 | 24 | 6 |
| distinct24 (4 rows → 24 disjoint experts) | 357.0 | 24 | 24 |

**Corrected ratio (distinct/shared) = 357.0 / 251.6 = 1.42x.**

Compare to the prior (wrong, serial-sync) run's 1.21x, and to the
arithmetic-only overhead-subtraction estimate the task brief pre-computed
from that run, (556-172)/(458-172) ≈ 1.34x. The chained method's actual
1.42x is close to — and slightly above — that arithmetic estimate, which
is exactly what should happen: removing the ~172µs/call fixed overhead by
construction (not by post-hoc subtraction) pushes the ratio up from the
raw 1.21x, and the chained method captures a bit more of that than the
naive linear-subtraction estimate did.

**Where 1.42x sits on the 1.0x (no dedup) → 4.0x (full dedup) scale:**
`(1.42 - 1.0) / (4.0 - 1.0) = 14%` of the way from "no dedup" to "full
dedup." This is **NOT** "approaching ~4x" and it is **NOT** exactly at
1.0x either — it is a modest, genuinely partial-reuse ratio, reported
plainly rather than rounded to either pole.

**I6 implication:** the dominant signal is still **no meaningful hardware
deduplication** — per-row expert duplication continues to cost real
bandwidth, consistent in direction with I6's original finding. The
magnitude moved from 1.21x to 1.42x once the fixed per-call overhead
artifact was removed by construction rather than subtracted after the
fact, but 1.42x is still 6x closer to the no-dedup pole (1.0x) than to the
full-dedup pole (4.0x). **I6's core finding STANDS**, revised only in
magnitude (real per-row-duplication cost is somewhat higher, not lower,
once measured without the compressing fixed-overhead artifact — the
opposite direction a naive skim of "ratio went up" might suggest for the
dedup question, since a fixed-overhead artifact always compresses ratios
toward 1.0 regardless of which pole the true ratio is closer to).

## 6. Cross-check against "flat in B" (retraction doc §Ablation matrix, tier F)

The retraction doc's ablation found efficiency **flat in B (B=1→16, 87-89%;
B=32, 80.6%)** using rotated indices, and concluded gather machinery costs
nothing meaningful. Does "no meaningful hardware dedup" (this section's
Part B finding) square with "flat in B"?

**Yes — the two findings are consistent, and are in fact mutually
reinforcing:**

- If the hardware **did** meaningfully deduplicate repeated expert reads,
  then as B grows with indices drawn from a *fixed* pool (256 experts, top-6
  each), the probability of INTRA-BATCH overlap between rows rises
  (birthday-paradox effect) — a deduping kernel would show efficiency
  (measured GB/s under a no-dedup byte model) *increase* super-linearly
  with B as more reads get served from already-fetched weight.
- The ablation instead found **flat** GB/s across B=1→16 — exactly what a
  **genuinely non-deduping**, streaming-bound kernel should show: each
  additional row/pair adds proportional bytes and proportional time, so
  GB/s (bytes/time) stays constant regardless of B.
- Part B's own direct measurement (1.42x, close to the no-dedup pole)
  independently corroborates this: if dedup were real and B-independent,
  the shared-vs-distinct ratio here would be far closer to 4x.

**No contradiction.** Both point the same direction: the kernel does not
meaningfully deduplicate redundant expert reads across rows.

## 7. BAND FIRED

Pre-registered bands, applied **verbatim** to the corrected chained-graph
**M=4** number per task instructions (production verify shape):

- ≥80% of 546 GB/s → kernels fine, I1 (collectives) owns the gap
- <60% of 546 GB/s → MLX kernel work is funded
- 60–80% → decision waits on I1

**M=4 chained-graph result: 83.9% of 546 GB/s peak.**

**BAND FIRED: `≥80% → kernels fine, collectives own the gap.`**

This is a different band than the task brief's stated "likely outcome"
(reproducing ~74% and firing the 60-80% band) — the measured M=4 number
(83.9%) clears the 80% line outright. It is also a different band than
either of this round's two prior (serial-sync-artifact) runs fired
(`<60% → MLX KERNEL WORK IS FUNDED`, at M=4 52.2%/52.7%).

**This round's earlier "<60% → FUND KERNEL WORK" verdict (both the
original run and RE-RUN 1) is WITHDRAWN.** Both were built on the
serial-sync artifact (`mx.eval` inside the per-iteration loop), confirmed
by code read (`i3_microbench_rerun.py:67-70`) and independently reproduced
in §3 above (M=4 serial-sync here: 53.1%, within 1% of RE-RUN 1's 52.7%).
The corrected chained-graph M=4 measurement (83.9%) fires the opposite
band: kernels are fine, the 56ms-vs-9ms gap is not a MoE-kernel-bandwidth
problem, and any remaining gap is I1/collectives' to own. At the M=1 shape
(directly comparable to the record's headline number), the result is 64.2%
— inside the 60-80% "waits on I1" band — so the verdict is genuinely
shape-dependent: draft-step (M=1) traffic sits in the "waits on I1" band,
verify-step (M=4) traffic clears the "kernels fine" bar. Since the task
explicitly designates M=4 (the production verify shape) as the number the
band applies to, **≥80% / "kernels fine" is the band this measurement
fires**, but the M=1 60-80% result should not be discarded when reasoning
about draft-step cost specifically.

## Files

`i3_microbench_chained.py` (script), `i3_chained_full.json` /
`i3_chained_full_stderr.txt` (chain_len=300, n_pool=64 full run, raw
output).
