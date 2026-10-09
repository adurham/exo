# Q1 — EXL3-fused dense kernel vs NATIVE mx.quantized_matmul (pricing)

**Scope:** measurement only. Nothing re-quantized, live model untouched, no POST, no deploy.
**Node/date:** studio2 (m4-2, rank0), 2026-10-09. mlx `0.32.3.dev20260918+603f16eb7`, production venv python.
**Shapes:** layer-20 dense roster (18 linears), **TP=2 rank-0 sharded** (matches production per-rank reads).
**Weights:** REAL — decoded from the EXL3 2.9bpw checkpoint via `decode_full_mlx`; re-quantized in-memory with `mx.quantize(W, group_size=64, bits=B)`.
**Timing:** whole-slice chained into ONE lazy graph + ONE `mx.eval`; and K-batched (K=8) per-call p95. warm≥3, reps≥9.

## Rate table (ms per dense layer, per-rank; 18-linears whole-slice)

| arm | m=1 whole | m=1 K/call med | m=4 whole | m=4 K/call med | m=4 K/call p95 | GB/s (bytes_moved @ m4 whole) |
|---|---|---|---|---|---|---|
| **EXL3 prod (fused, Sharded)** | 1.165 | 0.569 | 0.945 | 0.774 | 0.780 | 70.7 (trellis 66.7 MB) |
| native q4 (raw qmm, no Had) | 0.298 | 0.159 | 0.374 | 0.229 | 0.240 | 165.7 (q4 62.0 MB) |
| native q4 **+Hadamard** (drop-in) | 0.409 | 0.217 | 0.483 | 0.314 | 0.317 | 128.4 |
| native q5 (raw) | 0.366 | 0.179 | 0.375 | 0.249 | 0.250 | 202.3 (q5 75.8 MB) |
| native q5 +Hadamard | 0.403 | 0.247 | 0.482 | 0.338 | 0.341 | 157.2 |
| native q6 (raw) | 0.366 | 0.202 | 0.375 | 0.251 | 0.252 | 238.9 (q6 89.6 MB) |
| native q6 +Hadamard | 0.482 | 0.277 | 0.482 | 0.336 | 0.339 | 185.8 |

- **"raw qmm"** = a bare `mx.quantized_matmul` swap, *without* the EXL3 pre/post Hadamard rotations → not bit-comparable to EXL3 output (it drops the input rotation), so its cosine is only indicative. Shown because it was the requested arm.
- **"+Hadamard"** = the fair drop-in: `prepare_xh → qmm → finish_y` keeps EXL3's rotations → this is the realistic alternative path.
- All shapes have `in_features % 64 == 0`; **no shapes skipped** (`skipped: []`).

## Rate ratio vs EXL3 (m=4) and projected ms/round (40 dense layers × 1 m=4 pass)

| arm | ratio (m4, K/call) | ratio (m4, whole) | projected ms/round (whole×40) |
|---|---|---|---|
| EXL3 prod | 1.00× | 1.00× | **37.8 ms** (m1: 46.6 ms) |
| native q4 raw | **3.38×** | 2.52× | 15.0 ms |
| native q4 +Hadamard | **2.47×** | 1.95× | 19.3 ms |
| native q5 raw | 3.11× | 2.52× | 15.0 ms |
| native q5 +Hadamard | 2.29× | 1.96× | 19.3 ms |
| native q6 raw | 3.08× | 2.52× | 15.0 ms |
| native q6 +Hadamard | 2.30× | 1.96× | 19.3 ms |

- EXL3 dense slice measured here = **37.8 ms/round** at m=4 (whole-graph, sharded). The older "~51 ms/round" closure figure was on unsharded shapes / heavier load; the sharded per-rank reality is lower.
- Native-q4 drop-in projects **~19 ms/round**, i.e. **~18.5 ms/round** less than EXL3 on the dense slice.
- Trellis-equivalent GB/s (bytes-moved/ms) is ~2.5–3× higher for qN because q4 streams 62 MB vs EXL3's 66.7 MB trellis (not fewer bytes) — the win is **ALU/issue**, not bandwidth: the fused trellis decode (k=7 SWAR) is the cost, not the read.

## Cosine fidelity — fidelity to **bf16 EXL3-decoded W**, NOT model quality

| bits | cos (m=1) | cos (m=4) |
|---|---|---|
| q4 | 0.99589 | 0.99588 |
| q5 | 0.99903 | 0.99903 |
| q6 | 0.99976 | 0.99976 |

> A qN re-quant of an **already-2.9bpw-lossy** tensor measures fidelity to EXL3, not to the original model. Real end-task quality requires the **original** weights and is **UNMEASURED**. These cosines are a numerical-fidelity read only.

## OWNER-DECISION — TRIGGERED ✅

The qN rate at m=4 is **≥2×** the EXL3 rate: **3.38×** (raw q4) / **2.47×** (Hadamard-kept q4), and ≥2× for q5/q6 as well.

**A quant-format change (EXL3 2.9bpw trellis → native q4/q5 group_size=64) is QUALITY-GATED — NOT to be acted on from this measurement.** This prices the upside; the gate is real-quality validation against the original model plus a full-model two-node run (the row-limit / GPU-timeout history in `exl3_linear.py` shows layer-level wins have not always survived full-depth). Recommend only as a tracked experiment.

## Experts (optional arm) — PARKED

Pricing the MoE expert path was in budget only if cheap (<15 min): it is not. Layer-20 has **384 experts/layer**, each ~71 MB decoded fp16 (w1+w3+w2) → materializing one layer's experts for re-quantization is ~27 GB and the native path is `gather_qmm` on a stacked layout, a different kernel from what we priced. **Parked, not measured.**

## Artifacts
- Script: `bench/p20_pricing_q1_dense.py`
- Raw JSON: `docs/benchmarks/phase20-throughput/raw/pricing/q1/p20_pricing_q1_dense_layer20.json`
- Stdout: `docs/benchmarks/phase20-throughput/raw/pricing/q1/p20_pricing_q1_dense_layer20.stdout.txt`
