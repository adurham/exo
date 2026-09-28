# Phase 13 -- deepseek_v41 model file + EXL3 builder (plan phase 3, first cut)

Date: 2026-09-28
Host: gateway (build) + macstudio-m4-1 (gate; M4 Max, mlx 0.32.2, py3.14,
`~/phase1-exl3` venv, live cluster untouched)
Status: **model package lands and matches the reference forward.** Not yet wired
into `mlx_lm.load` or exo.

## Bottom line

The V4.1 text stack now exists as a real package in the serving fork,
`mlx_lm.models.deepseek_v41` (`adurham/mlx-lm` branch `feat/dsv41-exl3`,
commit `bd1bfd1`, signed). It builds straight from the EXL3 checkpoint and runs
the full 40-layer forward on the fast EXL3 kernels. Checked against the p30/p35
reference forward on the same 531-token prompt:

- **Each layer on its own matches the reference**: cosine >= 0.99983 on all 40
  layers (plan bar: 0.999).
- **End result matches**: teacher-forced NLL **1.0030**, top-1 **78.3%** --
  the reference was 1.003 / 78.3%.

## What was built

- Vendored the PipeNetwork port's model modules (config, layers, hc, fakequant,
  sparse attention, compressor, indexer, attention, engram, moe, cache, model,
  mtp, dequant) with relative imports + Apache-2.0 LICENSE + README.
- Three small local changes: MoE `group` hook (all_sum of routed partials
  before the replicated shared expert -- the phase-12 TP convention); `wo_a`
  may be a grouped module; compressor `wkv`/`wgate` may be quantized modules.
  The original `nn.Linear` paths are unchanged.
- New `exl3_build.py`: `build_block` / `build_model`. `EXL3Linear` for every
  trellis group (incl. the 8 `wo_a` slices and the 6-bit head), stacked
  `EXL3SwitchGLU` experts via the phase-12 loader (optional rank slice),
  engram rows read on demand from the native release, strict accounting
  (every param from the checkpoint; only `gate.bias_vl` is allowed absent --
  the text runtime never selects it).

## Gate (p46)

`scripts/p46_model_gate.py`, layer-major (build one block, run, free):

| mode | worst per-layer cos | end NLL mean / median | top-1 |
|---|---:|---:|---:|
| isolated (each layer fed the reference input) | **0.99983** (L18) | **1.0030 / 0.0288** | **78.3%** |
| chained (own output feeds the next layer) | 0.8227 (L39); 0.98 by L17 | 1.0231 / 0.0325 | 76.8% |
| p35 reference | -- | 1.003 / 0.029 | 78.3% |

Reading the chained row: the reference ran reconstructed-bf16 weights through
plain matmuls; this path runs the fp16 EXL3 kernels. Small per-layer rounding
differences compound through 40 layers of hyper-connection residuals (L39's
low cos is on its rms-36 outlier channel). The isolated run -- every layer
exact on the reference input -- rules out a wiring error, and the end-quality
cost is small (NLL +0.02, top-1 -1.5 pts). Still, it is a real numerics
difference vs the bf16 reference, not zero.

Wall clock: per-layer forward 0.2-0.3 s for 531 tokens (layer-major, not a
serving number); build 1-1.6 s/layer; peak 11.3 GB with one layer resident.

## Next

1. `mlx_lm.load` entry (model_type `deepseek_v41`, `quant_method: exl3`) and
   a decode loop over the full resident model (single node first -- 105 GB/rank
   won't fit one node at world=1, so test with TP=2 or a layer subset).
2. exo integration: auto_parallel branch using `world=2` + `MoE.group`, model
   card, generator; MTP head through `EXL3Linear` (mtp_bits=4).
3. First real end-to-end tok/s.

## Artifacts

- `raw/p46-isolated.log`, `raw/p46-chained.log`
- `scripts/p46_model_gate.py`
- Fork: `adurham/mlx-lm` `feat/dsv41-exl3` @ `bd1bfd1`. exo's recorded
  mlx-lm pin still intentionally `5c5328b` (inert until exo imports it).
