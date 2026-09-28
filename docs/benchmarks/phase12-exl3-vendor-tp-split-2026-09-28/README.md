# Phase 12 -- EXL3 kernels vendored into mlx-lm + TP split convention proven

Date: 2026-09-28
Host: gateway (vendor build) + macstudio-m4-1 (gates; M4 Max, mlx 0.32.2, py3.14)
Status: **both targets PASS** -- kernels vendored + bit-identical; the plan's
"main integration edit" (tensor-group slicing for TP) is now a proven recipe.

## Bottom line

1. **The EXL3 kernel set is vendored into the serving fork** as
   `mlx_lm.models.exl3` (import surface: `EXL3SwitchGLU`, `EXL3Linear`),
   branch `feat/dsv41-exl3` commit `9ea86f9` on `adurham/mlx-lm` (signed,
   pushed; branch, not main). 21 files, +4,457 lines, LICENSE + NOTICE +
   README included; `setup.py` registers the new packages/packagedata (a
   wheel built without that entry silently omits the module).
2. **The vendored tree is bit-identical to upstream ponyexl3** on real
   checkpoint tensors: 9/9 paths (EXL3SwitchGLU R=1/4/8 for both `silu` and
   `silu_clamp`; `EXL3Linear` on the k=6 head group at R=1/4;
   `reconstruct_public_mlx` on an expert w1 matrix) -- md5 + `mx.array_equal`
   match on every path.
3. **A 2-way TP split of the EXL3 expert module is exact**: slicing gate/up
   by OUT-tile on the intermediate (H) axis and the down projection by K-tile
   on the same H range, then summing the two half-width partials, reproduces
   the full module to fp16 rounding (max|diff| 2.4e-4 at R=1 / 4.9e-4 at R=4
   against max|full| 0.56/1.26; cos = 0.9999999). This matches the engine's
   existing convention (all-to-sharded on the gate/up output dim,
   sharded-to-all on the down input dim, all_sum of partials) -- no division,
   both ranks hold all experts at half intermediate width, same as V4 today.

## 1. What was vendored, and what changed

Source of record: the **patched working tree** on `macstudio-m4-1`
(`~/repos/ref/PonyExl3`, upstream `8e7fa6b` + the 2026-09-27/28 kernel work;
`exl3_moe.py` md5 `9f9bca31...`, `gemv_metal.py` md5 `3d5f69a0...` -- verified
at vendoring time). Upstream `main` alone does NOT contain these changes.

Scope: the full minimal import closure of the two exported classes --
12 modules under `mlx/` (`decode, exl3_linear, exl3_moe, gemv_metal, hadamard,
layer_state, metal_kernels, ops, perm, reconstruct, signs, stripe`) plus
4 numpy reference modules under `ref/` (kept: the Metal code builds its
codebooks through them). Deliberately excluded (not in the closure):
`exl3_qmv/qmm/fused/weights/native/mtp/generate/model` plumbing.

Local changes: (1) absolute `ponyexl3.*` imports -> relative, including the
four dynamic `__import__("ponyexl3.mlx.gemv_metal", ...)` sites in
`exl3_moe.py` (now `f"{__package__}.gemv_metal"`); (2) provenance header per
file; (3) `__init__.py` reduced to the two-class surface. **Numerical code
unchanged** except the three node-tree changes: v3 chunked prologue, v4 ALU
opts defaulted ON (SWAR + direct-read), additive `silu_clamp` activation.

## 2. Gate A -- vendored == upstream (bit-identical)

`raw/vendor-equiv.out`: 9 comparisons, all PASS. Representative md5s (vendored
vs upstream, equal): head-group `EXL3Linear` R=1 `986efc93...`, R=4
`a109c963...`; `EXL3SwitchGLU` R=1 `f84eec8f...`, R=4 `1e1152a7...`, R=8
`6d6afcc7...` (identical for `silu` and `silu_clamp` on this input);
`reconstruct_public_mlx` w1 expert0 `8dc2627a...`.

## 3. Gate B -- TP split recipe (the plan's "main integration edit")

Stacked layout (from the kernel sources, V4.1 dims, layer 1):
`gu_trellis (320, 2*E*144, 48)` = `[gate e0..eE-1 | up e0..eE-1]` x H-tiles;
`dn_trellis (144, E*320, 48)` = H(K)-tiles x D-out-tiles; `gu_suh (E,2,D)`,
`gu_svh (E,2*H)`, `dn_suh (E,H)`, `dn_svh (E,D)`.

Rank-r slice (r in {0,1}), contiguous intermediate split of H=2304:
- gate/up trellis: keep H-tiles `[r*72, (r+1)*72)` -- reshape to
  `(in_tiles, 2, E, 144, P)`, slice `[:, :, :, t0:t1, :]`, reshape back.
- `gu_svh`: keep `[r*1152, (r+1)*1152)` of each expert's 2*2304 block.
- down trellis: keep K-tiles `[r*72, (r+1)*72)`; `dn_suh` same H range.
- `gu_suh` and `dn_svh` replicated unchanged.

`raw/p42-split-equiv.out`: R=1 max|diff| 2.441e-4 (max|full| 0.5645),
R=4 4.883e-4 (1.264), cos 0.9999999 both; control (one half alone) deviates
0.7292 of max -- halves are genuine partials, not accidental copies.

**Convention mapping:** gate/up slice = the engine's `all-to-sharded` on the
output dim; down slice = `sharded-to-all` on the input dim; final
`all_sum` of the two partials. So the V4 `DeepseekV4ShardingStrategy` pattern
carries over to EXL3 experts unchanged -- "both ranks hold ALL experts at
half intermediate width", exactly as today's MoE-only TP (see the 2026-08-16
correction in auto_parallel.py; the plan doc's older "experts split by id"
phrasing is superseded).

## 4. What is NOT done yet (next bricks, in order)

1. EXL3 **loader module** for mlx-lm: checkpoint + layer idx -> stacked
   `EXL3SwitchGLU` (+ `EXL3Linear` for dense/MTP/head), with the rank slice
   above. Prototype logic exists (p30's `build_exl3_experts` + p2 harness);
   needs to become a first-class module with tests.
2. `deepseek_v41.py` model file (plan phase 3, the 10-step port; the
   `deepseek-v41-mlx` port is the validated reference for every step).
3. exo integration: `auto_parallel` branch, model card, generator wiring.
4. Remaining plan phases (Engram row store, cache wrapper, JIT lifecycle).

## Artifacts

- `raw/vendor-equiv.out` -- gate A transcript (9/9 PASS)
- `raw/p42-split-equiv.out` -- gate B transcript (PASS + control)
- `scripts/`: `p42_tp_split_equiv.py`, `exl3_vendor_equiv.py` (also in the
  session scratch `exl3patch/`)
- Fork: `adurham/mlx-lm` branch `feat/dsv41-exl3` @ `9ea86f9` (signed)
  (vendor manifest: `exl3patch/vendor-fork/vendor-manifest.txt`)
