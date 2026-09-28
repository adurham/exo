# Phase 4 — EXL3 ↔ production MTP compatibility, and a DSpark guard false-FAIL

Date: 2026-09-27/28
Host: hermes-gw-01 (LXC gateway) + macstudio-m4-1 (V4.1 build box)
Status: findings closed; one scoped production bug fixed, signed, pushed
(`72c678ee`)

Phases 1–3 asked whether DeepSeek-V4.1-Flash 2.9bpw (EXL3) could be made to run
on the two-node cluster. This phase asks the deployment question that follows:
**can production's own DSpark/MTP speculative path consume that checkpoint's
draft head, or does the head have to come from the native checkpoint?** It also
follows up, out of the logs, on a guard that had been reporting FAIL on every
model load.

## Bottom line

1. **The EXL3 draft head is not loadable by production — closed, no
   workaround.** Deploy the hybrid: **body from EXL3, draft head from the
   native checkpoint.** The substitution costs **+0.69 GB** total
   (**+0.35 GB/rank** at TP=2).
2. **A load guard had been reporting a false FAIL on every load of every
   quantized DSpark head, forever.** Fixed, 7 regression tests, signed, pushed
   as `72c678ee`.
3. **Production is running spec decode correctly right now** — native head
   attached, γ=3, no fallback. Its *acceptance telemetry is off by default and
   has never fired*, so the acceptance rate is still unmeasured and needs a
   relaunch (Adam's call).

## 1. EXL3 MTP format vs production's loader

All 4,836 `mtp.*` tensors in the EXL3 checkpoint were enumerated from its own
safetensors headers (`p7c_exl3_loadable_and_layers.py`):

| leaf | count |
|---|---|
| `suh` / `svh` / `mul1` / `trellis` | 1198 each |
| `weight` | 20 |
| `bias` | 3 |
| `attn_sink` | 3 |
| `hc_attn_base` / `hc_attn_fn` / `hc_attn_scale` | 3 each |
| `hc_ffn_base` / `hc_ffn_fn` / `hc_ffn_scale` | 3 each |

Decisive numbers:

- quantized bases **lacking** a `.scale` leaf: **1198 / 1198**
- quantized bases carrying a `.weight`: **0**
- files in `mlx-lm` + the mlx engine mentioning `exl3`/`trellis`: **0**

Example — `mtp.0.attn.wkv` has leaves `['mul1', 'suh', 'svh', 'trellis']`, no
`.weight`, no `.scale`. Production's sanitizer/loader expects affine
`.weight` + `.scale`. **EXL3's draft head cannot be loaded by production, and
production contains no EXL3-aware loading path at all.**

Related earlier finding (phase 1, unchanged): EXL3 MTP also has `bias` but no
`bias_vl`. That turns out **not** to be the blocker it looked like — see §2.

### Why the hybrid, and what it costs

Byte counts straight from the safetensors headers (`p7f_mtp_bytes.py`), no
dtype guessing, nothing loaded:

| | EXL3 | native |
|---|---|---|
| directory total | 210.60 GB (39 shards) | 244.20 GB (12 of 48 shards present) |
| `mtp.*` bytes | 7.243 GB / 4836 tensors | 7.933 GB / 2401 tensors |
| `mtp.*` share | 3.44% | 3.25% |

- `mtp.*` lives in two EXL3 shards: `model-00039` (4.847 GB) and
  `model-00038` (2.397 GB).
- Native `mtp.*` lives in shards 44/45/46 (2.653 / 2.574 / 2.706 GB).
- EXL3 body-only = **203.36 GB**; hybrid total = **211.29 GB** →
  **+0.690 GB over EXL3 as-is**, ≈ **+0.345 GB/rank**.

The draft head is ~3.4% of the directory, so swapping it is nearly free in
both bytes and risk — and the native head is the one production already loads
cleanly (§3).

## 2. `bias_vl` is a red herring — the gate allocates it conditionally

Phase 3 left open whether EXL3's missing `bias_vl` blocks the DSpark path. It
does not. In `mlx-lm/mlx_lm/models/deepseek_v4.py` the MoE gate declares:

```python
self.vl = config.vision_n_layers > 0
```

`e_score_correction_bias_vl` exists only for vision configs. So a
non-vision DSpark head legitimately has no `bias_vl`, and its absence is not a
defect in the EXL3 checkpoint.

## 3. The DSpark load guard was reporting a false FAIL — fixed

`_log_dspark_load_guard` (`src/exo/worker/engines/mlx/utils_mlx.py`) diffed the
**already-quantized** attached head against a **freshly-constructed,
unquantized** `DeepseekV4DSparkModule`. `nn.quantize` adds a `.scales` key per
quantized projection, so the loaded side legitimately held more keys; the guard
counted them as `extra` and flipped `param_tree_assert` to FAIL on **every load
of a quantized head, forever**.

Live on Vision-Exp, verbatim from the log:

```
[DSPARK-GUARD] provenance=native source=/Users/.../DeepSeek-V4-Flash-Vision-Exp
  param_tree=118/84 missing=0 extra=34 param_tree_assert=FAIL
  block_size=5 markov_rank=256 n_stages=3 taps=[40, 41, 42]
  noise_token_id=128799 CHECKPOINT_PROVENANCE=MATCHED
[DSPARK-GUARD] param-tree mismatch — missing=[]
  extra=['main_proj.scales', 'stages.0.attn.wkv.scales', ...]
```

The arithmetic pins it to a methodology artifact:

```
118 - 84 = 34   ==   25 mxfp8 + 9 mxfp4   (the overlay's own tally)
```

and every one of the 34 reported extras ends in `.scales`. Meanwhile
**`missing=0` was passing all along** — every parameter the head needs had
arrived. So the one line meant to catch the real DSpark incident (the head
silently keeping random init under `strict=False`) was permanently stuck at
FAIL and had taught readers to ignore it.

**Fix:** normalize quantization artifacts away on **both** sides before the
diff. `X.scales` maps onto the `X.weight` it quantizes — **not** to a bare `X`;
mapping to a bare stem leaves every extra unmatched and silently voids the fix
(there is a test for exactly this). The guard also logs `quant_keys=<n>` so the
normalization is visible rather than implicit, and still reports raw
`param_tree` counts.

Verified before pushing:

- new regression test, 7 cases, with a **RED/GREEN proof** against the old
  comparison — old logic FAILs the quantized-head fixture (16 extras, all
  `.scales`), new logic PASSes, and genuine missing/extra keys still FAIL in
  both directions;
- tests deliberately avoid importing `mlx`/`mlx_lm` so they actually run on the
  Linux gateway (the pre-existing guard test cannot be collected there —
  reproduced on the untouched tree);
- basedpyright **199 errors before and after** (zero delta), ruff clean.

Scope note: log-line only, never raises, production was **not** relaunched.
Takes effect at the next worker start.

## 4. Acceptance rate: still unmeasured, and now we know exactly why

Production's acceptance telemetry is **off by default** and **has never
fired**:

- `EXO_DSV4_MTP_LOG_INTERVAL` defaults to `0` → neither
  `mtp_batch_generator`'s `accept_rate=...` line nor `dsv4_mtp`'s
  `[MTP] cycles=... mean_accept=x/3 hist=...` line is emitted;
- `EXO_DSV4_SPEC_SHADOW` defaults to off → no `[DSPARK-SHADOW]` summary;
- a **530 MB** `exo.log` contains **zero** acceptance lines.

What production *is* doing (verbatim, 2026-09-27 19:57–19:58):

```
DSpark draft head attached from .../DeepSeek-V4-Flash-Vision-Exp
  (NATIVE checkpoint-bundled head, 118 tensors, 3 stages,
   block_size=5, taps=[40, 41, 42]).
DSv4 MTP speculative decoding enabled (γ=3, T=0.0)
```

No fallback lines. To measure the rate we need a relaunch with
`EXO_DSV4_MTP_LOG_INTERVAL=64`; the first `[MTP] cycles=64 mean_accept=...`
lands after 64 spec cycles. (`EXO_DSV4_SPEC_SHADOW=1` would give a
*would-accept* measurement but it forces `n_accepted=0`, so it changes decode
behavior; `EXO_DSV4_MTP_PROFILE=N` additionally dumps per-phase timing.)

**A 4-layer reduced body cannot answer this**: it drafts ~randomly (0/5
accepted over 47 rounds in phase 3). The number needs the full 40-layer model,
and only the local shards 0–3 are present (12 of 48 shards; 35 layers absent,
L14 partial) — so a deeper quick test needs more downloading.

## 5. Deployment shape implied by this phase

- **Body**: EXL3 (2.9bpw, 203.36 GB) + streamed Engram tables from native
  shards 47+48 (phase 1/2; 2.21 ms/token, ~2% of budget).
- **Draft head**: native `mtp.*` (7.933 GB, 2401 tensors) — production loads
  this shape cleanly.
- **EXL3 kernels**: still unported into production. They live only in the
  reference repo (PonyExl3); `mlx-lm`/the mlx engine have zero exl3/trellis
  references. Until they are ported, EXL3-format weights cannot run in
  production at all — this phase's finding is a *prerequisite* to that port,
  not a substitute for it.

## Artifacts

- `scripts/p7c_exl3_loadable_and_layers.py` — EXL3 MTP tensor-form census +
  local-shard layer coverage
- `scripts/p7e_guard_arithmetic.py` — the 118/84/34 quantization-config
  arithmetic
- `scripts/p7f_mtp_bytes.py` — body/draft byte split from safetensors headers
- `scripts/red_green_guard.py` — RED/GREEN proof for the guard fix
- `scripts/scan_acceptance.sh`, `scripts/read_accept_logs.sh`,
  `scripts/find_acceptance.sh` — log forensics for acceptance telemetry
- `raw/p7c.out`, `raw/accept.out` — captured output (md5s matched both sides)
- the guard fix itself ships in commit `72c678ee` with its test file

## Raw evidence index

| file | what it shows |
|---|---|
| `raw/p7c.out` | 4836 EXL3 MTP tensors; 1198 trellis-quantized; 0 with `.weight`; 0 exl3-aware files in production; layers 0–3 only |
| `raw/accept.out` | guard FAIL on two loads; native head attached; γ=3; **zero** acceptance lines in 530 MB of log |
