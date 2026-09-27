# Phase 1 — EXL3 format validation (dealignai DSv4.1-Flash UNCENSORED EXL3 2.9bpw)

**Verdict: PASS on the plan's gate.** The EXL3 codec decodes this model
bit-exactly on every tensor class, the 1.4.2 container is accepted, and the
per-rank memory budget now replaces the plan's 105 GB estimate with a
measured number — which lands **above** the plan's 98 GB trigger, so
section 5's mitigations are mandatory rather than optional.

Taken 2026-09-27 on `macstudio-m4-1`, throwaway venv `~/phase1-exl3`
(PonyExl3 0.3.0 built from source, mlx 0.32.2, Python 3.14.2), fully
isolated from the production exo venv. The live cluster was up throughout;
`memory_pressure` never dropped below 44% free and the API kept serving.

## What this is and isn't

It answers the plan's phase-1 question — "can PonyExl3 read this file at
all" — and produces the real memory budget. It is **not** a quality
measurement and **not** a performance measurement: nothing here says
anything about tok/s on M4 Max, and nothing here was run through exo.
Every number below is from real tensors in the real 210.6 GB repo, not
synthetic data.

Chip note: the cluster nodes are **Apple M4 Max** Mac Studios (16 cores,
128 GB unified each — verified on both machines 2026-09-27 via
`sysctl machdep.cpu.brand_string`). Older docs in this repo say "M4 Ultra";
that part does not exist in the M4 generation, and those docs are wrong.

## 1. Format acceptance

`quantization_config.json` parses and is accepted: `quant_method=exl3`,
`version=1.4.2`, `codebook=mul1`, `avg_bits=2.9`, `head_bits=6`,
`mtp_bits=4`. `config.json` declares `model_type=deepseek_v41`,
40 layers, 384 routed experts/layer, `hc_mult=4`, Engram layers `[1,14]`,
3 DSpark stages — matching the plan's phase-3 targets.

The plan's conditional ("if PonyExl3 pins an older exllamav3 version, diff
the tensor layouts") is **moot**: PonyExl3 declares no exllamav3
dependency at all (`dependencies = mlx, mlx-lm, numpy`; exllamav3 appears
only as a cited CUDA reference in tests and docs). There is no version pin
to reconcile — and the bit-exact decode below is direct evidence the
layouts agree.

## 2. Bit-exact decode — the gate

Two comparisons per tensor group, both against PonyExl3's own numpy CPU
reference:

- **inner decode** (`decode_packed_trellis` vs `reconstruct_inner_mlx`) —
  the trellis codec itself: **bit-exact, `max_abs_diff = 0.0`, on every
  group tested.** This is the plan's pass condition.
- **full layer** (`reconstruct_public_weights` vs `reconstruct_public_mlx`)
  — inner + Hadamard + sign folding. Differs by at most **1 fp16 ULP**
  (see §4). Not bit-exact, and the reason is understood, not hand-waved.

Groups covered (all real tensors from the shipped shards):

| class | module | trellis | bits | inner decode |
|---|---|---|---|---|
| routed expert | `layers.3.ffn.experts.0.w1` | (320,144,48) | 3 | bit-exact |
| attention proj | `layers.0.attn.wq_b` | — | 5 | bit-exact |
| attention proj | `layers.0.attn.wq_a` | (320,80,96) | 6 | bit-exact |
| head | `head` | (320,8080,96) | 6 | bit-exact |
| shared expert | `layers.11.ffn.shared_experts.w1` | (320,144,64) | 4 | bit-exact |
| MTP routed expert | `mtp.0.ffn.experts.10.w1` | (320,144,64) | 4 | bit-exact |
| MTP attention | `mtp.0.attn.wo_a.slice.0` | (256,64,64) | 4 | bit-exact |
| MTP main proj | `mtp.0.main_proj` | (960,320,64) | 4 | bit-exact |

The MTP rows are the plan's "one 4-bit MTP tensor" requirement, extended to
three. They need a raw-shard path: **`tensor_storage` in
`quantization_config.json` carries zero `mtp.*` entries** (47,002 modules,
none of them MTP), so `layer_meta_from_config` raises `KeyError` on any
`mtp.*` key and the normal loader cannot build these layers. The tensors
exist and decode fine; they just have to be read with `safe_open` and the
`EXL3Layer` constructed by hand.

The plan's named API path also works as written: `ponyexl3.mlx.weights.load_safetensors`
on a shard returns a full tensor dict (4,681 tensors), its keys match the
safetensors header exactly, the trellis bytes are identical to
`safe_open`, and rebuilding an `EXL3Layer` from that dict decodes
bit-exactly.

## 3. Per-rank memory budget — the real number, and it fails the trigger

The plan: *"Walk every shard header and tabulate bits per tensor by
component… This is the real memory budget."* Measured by reading all 39
shard headers (`data_offsets`, so these are exact stored bytes, not
estimates) — 192,452 tensors, 210.58 GB total:

| component | tensors | GB | share |
|---|---:|---:|---:|
| routed experts | 184,320 | 196.03 | 93.1% |
| MTP (3 stages) | 4,836 | 7.24 | 3.4% |
| attention | 2,132 | 3.22 | 1.5% |
| embed | 1 | 1.32 | 0.6% |
| vision tower | 266 | 0.97 | 0.5% |
| shared experts | 480 | 0.80 | 0.4% |
| head | 4 | 0.50 | 0.2% |
| router + norm glue | 400 | 0.32 | 0.1% |
| other | 12 | 0.18 | 0.1% |

Expert sanity: 40 MoE layers × 384 experts × 3 matrices = 46,080, all
present. **Mixed precision inside the "2.9bpw" label:** routed experts are
3-bit in 35 layers but **2-bit in layers 18–22** (verified two ways —
`bits_per_weight=2` in the config, and `packed_size=32 → k=2` in the raw
trellis versus 48 → k=3 elsewhere). Worth knowing before anyone assumes a
uniform expert path.

Per-rank resident weight bytes under the V4 rule (experts split across 2
ranks, everything else replicated), applying the plan's mitigations
cumulatively:

| scenario | GB/rank | vs 98 GB trigger |
|---|---:|---|
| R0 raw (experts/2 + all replicated) | **112.56** | OVER |
| R1 − vision tower | 111.59 | OVER |
| R2 − MTP tensors | 104.35 | OVER |
| R3 − head/2 − embed/2 (shard head+embed) | 103.44 | OVER |
| R4 − coldest 10% of rank experts (SSD stream) | **93.64** | ok |

**Conclusion for section 5:** R0 (112.56 GB) exceeds the plan's 98 GB
trigger, so mitigations 1–4 go from optional to mandatory. And R0 + the
V4-Flash precedent's ~16 GB of runtime overhead = **128.6 GB against the
115 GB wired limit** — which is worse than the plan's 105+16=121 GB
estimate, because the real weight number is 7 GB higher than assumed.
Only the full mitigation stack (vision + MTP dropped, head/embed sharded,
coldest 10% streamed) lands inside the limit at 93.64 + 16 ≈ **110 GB**,
with the section-5 caveat that if 1–4 together don't clear it with 32K
context, raising `iogpu.wired_limit_mb` is required and the decode
estimate moves down.

## 4. The full-layer delta, localized (why it isn't bit-exact)

The only non-exact step is the outer transform, and it is fully
characterized. Isolated on two modules by comparing each stage
independently:

- **sign unpack: bit-exact** (`suh`/`svh` handling agrees exactly).
- **Hadamard (left and right), same fp32 input: differs at fp32 epsilon** —
  max abs 2.4e-06 on values of mean magnitude ~0.66, i.e. ~3.6e-06
  relative. Present on ~85% of elements, which is the signature of a
  **summation-order difference**, not a logic error: the ref does
  `had @ x[mask]` in numpy, the MLX path uses its own reduction order.
- **Pre-cast fp32 full reconstruction: same scale** (max 5.2e-08 on the
  6-bit attention projection, 4.8e-07 on the shared expert).
- **Post-cast fp16: 0.15% of elements differ, every one by exactly 1 ULP**
  (`max_diff_in_fp16_ulp = 1.0`, `all_diffs_within_1_ulp = true`). Of
  5,973,838 fp32-differing elements, 5,963,853 round to the *same* fp16
  value and only 9,985 cross a boundary.

So the fp16 mismatch is the fp32 noise surfacing at the cast, nothing
more. For scale: PonyExl3's own test suite asserts `atol=0.07` on this
exact path (`test_mlx_reconstruct.py:41`, 1000× the observed 6.1e-05), and
its CUDA↔MLX parity audit (`docs/drifts_investigation.md`) records this
same fp16-kernel-rounding class as expected cross-platform behavior with
a top-1 argmax match on a real model. The plan's stricter literal
"bit-exact" holds for the codec (the thing that would indicate a format
misread); it does not hold for the outer transform in fp16, and the honest
statement is **exact codec, 1-ULP-composed weights**.

## 5. Gate disposition

| plan requirement | result |
|---|---|
| version 1.4.2 accepted | PASS |
| bit-exact decode, routed expert | PASS |
| bit-exact decode, attention projection | PASS |
| bit-exact decode, 6-bit head | PASS |
| bit-exact decode, 4-bit MTP | PASS (3 groups, raw-shard path) |
| per-rank byte table | DONE (exact, from shard headers) |
| per-rank bytes vs ~98 GB | **112.56 GB → mitigations mandatory** |

Phase 1 is closed. The immediate consequence for phase 5's model card is
that the resident footprint must be computed from this table (not the
on-disk 210.6 GB, and not the plan's 105 GB estimate) and placement must
assume the mitigation stack.

## Raw artifacts

`raw/phase1-run.log`, `raw/phase1b-run.log`, `raw/phase1b-results.json`,
`raw/phase1c-run.log`, `raw/phase1c-results.json`, and the three runner
scripts (`phase1_exl3_validate.py`, `phase1b_exl3_mtp_budget.py`,
`phase1c_localize.py`, kept in the session scratch dir and reproduced on
node 1 as `/tmp/`). The phase-1b "6/9 passed" verdict line in its own log
is the addendum script's raw output — its three "failures" are the
full-layer 1-ULP class characterized in §4, not gate failures; the gate
rows above are the correct reading.
