# I11 STEP 1 — Is 5-bit FAST in the deployed MoE `gather_qmm` path?

**Question (pre-registered, `round8/PRE-REGISTRATION.md` §"I11 step 1"):** at the deployed
expert shapes, M=4, does 5-bit `gather_qmm` run in the fast templated kernel path, or does it
fall to a slow generic path?

**Answer: 5-bit is FAST.** Measured 471.04 µs/call vs 6-bit's 543.37 µs/call = **0.8669×**,
inside the pre-registered `FAST` band (`≤ 0.90×`). 4-bit is also FAST at **0.7183×**.

**⚠ SEPARATE FINDING THAT RESHAPES I11 — see §7.** The deployed routed experts are **already
4-bit** (`mxfp4`, group_size=32), not 6-bit. This is a premise-level finding about the I11
framing, not a result of the microbench, and it is reported separately from the measurement
above. The measurement stands on its own; §7 is what the parent needs to act on.

---

## 1. KERNEL-SOURCE FINDING — 5-bit IS in the fast templated path

### Which mlx is actually deployed (verified, not assumed)

The venv's mlx is an editable/local install pointing at the in-repo submodule, **not** PyPI:

- `~/repos/exo/.venv/lib/python3.13/site-packages/mlx-0.32.1.dev20260822+e40a416b2.dist-info/direct_url.json`
  → `{"url":"file:///Users/adam.durham/repos/exo/mlx","dir_info":{}}`
- `~/repos/exo/mlx` HEAD = `e40a416b20851d118b061b3a57d8cab70f5756de` — the `e40a416b2` in the
  version string. (Note: `~/repos/mlx` is a *different, unrelated* checkout at `1fe020ed`; the
  deployed one is the **exo submodule** `~/repos/exo/mlx`.)
- `quantized.cpp` is byte-identical on this MacBook and on macstudio-m4-1
  (md5 `fb3b76561a61168b6e6b6e59082c64a3` both), and no `.cpp/.h/.metal` under `mlx/` is newer
  than the built `core.cpython-313-darwin.so` (Aug 22 00:07) — **the source read below is the
  source that was compiled.**

### The line numbers in the brief are real, but they are NOT the load-bearing evidence

The brief pointed at `quantized.cpp` ~2051 and ~2186. Both exist and both mention 5-bit:

- `mlx/backend/metal/quantized.cpp:2051` — in `quantize_dequantize()`:
  `int packs_per_int = (bits == 3 || bits == 5) ? 8 : bits == 6 ? 4 : 8 / bits;`
- `mlx/backend/metal/quantized.cpp:2186-2188` — the same expression in `fast::Quantize::eval_gpu`.

But these are in the **quantize/dequantize** ops, not in `gather_qmm`. Per the brief's own
instruction ("do not guess from the presence of a `case 5:`"), the dispatch and templating
logic is what actually decides, so:

### Evidence A — 5-bit is instantiated through the identical macro chain as 4- and 6-bit

`mlx/backend/metal/kernels/quantized.metal:174-182`:

```
#define instantiate_quantized_all()  \
  instantiate_quantized_groups(2)    \
  instantiate_quantized_groups(3)    \
  instantiate_quantized_groups(4)    \
  instantiate_quantized_groups(5)    \   <-- 5-bit, same macro as 4 and 6
  instantiate_quantized_groups(6)    \
  instantiate_quantized_groups(8)
instantiate_quantized_all()
```

`instantiate_quantized_groups(bits)` (`:169-172`) expands to group sizes 128/64/32 →
`instantiate_quantized_types` (`:164-167`, float/float16/bfloat16) → `instantiate_quantized_funcs`
(`:154-162`), which includes the gather kernels this path uses:

- `:102-105` `affine_gather_qmv_fast`, `affine_gather_qmv`, `affine_gather_qvm`, `affine_gather_qmm_n`
- `:108-109` `affine_gather_qmm_t` (aligned, both true/false)
- `:151-152` `affine_gather_qmm_rhs_nt` / `_nn`

**5-bit gets the exact same set of compiled gather kernels as 4- and 6-bit.** There is no
separate/degraded expansion for it.

### Evidence B — 5-bit has a dedicated packing, in the same class as 3-bit

`mlx/backend/metal/kernels/quantized.h:17-26`:

```
template <int bits, int wsize = 8>
inline constexpr short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}
template <int bits, int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}
```

5-bit packs **8 values into 5 bytes** — exact, no waste (8×5 bits = 40 bits = 5 bytes). This
is a real dedicated packing, the same structural treatment 3-bit gets (8 values in 3 bytes),
which is what the brief reported. `load_vector` etc. have explicit `bits == 5` arms
(`quantized.h:72-85`, and likewise at `:152, :246, :348, :439, :530`), and every relevant
`static_assert` admits 5: *"Template undefined for bits not in {2, 3, 4, 5, 6, 8}"*
(`quantized.h:30-33, 109-112, 198-201, 300-303, 397-400, 487-490`).

### Evidence C — the M=4 gather_qmm dispatch has NO bit-width-dependent branch

`GatherQMM::eval_gpu` (`quantized.cpp:1840`) at our shape takes this path:

- `:1889-1895` — the `gather_qmv_rhs` fast path is gated on `bits_ == 4 || bits_ == 8`, i.e. it
  **is** bit-width-dependent — **but it also requires `M == 1 && B >= 16`**. At M=4 this branch
  is not reachable for *any* bit width, so it cannot advantage 4-bit over 5-bit here.
- `:1914`, `:1940` — `gather_qmm_rhs` / `gather_qmm_rhs_lhs` require `right_sorted_ == true`
  (and `B>=16` / `M>=16`). Production passes `sorted_indices=False` at this shape (see §3), so
  these are skipped. **No bits gate.**
- `:1861` `int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;` — on M4 Max the
  architecture reports as `applegpu_g16s` (confirmed live via `mx.device_info`), so `arch_size`
  is `'s'` → the `default:` case of `get_qmv_batch_limit` (`:86-128`). With
  `D=4096, O=1024` (`D<=4096 && O<=4096`) that returns **10 or 12** depending on the arch-gen
  branch — **either way ≥ 10 > M=4**, so the `M >= vector_limit` matmul branch at `:1962` is
  **not** taken. The conclusion does not depend on which arch-gen branch is taken.
- `:1983-2001` — `transpose_` is true → **`gather_qmv`**. **No bits gate.**

Inside `gather_qmv` (`:1058-1122`), the fast-vs-slow choice is
`bool fast = N % bn == 0 && K % 512 == 0;` (`:1084`, `bn = 8`) — **a function of N and K only,
never of `bits`.** At the deployed shapes both gate/up (N=1024, K=4096) and down (N=4096,
K=1024) satisfy it, so **all three projections take `affine_gather_qmv_fast` at 6, 5 and 4 bits
alike.** The kernel name is assembled at `:1085-1092` as
`affine_gather_qmv_fast_bfloat16_t_gs_32_b_<bits>` — the same templated kernel family, differing
only in the `bits` template parameter.

The one `is_power_of_2(bits)` gate in the file (`:1757`) is in `dispatch_qmv`, which serves the
**non-gather** `QuantizedMatmul` path and additionally requires `K == 128 || K == 64`. It is
irrelevant to `gather_qmm` at these shapes. **This was the most plausible mechanism for a
non-power-of-two penalty, and it does not apply here.**

**Conclusion: 5-bit is genuinely in the fast templated `gather_qmv_fast` path — not a generic
fallback.** The measurement in §4 independently confirms this (time tracks bytes almost
exactly, which a generic path would not do).

*Honest limitation:* I attempted to prove the dispatched kernel *name* empirically via
`mx.metal.start_capture()` and via an on-disk kernel-name cache. The capture bundle stores
names in a compressed `index` member that `strings` does not reach, and this mlx build keeps no
scriptable on-disk kernel cache — the same "per-kernel data not scriptably extractable"
limitation already recorded in `docs/p01-switch-mlp-gputrace-recapture-2026-08-29.md`. So the
kernel-identity claim rests on the source read above **plus** the proportional-scaling
evidence, not on a captured kernel name. Flagged rather than papered over.

---

## 2. SHAPES USED AND THEIR PROVENANCE

From `~/.exo/models/deepseek-ai--DeepSeek-V4-Flash-0731/config.json` on macstudio-m4-1 — the
checkpoint the live cluster actually has loaded (confirmed via the API: `/state` reports
`"modelId": "deepseek-ai/DeepSeek-V4-Flash-0731"`, `nLayers: 43`, `hiddenSize: 4096`,
`worldSize: 2`):

| Field | Value | Source |
|---|---|---|
| `num_hidden_layers` | 43 | config.json (matches the brief's "43 hidden layers") |
| `hidden_size` | 4096 | config.json |
| `moe_intermediate_size` | 2048 | config.json |
| `n_routed_experts` | 256 | config.json |
| `num_experts_per_tok` (top-k) | 6 | config.json |
| `n_shared_experts` | 1 | config.json (not part of the routed `gather_qmm` path) |
| TP world size | 2 | live `/state` `worldSize: 2`; `MLX_JACCL_SHARDING_MODE=Tensor` on the running process |
| **per-rank expert intermediate** | **1024** = 2048 // 2 | TP=2 shard |

Resulting per-rank tensor geometry benched (all 3 projections, matching `SwitchGLU`):

- `gate_proj`, `up_proj`: `(256, 1024, 4096)` — out=1024, in=4096
- `down_proj`: `(256, 4096, 1024)` — out=4096, in=1024
- **M = 4 rows**, top-k = 6 → **24 (row, expert) pairs per call**
- **group_size = 32 for all three arms** (held constant across 6/5/4, as required)

These are the same per-rank shapes the R1 I3 microbench used
(`round1/i3_microbench_chained.py:41-47`), which is the campaign's method of record.

---

## 3. METHOD

**Chained-graph construction — serial-sync was NOT used anywhere.**

Harness: `tmp/perf-campaign-2/round8/i11_gather_qmm_bits_microbench.py`, adapted from
`round1/i3_microbench_chained.py` (the CHAINED version adopted after the 2026-08-22 serial-sync
retraction). Specifically:

1. **Dependency chain, one eval.** `CHAIN_LEN = 300` SwitchGLU-equivalent forwards, each one a
   genuine data dependency of the next (`carry = x + 1e-9 * mean(out, axis=-2)`), with a
   **single `mx.eval()` at the very end** of the whole chain, followed by `mx.synchronize()`.
   There is **no `mx.eval()` inside any timing loop** — the artifact that invalidated R1's first
   two runs is structurally absent.
2. **Rotated routing indices.** A pool of `N_POOL = 64` independently drawn `(4, 6)` uniform
   index sets is cycled through the 300 calls, so no expert weight set can sit warm in cache
   across calls (the "fictitious cache" artifact class from the 2026-08-29 recapture doc).
3. **Measured empty-graph baseline, subtracted.** The identical carry chain with the three
   `gather_qmm` calls removed was timed (median **12.63 µs/call**, range 12.10–18.89) and
   subtracted, so the reported µs/call is the **marginal cost of the MoE kernels**, not the
   harness scaffolding. Both raw and net numbers are in the JSON; the table below is net.
4. **Elision detector.** µs/call was measured at chain lengths 100/200/300 per arm and is flat —
   6-bit: 552.59 / 550.17 / 550.97; 5-bit: 480.11 / 479.87 / 479.56; 4-bit: 400.63 / 399.62 /
   399.69. A chain being optimised away would show µs/call collapsing as length grows. It does
   not: **the 300 dependent calls are really executing.**
5. **Production call signature.** `mx.gather_qmm(..., rhs_indices=idx, transpose=True,
   group_size=32, bits=B, mode="affine", sorted_indices=False)` — exactly
   `QuantizedSwitchLinear.__call__` (`mlx-lm/mlx_lm/models/switch_layers.py:76-91`).
   `sorted_indices=False` is correct at this shape: `SwitchGLU.__call__` sets
   `do_sort = indices.size >= 64` (`switch_layers.py:182`) and M=4 gives `indices.size = 24 < 64`.
6. **Bytes from real array `.nbytes`**, never assumed: weight + scales + biases summed from the
   actual quantized arrays, divided by 256 experts, times 24 pairs.
7. **n = 5 reps per arm**, ranges reported, never bare means.

**Why affine mode.** The `mxfp` family in this build defines **only** 4-bit and 8-bit:
`ops.cpp:4976` `int expected_bits = mode == QuantizationMode::Mxfp8 ? 8 : 4;` and it throws
otherwise (`:4984-4989`). **There is no `mxfp5` or `mxfp6`.** A 6-vs-5-vs-4 comparison is
therefore necessarily an **affine**-mode comparison — which is also the mode real
`mlx-community` 5-bit/6-bit conversions use. Stated explicitly rather than silently assumed.
All three arms use affine + group_size 32, so the comparison is internally consistent.

**Run location:** macstudio-m4-2 (`192.168.86.202`, the non-API node), via the node's own
`~/repos/exo/.venv/bin/python`. **No cluster process was killed, restarted, relaunched, or
reconfigured** — see §6.

**One real harness bug found and root-caused, not worked around.** Early runs died with
`[METAL] Command buffer execution failed: Insufficient Memory`. Rather than adding a retry and
moving on, I instrumented it: a single arm is only ~2.82 GB, but holding all three arms resident
simultaneously (~7.3 GB) on top of the live runner's resident working set intermittently exceeds
the Metal working-set limit. Fix: **one arm resident at a time**, built/measured/freed inside
each rep. Interleaving is preserved at the rep level (§6), and rebuilding per rep also means each
rep's weights land on different physical pages, which guards against a fixed-page residency
artifact. A bounded back-off retry remains only as a guard against transient pressure from the
live cluster; it did not fire during the reported run.

---

## 4. RESULTS

`chain_len=300`, `n_pool=64`, `reps=5`, `group_size=32`, `mode=affine`, M=4, per-rank shapes.
µs/call is **net of the measured 12.63 µs/call empty-graph baseline**. Ranges are min–max over
the 5 reps.

| bits | µs/call (range over n=5) | median µs/call | achieved GB/s | % of 546 GB/s | ratio vs 6-bit | byte ratio | **band verdict** |
|---|---|---|---|---|---|---|---|
| **6** (baseline) | **540.76 – 544.35** | 543.37 | 486.3 | 89.1% | 1.000 | 1.000 | — (reference) |
| **5** | **469.84 – 472.64** | 471.04 | 480.8 | 88.1% | **0.8669** | 0.8571 | **FAST** |
| **4** | **389.63 – 392.86** | 390.29 | 483.6 | 88.6% | **0.7183** | 0.7143 | **FAST** |

Pre-registered bands applied verbatim (FAST ≤ 0.90×; MARGINAL 0.90–0.98×; SLOW ≥ 0.98×):

- **5-bit: 0.8669× → FAST.** Also below the 0.833× proportional expectation? No — 0.8669 is
  slightly *above* 0.833, i.e. very slightly sub-proportional, but comfortably inside the FAST
  band. Reported precisely rather than rounded to "proportional".
- **4-bit: 0.7183× → FAST** (proportional expectation 0.667×; likewise marginally
  sub-proportional).

**Bytes per call** (real `.nbytes`, weight + scales + biases, ÷256 experts × 24 pairs):

| bits | bytes/expert (3 projs) | bytes/call | weight lastdim (gate/up) |
|---|---|---|---|
| 6 | 11,010,048 | 264,241,152 | 768 uint32 |
| 5 | 9,437,184 | 226,492,416 | 640 uint32 |
| 4 | 7,864,320 | 188,743,680 | 512 uint32 |

**The decisive evidence that 5-bit is not on a generic path:** measured time ratio (0.8669)
tracks the byte ratio (0.8571) to within 1.1 points, exactly as 4-bit does (0.7183 vs 0.7143,
0.4 points). A slow generic path would show time *decoupled* from bytes — 5-bit would cost
roughly as much as (or more than) 6-bit despite moving 14% fewer bytes. It does not. The tiny
excess over the byte ratio is the same small fixed per-dispatch cost visible in all three arms.

**Bandwidth-plausibility sanity check (explicitly required):** achieved bandwidth is
**480–486 GB/s, i.e. 88–89% of the 546 GB/s M4 Max spec**. No arm exceeds peak (which would
indicate a cache artifact or a byte-accounting error), and no arm is implausibly low. This sits
right on top of the independently established figures in the record for this same kernel —
`docs/PERFORMANCE_HISTORY.md` reports fused_gate_up at 531 GB/s (97.5% of spec) and down_proj at
482 GB/s (88.5% of spec) from the 2026-08-29 `MLX_GPU_TIME=1` per-stage attribution. **These
numbers are measuring the kernel, not dispatch overhead** — corroborated three ways: (a) the
subtracted empty-graph baseline is only 12.63 µs vs 390–543 µs of signal, so scaffolding is ~2-3%
of the measurement; (b) µs/call is flat across chain length 100→300; (c) achieved GB/s lands in
the same band as the record's independent instrument.

---

## 5. CONTENTION

**Could live-cluster GPU contention have biased the arms? Yes in principle, and it was
explicitly controlled for and measured.**

- The cluster was **LIVE on the production 6-bit config throughout** (that is a hard constraint
  of this round, not an oversight), and other round-8 work shares these nodes.
- **Idle-state check.** Immediately after the benchmark finished, `exo_gpu_usage_ratio` was
  **0.028 on the bench node (m4-2)** and **0.026 on m4-1** — i.e. the live runner was
  essentially idle, consistent with the R1 protocol's readings (0.027–0.031).
- **The ~0.98 readings sampled *during* my run are my own benchmark**, not a competing workload:
  the ratio returns to 0.028 the moment the bench exits. I sampled `exo_gpu_usage_ratio` before
  and after **every single timed block** (recorded per-rep in the JSON under
  `arms.<bits>.per_rep`) precisely so this could be checked rather than asserted.
- **Arms were interleaved.** Every rep visits all three bit widths, and the visit order
  **alternates each rep**: `[6,5,4] / [4,5,6] / [6,5,4] / [4,5,6] / [6,5,4]`. Any monotonic drift
  (thermal, or a competing job ramping) is therefore spread across all three arms instead of
  loading onto whichever ran first. This is recorded in the JSON as `rep_order`.
- **Observed spread is tiny and does not overlap between arms.** Per-arm ranges are
  6-bit 540.76–544.35, 5-bit 469.84–472.64, 4-bit 389.63–392.86 — each arm's full n=5 range is
  under 0.7% wide, and the three ranges are **cleanly separated with no overlap**. A contention
  artifact large enough to manufacture a 13% gap between 6-bit and 5-bit would have to have
  landed on the same arm in all five reps despite the alternating order, while leaving each
  arm's own spread under 1%. That is not a plausible failure mode here.
- **Residual honest caveat:** a microbench run on a node that is *also* hosting a live inference
  runner is never as clean as a quiesced node. The arms are internally consistent and the
  ordering was counterbalanced, so I consider the *ratios* (the thing the verdict turns on)
  trustworthy; the *absolute* µs/call could shift by a few percent on a fully idle machine.

---

## 6. CLUSTER UNTOUCHED

No process was killed, restarted, relaunched, or reconfigured. The exo PIDs and their
`etime` advanced monotonically across the whole task (9390/9391/9392 on m4-1 from `59:40`
at task start through `01:22:41` at the end) — a restart would have reset `etime` and changed
PIDs. Verified again at the end of the task; see the final summary for the live PID line.

One incident, reported rather than hidden: an early SSH command of mine had a quoting bug that
caused the remote shell to begin executing `start_cluster.sh`. It aborted on its own at the
Thunderbolt-topology discovery step ("CRITICAL ERROR: Could not map Studio-to-Studio Thunderbolt
topology!") **before** touching any running process, and the PIDs above confirm the live cluster
was unaffected. I switched to a `scp`-a-script-then-`bash` pattern for all subsequent remote
work.

---

## 7. ⚠ FINDING THAT RESHAPES I11: the deployed routed experts are ALREADY 4-bit

This is **not** a microbench result — it is a premise check I ran because the measurement's
value depends on it. Reported separately and explicitly.

**The routed experts of the deployed checkpoint are stored and loaded at 4 bits (`mxfp4`,
group_size 32), not 6 bits.** Evidence:

1. **On disk** (`~/.exo/models/deepseek-ai--DeepSeek-V4-Flash-0731`, read via safetensors
   headers and via `mx.load`):
   - `layers.3.ffn.experts.0.w1.weight` → dtype `int8`, shape `(2048, 2048)`.
     The logical tensor is 2048×4096 (`moe_intermediate` × `hidden`); 2048×4096 values at
     **4 bits** = 4 MB = exactly the stored `2048×2048` int8 bytes. **Two 4-bit values per byte.**
   - `layers.3.ffn.experts.0.w1.scale` → dtype `uint8` (`F8_E8M0`), shape `(2048, 128)` →
     4096/128 = **group_size 32**.
   - By contrast `layers.3.ffn.shared_experts.w1.weight` → `F8_E4M3`, shape `(2048, 4096)` =
     **8-bit**. Routed experts and shared experts are at *different* precisions.
2. **In the loader**: `config.json` has `quantization_config.quant_method = "fp8"` and
   `expert_dtype: "fp4"`. `mlx-lm/mlx_lm/utils.py:548-556` routes `quant_method == "fp8"` +
   `model_type == "deepseek_v4"` into `make_quantization_config(model)`, which
   (`mlx-lm/mlx_lm/models/deepseek_v4.py:952-984`) assigns
   `mxfp4 = {"group_size": 32, "bits": 4, "mode": "mxfp4"}` to every `.ffn.switch_mlp.*_proj`
   (the routed experts) and `mxfp8` (8-bit) to shared experts, attention and MTP projections.
3. **The mxfp override is confirmed active**: `utils.py`'s `_is_mxfp_override` keeps the mxfp4
   override only when the on-disk scales are `uint8` — and they are `uint8` (point 1). So the
   routed experts really do load as `mxfp4`/4-bit.
4. The R1 record already assumed exactly this: `round1/i3_microbench_chained.py:49-51` sets
   `EXPERT_GROUP_SIZE=32, EXPERT_BITS=4, EXPERT_MODE="mxfp4"`, and `PERFORMANCE_HISTORY.md`
   repeatedly describes the production decode shape as "mxfp4 g=32 b=4".

**Why this matters for the decision package.** I11's stated physics is "bandwidth-bound decode
× −33% active expert bytes at 4-bit," framed as a 6→5 or 6→4 move for routed experts. But the
routed experts — the tensors the `gather_qmm` decode path streams, and the entire basis of that
byte-reduction argument — **are already at 4 bits.** There is no 6→4 saving available on them,
because that move has already been made. The "6-bit" label on the shipped config does not
describe the routed-expert precision.

**What I am NOT claiming:** I have not established what the campaign's "shipped 6-bit config"
label *does* refer to (candidates: an overall size label, or other tensor groups —
note `start_cluster.sh:359` says "~100 GB/rank at 6-bit" while the checkpoint is 155 GB total
across 2 ranks). Resolving that is outside this task's scope and needs a decision from the
parent. I flag it because **the I11 evidence package should not be built on a 6→4 routed-expert
byte saving that does not exist**, and because the pre-registered step-3 decode measurement
would otherwise be comparing arms that differ less than expected.

The microbench above remains valid and answers the question asked — *the 5-bit kernel is fast*
— and it also establishes the real, measured cost curve of this kernel across 6/5/4 bits at the
deployed shapes, which is reusable regardless of how the labelling question resolves.

---

## 8. BOTTOM LINE

**5-bit is FAST (0.8669× the 6-bit µs/call, inside the pre-registered `≤0.90×` FAST band), so on
kernel-speed grounds 5-bit survives and the user would get a 3-way choice rather than a binary
6→4** — with the significant caveat in §7 that the deployed routed experts are already 4-bit,
which the parent must resolve before the 6→5/6→4 framing of the I11 package can stand.

---

## FILES

- `i11_gather_qmm_bits_microbench.py` — the chained-graph harness (this task's measurement)
- `i11_full_results.json` — full raw results incl. per-rep samples, per-rep GPU-usage readings,
  `rep_order`, scaling check, real array shapes/nbytes
- `i11_full_run.log` — complete stderr of the reported run
- `i11_shape_probe.py` — empirical quantized (shape, dtype) convention probe per bit width
- `i11_deployed_expert_dtype.py` — the §7 premise check against the resident checkpoint
- `i11_mem_probe.py`, `i11_mem_instrument.py`, `i11_oom_isolate.py` — the OOM root-cause
  instruments (§3)
- `i11_kernel_capture.py` — the kernel-name capture attempt whose limitation is disclosed in §1
