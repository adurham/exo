# Quantized SDPA kernel design (MiniMax Phase 2)

Design doc for the lever identified as Phase 2 in
[`minimax-decode-optimization.md`](./minimax-decode-optimization.md).
Implementation is the only remaining decode lever after the Phase 1
QK-norm rewrite proved decode-neutral.

## Problem

Today, `mlx_lm/models/base.py:117-132` runs a two-step attention path
when the KV cache is quantized:

```python
if hasattr(cache, "bits"):
    dk = mx.dequantize(*keys, group_size=cache.group_size, bits=cache.bits)
    dv = mx.dequantize(*values, group_size=cache.group_size, bits=cache.bits)
    return mx.fast.scaled_dot_product_attention(queries, dk, dv, ...)
```

For MiniMax-M2.7 at 66K context, 8 KV heads, head_dim 128, 5-bit KV with
`group_size=64`:

- **Packed K**: 66K × 8 × 128 × 5 bits ÷ 8 bits/byte ≈ 5.4 MB + ~2 MB
  scales + ~2 MB biases per layer. ~10 MB reads per K (and per V).
- **Dequantized K (bf16)**: 66K × 8 × 128 × 2 bytes = 135 MB per layer.
  Written by `mx.dequantize` to VRAM, then immediately read by SDPA.

Net round-trip per layer per decode token is ~270 MB of K traffic
(write-then-read) + 270 MB of V traffic = **540 MB per layer × 62 layers
≈ 33 GB per decode token**. At Apple Silicon's ~400 GB/s unified-memory
bandwidth that's **~83 ms/token of bandwidth time**. At 17 tok/s
(~59 ms/token) this is the dominant cost. The profile confirms:
attention is 70 % of wall time, ~41 ms/token.

## Solution

Fuse the dequantize into the SDPA kernel so K/V read is packed once and
dequantized in-register per tile. Eliminates the dequantized-K/V
VRAM write + re-read. Expected savings:

- Bandwidth per layer drops from ~540 MB → ~20 MB (packed reads only).
- At 400 GB/s that's ~1.3 ms → ~0.05 ms per layer, saving ~1.25 ms/layer.
- Across 62 layers: ~78 ms/token saved.
- Decode improves from 17 tok/s → **~25 tok/s** (+47 %) in the limit.
- More realistic with compute floor: **+20–30 % decode**
  (matches the research-agent estimate).

## Approach — new kernel in the adurham/mlx fork

Two options considered:

1. **Fork the MLX fast-SDPA kernel into exo as a custom `mx.fast.metal_kernel`.**
   Self-contained in exo, no mlx-fork changes, similar pattern to the
   qwen3_5_moe patches. Downside: duplicates MLX's SDPA algorithm inside
   exo; tracking upstream kernel changes becomes a merge chore.
2. **Add a new kernel + Python entry point in the adurham/mlx fork.**
   Reuses the sdpa_vector_2pass infrastructure, fits the upstream
   extension pattern, potentially upstreamable. Downside: larger fork
   diff against ml-explore/mlx.

**Going with option 2** (mlx fork). Rationale: the existing
`sdpa_vector_2pass_1` kernel is the right template — we want the same
FlashDecoding 2-pass structure, we just need a quantized K/V load path.
Copy-pasting the algorithm into an exo custom kernel would be a lossy
adapter; keeping it alongside its siblings in the fork is cleaner.

## Kernel specification

### New Metal kernel: `sdpa_vector_2pass_1_quant`

Template parameters:

```cpp
template <typename T, int D, int V = D, int BITS, int GROUP_SIZE>
[[kernel]] void sdpa_vector_2pass_1_quant(
    const device T* queries        [[buffer(0)]],   // unchanged: bf16 Q
    const device uint32_t* k_packed[[buffer(1)]],   // was: const device T* keys
    const device T* k_scales       [[buffer(2)]],   // NEW
    const device T* k_biases       [[buffer(3)]],   // NEW
    const device uint32_t* v_packed[[buffer(4)]],   // was: const device T* values
    const device T* v_scales       [[buffer(5)]],   // NEW
    const device T* v_biases       [[buffer(6)]],   // NEW
    device T* out                  [[buffer(7)]],
    device float* sums             [[buffer(8)]],
    device float* maxs             [[buffer(9)]],
    const constant int& N          [[buffer(11)]],
    const constant size_t& k_head_stride [[buffer(12)]],
    const constant size_t& k_seq_stride  [[buffer(13)]],
    const constant size_t& v_head_stride [[buffer(14)]],
    const constant size_t& v_seq_stride  [[buffer(15)]],
    const constant float& scale    [[buffer(16)]],
    // ...optional mask buffers, function constants...
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]])
```

### Layout invariants (derived from `QuantizedKVCache`)

Storage for one head, one sequence step (see `cache.py:232-283`):

- `k_packed`: shape `[..., steps, head_dim / (32/BITS)]`, dtype `uint32`.
  For BITS=5: each `uint32` packs 6 values (6×5=30 bits used, 2 bits
  unused). For head_dim=128 that's `⌈128/6⌉ = 22` uint32s per step per
  head.
- `k_scales`, `k_biases`: shape `[..., steps, head_dim / GROUP_SIZE]`,
  dtype matches Q (bf16). For head_dim=128, GROUP_SIZE=64 → 2 scales
  per step per head.

Strides in uint32/bf16 elements are computed at dispatch time and
passed in.

### Dequantize-in-register loop (the hot inner section)

Existing kernel, line 275–276 of `sdpa_vector.h`:

```cpp
for (int i = 0; i < qk_per_thread; i++) {
    score += q[i] * keys[i];
}
```

Becomes (for BITS=5, qk_per_thread=4, one thread handles a contiguous
4-element slice of head_dim per key position):

```cpp
// Each thread's 4-element slice may span 1 or 2 packed uint32 words.
// Compute which uint32(s) contain the slice at element-offset
// `simd_lid * qk_per_thread`, extract the 5-bit values, apply
// scale/bias for the containing group, accumulate score.

const int elem_offset = simd_lid * qk_per_thread;
const int group_idx = elem_offset / GROUP_SIZE;
const float scale_k = float(k_scales_thread[group_idx]);
const float bias_k  = float(k_biases_thread[group_idx]);

const int values_per_word = 32 / BITS;  // 6 for BITS=5
const int word_idx = elem_offset / values_per_word;
const int lane_in_word = elem_offset % values_per_word;

uint32_t w = k_packed_thread[word_idx];
// Handle boundary crossing: at BITS=5, 4 values fit in one uint32
// if lane_in_word <= 2 (positions 0..3 fit in bits 0..19), else span.
uint32_t w_next = (lane_in_word + qk_per_thread > values_per_word)
                  ? k_packed_thread[word_idx + 1] : 0;

constexpr uint32_t MASK = (1u << BITS) - 1u;
float k_deq[qk_per_thread];
for (int j = 0; j < qk_per_thread; j++) {
    const int pos = lane_in_word + j;
    uint32_t raw;
    if (pos < values_per_word) {
        raw = (w >> (pos * BITS)) & MASK;
    } else {
        raw = (w_next >> ((pos - values_per_word) * BITS)) & MASK;
    }
    // MLX dequantize is: x = (raw * scale) + bias  (affine)
    k_deq[j] = float(raw) * scale_k + bias_k;
}

U score = 0;
for (int j = 0; j < qk_per_thread; j++) {
    score += q[j] * k_deq[j];
}
score = simd_sum(score);
```

Same transformation for V on the accumulator update (line 293):

```cpp
for (int i = 0; i < v_per_thread; i++) {
    // ... dequantize v_deq[i] from v_packed, v_scales, v_biases ...
    o[i] = o[i] * factor + exp_score * v_deq[i];
}
```

### Dispatch (C++ side, `scaled_dot_product_attention.cpp`)

New function `sdpa_vector_2pass_quant` mirrors `sdpa_vector_2pass` with:

- Additional array arguments for `k_scales`, `k_biases`, `v_scales`,
  `v_biases`.
- Kernel name suffix `_quant_<BITS>_<GROUP_SIZE>` so different bit
  widths get their own specialized kernels (MLX already does this for
  template specialization via `get_kernel` + function constants).
- Same block heuristic as the bf16 variant — the decode algorithm is
  identical; only the K/V load changes.

### Python entry point

Either:

- New `mx.fast.scaled_dot_product_attention_quant(q, k_triple, v_triple, ...)`
  where `k_triple = (packed, scales, biases)`.
- Or extend existing `mx.fast.scaled_dot_product_attention` to accept
  a `quantized_kv=True` flag plus the triples.

Leaning toward **a new dedicated function** — cleaner API, no
load-bearing ambiguity about argument types. The fused-SDPA overload
can continue to accept bf16 K/V unchanged.

### Call site update (mlx-lm)

```python
# mlx_lm/models/base.py:117-141 becomes:
if hasattr(cache, "bits"):
    return mx.fast.scaled_dot_product_attention_quant(
        queries,
        k_triple=keys,   # (packed, scales, biases) from QuantizedKVCache
        v_triple=values,
        scale=scale,
        mask=mask,
        sinks=sinks,
    )
else:
    return mx.fast.scaled_dot_product_attention(queries, keys, values, ...)
```

## Supported configurations (v1 scope)

- **Bit widths**: 4, 5, 8. MiniMax runs at 5-bit; Huihui / others may
  want 4-bit. 8-bit included for completeness — may also help models we
  haven't deployed yet.
- **Head dims**: 64, 128 (first-class sizes in MLX, cover MiniMax /
  Qwen3.5 / most current models).
- **Group sizes**: 64 (the default everywhere in mlx-lm).
- **Batch size**: 1 (decode only). Prefill already uses the full
  `sdpa_full_self_attention_metal` Steel MMA path which has separate
  quant support considerations — out of scope for v1.

Other configurations fall back to the existing dequantize-then-SDPA
path.

## Correctness plan

1. **Unit test at the MLX level**: synthesize random K/V, quantize with
   `mx.quantize`, compute reference via `mx.dequantize` +
   `mx.fast.scaled_dot_product_attention`, compare against the new
   kernel output. Tolerance: bf16-precision absolute/relative diff.
   Cover head_dim ∈ {64, 128}, bits ∈ {4, 5, 8}, N ∈ {256, 8192, 66K}.

2. **Integration test via MiniMax sanity generation**: same seeded
   prompt before / after, assert token-level output match within some
   small divergence budget (quant path is numerically identical if the
   kernel is correct).

3. **Profile re-run**: `EXO_PROFILER=spans` after integration, confirm
   the `attn` span shrinks ~30–40 % (bandwidth-bound path) and total
   decode wall-time moves accordingly. (Was `EXO_MINIMAX_TRACE=1` — now
   generic via `mlx_lm/profiler.py`.)

## Perf targets

From the bandwidth analysis:

- Attn total wall-time: 180 s → ~110–130 s (−30–40 %) over the 200-token
  profile run.
- Pure decode: 17 tok/s → **21–23 tok/s** (+20–35 %).
- TTFT: likely small improvement too (~5 %), since prefill also reads
  KV but has fewer read passes per token.

If we see less than +15 % decode, the diagnosis is compute-bound (not
bandwidth-bound), which means dequantize latency wasn't the dominant
cost and the whole exercise was misdirected. That's a falsifiable exit
condition — not a sunk-cost trap.

## Risk register

- **5-bit packing layout**: MLX's packing into `uint32` words may differ
  from the straight `(pos * BITS) % 32` layout I sketched above. First
  implementation step is reading MLX's existing
  `mlx/backend/metal/kernels/quantized.h` dequantize kernel to match
  exact bit layout. If it diverges from the sketch, update the kernel
  logic to match.
- **Boundary-crossing reads**: At BITS=5, if a thread's 4-element slice
  starts at position 3 of a 6-slot word, the 4th element spills into
  the next word. The sketch handles this, but there's room for
  off-by-ones — needs careful unit tests.
- **Register pressure**: dequantize adds a few floats per-thread. Should
  fit comfortably within M4 simdgroup register budget (thousands of
  32-bit regs per simd), but worth confirming with Xcode GPU frame
  capture.
- **Metal shader compilation time**: adding new template specializations
  expands the kernel cache. Should be fine — MLX already caches many
  per-head-dim variants.
- **Upstream divergence**: if ml-explore/mlx lands its own quantized
  SDPA (issue #2955 is closed but they may revisit), we'd want to merge
  rather than maintain two. Worth a heads-up PR or issue once v1 ships.

## Implementation plan

### Session 1 (this one, ~remaining budget)

Done (deep-read + design doc, this file).

### Session 2 — kernel skeleton (0.5–1 day)

1. Read `mlx/backend/metal/kernels/quantized.h` to nail the exact MLX
   bit-packing layout for BITS=4, 5, 8.
2. Copy `sdpa_vector_2pass_1` → `sdpa_vector_2pass_1_quant` in
   `sdpa_vector.h`. Wire the new buffers through without changing
   behavior (still uses bf16 loads for the initial compile test).
3. Add the C++ dispatch function + Python binding stub.
4. Compile, confirm no regressions on the bf16 path.

**Session 2 actual outcome:**

Steps 1 and 2 landed (commit TBD). Key findings from reading
`quantized.h`:

- MLX's 5-bit packing is **8 values per 5 bytes**, not uint32-aligned
  (my original sketch in "Kernel specification" above is wrong for
  5-bit; left in the doc as a pedagogical example but the kernel uses
  the real layout).
- `quantized.h` exposes reusable helpers that avoid re-deriving the
  bit layout: `load_vector<T,U,VPT,bits>` (pre-scales the input),
  `qdot<U,VPT,bits>` (computes packed-weight dot product), and
  `dequantize<U,N,bits>` (writes a threadgroup array of dequantized
  values). The kernel uses `load_vector` + `qdot` for the K side so
  the quant layout is handled by MLX-blessed code. For V we need a
  per-thread dequantize (vs the threadgroup-writing helper); that's
  tracked in the kernel's TODO comments.
- **Discovered alignment issue at D=128, BITS=5**: the straightforward
  BD=32 / qk_per_thread=4 layout doesn't align to 5-bit's 8-value pack
  factor. Two options: widen qk_per_thread to 8 (halving the active
  thread count), or shift to BD=16. Deferred to session 3.

Landed:
- `mlx/backend/metal/kernels/sdpa_vector_quant.h` — WIP kernel draft,
  includes complete K-side with `load_vector`+`qdot`, BITS=8 fallback
  on V-side, TODOs for 4-bit and 5-bit V unpack.
- **Not yet instantiated** in `scaled_dot_product_attention.metal`, so
  it compiles as dead code. Safe to deploy — no risk to the running
  cluster until we explicitly wire it in.

Steps 3 and 4 (C++ dispatch + Python binding + smoke test) move to
Session 3.

### Session 3 — dequantize path + unit tests (1–2 days)

5. Replace the `keys[i]` / `values[i]` loads with the dequantize logic
   from the design.
6. Write the unit tests at each of (BITS × HD × N) combos listed above.
7. Fix bit-layout bugs as they surface. Iterate until tests pass.

**Session 3 actual outcome (mlx commit `f47e93a7`):**

Steps 5 landed; 6–7 deferred to session 4 because the unit-test
deliverable is gated on a callable Python entry point, and adding a new
`ScaledDotProductAttentionQuant` primitive + `mx.fast.scaled_dot_product_attention_quant`
binding is a cohesive unit of ~4 files / ~300 LOC that deserves its own
session.

Landed:
- `sdpa_vector_quant.h` — kernel body complete. `dequantize_thread<U,N,bits>`
  helper added (thread-local mirror of quantized.h's `dequantize<>`).
  V-side per-thread unpack wired for bits ∈ {4, 5, 8}. BITS=5 × D=128
  alignment resolved by widening `qk_per_thread` /`v_per_thread` to
  `max(D/32, pack_factor)`; for 5-bit at D=128 that's 8 values/thread
  and 16 active simd lanes (inactive lanes still safely contribute 0 to
  `simd_sum`).
- `scaled_dot_product_attention.cpp` — `sdpa_vector_2pass_quant(...)`
  dispatch mirroring the bf16 2-pass path, accepting
  `(k_packed, k_scales, k_biases, v_packed, v_scales, v_biases)`.
  Reuses the block heuristic and the unchanged `sdpa_vector_2pass_2`
  reduction. Decode-only v1 (mask/query_transposed fixed false).
- `scaled_dot_product_attention.metal` — 18 instantiations (3 dtypes
  × 2 head_dims × 3 bit-widths, group_size=64). The TU now transitively
  includes `steel/gemm/gemm.h` + `quantized_utils.h` to match
  `quantized.metal`.

Verified:
- `make mlx-metallib` compiles clean.
- `make mlx` links clean.
- `./tests/tests` — all 244 C++ test cases pass. Bf16 SDPA path
  unchanged.

**Scoping call for session 4:**

The dispatch function in the cpp is ready-to-call but currently
unreferenced. Wiring it through MLX's public API requires one of:

- **(A) New primitive** `ScaledDotProductAttentionQuant` in
  `fast_primitives.h` + new `fast::scaled_dot_product_attention_quant`
  entry in `fast.h`/`fast.cpp` + new Python binding in `python/src/fast.cpp`.
  Cleanest long-term, upstream-friendly, ~300 LOC across ~4 files, needs
  a wheel rebuild + uv.lock bump to exercise.
- **(C) Exo-side `mx.fast.metal_kernel`** wrapper that duplicates the
  kernel source into Python. Faster to validate but loses the benefit
  of the in-fork kernel and diverges from the design-doc plan.

The design doc chose (A). Session 4 should execute it unless the cluster
risk of a wheel-rebuild cycle changes the calculus — in which case (C)
becomes a faster-to-validate fallback.

### Session 4 actual outcome (mlx commit `21fd6384`)

Executed plan (A). **All 13 unit tests pass locally.** The v1 decode
kernel is correctness-verified against the dequantize-then-SDPA
reference path.

Landed:
- `fast_primitives.h`, `fast.h`, `fast.cpp` — new `ScaledDotProductAttentionQuant`
  primitive and `fast::scaled_dot_product_attention_quant` entry point.
- `scaled_dot_product_attention.cpp` — `ScaledDotProductAttentionQuant::eval_gpu`
  implementation that dispatches to `sdpa_vector_2pass_quant`.
- `python/src/fast.cpp` — `mx.fast.scaled_dot_product_attention_quant(q,
  k_packed, k_scales, k_biases, v_packed, v_scales, v_biases, *, scale,
  group_size, bits, do_causal=False, sinks=None)`.
- `python/tests/test_fast_sdpa_quant.py` — 13 test cases covering the
  full v1 grid: bits ∈ {4, 5, 8} × head_dim ∈ {64, 128} × N ∈ {256,
  8192, 66560}, plus fp16/fp32 dtype coverage, causal masking, and
  the prefill fallback path.

Kernel refinements (made during test-driven debugging):
- Switched the K dot-product from `qdot` to explicit `dequantize_thread`
  + manual dot. Reason: `qdot<bits=4>` requires values_per_thread % 4
  == 0 because of its positional Q-prescale contract, which would have
  forced `qk_per_thread > D/32` at D=64; explicit dot is simpler and
  matches the V side's approach.

Debugging note (for future sessions): setuptools' `build/lib.macosx*`
and `build/temp.macosx*` directories are **not** automatically
invalidated by uv when source headers change. If you edit a `.metal` or
`.h` file and re-run `uv pip install mlx`, the wheel may re-install but
ship the stale metallib. Workaround: delete those build dirs manually
before each reinstall during kernel iteration.

### Session 5 — integration + live-cluster profile

Next step per the original plan: wire the new binding into mlx-lm's
KV-cache attention path (`mlx_lm/models/base.py:117`) so MiniMax
attention actually uses it. Deploy via `start_cluster.sh` and profile
with `EXO_PROFILER=spans` to verify the predicted 20–30% decode
speedup.

#### Session 5 actual outcome (mlx `b1824c0e`, mlx-lm `77ed380`, exo `919c7c49`)

Integration landed, v1 unit tests still 13/13 green. Live-cluster bench
did NOT confirm the 20–30% decode speedup at the baseline config.

**Cluster decode tok/s at 66K context (200-tok generation) —
2-node M4 Studio tensor-parallel over jaccl RDMA, Huihui scouts
co-resident:**

| Config                     | Decode tok/s | vs 17 tok/s baseline |
| -------------------------- | ------------ | -------------------- |
| 5-bit KV, trace off        | 16.18        | **~flat (−5%)**      |
| 8-bit KV, trace off (warm) | ~11.1        | (no 8-bit baseline)  |
| 5-bit KV, trace on         | 2.24         | serialization artifact |
| 8-bit KV, trace on         | 5.74         | serialization artifact |

`EXO_PROFILER=spans` forces `mx.eval()` at every span boundary; the
forced fences serialize the new quant path much harder than the old
dequantize+SDPA path (quant dispatches 2 kernels per call vs 1 copy +
1 SDPA for the dequant path, so each fence serializes more work). Must
be disabled to measure real inference speed.

**Local microbench on the patched wheel, same (B=1, n_kv=2, head_dim=128)
shape the cluster runs:**

| bits | N=66K speedup | N=74K speedup |
| ---- | ------------- | ------------- |
| 4    | 1.28×         | 1.29×         |
| **5**| **0.96×**     | **0.90×**     |
| 8    | 1.52×         | 1.46×         |

At bits=5, head_dim=128 the kernel widens `qk_per_thread` / `v_per_thread`
to 8 to align with MLX's "8 values per 5 bytes" packing (noted in session
3). That halves active simd lanes (16 of 32) and cancels the
bandwidth-savings advantage of fusing the dequantize. At bits ∈ {4, 8}
the packed layout aligns cleanly with `qk_per_thread=4`, all 32 lanes
stay active, and the kernel wins 1.3–1.5×.

**Root cause vs the design-doc bandwidth analysis:** the analysis
modelled K/V-read bandwidth but not thread occupancy. 5-bit × head_dim=128
on M4 simdgroups is compute-starved by the pack-factor alignment, not
bandwidth-starved — so saving bandwidth doesn't speed decode up.

**A secondary bug was fixed along the way** (mlx `b1824c0e`): the v1
`ScaledDotProductAttentionQuant::eval_gpu` demanded every K/V input
be row-contiguous, which forced a `contiguous_copy_gpu` on every
`QuantizedKVCache.state` slice (~2.3 GB/decode-token at 74K). Without
that fix the first cluster run landed at 5 tok/s (3× regression). The
fix brings cluster behavior back in line with the microbench.

**Phase 2 verdict (abort criterion triggered):** per the exit rule at
the bottom of this doc, <15% decode gain on the cluster at the baseline
config is a stop signal. Halting Phase 2; not proceeding into Phase 3.

### Session 5 continuation (2026-04-24): half-pack fix + NOOP-sweep diagnosis

Took another pass at the bits=5 occupancy problem via a **branch-free
half-pack dequantize** (mlx `f784d2c3`). Paired two lanes per 5-byte
pack group using a lane-dependent shift amount instead of a divergent
`if (is_high)` branch (the first divergent attempt benched at 0.88×
— worse than the original, because SIMT serializes divergent
branches). With the branch-free rewrite the full simdgroup runs one
unified instruction stream with per-lane shift amounts.

Microbench on the cluster production shape (B=1, n_kv=2, head_dim=128):

| bits | N=66K (pre-fix) | N=66K (post-fix) |
| ---- | --------------- | ---------------- |
| 4    | 1.28×           | 1.24×            |
| **5**| **0.96×**       | **1.19×**        |
| 8    | 1.52×           | 1.71×            |

**Cluster A/B with Huihui scouts + prediction-bot paused** (5 warm
runs each, ±0.4 s wall variance):

| Build                       | Decode tok/s | vs 18.60 baseline |
| --------------------------- | ------------ | ----------------- |
| Pre-fix (b1824c0e)           | 18.61        | —                 |
| Post-half-pack (f784d2c3)    | 18.60        | **+0.0 %**        |
| + MoE gate/up fusion (6c295735) | 18.49    | −0.6 %            |

Kernel-level win is real and measurable in isolation; cluster-level
win is **zero**. After confirming the kernel is bit-exact vs
dequantize+SDPA reference on production shapes, the disconnect is not
a kernel bug — it's a **scheduler / dispatch-overhead disconnect**.

### NOOP sweep (2026-04-24): where the 54 ms/token actually goes

With three consecutive 0 % cluster results from kernel / dispatch-
level optimizations, we added section-level noop gates
(`EXO_MINIMAX_NOOP_{ATTN,SDPA,MOE,ALLSUM}`) that replace a section's
output with a shape-preserving zero tensor (answers become garbage —
purely a timing tool). Four 5-run benches on the same clean
environment:

| Config         | Wall (s) | Decode tok/s | Section removed → ms/token saved |
| -------------- | -------- | ------------ | -------------------------------- |
| Baseline       | 258.3    | 18.60        | —                                |
| NOOP_ALLSUM    | 253.8    | 19.29        | RDMA ≈ **2 ms/token** (3.7 %)    |
| NOOP_MOE       | 202.6    | 23.35        | MoE ≈ **11 ms/token** (21 %)     |
| NOOP_ATTN      | 87.0     | 55.04        | attention ≈ **36 ms/token** (66 %) |
| NOOP_SDPA      | 87.1     | 54.87        | SDPA ≈ **36 ms/token** (66 %)    |

**Definitive finding:** NOOP_ATTN and NOOP_SDPA are equal within
noise. The entire 35.8 ms/token attention section's cost is the
single SDPA kernel call per layer — Q/K/V projections, RoPE, KV
cache update, RMSNorm all combined contribute **zero measurable
time**.

**Per-SDPA-call on cluster = 35.8 ms ÷ 62 layers = 577 µs.**
Standalone microbench on M4 base = 1000 µs. Cluster is M4 Ultra with
~4× the GPU cores, so pure kernel compute ≈ 250 µs/call. That
leaves **~327 µs/call of fixed per-dispatch overhead** that no
kernel-compute optimization can touch.

**Why Phase 2's kernel win was flat on cluster:** 1.19× microbench
improvement only affects the 250 µs compute portion. 19 % of 250 µs
= 47 µs saved per call × 62 layers = 2.9 ms/token = ~5 % decode.
That's entirely within the 5-run bench's ±0.4 s wall-time noise
floor. The kernel work is correct and gives the predicted shave on
*its* piece of the budget — it's just that its piece is small
compared to the fixed-overhead piece.

### Implications for the next push

Kernel-compute wins on any algorithm (half-pack, TurboQuant, anything
else) are **capped at ≤ +25 % decode** on this cluster because the
250 µs compute floor is at most 46 % of the per-call time. Even an
infinitely fast SDPA kernel leaves 327 µs/call × 62 = 20 ms/token of
overhead → max 34 ms/token → 29 tok/s ceiling.

To break past that, the 327 µs fixed overhead must be attacked:

1. **`mx.compile` the decoder forward** — if the overhead is MLX
   graph-build + Metal command-buffer encoding, compiling the
   attention block once and reusing dispatches eliminates it.
   Potential: up to +55 % decode if the full 20 ms overhead is
   graph-building. Effort: 1–2 days.
2. **Batch multiple layers' SDPA into one dispatch** — custom Metal
   kernel that processes 2–4 layers of attention per call. Cuts
   per-call overhead by that factor. Effort: 5–7 days, uncertain.
3. **Reduce `blocks=1024` at 66K** — 1024 intermediate / sums / maxs
   buffers per call = allocator pressure that may be part of the
   327 µs. Trivial tuning experiment. Effort: 0.5 day.

`mx.compile` is the highest-expected-value move and the one the
NOOP sweep points at most directly.

**5-bit kernel variant (now mostly moot, but for reference):** the
original write-up listed `BD=16` / lane-vectorized dequant as the
"make 5-bit actually fast" path, or a `use_fallback` gate to route
bits=5 × head_dim=128 back to dequantize+SDPA as a minimum-invasive
safety net. The branch-free half-pack path landed in session 5
continuation (`f784d2c3`) and made bits=5 microbench-competitive
(1.19× at production shape). Neither moves the cluster number because
the bottleneck is the fixed per-dispatch overhead, not kernel
compute.

### Session 4 — integration + profile (1 day)

8. Update `mlx_lm/models/base.py:117` to call the new entry point when
   `cache.bits` is set.
9. Push mlx-lm fork + bump uv.lock + commit exo + deploy via
   `start_cluster.sh`.
10. Run the profile driver. Collect span breakdowns before/after.
11. Report. Decide on Phase 3 based on actual decode gain.

### Session 5+ (contingency) — boundaries / perf polish (0.5–1 day)

12. Tune block heuristic if warranted; Xcode frame capture for register
    spill / occupancy check.
13. Write up learnings for the memory + plan doc; open an issue
    upstream if the kernel is clean enough.

Total: **3.5–5 days** if things go smoothly; 5–8 days with expected
bit-layout debugging.

## Exit criteria

- **Ship v1** when unit tests pass for (bits=5, head_dim=128, HD 64)
  AND integration test produces identical token output vs dequantize
  path AND profiling shows ≥15 % decode speedup.
- **Abort** if, after 3 full days of kernel work, no measurable
  speedup vs the dequantize-then-SDPA path is reproducible on the live
  cluster — the bandwidth analysis would be wrong and we'd have to
  re-evaluate.

### 2026-04-24 session: unquant-port attempt + definitive Phase 2 close

One more kernel-compute swing before accepting the NOOP-sweep's
conclusion. Ported four optimizations from the adurham/mlx unquant
SDPA kernel (the 2-pass variant that mlx main uses) into the quant
kernel:

1. Pre-scaled Q hoisted into thread-local register array before the
   outer loop (mirrors `sdpa_vector_2pass_1`'s pattern).
2. V memory-read hoisted to the top of each iteration, parallel with
   K — lets the GPU issue both memory reads simultaneously. Matches
   upstream commit `cb95faf9`.
3. 5-byte half-pack group loaded once via `uint64` byte-assembly for
   bits=5. Reduces 8 byte reads/value to 5 byte reads + register-local
   extraction.
4. Contiguous-chunk access (each block processes positions
   `[block_idx * chunk_size, +chunk_size)` sequentially instead of
   striding by `blocks`). Matches upstream commit `0bf7df7f`.

All four landed in mlx `1f6eb6bd` and exo `15a19500`. All 13 unit
tests pass unchanged.

**Local M4 Max microbench at MiniMax production shape** (B=1, n_q=24,
n_kv=4, head_dim=128, N=66560, bf16 queries):

| bits | before | after | delta |
| ---- | ------ | ----- | ----- |
| 4    | 1363 µs | 940 µs  | −31 % |
| 5    | 1828 µs | 1300 µs | −29 % |
| 8    | 1278 µs | 950 µs  | −26 % |

**Cluster decode: 0 %.** Identical before/after at both short (500-
token) and long (50K-token) contexts. Same result as every prior
kernel-compute optimization attempt.

**Root cause confirmed via bf16-KV control experiment:** deployed
MiniMax with `MINIMAX_KV_CACHE_BITS=0` (disable quant entirely, use
bf16 KV cache). At 50K context, bf16 KV path is 3× faster than quant
in local microbench (712 µs vs ~2000 µs). **Cluster decode with bf16
KV: 27.79 tok/s. Cluster decode with 8-bit quant KV: 27.84 tok/s.**
Within 0.5 % — the 3× kernel-compute difference is invisible on the
cluster. Dispatch-scheduling, not compute, is the bottleneck.

The kernel is correct, ported, upstream-parity-clean, and fast in
microbench. It's not the cluster lever. Shipped anyway because it
costs nothing on cluster and wins +26-29 % on single-node M4 Max runs.

**Note on earlier 327 µs "fixed overhead" math:** the original
analysis assumed M4 Ultra = 4× M4 Max GPU cores. Actual ratio is ~2×
(40 → 80 cores). Corrected fixed-overhead estimate is smaller, but
the conclusion still holds: kernel-compute optimizations don't move
cluster tok/s.

**Only cluster lever that worked in this session:** `MLX_SDPA_BLOCKS`
env var added in mlx `1f6eb6bd` overrides the 2-pass blocks heuristic.
Sweep on cluster found a sharp peak at `blocks=88` giving +6.5 %
decode (26.14 → 27.86 tok/s at 50K context). Exo `6ae331fe` forwards
the env var through `start_cluster.sh` to runner processes. See
`memory/minimax_sdpa_blocks88.md` for the sweep data + cliff at
blocks=92.

**Phase 2 is definitively closed.** Next attack surface is dispatch-
count reduction via kernel fusion — see
[`minimax-fused-attention-prompt.md`](./minimax-fused-attention-prompt.md)
for the Phase 3 session prompt.
