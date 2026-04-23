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

3. **Profile re-run**: `EXO_MINIMAX_TRACE=1` after integration, confirm
   the `attn` span shrinks ~30–40 % (bandwidth-bound path) and total
   decode wall-time moves accordingly.

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
