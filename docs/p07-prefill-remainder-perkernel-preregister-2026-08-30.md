# P07 pre-registration — per-kernel GPU capture of prefill's remaining non-GEMM remainder (2026-08-30)

**Gates and criteria below are registered BEFORE any measurement is taken.**
Nothing in this document may be edited after the first capture runs; results go
in a separate results doc.

## 1. Why this is not a re-litigation of T10

T10 (`docs/t10-final-decomposition-closed-2026-08-22.md`) decomposed prefill's
28.8% non-GEMM remainder and closed it: "no async-fence-class hidden bug."
That closure was honest and is not being disputed. It named its own reopening
condition explicitly:

> "No further prefill-side dispatch/gate-hunting is recommended without new
> evidence (e.g. ... a new methodology that reveals something the span-profile
> approach can't see)."

Two things now satisfy that condition:

1. **T10's span table is stale by construction.** Two fused Metal kernels
   shipped default-ON *after* T10 closed, both targeting spans inside the
   remainder — `hc_expand` (2026-08-24, `deb1c8a6d`, 8.66x op-path speedup,
   +3.87% e2e @70.5K) and `hc_collapse` (2026-08-25, `99f5f96b8`, 2.47x span
   speedup, +1.89% e2e, depth-verified at 300K/500K). The remainder T10
   described no longer exists at the size it described.
2. **Per-kernel GPU capture at PREFILL shape has never been done.** P03
   (2026-08-30) ran the first per-kernel capture of the small-op bucket, but at
   DECODE shape (L=1, L=4). Every "at ceiling" verdict inside the prefill
   remainder rests on FLOP/bandwidth arithmetic or CPU-side microbenches, not
   on measured GPU kernel time.

The second point is load-bearing, and the indexer decomposition proves the
span-profiler is structurally blind here in a way that is already documented:

> `indexer.score` (8.34 µs/call) and `indexer.topk` (5.81 µs/call) inner spans
> "only measure **lazy-graph build time**, not GPU compute — sanity: if the
> score GEMM at 220K (14.42 GFLOP) executed in 8.34 µs it would be 1.7 PFLOPS,
> physically impossible" (`indexer-prefill-decomposition-2026-08-24.md:116-119`)

and that doc's own caveat (:254): **"The 'score GEMM at ceiling' claim is
inferred, not directly measured."** Under MLX's lazy executor, a span timer
that does not end at an eval barrier measures graph construction, not work.
That is exactly what per-kernel capture fixes and span-profiling cannot.

## 2. Corrected remainder composition (the real starting point)

Recomputed from raw ms totals in `docs/dsv4-220k-prefill-span-profile-2026-08-18.md`,
adjusted for kernels shipped after T10. Percentages are of total prefill wall time.

| Span | T10 (08-22) | Change since | Today (est.) | GPU-kernel measured at prefill shape? |
|---|---|---|---|---|
| `moe.all_sum` | 9.5% | none | 9.5% | **OUT OF SCOPE** — closed separately |
| `moe.post_combine` (incl. shared_experts fwd) | 4.2% | none | 4.2% | NO — FLOPs arithmetic + CPU microbench |
| `attn.indexer` | 4.0% | decomposed 08-24, nothing shipped | 4.0% | NO — explicitly inferred (:254) |
| tail spans (7: rope_in/out, kv_cache, mask, norms, compressor) | ~2.5% | none | ~2.5% | NO — dispatch-floor arithmetic only |
| HyperConnection (`attn_hc`/`ffn_hc`) | 4.6% | **hc_collapse shipped 08-25 (2.47x)** | **~1.9%** | Partially — kernel benched, not at prefill shape in situ |
| `moe.gate` | 0.9% | none | 0.9% | NO — CPU microbench (1.54x) |
| `hc_expand` (`attn`/`ffn_residual`) | 4.4% | **hc_expand shipped 08-24 (8.66x)** | **~0.5%** | Partially — same as above |
| **Total remainder** | **~28.8%** | | **~23.5%** | |
| **Non-`all_sum` remainder (the target)** | **~19.3%** | | **~14.0%** | |

**First real finding of this phase, before any new measurement:** ~5.3pp of the
original 19.3% was a genuine, fixable bottleneck — and it is *already fixed and
shipped*. The open question is now scoped to ~14.0%, not 19.3%.

`lm_head` is NOT in the prefill remainder: the mxfp8 ship commit (`80ec8ec03`)
records "Prefill unchanged (378.3 vs 378.0)", and `model.lm_head` is 0.2% of
prefill wall (once per chunk, not per layer). Decode-only win.

## 3. Pre-registered verdict gates (per span)

For each span, measure at real production prefill shape (`L=2048` chunk,
`L_band=1024` per rank under `EXO_DSV4_SEQ_SPLIT=1`, TP=2, bf16):
- **GPU-busy kernel time** via `MLX_GPU_TIME=1` (real `GPUEndTime-GPUStartTime`)
- **dispatch count** via `MLX_DISPATCH_COUNT=1`
- **the correctly-denominated ceiling** — chosen per-op by regime, decided and
  written down BEFORE computing efficiency.

**Regime choice is itself a pre-registered gate.** T10 banked the lesson that
applying a FLOPs ceiling to a memory-bound op produced a bogus "~100x over
ceiling" that collapsed to 1.93x once corrected. So: an op is denominated
against bandwidth iff arithmetic intensity < 10 FLOP/byte, against FLOPs
otherwise; the computed intensity is reported alongside every verdict.

Verdicts, assigned mechanically:

| Verdict | Criterion |
|---|---|
| **AT CEILING** | ≥70% of correctly-denominated ceiling. Closed, no action. |
| **DISPATCH/LATENCY FLOOR** | GPU-busy time <40% of span time AND dispatch count already minimal for the op's semantics (i.e. cost is latency/sync, not work, and cannot fall without reducing op COUNT). |
| **REAL HEADROOM** | <40% of ceiling AND ≥1.0% of prefill wall time AND a concrete mechanism identifiable. |
| **MODEST** | 40–70% of ceiling. Report mechanism + e2e estimate; ship only if ≥1% e2e and quality-neutral. |

**Ship gate for any candidate fix (all four required):**
1. Predicted e2e win ≥1.0% of prefill wall, using the hc_expand-proven triage
   product: `span_share × per-op-reduction ≈ e2e win` (that product predicted
   3.85% vs 3.87% measured — use it BEFORE building anything).
2. The fix reduces **dispatch COUNT**. Per the lever-exhaustion map, every
   fusion that merely re-implemented an MLX primitive failed once pipelined
   (indexer fused kernel 0.54x = slower; wq_a+wkv -0.48%). Only
   dispatch-count-reducing fusions have ever worked here.
3. Validated with a **pipelined** microbench, never per-call-in-isolation —
   MLX's async executor already overlaps dispatch across the op chain, so
   isolated per-call savings estimates are systematically inflated.
4. Numerics gate: mean relative error <0.2% against the production path.
   (hc_expand's 1.41x bf16-comb variant was rejected at 1.08% and that
   rejection was upheld multi-seed — this threshold is precedent, not new.)

**Phase-level conclusion gate:** if every span lands AT CEILING or
DISPATCH/LATENCY FLOOR, the honest verdict is *irreducible floor* and the
remainder is closed for good on this methodology — recorded as such, not
softened into "needs more investigation."

## 4. Secondary item — the SDPA ceiling-denominator audit

T7 flagged but never checked whether `attn.sdpa`/`attn.sdpa.compressed` ceiling
denominators were inflated by causal-masking / MLA-absorption FLOP over-count.
Audited this phase (code reading + arithmetic, `bench/attn_production_class_bench.py`):

- **MLA absorption: not in play.** DSv4-Flash's config has no `kv_lora_rank`
  (unlike v2/v3); KV projects directly `hidden(4096) → head_dim(512)` and is
  stored at full head_dim. The naive D=512 formula is correct. No over-count.
- **`attn.sdpa` (sparse): denominator CORRECT.** `_sparse_pooled_attention_inner`
  computes a **dense** local matmul over all 2175 local keys and applies the
  causal mask *after* (`_apply_score_mask`) — the masked positions really are
  computed. Counted FLOPs match real work. **61.7% stands.**
- **`attn.sdpa.compressed`: denominator INFLATED ~1.65x.** The bench counted
  full `L_band × CATTN_KV` work using a synthetic 95%-dense mask, but production
  passes a real **causal** mask to `mx.fast.scaled_dot_product_attention`.
  Real causal work = 158.3 GFLOP vs 261.3 GFLOP counted.
  Corrected efficiency = 79.1% / 1.65 = **~48%**, not 79.1%.

**This is UNDETERMINED pending one runtime measurement**, and is registered here
as a pre-declared test rather than a conclusion: the corrected ~48% assumes the
fused kernel actually exploits the causal mask to skip work. Decisive test —
time `mx.fast.scaled_dot_product_attention` at production shape with a causal
mask vs a dense mask:
- causal materially faster than dense → kernel exploits the mask → real
  efficiency ≈48%, and this span has **more** headroom than T7 believed
  (it was recorded as "at ceiling, dead end").
- causal ≈ dense → kernel does the full work → **79.1% stands**, denominator
  fine, no correction needed.

Note the direction: this correction, if confirmed, does not shrink the
opportunity — it means a span previously written off as at-ceiling has real
headroom. It sits in the 71.2% GEMM-covered majority, not in the remainder.

## 5. Method constraints (inherited, non-negotiable)

- Capture runs as a **standalone process beside the live runner**, never inside
  it (p01/P03-proven). Target: zero cluster relaunches.
- Verify all production flags before AND after; confirm continuous runner
  uptime; real post-capture smoke test before any capture is treated as valid.
- `.eval()` every constructed module before timing — `nn.Module.training`
  defaults True and silently benchmarks HyperConnection's slow pure-MLX path
  (8.8x error, cost a false lead once already).
- Never force evaluation with arithmetic that cancels (`*0`) — MLX constant-folds
  the whole graph away and reports a fictional ~500,000 GB/s.
- Rotate input banks past the L2 boundary before quoting any GB/s.
- Confirm the expected number of distinct gputrace bundles exists after capture
  (gputrace fails silently on filename collision).
- Every headline number re-derived against raw `results.json` before commit.
