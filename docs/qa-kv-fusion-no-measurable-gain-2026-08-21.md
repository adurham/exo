# wq_a+wkv fusion: implemented, correct, but no measurable decode gain — 2026-08-21 (session 2, part 8)

## What was built

Per a Fable review flagging this as the direct structural analogue of
the validated MoE gate+up fusion win, restored the previously-removed
`_try_fuse_two_quantized_linears`/`_fused_quantized_matmul` helpers
(originally removed 2026-06-18 as part of a BUNDLED fusion removal after
a BS>1 degeneration bug — that removal never isolated which specific
fusion in the bundle caused the bug) scoped to ONLY the wq_a+wkv fusion,
gated behind a new `EXO_DSV4_QA_KV_FUSED` flag, default OFF, explicitly
documented as c=1-only pending separate B>1 re-validation.

Code changes (both committed and pushed before any cluster deploy, per
the standing exo git workflow):
- `mlx-lm` submodule (`adurham/mlx-lm` commit `2842133`): restored the
  fusion helpers + added `fuse_qa_kv_weights()` and the gated
  `_project_qa_kv` branch to all three DSv4 attention classes
  (`LocalAttention`, `CompressedAttention`, `SparseCompressedAttention`
  — all had byte-identical unfused `_project_qa_kv` bodies).
- `exo` (commit `8da34cb85`): wired the install call into
  `auto_parallel.py`'s per-layer loading loop (applied directly to
  `layer.attn`, no sharding-aware wrapper needed since wq_a/wkv are
  unsharded/per-rank-duplicated, unlike `ffn.switch_mlp`), bumped the
  mlx-lm gitlink, wired the env var through `start_cluster.sh`.

## Verification before cluster deploy

Offline bit-exactness test (small synthetic quantized linears,
group_size=64, bits=4, matching the real production quant scheme):
concatenated fused output split back into both halves, compared against
the unfused reference computation. **Max abs diff: 0.0 for both q and kv
halves** — genuinely bit-exact, not just "close enough."

## Cluster deploy and real-hardware test

Deployed via the standard git workflow (`start_cluster.sh` resets both
Studios to the pushed commit, reinstalls the venv). Verified
`EXO_DSV4_QA_KV_FUSED=1` live on both nodes via `ps aux`, confirmed no
"fusion failed" warning in either node's log (clean install on all 43
layers × 2 ranks).

Correctness: 100K-context needle-in-haystack passes cleanly
(`FALCON-MERCURY-7749`, correct single-pass reasoning). A separate
short-prompt quality check (CAP theorem explanation) also coherent and
correct.

Decode A/B (`bench/decode_probe.py`, same methodology as the gate+up
validation — 8 reps, small prompt, `bench: true` to force full-length
generation):

| Config | n | mean decode tok/s | stdev |
|---|---|---|---|
| gate+up fusion + qa_kv fusion (this test) | 8 | 18.789 | 0.141 |
| gate+up fusion only (prior validated result) | 8 | 18.879 | 0.158 |
| No fusion (baseline) | 8 | 18.328 | 0.173 |

**qa_kv fusion's incremental contribution over gate+up alone: -0.48%**
(essentially flat — well within the ~0.15-0.17 stdev noise band of
either measurement, not a real effect in either direction). Combined
gain over baseline (both fusions together, +2.52%) is actually slightly
LESS than gate+up alone's +3.01%, again consistent with "no real
additional effect, sampling noise."

## Conclusion

**Implemented correctly (bit-exact offline, correct on real hardware at
depth), but produces no measurable decode throughput benefit at c=1.**
Plausible explanation not independently verified this session: wq_a
(hidden_size → q_lora_rank) and wkv (hidden_size → head_dim) are much
smaller matmuls than gate_proj/up_proj (hidden_size → MoE intermediate
width, typically far larger) — the fixed per-dispatch overhead this
fusion removes may already be a smaller fraction of these two ops'
combined cost, or MLX's async scheduling may already be overlapping
these two independent (same-input, no data dependency between them)
dispatches well enough that removing one dispatch doesn't remove much
real wall-clock time.

**Recommendation: do not enable `EXO_DSV4_QA_KV_FUSED` in production.**
No measurable benefit to justify carrying an unvalidated-at-B>1 code
path. The code is committed and available (correctly implemented, bit-
exact, gated OFF by default) in case future work wants to revisit it —
e.g. if a future architecture change makes wq_a/wkv larger, or if a
direct GPU trace (the Instruments Metal System Trace flagged as
still-needed by the earlier roofline doc) reveals these specific
dispatches ARE a measurable gap that this decode probe's methodology
isn't sensitive enough to detect.

## Honest self-assessment on methodology

This result is a good illustration of why the standing rule ("never
quote t/s without real generated text, verify before declaring a win")
matters in both directions — it would have been easy to declare this a
"second confirmed win" by only checking correctness and skipping the
proper A/B against the already-fused baseline. The -0.48% number forced
an honest "no effect" conclusion instead of an inflated one.
