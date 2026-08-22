# Flag sweep completion + one real unexplored finding — 2026-08-21 (session 2, part 12)

## Context

Following up on gaps flagged when the user reviewed the todo list: two
zero-risk code-reading items and the flag sweep were incomplete or
overstated relative to what was actually done. Completing them here.

## attn.sdpa (isolated read)

`LocalAttention`'s (and by extension the other two attention classes')
`attn.sdpa` span calls `scaled_dot_product_attention` from
`mlx_lm/models/base.py`, which routes to `mx.fast.scaled_dot_product_attention`
(or the quantized-KV variant `mx.fast.scaled_dot_product_attention_quant`
when the cache carries quantization bits) — MLX's own native fused
Metal SDPA kernel, already the fastest primitive MLX offers for this
op. The wrapper also already implements a batch-row-split optimization
(`MLX_LM_SDPA_ROWSPLIT`, default on) working around a documented MLX
batched-SDPA degradation at `B>1, 1<L<=8` (measured 3.7-7.6x slower than
row-split at that shape on this exact hardware). **No further fusion or
optimization opportunity exists at this level** — this is already a
mature, previously-tuned path with its own documented benchmark
history. Confirms item #4 from the todo list as a genuine (not just
claimed) negative result.

## attn.o_proj (isolated read)

`wo_a` → `wo_b` is a sequential two-stage low-rank output projection
(`MultiLinear` down-projection feeding a regular `Linear` up-projection),
NOT a same-input parallel pair like the two fusions validated tonight
(gate+up, wq_a+wkv). `wo_a`'s output IS `wo_b`'s input — there is no
shared-input concatenation opportunity here. Collapsing this chain into
one matmul would require either dequantizing and multiplying the two
weight matrices together at load time (destroying the quantization
structure the model ships with, likely at real accuracy cost) or a
genuinely new kernel design — a different and larger scope than the
"concatenate two same-input quantized linears" pattern that worked
twice tonight. **No simple fusion exists here.** Confirms item #5 as a
genuine negative result, not a skipped one.

## Flag sweep: verified rather than assumed

Checked the remaining flags identified but not fully resolved in the
original sweep, against BOTH their code defaults and the live
production process env (`ps aux` on both nodes):

- `EXO_DSV4_DECODE_NODE_DIET` (default 1), `EXO_DSV4_EXACT_TOPK`
  (default 1), `EXO_DSV4_SINGLE_GATHER` (default 1) — all three default
  ON and are NOT overridden in production, meaning **they are already
  the active code path**. Nothing to test; the "off" state was never
  the live behavior to begin with.
- `EXO_DSV4_CATTN_LSPLIT_MAX_L` (default 8) — a tuning constant, not a
  binary lever; already applied at its default value in production.
- `EXO_DSV4_BATCH_INVARIANT_MM` (default off) — a correctness-safety
  fallback (per its own code comment: forces per-row gemv computation
  for batch-invariance, explicitly trading speed for a batching
  guarantee). Enabling it can only ever be SLOWER by design — not a
  real performance lever, correctly off.

## One genuine, real, unexplored finding: `EXO_DSV4_DSPARK_NATIVE`

Unlike the flags above, this one is NOT already resolved. Confirmed via
live `ps aux`: `EXO_DSV4_DSPARK=1` (DSpark speculative-decode draft head
active, `EXO_DSV4_MTP=0`) but `EXO_DSV4_DSPARK_NATIVE` is unset (default
0 — the separately-converted local-overlay draft head path,
`_overlay_dsv4_dspark`, gated by `EXO_DSV4_DSPARK_DIR`).

The code's own comment (`utils_mlx.py`, 2026-08-04) states explicitly:

> `EXO_DSV4_DSPARK_NATIVE=1` selects the checkpoint's OWN bundled
> mtp.0/1/2.* DSpark head instead of the separately-converted local
> head — use this for checkpoints (**e.g.
> `deepseek-ai/DeepSeek-V4-Flash-0731`**) that ship their own trained
> draft weights, so verify runs against a head actually trained on
> THIS checkpoint's hidden-state distribution rather than a different
> (e.g. preview) checkpoint's.

**The model string named in that comment as the intended use case,
`deepseek-ai/DeepSeek-V4-Flash-0731`, is the exact model currently
running in production tonight** — and production is running with the
flag OFF (the fallback/mismatched-head path), not ON (the
comment-recommended path for this exact checkpoint).

This is categorically different from every other lever tested tonight:
it's not a microsecond-level dispatch/fusion optimization, it's a
speculative-decode DRAFT HEAD selection — the practical effect (if the
comment's reasoning holds) would be on MTP/DSpark draft acceptance rate
(a higher-quality, checkpoint-matched draft head accepts more
speculative tokens per verify cycle, raising effective decode
throughput indirectly), not a direct kernel/dispatch win. It also
carries real risk profile questions the other levers didn't: does the
checkpoint actually ship a bundled native head? Does switching draft
heads mid-deployment risk a quality regression if the native head is
less mature than the long-used local-overlay path? These are exactly
the kind of unknowns that warrant a careful, dedicated test — not a
quick toggle-and-check like tonight's env-var levers — and per the
explicit "stop live-hardware testing tonight" direction from review #4,
this was NOT tested live this session.

**Correction (2026-08-21, same session): OUT OF SCOPE for this line of
work.** User caught this: MTP/DSpark speculative-decode draft-head
acceptance rate is a decode-phase-only mechanism — prefill processes
the full prompt in one forward pass with no speculative
drafting/verification cycle at all, so `EXO_DSV4_DSPARK_NATIVE` has
zero mechanism to affect prefill throughput, which is the focus of this
optimization line. Deferred, not deleted — still a real, well-reasoned,
untested candidate for a FUTURE decode-specific session, but does not
belong in the prefill-focused queue (items 12-14 in the campaign
summary, all of which genuinely apply to both prefill and decode since
`moe.all_sum`/`moe.switch_mlp` fire on every layer regardless of phase).
