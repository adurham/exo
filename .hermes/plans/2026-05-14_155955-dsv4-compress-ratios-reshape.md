# DSv4 compress_ratios reshape — reduce sparse-attention layer count for decode-time wall

## Goal

Reduce the number of expensive SparseCompressedAttention layers in DSv4-Flash
by reshaping `compress_ratios`. The May 14 NOP sweep proved sparse_attn is the
#1 non-MoE lever (+31% at 100K). The architectural lever proven WRONG (expert
co-location, no cross-rank fetch to optimize) and the kernel lever proven
WRONG (1.23x pipelined < 1.7x gate) both shifted the headroom search HERE.

Reshape options (in order of invasiveness):

  Option-A (CHEAPEST): demote a subset of sparse layers (ratio=4) to local
    (ratio=0). Removes both the indexer AND the sparse-pooled SDPA path on
    those layers. Per-layer wall savings: ~143 µs sparse + indexer overhead
    on top of that. But: those layers lose long-range attention entirely
    (sliding_window=128 only). Quality risk: HIGH on long context.

  Option-B (MEDIUM): demote sparse (ratio=4) to compressed (ratio=128) on a
    subset of layers. Layer still has a compressor and CompressedAttention,
    but NO indexer and NO sparse-pooled gather. Saves ~70 µs/layer on
    promoted layers. Lower quality risk than A because the layer still
    sees a full compressed view of the past.

  Option-C (MOST INVASIVE): change MoE/attention layer counts (e.g.,
    interleave attention-only or FFN-only layers). Out of scope — touches
    the whole architecture.

This plan targets **Option-B**. Demote the LATEST sparse layers (which routing
data shows have HIGHEST concentration and may benefit LEAST from sparse
indexing) to compressed, and measure.

## What the data justifies

From the NOP sweep at 100K c=1 MTP-off (29.2 t/s baseline):

  - sparse_attn NOP: 38.3 t/s (+31.2%)  → 21 layers worth of savings
  - compressed_attn NOP: 32.9 t/s (+12.7%)  → 20 layers worth of savings
  - indexer NOP: 33.4 t/s (+14.4%)  → 21 indexers (sparse layers)
  - per-layer sparse cost: ~143 µs (3 ms / 21 layers)
  - per-layer compressed cost: ~70 µs (1.4 ms / 20 layers)
  - per-layer sparse→compressed savings: ~73 µs

If we demote 8 of the 21 sparse layers to compressed:
  - Savings: 8 × ~73 µs = ~0.58 ms/token
  - At 28.5 ms baseline: 1000/(28.5-0.58) = ~35.8 t/s = +6.5%

If we demote 12 sparse layers:
  - 12 × 73 = ~0.88 ms = ~36.4 t/s = +8% but with bigger quality risk.

This is **smaller** than the kernel-fusion plans projected, but it's BACKED
by direct NOP measurement — no speculation about dispatch overhead or memory
bandwidth. We KNOW what skipping a sparse layer costs because we measured it.

## Current state (the actual loaded ratios)

From `~/.exo/models/mlx-community--DeepSeek-V4-Flash-8bit/config.json`:

  L00:   0   LocalAttention
  L01:   0   LocalAttention
  L02:   4   SparseCompressedAttention   ← demotion candidate
  L03: 128   CompressedAttention
  L04:   4   SparseCompressedAttention   ← demotion candidate
  L05: 128   CompressedAttention
  ... [alternating 4/128 pattern]
  L42:   4   SparseCompressedAttention   ← demotion candidate
  L43:   0   (MTP layer, separate from num_hidden_layers=43)

So `compress_ratios` has 44 entries; the last (L43) is the MTP block we
don't touch. Of the 43 hidden layers:
  - 3 LocalAttention (L00, L01, plus MTP L43)
  - 21 SparseCompressedAttention (even-indexed L02..L42 stepping by 2)
  - 20 CompressedAttention (odd-indexed L03..L41 stepping by 2)

Wait — actually L00 and L01 are BOTH local. The MTP block (L43) is also
local but is configured separately from the 43-layer main body. So
the 21 sparse / 20 compressed / 2 local pattern WITHIN the body holds.

## The hard part: checkpoint mismatch

Trained weights expect each layer to have the modules its trained
compress_ratio dictates:

  - ratio=0 (LocalAttention):  wq_a/wq_b, kv projections, RMSNorms.
  - ratio=128 (CompressedAttention): all of LocalAttn + `compressor.*`
    weights for the compressed pool.
  - ratio=4 (SparseCompressedAttention): all of CompressedAttn +
    `indexer.*` (wq, weights_proj, RoPE) for the sparse selection.

If we demote `L02: 4 → 128`, the checkpoint has `layers.2.attn.indexer.*`
keys that the new CompressedAttention has no slot for. mlx-lm's `sanitize`
in deepseek_v4.py:2281 doesn't filter these. With `load_weights(strict=True)`
this would error.

**Two options to handle this:**

  Option B.1 (CONSERVATIVE): Add a sanitize-time filter that drops orphan
    `indexer.*` keys when the runtime compress_ratio differs from trained.
    Quality risk: we're throwing away learned routing/selection that the
    layer's other weights presumably co-adapted to.

  Option B.2 (RIGOROUS): Add a compatibility layer that LOADS the indexer
    weights but uses them differently — e.g., keep the indexer running but
    swap to compressed-only attention output. Lower quality risk but more
    code.

This plan starts with B.1 because it's the SIMPLER experiment. Quality
gate after each demotion determines whether to escalate to B.2.

## Why NOT just touch index_topk first?

We already tried: `EXO_DSV4_INDEX_TOPK=96` made things WORSE (28.5 → 33.2 ms
fwd_build at 100K). Lower topk pushes MORE pool positions into the sparse
fallback path (pooled.shape[1] > index_topk fails more often). Documented
in skill `distributed-bottleneck-attribution.md` and warm memory.

This plan is OPPOSITE of that — fewer SPARSE LAYERS rather than smaller
topk. We're cutting layers from the slow path, not slowing the slow path
on more layers.

## Step-by-step plan

### Phase 0 — Pick demotion candidates (0.5 day)

The 21 sparse layers are L02, L04, ..., L42. Which 4-8 to demote first?

Three candidate selection heuristics:

  H1 (LATEST layers first): demote L34, L36, L38, L40, L42 — the last 5
    sparse layers. Rationale: later layers have shorter "remaining"
    forward path. If they make a worse choice, the model has less time
    to recover. But empirically transformer architectures often have
    LATE layers doing the most important "decision making" — so this is
    risky.

  H2 (EARLIEST layers first): demote L02, L04, L06, L08, L10 — the first 5.
    Rationale: early layers do more "input recognition" — they may benefit
    LESS from long-range sparse attention since they're still building
    representations from tokens near the current position.

  H3 (LEAST-USEFUL-INDEXER): use the May 14 routing histogram to find
    layers where the sparse_attn output contributes LEAST to the residual.
    We don't have this signal yet — would need a new probe that NOPs each
    sparse layer INDIVIDUALLY and measures quality delta. Out of scope
    for phase 0; revisit if quality concerns force it.

This plan starts with **H2 (earliest first)** because:
  (a) Earliest sparse layer pool sizes are SMALLEST (compressor pool grows
      with context length), so sparse-vs-compressed differential is
      smallest there — least quality cost per demotion.
  (b) MoE concentration data shows late layers (L14-L42) have HIGHER
      per-layer concentration (top-32 = 60-90% vs L02-L10 = 30-55%).
      Late layers are more specialized — less safe to demote.
  (c) Compressed pool growth pattern: compress_ratio=4 means pool grows
      every 4 tokens, ratio=128 every 128 tokens. Early layers' pools
      grow fastest under ratio=4; demoting to 128 means the layer sees
      a coarser view but with much smaller pool size growth.

### Phase 1 — Single-layer demotion smoke test (1 day)

1. Implement env-gated override: `EXO_DSV4_DEMOTE_SPARSE="2,4,6"` would
   change L02, L04, L06 from ratio 4 → 128. Format: comma-separated
   integer layer indices.

2. Modify `mlx-lm/mlx_lm/models/deepseek_v4.py:ModelArgs.__post_init__`:
   parse the env var and rewrite `self.compress_ratios[i] = 128` for
   each `i` in the list. Validate i is a sparse layer originally.

3. Modify `sanitize()` (line 2281) to drop orphan indexer weights:
   ```python
   demoted = set(int(x) for x in
                  os.environ.get("EXO_DSV4_DEMOTE_SPARSE", "").split(",")
                  if x.strip().isdigit())
   for k in list(new_weights.keys()):
       parts = k.split(".")
       if (len(parts) >= 4 and parts[0] == "layers"
           and parts[2] == "attn" and parts[3] == "indexer"
           and int(parts[1]) in demoted):
           del new_weights[k]
   ```

4. Wire env var through start_cluster.sh.

5. **Smoke test (single-node first, before cluster):**
   `EXO_DSV4_DEMOTE_SPARSE=2 uv run python3 -m mlx_lm.generate ...` — make
   sure the model loads at all with one demoted layer.

6. If load succeeds, run a short greedy decode (50 tokens, temp=0) with
   a fixed prompt. Compare against baseline (`EXO_DSV4_DEMOTE_SPARSE=""`).
   Tokens that match for the first ~20 are a good smoke signal.

### Phase 2 — Quality gate (1 day)

For each demotion config, run:

1. **8K c=1 quality probe**: needle-in-haystack at 8K. Did the model still
   find the needle? Lower context = less reliant on sparse layers, should
   PASS easily.

2. **100K c=1 quality probe**: needle at 100K (the bench/quality_probe_dsv4
   harness). This is the real test — sparse layers exist precisely for
   long-range. The model MUST still find the needle.

3. **Decode coherence**: a 256-token continuation on a creative prompt
   ("Once upon a time..."). Read the output. Does it look like the model
   it was, or has it degraded into noise?

Quality gates per demotion count:
  - 1 layer demoted: needle MUST still be found. Tolerance: 100%.
  - 3 layers demoted: needle MUST still be found. Tolerance: 100%.
  - 5 layers demoted: needle MUST still be found.
  - 8 layers demoted: needle SHOULD still be found (1 failure of 3 trials).
  - 12 layers demoted: STOP. Below this is likely going to break things.

### Phase 3 — Throughput measurement (1 day)

Only after a configuration passes quality gates:

  - 100K c=1 MTP-off via `concurrent_bench.py --concurrency 1 ...
    --prompt-words 75000 --max-tokens 256`.
  - Compare against champion 29.47 t/s.
  - Target: each demoted layer should add ~73 µs of savings, so:
    - 4 layers → +1% throughput (in-noise)
    - 8 layers → +2-3% throughput
    - 12 layers → +3-4% throughput

  Wait. Re-read the May 14 NOP data:
    - sparse_attn NOP (skip ALL 21) = +31% (3 ms saved)
    - That means EACH sparse layer's WALL contribution is ~143 µs (3/21).
    - Demoting 4→128 turns a sparse layer into compressed.
    - Compressed-attn NOP (skip ALL 20) = +13% (1.4 ms saved).
    - Each compressed layer = ~70 µs.
    - So sparse→compressed conversion saves 143-70 = ~73 µs/layer.
    - 4 layers × 73 µs = 0.29 ms = ~1% throughput. Marginal.
    - 8 layers × 73 µs = 0.58 ms = ~2% throughput.
    - 12 layers × 73 µs = 0.88 ms = ~3% throughput.

  This is MUCH SMALLER than the headline NOP numbers suggested. The reason
  is that we're NOT removing the attention layer — we're DOWNGRADING it
  from sparse to compressed. The compressed path is still expensive.

  **REVISED EXPECTATION:** +2-4% throughput per 8 demoted layers, AT MOST.
  29.47 → ~30.5 t/s. This is a smaller payoff than I initially advertised.

  To get the BIG win (the +31% from sparse_attn NOP), we'd need to skip
  the sparse path ENTIRELY — i.e., demote sparse → LOCAL (ratio=0). That
  loses the compressor pool too. Quality risk is much higher.

### Phase 4 — Decision

Gate: ≥ 5% throughput AND quality passes (needle found at 100K, decode
coherent). If both met, ship behind `EXO_DSV4_DEMOTE_SPARSE` flag,
default empty.

If quality fails OR throughput < 5%: abandon. Move to escalation:

  - Try LOCAL demotions instead (4 → 0). Bigger wall savings (sparse 143
    µs → 0 ~= save full per-layer cost). But sliding_window=128 means
    those layers see ONLY 128 most-recent tokens. At 100K that drops
    99.9% of context. Quality risk: SEVERE.

  - Try a HYBRID: demote sparse → compressed on early layers; demote
    compressed → local on a few middle layers. Stacks the savings.

  - Accept the ceiling at ~30 t/s.

## Files to change

mlx-lm submodule:
  - `mlx_lm/models/deepseek_v4.py`:
    - `ModelArgs.__post_init__`: parse `EXO_DSV4_DEMOTE_SPARSE` env var
      and rewrite `self.compress_ratios[i] = 128` (or `= 0` in escalation).
    - `sanitize()`: drop orphan `layers.N.attn.indexer.*` keys for
      demoted layers.

exo repo:
  - `start_cluster.sh`: propagate `EXO_DSV4_DEMOTE_SPARSE` env.

No new files needed.

## Tests / validation

Smoke (single-node):
  - Model loads with `EXO_DSV4_DEMOTE_SPARSE=2` (no missing-key errors,
    no extra-key errors).
  - 50-token greedy decode at temp=0 produces non-garbage output.

Quality (cluster):
  - `bench/quality_probe_dsv4.py --target-tokens 100000` — needle found.
  - Decode coherence check by human inspection.

Throughput (cluster):
  - `bench/concurrent_bench.py --concurrency 1 --iterations 2
    --max-tokens 256 --prompt-words 75000`
  - Compare median agg_tps to champion 29.47.

## Risks, tradeoffs, open questions

**Risk 1: Sparse layers may be doing critical work.** They exist because
DSv4 was trained to use them. Removing the sparse selection on 8/21 of
them is non-trivial damage. Mitigation: phase 2 quality gate. If needle
fails, we stop.

**Risk 2: The compressor weight inside the demoted layer may have
trained-with-sparse-attention adaptations.** Even keeping the compressor
but dropping the indexer means the layer's output distribution shifts.
Mitigation: phase 2 decode coherence check.

**Risk 3: The savings projection assumes the per-layer wall delta from
NOP is the per-layer cost.** This is correct for "isolated cost" but
the cluster's all_sum fence is the dominant cost overall. If demoting
a sparse layer doesn't ALSO remove a sync point, the all_sum fence still
fires — savings would be SMALLER than projected. Same lesson as the MoE
and sparse_attn fused-kernel failures.

**Risk 4: Indexer NOP also gave +14%. That's separate from sparse_attn
NOP's +31%.** Demoting sparse→compressed eliminates the indexer too, so
we might capture some of the +14% indexer gain on top. But the additive
ceiling is bounded by `(143 µs sparse + indexer overhead) - 70 µs
compressed = ~120 µs/demoted-layer` at most. 8 layers × 120 µs = ~1.0 ms
= 3.5% — still in the 2-4% range, not transformative.

**Risk 5: This may be a small lever.** Honest assessment: the projected
+2-4% is modest. The user has been clear that we should "do the work
inline or via delegation; don't punt back" and aggressively questions
premature ceilings. This plan ACKNOWLEDGES the ceiling but proposes
the cleanest remaining experiment to confirm or refute it.

**Open question 1:** Does demoting actually remove the sparse-attn
all_sum sync? Probably not — the all_sum is per-attention-layer (any
attention layer), not per-sparse-layer. So this lever does NOT
attack the cluster fence cost. Re-read auto_parallel.py to confirm.

**Open question 2:** Would demoting sparse → local on JUST L00..L05
(the first 4-6 sparse layers, which have smallest pool sizes anyway)
be safer than going compressed? Smaller pool means smaller difference
between sparse and local at those layers. Worth a phase-1 sub-experiment.

**Open question 3:** Is there a published DSv4 ablation showing
sensitivity to compress_ratios in published work? If so, that gives us
a prior on quality risk. Probably not since DSv4-Flash is recent.

## Rollback path

Champion tag `champion-2026-05-13-29.47` on both repos.
Rollback script `~/.hermes/scripts/rollback_to_champion.sh`.
`EXO_DSV4_DEMOTE_SPARSE` defaults to empty string = no demotion = champion
behavior.

## Why this is the right (last reasonable) target

After three plan abandonments (MoE kernel, sparse_attn kernel, expert
co-location), the remaining levers in the search space are:

  - Lever B (compress_ratios reshape): THIS PLAN. Architectural, env-gated,
    backed by direct NOP measurement.
  - Lever C (mixed-precision quant on cold experts): now known to be
    decode/prefill divergent, harder to do safely.
  - Accept the ceiling at champion 29.47 t/s.

This plan is the cleanest remaining experiment. The projected payoff is
modest (+2-4%, getting champion to ~30-31 t/s) but it's the most likely
to actually work because:

  (a) NOP data measured the savings DIRECTLY — no microbench extrapolation.
  (b) The change is small (one env var, ~30 lines of code).
  (c) Quality gate is well-defined (needle probe is reliable).
  (d) Rollback is trivial.

If phase 1-2 succeed and we get +5% real-cluster, that's the new champion.
If not, we accept the ceiling.

**Expected delivery:** 29.47 → 30.5-31.0 t/s, 4-8 layers demoted,
~1 week of work.
