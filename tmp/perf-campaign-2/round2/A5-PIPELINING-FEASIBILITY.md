# A5 — Cross-layer pipelining feasibility in the DSv4 forward

**Question:** Can layer N's MoE `all_sum` overlap layer N+1's attention?

Read-only source analysis. Line numbers from the working tree at
`/Users/adam.durham/repos/exo`:
- `mlx-lm/mlx_lm/models/deepseek_v4.py` (7559 lines)
- `mlx-lm/mlx_lm/models/hyper_connection.py` (645 lines)
- `mlx-lm/mlx_lm/models/switch_layers.py`
- `src/exo/worker/engines/mlx/auto_parallel.py` (2215 lines)
- `mlx/` submodule (jaccl / MLX transforms), for the collective mechanism only

Nothing was run. No source modified. This document is the only file written.

Mechanism facts about jaccl (CPU communication stream, `AllReduce::eval_gpu`
throws, Fence = whole-payload `input_coherent` + device-wide `memoryBarrier`
on the GPU side and a host busy-spin on the CPU side) are taken as ESTABLISHED
from A2/A3 this round and are not re-derived here.

---

## STEP 1 — The true data dependency chain (THE CRUX)

### 1.1 What the top-level loop does

`DeepseekV4Model._forward_steps` runs a plain sequential residual stream. There
is no branch, no double-buffer, no lookahead:

```
7045|        for _ap_i, (layer, layer_cache) in enumerate(zip(self.pipeline_layers, cache)):
7046|            h = layer(h, mask, layer_cache, inputs)
```

`h` is rebound in-place each iteration. Layer N+1 is called with exactly the
`h` that layer N returned. `mask` and `inputs` (= `input_ids`) are loop
invariants computed once before the loop (`:6907-6928` for the mask,
`:6880-6892` for the embed + hc broadcast), so those are already hoisted and
are not per-layer work.

### 1.2 What layer N returns

`DeepseekV4Block.__call__`, production (non-rowseq, non-verify) path:

```
5370|        residual = h
5371|        with span("layer.attn_hc"):
5372|            x, post, comb = self.attn_hc(h)
...
5375|            normed = finalize(self.attn_norm(x))
...
5435|            x = self.attn(normed, mask=mask, cache=cache)
5437|        with span("layer.attn_residual"):
5438|            h = finalize(hc_expand(x, residual, post, comb))
5439|
5441|        residual = h
5442|        with span("layer.ffn_hc"):
5443|            x, post, comb = self.ffn_hc(h)
5445|        with span("layer.ffn_norm"):
5446|            normed = finalize(self.ffn_norm(x))
5448|        x = self.ffn(normed, input_ids)
5450|        with span("layer.ffn_residual"):
5451|            out = finalize(hc_expand(x, residual, post, comb))
...
5455|            return out
```

Line **5451 is the residual-add**. `x` on that line is the return value of
`self.ffn(...)` at 5448, i.e. `DeepseekV4MoE.__call__`.

DSv4 is NOT a plain `h + attn + moe` transformer — it uses HyperConnections
(hc_mult=4 parallel residual streams). But the substitution is strictly
*tighter*, not looser: `hc_expand` mixes the FFN output into **every one of
the 4 residual streams** via a dense per-token `comb` matrix:

```
hyper_connection.py:486|@mx.compile
hyper_connection.py:487|def _hc_expand_op(x, residual, post, comb):
hyper_connection.py:488|    y = post[..., None] * x[:, :, None, :].astype(mx.float32)
hyper_connection.py:489|    y = y + mx.matmul(comb.swapaxes(-1, -2), residual.astype(mx.float32))
hyper_connection.py:490|    return y.astype(x.dtype)
```

`x` (the MoE output) is broadcast into all `hc_mult` output streams with
per-stream weight `post`. So **every element of layer N's output `out` depends
on the MoE output**, not just one of four streams. There is no "clean" stream
that bypasses the FFN.

### 1.3 Where the MoE output comes from — the all_sum IS the output

`DeepseekV4MoE.__call__`, tail:

```
3074|            if self.sharding_group is not None:
3075|                with span("moe.all_sum"):
3076|                    y = mx.distributed.all_sum(y, group=self.sharding_group)
...
3150|                        mx.eval(y)
3151|                    y = finalize(y)
3152|            return y
```

The `all_sum` result is rebound to `y` and `y` is the return value. There is no
alternate return path. `self.sharding_group` is set unconditionally for every
main layer by the TP sharder:

```
auto_parallel.py:1110|            layer.ffn.sharding_group = self.group
```

and the reduction is a genuine cross-rank partial-sum, because both expert and
shared-expert `down_proj` are sharded-to-all (each rank holds a slice of the
reduction dimension and produces a PARTIAL result for the full output shape):

```
auto_parallel.py:1128|            self.sharded_to_all_linear_in_place(layer.ffn.shared_experts.down_proj)
auto_parallel.py:1131|            self.sharded_to_all_linear_in_place(layer.ffn.switch_mlp.down_proj)
```

(This is the same algebra that killed prior attempt #3 — moving the `all_sum`
earlier left `shared_out` unreduced.)

### 1.4 What layer N+1's attention CONSUMES

Layer N+1 enters at 5370 with `h` = layer N's `out`. Its attention consumes
`normed = attn_norm(attn_hc(h)[0])`. Both consumers are **whole-tensor**:

- `HyperConnection.__call__` (`hyper_connection.py:441-482`) computes
  `z = mx.fast.rms_norm(y.flatten(-2), ...)` at `:468` — an RMSNorm over the
  flattened `hc_mult*hidden` axis, so every output element depends on every
  input element of that token. Nonlinear, non-decomposable.
- `nn.RMSNorm` at 5375 — same property.

Then inside `SparseCompressedAttention.__call__` every branch reads `x`:
compressor at `:4763`, `_project_qa_kv(x)` at `:4778`, `kv_norm(kv_pre)` at
`:4790`. Under seq-split the kv/compressor side deliberately stays **FULL on
every rank** (`:4734-4738` comment, `:4790`), so not even a row-band of the
input can be consumed early.

### 1.5 The chain, written out

```
layer N:   moe.all_sum(y)                        deepseek_v4.py:3076
             ↓  (return value)                   deepseek_v4.py:3152
           x = self.ffn(normed, input_ids)       deepseek_v4.py:5448
             ↓  (hc_expand mixes x into ALL 4 residual streams)
           out = hc_expand(x, residual, ...)     deepseek_v4.py:5451
             ↓  (loop rebinds h = out)           deepseek_v4.py:7046
layer N+1: x, post, comb = self.attn_hc(h)       deepseek_v4.py:5372
             ↓  (RMSNorm over flattened hc*hidden — every elem ← every elem)
                                                 hyper_connection.py:468
           normed = self.attn_norm(x)            deepseek_v4.py:5375
             ↓
           self.attn(normed, ...)                deepseek_v4.py:5435
             ↓  compressor / _project_qa_kv / kv_norm all read the FULL tensor
                                                 deepseek_v4.py:4763, 4778, 4790
```

**VERIFIED: the standard-transformer framing holds here, and DSv4's
HyperConnection structure makes it strictly stronger, not weaker.** Layer N+1's
attention input is a total function of layer N's `all_sum` output. Every
element depends on every element, through two RMSNorms. This is a TRUE data
dependency with zero decomposable slack.

---

## STEP 2 — Independent GPU work available to fill the gap

Candidates checked, one by one.

**(a) Weight prefetch / dequantization for layer N+1's experts — DOES NOT
EXIST as separable work.** There is no dequant op in the graph to hoist. MLX's
quantized path dequantizes *inside* the matmul kernel: `SwitchGLU.__call__`
(`switch_layers.py:177-203`) calls `self.up_proj/gate_proj/down_proj` (which
are `SwitchLinear` → `mx.gather_qmm`) directly on the already-quantized
weights; there is no producer node emitting an fp16 weight tensor that could be
scheduled early. Separately, the whole model is already wired-resident
(`src/exo/worker/engines/mlx/utils_mlx.py:1731-1752` calls `mx.set_wired_limit`
sized to the model), so there is no page-in/prefetch to issue either. Also
note prior campaign finding: bf16 (pre-dequantized) expert weights measure the
SAME kernel efficiency as quantized in the ragged path — pre-dequant is
independently a dead end.

**(b) The MTP / DSpark draft head — no work independent of the current layer,
and NOT LIVE DURING PREFILL AT ALL.** `DeepseekV4MTPModule.__call__` takes
`prev_hidden` (`:5514`) and immediately does `h_normed = self.hnorm(prev_hidden)`
(`:5546`) — `prev_hidden` is the *final* `pre_norm_out` of the full 43-layer
main forward, so it depends on layer 42's all_sum, i.e. on ALL of them. The
DSpark head is the same shape: it consumes `_DSPARK_CTX["hiddens"]`, populated
at `:7048-7049` from `h.mean(axis=2)` after tapped layers 40/41/42 — again
downstream, never upstream. And speculative drafting is a decode-phase
mechanism; prefill runs one forward with no draft/verify cycle.

**(c) Attention indexer / compressor / pooling — NOT precomputable ahead of the
residual stream.** `self.compressor(x, comp_cache, offset)` (`:4763`) and the
indexer (`:4828`) both take `x`, which is `attn_norm(attn_hc(h))` for the
CURRENT layer. Each layer has its own `compressor`, `indexer`, and its own
`comp_cache`/`idx_cache` (`:4729-4730`). Layer N+1's pooled KV cannot be built
from layer N's activations — the weights differ and the input is the layer's
own normed hidden. Nothing here is layer-invariant.

**(d) Per-layer work depending only on the layer INPUT and not on the
collective output — one instance found, and it is negligible.** In
`_hash_gate_route`, the expert INDICES are a pure token-ID lookup:

```
 990|    inds = tid2eid[input_ids]
```

`input_ids` is a loop invariant (`:7046` passes the same `inputs` to every
layer), so `inds` for the hash layers is computable at forward entry, before
any layer runs. But: (i) `config.num_hash_layers = 3`, so this applies to
layers 0-2 only out of 43; (ii) the accompanying `weights` on line 991 needs
`scores`, which needs `logits = (x @ weight.T)` at `:988` — activation-
dependent; (iii) it is an int32 gather of shape `(B, L, top_k)`. This is
microseconds of GPU work against a ~5-12 ms collective. It is not a filler.

**(e) A different SEQUENCE CHUNK at the same layer — this is the only genuinely
independent GPU work in the system, and it is not cross-layer.** Chunk B's
layer-N attention does not depend on chunk A's layer-N `all_sum`. This is the
already-documented "lever 2 / seq-chunk overlap" (laptop-measured ~1.1-1.15x,
bit-identical output, never integrated into the production forward loop). It is
out of scope for the question as posed but is the honest answer to "what *can*
fill the gap."

**Honest finding: apart from (e), which is a different axis entirely, there is
NO independent GPU work anywhere in the DSv4 forward that could fill a
layer-N-collective bubble.** The forward is a strictly serial 43-iteration
chain with all loop-invariant work already hoisted above the loop.

---

## STEP 3 — Structural blockers in the compressed-attention / seq-split path

These do not *cause* the impossibility (Step 1 already does), but they are real
and would independently defeat a software-pipelining scheme.

**3.1 `sharding_group` on attention means attention ends in its OWN collective.**

```
auto_parallel.py:1122|            if _DSV4_SEQ_SPLIT and type(layer.attn).__name__ in (
auto_parallel.py:1123|                "SparseCompressedAttention",
auto_parallel.py:1124|                "CompressedAttention",
auto_parallel.py:1125|            ):
auto_parallel.py:1126|                layer.attn.sharding_group = self.group
```

Setting `attn.sharding_group` activates the query-row-split path, whose tail is
a collective:

```
5052|            if _seq and _sg is not None:
...
5063|                with span("attn.all_gather"):
5064|                    if (
5065|                        _SEQ_SPLIT_GATHER_VIA_ALLSUM
5066|                        and self.sharding_group is not None
5067|                    ):
...
5083|                            mx.distributed.all_sum(
5084|                                _full, group=self.sharding_group
5085|                            )
```

Of the 43 layers, `compress_ratios` in the checkpoint config
(`DeepSeek-V4-Flash-0731/config.json`, truncated to `num_hidden_layers=43`)
gives ratio 0 for exactly layers **0 and 1** (LocalAttention, no seq-split —
`auto_parallel.py:1120-1121` explicitly refuses to set their `sharding_group`)
and 4/128 alternating for layers 2-42 (21× ratio-4, 20× ratio-128). So **41 of
43 layers have TWO GPU→CPU→GPU collective handoffs per layer, not one**:
`attn.all_gather` (via zero-padded all_sum) then
`moe.all_sum`. A pipelining scheme would have to hide both, and they are
back-to-back on the same serial chain (attn collective → hc_expand → ffn_hc →
ffn_norm → MoE → MoE collective).

**3.2 The seq-split band offers no early-availability slack.** The split is
computed per-layer from the CURRENT layer's `L` (`:4740-4753`) and only the
q-side is banded; `kv`, `compressor`, and `pool` stay FULL on every rank by
design ("so every cache + pool stays bit-identical, zero coherence risk",
`:219-221`). And the gather is a full-`L` reconstruction (`:5074-5086`), so the
attention output is not partially available either.

**3.3 jaccl pins ONE CPU stream per group and both ranks must issue collectives
in the SAME ORDER.** `JACCLGroup` constructs `communication_stream_(new_stream(
Device::cpu))` once and `communication_stream()` ignores the caller's stream
(`mlx/distributed/jaccl/jaccl.cpp:88-95`); `MeshGroup::all_sum` takes
`collective_mutex_` (`mlx/distributed/jaccl/lib/jaccl/mesh.cpp:1902-1907`).
The ctor comment states the reason explicitly: divergent dispatch order between
ranks makes the QP see mismatched post_send/post_recv interleavings and UC's
per-QP FIFO matching corrupts sends into the wrong recv buffers — this is the
documented c=2 γ=2 MTP corruption mechanism.

Implication: (i) two collectives can never actually be in flight
simultaneously — they serialize on one worker thread behind one mutex, so even
with independent work you could hide at most one collective at a time; (ii)
any reordering scheme must be **rank-symmetric and deterministic** — a
data-dependent or timing-dependent issue order is a correctness hazard, not
just a perf risk.

---

## STEP 4 — VERDICT

### **(P1) Cross-layer pipelining is structurally IMPOSSIBLE without changing the model's math.**

**The exact dependency that blocks it:** the value returned by
`mx.distributed.all_sum` at `deepseek_v4.py:3076` is the value returned by
`DeepseekV4MoE.__call__` at `:3152`, which is the `x` argument to `hc_expand`
at `:5451`, whose output is layer N's `out`, which the model loop rebinds to
`h` at `:7046`, which is the sole argument to layer N+1's `attn_hc` at `:5372`,
whose first op is an RMSNorm over the flattened `hc_mult*hidden` axis
(`hyper_connection.py:468`) — making every element of layer N+1's attention
input a function of every element of layer N's `all_sum` output.

There is no partial ordering to exploit: not per-residual-stream (`hc_expand`
mixes the MoE output into all 4 streams via `post` and `comb`,
`hyper_connection.py:488-489`), not per-row/band (seq-split keeps kv/compressor
full-`L`, `deepseek_v4.py:4734-4738`), and not per-channel (two RMSNorms
reduce across the full feature axis). Any scheme that starts layer N+1's
attention before layer N's `all_sum` retires is computing a different function.

And per Step 2, there is no *other* work to substitute: no dequant node exists
to hoist, weights are already wired-resident, MTP/DSpark are strictly
downstream of the whole layer stack and decode-only, and every per-layer
attention structure consumes that layer's own normed hidden.

**What it could have hidden (ESTIMATE, not a measurement):** from the
2026-08-21 sync-span breakdown at 100K context, `moe.all_sum` ≈ 8.2% and
`attn.all_gather` ≈ 8.5% of prefill wall — ~16.7% combined. A perfect
cross-layer pipeline would target roughly that envelope. **Treat this number as
a soft upper bound only**: sync-span mode inflates collective spans by forcing
a GPU sync at every span boundary (this repo's own documented profiler pitfall),
and jaccl's own ENTER/EXIT timing measured ~5 ms median / 9 ms p90 per call
against a 12.4 ms/call span-profile figure. The realistic hideable fraction is
somewhere in the high single digits to mid teens percent — and it is moot,
because the mechanism to hide it does not exist.

**Not P2:** P2 requires a named change with a blast radius. Every change that
would make cross-layer pipelining legal is a change to what the model computes
(dropping or approximating the residual dependency), which is out of scope for
a perf campaign and would not be "a specific named change" so much as a
different model.

**Not P3, emphatically.** Three prior attempts in this neighborhood failed, and
this analysis explains why in a way consistent with all three: `_fence_every_n`
(#1) tried to defer the fence without removing the dependency and paid 23% in
lazy-graph accumulation; the MoE gate+up fusion (#2) was orthogonal; the
intra-layer `all_sum` reorder (#3) violated the partial-sum algebra that
`sharded_to_all` on `down_proj` establishes (`auto_parallel.py:1128,1131`).
None of them had independent work to overlap with, because there isn't any.

**The adjacent thing that IS possible** (stated for completeness, not proposed
here): sequence-chunk pipelining — overlapping chunk A's layer-N collective
with chunk B's layer-N attention. That is a different axis (same layer,
different data), it is the only source of genuinely independent GPU work in
this forward, and it has a real ~1.1-1.15x laptop-measured result with
bit-identical output. It is not cross-layer pipelining and this document does
not scope it.

---

## CONFIDENCE

**HIGH (≈0.95) on the P1 verdict.** The dependency chain is five explicit
rebinds in straight-line Python with no branch, and both intervening ops are
full-axis RMSNorms. This is a code-reading conclusion that does not depend on
any measurement, any env-var configuration, or any assumption about MLX's
scheduler. The residual 5% is the possibility of a code path I did not read
that bypasses `DeepseekV4Block.__call__`'s production branch (see below).

**HIGH (≈0.9) on "no independent GPU work exists."** Checked all four
candidates the brief named plus the loop-invariant hoisting. The one thing I
found (hash-layer `inds`) is real but ~3 layers of int32 gather.

**MEDIUM on the quantitative "fraction hideable" estimate (~8-17%).** Explicitly
an estimate from a sync-mode profile with a known inflation bias, cross-checked
against a jaccl-internal timing figure that disagrees by ~2x.

---

## WHAT I COULD NOT DETERMINE

1. **Non-production block branches were not traced end to end.**
   `DeepseekV4Block.__call__` has a `_VERIFY_ROWSEQ_FULLBLOCK` per-row branch
   (`:5233-5368`) and a rowseq attention branch (`:5381-5433`), both gated on
   small `L` (≤ `_VERIFY_ROWSEQ_MAX_L`) and on env vars. I confirmed they end
   in the same `hc_expand(ffn_out, residual, ...)` structure (`:5354-5364`) and
   so carry the same dependency, but I did not exhaustively verify every gate
   combination. They are decode/verify paths, not prefill.
2. **Whether the two per-layer collectives (`attn.all_gather` then
   `moe.all_sum`) ever partially overlap in practice** — the mutex and single
   CPU stream say no, but I did not verify against a live trace, and this
   analysis was conducted without touching the cluster.
3. **The real per-call collective cost.** The two available figures (~5-12 ms
   from jaccl-internal / span profile vs. the retracted 178 ms derived number)
   were not re-measured here, so the "fraction hideable" band stays wide.
4. **Whether the DSpark path (`EXO_DSV4_DSPARK_NATIVE`) changes anything.** It
   is decode-only by construction and its ctx taps are downstream
   (`:7048-7049`), so I am confident it does not — but I did not read
   `pp_speculation.py`'s driver to confirm there is no prefill-time draft-head
   invocation.
