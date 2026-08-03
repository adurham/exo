# DSpark Native Head for DeepSeek-V4-Flash-0731 (2026-08-03)

**STATUS: NOT STARTED. Planning only.** This doc is the handoff for the next
session — everything here is grounded in actual code reads, not guesses, but
zero implementation has happened yet.

## TL;DR for whoever picks this up

`deepseek-ai/DeepSeek-V4-Flash-0731` is deployed and live on the cluster
(correctness verified, throughput verified). DSpark speculative decoding is
currently **disabled** (`EXO_SPECULATIVE=0 EXO_DSV4_DSPARK=0`, a manual
runtime override — `start_cluster.sh`'s own defaults are still
`EXO_SPECULATIVE=1 EXO_DSV4_DSPARK=1`, so a plain relaunch reverts to DSpark
on) because a controlled A/B showed no throughput benefit from DSpark on this
checkpoint. The likely cause: the DSpark draft head currently in use is
trained on the **preview** checkpoint's hidden states, not `-0731`'s (which
was re-post-trained for agentic tasks and may have shifted its internal
representations enough to hurt draft acceptance). `-0731`'s checkpoint ships
its **own** trained DSpark head, but exo's loader doesn't know how to read it
yet — this doc is the plan to fix that.

## Background: how we got here

Session timeline (2026-08-02/03, see warm memory facts 1142-1156 for full
detail, `memory(action='recall', query='DSpark native 0731')`):

1. Deployed `-0731`, verified architecture/tokenizer/encoding compatibility.
2. Fixed two real bugs found along the way: stale `reasoning_effort` prompt
   semantics in the vendored encoder (commit `1cd33b74a`), and a misleading
   "could not find MTP weights" warning that fired from a dead code path for
   all DSv4 models (commit `b637520d4`).
3. Ran a controlled throughput A/B (list-of-100-primes task, temp=0,
   max_tokens=300, forces `finish_reason=length` so decode is measured at
   steady state, not skewed by early convergence):
   - **DSpark ON** (preview-vintage local head): 3 runs, mean 23.62 tok/s
   - **DSpark OFF** (no speculation): 6 runs, mean 24.68 tok/s
   - DSpark-off was at parity or slightly ahead — the opposite of the
     documented ~24-33 tok/s DSpark champion range (which was measured
     preview-checkpoint-on-preview-checkpoint, i.e. head and weights matched).
4. Left DSpark off as the live config given the data, and this plan was
   written to fix the actual root cause rather than leave it off forever.

## The problem in one sentence

`src/exo/worker/engines/mlx/utils_mlx.py`'s `_overlay_dsv4_dspark()` always
loads the DSpark draft head from a **separate local directory**
(`EXO_DSV4_DSPARK_DIR`, default `~/.exo/models/local--DeepSeek-V4-Flash-DSpark-MTP`,
converted from the **preview** checkpoint's `mtp.*` shards) — it never reads
the currently-loaded model's own bundled `mtp.*` weights. This was fine when
the only model in play was the preview, but `-0731` ships its own (different,
re-trained) DSpark head that's currently going completely unused.

## What's structurally different between the two head formats

Confirmed via `mx.load()` on both the local converted head and `-0731`'s own
`model.safetensors.index.json` (fact 1145):

| | Local converted head (in use today) | `-0731`'s own bundled head |
|---|---|---|
| Location | Separate ~10GB file, `decoder.N.*` prefix | Inside the main checkpoint, `mtp.N.*` prefix |
| Attention | `decoder.N.attn.{wq_a,wq_b,wkv,wo_a,wo_b,kv_norm,q_norm,attn_sink}` | `mtp.N.attn.{...}` — **same names, 1:1 match** |
| Hyper-connections | Nested: `decoder.N.attn_hc.{base,fn,scale}`, `.ffn_hc.*`, `.hc_head.*` | Flat: `mtp.N.hc_attn_base/fn/scale`, `mtp.N.hc_ffn_base/fn/scale`, `mtp.2.hc_head_base/fn/scale` (only last stage) |
| Expert FFN | **Fused**: `decoder.N.ffn.switch_mlp.{gate_proj,down_proj,up_proj}.weight/scales` + `gate.e_score_correction_bias` | **Unfused**: `mtp.N.ffn.experts.{0..255}.{w1,w2,w3}.weight/scale` + `mtp.N.ffn.gate.{weight,bias}` |

The attention subtree is a free win (same names). Hyper-connections need
simple key restructuring (flat → nested). The expert FFN is the real new
work — 256 separate per-expert tensors need to be fused into one stacked
tensor per stage per weight (`w1`/`w2`/`w3`).

## Target module (what we're loading weights INTO)

`DeepseekV4DSparkModule` in `mlx-lm/mlx_lm/models/deepseek_v4.py` (class def
~line 5924). Its `.stages[i].ffn` is a `DeepseekV4MoE` (~line 2631) wrapping
a `SwitchGLU` (`mlx-lm/mlx_lm/models/switch_layers.py` ~line 161). Confirmed
via `SwitchLinear.__init__` (~line 94-105 in `switch_layers.py`):

```python
self.weight = mx.random.uniform(
    low=-scale, high=scale,
    shape=(num_experts, output_dims, input_dims),
)
```

So the fusion target shape is `(256, output_dims, input_dims)` — literally
`mx.stack()` over the 256 per-expert `[output_dims, input_dims]` tensors, in
expert-index order (0 through 255). Not exotic, just needs doing.

`MoEGate.weight` / `.e_score_correction_bias` (`deepseek_v4.py` ~line 2561)
map directly by shape to the checkpoint's `mtp.N.ffn.gate.weight` /
`.gate.bias` — no fusion needed, verify the bias key name matches exactly
(`e_score_correction_bias`) before wiring it up.

## Reference implementation to copy/adapt

`_overlay_dsv4_dspark()` in `src/exo/worker/engines/mlx/utils_mlx.py`
(~line 636) is the existing, working overlay for the OLD head format. Use it
as the structural template — same overall shape (load raw safetensors, remap
keys via a dict/regex, build `DeepseekV4DSparkModule`, per-layer quant-scheme
inference from `.scales` presence, `nn.quantize` + `load_weights`), just with
a different weight source and a real fusion step added.

## Concrete implementation steps

1. **Keep `mtp.*` weights around at load time.** Currently
   `DeepseekV4Model.sanitize()` in `deepseek_v4.py` strips `mtp.*` keys
   unless `EXO_DSV4_MTP=1` (the OLD single-head self-chaining mechanism, NOT
   DSpark). Need a new env gate — e.g. `EXO_DSV4_DSPARK_NATIVE=1` — that
   tells `sanitize()` to keep `mtp.*` around specifically so the new overlay
   function can consume it, without accidentally also enabling the old
   single-head mechanism.

2. **Write `_overlay_dsv4_dspark_native()`** (new function, or parameterize
   the existing one) that:
   - Reads `mtp.*` tensors from the already-loaded model's own weights
     (not a separate file — they're already in memory once step 1 keeps
     them) or re-reads them from the checkpoint's safetensors shards
     directly if that's cleaner given how `sanitize()` and load ordering
     work.
   - Remaps attention subtree: `mtp.N.attn.*` → `stages.N.attn.*` (should
     port almost verbatim from the existing `_remap()` logic given the
     confirmed 1:1 name match).
   - Remaps hyper-connections: flat `mtp.N.hc_attn_base/fn/scale` →
     nested `stages.N.attn_hc.{base,fn,scale}`; same pattern for `ffn_hc`
     and (stage 2 only) `hc_head`.
   - **Fuses expert FFN weights**: for each stage N and each of
     `w1`/`w2`/`w3`, gather `mtp.N.ffn.experts.{0..255}.{w1,w2,w3}.weight`
     (and `.scale` if quantized) in expert-index order, `mx.stack()` along
     axis 0 into the `(256, out, in)` shape `SwitchLinear` expects.
   - Maps gate directly: `mtp.N.ffn.gate.weight` → `stages.N.ffn.gate.weight`,
     `mtp.N.ffn.gate.bias` → `stages.N.ffn.gate.e_score_correction_bias`
     (confirm bias key name before committing).
   - Handles quantization: the checkpoint's expert weights are native
     fp8/mxfp4 per `config.json`'s `quantization_config`. Check whether the
     existing overlay's per-layer scheme-inference block (`.scales` presence
     → mxfp4/mxfp8 heuristic, ~line 700+ in `utils_mlx.py`) can be reused
     directly, or whether the fp8-native format needs a dequant-then-requant
     step before `nn.quantize`. This is the part most likely to need
     iteration/debugging — don't assume it's a clean drop-in without testing.

3. **Wire the new gate into `start_cluster.sh`** — add
   `EXO_DSV4_DSPARK_NATIVE` to the env allow-list (~lines 680-870, see the
   fork's own pitfall #51: new `EXO_DSV4_*` vars need a code patch PLUS a
   launcher line to actually propagate; `scp` is blocked, deploy via
   `git push origin main`, editable install picks it up on next
   `start_cluster.sh` run).

4. **Test.**
   - Confirm the overlay attaches without error and logs a sane tensor
     count/shape summary (mirror the existing
     `"DSpark draft head attached from ... (N tensors, 3 stages...)"` log
     line for an easy sanity check).
   - Re-run the exact controlled A/B from this session (list-of-100-primes,
     temp=0, max_tokens=300, `finish_reason=length`) comparing: native-head
     DSpark-on vs. old-head DSpark-on vs. DSpark-off. 3-6 runs each is
     enough (per-fork pitfall: c=1 has no bistability, no need for a long
     sweep).
   - Check draft acceptance rate via `exo.log`'s
     `"drafted tokens accepted"` line — should be meaningfully higher with
     the native head than the ~64% seen with the preview-vintage head if the
     hidden-state-mismatch theory was actually the throughput bottleneck.
   - Run the existing correctness smoke tests (plain chat "capital of X",
     tool-calling) to make sure nothing regressed.

## Effort/risk estimate

Genuinely non-trivial — new `sanitize()` gating, a new remap function, and
real fusion logic, not a quick patch. Estimate as a dedicated multi-hour
session, not something to tack onto other work. The quantization-handling
step (4c above) is the most likely place to hit real friction.

## Where to find things

- Live model card: `resources/inference_model_cards/deepseek-ai--DeepSeek-V4-Flash-0731.toml`
- Existing overlay to copy from: `src/exo/worker/engines/mlx/utils_mlx.py::_overlay_dsv4_dspark` (~line 636)
- Target module: `mlx-lm/mlx_lm/models/deepseek_v4.py::DeepseekV4DSparkModule` (~line 5924)
- MoE/SwitchGLU internals: `mlx-lm/mlx_lm/models/deepseek_v4.py::DeepseekV4MoE` (~line 2631), `mlx-lm/mlx_lm/models/switch_layers.py::SwitchLinear`/`SwitchGLU` (~line 94-161)
- Warm memory: `memory(action='recall', query='DSpark native 0731 remap')` pulls facts 1142-1156, the full session narrative including the raw A/B numbers and every structural finding this doc summarizes.
