# T10: HyperConnection real-code investigation — a promising lead FALSIFIED by direct production verification; real methodology lesson — 2026-08-22 (session 4)

## Why this check

Per T10's mandate (decompose prefill's 28.8% non-GEMM wall-time
remainder for a decode-fence-shaped hidden bug), read the real code for
the largest remainder spans by wall-time share: `layer.attn_hc`/`ffn_hc`
(2.3%+2.3%=4.6%) and `layer.attn_residual`/`ffn_residual`
(2.2%+2.2%=4.4%), the `HyperConnection` module's forward and expand
paths.

## Initial finding — looked like a real, decode-fence-class bug

Read `HyperConnection.__call__` (`mlx-lm/mlx_lm/models/hyper_connection.py`):
a gate selects between `_hc_ops` (pure-MLX, 20 real Sinkhorn iterations
as separate small ops — the real default `hc_sinkhorn_iters=20`,
confirmed via `deepseek_v4.py` ModelArgs) and `_hc_kernel` (a
hand-written fused Metal kernel doing identical math in one dispatch):

```python
use_ops = (
    self.training
    or mx.default_device() != mx.gpu
    or not mx.metal.is_available()
    or os.environ.get("EXO_HC_USE_OPS") == "1"
)
hc_func = _hc_ops if use_ops else _hc_kernel
```

A standalone pipelined microbench at the real production shape (B=1,
L=2048, hc_mult=4, hidden=4096, matching the exact prefill chunk size)
reproduced the real span-profile numbers closely (5216µs combined
hc+hc_expand vs the real profile's 5953µs — same order of magnitude,
confirming this cost is real, not a measurement artifact). Isolating
just the ops-vs-kernel dispatch found **a real 4.32x speedup**
(_hc_ops: 1517µs/call, _hc_kernel: 351µs/call), with **numerically
near-identical output** (max abs diff 4.9e-4, well within bf16
rounding — `post`/`comb` outputs bit-identical). This looked exactly
like the async-fence bug's shape: a real, unused fast path gated behind
a condition nobody verifies.

**Root cause candidate identified**: my own standalone microbench never
called `.eval()` on the constructed `HyperConnection` module, and MLX's
`nn.Module.training` defaults to `True` — so my microbench used
`_hc_ops` purely by construction, independent of what production
actually does.

## Verification — Fable consult flagged this correctly as unverified, real production check performed

Before treating this as a real bug (matching the project's standing
"verify from real signals, don't guess" rule), consulted Fable, which
correctly flagged that reading code alone does not settle the question
— any of the four gate conditions could differ in the live process from
what static reading suggests, and load-time state (`training=False`
after `model.eval()`) does not guarantee no later code path flips it
back.

**Direct production verification performed**: loaded the REAL
DeepSeek-V4-Flash-0731 checkpoint via `mlx_lm.utils.load_model`
(the exact function `exo`'s real TP loading path — `shard_and_load()`
in `utils_mlx.py` — calls), at the exact real model path
(`/Users/adam.durham/.exo/models/deepseek-ai--DeepSeek-V4-Flash-0731`,
confirmed via exo's own `build_model_path()` helper, not guessed). Real
result:

```
Top-level model.training: False
inner model.training: False
layer[0].training: False
layer[0].attn_hc.training: False
layer[0].ffn_hc.training: False
mx.default_device(): Device(gpu, 0)
mx.metal.is_available(): True
EXO_HC_USE_OPS env: None
```

**All four gate conditions clear the fast-kernel path**: `training` is
genuinely `False` (confirmed `mlx_lm.utils.load_model` does call
`model.eval()` before `load_weights()`, and this DOES recursively
propagate to `HyperConnection` submodules constructed during initial
model construction — verified experimentally that MLX's `.eval()`
correctly sets `training=False` on already-constructed nested
submodules); device is genuinely `gpu`; Metal is genuinely available;
the escape-hatch env var is genuinely unset on the live runner process
(confirmed via `ps eww <pid>` on the actual production PID, not
inferred). **`_hc_kernel` (the fast, fused path) does fire in
production. This is NOT a bug.**

## Conclusion — a real, honest negative result

The 4.32x speedup and the elevated `layer.attn_hc`/`ffn_hc` span costs
in the 220K prefill profile are real, but **the cause is NOT an unused
fast path** — production already uses the fast kernel. The real
explanation for these spans' cost (2.3% each, ~4.6% combined) is
therefore something else: either the fused kernel's own real cost at
this shape (351µs/call, confirmed via the isolated benchmark — smaller
than the full `HyperConnection.__call__`'s ~2854µs, meaning the
rms_norm+matmul precursor and `finalize()`'s `mx.eval()` sync overhead
dominate the span's measured cost, not the Sinkhorn/collapse step
itself), or genuine, currently-irreducible real compute + dispatch
cost for this architecture's real per-layer HyperConnection mechanism.

**This closes the specific HyperConnection lead as NOT a hidden bug.**
The methodology error (microbench not matching production's real
`.eval()` state) is a genuinely reusable lesson: **always verify a
training/eval-mode-gated code path against the REAL model loading
function in a live-equivalent environment, not a standalone
reconstruction, before drawing a performance conclusion from it** —
this is a distinct and easy-to-miss trap from the decode-fence bug's
actual cause (an owner-registration gate that was ACTUALLY broken;
this one looked structurally similar but wasn't).

## What remains open for T10

The HyperConnection candidate is closed. The broader T10 mandate
(decompose the ~19.3% non-all_sum portion of prefill's 28.8% remainder)
is **not fully closed** — this session investigated the single largest
candidate spans (attn_hc/ffn_hc, 4.6% combined) and found them
genuinely optimized already. Remaining un-investigated spans from the
same remainder: `layer.attn_residual`/`ffn_residual` (hc_expand path,
4.4% combined — the `_hc_expand_op` benchmark showed 2361µs/call,
not yet decomposed into its own rms_norm-vs-matmul-vs-eval breakdown),
`attn.indexer` (4.0%), `moe.gate`/`post_combine` (5.1% combined). None
of these have been read/benchmarked this session. A future continuation
of T10 should apply the SAME rigor (read real code, build a
production-config-verified microbench, cross-check against a live
process before concluding) to these remaining spans.

## Standing lesson for the repo

Added explicitly here since it's a genuinely new, reusable pitfall not
previously documented in this codebase's own meta-lessons section:
**a training/eval-mode-gated fast-path check requires verifying against
`mlx_lm.utils.load_model`'s real `model.eval()` propagation, not a
standalone module reconstruction** — MLX's `nn.Module.training`
defaults to `True`, and any microbench that constructs a module
directly (rather than through the real load path) will silently take
the slow/training-mode branch of ANY `self.training`-gated code,
producing a real-looking but production-irrelevant performance number.
