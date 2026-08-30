# P04 — Sinkhorn truncation numerics validation (2026-08-30)

**Scope: 100% offline/local laptop work — ZERO cluster contact.** No SSH, no
production API, no relaunches, no benchmark against the live cluster. This is
the code+numerics prerequisite the P03 doc flagged as the gate for its HC
Sinkhorn-truncation fusion target (`docs/p03-smallop-bucket-gputrace-2026-08-30.md`,
§"three concrete targets": hc_collapse = 30.4% of the spec-ON decode cycle at
5.24 ms/cycle; "truncating to ~4-5 iterations (numerics-gated) projects the
family from 5.24 -> ~1.5-2 ms/cycle").

Two deliverables:

1. **`EXO_HC_SINKHORN_ITERS` env-override knob** in
   `mlx-lm/mlx_lm/models/hyper_connection.py` (`HyperConnection.__init__`),
   following the existing `EXO_HC_USE_OPS` diagnostic-knob pattern. Minimal,
   purely additive: overrides only the iteration COUNT at construction; both
   execution paths and the kernel source are untouched. Unset/invalid →
   falls back to `config.hc_sinkhorn_iters` (bit-identical to today).
2. **Offline numerics validation** of truncation safety at
   {20, 10, 5, 4, 3, 2} iterations, through the REAL production code paths
   (both the pure-MLX ops path and the fused Metal kernel path), with the
   real DeepSeek-V4-Flash-0731 checkpoint config values.

Artifacts: `tmp/p04-sinkhorn-truncation-20260830/` (numerics_check.py,
numerics_results.json, stdout log reproduced below).

## 1. The code change (mlxl-lm submodule, uncommitted per instructions)

```diff
--- a/mlx_lm/models/hyper_connection.py
+++ b/mlx_lm/models/hyper_connection.py
@@ HyperConnection.__init__ (line ~412)
         self.hc_mult = config.hc_mult
         self.sinkhorn_iters = config.hc_sinkhorn_iters
+
+        # Tuning knob (2026-08-30): override the Sinkhorn iteration count
+        # from the environment, without touching the checkpoint config.
+        # P03 GPU trace (docs/p03-smallop-bucket-gputrace-2026-08-30.md)
+        # found the hc_sinkhorn_iters=20 loop is 30.4% of the production
+        # spec-ON decode cycle — pure sequential-barrier latency, since the
+        # comb matrix is only 4x4. Both execution paths already thread the
+        # count through (the fused kernel takes it as the ITERS template
+        # param), so only the count itself needs overriding. Off / unset /
+        # invalid values fall back to config.hc_sinkhorn_iters — bit-identical
+        # to today's behavior. Same pattern as EXO_HC_USE_OPS (2026-06-09).
+        env_iters = os.environ.get("EXO_HC_SINKHORN_ITERS")
+        if env_iters is not None:
+            try:
+                parsed = int(env_iters)
+            except ValueError:
+                parsed = 0  # invalid -> fall back to config below
+            if parsed > 0:
+                self.sinkhorn_iters = parsed
+
         self.hc_eps = config.hc_eps
```

Knob behavior verified live (fresh module per env, real class):
`unset → 20`, `4 → 4`, `bogus → 20`, `0 → 20`, `-3 → 20`.

Note the knob is read at `__init__`, i.e. at model-load time — a runner
restart is required to change it (same as any env knob read at construction;
documented here so nobody expects a live toggle).

## 2. Method

- **Real config**: all values read from the cached
  `DeepSeek-V4-Flash-0731/config.json` and asserted against expectations:
  `hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6, hidden_size=4096,
  rms_norm_eps=1e-6, num_hidden_layers=43`.
- **Real code paths**: every measurement goes through the REAL
  `HyperConnection.__call__` — the ops path (`EXO_HC_USE_OPS=1`, `@mx.compile`d
  `_hc_split_sinkhorn_ops`) AND the fused Metal kernel path (production path:
  module `.eval()`'d, env unset, L=64 divisible by the precursor's R_TILE=4,
  so `_hc_precursor_fused` + `_hc_sinkhorn_collapse_kernel` with ITERS as a
  template param run for real on the laptop's GPU). The fused path is the
  production path on the studios; the ops path is the numerics reference.
  **Cross-path equivalence at iters=20: bit-identical (max_abs=0 on all three
  outputs)** — first local confirmation that the fused kernel and the ops path
  compute the same function, and that the new knob threads the ITERS template
  param correctly (the fused path's divergence table is nonzero and IDENTICAL
  to the ops path's, which is only possible if the template param took effect).
- **Inputs**: synthetic, decode-shaped `(1, 64, 4, 4096)` (B, L, hc_mult,
  hidden), x ~ N(0,1). The real trained weights (`fn`, `base`, `scale`) are
  NOT on this laptop (the 6.1 MB HF cache snapshot is config+tokenizer only;
  weights live on the studios, out of scope per standing rules) — this is the
  stated deviation from full production realism, mitigated as follows.
- **Parameter realism (the hard part — two traps found and fixed)**:
  1. `fn` drawn at the naive init scale `1/sqrt(fan_in)` produces softmax
     logits with std ≈ 0.08 — a nearly-flat softmax that converges in 1-2
     iterations and makes iters=2 measure *bit-identical* to iters=20. That is
     a false "truncation is free" answer; trained logits are NOT at init
     scale. Fixed by empirically calibrating `fn` so `mixes` (the softmax
     logits) hit a target std: **1.0 = "realistic O(1) logits"** (the
     standard operating regime of a trained 4-way softmax with O(1) bias) and
     **4.0 = "wide-logit stress x4"** (adversarial bound).
  2. Scaling the *input* x is a no-op for Sinkhorn — `rms_norm(x)` is
     scale-invariant, so the case axis had to be the fn/logit scale, not x.
  3 param draws × 4 input seeds per cell; worst case reported.
- **Harness-integrity trap found (worth recording)**: `uv run python
  script.py` from the mlx-lm dir resolves `import mlx_lm` to the STALE
  non-editable copy in `exo/.venv/site-packages` (knob absent, constructor
  silently ignores the env var) unless `sys.path[0]` is forced to the
  submodule. The first full run measured "zero divergence everywhere" purely
  because of this. The script now pins `sys.path` to the submodule and
  asserts the imported class contains the knob. Anyone running standalone
  numerics against this fork's mlx-lm submodule should do the same.

## 3. Divergence vs iters=20 baseline

Both paths give identical numbers (fused kernel = ops path); table shows the
shared values. Worst case over 3 param draws × 4 inputs. pre/post/collapsed
are structurally Sinkhorn-independent (pre/post are sigmoids computed before
Sinkhorn; collapsed = pre·y and never touches comb) — their measured zero
divergence across every cell confirms this and validates the harness.

comb is the only Sinkhorn-dependent output — and it matters: comb is the
residual-stream mixing matrix consumed by `hc_expand` on the next sublayer,
applied 2× per layer (attn_hc + ffn_hc) × 43 layers = 86 times per forward.

| case | iters | comb max_abs | comb mean_abs | comb mean_rel |
|---|---|---|---|---|
| realistic O(1) | 20 | 0 (baseline) | 0 | 0 |
| realistic O(1) | 10 | **1.073e-02** | 1.796e-04 | 2.032e-03 |
| realistic O(1) | 5 | **5.545e-02** | 2.184e-03 | 1.979e-02 |
| realistic O(1) | 4 | **8.610e-02** | 4.126e-03 | 3.518e-02 |
| realistic O(1) | 3 | **1.373e-01** | 8.397e-03 | 6.709e-02 |
| realistic O(1) | 2 | **2.308e-01** | 1.880e-02 | 1.410e-01 |
| wide x4 | 10 | 8.704e-02 | 6.112e-03 | 1.518e-01 |
| wide x4 | 5 | 5.295e-01 | 2.868e-02 | 5.692e-01 |
| wide x4 | 4 | 5.948e-01 | 3.995e-02 | 8.666e-01 |
| wide x4 | 3 | 6.787e-01 | 5.465e-02 | 1.634e+00 |
| wide x4 | 2 | 7.858e-01 | 7.492e-02 | 4.229e+00 |

pre / post / collapsed: exactly 0.0 (max, mean, rel) in every cell, both
paths, both cases.

## 4. Measured Sinkhorn convergence curve (the "converges fast" claim, tested)

Realistic O(1) logits, one representative forward's comb tensor, residual
row-sum deviation after each full row+col iteration (columns are re-normalized
each iteration so col deviation sits at the eps floor by construction):

```
iter   max|row-1|   max|col-1|   step_delta
   1    3.123e-01    1.431e-06    2.168e-01
   2    1.578e-01    1.252e-06    9.587e-02
   3    8.936e-02    1.132e-06    4.227e-02
   4    7.206e-02    1.132e-06    1.943e-02
   5    5.678e-02    1.132e-06    1.093e-02
   6    4.407e-02    1.073e-06    7.308e-03
   7    3.384e-02    1.073e-06    5.048e-03
   8    2.578e-02    1.073e-06    3.726e-03
   9    1.954e-02    1.073e-06    2.821e-03
  10    1.475e-02    1.132e-06    2.124e-03
  11    1.110e-02    1.132e-06    1.594e-03
  12    8.334e-03    1.132e-06    1.194e-03
  13    6.247e-03    1.132e-06    8.927e-04
  14    4.678e-03    1.073e-06    6.669e-04
  15    3.499e-03    1.073e-06    4.981e-04
  16    2.616e-03    1.073e-06    3.719e-04
  17    1.955e-03    1.073e-06    2.775e-04
  18    1.460e-03    1.073e-06    2.072e-04
  19    1.091e-03    1.132e-06    1.545e-04
```

Wide-logit x4 stress case:

```
iter   max|row-1|   max|col-1|   step_delta
   1    9.879e-01    2.325e-06    3.159e-01
   5    6.925e-01    1.669e-06    1.128e-01
  10    9.384e-02    1.192e-06    2.483e-02
  15    5.293e-02    1.132e-06    4.465e-03
  19    4.164e-02    1.132e-06    2.421e-03
```

**Verdict on the claim: "Sinkhorn converges fast" is NOT supported at these
logit scales.** Convergence is geometric with ratio ≈ 0.67/iter (realistic
case) — slow enough that even the FULL 20 iterations end at ~1.1e-3 residual,
and truncating to 4-5 iterations stops at 5.7-7.2e-2 residual. The measured
comb divergence at each truncation point matches the curve's residual at that
iteration (e.g. iters=5 → 5.5e-2 measured ≈ 5.7e-2 curve residual), which is
the internal-consistency check. The wide-logit case doesn't even converge
within 20 iterations (plateau ~4e-2 by iter 19) — 20 iterations is not
"way past convergence" for it, it IS the operating point. The
Hyper-Connections paper's convergence claim may hold for their training-time
logit distribution, but the checkpoint's chosen `hc_sinkhorn_iters=20` is
consistent with a model trained to expect 20-iteration comb values, whatever
the paper says.

## 5. Interpretation and honest recommendation

**The P03 projection — truncate 20 → 4-5 iterations — is numerically NOT
safe as a default.** The evidence:

- At 4-5 iterations the comb matrix per-application error is up to
  5.5-8.6e-2 max abs (realistic case; 0.53-0.59 wide). This is not
  fp-rounding noise (1e-6); it is a 2-3 order-of-magnitude-larger,
  systematic perturbation of the residual-stream mixing matrix.
- This error is applied 86 times per forward (attn_hc + ffn_hc × 43 layers).
  The residual stream accumulates; a systematic per-layer perturbation of
  the mixing matrix compounds across depth rather than averaging out. We did
  NOT measure 43-layer accumulation (would need the real weights) — so the
  compounding magnitude is unmeasured, but the direction (compounding, not
  cancelling) is structural: every hc_expand consumes the same direction of
  truncation bias within a layer pair.
- The one clean finding in truncation's favor: pre/post/collapsed (the
  current sublayer's outputs) are BIT-UNAFFECTED by truncation — the error
  enters only through the next sublayer's residual mixing. So the damage is
  delayed-and-distributed, not immediate — which is exactly the kind of
  perturbation that shows up as subtle generation-quality drift rather than
  crashes.

**Minimum safe truncation, based on this data:**

- **iters=10** is the only defensible candidate: 1.07e-2 max / 2.0e-3 mean-rel
  (realistic), 8.7e-2 max (wide stress). It halves the Sinkhorn loop, not the
  4-5× the P03 projection wanted — the P03 arithmetic (5.24 → ~1.5-2
  ms/cycle) does NOT survive contact with the numerics at 4-5 iters.
- **iters ≤5**: unsafe to even live-test without a quality gate; the per-application
  error (5e-2 max) is the same order as the residual mixing weights themselves.
- **Honest bottom line: no truncation below N=10 is supported by this data,
  and even N=10 is "plausibly tolerable, unproven" — not "safe".** Any live
  throughput test MUST be paired with a generation-quality gate (the
  exo-local-vs-cloud-dsv4 probe suite is the established gate), and the
  knob makes that test reversible per-request (set env → restart runner).

**Open question that would upgrade this analysis from synthetic to real:**
the actual trained comb-logit distribution. The realistic case assumes O(1)
softmax logits; the real checkpoint's could be tamer (supporting deeper
truncation) or wilder (refuting even 10). A read-only diagnostic on a studio
(dump one decode cycle's mixes/comb logits — no relaunch, standalone process,
the p01/p03 recipe) would pin the exact logit std and let this same offline
script re-run with the real distribution. That is the cheapest next step and
should precede any live truncation test. It was NOT done here (laptop-only
scope, per instructions).

## 6. Status

- Code change: applied to the mlx-lm submodule working tree, UNCOMMITTED
  (per instructions — supervisor commits submodule + parent pointer).
- Docs: this file + PERFORMANCE_HISTORY.md entry. Uncommitted likewise.
- Cluster: untouched, zero contact, verified by construction (no SSH, no
  API calls in any command run this session).
- NOT verified here: 43-layer compounding with real weights; live throughput
  delta; generation-quality impact at any iteration count. Those require the
  explicitly-gated live follow-up.