# PRE-REGISTRATION — verify-phase decomposition & context-depth scan

**Written:** 2026-09-01, BEFORE any depth-scan data was collected.
**Timestamp anchor:** cluster runners PID 83029 (m4-1) / 85554 (m4-2), lstart
`Tue Sep 1 16:19:35/37 2026` — unchanged from the V3 run, profiling flags
(`EXO_DSV4_MTP_PROFILE=50`, `EXO_DSV4_RB_PROFILE=1`) verified live on the
**genuine** runner PIDs (not the SCREEN/login/zsh wrappers that also match
`-m exo` — see v3/RESULTS.md §4 for that false-positive trap).

At the moment of writing, the ONLY depth measured is the V3 baseline:
89,408 prompt tokens → 33.99 t/s decode, 425.7 t/s prefill, 68.85 ms cycle
(verify 56.09 / draft 9.19 / accept 2.87 / rollback 0.71).
Nothing is known about 150K or 250K behavior.

---

## H0 (null) — no context-dependence

Decode t/s at 150K and 250K stays within the V3 run-to-run noise band
(33.99 ± 2σ = 30.4 – 37.6 t/s), and verify stays within 56.1 ± 10 %.

**Consequence if H0 holds:** the real-usage 20 t/s CANNOT be attributed to
context depth at all, and the entire framing of this task is wrong — the gap
would have to live in something the synthetic benchmark does not reproduce
(concurrency, `max_tokens` distribution, real-session prompt composition,
KV-cache reuse/eviction, JIT aux-model contention). That is a real and
reportable outcome, not a failure.

## H1 — attention/KV-gather bound (structural)

Verify grows monotonically and roughly linearly with context; decode fits
`t/s ≈ (1+acc) / (a + b·ctx)` with `b > 0` significant.

**Pre-registered quantitative bar for "linear":** verify(250K)/verify(89K)
≥ 2.0 would indicate a slope at/above proportional-to-context. Between 1.3
and 2.0 = sublinear growth (a mix of fixed + scaling cost). < 1.3 = NOT
attention-bound in the structural sense.

**Lever if H1:** per-token attention cost over pooled KV — indexer top-k
window, gather efficiency. NOT snapshot/rollback (already closed at 0.218 %).

## H2 — acceptance collapse (spec-efficiency loss)

Cycle time stays roughly flat but acceptance/cycle falls at depth, so
`(1+acc)/cycle` drops.

**Pre-registered bar:** acceptance drop ≥ 15 % relative from 89K→250K with
cycle-time growth < 20 % ⇒ H2 is the dominant term.

**Lever if H2:** draft quality at depth (MTP head behavior over long context),
a completely different fix from H1.

## H3 — fixed overhead / non-decode time

The wall-clock t/s the *user* perceives includes TTFT (prefill of a 150K+
prompt is tens of seconds) and per-request overhead. If measured pure-decode
t/s at 150K is much higher than the ~20 t/s felt in real usage, the gap is
partly an accounting artifact — the user is experiencing
`total_tokens / total_wall`, not steady-state decode.

---

## Gate: what counts as "a lever exists"

A lever is only claimed if BOTH:
1. a specific phase/sub-phase is measured (not inferred) to grow with context
   and account for ≥ 25 % of the cycle-time increase 89K→250K, AND
2. an existing counter or a source-level read identifies *which* operation
   inside that phase carries the growth.

Anything weaker is reported as "no actionable lever found; here is the
structural reason," per the task's root-cause-framing constraint.

## Decomposition arithmetic (pre-committed)

The 20→34 t/s gap will be split as:
- **context-scaling term**: (t/s at 89K) − (t/s at the real-usage depth),
  measured directly by the scan.
- **acceptance term**: recomputed holding cycle time at its 89K value and
  substituting the measured deep-context acceptance.
- **residual/fixed term**: whatever remains after the two above, explicitly
  labeled as unexplained rather than silently absorbed.

If the residual exceeds either named term, the honest conclusion is that the
synthetic benchmark does not reproduce real usage, and I will say so.

## Method constraints (pre-committed)

- `[MTP-PROF]` dumps are **cumulative running means**. Per-interval values MUST
  be reconstructed as `(mean_k·n_k − mean_{k−1}·n_{k−1}) / (n_k − n_{k−1})`,
  cycle-count weighted, restricted to each bench window. Using
  `statistics.mean` over dump lines double-counts early cycles (v3/RESULTS.md §1).
- `rb_pool_restores` is a COUNT mislabeled with an `ms` suffix — never read it
  as a time (v3/RESULTS.md §2).
- Cluster idle (`gpu_usage_ratio < 0.1`) checked before every depth.
- 5 scored iterations + 1 warmup per depth, concurrency 1, sequential.
- No relaunch, no config change, no code change.
