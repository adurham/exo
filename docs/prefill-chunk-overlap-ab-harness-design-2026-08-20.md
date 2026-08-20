EXO_PREFILL_CHUNK_OVERLAP controlled A/B: design doc (2026-08-20)
=================================================================

Status: DESIGN ONLY. Nothing here has been run. No live cluster access was
used to produce this document. A future session with explicit user approval
runs `scripts/prefill_overlap_ab.py` ONCE (it does both arms in a single
batched session) and then fills in the Results section.

Motivation
----------

`docs/prefill-chunk-overlap-live-test-2026-08-20.md` recorded an ambiguous
live result with `EXO_PREFILL_CHUNK_OVERLAP=1` (commit `b7aa41920`):

- Trial 1: expected `OVLP-9471-CHK-3433`, got `SECLP-9473-CHK-3433`.
- Trial 2: correct but truncated at `max_tokens=50` (`finish_reason: length`)
  -- a weak, uninterpretable result.
- No `flag=0` control on identical prompts, so the baseline miss rate for
  this prompt style at ~38K context is UNKNOWN.

The corruption signature is the interesting part: the `CHK-3433` suffix was
recovered exactly while the `OVLP-9471` prefix was wrong. That is NOT total
recall failure. It is consistent with *one region* of the KV cache being
stale or partially materialized -- i.e. a specific chunk boundary -- which is
exactly the failure mode the depth-1 `mx.async_eval` double buffer could
produce if `mx.eval(_prev_cache_sync)` does not actually fence the previous
chunk's cache write against the next chunk's forward read.

This design turns that guess into a measurement.

Mechanism under test (read from source, not assumed)
----------------------------------------------------

`src/exo/worker/engines/mlx/generator/generate.py`, `prefill_batched()`
chunk loop (~line 1416):

```python
_overlap = os.environ.get("EXO_PREFILL_CHUNK_OVERLAP", "0") == "1"
...
if _overlap:
    if _prev_cache_sync is not None:
        mx.eval(_prev_cache_sync)          # depth-1 double buffer
    _cache_state = [c.state for c in batched_cache]
    mx.async_eval(_cache_state)
    _prev_cache_sync = _cache_state
else:
    mx.eval([c.state for c in batched_cache])
```

Two chunk-size facts that MUST be respected by the prompt generator, both
verified in source:

1. `prefill_batched()` uses `prefill_step_size` **undivided**
   (line ~1424: `n_to_process = min(prefill_step_size, max_length - offset)`).
   So with `EXO_PREFILL_STEP_SIZE=2048` the boundaries in the code path that
   contains the overlap flag land at token offsets 2048, 4096, 6144, ...
2. The *serial* `prefill()` path divides:
   `prefill_step_size = prefill_step_size // min(4, group.size())`
   (line ~496). On the 2-rank cluster that is 1024. The overlap flag is NOT
   in that path, but the harness records which path ran (via the
   `prefill_batched`/`prefill.chunkN` trace names) so a boundary analysis is
   never done against the wrong grid.

The harness therefore places needles against a configurable
`CHUNK` (default 2048) and additionally emits a 1024-grid re-analysis so a
mis-assumed grid cannot silently invalidate the localization.

Design decisions and why
------------------------

**(1) Multiple needles per prompt, at chosen token offsets.**
One needle per prompt only tells you "this trial was wrong". Eight needles
per prompt, each anchored to a different chunk boundary, tells you *which
boundary* is wrong -- and, critically, whether the failures cluster on
boundaries at all. Each prompt carries needles at:

- `k*CHUNK - delta` (needle text ends just BEFORE a boundary),
- `k*CHUNK + delta` (needle text starts just AFTER a boundary),
- `k*CHUNK` straddling (needle text spans the boundary itself -- the
  highest-value placement, since a needle split across two chunks is the
  only one whose tokens live in both the producer and the consumer chunk),
- plus 2 CONTROL needles placed at chunk-interior offsets (far from any
  boundary) in every prompt.

The control needles are the load-bearing part of the design. If boundary
needles fail at a materially higher rate than interior needles *within the
same prompt and the same arm*, that is a within-prompt, within-arm contrast
that no amount of "the model is just bad at long-context recall" can
explain away.

**(2) Distinct, high-entropy, format-fixed secret strings.**
Format: `NDL-<slot>-<AAAA>-<9999>-<BBBB>` (word-digit-word). Two reasons:
- the observed corruption mutated *part* of the string, so scoring must be
  partial-credit-aware: the harness records exact match, per-field match,
  and normalized edit distance, not just a boolean.
- fixed format means a wrong answer that still matches the format is
  distinguishable from a formatting/refusal failure.

Every secret is generated from a per-prompt seed and is unique across the
whole run, so a cross-contaminated answer (needle 3's value returned for
needle 5) is detectable and is itself strong evidence of cache-region
mixing.

**(3) `max_tokens` generously high.**
`MAX_TOKENS = 4096` by default. DSv4-Flash is a thinking model: reasoning
tokens are billed against the same budget, and the live test's trial 2 was
ruined by `max_tokens=50`. The harness additionally treats any
`finish_reason == "length"` result as **INVALID, not as a failure**, and
records it separately so a truncated trial can never be scored as a
correctness miss. If the invalid rate exceeds 10% the run is aborted and
`MAX_TOKENS` is raised.

**(4) Answer format that keeps the graded region short.**
The model is asked to emit one `SLOT<i>=<value>` line per needle, in order,
after its reasoning. Parsing is by regex per slot, so a missing slot is
scored `MISSING` (distinct from `WRONG`), and reasoning-block chatter cannot
be mistaken for an answer.

**(5) Paired design, identical prompts, single session.**
Each prompt id `p` is run under BOTH arms. Prompts are byte-identical across
arms (generated once, cached to disk, replayed). This makes the comparison
a paired one and lets McNemar's test be used on the discordant pairs.
Arm order is INTERLEAVED per prompt (flag=0 then flag=1 for the same prompt,
before moving to prompt p+1) so cluster drift, thermals, and time-of-day
effects hit both arms equally.

**(6) Cache must be cold for every trial.**
`cached_tokens: 0` is asserted on every response (`usage.prompt_tokens` and
the exo prefix-cache field). Every prompt has a unique random preamble, so a
prefix cache cannot short-circuit prefill and thereby skip the chunk loop
entirely -- which would make the whole test vacuous. Any trial reporting
non-zero cached tokens is marked INVALID and re-run once with a fresh nonce.

**(7) Logit-level comparison.**
Token-level agreement can hide real numerical divergence. exo's OpenAI
adapter DOES expose logprobs end-to-end -- verified in source:

- `src/exo/api/types/api.py`: request fields `logprobs: bool | None`,
  `top_logprobs: int | None`; response `Logprobs`/`LogprobsContentItem`
  with `token`, `logprob`, `top_logprobs`.
- `src/exo/api/adapters/chat_completions.py:172` forwards them into the
  generation task; :196-224 (streaming) and :318-350 (non-streaming) build
  the response objects.
- `src/exo/shared/types/text_generation.py:157` carries `logprobs` /
  `top_logprobs` into the runner task.
- `src/exo/worker/engines/mlx/generator/generate.py:1864 extract_top_logprobs`
  and :2377-2408 populate them from the real full-vocab logprob array.

So NO instrumentation of `generate.py` is required. The harness sends
`logprobs: true, top_logprobs: 20` and captures, for every generated token,
the sampled token's logprob plus the top-20 distribution.

Because sampling is stochastic, the harness runs at `temperature: 0` (greedy)
for the graded arms so token sequences are directly comparable, and compares:

- **`logprob_correct`** (recorded, analyzed manually if a gate fires):
  the chat API cannot force-decode an arbitrary continuation, so the
  harness instead stores the FULL top-20 distribution at every generated
  position. Because the answer format puts the secret immediately after a
  literal `SLOT<i>=` anchor, the positions where the model emits each
  secret's tokens are locatable in the recorded stream, and the logprob the
  model assigned to the correct token at each of those positions can be
  read off the top-20 list. If the correct token falls outside the top-20
  at some position, that is a censored observation and must be treated as
  a bound (`<= min(top20)`), never imputed. This is a post-hoc drill-down,
  not one of the three automated gates.
- **`top1_agreement`**: fraction of generated positions where flag=0 and
  flag=1 chose the same top-1 token on the identical prompt.
- **`mean_abs_logprob_delta`** and **max delta** over the aligned prefix of
  positions, flag=0 vs flag=1. Under a *pure scheduling* change (the flag's
  stated intent) this should be ~0 modulo GPU nondeterminism; a genuinely
  stale-cache read produces large deltas that are *localized* to the tokens
  recalling the affected needle.

The logit comparison is the sharpest instrument here because it does not
need a wrong answer to fire. It can show divergence even on trials where
both arms answer correctly -- which is exactly what a 1-in-N race needs.

**(8) Nondeterminism floor must be measured, not assumed.**
The harness includes a third arm, `flag0_repeat`: the SAME prompt run twice
under flag=0. This establishes the run-to-run logprob noise floor of the
cluster (MoE routing, collective reduction order, and GPU scheduling are
not bit-deterministic). Any flag=1 divergence is judged against that floor,
not against zero. **Without this arm the logit comparison is
uninterpretable**, and it is cheap (it reuses prompts already generated).

Context lengths
---------------

Needles must be spread across enough chunks to localize, and the run must
fit in one session. Default plan, `N = 12` prompts per arm:

| prompts | approx ctx | chunk boundaries covered (@2048) |
|---------|-----------|----------------------------------|
| 4       | ~20K      | 2..9                              |
| 4       | ~40K      | 2..19 (sampled)                   |
| 4       | ~80K      | 2..39 (sampled)                   |

The ~40K tier reproduces the original anomaly's regime (38K). The ~20K tier
gives a fast, high-trial-count regime. The ~80K tier probes whether the
effect grows with chunk count (a race that fires per-boundary with
probability `p` gives failure probability `1-(1-p)^n_chunks`, so the
per-prompt failure rate SHOULD rise with context length if this is a real
per-boundary race -- a strong, falsifiable prediction that distinguishes it
from generic long-context degradation, which also rises with length but
does NOT concentrate on boundary needles).

Total trials: 12 prompts x (flag0 + flag1 + flag0_repeat) = 36 generations
in one session (two cluster relaunches).

Statistical decision rule
-------------------------

Three independent gates. The verdict is decided by the FIRST gate that
fires, in order. All thresholds are fixed HERE, before any data is seen.

**Gate A -- paired token-level (McNemar, exact binomial).**
Unit of analysis is the *needle*, not the prompt: 12 prompts x 8 needles =
96 paired observations per arm. Let `b` = needles correct under flag=0 but
wrong under flag=1, and `c` = wrong under flag=0 but correct under flag=1.
Discard concordant pairs. Under H0 (flag has no effect), `b ~ Binomial(b+c, 0.5)`.

- **REAL BUG** if the one-sided exact binomial p-value <= 0.05 AND `b - c >= 4`.
  (The absolute-difference floor prevents declaring a bug on `b=3, c=0`,
  which is p=0.125 anyway, and guards against a single pathological prompt.)
- **NO EFFECT DETECTED at this power** if `b + c <= 2` and Gates B and C
  also fail to fire.
- **INCONCLUSIVE** otherwise -> extend to N=24 prompts in a follow-up
  session rather than guessing.

**Gate B -- boundary localization (the diagnostic gate).**
Within the flag=1 arm only, compare boundary needles (6 per prompt = 72)
against interior control needles (2 per prompt = 24) using Fisher's exact
test, one-sided.

- **REAL BUG, LOCALIZED** if p <= 0.05 and the boundary-needle error rate is
  at least 3x the interior rate. Additionally report the per-boundary error
  histogram: if errors concentrate on a small set of `k` values (e.g. every
  boundary, or only the first, or only late ones), that is the actionable
  localization the whole exercise exists to produce.
- Gate B firing while Gate A does not is still a positive finding, because
  Gate B is a within-arm contrast and is therefore immune to any baseline
  difference between arms.

**Gate C -- logit divergence vs the measured noise floor.**
Let `D_noise` = the distribution of `mean_abs_logprob_delta` between the two
flag=0 runs of the same prompt (the `flag0_repeat` arm, 12 values). Let
`D_flag` = the same statistic between flag=0 and flag=1 (12 values).

- **REAL NUMERICAL DIVERGENCE** if `median(D_flag) > max(D_noise)`, or if a
  Mann-Whitney U test on the two 12-value samples gives p <= 0.05 AND
  `median(D_flag) >= 2 * median(D_noise)`.
- Also flag if `top1_agreement` between arms is below the *minimum*
  `top1_agreement` observed within the flag=0 repeat pairs.
- A Gate C fire with Gates A and B silent means: the flag DOES perturb
  numerics measurably, but the perturbation has not yet been shown to
  change answers. That is a "do not ship, keep investigating" verdict, not
  a clean pass.

**Overall verdict table**

| A | B | C | Verdict |
|---|---|---|---------|
| fire | fire | any | REAL BUG, localized to specific boundaries. Do not enable. Fix the fence. |
| fire | - | any | REAL correctness regression, not boundary-localized. Do not enable; widen investigation beyond the double buffer. |
| - | fire | any | REAL boundary-localized effect at sub-threshold global rate. Do not enable. |
| - | - | fire | Numerics perturbed, correctness impact unproven. Do not enable; extend N. |
| - | - | - | No effect detected at N=12/96 needles. NOT "proven safe" -- report the achieved power (see below) and let the user decide. |

**Power honesty.** With 96 paired needles, Gate A has roughly 80% power to
detect an induced per-needle error rate around 6-7% against a low baseline.
It has POOR power against a 1% rate. The harness prints the minimum
detectable effect for the observed baseline so the final report can never
claim "safe" when it only established "not catastrophic". This must be
stated in any conclusion.

Single-session execution contract
---------------------------------

Live cluster access requires per-session user approval, so the harness does
everything in ONE approved session:

1. Generate + cache all prompts locally (no cluster needed) -- can be done
   ahead of time with `--generate-only`.
2. Relaunch the cluster with `EXO_PREFILL_STEP_SIZE=2048` and the flag
   **unset** (arm A). Run all flag=0 trials + all flag0_repeat trials.
3. Relaunch ONCE with `EXO_PREFILL_CHUNK_OVERLAP=1` on BOTH ranks
   (verify per-rank env before any generation -- per-rank flag skew is the
   documented top risk of this mechanism). Run all flag=1 trials.
4. Revert to standing baseline.
5. Analyze from the on-disk JSONL. Analysis needs no cluster and can be
   re-run offline (`--analyze-only`).

Exactly two relaunches. The harness writes every response to JSONL
incrementally so a mid-run failure loses nothing and the analysis can be
completed offline.

Pre-flight (must all pass before generating)
--------------------------------------------

- Both ranks report the same `EXO_PREFILL_CHUNK_OVERLAP` and
  `EXO_PREFILL_STEP_SIZE`.
- `EXO_DSV4_MTP=0` and `EXO_SPECULATIVE=0` for BOTH arms. Speculative decode
  introduces its own correctness surface (see skill
  `exo-speculative-decode-correctness`) and would confound this test entirely.
- `temperature: 0` on every graded call.
- A single smoke prompt returns `logprobs` populated. If logprobs come back
  null, STOP -- Gate C is dead and the run is worth much less; report that
  rather than proceeding silently.
- No concurrent bench/probe processes on the cluster (documented pitfall:
  overlapping GPU load has previously been misread as a code regression).

Known limitations, stated up front
-----------------------------------

- Token offsets are computed with the local checkpoint tokenizer. The
  chat template adds a prefix of unknown-at-generation-time length, so
  absolute offsets are *approximate* until the first response returns
  `usage.prompt_tokens`. The harness calibrates: it measures the template
  overhead once with a probe request, then re-generates the offsets with
  the measured offset applied. Boundary placement is only as good as that
  calibration -- hence `delta` defaults to 24 tokens (well inside the
  calibration error budget) and hence the straddling needles, which are
  robust to +/- a few tokens of drift.
- This tests `prefill_batched()`. If the cluster routes a given request to
  the serial `prefill()` path, the flag is not even read. The harness
  asserts the batched path ran before trusting any boundary analysis.
- A negative result bounds the effect size; it does not prove correctness.
