# DSpark MTP @352.6K — correctness + harness verification of the shipped production config (2026-08-28)

Closes the two open verification gaps from the 352.6K campaign: "no correctness
issues" and "no harness issues" were previously extrapolated from gates run at
100K on commit `dda9237` — BEFORE the bookkeeping trim (`a36de3e28`),
draft-epilogue fusion (`0fd40a94a`+`71a8ffb9c`), DSpark TP-shard fix
(`2d85ccdcb`), and JIT wait fix (`75d2402dd`) stacked on top. This session
re-verified against the EXACT shipped config, live, at 352.6K.

Pre-registration + amendments (written before the data they bind):
`/tmp/ab/g0_352/PREREGISTER.md`. All artifacts: `/tmp/ab/g0_352/`.

## Config under test (verified live via `ps eww` on BOTH nodes, every launch)

- exo `75d2402dd` (laptop HEAD `7e9e38274` = same + 3 docs-only commits),
  mlx-lm `d098642`, both studio repos clean, installed mlx-lm == submodule
  (diff -q), exo editable (runners import `~/repos/exo/src` live).
- verbon3 env = production baseline + `EXO_DSV4_DSPARK_TP_SHARD=1` +
  `EXO_MLX_CLEAR_CACHE_INTERVAL=64`; `EXO_DSV4_BOOKKEEP_FAST` and
  `EXO_DSV4_DRAFT_EPILOGUE` ABSENT (= "0" = OFF; both parse
  `os.environ.get(F,"0")=="1"`, so absent ≡ explicit 0).

## Static audit of the stacked commits (OFF-path safety)

- Bookkeeping trim: pure `if _BOOKKEEP_FAST:` branches; OFF path is a rename +
  typed accessor (`_int_at`), bitwise-identical semantics.
- Draft-epilogue fusion: `_dspark_sample` + the inline draft call hoisted
  UNCHANGED into the `else` branch; OFF ⇒ `_cached=None` ⇒ else; the new
  per-uid cache dict stays empty.
- Submodule dda9237..d098642: ONLY the `_EXACT_TOPK_PARAM_CAP` env-gate
  (default 64 == the old hardcoded 64). Zero behavioral delta.
- TP shard: `auto_parallel.py` only — draft-head numerics only; committed
  tokens are Gate-A-shielded (strict argmax vs TARGET verify logits).
- JIT wait fix: `src/exo/api` only; no inference path.

## Phase 1 — Tier-1 short-ctx byte-identity (live stacked config): PASS

7-prompt degen set (spec_degen_capture.py, /v1, temp=0, max_tokens=512):

- `sys_capital_france`, `sys_count_to_five`: **byte-identical to the Aug-26
  spec-OFF captures** (content AND reasoning). The stack did not change
  short-ctx rowseq behavior.
- `sys_primary_colors`: byte-identical to the **Aug-26 spec-ON trajectory**
  (262c/1228c) — the documented, accepted 0.023%/row MoE rowseq residual,
  unchanged in position and content by TP_SHARD.
- Determinism: 2 consecutive live captures — **7/7 prompts byte-identical**
  run-to-run (including all four length-truncated long prompts).

## Phases 2-5 — G0''-style drift gate at 352.6K (fixed frozen prompt)

One frozen 352.6K prompt (`prompt_352k.txt`, nonce fixed once;
`usage.prompt_tokens=352600` exactly, `cached_tokens=0`, TTFT ~1000-1060s on
every run = every run genuinely re-prefilled; /bench + use_prefix_cache=false).

| arm | config | runs | result |
|---|---|---|---:|
| B (batched, production) | verbon3 | 2 (across TWO separate instance launches) | **byte-identical to each other**; 510 tok natural end |
| R (rowseq) | verbon3 + VERIFY_BATCH=0 | 1 | 305 tok natural end |
| S (spec-off base) | stripped (+CLEAR_CACHE=64) | 3 sequential | **all 3 byte-identical**; 1965 chunks, finish=length |

- **Every arm is internally byte-deterministic.** The batched path is
  deterministic even across a full cluster relaunch (B0 vs B1).
- Arms differ from each other deterministically: B-vs-R first divergence at
  char 24, R-vs-S at char 54 — early low-margin argmax flips from the known
  bounded MoE numerics residual, then textual divergence. Same class as the
  accepted Tier-1 `primary_colors` divergence; NOT noise, does NOT randomize.
- All three arm outputs are coherent, correct summaries (manually inspected;
  zero U+FFFD; no repetition).
- **Formal pre-registered envelope bar (mismatch(B,R) ≤ base drift): FAIL,
  recorded honestly — but the bar is degenerate**: measured base drift = 0.0%
  (the denominator the bar assumed nonzero), so ANY deterministic path
  difference fails it. Amendment 3 (pre-registered before the follow-up runs)
  reframes the meaningful correctness property: per-arm determinism (PASS) +
  bounded deterministic divergence (verified) + output quality (verified).

## HARNESS FINDING (the significant one): the "base is nondeterministic at depth" premise does not reproduce under a fixed prompt

The shipped G0'' gate's denominator ("base-vs-base run-to-run drift **99.3%**
at 100K") and the campaign's "295 vs 977 tokens, same prompt" nondeterminism
claims were measured with `build_prompt()` called **fresh per run — which
embeds a fresh uuid4 nonce in the header**. Those "same prompt" runs were not
byte-identical prompts.

Re-tested with byte-identical fixed prompts on the current config:
- 100K spec-off pair: **byte-identical** (979 chunks each, past the natural
  EOS point — EOS-banned continuation included).
- 352.6K spec-off triple: **byte-identical**.

⇒ At the shipped config, base decode is **byte-deterministic at depth under a
fixed prompt**. The historical 99.3% (and the promotion doc's "the batched
path adds less noise than the base already has") is best explained as a
fresh-nonce harness artifact (the batch-invariant MLX flags were already
default since 2026-07-10 `69a770084`, so flag drift does not explain it; the
prompt nonce does).
The 24-run THROUGHPUT verdicts are unaffected (tok/s doesn't care about the
nonce). The G0'' PASS/FAIL logic specifically should not be cited going
forward without this caveat; the honest current statement is stronger anyway:
**batched verify is deterministic; it selects a different-but-valid greedy
trajectory than rowseq/sequential (bounded numerics residual, quality
verified).**

## Phase 3/5b — depth soak on the live production config: PASS

- Combined B0 run: 28.95 tok/s fixed-window, median gap 0.1ms, max 169ms.
- 5b long-output soak (same corpus, exhaustive-restatement task): **4000/4000
  tokens, finish=length, 30.46 tok/s, median gap 0.08ms, p99 144ms, max gap
  376ms** — ~1200+ spec cycles at depth, zero stalls (collapse signature = 2-4s
  gaps; nothing within 10x of that). mean_accept 2.438/3 during the long run.
- Swap: 50MB / 72MB peak (bars <500MB); zero WC_ERR / segfault / transport
  fault / degeneration lines in either node log for the whole session (the
  only WC_ERR in the verbon3 log remains the documented 08:52 incident).
- **Factual recall check** (corpus is procedural ⇒ ground truth computable):
  34 verifiable section claims in the soak output (3 config numbers, 31 stage
  pairs) — **0 wrong**.
- **Post-soak state check**: full 7-prompt Tier-1 rerun on the same instance
  immediately after the deep session — **7/7 byte-identical** to pre-soak.
  Request-boundary reset after a deep spec session is clean.

## Harness cleanliness audit of the shipped 352.6K verdict artifacts

- `summary_von3.jsonl` reproduces the published verdict exactly: 8 genuine
  runs (2 voided slots recorded as 0.0/n=0, excluded), 0 collapses, median
  28.44 vs stripped-OFF 24.19 (n=9) = **+17.57%**, bootstrap CI
  [+8.76, +27.79] — matches the docs to the decimal.
- Pre-fix gate reproduces: n=16, 4 collapses, median 19.87, −17.87%,
  CI [−57.5, +9.8] — FAIL as documented.
- Raw run JSONs cross-check against both summaries: 0 mismatches.
- von3 telemetry CSVs reproduce the doc claims exactly: swap_used peak
  97MB/80MB; wired peak 92.0/93.5 GB; pageouts-delta 44/49.
- Node-log Tracebacks during launches are pre-existing model-card pydantic
  validation noise (6bit/4bit/Qwen cards), unrelated to serving.
- Every launch in this session: env asserted via `ps eww` on both nodes,
  logs rotated per launch, SIGTERM-only, screens cleaned, /state idle-checked.

## Verdict

- **Correctness at 352.6K on the shipped config: VERIFIED** — deterministic
  per-config output (incl. across relaunches), short-ctx byte-identity vs
  spec-off intact, bounded deterministic path divergence of the known residual
  class (no new divergence source from the 4 stacked changes), clean deep soak
  with verified factual recall, clean request-boundary state.
- **Harness: one real methodological artifact found and documented** (fresh-
  nonce prompts inflating historical "base nondeterminism"); all shipped
  verdict numbers reproduce from raw artifacts; no evidence of any other
  harness contamination.
- **Cluster end state: verbon3 production config restored**, env-asserted,
  cold JIT auto-place + Paris smoke PASS (exercising the `75d2402dd` wait
  path once more).
