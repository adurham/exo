# Pre-registered: 352.6K correctness + harness verification of the shipped stacked config (2026-08-28)

Live config under test (verbon3 = production): exo `75d2402dd` + mlx-lm `d098642`,
env per /tmp/verbon3_launch.sh (VERIFY_BATCH=1 MIN_CTX=8192, TP_SHARD=1,
CLEAR_CACHE_INTERVAL=64, GAMMA=3, MTP=1, DSPARK=1, BOOKKEEP_FAST/DRAFT_EPILOGUE absent=OFF).

## Static audit (done before any run)
- BOOKKEEP_FAST (a36de3e28): pure if/else; OFF path identical semantics (rename + `_int_at` typed accessor).
- DRAFT_EPILOGUE (0fd40a94a + 71a8ffb9c): `_dspark_sample` hoisted UNCHANGED; inline draft call identical in else-branch; OFF -> `_cached=None` -> else. New `_dspark_next_draft` dict stays empty when OFF.
- Both flags parse `os.environ.get(F,"0")=="1"` -> absent == "0" == OFF. Verified absent in live env (ps eww, both nodes).
- dda9237..d098642 (submodule): ONLY `_EXACT_TOPK_PARAM_CAP` env-gate, default 64 == old hardcoded 64. Zero behavioral delta.
- TP_SHARD: auto_parallel only, draft-head MoE FFN sharding (draft numerics only; committed tokens Gate-A-shielded vs target argmax; CAN legitimately shift acceptance trajectory).
- JIT wait fix (75d2402dd): src/exo/api only. No inference path.
- Laptop HEAD 7e9e38274 = 75d2402dd + 3 docs-only commits. Deployed = same inference code.

## Phase 1 — Tier-1 short-ctx (live stacked config, /v1, temp=0, max_tokens=512)
spec_degen_capture.py 7-prompt set. PRIMARY comparator: Aug-26 spec-OFF captures
(/tmp/ab/tier1/degen_specoff_512.json) — base greedy is bit-deterministic at short ctx
and unaffected by every stacked change. Aug-26 spec-ON rowseq captures = secondary/informational
(TP_SHARD may legitimately shift the residual-sensitive trajectory).
BARS:
- sys_capital_france, sys_count_to_five (short, finish=stop): byte-identical (content AND
  reasoning_content) to spec-OFF captures. Any diff = REGRESSION (hard fail).
- sys_primary_colors: PASS iff byte-identical to spec-OFF OR (divergent AND deterministic:
  3/3 identical across immediate reruns — the documented 0.023%/row MoE rowseq residual class,
  accepted at promotion). New NONdeterminism = FAIL.
- 4 length-truncated prompts: recorded, no bar (same as the Aug-26 protocol).

## AMENDMENT 3 (13:55, after G0'' arms landed, BEFORE the 100K base pair runs)
Measured at 352.6K: S1=S2=S3 BYTE-IDENTICAL (base deterministic at depth under the fixed
prompt + current config) and B0=B1 BYTE-IDENTICAL (batched deterministic). B/R/S are
deterministic DIFFERENT trajectories (first div: B-vs-R char 24, R-vs-S char 54) — the
same class as the documented Tier-1 0.023%/row MoE residual, NOT noise. The pre-registered
envelope bar (drift <= base drift) is therefore DEGENERATE here: base drift = 0, and any
deterministic path difference "fails" it. The formal FAIL is recorded honestly. Two
interpretive follow-ups, pre-registered now:
- 100K fixed-prompt spec-off pair (S100k_0 vs S100k_1, current config): tests whether the
  shipped G0'' denominator ("base-vs-base 99.3% at 100K") reflected real base
  nondeterminism or a fresh-nonce-per-run harness artifact (p3 build_prompt embeds a
  uuid nonce; the Aug-27 runs generated a NEW prompt per run).
- Verdict framing: correctness at depth = per-arm determinism + bounded deterministic
  path divergence + healthy output quality (soak factcheck) — NOT drift-envelope pass,
  which is unfalsifiable when the base is deterministic and vacuous when it isn't.

## AMENDMENT 2 (12:07, after B landed, BEFORE any R/S run — no R/S data exists)
- DISCOVERED: production spec path runs EXO_DSV4_SPEC_EOS_BAN=0 (natural-end semantics,
  documented in the baseline doc — the spec-path EOS ban caused the degen bug). /bench's
  EOS ban therefore does NOT apply to spec-ON arms: B ended finish=stop @513 tokens
  (in family with von3 natural ends 338-1196). The Phase-3 bars "finish=length" and
  "n_chunks==4000" were MIS-REGISTERED against actual production semantics — reported
  as bar-spec errors, NOT waived silently. All health bars passed (28.95 tok/s, median
  gap 0.1ms, max 169ms, zero faults, zero swap delta, post-soak Tier-1 7/7 identical).
- G0'' verdict window: mismatch fractions computed over the UNIFORM window
  min(len(B),len(R),len(S1..S3)) for ALL pairs (binding), full pairwise min-window
  numbers reported informationally. Rationale: spec arms end naturally (~500), S arms
  are EOS-banned (2000); unequal windows inflate the S-envelope (frac -> 1 past first
  divergence) and would weaken the gate. First-divergence indices reported for all pairs.
- ADDED Phase 5b (genuine long soak at depth, replaces the impossible 4000-token bar):
  after production restore, ONE run with a NEW long-output prompt (same 352.6K corpus,
  task = exhaustive section-by-section restatement) max_tokens 4000. Bars: finish=length
  OR n_tokens>=2000; zero fault lines; median gap <300ms; max gap <5s; swap delta
  <500MB/node. This yields >=600-1200 uninterrupted spec cycles at depth on the
  production config.

## Phase 2/4/5 — G0'' drift gate at 352.6K (/bench endpoint, EOS-banned, use_prefix_cache=false)
AMENDMENT (pre-registered 11:45, BEFORE any deep run): arm B and the Phase-3 soak are ONE
run (max_tokens=4000). temp=0 argmax decode is prefix-stable, so its chunk stream's first
min-len positions are the arm-B comparator; the soak bars read the full 4000. If the soak
run faults/collapses, BOTH phase verdicts fail (honest coupling). R and S arms stay 2000.
ONE fixed prompt (nonce frozen, built once via p3_depth_anchor_probe.build_prompt(352600),
saved /tmp/ab/g0_352/prompt_352k.txt, byte-identical for ALL runs). temp=0, max_tokens=2000.
/bench bans EOS on every arm equally (fixed-length streams; removes natural-end length variance).
use_prefix_cache=false on /bench is honored server-side (generate.py:2032) -> no KV prefix reuse
even with identical prompts back-to-back. Chunk-positional comparison (each SSE delta = one
detokenizer emission), mismatch fraction over min length + first-divergence index.
ARMS:
- B (batched): live verbon3 config. 1 run.
- R (rowseq): relaunch verbon3 env with EXO_DSV4_VERIFY_BATCH=0, all else identical. 1 run.
  Arm assertion: ps eww shows VERIFY_BATCH=0 on both nodes (+ MTP-PROF verify_ms sanity).
- S (spec-off base): relaunch stripped env == verbon3 minus spec (EXO_SPECULATIVE=0,
  EXO_DSV4_MTP=0, EXO_DSV4_DSPARK_FORCE_LOAD=0), KEEPING CLEAR_CACHE_INTERVAL=64 and all
  MLX/batch-invariant knobs for numerics parity. 3 runs -> 3 pairwise base-vs-base drifts.
BARS (pre-registered):
- PASS iff mismatch(B,R) <= median(pairwise mismatch among S1,S2,S3).
- Fallback if the base is (near-)deterministic at 352.6K (median base mismatch < 5%):
  bar becomes mismatch(B,R) < 5% too — no hiding behind a nonexistent drift envelope.
- Report all first-divergence indices; no post-hoc bar adjustment.

## Phase 3 — Depth soak on live config (before any relaunch)
p3-style /bench run at 352.6K, max_tokens=4000 (~1100+ spec cycles at a~3.25/cycle), temp=0.
BARS: HTTP 200; finish=length; n_tokens=4000; overall decode >= 15 tok/s; median inter-token
gap < 300ms; max gap < 5s; zero jaccl faults/WC_ERR/segfault/kill-switch lines in either node
log during the window; swap delta < 500MB per node (vm.swapusage before/after).
POST-SOAK cross-request state check: immediately rerun sys_capital_france (/v1, same instance):
byte-identical to Phase-1 capture. Exercises request-boundary reset after a deep spec session.

## Phase 6 — End state
Relaunch verbon3 (production). Paris smoke + ps eww env check. Cluster left healthy.

## Harness cleanliness bars (all phases)
- Runner env verified via `ps eww <pid>` on BOTH nodes for every launch (never argv).
- Repo SHAs on both studios == 75d2402dd, trees clean; installed mlx-lm == submodule d098642
  (diff -q of deepseek_v4.py + dsv4_mtp.py vs site-packages).
- Node logs rotated per launch (append-mode log must be launch-scoped for greps).
- One probe at a time; SIGTERM only; screens cleaned; /state checked idle before every relaunch.
- All artifacts under /tmp/ab/g0_352/ + verdict JSON; every number in the final doc traceable to a file.
