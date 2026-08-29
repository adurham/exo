# Pre-registered: P1–P4 forward-plan campaign (2026-08-28)

**Scope:** executes §6 of `dspark-mtp-master-history-2026-08-28.md` — P1 (draft-epilogue
fusion A/B), P2 (c=2 validation), P3 (verbon3 ablation), P4 (same-regime verify-cost
curve) — in priority order. Registered BEFORE any run. Design reviewed by second-opinion
consult 2026-08-28 (gate taxonomy, n-size criticisms, P4 second-regime arm, Bug-3
denominator fix all incorporated pre-registration).

**Stack under test:** exo `75d2402dd` + mlx-lm `d098642` (deployed on both studios,
unchanged since Aug-28 verification). No code changes anywhere in this campaign —
measurement + docs only. All flags exercised already exist.

**Fixed-prompt discipline (fresh-nonce lesson):** every prompt frozen to disk ONCE
before any run and byte-identical across all arms/runs that use it. No
`build_prompt()`-per-run anywhere.

**Launch plan (7 sessions, block design — env flags read at import require relaunch
per arm):**

| # | Config | Used for |
|---|--------|----------|
| L0 | CURRENT live runner (verbon3, EPI-OFF; up since Aug-28, env verified via ps eww) | P1-OFF arms, P4a sweep, P2 correctness + spec-ON throughput |
| L1 | verbon3 + `EXO_DSV4_DRAFT_EPILOGUE=1` | P1-ON arms + Tier-1 |
| L2 | stripped spec-OFF (Aug-28 `specoff_stripped_launch.sh`) | P2 c=2 spec-OFF throughput control |
| L3 | verbon3 with `EXO_MLX_CLEAR_CACHE_INTERVAL=0` (TP_SHARD alone) | P3 arm-SHARD |
| L4 | verbon3 with `EXO_DSV4_DSPARK_TP_SHARD=0` (CLEAR_CACHE alone) | P3 arm-CACHE |
| L5 | verbon3 + `EXO_DSV4_VERIFY_BATCH=0` (rowseq-forced) | P4b rowseq points |
| L6 | verbon3 stock (production restore) | end state + cross-session anchor run |

L0 ordering (pre-registered so P4a cannot contaminate P1 timing): P1-OFF throughput
runs FIRST, then P4a sweep, then P2 legs.

Hygiene every launch: SIGTERM only; runners verified gone before relaunch; one probe
at a time; `ps eww` env verification on both nodes; log offset recorded per launch
(greps scoped to offset, live log never truncated); `/state` idle check before
relaunch; artifacts under `/tmp/ab/p1p4/`; every number in the results docs traceable
to a file.

---

## P1 — Draft-epilogue fusion A/B (`EXO_DSV4_DRAFT_EPILOGUE=1` vs `0`)

Design doc: `dspark-draft-epilogue-fusion-2026-08-27.md`. Theoretical +16% @100K
(draft 10.8 ms overlapped with ~10.3 ms tail). Design predicts the epilogue-computed
draft is IDENTICAL to the inline draft (same anchor + ctx-KV, deterministic), so
committed tokens should be byte-identical ON vs OFF.

**Runs:**
- L0 (OFF): 6× 100K + 3× 352.6K, `/bench`, `use_prefix_cache=false`, temp=0,
  long-output frozen prompts (`prompt_100k_long.txt` NEW — same corpus recipe +
  restatement tail; `prompt_352k_long.txt` = Aug-28 Phase-5b file), max_tokens
  1500 (100K) / 2500 (352.6K).
- L1 (ON): Paris smoke; Tier-1 7-prompt set (/v1, 512 tok, temp=0); then same
  6× 100K + 3× 352.6K with the SAME frozen prompts.

**Pre-registered gates:**
1. **Tier-1:** EPI-ON captures byte-identical (content + reasoning_content) to the
   Aug-28 live-stacked EPI-OFF captures (`/tmp/ab/g0_352/tier1_live.json`) on all 7
   prompts (same config modulo EPI; draft identical by design ⇒ identity expected).
2. **Cross-arm byte-identity at depth:** ON run-streams byte-identical to OFF
   run-streams at BOTH depths (same prompt).
   *Failure taxonomy (pre-registered):* if ON≠OFF but ON is internally
   byte-deterministic (gate 3) AND the divergence is the documented
   deterministic-trajectory class (stable across reruns), classify as
   "deterministic numeric shift" — gate 2 DOWNGRADED (not auto-FAIL), mechanism
   gates still apply, and the divergence must be root-caused before any promote
   decision (design says drafts are identical — a shift indicates scheduling-order
   numerics or a design-analysis error; either must be understood).
   Nondeterministic divergence = hard FAIL.
3. **Within-arm determinism:** all runs of the same (arm, depth) byte-identical to
   each other.
4. **Throughput @100K:** shared-min-window tok/s (window = min chunk-count across
   all 12 runs at that depth; tok/s = (W−1)/(t[W−1]−t[0]) per run). Primary
   criterion: median(ON) − median(OFF) ≥ +8% AND min(ON) > max(OFF). (Bootstrap CI
   reported informationally only — n=6/arm is too lumpy for a CI-based bar; noted
   pre-registration.) If gate 2 took the taxonomy branch, the window is
   min(shared-prefix length, min chunk-count) — pre-registered now.
5. **352.6K health:** no collapses (every run ≥ 15 tok/s overall), swap delta
   < 500 MB/node (telemetry sampler), zero jaccl fault/WC_ERR/kill-switch lines in
   either node's log window. Throughput @352.6K reported with the same window
   method (informational + directional; n=3/arm — no hard % bar at depth).
6. **Mechanism:** MTP-PROF windowed means — ON-arm consume cycles show draft phase
   ≈ 0 ms vs OFF ~10.8 ms; a new epilogue-draft cost may appear inside the cycle
   total. Expected exceptions (pre-registered): the FIRST cycle of each stream and
   any post-invalidation cycle run the inline draft (draft_ms reappears on those
   cycles); at γ-prune boundaries the consume path re-prunes the cached full-width
   draft. Acceptance mean/γ must be statistically indistinguishable ON vs OFF
   (identical drafts ⇒ identical acceptance when gate 2 passes byte-identical).

**PROMOTE bar (flip default ON in start_cluster.sh):** gates 1–3 clean (or gate-2
taxonomy branch fully root-caused as benign), gate 4 passed, gate 5 clean. Otherwise
flag stays default-OFF and the results doc records why.

**Long-output fallback (pre-registered):** if any spec arm ends naturally < 300
chunks at 100K (window too small), a longer-tail prompt variant is generated ONCE
and BOTH arms rerun — trigger and remedy fixed now.

## P2 — c=2 validation under the promoted config

**AMENDMENT P2-1 (2026-08-28 19:20, after L0 stage-2 short-c2 legs landed, BEFORE
L2 runs):** all 3 short-c2 spec-ON repeats hit the degeneration kill-switch —
deterministic `token cycle period=3 '.</think>Paris'` at completion_token=61 on the
cap stream (uid varies), second stream aborted collaterally mid-phrase (its text
also shows unstopped repetition: "One...five." ~2.5×). c=1 on the SAME launch was
clean (Aug-28 Tier-1). ADDED to L2 (spec-OFF stripped config): 3 repeats of the
identical concurrent short pair via /v1 — decides spec-specific vs
batched-generator-c=2-generic before any conclusions are drawn. Both outcomes
pre-committed: spec-OFF clean ⇒ finding attributes to the spec c=2 path (rowseq
B×L=8); spec-OFF degenerates too ⇒ attributes to the shared c=2 EOS/stop handling,
and the P2 verdict says so.

**AMENDMENT P2-2 (2026-08-28 19:35, during L0 stage-2, before Bug-3/throughput legs):**
observed mechanics of the short-c2 degeneration, recorded for the verdict: each of
the 3 repeats hit the kill-switch (`action=error`) on the cap stream at token 61,
and the ABORT PATH then crashed the whole runner —
`ValueError: [reshape] Cannot reshape array of size 2 into shape (1,1,1,1)` in
mlx-lm `cache.py:2050 fetch_overlap_carry` — killing BOTH streams (availability
bug, distinct from the degeneration itself) and forcing instance deletion + JIT
reload. Consequence: c2deep repeat 0's B stream landed in the reload window →
503; that slot is VOIDED (not evidence about the batched path), repeat 0 A ran
effectively as c=1-deep. Repeats 1+ (both streams admitted concurrently) are the
binding contamination/determinism evidence. The reshape crash is a NEW finding
to file under P2 results: the BS=2 degen-abort cleanup path is broken.

Gap (h): every campaign run was c=1; batched verify at B=2 (deep) and rowseq at
B×L=8 (short) are production-unvalidated. June Bug-3 residual (80% adversarial
final-digit flip, PP-era) never re-tested under TP batched verify.

**Correctness leg (L0), all temp=0:**
- **Short c=2 (rowseq path):** 3 repeats of two CONCURRENT /v1 streams —
  sys_capital_france + sys_count_to_five (Tier-1 prompts, 512 tok). Bars:
  per-stream byte-determinism across the 3 repeats; each stream's output compared
  to its c=1 Tier-1 capture — byte-identical is the expected strongest pass;
  deterministic-divergent-bounded = pass-with-note (B=2 rowseq batch-invariance
  never promised); nondeterministic or cross-stream content swap = FAIL.
- **Deep c=2 (batched B=2 path):** 3 repeats of two concurrent 100K streams with
  DISTINCT frozen prompts carrying distinct embedded reference codes
  (A: "ALPHA-7749", B: "BRAVO-3317"; tail asks to state the code). Bars:
  per-stream determinism across repeats; each stream returns its OWN code
  (contamination check); zero degeneration/kill-switch/fault lines; c=2-vs-c=1
  trajectory difference is EXPECTED (batch-invariance fails 0-ulp — documented) and
  is not a bar.
- **Bug-3 needle (batched B=2):** 6 frozen 100K prompt variants, each embedding an
  activation code ending in a distinct digit, tail = "Reply with ONLY the
  activation code." Run as 3 concurrent pairs. Metric: flips/6 (deterministic
  system ⇒ per-variant flip is 0/1; the PP-era "80%" was across nondeterministic
  repeats — denominator fixed to variants, pre-registered). Informational bar:
  report the rate; a deterministic flip is the KNOWN bounded residual class, not a
  new regression; >4/6 flips OR any non-final-digit corruption = escalate to
  investigation before any c=2 production claim.
- **Skipped (pre-registered):** math_digit_sum-style self-verification loop probes
  at c=2 — the batched-verify self-doubt class is documented behavior-level, not
  c=2-specific; out of scope here.

**Throughput leg:** 3 concurrent-pair runs @100K spec-ON (L0; streams =
`prompt_100k_long.txt` + `prompt_100k_long_B.txt`) vs 3 pairs spec-OFF (L2, same
prompts). Metric: per-stream AND aggregate shared-min-window tok/s (window = min
chunk-count across all c=2 runs at the depth — equalizes the natural-end vs
EOS-ban length-regime mismatch, pre-registered). Report deltas; NO promotion claim
unless both correctness legs clean. c=1-vs-c=2 per-stream cost reported
informationally against P1-OFF singles.

## P3 — verbon3 ablation @352.6K (TP_SHARD vs CLEAR_CACHE=64)

Both fixes shipped bundled; individual contributions unmeasured.

**Arms (3× 352.6K each, `prompt_352k_long.txt`, telemetry sampler running):**
- arm-SHARD (L3): `TP_SHARD=1, CLEAR_CACHE_INTERVAL=0` → isolates the
  clear-interval throughput penalty (vs P1-OFF "both" trio) and whether the shard
  alone eliminates collapse.
- arm-CACHE (L4): `TP_SHARD=0, CLEAR_CACHE_INTERVAL=64` → head fully replicated
  (~+3–3.5 GB/node); collapse risk accepted — that is the point. Run timeout
  3600 s hard cap per run; a collapsed run is recorded, not retried.
- arm-BOTH: reuse P1-OFF L0 352.6K trio (identical config). Cross-session
  validity anchor (pre-registered): ONE fresh 352.6K run on L6 (production
  restore) — if within ±5% of the L0 trio median, reuse stands; if not, the
  ablation deltas are reported with a session-confound caveat.

**Outputs (pre-registered):** CLEAR_CACHE=64 throughput cost at depth = arm-SHARD
median vs arm-BOTH median (the 5–15% estimate has never been measured at depth);
shard memory contribution = wired/swap peaks arm-CACHE vs arm-BOTH; collapse
tallies per arm (n=3 distinguishes ~0 from ~1 rates only — no percentage claims
from 1/3, pre-registered). No hard pass/fail bar — this is a quantification
session; results feed the production-env recommendation.

## P4 — same-regime verify-cost curve + MIN_CTX placement

The "1455.8 ms @14K" figure was FULLBLOCK-regime; no same-regime point exists.
MIN_CTX=8192 placement is unvalidated. NOTE (pre-registered asymmetry): rowseq
below 8K is ALSO a correctness guarantee (short-ctx byte-identity) — the placement
verdict can only move MIN_CTX UP, never below ~8K, regardless of perf.

**P4a (L0, production batched config, after P1-OFF runs):** one run each at
4K and 7.5K (rowseq regime; 7.5K + ≤600 gen stays < 8192 — the 8K straddle point
is deliberately avoided, pre-registered) and 9K, 14K, 32K, 64K (batched regime;
9K is batched from cycle 1). Frozen prompts per depth (summarise tail),
max_tokens 600. If a run ends < 150 chunks (< ~50 cycles), rerun once with the
long-tail variant of that depth (pre-registered remedy). 100K point = P1-OFF runs.
- Extraction: MTP-PROF cumulative dumps → windowed per-run means
  ((mean_b·n_b − mean_a·n_a)/(n_b − n_a) between the last dump before run start
  and last before run end), per phase (draft/verify/accept/total). Per-point
  distributions (min/max within window) reported, not just means.
- SPEC_TRACE per-cycle lines used instead if they carry per-cycle verify ms
  (checked on-node; whichever source is used is named in the results doc).

**P4b (L5, rowseq-forced `VERIFY_BATCH=0`):** runs at 14K and 32K, same prompts →
rowseq verify_ms ABOVE the boundary. This retires the 1455 ms claim with a
same-stack rowseq measurement and gives the second regime at the depths where the
placement decision lives.

**Pre-registered verdict rule:** MIN_CTX=8192 placement SUPPORTED if batched
verify_ms < rowseq verify_ms at 14K (and 32K); if rowseq is cheaper at 14K, the
recommendation is to raise MIN_CTX to above the measured crossover. The 4K/7.5K
batched-side comparison is NOT available (batched forbidden below 8K by
correctness policy) — curve shape at those points is informational (rowseq cost
growth vs depth). Additionally: verify_ms vs ctx curve (both regimes where
measured) is THE deliverable; the "cliff" question is closed by stating the
measured batched 14K point against the historical 1455.8 ms FULLBLOCK number.

---

## Deliverables

Per phase: results doc in `docs/` + PERFORMANCE_HISTORY chronological entry +
master-history §5/§6 status updates; commits after every milestone (docs-only
expected; any code change would go through the submodule-first commit discipline).
End state: cluster restored to stock verbon3 (L6), Paris smoke + anchor run +
`ps eww` verification recorded.
