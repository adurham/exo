# CAMPAIGN 2, ROUND 10 - PRE-REGISTRATION

This document is committed BEFORE any round-10 measurement, and the governing statistic is fixed here in advance.

## 1. GOVERNING STATISTIC: the RESIDUAL

The governing statistic is the residual, defined exactly as:

```
residual_ms = prefill_s*1000 - ((prompt_tokens - 1) / prompt_tps) * 1000
```

Field provenance (all from bench/long_decode_probe.py's output JSON):
- prefill_s = client-observed TTFT, JSON field `prefill_s`
- prompt_tokens = JSON field `prompt_tokens`
- prompt_tps = JSON field `server_stats.prompt_tps` (server-side, timed INSIDE prefill())

Use (prompt_tokens - 1) because prefill() receives prompt_tokens[:-1]; using the raw count shifts the bucket by <5ms.

Raw TTFT is DEMOTED to diagnostic-only this round - the exact inverse of R9.

## 2. WHY THE RESIDUAL GOVERNS (justification already on record, not invented after seeing data)

- The rendezvous sleep is pre-prefill() BY CODE: runner.py:580 gates it, :582 arms the deadline, the drain loop :594-620 completes before prefill() is entered. The window therefore CANNOT enter the in-prefill compute term.
- The in-prefill compute term is arm-INDEPENDENT noise, PROVEN by its sign flip across R9's two independent pairs: +861 ms on pair 1 versus -132 ms on pair 2 on the 2K instrument. A constant sleep cannot change sign.
- Therefore raw TTFT = (arm-sensitive residual) + (arm-independent noisy compute term), and the residual is the ONLY bucket the knob can occupy.

## 3. GOVERNING INSTRUMENT: the SHORT (~200-token) prompt

Fixed in advance for reasons independent of any round-10 value:

(a) the round brief mandates n=25 short reps for the confirmatory pair;
(b) R9 section 8 recommendation 2 (the closing design) specifies short;
(c) mechanism: the residual bucket also contains prompt-size-dependent terms (prompt serialization/transfer, task dispatch, the all_gather barrier). The short instrument minimizes non-window content in the residual, so a 200 ms constant window is a larger fraction of a smaller bucket.

The 2K residual is reported as a SECONDARY DIAGNOSTIC and does NOT veto. If the 2K residual gap falls outside [150, 250] ms it MUST be named prominently in the REPORT as a limitation.

### 3.1 DISCLOSURE (integrity)

R9's REPORT section 2.2 already published short and 2K residual MEDIANS. This pre-registration is therefore NOT written blind to them. It IS written blind to (i) all round-10 fresh-pair data, and (ii) the per-rep residual distributions and ranges, which R9 never published. The instrument choice (short) is inherited from the round brief and R9 section 8, not selected this round. Recorded explicitly so a reader can discount it appropriately.

## 4. THE SHIP BAND - applied verbatim to the FULL boot set

The full boot set = R9's four boots (residual recomputed from their raw JSONs) PLUS round-10's fresh confirmatory pair = 3 RV=200 boots and 3 RV=0 boots.

Define spread(RV200) = max - min across the RV=200 boots' short residual MEDIANS.

- C1: min(RV=200 short residual medians) - max(RV=0 short residual medians) > spread(RV200). In words: every RV=0 boot's short residual median lies below every RV=200 boot's, by more than the RV=200 arm's own between-boot spread.
- C2: the pooled short residual gap lies in [150, 250] ms, where pooled gap = median(all RV=200 short residual reps) - median(all RV=0 short residual reps), reported as a positive magnitude; the sign must be RV=0 LOWER.

SHIP requires C1 AND C2. Anything else -> HOLD, and the REPORT must name which condition failed and by how much.

### 4.1 Required transparency breakdowns (do not gate, but MUST be reported)

Apply C1/C2 three times and report all three: (i) R9-only recompute, 4 boots; (ii) fresh-pair-only, 2 boots; (iii) full set, 6 boots. The FULL SET is the governing application. If the full set passes but either sub-analysis fails, that MUST be named in the REPORT as a caveat on the ship.

## 5. OUTLIER POLICY

NO outlier exclusion, none, at any stage. Medians and FULL RANGES are reported for every boot. Medians are used precisely because they are robust; excluding reps would be an unfalsifiable degree of freedom. n=25 short reps per fresh boot.

## 6. BYTE-IDENTITY GATE (HARD - can only veto a ship, never enable one)

Three prompts (short / 2K / 89K), temp=0, fixed --run-id r10id, matched --max-tokens across arms. Compare `reasoning_content` + `content` CONCATENATED (DSv4 spends small budgets entirely in reasoning_content, so comparing content alone compares two empty strings and reports a false PASS). TWO captures per arm per prompt; within-arm compared FIRST. Every rep must show prefix_cache_hit = none.

Decision rule, fixed now:
- within-arm identical on BOTH arms AND cross-arm identical -> PASS
- within-arm identical on BOTH arms AND cross-arm differs -> HARD FAIL -> HOLD
- within-arm differs on EITHER arm -> that prompt's cross-arm comparison is VOID (nondeterministic regime); it neither passes nor fails and does not block a ship otherwise supported. An arm that cannot reproduce ITSELF cannot testify against the other arm.

## 7. CLEAN-LOGS VETO (HARD)

On the RV=0 boot, under mixed short + 89K traffic: zero errors, zero rank disagreement, zero task-set mismatch, zero 'out of sync' / 'closed communication'. The pre-existing background warnings enumerated in R9 section 2.3 (HF catalog poll for GLM-4.7-8bit-gs32, invalid model cards, mx.metal.get_*_memory deprecations, transformers rope notice, normal [jaccl-v2] trace, the error.svelte.js build-artifact filename) are excluded BY NAME. Any other error -> HOLD.

## 8. RV VERIFICATION (HARD)

The arm's EXO_BATCHED_PREFILL_RENDEZVOUS_MS value is read off the REAL runner PIDs via `ps eww` on BOTH nodes before any rep on that arm. Never infer it from the launch command. A boot whose PIDs disagree with the intended arm is VOID and its reps are discarded.

## 9. TASK 3 - R7 steel-BI 89K self-control (decision rule fixed now)

On the RV=200 boot (production config; MLX_STEEL_BATCH_INVARIANT defaults to 1 at start_cluster.sh:269), capture the 89K prompt at temp=0 THREE times at a fixed run-id and identical max-tokens.

- All three byte-identical -> R7 section 4's 89K leg STANDS as-is.
- Any two differ -> R7 section 4's 89K leg is VOID (same-arm nondeterminism at 89K on the production config), and only R7's <8192 and 5-fixed-prompt legs stand.

MLX_STEEL_BATCH_INVARIANT is NOT flipped this round. This is a control on existing evidence, not a re-test.

## 10. SHIP ACTION, fixed in advance

SHIP -> start_cluster.sh:136 changes from : "${EXO_BATCHED_PREFILL_RENDEZVOUS_MS:=200}" to :=0, with a dated comment citing R7/R9/R10, the measured residual gap, and the V4 c>=2 decision. KEEP the knob - the env override must still work. Relaunch onto the new default with NO env override and verify RV=0 on the real PIDs, proving the default itself took effect. Plus API 200 and clean logs.

HOLD -> start_cluster.sh untouched at 200; cluster restored to production RV=200.

## 11. CONSTRAINTS

Time-box 3h. 2-3 relaunches pre-registered. No pushes. Never `git add -A`. Never `git stash`. No new harness - reuse R9's instrument bench/long_decode_probe.py (the R7 A2 script) unmodified. Ranges, never bare means.
