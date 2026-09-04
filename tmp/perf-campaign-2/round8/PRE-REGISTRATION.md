# CAMPAIGN 2, ROUND 8 — PRE-REGISTRATION
Written BEFORE any measurement. The last throughput round of campaign 2.
Authority: docs/PERFORMANCE_HISTORY.md "CAMPAIGN 2 — REVIEW AFTER R7" (decode at a physics
floor; prefill exhausted modulo the I12 audit; I11 is user-gated, not cluster-gated).

## Scope
1. I12 — serial-vs-batched prefill parity CODE AUDIT (measurement ONLY on a found bypass).
2. I15 (kernel-launch count) + I9 (GPU clock) one-shot diagnostics, zero relaunch.
3. I11 — expert-precision EVIDENCE PACKAGE. **NOT shipped.** Cluster ends on 6-bit.

## Environment observed at round start (2026-09-04)
- Cluster LIVE, ~28 min uptime, shipped 6-bit config, TP worldSize=2 (`MLX_JACCL_SHARDING_MODE=Tensor`).
- node1 = macstudio-m4-1 (192.168.86.201), node2 = macstudio-m4-2 (192.168.86.202).
- Free space `/System/Volumes/Data`: node1 **111 GiB**, node2 **134 GiB**.
  Binding headroom = **min = 111 GiB**. Deployed DSv4-Flash weight sets are ~144-165 GB each.
- Gate scripts VERIFIED TO EXIST before citing (R7 caught a brief citing a nonexistent gate):
  `bench/long_decode_probe.py` (215 L), `bench/ab_probe_tier1.py` (144 L),
  `bench/quality_probe_dsv4.py` (424 L). All three present.
  NOTE: R7 established `ab_probe_tier1.py` does **not** self-report "7/7" — do not cite that band.

## BANDS — pre-registered verbatim

### I12 (audit verdicts, per item)
Items: tiled compressed SDPA (`EXO_DSV4_QUERY_TILED_SDPA`, P09, +7.2%); exact fused top-k
(`EXO_DSV4_EXACT_TOPK_PREFILL`, P08, +1.6%); the indexer path; prefix-cache keying/snapshot
insertion; `EXO_PREFILL_STEP_SIZE` chunking; clear-cache cadence.
- **SHARED** — an unbroken file:line call chain from the SERIAL driver (generate.py ~866) reaches
  the optimization. Close.
- **BYPASSED** — the optimization is reachable only under `prefill_batched` / a queue_len>=2 gate.
  This is the real gap.
- **AMBIGUOUS** — chain cannot be established either way. **Treated as a gap**, not as SHARED.
A verdict with no file:line citation is not a verdict.
**If ANY item is BYPASSED:** pre-register that item's ORIGINAL campaign-1 A/B delta as the
expectation (tiled SDPA +7.2%, exact-topk +1.6%), then ONE measurement (prefill is boot-stable to
0.02%, so a single relaunch serial-path-fixed vs current is admissible). Ship only if it reproduces
within the pre-registered direction and magnitude. If all SHARED: close I12, audit IS the deliverable.

### I15 — kernel launches + command-buffer commits per decode STEP
Instruments: Metal capture / `MLX_GPU_TIME=1` + the R4 `EXO_DECODE_PROBE=1` instruments already in
the stack. **NOT** the `[MTP-PROF]` profiler (it `mx.eval`s every cycle and would corrupt the number).
Reference shape: 43 layers x (attention + router + experts + norms).
- **> 500 launches/step** -> launch overhead (~10-20us each on Metal) is 5-10ms of a ~65ms cycle;
  the dead/removed-06-18 `COMPILE_LAYER` idea would need REBUILDING as a real lever. **Scope it, do NOT build it.**
- **< 200** -> close I15.
- **200-500** -> band not defined by the brief; registered NOW as **INCONCLUSIVE**: report the raw
  count, do NOT declare a lever, do NOT close. Closing requires <200; declaring a lever requires >500.

### I9 — GPU frequency, one node, one 89K request
`sudo -n powermetrics --samplers gpu_power` during decode vs prefill vs idle.
- decode clock **>= 15% below** prefill clock -> a systems lever exists. **Report it; do NOT tune.**
- otherwise -> close I9. Low prior (bandwidth-bound decode is largely clock-insensitive); ride-along only.

### I11 step 1 — is 5-bit FAST? (MoE `gather_qmm`, M=4, deployed shapes, chained-graph method)
Proportional-scaling expectation vs the 6-bit baseline us/call: 5-bit 0.833x, 4-bit 0.667x.
- **FAST**: us/call <= 0.90x the 6-bit us/call.
- **MARGINAL**: 0.90x < us/call < 0.98x — a gain exists but is sub-proportional; report as such.
- **SLOW / generic path**: us/call >= 0.98x of 6-bit (or worse). If 5-bit is SLOW, **5-bit is dropped
  from the package** and the user's choice is the binary 6->4, exactly as the R7 review anticipated.
Report achieved GB/s AND absolute us/call. Serial-sync method is forbidden (R1 artifact); the R1 I3
chained-graph harness is the method of record.

### I11 step 2 — DISK GATE (hard, evaluated before any conversion)
Required: `min(free_node1, free_node2) >= size(new weight set) + 20 GiB safety margin`.
Observed headroom is **111 GiB** against ~150 GB/set, so:
- Retaining 6-bit AND 5-bit AND 4-bit simultaneously is **presumed INFEASIBLE**.
- The 6-bit set is **never** removed (it is the deployed config and the rollback path).
- **NO existing model directory may be deleted, moved, or pruned** to make room. Freeing disk is a
  USER decision, not an autonomous one. If the gate fails, step 2 delivers a **costed conversion
  plan** (bytes, wall-clock, exact target paths, what would have to be freed) instead of weights.
- If only ONE precision fits, prefer **4-bit** (larger ceiling; and it is the binary choice if 5-bit is SLOW).

### I11 step 3 — decode measurement (the ONLY step needing relaunches)
Ruler: round-6 ruler ONLY — `stats.generation_tps`, ~88K prompt, n=3 reps, **ranges never means**.
Arms in order: 6-bit (A) -> {4-bit and/or 5-bit-if-fast} -> 6-bit (B). **Both 6-bit boots mandatory**
(they bracket the 6+ t/s cold-boot decode drift that cost a 5-phase campaign on 2026-08-31).
- A precision's decode gain counts as REAL only if its n=3 range does **not overlap** the union of
  the two 6-bit ranges (A and B). Overlap -> "not separable from boot drift", not a win.
- Bit-equivalence is **NOT** the gate for I11 (a precision change is expected to diverge).
  The quality battery is the gate.

### I11 step 4 — quality battery (8 prompts, temp=0, side-by-side)
2 long-context (89K needle + a ~60K real-session-style prompt), 2 code, 2 reasoning
(`math_digit_sum` from the record), 2 tool-call (DSML). Scored: needle exact-match, tool-call parse
correctness, diff of 6-bit vs 4-bit outputs, P06-era top-1 logit-flip rate on real hidden states.
Perplexity is a **secondary number only** — insufficient alone for MoE, since expert routing changes.

## STOP / FALLBACK (registered now, not after seeing results)
Time-box 5h. Per the brief, **Task 1 + Task 2 + Task 3 steps 1-2 is a VALID round** if the
relaunches do not fit. If step 3 cannot complete, `I11-DECISION.md` ships with the measured-t/s row
marked NOT MEASURED and the user authorizes step 3 separately. A partial package is reported as
partial; no projected t/s number is ever presented as measured.

## HARD CONSTRAINTS
- Cluster ends **HEALTHY on the SHIPPED 6-bit config**, verified on real PIDs + a coherent temp=0
  completion. **I11 IS NOT SHIPPED.**
- NO pushes (local commits only; supervisor pushes). Never `git add -A`. Never `git stash`.
- Quantized weights go to the models dir, **NOT** the repo.
- Name only gates that EXIST — verify every gate script before citing it.
- `gh` requires `--repo adurham/exo`.
