# ROUND 3 — PRE-REGISTRATION (written BEFORE any measurement was taken)

PM: delegation subagent. Timestamp: written prior to dispatching A1/A3/B/C.
Purpose: fix every gate and every verdict mapping in advance so no criterion can be
chosen after seeing a result. This file is append-only; nothing below is edited after
results land.

---

## A1 — API verification gate (the ship/no-ship gate for the pad-strip)

Experiment: three requests to the exo server, identical except for a prior ASSISTANT
message's `reasoning_content` field:
  (a) key ABSENT
  (b) key present as ""
  (c) key present as " "   <- current client behavior (the pad)

Observation required: the SERVER-SIDE rendered prompt / tokenization for each variant.
Reading the client or template source alone does NOT satisfy this gate.

### GATE CRITERIA (all four must hold to ship the pad-strip)
- **G1 — no error.** All three variants return HTTP 200 with a well-formed completion.
- **G2 — absent is clean.** Variant (a)'s rendered prompt contains no artifact
  attributable to the missing field: no literal `None`/`null`/`NoneType`, no dangling or
  empty reasoning delimiters, no duplicated/dropped turn markers.
- **G3 — divergence is localized.** (a) vs (c) differ ONLY in the reasoning slot: the pad's
  own token(s) and nothing else. No re-segmentation of surrounding text, no differing
  turn-structure tokens, no cascade. (This is fable's named "silent mishandling" check —
  three wildly different sequences would mean the template folds the field into the prompt
  in a load-bearing way.)
- **G4 — absent is not worse.** Variant (a) does not lengthen the prompt or add tokens
  relative to (c).

GATE PASS = G1 AND G2 AND G3 AND G4 -> ship the provider-scoped pad-strip.
GATE FAIL = any of G1-G4 fails -> do NOT ship. Report the mechanism, Fix B stays NO-GO.

Note: (a) vs (c) SHOULD differ by the pad token — that is the whole point and is not a
failure. G3 constrains the SHAPE of the difference, not its existence.

---

## A3 — LCP re-probe

### Bands: IDENTICAL to round 2, quoted verbatim, not renegotiated
- **PASS:** median LCP_coverage >= 90% AND p10 >= 70%
- **FAIL:** median < 60% OR p10 < 20%
- **INDETERMINATE** (60-90% median, or p10 20-70%): scored as PRACTICAL-FAIL

### Methodology lock
Same 54 turn pairs, same `probe_lcp.py`, same tokenizer, same variant A/B bounding.
The ONLY permitted change is the treatment of the `reasoning_content` pad in the
re-fed assistant message. Any other methodology change voids the comparison.

### OPTIMISM HANDLING — CHOICE MADE IN ADVANCE: **option (b)**
Fable offered (a) persist raw decode ids so the probe stops being optimistic, or
(b) explicitly downgrade the claim and file the proxy fix as a shipping prerequisite.

**We take (b), and (a) is not available for this dataset.** The 54 pairs under test were
emitted in a session that has already ended; raw decode token ids for those turns do not
exist in `state.db`, `requests.jsonl`, or the exo event-log archive, and cannot be created
retroactively. Fable also explicitly preferred the same-data re-probe over waiting for a
new session, because it tests the MECHANISM directly and n=9 pad-related pairs is too thin
for a statistical claim. Same-data therefore forces (b).

**Consequence, registered now:** a passing re-probe CANNOT produce a GO this round. The
ceiling is CONDITIONAL-GO. The proxy fix is a hard prerequisite, not a nice-to-have.

### VERDICT MAPPING (fixed in advance)
| Re-probe result | Fix B verdict |
|---|---|
| FAIL bands | **NO-GO** — robust (a FAIL under an optimistic proxy is a real FAIL) |
| INDETERMINATE | **NO-GO** for round 3 (PRACTICAL-FAIL per the round-2 band definition) |
| PASS bands | **CONDITIONAL-GO** — never GO. Condition below. |

### The pre-registered CONDITION attached to CONDITIONAL-GO
Fix B may ship only after BOTH:
- **C1 (de-optimize the proxy):** raw decode token ids are persisted for >= 1 fresh
  session, and the probe is re-run against those real ids — not reconstructions — and
  passes these same bands. This retires the optimism, it does not re-litigate it.
- **C2 (fleet value):** the session-length distribution (N >= 20-30 sessions) establishes
  that bucket B's share is materially above the ~2-4% seen in short sessions. Round 2
  measured B-share to be strongly length-dependent (2.09% at 2 turns -> 21.91% at 55);
  21.9% is a long-session figure, not a fleet figure. A mechanism that works is not the
  same as a mechanism worth shipping.

C2 is a value gate, not a correctness gate, and is listed because CONDITIONAL-GO must
name every condition up front rather than discovering them later.

---

## B — SDPA 2-length timing (script prepared this round, NOT run)

Pre-registered decision rule, set before any timing exists:
- Let R(L) = per-call absolute SDPA time at per-rank query rows doubled, measured at
  context length L, from >= 5 calls at each length, each call DIRECTLY wrapped in
  perf_counter with an explicit GPU sync before the end timestamp.
- **Real multiplicative constant (matters at 250K):** the ~4.06x per-call ratio holds at
  BOTH 12K and 64K (both within [3.0, 5.0], and |R(64K) - R(12K)| / R(12K) <= 25%).
- **Fixed-overhead artifact (does not matter):** R shrinks materially with length —
  R(64K) <= 2.2, or R(64K) < 0.6 * R(12K).
- **INDETERMINATE:** anything else. Report as indeterminate; do not narrate a story.
- Mandatory reductio before quoting any per-call number: calls x per-call <= measured wall.

---

## C — Decode instrumentation (design only this round)

No measurement, therefore no numeric gate. The deliverable is judged on whether all four
of fable's named pitfalls are structurally handled in the code sketch, not merely
mentioned in prose: lazy-eval sync, cross-rank clock skew, first-call warmup, and
disjoint timer placement.

---

## Standing constraints for round 3
1. Root-cause only. No mitigations.
2. Zero cluster cost except A1's short verification calls. B is PREP-ONLY.
3. No commits to `~/repos/exo` (supervisor owns those). The hermes-agent pad-strip commit
   is authorized for the PM **only after the A1 gate passes**.
4. No boot experiments. No P16.
5. Do not re-propose NOT-FUNDED items.


---

## AMENDMENT (appended AFTER results were seen — labeled as such, not backdated)

Everything above this line was written before any measurement. The following was added
after the round-3 results landed, on reviewer advice. It only makes conditions STRICTER;
no gate, band, or verdict mapping above was altered.

**C1 gains a composition requirement.** As originally written, C1 could be satisfied by a
fresh session that happened to be short and tool-light — which would never exercise the
tool-call region, the exact place the current proxy matches *by construction* and is
therefore structurally blind. C1 now additionally requires that the fresh session contain
multi-invoke tool-call pairs, so the de-optimized re-probe actually tests the blind spot
rather than routing around it.

**New round-4 item (not a gate):** the shipped fix is client-side, but the underlying
defect class is server-side and survives it. Any other client that feeds a
leading-divergent byte into the re-fed region silently zeroes the cache, and
`cached_tokens = 0` is the only symptom. Filed as a monitoring/normalization item.
