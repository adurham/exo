# Pack review record — 2026-10-18

Second review pass, on the EVIDENCE PACKS (`evidence.md`, `evidence-live.md`) rather than the
writeup. The packs are the authority the writeup rests on, so a defect here propagates.

## Method

- Fable consulted for the plan/acceptance bar and to rule on findings. 2 DSv4.1-Flash subagents
  executed (one failed on reasoning-token exhaustion, was salvaged and re-split into two).
- Parent re-verified every finding against the pack source before listing.
- Four items needed pre-checks before applying; all four resolved against source:
  Q01 counter is positional (`engine.py:1191-1192`), Q02 1024-block alignment is real
  (`session.py:158`), Q09 rows≡tokens (`PERFORMANCE_HISTORY.md:10398`), Q13 loop rate is 372.

## Fable's governing policy

**Never overwrite a sourced figure.** Replacing a sourced value with a recomputed one severs
the chain back to the source doc. Corrections are APPEND-WITH-LABEL: the original keeps its
basis, the recomputed value is appended with its own basis, and an unidentifiable basis is
tagged PROVENANCE-WEAK. This is now written into both packs as a standing edit policy.

## Rulings and outcomes

17 items. Fable ruled: 9 ACCEPT, 7 ACCEPT-MODIFIED, 1 REJECT (Q02's replacement — the two
figures are distinct real quantities, not a typo).

| ID | Verdict | Outcome |
|---|---|---|
| Q01 | ACCEPT-MODIFIED | FIXED — positional reading confirmed; convention stated; extrapolation marked a lower bound |
| Q02 | REJECT (replacement) / ACCEPT-MODIFIED | FIXED — both quantities kept and named; 1024-block alignment explained |
| Q03 | ACCEPT-MODIFIED | FIXED — +36.71% kept and labelled as the paired statistic; +34.9% ratio added |
| Q04 | ACCEPT | FIXED — one unit rule stated |
| Q05 | ACCEPT-MODIFIED | FIXED — relabelled as a 0.1 GB discrepancy, not rounding; mechanism marked unrecoverable |
| Q06 | ACCEPT | FIXED — CONFLICT prepended |
| Q07 | ACCEPT | FIXED — leading "none" -> CONFLICT |
| Q08 | ACCEPT | FIXED — depth cell matched to row 3 |
| Q09 | ACCEPT-MODIFIED | FIXED — CONFLICT kept; rows≡tokens established and noted |
| Q10 | ACCEPT-MODIFIED | FIXED — 30.4 kept (it IS the n-1 figure); convention annotated |
| Q11 | ACCEPT-MODIFIED | FIXED — percentage kept, tagged PROVENANCE-WEAK, ratio added |
| Q12 | ACCEPT-MODIFIED | FIXED — same treatment |
| Q13 | ACCEPT-MODIFIED | FIXED — 55.8 s -> ~65 s (255.06 - 70,656/372) |
| Q14 | ACCEPT | FIXED — CONFLICT/CONVENTION -> CONVENTION-AMBIGUOUS |
| Q15 | ACCEPT | FIXED — live storageSize added as a third value |
| Q16 | ACCEPT | FIXED — CONFLICT token carried |
| Q17 | ACCEPT | FIXED — PROVENANCE-WEAK/DERIVED tagged |

REFUTED: 0. AMBIGUOUS: 0.

## Structural additions (Fable's request)

- RAW STATUS table in both packs: per-artifact EXISTS / ROTATED-OUT / GONE, with the pack
  sha256 and the dated edit policy.
- SOLE-SURVIVOR tags on figures whose only surviving source is the pack.
- A "still checkable WITHOUT raw" note (source on disk, derived arithmetic).

## Writeup inheritance

Four writeup lines quote corrected pack values and were updated to follow: §9.4 (Q02 both
figures, Q01 lower-bound caveat) and §10.8 (Q01 caveat).

## Still open

- The packs' raw artifacts are gone; figures tagged SOLE-SURVIVOR cannot be re-checked against
  raw at all.
- §11 has still never been executed end to end by a third party.
