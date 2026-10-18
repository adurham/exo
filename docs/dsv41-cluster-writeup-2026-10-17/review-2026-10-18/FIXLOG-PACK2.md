# Pack review record, round 2 — 2026-10-18

Third review pass overall, second on the packs. Character differs from round 1: the raw
artifacts are gone, so most findings are provenance findings rather than number errors.

## Method

- Fable consulted for the plan/acceptance bar and to rule. 3 DSv4.1-Flash agents executed
  (deleg_3cf17139 tasks 1/2/4). Parent re-verified all 38 raw findings, deduped to 27.
- Agent line numbers were taken pre-fix; located by content instead.

## Fable's governing refinement

**Three states, not two.** UNVERIFIED was wrong for rotated artifacts — it means "nobody
checked this", but these *were* checked and the source then rotated. Now:
- SOURCED — artifact extant, re-checkable.
- RAW-LOST — quoted verbatim at capture; artifact since rotated; the quote is the sole
  surviving record. Date-stamped with the absence check.
- PROVENANCE-WEAK — basis never identified.
UNVERIFIED reserved for figures never checked at all.

## Urgent actions taken first

- Extracted the R04 corroborating lines from the 10-17 archive **before it rotates** →
  `preserved/loop-guard-env-from-20261017-archive.txt`.
- Copied the three surviving guard dumps into `preserved/` with sha256 recorded. These are the
  only re-checkable artifacts left and they lived on one machine's disk.

## Rulings and outcomes

27 items. Fable ruled: 12 ACCEPT, 14 ACCEPT-MODIFIED, 1 REJECT (R04's re-point — the later boot
differs in source tag and rank, so substituting would misattribute).

| ID | Verdict | Outcome |
|---|---|---|
| R01 | ACCEPT | FIXED — 166 -> 168 (misquote of a live source) |
| R02 | ACCEPT-MODIFIED | FIXED — past tense, RAW-LOST, quote marked sole survivor |
| R03 | ACCEPT-MODIFIED (downgraded from BLOCK) | FIXED — stated as a fact; server count authoritative; delta immaterial |
| R04 | REJECT (re-point) / ACCEPT-MODIFIED | FIXED — original kept RAW-LOST; corroborating line appended; extracted |
| R05-R08, R10, R11 | ACCEPT-MODIFIED | FIXED — RAW-LOST labels, date-stamped, figures stand as captured |
| R09 | ACCEPT | FIXED — PROVENANCE-WEAK (true provenance defect: no filename ever recorded) |
| R12-R16, R20, R21, R23, R24 | ACCEPT | FIXED — citation anchors appended |
| R17 | ACCEPT-MODIFIED | FIXED — dead pointers REPLACED with the master-history path |
| R18 | ACCEPT-MODIFIED | FIXED — retraction stated inline (:35 4.32x, :91 retracted) |
| R19 | ACCEPT-MODIFIED | FIXED — status transition stated (:4444 NOT-YET-FIXED, :10307/10751 detail) |
| R22 | ACCEPT | FIXED — 2.92 -> 2.95 both cells; no 34.2 basis existed |
| R25 | ACCEPT-MODIFIED | FIXED — inline at all three cited lines + RAW STATUS row; discriminating pattern given |
| R26 | ACCEPT | FIXED — dump cited with host, path, task_id, sha256 |
| R27 | ACCEPT (refutation) | NO-CHANGE — citation already carried :678 |

REFUTED: 1. AMBIGUOUS: 0.

## Still open

- The packs' raw artifacts are gone; RAW-LOST figures cannot be re-checked against raw at all.
- §11 has still never been executed end to end by a third party.
- Retention is ~5 archives; further rotation will take the remaining corroborating lines.
