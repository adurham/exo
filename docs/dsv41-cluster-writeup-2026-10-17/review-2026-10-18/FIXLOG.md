# Review record — 2026-10-18

Independent review pass on `WRITEUP.md`, run after the 2026-10-17 rounds closed.

## Method

- Reference model (Fable) consulted for the review PLAN and acceptance bar, then again to
  RULE on the consolidated findings. The line-by-line work was executed by 4 cheap
  DSv4.1-Flash subagents on a disjoint split (causal attribution / reverse sweep /
  head-body+status drift+scrub / §11 executability+§9 baselines).
- Every finding was re-verified by the parent against its cited source before being listed.
  Nothing was applied on a subagent's say-so.
- Acceptance bar: line anchor + verbatim source cite + proposed replacement + class tag;
  world knowledge is not a source (unsourced numbers become NOTE, not rewrites); deliberate
  choices excluded (§10 open by design, §12.6 structure, neutral placeholders, untruncated
  §11 ids).

## Rulings and outcomes

21 consolidated items. Fable ruled: 15 ACCEPT, 6 ACCEPT-MODIFIED, 1 REJECT.
All 20 accepted items applied. Verdict log below.

| ID | Verdict | Outcome | Source cite |
|---|---|---|---|
| R01 | ACCEPT | FIXED — `gu_svh`/`dn_suh` -> `gu_suh`/`dn_svh` (pairs were swapped) | mlx-lm `6391efc:loader.py:11,203,207` |
| R02 | ACCEPT-MODIFIED | FIXED — reworded to the diff's actual mechanism; anchor moved to the diff body | mlx `b4871e8` diff `@@ -782,10 +782,43 @@` |
| R03 | ACCEPT-MODIFIED | FIXED — scoped to the all_reduce family; confirmed the sentence covers both collectives | WRITEUP.md:682; `mesh_impl.h:1935` |
| R04 | ACCEPT | FIXED — `mkdir -p prompts` added | `git ls-tree b88d65dd` -> 0 |
| R05 | ACCEPT-MODIFIED | FIXED — kept "over the tailnet", named the hardcoded constants | `start_cluster.sh:1025-1028,1113` |
| R06 | ACCEPT | FIXED — open-vs-shipped reconciled | WRITEUP.md:1389 |
| R07 | ACCEPT | FIXED — data path -> deploy path | WRITEUP.md:329 |
| R08 | ACCEPT-MODIFIED | FIXED — "Every" kept; D4.4 confirmed to render the SAME knob, not a different pin | WRITEUP.md:583 |
| R09 | ACCEPT | FIXED — stale basis row annotated | WRITEUP.md:1386 |
| R10 | ACCEPT | FIXED — "Two caveats" -> "Three caveats" | WRITEUP.md:942-945 |
| R11 | ACCEPT-MODIFIED | FIXED — 30.4 is a first-token-excluded convention, not bad arithmetic; convention stated in §9.1 | evidence-live.md:129 |
| R12 | ACCEPT-MODIFIED | FIXED — 299008 vs 299946 vs ~98K all stated; origin marked unresolved | evidence-live.md:704; WRITEUP.md:943 |
| R13 | ACCEPT | FIXED — elision noted | evidence-live.md:296-303 |
| R14 | ACCEPT | FIXED — checkout the pinned SHA; detach noted | `git rev-parse main` |
| R15 | ACCEPT | FIXED — script runs ON the node | `exo_graceful_shutdown.sh:5-6` |
| R16 | ACCEPT | FIXED — line 768-773 -> 807, revision pinned to `b1778183` | mlx `b1778183:device.cpp:807` |
| R17 | ACCEPT | FIXED — scope to measured depths | evidence-live.md:687,864 |
| R18 | **REJECT** | NO-CHANGE-NEEDED — a writeup more conservative than its own pack is not a defect; the pack's "memory-side" is not established as measurement-backed | evidence-live.md:211 |
| R19 | ACCEPT-MODIFIED | FIXED — ceiling bounds the default only; request value passes through uncapped | `reasoning_budget.py:91-92` |
| R20 | ACCEPT-MODIFIED | FIXED — +44% stated; comparability caveat added | evidence.md:291 |
| R21 | ACCEPT-MODIFIED | FIXED — explicit 100K/300K range; shallower rows stated | evidence-live.md rows |

REFUTED: 0. AMBIGUOUS: 0.

## Fifth-pass items (Fable's list of what line-anchored agents structurally miss)

1. **Fix interaction** — reread the §12.6 truncate/escape region and the L941-958 caveat block
   after applying. Done; one flow break introduced by R17 was caught and repaired.
2. **Pack as ground truth** — everything verified "against the pack" is verified only to the
   pack; nobody checked the pack against raw logs. Still open (see below).
3. **Revision drift in cites** — R16's line number pinned to `b1778183`. `mlx-lm 6391efc`
   confirmed to be the loader's introducing commit (2026-09-28), not a stale HEAD reference.
4. **Universal/negative claims** — grepped and audited; all sampled hits are sourced
   (e.g. L354 "every deploy" -> `docs/incidents/pkill-self-match-forces-sigkill-2026-09-10.md:41-59`).
5. **199-vs-200 tok/s convention** — stated once in §9.1 so all throughput figures are covered.
6. **Deliberate exclusions** — placeholders grepped (consistent, and explained in the header);
   the writeup's own hardware/spec assertions checked: 552B/16B active is sourced to
   `PERFORMANCE_HISTORY.md:12089` (model-card authoritative) and `deepseek-v41-exl3-plan.md:65`;
   `max_qp=3` is read live and already marked observed/undocumented.

## Still open after this pass

- **Pack-vs-raw-log check** (fifth-pass item 2): the evidence packs are treated as authority
  throughout. A pass that re-derives pack figures from the raw captures/logs has not been run.
- **§11 has never actually been executed** end to end by a third party; the fixes make it
  runnable on paper only.

## Files

- `s1-causal.md`, `s2-reverse.md`, `s3-drift.md`, `s4-repro.md` — raw agent findings.
- `CONSOLIDATED.md` — the pre-verified, item-ID'd list the ruling was made against.
- `SALVAGE-fable-interrupted.md` — raw output from an earlier interrupted Fable batch.
