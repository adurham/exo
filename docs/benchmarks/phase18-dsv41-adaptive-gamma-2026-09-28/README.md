# Phase 18 -- one-sync spec loop + adaptive draft length: 22.9 tok/s mean

Date: 2026-09-28. Both nodes, TP=2, standalone. 5 prompts x 192 tokens
(longer and wider than phase 17). Production stopped cleanly and restored.

| loop | mean tok/s |
|---|---:|
| plain | 15.4 (prompt 0 plain run was an outlier at 9.25; others 16.9-17.0) |
| old loop, gamma 3 | 21.6 |
| one host sync per round, gamma 3 | 21.9 |
| one sync + adaptive gamma | **22.9** |

Per prompt, adaptive: 21.4 / 29.3 / 20.7 / 19.1 / 24.3.

- The Fable consult estimated ~10 ms/round of host-sync overhead. Measured:
  removing the second sync gained 0.3 tok/s -- the estimate did not hold.
  Round time stays ~110 ms at gamma 3 (draft 11 + verify 88 + ~11 other).
- Adaptive gamma (per-round, from observed per-position acceptance, costed
  with the measured verify times) mostly picks gamma 2 on low-acceptance
  prompts and 3 on the code prompt: +1.1 tok/s mean.
- The phase-17 mean (23.5) was on 3 prompts x 96 tokens; this wider set
  (incl. French Revolution summary, 17.8 at g3) is the more honest number.
- Output: all five adaptive texts read, fluent and on-task.

Where the ~11 ms/round unaccounted goes is still open (not the sync).
