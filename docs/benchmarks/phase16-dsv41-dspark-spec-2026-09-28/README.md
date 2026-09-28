# Phase 16 -- DSpark speculative decode on DSv4.1: 16.5 -> 22.7 tok/s (gamma=3)

Date: 2026-09-28. Both nodes, TP=2, standalone (not via exo). Production
stopped cleanly for each run and restored (fresh pids, env verified, real
completion OK).

## Bottom line

Speculative decode works end to end. Mean over 3 prompts, 96 tokens each:

| gamma | tok/s | accepted drafts/round |
|---:|---:|---:|
| plain | 16.5 | -- |
| 2 | 22.3 | 1.30 |
| **3** | **22.7** | 1.60 |
| 4 | 20.2 | 1.66 |

Per prompt at gamma=3: 21.6 / 25.4 / 20.9 tok/s. Short of 25 on average.

## Measured inputs

- Plain step 60.6 ms; draft round 13.1 ms (3 stages, width 3-5).
- Verify forward by rows: R1 59.5, R2 76.9, R3 89.6, R4 99.7, R5 114.2,
  R6 123.3 ms. Each extra row costs ~13 ms -- far more than the reduced-model
  projection assumed; this, not acceptance, is what caps the gain.
- Teacher-forced acceptance (first draft token = greedy's next token):
  74-92% per prompt; mean accepted prefix of 5 drafts 1.7-3.2.
- Draft head: 7.24 GB, loaded through the EXL3 builder, experts TP-split
  (split exact: cos 0.999997); peak 110.4 GB/rank.

## Bug found and fixed on the way

First run: acceptance 7-12% (speculation ran at 0.5 tok/s). Fixes, both to
match the DeepSeek reference: the draft head reads the INPUT of layers 37-39
(the port fed their OUTPUT), and the draft KV gets the fp8 activation
fake-quant. After the fix: 74-92%. Offline replay on the saved trace shows
input-vs-output taps moves acceptance (prefix 1.89 -> 2.47) but does not by
itself explain the 7% live figure; the exact live mechanism of that first run
was not isolated further -- the fixed code is reference-correct and measured.

## Correctness note

Chunk verify is not bit-identical to plain greedy (known body property,
same tradeoff production DSv4 makes at short context). Identical prefix vs
greedy: 14-77 of 97 tokens depending on prompt. Prompt-0 text at gamma=3 is
fluent and correct (sampled below); the other prompts' spec text was not
eyeballed yet -- do that before calling output quality equal.

> The sky appears blue because of a phenomenon called Rayleigh scattering.
> ... Blue and violet light have the shortest wavelengths, so they scatter the most

## Next levers (ranked by measured cost)

1. Verify row cost (~13 ms/row): the body at R>1 -- sparse attention and
   indexer at R rows, EXL3 R>1 paths. Halving it puts gamma=3 near 28 tok/s.
2. Draft round 13.1 ms: fused projections already on; compile the rest.
3. Adaptive gamma / confidence-truncated drafts (prompt 2 prefers gamma=2-3).

## Artifacts

`raw/p56-run1-lowaccept.log` (7% bug), `raw/p56-run2.log` (fixed),
`scripts/p56_spec.py`, `scripts/p58_accept_offline.py`, `scripts/p59_draft_debug.py`,
`scripts/p60_variants.py`.
