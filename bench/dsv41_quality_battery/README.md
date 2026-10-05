# DSv4.1 quality battery + soak tooling (repo copy)

Persisted from `~/.hermes/cache/scratch/quality_battery/` + `laddercheck/` +
`1m_soak3.sh` so the tooling survives scratch pruning.

## What each file is

- **battery.py** — the live quality battery (the ship gate for precision changes
  on the DSv4.1 path). Modes: `plan|build|eval|all|selftest|aggregate` plus
  per-phase `needles|prose|tools|park`. Deep-context build with stratified
  planted needles, multiturn probes (LCP lands ABOVE the prompt-end checkpoint so
  they are ladder refeeds, ~seconds), free-prose with the lm_head
  glued-fragment detectors, tool-call probes, parked-restore probe.
  Probe-repair history (2026-10-05): reasoning-channel fallback for needle/park
  checks; city-string containment in tool arg compare; `dsml_leak_in_content`
  classification; prose budget raised to >=4096 with `reasoning_effort="low"`;
  noisy `same_script_glue` heuristic demoted to ADVISORY (cannot tell
  "beginner"=begin+ner from a real glued fragment without a dictionary);
  `REASONING_ONLY` verdict (content empty, answer in reasoning — a known
  temp-0 checkpoint behavior; diffable across arms, never a false FAIL).
- **compare.py** — two-phase diff + PASS/REVIEW/FAIL ship verdict (A=baseline,
  B=candidate). Tracks REASONING_ONLY/ERROR rates as signals; text change on
  free-form prose is a NOTE (temp-0 divergence is expected), not a gate.
- **reverdict.py** — re-verdict a label's stored prose bodies offline with the
  current detector code (no cluster needed) — used when the detector changed
  after a phase was captured.
- **ladder_exactness.py** — the reuse-ladder exactness gate: T1 build, T2
  ladder refeed, T3 exact repeat; gates A2==A3 (same build) and A2==A2_prev
  (cross-build, `--a2-prev`).
- **1m_soak3.sh** — the r1M re-proof soak (r160 cold build, then TRUE delta
  rungs r500/r750/r1m + over-cap refusal), on the ladder build.
- **runbook.md** — the two-phase relaunch + A/B procedure (env flip via the
  LAUNCHED process env; DSV41_INDEXER_ROW_BF16 is read at import).

## Notes

- The battery drives the CLUSTER API (`http://macstudio-m4-1.tail19c543.ts.net:52415`);
  it does not touch the repos.
- `_mock_integration_test.py` is the offline pipeline test (mock server).
- Free-form prose at temp=0 always diverges a hair between arms (0/5
  byte-identical in the originating incident); the detectors plus the
  REASONING_ONLY rate are the real gates.
