G1 — gen-prompts + validation harness. Repo head b88d65dd330478e8e1b8a565848af3c863be7a06 (pinned; worktree uncommitted, no commits made).
READ evidence-live.md:122,207,264-306,318-340,384-414,736,842-922 — bodies are RAW-LOST; only counts, request fields, and the guard_banana body survive verbatim.
FIND body shape = ONE user message, no system prompt (evidence-live:322-333 quoted body; all completions are single-paragraph answers).
FIND counts on the obs lines INCLUDE chat framing: exo's DSV4 encoder + local tokenizer reproduce guard_banana at exactly 26 prompt tokens (content 22), so target the rendered prompt, not the content.
MEASURE corpus at pinned SHA, docs/**.md + top-level *.md sorted: 314 files, 5,925,808 chars, 1,755,007 tokens.
DECIDE exclude docs/dsv41-cluster-writeup-2026-10-17/ — added by the same merge the repro pins, and it quotes the answers (self-reference).
DECIDE import exo's own DSv4 encoder from the pin (blob 7b785a2d93df) instead of re-implementing chat framing; tokenizer sha256 8f9f37ca… matches the writeup's id.
GOTCHA encoder rejects message text containing the DSv4 image placeholder token; docs/DSV4_VISION_PORT_PLAN_PHASE34.md contains 1 → substituted ASCII marker, counted in manifest.
GOTCHA BPE boundary merges make some targets unreachable by prefix slicing (160,004 skipped: n+65 vs n+66); fixed with 0-8 newlines between corpus and question, k=0 preferred.
WROTE scripts/gen-prompts.py — 11 bodies, all prompt-token targets hit exactly; 6 s cold.
WROTE scripts/validate-prompts.py — checks (i) exact bytes, (ii) tokens, (iii) quoted substrings, (iv) bytes ±5%, (v) schema, (vi) determinism, plus a prefix-chain assertion.
RESULT (i) FAIL for prompts_160k (-0.89%) and prompts_1m (-1.38%) — expected; (i) PASS only for guard_banana, which is a transcription of the full quote (318 B byte-identical). (ii)-(vi) PASS on all 11.
RESULT prefix chain PASS: 1k → 10k → 50k → 100k → 160k → 300k → 1M are literal prefixes, so the 8,192/49,152/299,008 cache geometry is reachable on a re-run.
WROTE docs/dsv41-cluster-writeup-2026-10-17/preserved/RECONSTRUCTION.md with the honesty ceiling stated: equivalent-class input, never "the original"; RAW-LOST figures stay RAW-LOST.
VERIFY rm -rf prompts && regenerate → identical sha256 for all 11 files; `gen-prompts.py --check` says OK on all 11; guard_banana 318 B is byte-identical to the quoted block at evidence-live.md:321-333.
NOTE §11.5 still says "the capture session's prompt files and their generator are not in this repo" — now stale; parent should point it at scripts/gen-prompts.py (no edit made here to avoid colliding with other packs).
END G1
