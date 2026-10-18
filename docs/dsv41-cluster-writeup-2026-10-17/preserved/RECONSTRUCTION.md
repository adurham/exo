# Reconstruction of the lost §11 prompt bodies

**Status: reconstructed, NOT recovered.** The originals were built in a scratch directory (`~/.hermes/cache/scratch/writeup/prompts/`) and were never committed; they are confirmed absent from the controller and node0 (§"RAW-LOST" notes throughout `evidence-live.md`). This directory now holds generated *equivalent-class* inputs so that §11 is runnable again.

The honesty rule this document is written under:

> A reconstruction does not retroactively source a RAW-LOST figure. Every byte size and token count in `evidence-live.md` and `WRITEUP.md` remains RAW-LOST. The generated bodies reproduce the *logged prompt token counts* and the *quoted structure*; they are not the bytes that produced the measured latencies, and no measured number is re-derived from them.

The ceiling of this work is therefore stated once, plainly: **equivalent-class input matching the logged token count and the quoted content — never "the original"**.

## What was reconstructed, and from what

| input | source of truth | reconstructed? |
| --- | --- | --- |
| message structure (one `user` message, no system prompt) | the quoted body shape in `evidence-live.md` §9.1 / §11.5 | yes, exact |
| request fields (`model`, `max_tokens`, `temperature`, `stream`, `enable_thinking`, `reasoning_budget_tokens`) | quoted bodies + §11.7 | yes, exact |
| corpus | "the repo's own `docs/**.md` plus the top-level handoff docs, concatenated in sorted order, truncated at the target depth" | approximated: `git ls-tree` of the pinned commit `b88d65dd3304`, 314 files, 5,925,808 chars, 1,755,007 tokens |
| tokenizer | `~/.exo/models/deepseek-ai--DeepSeek-V4-Flash/tokenizer.json`, sha256 `8f9f37ca…` (the id the writeup quotes) | exact |
| chat framing | exo's own DSv4 encoder at the pinned commit (`deepseek_v4_encoding.py`, blob 7b785a2d93df) | exact |
| prompt token count | the surviving `[DSV41] obs` lines | exact by construction (filled to target) |
| the question text | **not preserved anywhere** — only its shape is implied by the surviving completions (each starts `BOTTLENECK:` and names the document that quantifies the bottleneck) | reconstructed, not exact |
| byte size | quoted for two bodies, and fully quoted for a third (`guard_banana`) | **not reproducible** for the reconstructed bodies — check (i) below fails for them by design |

Two deliberate deviations from the original construction, both recorded because they change bytes:

1. The writeup's own directory (`docs/dsv41-cluster-writeup-2026-10-17/`) is excluded from the corpus. It was added by the same merge commit the repro pins (`b88d65dd3304`), i.e. *after* the capture, and it quotes the model's answers — including it would make the prompt self-referential.
2. 1 occurrence of the DSv4 image placeholder token in `docs/DSV4_VISION_PORT_PLAN_PHASE34.md` is replaced with the ASCII marker `[image-placeholder-token]`, because exo's encoder rejects message text containing that special token.

Each corpus body is `corpus_prefix + "\n"*k + question`, where `k ∈ 0..8` exists only because BPE boundary merges make some token counts unreachable by prefix slicing alone; `k=0` is used whenever it suffices. The manifest records the exact `k` per body.

## Validation

Checks are ordered strongest-first. (i) is the strongest *claim*, and it fails for every body whose bytes were reconstructed — which is the honest result. It passes for exactly one file, `guard_banana.json`, because that body is quoted in full in `evidence-live.md`: that pass is transcription, not recovery. (ii)–(vi) are the claims that are reachable for a reconstruction, and they pass.

| # | check | what a failure would mean |
| --- | --- | --- |
| (i) | exact byte size vs the quoted figure | the original bytes were recovered — **not achievable**, and passing would be suspicious |
| (ii) | prompt token count == the count on the surviving `obs` line | the body would not exercise the same context depth |
| (iii) | every verbatim substring the packs quote is present | the body would not contain the content the writeup claims it contained |
| (iv) | byte size within ±5% where quoted | the corpus family/truncation would differ materially from the original |
| (v) | JSON schema identical to the quoted request shape | §11's curl commands would not reproduce the same API surface |
| (vi) | regeneration is byte-identical | the generator would not be deterministic, so the directory could not be trusted |

### Per-file results

| file | (i) exact bytes | (ii) prompt tokens | (iii) quoted substrings | (iv) bytes ±5% | (v) schema | (vi) determinism |
| --- | --- | --- | --- | --- | --- | --- |
| `prompts_160k.json` | FAIL | PASS | PASS | PASS | PASS | PASS |
| `prompts_1m.json` | FAIL | PASS | PASS | PASS | PASS | PASS |
| `depth_d1k.json` | — | PASS | PASS | — | PASS | PASS |
| `depth_d10k.json` | — | PASS | — | — | PASS | PASS |
| `depth_d50k.json` | — | PASS | — | — | PASS | PASS |
| `depth_d100k.json` | — | PASS | PASS | — | PASS | PASS |
| `depth_d300k.json` | — | PASS | PASS | — | PASS | PASS |
| `guard_banana.json` | PASS | PASS | PASS | PASS | PASS | PASS |
| `guard_loop.json` | — | PASS | PASS | — | PASS | PASS |
| `limiter_default.json` | — | PASS | PASS | — | PASS | PASS |
| `limiter_96.json` | — | PASS | PASS | — | PASS | PASS |

`—` = not applicable: the writeup quotes no byte size for that body, so checks (i) and (iv) cannot be stated for it at all; for `depth_d10k` and `depth_d50k` no completion is quoted either, so there is no quoted content for check (iii) to match.

Structural check (not part of the five): the corpus bodies are literal prefixes of one another in target order (1003 → 10004 → 50004 → 100003 → 160004 → 300003 → 1000003 tokens). Result: **PASS**. That is what makes the recorded cache geometry plausible on a re-run: the deep rows can hit the shallow rows' prefix blocks, which is how the original saw `cached_tokens` 8,192 at 50 K, 49,152 at 100 K and 299,008 at 1 M.

### Numbers behind the table

| file | target prompt tokens | measured prompt tokens | content tokens | bytes | quoted bytes | Δ bytes | corpus prefix tokens | tail `\n` | sha256[:16] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `prompts_160k.json` | 160,004 | 160,004 | 160,000 | 554,038 | 558,995 | -4,957 (-0.89%) | 159,939 | 1 | `8a5aaf5b48766850` |
| `prompts_1m.json` | 1,000,003 | 1,000,003 | 999,999 | 3,344,717 | 3,391,475 | -46,758 (-1.38%) | 999,938 | 0 | `532b1818f5cb5684` |
| `depth_d1k.json` | 1,003 | 1,003 | 999 | 4,384 | — | — | 938 | 0 | `bbd0b0dc4f678c9f` |
| `depth_d10k.json` | 10,004 | 10,004 | 10,000 | 36,005 | — | — | 9,939 | 0 | `2ed3b069576588ad` |
| `depth_d50k.json` | 50,004 | 50,004 | 50,000 | 176,525 | — | — | 49,939 | 3 | `6ab13679ed549e94` |
| `depth_d100k.json` | 100,003 | 100,003 | 99,999 | 350,291 | — | — | 99,938 | 2 | `bcdc5f9aa8c1c172` |
| `depth_d300k.json` | 300,003 | 300,003 | 299,999 | 1,031,711 | — | — | 299,938 | 0 | `6bbf3893db69d4ed` |
| `guard_banana.json` | 26 | 26 | 22 | 318 | 318 | +0 (+0.00%) | 0 | 0 | `a2b4aa327c4479e3` |
| `guard_loop.json` | 48 | 48 | 44 | 417 | — | — | 0 | 0 | `ca98e723fe4b525b` |
| `limiter_default.json` | 52 | 52 | 48 | 397 | — | — | 0 | 0 | `266229f63655d903` |
| `limiter_96.json` | 52 | 52 | 48 | 429 | — | — | 0 | 0 | `07d2bbd12298204c` |

### Check (i) in words

- `prompts_160k.json`: quoted **558,995 B**, generated **554,038 B** → **-4,957 B** (-0.89%). FAIL, as expected: the original's exact corpus snapshot, selection boundary and byte-level formatting are not recoverable.
- `prompts_1m.json`: quoted **3,391,475 B**, generated **3,344,717 B** → **-46,758 B** (-1.38%). FAIL, as expected: the original's exact corpus snapshot, selection boundary and byte-level formatting are not recoverable.
- `guard_banana.json`: quoted **318 B**, generated **318 B** → **+0 B** (+0.00%). PASS — transcription of a body the writeup quotes in full, not evidence about any lost body.

The residual gap is a property of *which* docs land inside the first N tokens of *this* snapshot, not of a wrong method: the writeup's own figure (5.38 M chars) does not match any single obvious filter of the pinned tree, so the exact corpus membership at capture time is itself part of what was lost.

### Check (iii) detail

Only `guard_banana.json` has a body quoted verbatim in the writeup, and that sentence is reproduced character-for-character. For the corpus bodies there is no quoted body text to match; the check instead requires the document paths that the surviving completions cite — i.e. paths that must have been inside the context for those answers to be grounded.

| file | required substrings | result |
| --- | --- | --- |
| `prompts_160k.json` | `docs/moe-all-sum-dominant-cost-2026-08-19.md`, `docs/jaccl-internal-timing-allsum-transport-fast-2026-08-21.md`, `docs/allsum-sync-span-artifact-arithmetic-check-2026-08-22.md` | 3/3 present |
| `prompts_1m.json` | `docs/moe-all-sum-dominant-cost-2026-08-19.md` | 1/1 present |
| `depth_d1k.json` | `AGENTS.md`, `npm run build` | 2/2 present |
| `depth_d10k.json` | — | n/a — no quoted completion for this row |
| `depth_d50k.json` | — | n/a — no quoted completion for this row |
| `depth_d100k.json` | `docs/moe-all-sum-dominant-cost-2026-08-19.md`, `PREFILL_CLIFF_HANDOFF.md` | 2/2 present |
| `depth_d300k.json` | `docs/moe-all-sum-dominant-cost-2026-08-19.md`, `docs/moe-all-sum-skew-vs-comms-2026-08-19.md` | 2/2 present |
| `guard_banana.json` | `Output the word banana exactly 40 times, separated by single spaces, and nothing else. Do not explain.` | 1/1 present |
| `guard_loop.json` | `[SECTION` | 1/1 present |
| `limiter_default.json` | `A bat and a ball cost $1.10 in total.` | 1/1 present |
| `limiter_96.json` | `A bat and a ball cost $1.10 in total.` | 1/1 present |

All corpus bodies also contain the model id and the single `user` message with `role`/`content` only, which check (v) asserts structurally.

## What this does and does not license

**Licensed.** §11 can be run again: the curl commands in the writeup take these files unchanged, at the pinned commit, with the same request fields, and reach the same context depth to the token. The token-count column of the depth table (1003 / 10004 / 50004 / 100003 / 300003, and 160,004 / 1,000,003) is reproduced exactly, so re-running exercises the same shapes of prefill and the same prefix-cache geometry (the depth bodies are literal prefixes of one another, and of the 1M body, which is what produced the 8,192 / 49,152 / 299,008 cached-token hits).

**Not licensed.** Anything of the form "the original prompt", "the captured bytes", or "re-derived from the recovered body". In particular:

- the quoted byte sizes (558,995 B / 3,391,475 B) stay RAW-LOST and are matched only to within a few percent;
- the local tokenizer counts quoted in the writeup (159,999 for the 160 K body, 999,999 for the 1 M body) are the *content* counts of the originals; the generated content counts land at 160,000 and 999,999 — within one token — and the server-reported totals are hit exactly by adding exo's 4-token chat framing on top. Matching a count is not recovering a body;
- the question wording, and therefore any claim about how the model was asked, is a reconstruction from the shape of its answers;
- no measured latency, throughput or cache figure is recomputed, re-validated or "confirmed" by this directory. Those numbers rest on the quoted server lines alone, exactly as before.

## Reproducing this document

```bash
.venv/bin/python scripts/gen-prompts.py            # writes ./prompts
.venv/bin/python scripts/validate-prompts.py       # validates + rewrites this file
```

`prompts/GENERATED-MANIFEST.json` records the seed, the pinned commit, the tokenizer hash, the corpus inventory size and the per-body token/byte accounting. The generator is byte-deterministic given (commit, tokenizer, seed); check (vi) asserts that on every run.
