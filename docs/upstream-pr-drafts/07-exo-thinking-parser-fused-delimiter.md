**Repo:** exo-explore/exo
**Branch:** adurham:pr/thinking-parser-fused-delimiter
**Target:** exo-explore/exo:main
**Commits:** 1 (`170f9023`, cherry-picked from fork `08eac031`)
**Depends on:** none — isolated, applies cleanly on current main

---

# Title

`fix(runner): handle fused/embedded think delimiters in parse_thinking_models`

# Body

## Summary

`parse_thinking_models` (`src/exo/worker/runner/llm_inference/model_output_parsers.py`)
splits a token stream into reasoning vs. content by detecting the `<think>` /
`</think>` delimiters. It flips the `is_thinking` flag **only on exact string
equality** (`accumulated == think_end`). But the mlx-lm streaming detokenizer
emits `last_segment` deltas that can carry **multiple tokens' worth of text in a
single chunk**, so the delimiter can arrive:

- **fused** with neighbouring text — e.g. `"...Let's code.</think>def fibonacci(n):"` in one chunk, or
- **spanning** chunks such that the accumulation overshoots — e.g. `"</"` then `"think>def"` → `accumulated == "</think>def"`.

In both cases the exact-equality check never matches, the boundary is missed,
and the delimiter **plus the post-delimiter answer leak into the wrong stream** —
chain-of-thought prose ends up in `content` (or the answer ends up flagged as
`is_thinking`).

This PR adds substring/boundary detection that splits around the delimiter
wherever it appears in the chunk, while preserving the existing clean-token fast
path and prefix-buffering verbatim.

## Motivation / reproducer

Observed on DeepSeek-V4-Flash (a thinking model whose `</think>` is token id
128822). On longer generations the detokenizer frequently emits the closing
delimiter fused with the first tokens of the answer. The result is a `content`
field containing a literal `</think>` followed by reasoning text and then the
real answer — which breaks any downstream consumer that treats `content` as the
clean answer (e.g. a code grader sees a `SyntaxError`; a UI shows raw `</think>`
tags; OpenAI-API clients get reasoning bleed in `message.content`).

Minimal logical repro (the streaming detokenizer delivers the delimiter fused):

```
chunk 0: "reasoning "           (is_thinking=True)
chunk 1: "done.</think>def x():" (the bug: </think> fused mid-chunk)
```

Pre-fix: chunk 1 falls through to the default branch and is emitted with the
stale `is_thinking=True`, so `</think>def x():` is misclassified.
Post-fix: emits `"done."` as thinking, swallows `</think>`, emits `"def x():"`
as content.

## Approach

In `parse_thinking_models`, before the default emit:

1. Locate the **currently active** delimiter (`think_end` while thinking,
   `think_start` otherwise) as a **substring** of the accumulation.
2. Split around it: emit the pre-delimiter text with the current flag, swallow
   the delimiter, flip `is_thinking`, and re-process the remainder in a loop
   (handles multiple fused delimiters in one chunk).
3. The clean single-token fast path (exact equality) and the partial-prefix
   buffering path are **unchanged**, so existing behaviour and its regression
   tests are preserved. Prefix-buffered tokens are cleared (not drained) when
   the substring branch fires, to avoid double-emitting delimiter fragments
   (e.g. a buffered `"</"` followed by `"think>answer"`).

The fix is generic to any thinking model — it is not specific to DeepSeek.

## Test plan

Adds `TestThinkingModelsFusedDelimiter` to
`src/exo/worker/tests/unittests/test_runner/test_finish_reason_sse.py` (5 cases):

- `test_fused_end_delimiter_single_chunk` — `"done.</think>def..."` in one chunk
- `test_end_delimiter_spanning_two_chunks` — `"</"` then `"think>answer"`
- `test_clean_end_delimiter_token_regression` — delimiter as its own token (the
  original fast path) still works
- `test_fused_start_delimiter_single_chunk` — symmetric `<think>` case
- `test_no_thinking_passthrough` — no delimiters, all content

All 5 pass; the 26 pre-existing tests in that file remain green (31/31 total).
`parse_thinking_models` is pure Python (operates on `GenerationResponse`
objects, no mlx/Metal dependency), so the tests run on CPU.

## Production validation

Running as the default thinking-delimiter parser on a 2-node Mac Studio M4 Max
cluster serving DeepSeek-V4-Flash (8-bit), where it eliminated `</think>`
leakage into `content` on long reasoning generations.

## Notes

Cherry-picked clean onto current `main` (post-zenoh, 09f9ea31); touches only the
parser and its test file (2 files, +193 lines), no other coupling.
