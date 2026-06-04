# Thinking-Delimiter Parser — Fused / Embedded `</think>` Fix

**Status:** Shipped in fork, fork commit `08eac031` on `adurham/exo` main.
**Upstreamable:** Yes — `model_output_parsers.py` exists in `exo-explore/exo`
with the identical bug. See `docs/upstream-pr-drafts/`.

**TL;DR:** `parse_thinking_models` flipped `is_thinking` only on **exact string
equality** (`accumulated == think_end`). The mlx-lm streaming detokenizer emits
`last_segment` deltas that can carry multiple tokens' text, so the `</think>`
delimiter can arrive **fused** with neighbouring text in one chunk
(`"...done.</think>def fibonacci(n):"`) or **overshoot** the exact match when
split across chunks (`"</"` then `"think>def"` → accumulates `"</think>def"`).
Both cases miss the boundary, leaking the delimiter + the real answer into the
wrong stream — chain-of-thought prose dumped into `content`.

---

## 1. Location

`src/exo/worker/runner/llm_inference/model_output_parsers.py`,
`parse_thinking_models()` (~line 361). Routes a token generator into
`is_thinking`-flagged chunks by detecting the `think_start` / `think_end`
delimiters. Used by the DeepSeek-V3.2, DeepSeek-V4, and generic thinking-model
paths (all gated on `tokenizer.has_thinking`).

## 2. The bug

The boundary flip was:

```python
if accumulated == think_end and is_thinking:   # line 398 (pre-fix)
    is_thinking = False
    ...
```

`accumulated` grows by `response.text` each chunk and resets on any token that
isn't a clean prefix of a delimiter. The exact-equality check assumes the
delimiter always lands as its own standalone chunk. But the runtime mlx-lm
detokenizer (`BPEStreamingDetokenizer` / `NaiveStreamingDetokenizer`) returns
`last_segment` — the delta of decoded text — which can materialise **multiple
tokens at once** (BPE buffers incomplete bytes; the naive detokenizer commits on
newline boundaries). So a single chunk legitimately can be `"code.</think>def"`,
and `accumulated == think_end` is never true. The delimiter (and the answer
after it) then fall through to the default branch and are emitted with the stale
`is_thinking`, landing chain-of-thought prose in the `content` field.

## 3. Observed impact

On DeepSeek-V4-Flash-8bit, the `optimize_fibonacci` benchmark task failed
intermittently: content contained a literal `</think>` plus interleaved
reasoning prose, which a code grader then rejected as a SyntaxError. (Note: this
parser bug was a *contributing* layer; the deeper cause of that specific task's
failure was the separate MTP tie-break issue — see
`docs/mtp-tiebreak-losslessness-fix.md`. The parser fix is correct and
necessary regardless: reasoning leaking into `content` is a real correctness bug
for any consumer whenever a thinking model emits a fused delimiter.)

## 4. The fix

Add a **substring/boundary** detection branch before the default emit: locate
the active delimiter (`think_end` while thinking, `think_start` otherwise) as a
substring of `accumulated`, split around it — emit the pre-delimiter text with
the current flag, swallow the delimiter, flip `is_thinking`, and re-process the
remainder in a loop (handles multiple fused delimiters in one chunk). The clean
single-token fast path (exact equality) and the prefix-buffering path are
preserved verbatim, so existing behaviour and its regression tests are
unchanged. The prefix-buffered tokens (which mirror their text into
`accumulated`) are cleared rather than drained when the substring branch fires,
to avoid double-emitting delimiter fragments (e.g. a buffered `"</"` followed by
`"think>answer"`).

## 5. Tests

Added `TestThinkingModelsFusedDelimiter` in
`src/exo/worker/tests/unittests/test_runner/test_finish_reason_sse.py`:

1. `test_fused_end_delimiter_single_chunk` — `"done.</think>def..."` in one
   chunk → pre-text thinking, post-text content, delimiter swallowed.
2. `test_end_delimiter_spanning_two_chunks` — `"</"` then `"think>answer"`.
3. `test_clean_end_delimiter_token_regression` — delimiter as its own token
   (the original fast path) still works.
4. `test_fused_start_delimiter_single_chunk` — symmetric `<think>` case.
5. `test_no_thinking_passthrough` — no delimiters, everything is content.

Result: 31/31 in that file (5 new + 26 pre-existing), `basedpyright` clean on
the changed file.

## 6. Upstreamability

`exo-explore/exo` carries `model_output_parsers.py` with the identical
exact-equality bug (verified against `exo-explore/main`: same
`parse_thinking_models`, same line-398 check). The fix is generic to any
thinking model (not DSv4-specific) and ships with tests, making it a clean
Tier-1 upstream candidate. PR draft and tracker entry under
`docs/upstream-pr-drafts/`.
