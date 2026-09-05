# Seam-Rule Harness -- Findings (round 12, prep only, NOT wired into production)

Status: **RAN successfully** against a real local tokenizer and the real
vendored DSv4 encoder. Full raw output saved in this directory's run log
(reproduce with `uv run python run_seam_harness.py` from the repo root, or
`python3 run_seam_harness.py` from inside this directory with `transformers`
importable).

Tokenizer used: `/Users/adam.durham/.exo/models/deepseek-ai--DeepSeek-V4-Flash`
(exo's own on-disk model store -- this is the actual production checkpoint
directory, not a downloaded stand-in; no network access was made).

Vendored encoder used: `src/exo/worker/engines/mlx/vendor/deepseek_v4_encoding.py`
(loaded read-only via `importlib` from its real repo path; this harness does
not modify or import it as part of any production code path).

## 1. Round-trip identity

Ran on two message-list corpora (`plain_multiturn`, `toolcall_multi_result`,
the latter deliberately containing two tool results delivered **out of
call-order and as separate messages**, matching the pre-registration's
tool-merge concern):

- **All candidate SAFE seams (immediately after an added/special token, as
  discovered from the tokenizer's own `offset_mapping`, not guessed by
  string search) PASSED**: `cached_prefix_tokens + tok(suffix,
  add_special_tokens=False) == tok(full)` held exactly, for every one of the
  11 safe seams in `plain_multiturn` and 18 safe seams in
  `toolcall_multi_result`.
- **BOS was emitted exactly once** in `tok(full)` in every case tested, and
  remained exactly once in every prefix+suffix reconstruction (no
  double-BOS, no missing-BOS).
- **The one candidate UNSAFE seam tested per corpus (a derived mid-token
  split, i.e. an offset strictly inside a single BPE token's span) was
  correctly REJECTED** -- the prefix+suffix reconstruction did NOT equal
  `tok(full)`, confirming the harness's negative control has teeth (it does
  not just report "no exception raised").

This establishes: **for this tokenizer, on these two corpora, the seam rule
as stated (safe seams = immediately after an added/special token) is
sufficient at the tokenization level** -- no counterexample was found where
a nominally-safe seam produced a mismatch, and the harness demonstrably
detects mismatches when they occur (proven by the unsafe-seam negative
control).

## 2. Adversarial seam corpus

Four hostile fragments were each embedded as `<｜User｜>prefix {hostile}
suffix<｜Assistant｜>` and tested at both a candidate-safe seam (right after
`<｜User｜>`) and a candidate-unsafe seam (mid-token, offset 4, inside the
`<｜User｜>` token's own span):

| Case | Hostile content | Safe seam | Unsafe seam |
|---|---|---|---|
| `combining_chars` | base letter + 3 combining diacritics split from following text | PASS | correctly REJECTED |
| `emoji_zwj` | 4-codepoint ZWJ family emoji + separate emoji | PASS | correctly REJECTED |
| `digit_run` | 19-digit run | PASS | correctly REJECTED |
| `whitespace_run` | mixed spaces/tabs/newlines | PASS | correctly REJECTED |

No adversarial case broke the safe-seam invariant, and the unsafe-seam
negative control fired correctly in every case. **Caveat**: these four cases
only test seams that land *after* the hostile content began (i.e., the
hostile content is entirely inside one contiguous span on one side of the
seam). They do not test a seam landing **inside** the hostile span itself at
a genuinely added-token boundary, because none of these hostile fragments
contain an added/special token internally -- that scenario is structurally
excluded by the seam rule (a safe seam by definition cannot be inside a
non-added-token span), so it was not a gap to test, not an oversight.

## 3. Template position-invariance -- **the highest-value finding**

**RESULT: `render()` is NOT prefix-stable.**

On `plain_multiturn` (no tool calls), every prefix relationship held:
`render(msgs[:1])` through `render(msgs[:6])` were all byte-prefixes of the
next longer render, in order.

On `toolcall_multi_result` (two tool results delivered out of call-order as
separate messages), the very first non-trivial multi-tool-result state
**broke position-invariance**:

- `render(msgs[:4])` ends with the tool-result content in the order it was
  *received*: SF's result before NYC's.
- `render(msgs[:5])` (one more message appended) re-renders the SAME earlier
  turns but with the tool results **re-sorted into original call order**:
  NYC's result now appears before SF's.
- Concretely, `render(msgs[:4])`'s tail differs from `render(msgs[:5])`'s
  tail starting at character 403 -- `SF: 60F foggy` vs `NYC: 75F sunny` in
  that position. This is a real, measured divergence in already-committed
  prefix bytes, not a hypothetical.

This is exactly the mechanism the pre-registration named:
`merge_tool_messages()` / `sort_tool_results_by_call_order()` inside the
vendored `deepseek_v4_encoding.py` re-merge and re-sort tool-result messages
**every time `encode_messages()` is called**, and the sort key depends on
information (the assistant's tool-call order) that is fixed once but whose
*effect on already-rendered text* only becomes visible once enough
downstream messages accumulate to trigger a re-sort of blocks that were
previously in a different position.

### Why this matters for the token-cache decision

A token/prefix cache that keys off "the text rendered for the first N
messages" is **provably unsafe** for any conversation that exercises
multi-tool-result reordering, independent of whether the *tokenizer-level*
seam rule (property 1) holds. The tokenizer-level seam rule is a necessary
but not sufficient condition: even a perfectly seam-safe tokenizer cannot
save a cache whose *input text itself* is not stable at the position the
cache would have committed to disk/memory in an earlier turn. A round using
only single-tool-call or no-tool-call conversations would never observe this
and would ship a token cache with a live correctness bug that only surfaces
on multi-tool-call turns -- exactly the "burn cluster time discovering it is
unsafe" scenario this harness exists to prevent finding the hard way.

## 4. Normalizer detection

`tokenizer.json`'s `normalizer` field is:

```json
{"type": "Sequence", "normalizers": []}
```

This is **present in the schema but structurally empty** (zero
sub-normalizers), i.e. an identity transform. The harness reports this
explicitly rather than silently treating a present-but-empty normalizer as
either "safe" or "absent" by assumption -- for this specific tokenizer, an
empty `Sequence` normalizer cannot rewrite bytes across a seam, so it does
NOT invalidate the seam rule's safety argument. If a future checkpoint's
`tokenizer.json` ships a non-empty normalizer (e.g. NFC/NFKC folding,
lowercasing, whitespace collapsing), the harness's normalizer check will
flag it as "CAN invalidate seam safety" and that specific normalizer would
need to be individually vetted for seam-invariance before trusting property
1's results for that checkpoint.

## What this harness does NOT establish

- It was run against exactly two synthetic message-list corpora and four
  synthetic hostile fragments, chosen to exercise the specific mechanisms
  named in the pre-registration. It is not an exhaustive fuzz of the
  tokenizer or the encoder -- a later round could extend
  `seam_corpus.py`'s corpora before trusting a wider safety claim.
- It does not measure performance, memory, or cluster behavior at all. It is
  purely a correctness/safety harness for the seam rule and template
  stability, per the task scope.
- It does not evaluate the `thinking_mode="thinking"` path, multimodal
  payloads, or the `context=` (pre-encoded prefix) parameter of
  `encode_messages()` -- those are additional surface area a later round
  should decide whether to add to `seam_corpus.py`.
- Property 3's divergence was demonstrated on ONE specific tool-result
  reordering shape. It is not a proof that ALL tool-call conversation shapes
  are unsafe, nor a proof that single-tool-call conversations (no reorder
  possible) are unaffected -- the plain_multiturn corpus's clean PASS is
  weak evidence the no-tool-call path is fine, but was not adversarially
  stress-tested beyond that one corpus.

## Bottom line for the round-13 decision

- Tokenizer-level seam safety (property 1): **holds** for this checkpoint on
  the corpora tested, including hostile boundary content, with a normalizer
  that is empty and therefore non-threatening.
- Template-level position-invariance (property 3): **does NOT hold** for
  conversations with reordered multi-tool-result turns. Any prefix-cache
  design that assumes "text rendered so far never changes as more messages
  are appended" needs either (a) a cache-invalidation trigger keyed to
  detecting a tool-result-reorder event, or (b) to be scoped OUT of
  multi-tool-call conversations entirely, or the cache will silently serve
  stale/wrong prefix tokens on exactly the conversation shape most likely to
  appear in agentic tool-use sessions.
