# LCP-coverage probe — decode output vs next-turn re-fed prompt (54 turn pairs)

Offline, read-only, 2026-09-02. No cluster contact, no inference, no commits.
Feeds the Fix-B decision (retain decode KV in the prefix-cache trie); the PM
applies the pre-registered bands — this report deliberately contains NO
pass/fail verdict and NO build/abandon recommendation.

## 0. The one-paragraph answer

**47 of 54 turn pairs reproduce the decode output BYTE-IDENTICALLY as the next
turn's prompt (LCP_coverage = 1.000). 7 pairs score 0.000 — every one of them
because Hermes echoes a single-space `reasoning_content` pad on tool-call turns
whose decode produced NO reasoning, and the pad token sits where the decode had
no-`<think>`-opener straight-through output, so the very first re-fed token
diverges.** Raw distribution: median 1.000, p25 1.000, **p10 0.000**, mean
0.870, min 0.000, max 1.000. Token-weighted (recoverable decode tokens /
total decode tokens): **96.2%**. The bimodality is the finding: Fix B either
recovers a whole turn's decode KV (47/54 turns, 96% of B-bucket tokens) or
nothing (7 turns, 3.8%), with no middle ground — and the zero class is caused
by a one-token client-side placeholder, not by the model or the template.

## 1. Method (what was actually run)

- **Tokenizer:** the real DSv4 tokenizer,
  `deepseek-ai/DeepSeek-V4-Flash-0731/tokenizer.json` from the local HF cache
  (no download, no chars//4). Special-token ids verified before use:
  `<｜User｜>`=128803, `<｜Assistant｜>`=128804, `<think>`=128821,
  `</think>`=128822, `｜DSML｜`=128825.
- **Re-feed side (exact production path, reproduced offline):** Hermes wire
  shape (`agent/conversation_loop.py:2552-2660`: system first, assistant
  `reasoning_content` echoed, tool rows) → exo adapter
  (`api/adapters/chat_completions.py:62`, skip-empty + `exclude_none`) →
  `utils_mlx.render_chat_template` (consolidate system;
  `_strip_v4_thinking_markers` on assistant **content**) →
  `vendor/deepseek_v4_encoding.encode_messages` (merge_tool_messages →
  sort_tool_results_by_call_order → render_message). **Tools are in play on
  every turn of this session, and exo's encoder is tool_conditional: with tools
  present `drop_thinking` is DISABLED — every prior assistant turn re-renders
  its full `reasoning_content` + `</think>`.** Tokenization mirrors
  `cache.encode_prompt`: `tokenizer.encode(prompt, add_special_tokens=False)`.
- **Decode side (PROXY — the one idealization, labeled):** raw decode token ids
  are not persisted anywhere (checked: requests.jsonl is metadata-only; exo
  event log carries usage counts only; Hermes state.db stores text). Decode
  text is therefore RECONSTRUCTED from stored `reasoning` + `content` +
  tool_calls and re-tokenized. `reasoning` and `content` are stored verbatim
  (measured: `reasoning == reasoning_content` on 51/59 assistant rows; the 8
  exceptions are single-space pads on rows whose decode reasoning was empty),
  so the text is exact; what the proxy cannot see is (a) the model's true BPE
  segmentation at decode time and (b) the raw JSON spacing the model emitted
  inside DSML `string="false"` parameters (parse_dsml_output re-serializes with
  `json.dumps` before storage). Direction of bias: **optimistic** — canonical
  re-tokenization cannot reproduce decode-time segmentation quirks, and the
  tool-call region is identical to the re-feed *by construction* because both
  sides render from the same parsed arguments.
- **Sanity vs recorded counts (MEASURED):** proxy decode length sums to
  **41,413** tokens vs **41,414** recorded `completion_tokens` over the same 54
  turns (Δ = −1 token total; max per-pair error 0.02%). The reconstruction is
  the right turns at the right sizes.
- **Position invariant (MEASURED, then explained):** reconstructed
  `len(prompt(n)) − common_prefix(prompt(n), prompt(n+1)) = 0` on all 54 pairs —
  prompt(n) including its trailing `<｜Assistant｜><think>` header is a strict
  prefix of prompt(n+1). The recorded invariant
  `cached_tokens(n+1) = len(prompt(n)) − 2` is the **serve-side exact-hit
  reserve**, not a token mismatch: `cache.py:1301-1303` keeps at least one token
  out (`target = max_length − 1` on exact hits, plus DSv4 non-sliceable-layer
  snapshot granularity), so a perfect prefix still reports 2 uncached tokens.
  This does not affect the LCP metric, which compares decode ids against the
  re-fed region after the matched prefix.

## 1b. The critical requirement, answered head-on

**Do completions open with thinking, and does stripping destroy the position
alignment? NO.** This session's completions open with real `<think>…</think>`
thinking on 45/54 compared turns (96.0% of decode tokens). The
thinking-marker-strip path (`_strip_v4_thinking_markers`) fires on `content`,
which never contains markers (exo's parser splits reasoning into
`reasoning_content` at the source), so the strip is a **no-op on 54/54 turns**
and no RoPE position shift occurs. Separately, the encoder's tool-conditional
rule retains prior-turn `reasoning_content` (tools ⇒ `drop_thinking=False`), so
the thinking is re-fed, in place, verbatim. **The recoverable fraction of the
22% B-bucket does NOT approach zero — the measured ceiling is ~96% of B-tokens
from whole turns (plus the one-token reserve), and the residual 4% is a
one-token client placeholder artifact, not a structural loss.**

## 2. Raw distribution (MEASURED, n=54)

| statistic | LCP_coverage | LCP_tokens |
|---|---:|---:|
| n | 54 | 54 |
| min | 0.0000 | 0 |
| **p10** | **0.0000** | 0 |
| p25 | 1.0000 | 197.5 |
| median | 1.0000 | 431.5 |
| mean | 0.8704 | 738.0 |
| max | 1.0000 | 6479 |

Shape: strictly bimodal — 47 pairs at exactly 1.000, 7 pairs at exactly 0.000,
nothing in between (variant-B lower-bound proxy, below, adds 7 partial pairs
from multi-invoke block-shape ambiguity: p25 0.864, mean 0.841).

## 3. Divergence taxonomy (the 7 low scorers)

Every low scorer has the SAME cause, precisely identified:

**`reasoning-echo-space-pad` (7/7 zero-coverage pairs: 34→35, 48→49, 50→51,
51→52, 53→54, 56→57, 84→85).** On turns where the model emitted NO reasoning
(went straight `</think>`-less → content/tool-call), Hermes's reasoning-echo policy
for DeepSeek-family endpoints (`agent/message_sanitization.py:925`, the
DeepSeek V4 HTTP-400 empty-string workaround) pads the wire with
`reasoning_content: " "` (a single space, written at message-creation time —
confirmed in state.db: 8 assistant rows store `reasoning_content=' '` while
`reasoning` is empty). The re-fed prompt therefore renders `␣ + <think>`
where the decode output had `<think>` directly. First token diverges →
LCP = 0 despite the ENTIRE remaining decode (1003, 98, 118, 99, 99, 99, 46
tokens — 1,562 tokens total) being byte-identical on the wire. The two
no-reasoning turns that did NOT get a pad (`reasoning_content=None` rows
38→39, 63→64) score 1.000 — proving the pad token, not the absence of
thinking, is the cause.

**Zero occurrences** in this session of: thinking-strip position shift,
tool-call DSML re-serialization divergence, boundary BPE merge, system-prompt
mutation, `<｜latest_reminder｜>` injection, context-compaction rewrite. (No
compaction rows in the window; toolset byte-stable; strip a verified no-op.)

**Unmeasurable-but-real (proxy blind spot, 8 pairs):** turns with TWO tool
calls (39, 41, 42, 43, 45, 64, 72, 82). Variant A (single merged
`<｜DSML｜tool_calls>` block — what the re-feed renders) scores them 1.000.
Variant B (one block PER invoke — the other shape the model may have emitted;
exo's parser recovers both into one array, so the wire shape is not
persisted) scores 0.755–0.984 on them (e.g. 82→83: 5,148/6,491). If the model
splits multi-invoke turns into separate blocks, those turns lose coverage from
the first block close onward. Bounded between the A and B columns in the JSON.

## 4. What Fix B gets (MEASURED + DERIVED)

- 47/54 turns (87%): decode ids ≡ re-fed ids for the whole turn → decode KV is
  reusable from the first reasoning token through EOS, minus the 1-token
  exact-hit reserve the serve path already keeps.
- 7/54 turns (13%): 0 reusable tokens, but the divergence is ONE injected
  space token. A trivially small change (echo `""` as empty instead of `" "`,
  or teach the trie to skip the pad) converts these to 1.000 — that is a
  client-side one-line fix, separate from and complementary to Fix B.
- Token-weighted recoverable fraction of the B-bucket (41,413 proxy tokens):
  **96.2%** (variant A) / **88.4%** (variant B lower bound).

## 5. Per-pair table (MEASURED)

| pair | decode_len (proxy) | completion_tokens (recorded) | LCP (tok) | coverage | first_divergence_idx (prompt n+1) | cause |
|---|---:|---:|---:|---:|---:|---|
| 33->34 | 1982 | 1982 | 1982 | 1.000 | — | full-match |
| 34->35 | 1003 | 1003 | 0 | 0.000 | 2296 | reasoning-echo-space-pad |
| 35->36 | 337 | 337 | 337 | 1.000 | — | full-match |
| 36->37 | 201 | 201 | 201 | 1.000 | — | full-match |
| 37->38 | 646 | 646 | 646 | 1.000 | — | full-match |
| 38->39 | 64 | 64 | 64 | 1.000 | — | full-match |
| 39->40 | 1254 | 1254 | 1254 | 1.000 | — | full-match |
| 40->41 | 419 | 419 | 419 | 1.000 | — | full-match |
| 41->42 | 487 | 487 | 487 | 1.000 | — | full-match |
| 42->43 | 455 | 455 | 455 | 1.000 | — | full-match |
| 43->44 | 513 | 513 | 513 | 1.000 | — | full-match |
| 44->45 | 165 | 165 | 165 | 1.000 | — | full-match |
| 45->46 | 810 | 810 | 810 | 1.000 | — | full-match |
| 46->47 | 219 | 219 | 219 | 1.000 | — | full-match |
| 47->48 | 218 | 218 | 218 | 1.000 | — | full-match |
| 48->49 | 98 | 98 | 0 | 0.000 | 36992 | reasoning-echo-space-pad |
| 49->50 | 617 | 617 | 617 | 1.000 | — | full-match |
| 50->51 | 118 | 118 | 0 | 0.000 | 40143 | reasoning-echo-space-pad |
| 51->52 | 99 | 99 | 0 | 0.000 | 40537 | reasoning-echo-space-pad |
| 52->53 | 530 | 530 | 530 | 1.000 | — | full-match |
| 53->54 | 99 | 99 | 0 | 0.000 | 43487 | reasoning-echo-space-pad |
| 54->55 | 2903 | 2903 | 2903 | 1.000 | — | full-match |
| 55->56 | 689 | 689 | 689 | 1.000 | — | full-match |
| 56->57 | 99 | 99 | 0 | 0.000 | 48796 | reasoning-echo-space-pad |
| 57->58 | 455 | 455 | 455 | 1.000 | — | full-match |
| 58->59 | 413 | 413 | 413 | 1.000 | — | full-match |
| 59->60 | 1748 | 1748 | 1748 | 1.000 | — | full-match |
| 60->61 | 220 | 220 | 220 | 1.000 | — | full-match |
| 61->62 | 444 | 444 | 444 | 1.000 | — | full-match |
| 62->63 | 788 | 788 | 788 | 1.000 | — | full-match |
| 63->64 | 43 | 43 | 43 | 1.000 | — | full-match |
| 64->65 | 1541 | 1541 | 1541 | 1.000 | — | full-match |
| 65->66 | 1545 | 1545 | 1545 | 1.000 | — | full-match |
| 66->67 | 992 | 992 | 992 | 1.000 | — | full-match |
| 67->68 | 567 | 567 | 567 | 1.000 | — | full-match |
| 68->69 | 336 | 336 | 336 | 1.000 | — | full-match |
| 69->70 | 230 | 230 | 230 | 1.000 | — | full-match |
| 70->71 | 347 | 347 | 347 | 1.000 | — | full-match |
| 71->72 | 1687 | 1687 | 1687 | 1.000 | — | full-match |
| 72->73 | 2721 | 2721 | 2721 | 1.000 | — | full-match |
| 73->74 | 187 | 187 | 187 | 1.000 | — | full-match |
| 74->75 | 1248 | 1248 | 1248 | 1.000 | — | full-match |
| 75->76 | 182 | 182 | 182 | 1.000 | — | full-match |
| 76->77 | 314 | 314 | 314 | 1.000 | — | full-match |
| 77->78 | 1497 | 1497 | 1497 | 1.000 | — | full-match |
| 78->79 | 201 | 201 | 201 | 1.000 | — | full-match |
| 79->80 | 308 | 308 | 308 | 1.000 | — | full-match |
| 80->81 | 180 | 180 | 180 | 1.000 | — | full-match |
| 81->82 | 1156 | 1156 | 1156 | 1.000 | — | full-match |
| 82->83 | 6479 | 6480 | 6479 | 1.000 | — | full-match |
| 83->84 | 457 | 457 | 457 | 1.000 | — | full-match |
| 84->85 | 46 | 46 | 0 | 0.000 | 93097 | reasoning-echo-space-pad |
| 85->86 | 348 | 348 | 348 | 1.000 | — | full-match |
| 86->87 | 708 | 708 | 708 | 1.000 | — | full-match |
`full-match` = LCP == decode length (identical ids end to end).
`reasoning-echo-space-pad` = first re-fed token is the space pad, decode had none.

## 6. Honesty ledger

- MEASURED: everything in the table; the 51/59 verbatim-echo and 8-pad counts;
  the no-compaction/no-reminder window state; the tokenizer id checks; the
  Δ=−1 token reconstruction total; the full-prefix position invariant.
- PROXY (optimistic): decode token ids (re-tokenized stored verbatim text;
  true decode ids not persisted anywhere — searched requests.jsonl, the exo
  event-log archive, and state.db).
- CONSTRUCTED (identical-by-construction, not independent evidence): the DSML
  tool-call region of the decode proxy (rendered from the same parsed arguments
  the re-feed uses). JSON spacing a model may have emitted differently, and
  multi-invoke block shape, cannot be recovered offline — bounded by variants
  A/B in `lcp_probe.json`.
- NOT APPLIED (per task): pre-registered pass/fail bands and any
  build/abandon recommendation. PM applies them to `summary.lcp_coverage`
  (`p10 = 0.0000` is the pre-registered gate input).

Machine-readable companion: `findings/lcp_probe.json` (per_pair includes
variant-A and variant-B ids, divergence text snippets ±10 tokens, echo class,
finish_reason; summary carries the distribution block).
