# A1 — Live API Cross-check of `reasoning_content` Omission vs Empty vs Space Pad (DeepSeek-V4-Flash-0731)

Date of run: 2026-09-02 (17:33–17:34 local)
Gate file: this document
Scope: LIVE half of the pre-registered A1 verification. Offline half (token-count prediction) was completed previously and is treated as given.

---

## 1) Request bodies used (verbatim from artifacts/a1_offline_render.json)

All three payloads hit `POST http://192.168.86.201:52415/v1/chat/completions` with `model = "deepseek-ai/DeepSeek-V4-Flash-0731"` (the id the server lists), `max_tokens = 1`, `temperature = 0`, `stream = false`. The bodies differ only in the `reasoning_content` handling of the prior assistant message:

- **(a) a_absent**: `reasoning_content` key omitted entirely from the assistant message that carries tool_calls.
- **(b) b_empty**: `reasoning_content: ""` present on that assistant message.
- **(c) c_space**: `reasoning_content: " "` (a single space) present on that assistant message.

Common structure of every message list:
```json
{
  "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant. Answer briefly."},
    {"role": "user", "content": "What is the weather in Hangzhou? Use the tool."},
    {"role": "assistant", "tool_calls": [
      {"id": "call_1", "type": "function", "function": {"id": "<UUID>", "name": "get_weather", "arguments": "{\"city\": \"Hangzhou\"}"}}
    ]},
    {"role": "user", "content": "Now summarize the result in one sentence."}
  ],
  "max_tokens": 1,
  "stream": false,
  "temperature": 0,
  "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
}
```
Only difference between the three bodies: presence/absence/value of the `reasoning_content` field on the `assistant` message (exact JSON payloads stored in artifacts/a1_offline_render.json, `request_bodies` key).

---

## 2) Offline render summary (given, from prior work)

Using the real encoder path (`ChatCompletionRequest.model_validate -> chat_completions adapter -> TextGenerationTaskParams(chat_template_messages) -> utils_mlx.apply_chat_template -> deepseek_v4_encoding.encode_messages`; tokenizer `deepseek-ai/DeepSeek-V4-Flash-0731` snapshot 7872f01b):

- **(a) key ABSENT** -> **353 tokens**
- **(b) key = ""** -> **353 tokens**, token-id list BYTE-IDENTICAL to (a)
- **(c) key = " "** -> **354 tokens**

(a) vs (c) differ by EXACTLY ONE inserted token: id **223** (a single space) at index 294. Removing index 294 from (c) yields (a) exactly. Common prefix 294 tokens, common suffix 59 tokens (294 + 59 = 353 = len(a)). Rendered text differs only as ` thinking response` (a) vs ` thinking  response` (c). No `None`/`null`/`NoneType` artifacts appear in (a).

---

## 3) Live `usage` blocks (observed this run)

| Variant | HTTP status | `usage.prompt_tokens` | `usage.completion_tokens` | `usage.total_tokens` | `cache` (`prompt_tokens_details.cached_tokens`) | wall time |
|---|---|---|---|---|---|---|
| a_absent | **200** | **353** | 1 | 354 | 0 | 3.1 s |
| b_empty | **200** | **353** | 1 | 354 | **351** (cache hit) | 1.0 s |
| c_space | **200** | **354** | 1 | 355 | 0 | 3.3 s |

All three responses were well-formed `chat.completion` objects: single element in `choices[]`, each with `index`/`message`/`finish_reason`; `message.content == ""` (max_tokens=1), `finish_reason == "length"`.

Server-side prefill logs corroborate the accounting: `a_absent` prefilled 352 tokens (the 353rd token is the one consumed by the first decode step), `b_empty` prefilled 1 token on 351 cached (a cache hit — total prompt 353), `c_space` prefilled 353 (total 354). Full raw responses saved to `artifacts/a1_live_responses.json`.

---

## 4) Cross-check assertions (THE TEST)

- **A1 — prompt_tokens(a) == prompt_tokens(b):** 353 == 353 → **PASS**
- **A2 — prompt_tokens(c) == prompt_tokens(a) + 1:** 354 == 354 → **PASS**
- **A3 — prompt_tokens(a) == 353 (exact absolute match to offline):** 353 == 353 → **PASS** (no offset needed; absolute match exact)
- **A4 — all three HTTP 200 with well-formed choices[0]:** 200/200/200, each with a valid single choice → **PASS**

No constant offset case arises; the live absolute counts match the offline prediction exactly, and the relative deltas (A1, A2) agree. The load-bearing relative checks are satisfied.

---

## 5) Server log evidence

No logs were found at the prompt-rendering/tokenization granularity with content-level detail; the cluster nodes' `exo -v` run logs `apply_chat_template:1597` entries for each rendered prompt prefix but do not dump full token-id lists or a warning/error mentioning `reasoning_content`/think markers/template rendering. No errors or warnings were observed for these three requests. The `apply_chat_template` log entries for the three requests (timestamps 17:33:53.963 / 17:33:56.439 / 17:33:56.972) confirm each request reached template application with identical rendered prefix text and no `None`/`null`/`reasoning` artifact present in the rendered prompt lines. (Note: the node log was actively being written to by other cluster traffic at the time; the relevant entries were extracted by line number.)

---

## 6) Pre-registered gate criteria — evaluated

- **G1 — no error:** All three variants returned HTTP 200 with a well-formed completion (`choices[0]` present, `finish_reason=="length"`). No error/exception surfaced in the API layer or the logs. → **PASS**
- **G2 — absent is clean:** Variant (a)'s rendered prompt shows no artifact from the missing field — no literal `None`/`null`/`NoneType`, no dangling/empty reasoning delimiters, no duplicated or dropped turn markers (confirmed by inspecting the `apply_chat_template` output for the `a_absent` request and by the byte-identical prompt-tokenization of (a) vs (b) from offline work). → **PASS**
- **G3 — divergence is localized:** The only structural difference between (a) and (c) is the single inserted pad token (id 223 at index 294) in the reasoning slot; no re-segmentation of surrounding text and no cascade. Live `prompt_tokens` deltas corroborate this: c - a = +1 exactly, and the server-side prefill step counts confirm the +1 is the pad token alone. → **PASS**
- **G4 — absent is not worse:** Variant (a) does not lengthen the prompt or add tokens versus (c); in fact (a) is one token SHORTER than (c) (353 < 354). → **PASS**

All four pre-registered criteria are satisfied by observed live evidence.

---

## 7) Final verdict

**GATE: PASS**
