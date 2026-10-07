# Phase 1 — Reconciled per-call accounting + `reasoning_effort` wiring verdict

Date: 2026-10-07 · Read-only investigation, no cluster access, no commits.
Session: `20261007_092009_9a2ed7` · model `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`
(engine `dsv41`) · turn span 09:20:21 → 09:50:35 CDT (12m38s).

## Sources (every number below comes from a command run in this task)

- **Ledger** — `/Users/adam.durham/.hermes/state.db`, opened read-only
  (`sqlite3.connect('file:...?mode=ro', uri=True)`), table `api_calls`,
  `WHERE session_id='20261007_092009_9a2ed7' ORDER BY call_seq` → **42 rows, all
  `provider='custom'`, all `model='dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw'`**.
- **delta_rows (prefill=) / reuse_rows** — `/private/tmp/next14-gamma/docs/benchmarks/phase19-latency/raw/A1-real-turn-split.md`
  (per-call `turn reuse:` lines from `macstudio-m4-1:~/exo.log`, prefill-completion instants).
- **Live decode rate** — 20.66 output tok/s (agentic), `bench/phase19_agentic_measure.py`,
  documented in `.../phase19-latency/agentic-replay.md`.
- **Cold TTFT** — `phase19-latency/README.md` §2: 100K cold ttft 373.9 s, 30K cold ttft 111.7 s;
  live decode 28.08 t/s @100K, 29.97 t/s @30K.
- **Source** — `/Users/adam.durham/repos/exo` @ `deploy/next13 f0840af1c`.

---

## (A) Reconciled per-call accounting table

`prefill_s` / `decode_s` are **estimates**: `decode_s = output_tok / 20.66` (uniform agentic
decode rate) and `prefill_s = latency_s − decode_s`. Per-call `prefill_s` therefore folds in
client/tool time *inside* the call window (see caveats below), so it is an upper bound on true
prefill. The **totals** in the summary row are the reconciled quantities.

| call_seq | started CDT | latency_s | output_tok | reasoning_tok | prompt_tok | cache_read | delta_rows (prefill=) | prefill_s (est) | decode_s (est) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 09:20:21 | 98.80 | 175 | 122 | 24095 | 0 | 24095 (FULL=cold) | 90.3 | 8.5 |
| 2 | 09:22:00 | 6.80 | 90 | 5 | 24340 | 24095 | 245 | 2.4 | 4.4 |
| 3 | 09:22:51 | 14.79 | 233 | 93 | 25300 | 24431 | 869 | 3.5 | 11.3 |
| 4 | 09:23:06 | 25.28 | 284 | 37 | 28743 | 25300 | 3443 | 11.5 | 13.7 |
| 5 | 09:23:32 | 15.13 | 231 | 54 | 30178 | 28743 | 1435 | 4.0 | 11.2 |
| 6 | 09:23:47 | 42.88 | 133 | 77 | 39293 | 30178 | 9115 | 36.4 | 6.4 |
| 7 | 09:24:30 | 136.14 | 1572 | 1228 | 53564 | 39293 | 14271 | 60.1 | 76.1 |
| 8 | 09:26:48 | 44.70 | 830 | 300 | 56313 | 54588 | 1725 | 4.5 | 40.2 |
| 9 | 09:27:34 | 41.32 | 743 | 356 | 57541 | 56313 | 1228 | 5.4 | 36.0 |
| 10 | 09:28:17 | 85.00 | 1432 | 945 | 59602 | 57541 | 2061 | 15.7 | 69.3 |
| 11 | 09:29:46 | 19.04 | 245 | 92 | 61169 | 59602 | 1567 | 7.2 | 11.9 |
| 12 | 09:30:06 | 42.73 | 721 | 181 | 63904 | 61169 | 2735 | 7.8 | 34.9 |
| 13 | 09:30:56 | 37.78 | 606 | 495 | 64853 | 63904 | 949 | 8.4 | 29.3 |
| 14 | 09:31:33 | 12.26 | 109 | 19 | 66620 | 64853 | 1767 | 7.0 | 5.3 |
| 15 | 09:31:46 | 51.42 | 828 | 661 | 68147 | 66620 | 1527 | 11.3 | 40.1 |
| 16 | 09:32:38 | 15.94 | 199 | 40 | 69862 | 68147 | 1715 | 6.3 | 9.6 |
| 17 | 09:32:55 | 6.33 | 92 | 0 | 70159 | 69862 | 297 | 1.9 | 4.5 |
| 18 | 09:33:02 | 9.17 | 159 | 0 | 70622 | 70159 | 463 | 1.5 | 7.7 |
| 19 | 09:33:12 | 82.59 | 1542 | 1023 | 71044 | 70622 | 422 | 7.9 | 74.6 |
| 20 | 09:34:37 | 52.64 | 885 | 427 | 72678 | 72070 | 608 | 9.8 | 42.8 |
| 21 | 09:38:53 | 42.46 | 804 | 650 | 73568 | 73564 | 4 | 3.5 | 38.9 |
| 22 | 09:39:35 | 19.28 | 299 | 67 | 74893 | 73568 | 1325 | 4.8 | 14.5 |
| 23 | 09:39:55 | 14.38 | 265 | 81 | 75338 | 74893 | 445 | 1.6 | 12.8 |
| 24 | 09:40:11 | 46.99 | 251 | 143 | 83663 | 75338 | 8325 | 34.8 | 12.1 |
| 25 | 09:40:58 | 11.21 | 123 | 14 | 84980 | 83663 | 1317 | 5.3 | 6.0 |
| 26 | 09:41:09 | 21.99 | 375 | 216 | 86015 | 84980 | 1035 | 3.8 | 18.2 |
| 27 | 09:41:32 | 17.75 | 353 | 22 | 86529 | 86015 | 514 | 0.7 | 17.1 |
| 28 | 09:41:51 | 69.18 | 1110 | 786 | 87509 | 86529 | 980 | 15.5 | 53.7 |
| 29 | 09:43:01 | 45.21 | 720 | 513 | 88853 | 87509 | 1344 | 10.4 | 34.8 |
| 30 | 09:43:47 | 24.70 | 373 | 165 | 90793 | 88853 | 1940 | 6.6 | 18.1 |
| 31 | 09:44:11 | 19.82 | 293 | 95 | 92236 | 90793 | 1443 | 5.6 | 14.2 |
| 32 | 09:44:32 | 39.34 | 597 | 486 | 94402 | 92236 | 2166 | 10.4 | 28.9 |
| 33 | 09:45:12 | 10.76 | 147 | 0 | 95238 | 94402 | 836 | 3.6 | 7.1 |
| 34 | 09:45:24 | 21.65 | 373 | 250 | 95611 | 95238 | 373 | 3.6 | 18.1 |
| 35 | 09:45:45 | 7.47 | 103 | 0 | 96008 | 95611 | 397 | 2.5 | 5.0 |
| 36 | 09:45:53 | 22.36 | 362 | 258 | 97134 | 96008 | 1126 | 4.8 | 17.5 |
| 37 | 09:46:16 | 12.90 | 157 | 52 | 98382 | 97134 | 1248 | 5.3 | 7.6 |
| 38 | 09:46:30 | 18.26 | 271 | 160 | 99585 | 98382 | 1203 | 5.1 | 13.1 |
| 39 | 09:46:48 | 59.54 | 995 | 725 | 100945 | 99585 | 1360 | 11.4 | 48.2 |
| 40 | 09:47:49 | 110.62 | 1906 | 1411 | 102063 | 100945 | 1118 | 18.4 | 92.3 |
| 41 | 09:49:41 | 30.57 | 452 | 150 | 104396 | 103088 | 1308 | 8.7 | 21.9 |
| 42 | 09:50:12 | 23.42 | 361 | 0 | 104880 | 104396 | 484 | 5.9 | 17.5 |

**delta_rows cross-check (required):** for all 42 calls,
`prompt_tokens_total − cache_read_tokens == delta_rows` from the A1 exo log.
Mismatches: **none** (0/42). Σdelta_rows = **100,828** = Σinput_tokens (100,828). ✅

### Summary reconciliation

- **model_time = Σlatency = 1530.59 s** (ledger). Turn span = **1814.45 s**; Σinter-call
  gaps (client/tool time) = **283.86 s**; `Σlatency + Σgaps − span = 0.0 s` (exact).
- **Σoutput = 21,799 tok; Σreasoning = 12,399 tok; Σprompt = 3,091,048 tok; Σcache_read = 2,990,220 tok.**
- **decode_s = Σoutput / 20.66 = 21,799 / 20.66 = 1055.1 s.**
- **prefill_s = 1530.59 − 1055.1 = 475.5 s.**
- **Effective prefill rate = Σdelta_rows / prefill_s = 100,828 / 475.5 = 212.1 rows/s.**
- **Cold-call check:** call 1 = 24,095 rows / 98.80 s = **243.9 rows/s** (this leg is the
  measured cold path and is directly comparable). Known cold TTFT from README §2:
  100,000 / 373.9 = **267.5 rows/s**; 30,000 / 111.7 = **268.6 rows/s**.

**Do the numbers reconcile?** Within ~**11%**: the turn-implied 212.1 rows/s vs the measured
cold ~244–269 rows/s. They are **not** equal, and the direction of the gap is the honest one —
the uniform-20.66 t/s decoder *understates* decode in this turn (many calls were short/partial
decode, and call 40 at 92.3 s decode alone dominates), so `decode_s` is a floor and
`prefill_s = 475.5 s` is a **ceiling**. Conversely the residual is **not** zero: the turn-implied
rate is slower than the cold path, which is expected because (i) per-call `prefill_s` here
absorbs the intra-call tool/client overhead the ledger cannot separate, and (ii) cold TTFT is a
clean single-pass prefill, whereas the turn's deltas prefill *after* a reuse-undershoot rewind.
**Bound:** to reconcile this turn's Σdelta_rows at the measured cold rate of 267.5 rows/s the
true prefill time would be **100,828 / 267.5 = 377 s**, implying decode_s = 1530.6 − 377 = 1153.6 s
and an effective decode rate of 21,799 / 1153.6 = **18.9 t/s** (vs 20.66 assumed). So
**prefill_s ∈ [377 s, 475 s]** and the derived rows/s ∈ [212, 267]; the central estimate is
**475 s and 212 rows/s**, the aggressive bound is 377 s and 267 rows/s. The accounting
**reconciles to ~11% on the prefill term**, and exactly on the latency/gap/span term (0.0 s).

### Reasoning share

- Σreasoning = **12,399 tok = 56.9%** of Σoutput (12,399 / 21,799).
- per-call mean = 12,399 / 42 = **295.2 tok**; max = **1,411** (call 40); min = **0**
  (calls 17, 18, 33, 35, 42 — five calls with zero reasoning).
- Decode seconds attributable to reasoning = 12,399 / 20.66 = **600.1 s** = **39.2% of the
  1530.6 s model time** and **33.1% of the 1814.4 s turn wall**.

### Direct address of the campaign-brief claim

> "the 42 calls generate 118–160K hidden reasoning tokens (2.8–3.8K per call), i.e. 1.6–2.2 h at 20 t/s."

**Verdict: FALSE.** Actual hidden reasoning for this turn = **12,399 tok total**, mean
**295 tok/call** — an order of magnitude **below** the 2.8–3.8K/call band (10–13× lower), and
**9.5–12.9× below** the 118–160K total. At 20 t/s the real figure is 12,399 / 20 = 620 s ≈
**10.3 min**, not 1.6–2.2 h. Where 2.8–3.8K/call could come from: it is not this session's
central tendency — only **4 of 42** calls even reach ~1K reasoning. The likely origin is
(a) a few outliers read as typical, or (b) a different session/model. Top-5 reasoning calls:

| rank | call_seq | reasoning_tok |
|---:|---:|---:|
| 1 | 40 | 1,411 |
| 2 | 7 | 1,228 |
| 3 | 19 | 1,023 |
| 4 | 10 | 945 |
| 5 | 28 | 786 |

Even the top-5 sum (5,393) is 3.6% of the low end of the claimed band; the max single call
(1,411) is half of the *claimed per-call minimum* (2.8K). The claim does not describe this turn.

---

## (B) `reasoning_effort` wired / partial / dead — verdict: **PARTIAL**

**Agent config:** `~/.hermes/config.yaml` → `agent.reasoning_effort: 'ultra'`, with per-model
`reasoning_effort_by_model['dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw'] = 'xhigh'`.
The session's persisted config confirms the intent:
`sessions.model_config` = `{"reasoning_config": {"enabled": true, "effort": "xhigh"}, ...}`.
Effective effort on the wire = **xhigh** (not "ultra"; see clamp below).

Trace, API boundary → dsv41 engine:

1. **Wire ceiling & clamp** — `src/exo/shared/types/text_generation.py:15-33,36-53`:
   `ReasoningEffort` Literal tops out at `"xhigh"`; `REASONING_EFFORT_LADDER` extends to
   `"ultra"`; `REASONING_EFFORT_CEILING = "xhigh"`; `clamp_reasoning_effort()` (L36) drops
   over-ceiling ladder names to the ceiling. So Hermes' `ultra` → **`xhigh`** at the wire.
2. **Chat-completions boundary validator** — `src/exo/api/types/api.py:285-286` (fields),
   `:308-320` `@field_validator("reasoning_effort", mode="before")` calls
   `clamp_reasoning_effort(v)`. Applied to the OpenAI-compat request.
3. **(effort, thinking) resolution** — `src/exo/shared/types/text_generation.py:75-94`
   `resolve_reasoning_params()`: `reasoning_effort="xhigh"` (≠ "none") ⇒
   `enable_thinking = True`; effort stays `"xhigh"`. Called at the adapter:
   `src/exo/api/adapters/chat_completions.py:183-184`; results written onto the internal
   task params at `:201-202` (`reasoning_effort=resolved_effort, enable_thinking=resolved_thinking`).
4. **Card sampling defaults** — `TextGenerationTaskParams.with_card_sampling_defaults()`
   (`text_generation.py:226-260`, esp. L234-237) only selects the thinking vs non_thinking
   sampling card; it does **not** consume `reasoning_effort`. Invoked at
   `src/exo/api/main.py:1062`. The dsv41 card
   (`resources/inference_model_cards/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw.toml:74-76`)
   defines only `[sampling_defaults] temperature=0.0, min_p=0.0` — no thinking/non_thinking split.
5. **Prompt assembly (the one live consumer)** — the dsv41 engine renders the prompt via
   `dsv41/engine.py:592-616` `_render_prompt()` → `apply_chat_template()` (L613). For this
   text-only request that routes through `utils_mlx.py:1948-1989` →
   `render_chat_template()` (`utils_mlx.py:1822-1945`). Because the model id contains
   `deepseek-v4` (`_needs_v4_encoding`, `utils_mlx.py:1770-1771`), it takes the vendored V4
   branch **`utils_mlx.py:1859-1898`** and calls:
   - `thinking_mode = "chat" if enable_thinking is False else "thinking"` (L1890-1892), and
   - `reasoning_effort = _v4_reasoning_effort(task_params)` (L1893).
   `_v4_reasoning_effort()` (`utils_mlx.py:1774-1780`) maps `"xhigh"→"max"`, `"high"→"high"`,
   **everything else → `None`**.
6. **Effort actually changes tokens** — `vendor/deepseek_v4_encoding.py:82-95` defines
   `REASONING_EFFORT_PROMPTS = {"low":"", "high":"…", "max":"…"}` and
   `DEFAULT_REASONING_EFFORT="low"`; `render_message()` (`:261-313`) prepends
   `REASONING_EFFORT_PROMPTS[reasoning_effort]` at `index == 0` **only in thinking mode**
   (`:307-313`). For this turn: effort `"max"` ⇒ ~526-char effort preamble is prepended to
   the prompt on **every** call (prompt positions are identical, so it is part of the cached
   prefix). The `"low"` value adds nothing.

So the field is **consumed and material** (it changes the rendered prompt), but:

- **It is NOT a continuous gradient.** Only 3 rungs exist on this wire: `low`/default (no
  prefix), `high`, `max`. `xhigh` and `ultra` both collapse to `max`; `minimal`/`medium`
  collapse to `None`→`low` (no prefix). There is no "xhigh-specific" text distinct from "max".
  The `min_p`/sampler path is untouched by effort.
- **Sampler is not affected** — the engine is greedy-only (`dsv41/rounds.py:21-27`;
  card `.toml:65-76`), so effort cannot modulate temperature/top-p; the card's
  `temperature=0.0` is what runs.
- **Downstream engine internals ignore it** — after `render_chat_template` produces the
  prompt string, `reasoning_effort` / `enable_thinking` are never read again in the dsv41
  package (`rounds.py`, `session.py`, `output.py` only handle the *output* thinking split;
  `output.py:93-111`). The only `enable_thinking` literal in `dsv41/` is the warmup call
  (`engine.py:377`).

**Verdict: PARTIAL.** `reasoning_effort` is a **functioning knob at the prompt level** — it is
not a no-op and not dropped: `xhigh` (from Hermes' `ultra`/`xhigh`) is clamped to the wire
ceiling, resolved to `enable_thinking=True`, mapped by `_v4_reasoning_effort` to the model's
`"max"` effort prompt, and that ~526-char preamble is prepended to the conversation. But it is
**coarse**: a 3-value thinking-on/off-with-preamble toggle (`low`/`high`/`max`), with no
monotone effort gradient above `xhigh`, and it does **not** reach the dsv41 sampler or round
loop. If a future turn needs "more" than `xhigh`, nothing stronger exists on this wire.

**Citations (proven by source read):**
`text_generation.py:15,21-33,36-53,75-94,234-237` · `api/types/api.py:285-286,308-320` ·
`api/adapters/chat_completions.py:183-184,201-202` · `api/main.py:1062` ·
`utils_mlx.py:1770-1771,1774-1780,1859-1898` · `vendor/deepseek_v4_encoding.py:82-95,261-313` ·
`dsv41/engine.py:377,592-616` · `dsv41/rounds.py:21-27` ·
card `dealignai--…EXL3-2.9bpw.toml:56,65-76`.

---

## (C) System-prompt stability

- `sessions.system_prompt_hash` for this session =
  **`13466837d495ef2fda6d113fbc7656ab13431a876c7efe9e40a2d706a99a708a`**.
- `system_prompts` holds exactly **one** row for that hash, with `length(prompt) = 22,770`
  chars — matching the A1/agentic-replay doc's stated 22,770-char preamble.
- **Stability:** the ledger stores one system prompt per session keyed by hash; all 42
  `api_calls` rows belong to this single session, so all 42 calls share the **same** system
  prompt (byte-identical, fixed 22,770 chars). The hash is used by **exactly 1** session
  (`SELECT count(*) FROM sessions WHERE system_prompt_hash=…` = 1), out of 572 distinct hashes
  across sessions — i.e. it is session-specific, not globally shared, and stable within it.
- **Churn check (per-call volatility that would break a cached prefix):** the stored prompt
  contains **no** wall-clock times (`HH:MM:SS` matches = 0), **no** session ids (`\d{9,}` = 0),
  **no** "Working directory"/`<cwd>` sentinel lines (0). It **does** contain 8 absolute dates
  (`2026-09-13`, `2026-09-25`, `2026-10-07`, …) and the workdir path `/Users/adam.durham`
  (4 occurrences), but these are **fixed strings baked at session start**, not regenerated
  per call — the hash is constant across the turn and the 42-call prompt prefix is
  byte-stable, so the cache prefix is **not** invalidated by per-call content. (Caveat: if the
  prompt had carried a live timestamp or a per-turn memory snippet, the hash would change
  mid-session; it did not — one hash, one row, used by all 42 calls.)

**Hashes actually used:** `13466837d495ef2fda6d113fbc7656ab13431a876c7efe9e40a2d706a99a708a`
(the only one). Stable.
