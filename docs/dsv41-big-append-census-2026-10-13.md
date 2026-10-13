# DSv4.1 big-append prefill — Stage 0 offline census (P1, P2, §0.1–0.6)

**Date:** 2026-10-13. **Scope:** the plan `docs/dsv41-big-append-prefill-plan-2026-10-13.md`, pre-flight P1/P2 and Stage 0 only.
Fully offline: 0 exo API calls, 0 boots, 0 node writes. The only cluster contact was one read-only `scp` of
`chat_template.jinja`, `tokenizer.json`, `tokenizer_config.json` and `config.json` from studio2 (macstudio-m4-2), into `tok/`.
**Labels:** MEASURED = read from a log or db · DERIVED = arithmetic or reconstruction from MEASURED inputs · BOUND = ceiling · MODELED = cost model.
**Committed** by the orchestrator as `docs/dsv41-big-append-census-2026-10-13.md` + CSV
`docs/benchmarks/big-append-census-2026-10-13/census.csv` (signed); full working set archived at
`scratch/census-largeappend/` (calls_all.csv, recon_meta.json, summary.json, p1/, demo_02.json, scripts/, tok/).

## 0. VERDICT: close the felt-latency case. All three plan-mandated immediate-surface conditions hold.

| # | Surface condition | Result | Holds? |
|---|---|---|---|
| 1 | **P2: the ≥20K bucket shrinks by ≥25 %** | Cloud bucket: 270 calls / 20.39 M rows. Token-level, exo-ideal reuse: **68 calls / 5.00 M rows (−75.5 %)**. Plan-literal (prompt vs previous prompt): 282 calls / 11.11 M rows (−45.5 %). DERIVED | **YES** |
| 2 | **§0.2: per-turn reasoning-strip re-prefill** | **The reasoning strip itself does NOT fire for Hermes.** Hermes always sends `tools`, so `drop_thinking` is forced False, both in exo's encoder and in the checkpoint's own template. Prior reasoning stays in the prompt. **But a related per-turn re-prefill is found, conditionally.** exo re-renders prior tool calls in the **V4** DSML spelling (`<｜DSML｜tool_calls>`). The checkpoint's native spelling, which exo's own dsv41 parser is keyed on, is `<｜DSML｜ calls>`. If the model decodes its native spelling, every tool-call turn diverges inside the previous assistant turn and re-prefills all of it, reasoning included. That costs **median 524 / p90 9.3K / p99 28.1K rows per turn**, and total rows are 3.5× the ideal. DERIVED. Which spelling the model emits is NOT MEASURED | **YES (conditional)** |
| 3 | **§0.4: S < 8 %** | S = **1.07 %** (exo-ideal), 2.56 % (dialect case), 5.53 % (the old cloud-row target definition). MODELED. Every variant is <8 %, and S stays <2 % for exo-ideal even if decode were 2× faster | **YES** |

**Reading.**
- Big appends are a small slice of the modeled wall time (~1 % in the exo-ideal case). Decode is ~84 % of modeled wall: 419 of 498 h, because subagents run at reasoning effort=max and produce long outputs.
- 79 % of the genuine ≥20K rows that remain are class (iii) cold rebuilds after compaction or a model switch. They come from 5 interactive sessions, are cold from depth 0, and are 22K–532K in size. They are not deep appends.
- So no engine prefill lever can move felt latency perceptibly. Per the plan's own table, this is a **closure-with-evidence** outcome.
- The one live question worth Stage-1 time is the **cache-fidelity check (1D)**: which DSML spelling does the model actually emit? That decides whether the dialect re-prefill is real. Plan rule: a fidelity defect is surfaced same day.

**Also surface (new, not one of the three triggers): depth inflation.**
- On exo, Hermes prompts are **2.38× the cloud depth** (median; p90 3.81×). DERIVED. The reason is that reasoning is retained in every prior turn.
- 232 reconstructed calls in 3 sessions exceed the 1,048,576 KV cap and would fail with `CapacityError`.
- Hermes would also compact ~2.4× sooner on exo, which adds class-(iii) cold prefills that this census does not count (it replays cloud-era history).

S0 still stands: the cluster does not serve Hermes today. Every row below is a counterfactual replay of cloud-served sessions.

---

## P1. Provenance of 234 rows/s and its reconciliation with 309 rows/s and the 22 % indexer share

**Provenance (MEASURED, `idxprof6/REPORT.md` §1 plus a re-parse of `idxprof6/raw/macstudio-m4-1.exo.slice.log` via `scripts/parse_live.py` → `out/p1/`):**

| Field | Value |
|---|---|
| Request | `b32b6571` |
| Time | 2026-10-12 21:01:35Z |
| Build | live production deploy/next23b-qsplit @ 60008939 + mlx-lm 6a5c78f4 |
| Flags | QSPLIT=1, CKPT=1, chunk 2048, maxKv 1,048,576 |
| Request settings | cold, `use_prefix_cache=false`, enable_thinking=false |
| Prompt | 500,052 rows |
| Server prefill | 2,134.5 s → **234.3 rows/s** |

It was measured on the same build and flags as the 309 figure. **Neither of the plan's hypotheses is the cause** (the figure does not predate qsplit or HIER).

**Resolution (DERIVED, from the 242 MEASURED full-chunk walls of the same run):**

| Depth (median of full 2048-row chunks) | ~105K | ~205K | ~255K | ~300K | ~490K |
|---|---:|---:|---:|---:|---:|
| s/chunk | 6.84 | 8.18 | 9.33 | 9.63 | **10.73** |
| Marginal rows/s | 299 | 250 | 220 | 213 | **191** |

1. **234 rows/s is a cumulative average over depth 0→500K, not the marginal rate at 500K.**
   - The same run's cumulative rate at 104K is 295 rows/s. That is 4.5 % under idxprof3's 309, because the first chunk pair carries overhead.
   - The marginal chunk at ~490K takes 10.73 s, not the plan's 8.75 s (2048/234).
2. **The depth increment over 105K→490K is +3.9 s/chunk (MEASURED)**, not +2.1 s.
   - The plan's linear-indexer extrapolation (1.46 s × 490/105 = 6.8 s, i.e. +5.4 s) over-predicts the MEASURED +3.9 s, even if all of the increment were indexer.
   - The sdpa/KV term also grows with depth, so the indexer must be clearly **sublinear in depth**. That matches §1A of the indexer-profile doc: the HIER cost on ratio-1 layers is ~flat (layer 24: 82.9→89.0 ms over 50→130K) plus a constant ≈30 ms exact re-score.
3. Linear fit over the whole run (MODELED): t_chunk ≈ 6.26 s + 0.98 s per 100K, but the curve is not linear:
   - 0–50K is *slower* (7.1 s) than 50–100K (6.7 s).
   - There is a step between 200K and 250K.
   - This census uses the MEASURED binned curve (25K bins), not the two rates. The two-rate variant is reported alongside for comparison.

The plan's rule stands: re-measure in Stage 1A. The census results do not depend on it (S is 1.07 % with the curve and 1.11 % with the two-rate model).

## P2. How "new rows" was computed, and the token-level recount

- **MEASURED.** `idxprof2/out/agentic_calls.csv` `new_rows` = `api_calls.input_tokens` = ollama-cloud `prompt_tokens − prompt_tokens_details.cached_tokens` (`api_calls.extra.raw_usage`). That is the cloud's server-side cache. Its token counts come from the cloud's tokenizer and template.
- **50 of the 270 calls (8.17 M rows, 40 % of the bucket) report `cached_tokens=0` on a non-first call.** These are cloud cache misses. MEASURED.
- **Recount (DERIVED).** Method:
  - Every call of the 50 sessions behind the 270 was replayed: 8,363 calls.
  - 45 sessions with no compaction and no model switch (6,390 calls) were reconstructed message by message from `messages`, using the exact Hermes replay rules (`reasoning_content` verbatim, else `reasoning`, else `" "`).
  - Each was rendered with **exo's DSv4.1 encoder `vendor/deepseek_v4_encoding.py` @ 60008939** (route: `dsv41/engine.py::_render_prompt` → `utils_mlx.apply_chat_template` → `render_chat_template` → `_needs_v4_encoding` because the model id contains "deepseek-v4"; tokens via `cache.encode_prompt`, `add_special_tokens=False`). It was tokenized with the checkpoint `tokenizer.json`.
  - Each turn was then fed through a simulator of exo's prefix reuse: dsv41 `session.py` + mlx-lm `session_cache.py` @ 6a5c78f4. The simulator models the KV holding prompt + own decode minus the last token, checkpoints at every chunk end (spacing 1024 < chunk 2048), the 512-row margin rung, prompt end and turn end, keep-32 + offset 0, and rewind to the newest checkpoint ≤ LCP.
  - The 5 compaction/model-switch sessions (1,973 calls) are classified at the api_calls level. Their counts are DERIVED in cloud tokens.

| ≥20K-row bucket | Calls | Rows | Δ rows vs cloud |
|---|---:|---:|---:|
| Cloud (`cached_tokens`), MEASURED | 270 | 20,388,459 | — |
| Plan-literal: LCP vs the same session's previous **prompt** (DERIVED) | 282 | 11,113,323 | **−45.5 %** |
| exo-ideal: LCP vs the KV after the previous turn's own decode (DERIVED) | **68** | **5,000,265** | **−75.5 %** |
| exo, model decodes the native V4.1 DSML spelling (DERIVED) | 282 | 11,113,323 | −45.5 % |
| The 270 seed calls under exo-ideal: still ≥20K | 62 | 4,290,206 (in those 270) | — |

**Why the shrink happens:**
- The cloud figure counts cache misses (40 % of rows).
- Hermes's prompt-cache misses on the cloud also count the previous turn's output as new.
- On exo, the previous turn's output (reasoning + tool block) is already in the KV.
- The literal definition counts the previous assistant turn as new (its median is 540 tokens, but it can be up to 60K of reasoning).

## §0.1 Classes of the 270 seed calls (exo-ideal view)

| Class | Calls | Cloud rows | exo rows | Still ≥20K on exo | Depth p50 / p90 (exo) |
|---|---:|---:|---:|---:|---|
| (i) true append (reconstructed) | 191 | 12.73 M | 0.95 M | 22 | 250K / 767K |
| (i) append (api-level, cloud tokens) | 53 | 5.00 M | 0.55 M | 16 | 132K / 228K |
| (ii) divergent prefix (before previous prompt end) | **0** | — | — | 0 | — |
| (iii) compaction / model-switch cold rebuild | 23 | 2.59 M | 2.75 M | 23 | 0 (cold) |
| (iii) first call | 3 | 0.06 M | 0.04 M | 1 | 0 |

Over all 6,390 reconstructed calls, class (ii) occurs in only **37 turns (0.6 %)**, none of them ≥20K. DERIVED. Totals are DERIVED. Depths are exo depths and include retained reasoning and the real tool-schema block (11.1–14.0K tokens; DERIVED offset).

**Over all replayed calls, the 68 genuine exo-ideal ≥20K calls split as:**

| Class | Calls | Rows | Share of rows |
|---|---:|---:|---:|
| (iii) cold rebuild | 29 | 3.95 M | **79 %** |
| (i) true append, reconstructed | 23 | 0.60 M | 12 % |
| (i) append, api-level | 16 | 0.45 M | 9 % |

- Size: p50 33K, p90 149K.
- Depth at start: p50 21K, p90 190K.

**Under the native-dialect decode,** all 237 reconstructed big calls become "divergence inside the previous assistant turn". That fails class (i)'s definition but sits after the previous prompt end, so it is not class (ii). Labelled (ii′) here, it holds 60 % of the 11.1 M rows, and (ii′)+(iii) hold 96 %.

**Pre-registered readings:**
- **(ii)+(iii) ≥ 30 % of ≥20K rows: YES (79 % ideal; 96 % dialect case).** Promote L1 (prefix/cache fidelity: the dialect question) and the owner-side compaction behaviour.
- **Class (iii) dominates the ideal case:** the problem is cold prefill at shallow-to-mid size (22K–532K from depth 0). **Depth levers are irrelevant.**

## §0.2 Template reasoning-strip check: exact divergence index

Two consecutive real turns were rendered: session `20260913_004543_1a76eb`, calls 1→2. The previous assistant turn has a real 1,213-char reasoning block plus 2 tool calls. Output is in `out/demo_02.json` (`scripts/demo_02.py`).

| Render path | KV after turn N | Turn N+1 prompt LCP vs KV | Divergence | Rows re-fed vs ideal |
|---|---:|---:|---|---|
| tools present, model decodes the **V4** spelling (exo-ideal) | 4,477 | 4,477 (= KV) | none: clean append | 3,343 = ideal |
| tools present, model decodes the **native V4.1** spelling | 4,469 | **4,346** | **token 308 of the 439-token previous assistant turn**: KV `</think>\n\n<｜DSML｜ calls>` vs re-render `</think>\n\n<｜DSML｜tool_calls>` | 3,782 vs ideal 3,351 (rewind to the prompt-end checkpoint 4,038; whole previous turn re-fed) |
| no tools (`drop_thinking` path; not Hermes) | 4,223 | 3,791 | token **prompt_N − 1**: KV `<｜Assistant｜><think>` vs re-render `<｜Assistant｜></think>` (reasoning dropped) | rewind to the newest checkpoint ≤ 3,791 |

**Reasoning strip.**
- exo's V4 encoder drops prior `reasoning_content` only when no message carries tools (`_encode_messages_text`: `effective_drop_thinking = False` if any message has tools).
- The checkpoint's own `chat_template.jinja` does the same: `drop_thinking = not tools`.
- Hermes always sends tools, so **no reasoning-strip re-prefill occurs for Hermes**. It is quality-correct: prior reasoning stays in the prompt, which is what the checkpoint expects.
- The no-tools path (chat/dashboard) does strip reasoning and diverges at the last `<think>` token. That is a structural per-turn re-prefill. Its cost is bounded by checkpoint granularity (≤ ~2K rows); the bench logs show exactly this signature, MEASURED: `prompt=120142 lcp=120119 reuse=118784 prefill=1358`.

**Dialect re-prefill (DERIVED over 6,345 reconstructed non-first turns).**

| Metric | Value |
|---|---|
| Turns affected | 6,311 / 6,345 |
| Extra rows per turn | p10 113 / **p50 524 / p90 9,312 / p99 28,092** / max 59,857 |
| Total fed rows | 26.8 M vs 7.63 M ideal (**3.5×**) |

- It is an exact-prefix-safe fix candidate (L1). One option is to render prior tool calls in the checkpoint's own dialect, which is also arguably the model-correct prompt. Another is to reuse up to the true divergence.
- **Prerequisite (not measured here):** which spelling the model actually emits under exo's V4-spelled tools prompt. This is a Stage-1 1D item. It can also be checked read-only by grepping a production log for `｜DSML｜ calls`. The bench tool-call battery's outputs are not in the local log copies: 0 hits for either spelling in emitted text.

## §0.3 Concurrency census (MEASURED; cloud call intervals, all 10,572 DSv4.1 calls)

| Metric | Value |
|---|---|
| Call-seconds overlapping another in-flight call | **40.1 %** (79.2K of 197.6K s) |
| Calls overlapping at least one other | 36.0 % |

**L6 kill criterion (<5 %): NOT met.** L6 stays open on this criterion. The overlap is parallel subagents. On a single-stream cluster they would queue, so this is also a capacity and routing fact.

## §0.4 Felt-cost share S (MODELED)

Model inputs:
- Prefill: MEASURED idxprof6 chunk curve, plus the idxprof2 tail model (A0 0.0823).
- Decode: round = 87 ms @105K + 0.115 ms/1K, at 2.08 tok/round. This assumes the R-B fix restores the untiled trend above 254K. idxprof8 was not read.
- Tool/human time: MEASURED inter-call gaps from state.db, capped at 300 s per gap. The uncapped variant is also shown.
- Population: the 50 sessions, 8,363 calls.

| Variant | S | ≥20K prefill (h) | Modeled wall (h) | Prefill / decode / gaps (h) |
|---|---:|---:|---:|---|
| **exo-ideal, gaps capped** | **1.07 %** | 5.35 | 498 | 19.6 / 418.7 / 60.2 |
| exo-ideal, gaps uncapped | 0.92 % | 5.35 | 581 | 19.6 / 418.7 / 142.5 |
| exo-ideal, two-rate 309/234 (flagged) | 1.11 % | 5.52 | 497 | 17.8 / 418.7 / 60.2 |
| exo, native-dialect re-prefill | 2.56 % | 13.39 | 524 | 45.1 / 418.7 / 60.2 |
| cloud-row target (old definition), cloud depth | 5.53 % | 24.81 | 449 | 45.3 / 343.5 / 60.2 |
| subagent sessions only, exo-ideal | 0.15 % | 0.53 | 365 | — |

**Sensitivity:** with decode 2× faster, S is 1.85 % (ideal), 4.26 % (dialect) and 8.95 % (old definition only).
**S < 8 % in every exo variant → SURFACE** ("the felt-latency case is weak at any achievable speed-up").

## §0.5 Owner-side lever sizing (data only; MODELED; reconstructed sessions, exo-ideal rows)

Tool-result sizes (DSv4.1 tokens; 8,065 results): p50 436, p90 1,968, p99 7,130, max 30,283.

| Hermes tool-result policy | Share of all fed rows removed | Share of ≥20K-call rows removed | ≥20K calls left (of 23) | Prefill time saved (same depth) |
|---|---:|---:|---:|---:|
| cap 8K | 5.3 % | 45.9 % | 5 | 4.1 % |
| cap 16K | 1.5 % | 19.0 % | 10 | 1.3 % |
| cap 32K | 0.0 % | 0.0 % | 23 | 0.0 % |
| by reference (256-token stub) | 73.2 % | 91.1 % | 1 | 60.8 % |

These savings are percentages of *prefill* time, which is itself ~4 % of modeled wall.
- The ≥20K mass that dominates (class iii) is compaction rebuild, not tool-result size.
- The larger owner-side levers are:
  - **compaction behaviour:** compaction output re-prefills cold.
  - **model-switch policy.**
  - **routing (O2).**
  - **depth inflation:** reasoning is retained under tools, and 2.4× depth pushes 3 sessions past the 1M KV cap.

## §0.6 Routing context

**TTFT is NOT recorded in state.db.** `api_calls` has only end-to-end latency, and `extra.raw_usage` has no timing.

| Same 270 calls | p10 | p50 | p90 | p99 |
|---|---:|---:|---:|---:|
| Cloud end-to-end latency, incl. decode (MEASURED) | 3.0 s | **11.8 s** | 89.7 s | 193.6 s |
| Cluster prefill alone for the cloud-row deltas (MODELED) | 79 s | 145 s | 667 s | 2,112 s |
| Cluster prefill for the 68 genuine exo-ideal ≥20K appends (MODELED) | 79 s | 116 s | 505 s | 2,275 s |

- **Expectation gap:** the cluster's prefill alone is ≈10× the cloud's whole call.
- Over all calls, the modeled cluster wall (498 h) is ≈11× the cloud's summed call time for the same 8,363 calls (44.7 h MEASURED). That gap is mostly decode.

---

## Caveats

- **Counterfactual.** These are cloud-era sessions replayed through exo's renderer and reuse logic. Hermes's behaviour on exo would differ: earlier compaction because of 2.4× depth, and capacity failures above 1M.
- **Tool schemas are not stored in state.db.** Renders use a 1-tool dummy. The real tool-block size (11.1–14.0K tokens) is DERIVED from call 1 (cloud prompt − render) and added to depths. It only shifts depth, not deltas.
- **Exo-ideal assumes byte-identical re-rendering of the model's own output** (args JSON re-encode, whitespace). The V4 round trip was verified exact on the reconstructed text. Any real model-output-vs-render difference adds re-prefill (L1 territory).
- **Api-level sessions (5)** use cloud token counts and cloud deltas. Their appends include the previous output, so they are slightly pessimistic.
- Decode beyond 254K assumes the R-B fix works (idxprof8 outcome not read). The cost curve is extrapolated linearly beyond 500K.
- Production `EXO_DSV41_CHECKPOINT_*` env values were assumed to be the code defaults (1024 / 512 / 32).

## Artifacts (`/home/hermes/.hermes/cache/scratch/census-largeappend/`)

- `census.csv`: one row per seed ≥20K call (270). Columns:
  - class, cloud_miss
  - cloud vs literal vs exo-ideal vs dialect new rows
  - depths
  - divergence offset
  - modeled prefill and decode
- `out/calls_all.csv`: all 8,363 replayed calls. `out/summary.json`: every aggregate above. `out/demo_02.json`: the §0.2 render. `out/p1/`: the idxprof6 re-parse.
- `scripts/`:
  - `census.py`: reconstruction, render, tokenize and simulation (~15 min).
  - `analyze.py`: aggregates.
  - `demo_02.py`: the §0.2 render.
  - `parse_live.py`: the idxprof2 copy.
  - `vendor/deepseek_v4_encoding_60008939.py`: the exact encoder exo serves DSv4.1 with.
- `tok/`: checkpoint tokenizer + template + config (scp'd read-only). sha256:
  - `tokenizer.json`: c90dfa01…
  - `chat_template.jinja`: 7d135667…
- Run with the `dsv41-ws5` venv: `/home/hermes/work/dsv41-ws5/.venv/bin/python -I scripts/census.py && … scripts/analyze.py`.
