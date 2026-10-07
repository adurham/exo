# Phase 19 — agentic-vs-benign replay @ ~100K: the decode gap is acceptance, not the round wall

Date: 2026-10-07 (afternoon)
Cluster: 2× Mac Studio M4 Max, TP2 over jaccl RDMA
Model: `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw` (engine `dsv41`, greedy-only)
**Relaunches: 0. Cluster env untouched. Read-only cluster access except the inference requests themselves.**
Harness: `bench/phase19_agentic_measure.py` + `bench/run_campaign.py` (this branch).
Raw: `docs/benchmarks/phase19-latency/raw/phase19_agentic_r100k.json`.

## Headline

The synthetic→real decode gap is **entirely spec-decode acceptance**, not the round wall.

| arm | content | prompt tok | **decode t/s** | **ms/round** | **mean accepted /3** | committed/round | rounds | reasoning/content tok |
|---|---|---|---|---|---|---|---|---|
| benign | synthetic filler + "count 1..450" | 100,042 | **24.78** | **154.14** | **2.827** | 3.827 | 254 | 71 / 971 |
| agentic | **real session 20261007_092009_9a2ed7** | 91,046 | **20.66** | **154.03** | **2.188** | 3.188 | 314 | 1000 / 0 † |

† the agentic answer hit the `max_tokens` ceiling entirely inside `reasoning_content`
(`finish_reason=length`, `content` empty) — the known trap, and still a valid decode
measurement. An extra rep at `max_tokens=2000` also spent all 2000 tokens in reasoning
(20.22 t/s), confirming the model's effort on this instruction exceeds 2000 reasoning tokens.

* **ms/round is byte-identical across arms: 154.03 vs 154.14 ms (−0.07%).** The per-round
  wall does **not** care what the content is.
* **All of the −16.6% decode gap is acceptance:** 2.188 vs 2.827 accepted drafts/round
  (× 0.774) ⇒ committed tokens/round **3.19 vs 3.83** ⇒ 20.66 vs 24.78 t/s (ratio 0.833).
  Cross-checked two independent ways: `1 + mean_accepted` and `completion_tokens/rounds`
  agree to <0.15%; `decode_tps × ms_per_round/1000` reproduces committed/round to <0.2%.
* Acceptance is reproducible to **4 decimal places** across all 3 timed reps per arm
  (deterministic greedy), so this is a real content effect, not noise.

## What this closes about the 28.08 → 16–19 t/s complaint

Two independent, additive effects — neither is the round wall and neither is sampling
(the engine is greedy-only and both arms ran at `temperature=0`):

1. **Acceptance drag on real agentic content: ×0.833 (−16.6%).** Synthetic filler is
   *pathologically easy* to draft-verify (2.83/3 near saturation). Real agentic text —
   code, terminal output, tool JSON, prose — self-speculates worse (2.19/3). With gamma
   pinned at 3, that directly costs committed tokens per round.
2. **Same-day cluster-state penalty: ×0.880 (+13.7%).** Today's benign 100K round is
   **154.1 ms**, vs **135.6 ms** recorded earlier the same day for the same task/deploy.
   Acceptance is unchanged (2.827 vs 2.813), so this is a cluster/thermal/contention
   effect, not a model effect — the cluster is simply ~13% slower today.

28.08 t/s × 0.880 × 0.833 = **20.6 t/s**, which is exactly the agentic arm.

The real session's **16.7–17.2 t/s** (seq39 = 995 tok / 59.5 s; seq40 = 1906 tok / 110.6 s)
is the same two factors plus its true ~105K context (this replay is 91K) and its real
reasoning-heavy generation: **~50 % of the real session's output tokens were reasoning**
(12,399 reasoning / 21,799 output). Reasoning tokens decode through the same speculative
path, so they are not intrinsically slower per round — but the session's effective rate is
also diluted by prefill time amortised over short turns, which this steady-state harness
does not include.

## Method (and honest limits)

* **Agentic prompt** = the session's real system-prompt preamble (`state.db`
  `system_prompts[hash=13466837…]`, 22,770 chars) + **all 98 real session messages** in
  order (content + `reasoning_content` + `tool_calls`), rendered into **one flattened user
  message** — the *same request shape* the synthetic harness uses, so only the token
  *content* differs. A unique salt is prepended, exactly as the synthetic harness does, so
  rep0 is a genuine cold prefill and later reps are prefix-cache hits.
* **Reconstruction fidelity:** the full real session was **104,880 prompt tokens** at its
  max. The content recoverable from `state.db` (system prompt + all message content)
  reconstructs to **91,046 tokens**; the residual ~25K is the Hermes tool-schema block,
  which `state.db` does not store (`sessions.tool_names` is empty for this row). So the
  replay is at 91K, not 105K — the reported `prompt_tokens` is the **actual** value from
  the response `usage` block, as required.
* Reps: 1 cold (prefill / prompt_tps) + **3 timed** per arm, `max_tokens=1000`,
  `temperature=0`, `stream=true`, interleaved arms in one harness process.
* Stats read from the `: generation_stats` **comment** SSE frame
  (`mtp_cycles_cumulative` / `mtp_accepted_drafts_cumulative`); gamma pinned at 3 ⇒
  `rounds = Δcycles/3`, `mean_accepted = 3·Δaccepted/Δcycles`, `ms_per_round =
  decode_s/rounds`. Gamma=3 re-confirmed here: 971/3.827 = 253.7 vs 254 measured;
  1000/3.188 = 313.7 vs 314 measured.
* **Harness caveat (not used in the numbers above):** the stock `derive()`'s internal
  `mc = 1 + d_acc/d_cyc` and `gamma_implied` are off by a factor of gamma
  (1.524/1.629 here). `rounds`, `mean_accepted`, `ms_per_round`, `decode_tps` are correct;
  `gamma_implied` must be read as ≈3÷1.52 = 1.97… i.e. it is not a gamma estimate at all.
  Reported for the next reader; the analysis uses the acceptance-derived committed/round.

## Idle-guard (before every rep)

Rule: `state.db MAX(ended_at) provider='custom'` > 600 s **and** the newest non-own
cluster `POST /v1/chat/completions` in `~/.exo/exo_log/exo.log` > 600 s old. "Own"
requests are matched by the local send epoch (±5 s) recorded per rep; the response
`created` field could **not** be used for matching (the engine stamps it *after* prefill,
so it is minutes late for a 370 s-prefill cold rep).

* Benign run — first guard (12:19:37): last foreign POST 11:56:41 = **1376 s** old,
  last custom state.db call 09:50:35 = **8942 s** old → `ok: true`.
* Agentic run — first guard (12:36:02): last foreign POST 11:56:41 = **2362 s** old,
  last custom state.db call 09:50:35 = **9927 s** old → `ok: true`.
* Last guard (agentic rep3, 12:38:36): foreign POST still 11:56:41 = **2515 s** old,
  state.db **10081 s** old → `ok: true`.
* No foreign traffic was ever observed mid-run; the only failures were the guard
  mis-classifying the campaign's own requests before the send-epoch fix (logged in
  `guards.json`) and one log-window-saturation false positive, both handled by
  discarding/rerunning rather than reporting through them.
