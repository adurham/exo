# DSv4.1 reasoning-budget limiter — round-batched port into the dsv41 decode path

2026-10-13 (PHASE 2 of the degeneration guard). Repo-only work: no cluster operation, no node
edits, no relaunch. Unit tests only.

## Verdict

The GENERIC MLX path already owned a reasoning-budget limiter:
`generator/generate.py::make_reasoning_budget_limiter` — a logits processor that, once a still-open
reasoning block has run for `budget_tokens` (default `0.75 x max_output_tokens`, capped at 16384),
forces the model's next sampled token to be the think-END delimiter, so a self-doubt loop cannot
consume the entire generation budget and leave the answer channel empty. The custom `dsv41` engine
deliberately bypasses the generic generator (no per-token sampler; `rounds._one_round` argmaxes a
whole DSpark draft + verify BATCH at once), so it inherited **no** such limiter.

This change ports it into the dsv41 **round-batched** greedy decode, adds a structured
`error.code='repetition_loop'` to the degeneration guard's clean-fail, and parses an optional
per-request reasoning-budget field (request value wins; shared env default; absence tolerated).

## Provenance / refs

- Base: `adurham/exo` `main` @ `22e0d6b38` (guard Phase 1 shipped + live warn-staged; see
  `docs/dsv41-degeneration-guards-2026-10-13.md`). Work in a worktree of `main`.
- Reference implementation: `src/exo/worker/engines/mlx/generator/generate.py::make_reasoning_budget_limiter`
  and its suite `src/exo/worker/engines/mlx/tests/test_reasoning_budget_limiter.py` (adapted here to
  the round-batched feed).
- The env knobs are reused VERBATIM from `batch_generate` (`_REASONING_BUDGET_FRACTION`,
  `_REASONING_BUDGET_MAX_TOKENS`), so the two engines keep ONE source of truth for the DEFAULT budget.

## Mechanism (why a round needs a different driver)

A round commits a BATCH of tokens at once. If the budget is crossed in the MIDDLE of that batch, the
generic path (one token per forward) would simply never have produced the tokens past the threshold;
here they have ALREADY been computed and their rows are in the cache. So the round must:

1. commit only the accepted tokens **up to the threshold**;
2. **discard** the remaining accepted tokens **and the round's bonus token**, rolling the cache back to
   the committed prefix; and
3. make the **NEXT** committed token the think-END delimiter.

`dsv41/reasoning_budget.py::ReasoningBudgetLimiter.feed(batch)` decides the cut (a PURE function of
the committed ids + phase) and latches `next_force`; `rounds._one_round` enacts it:

- **cut**: `committed = limiter.feed(committed)`, then `spec.rollback(cache, snapshot, position +
  len(committed), stashes)` and `head.append_ctx(tapcat(taps)[:, : len(committed)])` — i.e. both the
  body cache and the DSpark draft window follow the SHORTENED prefix (previously `accepted + 1`).
- **force**: the round AFTER the cut is a plain single-row greedy STEP (no speculative draft) whose
  output is overridden to `think_end_id`. That is the **cold start** the spec requires (the draft window
  conditions on target hidden states, and the cut round was truncated mid-batch). The forced token's OWN
  KV is deliberately NOT computed in that round — the FOLLOWING round feeds it as its anchor (an
  off-by-one there would corrupt the stream silently). The forced round still advances the draft window by
  one row so it stays in step with the cache.

Phase initialisation: from the RENDERED prompt (`dsv41/output.py::_starts_in_thinking`), never inferred
from seeing markers in output — DSv4.1's template appends a literal ` thinking` suffix, so the model never
generates an open marker of its own. The anchor (the first GENERATED token) is fed to the limiter before
the first round, exactly as the generic limiter counts it.

Purity / rank agreement: `feed` reads only the committed ids, which are byte-identical on both TP ranks
under greedy, so both compute the same cut and the same force — no collective, no new hang class. For
that reason the generic path's wall-clock `max_seconds` trigger is **deliberately NOT ported**: a
rank-local clock would let one rank force while its peer kept decoding, wedging the peer in the model's
own TP `all_sum`.

## Structured error code

`degeneration.REPETITION_LOOP_ERROR_CODE = "repetition_loop"`. The guard's clean-fail terminal is a
`_final_response(reason="error", text=DEGENERATION_ERROR_TEXT, error_code=...)`; `error_code` is threaded
onto `GenerationResponse.error_code` (new optional field) and copied by `map_responses_to_chunks` onto
`ErrorChunk.error_code` — the same plumbing the `context_length_exceeded` refusal already uses. A client
can classify a degeneration kill WITHOUT string-matching the human message; the message is unchanged.

## Per-request reasoning budget

`reasoning_budget_tokens: int | None` added to `ChatCompletionRequest` (wire) and
`TextGenerationTaskParams` (internal), passed through by the chat-completions adapter.
`reasoning_budget.budget_tokens_from_request(request_value, max_tokens)` prefers a positive request value
and otherwise falls back to `min(max_tokens * EXO_REASONING_BUDGET_FRACTION, EXO_REASONING_BUDGET_MAX_TOKENS)`.
Absence is tolerated (`None` => default); `<= 0` (knob disabled) => limiter is `None`. Bench requests skip
the limiter (parity with the generic call site).

## Pitfall coverage (explicit)

| pitfall | handling |
| --- | --- |
| commit only up to the threshold, discard the rest + the bonus | `feed` returns a strict prefix; `committed_position = position + len(committed)` shortens the rollback |
| think_end as the NEXT token | the following round is a plain step overridden to `think_end_id` |
| think_end's KV = first input of the NEXT round, not already-cached | the forced round does NOT feed it as an output row; the round after feeds it as its anchor (test: `test_think_end_kv_is_computed_by_the_following_round`) |
| rollback DSpark draft KV + hidden to the committed prefix, discard pending drafts | `head.append_ctx(tapcat(taps)[:, : len(committed)])`; no pending draft state exists (drafts are per-round), and the first post-force round is a cold start |
| decision a pure function of committed count + phase; never a rank-local timer | `feed` reads only committed ids; `max_seconds` NOT ported |
| phase initialised from the rendered prompt | `_starts_in_thinking(prompt, tokenizer)`; the anchor is fed first |
| natural think_end in the same round BEFORE the threshold => do not force | `feed` admits a natural close and never latches (`test_a_natural_close_before_the_threshold_never_forces`) |
| think_end is a SINGLE token id | `safe_think_token_id` collapses a >1-token delimiter to `None` => limiter disabled (test) |
| test re-open of think, natural close before threshold, multi-turn prior think blocks | `TestFeedPhase` re-open/close cases + `test_a_following_request_still_applies_the_limiter` |

## Files changed

| file | change |
| --- | --- |
| `src/exo/worker/engines/mlx/dsv41/reasoning_budget.py` | **new** — the round-batched limiter (`ReasoningBudgetLimiter`, `make_reasoning_budget_limiter`, `budget_tokens_from_request`) |
| `src/exo/worker/engines/mlx/dsv41/rounds.py` | `_one_round` cuts the batch / forces the round; `_final_response` gains `error_code` |
| `src/exo/worker/engines/mlx/dsv41/engine.py` | build + feed the limiter in `_generate`, thread it into `_rounds`, set the guard's structured code |
| `src/exo/worker/engines/mlx/dsv41/degeneration.py` | `REPETITION_LOOP_ERROR_CODE` |
| `src/exo/shared/types/worker/runner_response.py` | `GenerationResponse.error_code` (optional) |
| `src/exo/worker/runner/llm_inference/model_output_parsers.py` | copy `error_code` onto the `ErrorChunk` |
| `src/exo/shared/types/text_generation.py`, `src/exo/api/types/api.py`, `src/exo/api/adapters/chat_completions.py` | `reasoning_budget_tokens` field (wire -> internal) |
| `src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_reasoning_budget.py` | **new** — the limiter suite (36 tests) |
| `src/exo/worker/engines/mlx/dsv41/tests/conftest.py` | `FakeTokenizer` exposes `think_start_id`/`think_end_id` (wrapper parity) |
| `src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_engine.py` | `_engine` helper gains `enable_thinking` / `reasoning_budget_tokens` / `bench`; the `max_output_tokens=2` cap test is made a NON-thinking turn (with thinking on, `budget = int(2*0.75) = 1` means the limiter correctly forces after the first token — not what that test is about) |

## Test evidence

Command (this machine, CPU-only, no checkpoint, no GPU):

```bash
cd <worktree>
EXO_DASHBOARD_DIR=<worktree>/dashboard/build \
PYTHONPATH=<worktree>/src:<worktree>/mlx-lm \
/Users/adam.durham/repos/exo/.venv/bin/python -m pytest \
  src/exo/worker/engines/mlx/dsv41/tests \
  src/exo/worker/tests/unittests/test_runner/test_loop_detection.py \
  src/exo/worker/engines/mlx/tests/test_reasoning_budget_limiter.py \
  -q -p no:cacheprovider
```

- **326 passed** = 231 dsv41 (baseline) + **36 new** + 32 generic loop-detection + 27 generic
  reasoning-budget.
- Whole `src/exo` tree WITHOUT the dsv41 dir: **1401 passed, 35 skipped, 208 deselected** (the shared-type
  additions — `GenerationResponse.error_code`, `reasoning_budget_tokens` — did not regress anything).
- Gates: `ruff check` → clean on every touched file; `ruff format --check` clean on every file this change
  touches or adds. `dsv41/rounds.py` and `dsv41/tests/test_dsv41_engine.py` still show ruff-format drift,
  but that drift is **PRE-EXISTING** (verified against a clean checkout of the base): they already
  reformat at `22e0d6b38`, and this change does not add to it.
- `basedpyright` was not run (not part of the acceptance gates).

What the new suite covers:
- factory/no-op contract (missing ids, `<= 0` budget, >1-token delimiter => None), and that
  `max_seconds` is NOT a parameter;
- budget resolution (request wins / default / capped / knob disabled);
- the phase machine over the committed stream: under/at/over the threshold, exact-boundary, natural close
  before vs after the threshold, closed block, RE-OPENED block with prior think history, purity/rank
  invariance;
- the round-level cut (shortened batch, shortened cache rollback + draft append) and the forced round
  (one plain step, no draft, commits `think_end`, phase closed);
- engine end-to-end: `think_end` at the budget point, content following it, the off-by-one KV fact, the
  request field lowering the point, absence using the default, bench skipping the limiter;
- the guard's `ErrorChunk.error_code == "repetition_loop"`.

## Residual risks / not verified

- **No live-cluster validation** (by instruction): the port is unit-verified only. The staged
  warn->armed guard rollout is now done separately (`main` @ `4e7b0183a`: stage-2 ARMED, `action=error`,
  verified on both ranks) — the structured `repetition_loop` code added here rides that armed path.
- **The reasoning-budget limiter's in-situ behaviour on TP2** (both ranks cutting + forcing together) is
  argued from purity + byte-identical committed ids, not measured live.
- **The draft window is not re-primed from scratch after a cut.** The post-force round is a cold start
  (it drafts nothing), and the window is rolled to the committed prefix; but the DSpark head's context KVs
  for the kept rows come from a truncated verify forward whose hiddens differ slightly from a fresh 1-row
  forward's (the documented body chunk-shape effect). This is a hint-only inconsistency: the target
  verifies every draft, so it cannot corrupt output.
- **The limiter is token-only.** A very slow/degraded decode can therefore take a long wall-clock time to
  reach its token budget; the generic path's `max_seconds` backstop was rejected on purpose (rank-local
  time is a hang class here). If a wall-clock backstop is ever wanted on dsv41 it must be driven from a
  rank-agreed quantity, not a per-rank clock.
- **`reasoning_budget_tokens` is not yet sent by any client.** The Hermes side is being taught to send a
  cap in a parallel task; until then the env default governs.
