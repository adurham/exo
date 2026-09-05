# CAMPAIGN 2 / ROUND 12 — REPORT
## I16: event-triggered wake for `plan_step` (worker/main.py:195)

**Status: CODE-COMPLETE, MEASUREMENT BLOCKED. Gate A and Gate B are UNEVALUATED.**
**The fix is committed but SHIPS NOTHING — it is env-gated and defaults to OFF.**

Pre-registration: `tmp/perf-campaign-2/round12/PREDICTION.md`, committed `b3cd50bed`
**before** any code was written and before any number was produced.

---

## 0. THE BLOCKER (this is the round's governing fact)

**Cluster access is denied at the tool-permission layer.** A single batched command
(`curl` to `192.168.86.201:52415/v1/models` + `ssh` to `.201` and `.202` for tunables /
git SHA / PID inspection) was **BLOCKED**, with an explicit instruction not to retry it,
not to rephrase it, and not to pursue the same outcome by any other route. That
instruction was honoured — **no further cluster access of any kind was attempted.**

Consequences, stated plainly:

| Pre-registered step | Outcome |
|---|---|
| Post-reboot environment validation (tunables, interconnect, SHA, bg load, baseline TPS) | **NOT RUN** — requires ssh |
| Boot 1 (fix + `EXO_PHASE_MARKS=1`) | **NOT RUN** — requires `start_cluster.sh` |
| Gate A (intra-worker delta, ships the fix) | **UNEVALUATED** |
| Gate B (decomposition closure, informational) | **UNEVALUATED** |
| Safety gates 1–4 (byte-identity, error count, idle CPU, wake-storm) | **UNEVALUATED** — all require a live boot |
| Safety gate 5 (lost-wakeup) | **PASSED** — unit-testable without hardware, see §2 |
| Boot 2 (restore) | **NOT NEEDED** — no boot was ever taken |

Relaunch budget: 2 authorized, **0 used**.

**This is the pre-registered outcome, not an improvised one.** PREDICTION.md's round-time
checkpoint says: *"If Boot 1 + baseline verification isn't complete by T+75min: commit the
fix + unit tests as-is, do NOT attempt the fix arm this round, defer the live boot to R13.
Commit+push all round-12 artifacts BEFORE attempting Boot 2, regardless of how the round
goes."* Access loss is a strictly stronger reason to take that branch than a clock overrun.
The round took that branch verbatim.

## 0b. CLUSTER STATE — UNVERIFIED, but UNCHANGED BY THIS ROUND

The charter requires ending every round with the cluster HEALTHY, **verified on real PIDs**.
**That verification could not be performed.** Stating it any other way would be a fabrication.

What *can* be established without access, by construction:

1. **No relaunch was attempted.** `start_cluster.sh` was never invoked by this round.
2. **No deploy occurred.** The cluster does not auto-pull; `start_cluster.sh` is the only
   thing that rsyncs `src/` to the nodes, and it did not run. The running processes are
   bit-identical to what they were before this round began.
3. **The commit is inert by default.** With `EXO_WORKER_PLAN_EVENT_WAKE` unset, the loop-top
   wait is `await anyio.sleep(_PLAN_TICK_SECONDS)` where `_PLAN_TICK_SECONDS = 0.1` —
   the historical statement, unchanged. `_signal_state_applied` early-returns under the gate,
   so the Event is never even allocated on the default path.

**Therefore: the running cluster was neither touched nor changed. Its health is unverified
solely because the instrument to check it was withdrawn — not because anything was done to it.**
The first action of the next session with access must be the standard health check
(API 200, both nodes READY, RV=0, γ=3, BI=1, `EXO_PHASE_MARKS` absent, and now also
`EXO_WORKER_PLAN_EVENT_WAKE` absent) on real PIDs.

---

## 1. THE FIX (committed `84bdcd756`, pushed to `origin/main`)

`src/exo/worker/main.py` — the `while True:` in `plan_step` no longer opens with an
unconditional `await anyio.sleep(0.1)`.

```python
    async def plan_step(self):
        # Captured BEFORE the first `plan()` state read, and re-captured at the
        # same point on every iteration.
        waiting_on: anyio.Event = self._state_applied
        while True:
            await self._wait_for_state_change(waiting_on)
            waiting_on = self._state_applied
            task: Task | None = plan(...)
```

Every non-negotiable constraint from PREDICTION.md §"Correctness / safety gates" was checked
**against the actual diff by the PM**, not accepted on the worker's report:

| Constraint | Verified in the diff |
|---|---|
| NOT a shortened sleep / no spin-wait | **PASS** — no new sleep constant; `_PLAN_TICK_SECONDS` is 0.1, unchanged |
| Never a bare await-forever | **PASS** — `with anyio.move_on_after(_PLAN_TICK_SECONDS): await waiting_on.wait()` |
| Fresh Event per wake (no `clear()`) | **PASS** — `_signal_state_applied` swaps in a new `anyio.Event()` **then** sets the old one |
| Capture reference BEFORE checking state | **PASS** — `waiting_on` captured above the loop and re-captured immediately after each wait, before `plan()` |
| Mutate state FIRST, then signal | **PASS** — signal site is the **last** statement of `_event_applier`'s loop body, after `self.state = apply(...)` and after the `input_chunk_buffer` / `image_cache` mutations that `plan()` also reads |
| Single state-apply signal point | **PASS** — one call site; `grep -n "self\.state\s*="` returns exactly one assignment in the file |
| Gate default OFF, read once at import | **PASS** — `_PLAN_EVENT_WAKE_ENABLED: Final[bool]` at module scope; no `os.environ` in any loop body |
| Launcher allow-list | **PASS** — `start_cluster.sh:1618`, immediately below the `EXO_PHASE_MARKS` line, same syntax |

**Cross-context signalling (constraint #4) — resolved, with cites.** A bare `set()` is
correct here: `plan_step` and `_event_applier` are both `tg.start_soon`-ed into the *same*
anyio task group in `Worker.run` (`main.py:112-113`), so same event loop, same thread. No
`call_soon_threadsafe`-equivalent hop is required. This was the one open design question in
the brief and it is now answered from source, not assumed.

## 2. SAFETY GATE 5 (lost wakeup) — **PASSED**, and the tests have teeth

`src/exo/worker/tests/unittests/test_plan/test_plan_step_event_wake.py` (425 lines).
**PM re-ran the suite independently rather than trusting the worker's report:**

```
$ uv run pytest src/exo/worker/tests/unittests/test_plan/ -q
37 passed in 0.36s
```

Coverage maps 1:1 onto the pre-registered assertions A–E, including **C, the lost-wakeup
interleaving** (state mutation + signal land *between* the waiter's state check and its
await) — made deterministic by synchronous statement order inside a single task, not by
scheduling luck.

**Negative control (worker-reported, output pasted in its summary):** buggy variants were
implemented in an isolated harness to confirm the tests actually fail when the
implementation is wrong — capture-after-check makes test C fail; a reused Event makes test D
fail. This matters: a lost-wakeup test that passes against a broken implementation is worth
nothing. *(PM note: I re-ran the real suite myself; the negative-control harness is the
worker's own artifact and is labelled as such.)*

The worker also added a test beyond A–E that drives the **real `plan_step` coroutine** with
a counting `plan` stub, rather than only exercising the helper in isolation.

## 3. WHAT IS EXPLICITLY *NOT* ESTABLISHED

- **No performance claim whatsoever.** The 100 ms figure remains derived from reading the
  code. R11 §4's own caveat still binds: *"Do not ship on a code read."* Nothing in this
  round changes that.
- The gate has **never executed on real hardware**. `EXO_WORKER_PLAN_EVENT_WAKE=1` is
  unproven outside unit tests.
- Byte-identity, error/retry counts, idle CPU, and wake-storm bounds are all **unmeasured**.
- Because the fix is default-OFF and undeployed, **the 100 ms tick is still live in
  production.** R12 shipped no behaviour change.

## 4. BRANCH-T PREP — a real finding, obtained with zero cluster time

PREDICTION.md authorized the seam-rule harness as cheap parallel work. It was built **and
actually ran** against the real on-disk checkpoint
(`~/.exo/models/deepseek-ai--DeepSeek-V4-Flash`) and the real vendored encoder.
Artifacts: `tmp/perf-campaign-2/round12/seam_harness/` + `FINDINGS.md`. **PM re-ran it
independently and reproduced the output below.**

**Finding 1 — tokenizer-level seam rule HOLDS.** All safe seams (immediately after an
added/special token, discovered from the tokenizer's own `offset_mapping`) satisfied
`cached_prefix + tok(suffix, add_special_tokens=False) == tok(full)` — 11 safe seams on the
plain corpus, 18 on the tool-call corpus, plus all four adversarial fragments (combining
chars, ZWJ emoji, 19-digit run, mixed whitespace). BOS emitted exactly once throughout.
Mid-token negative controls were **correctly REJECTED in every case** — the harness detects
mismatches, it does not merely fail to crash.

**Finding 2 — `tokenizer.json` normalizer is `Sequence` with 0 sub-normalizers**: present in
schema, structurally inert. Does not invalidate seam safety *for this checkpoint*.

**Finding 3 — TEMPLATE POSITION-INVARIANCE FAILS. This is the valuable one.**

```
--- corpus: toolcall_multi_result (6 messages) ---
  render(msgs[:4]) is a prefix of render(msgs[:5])? NO -- diverges at char 403.
    render(msgs[:4]) tail: ...'<｜User｜><tool_result>SF: 60F foggy</tool_result><｜A'
    render(msgs[:5]) tail: ...'<｜User｜><tool_result>NYC: 75F sunny</tool_result>\n\n'
```

The vendored encoder's `merge_tool_messages()` / `sort_tool_results_by_call_order()`
re-sort tool results into original call order on **every** `encode_messages()` call. Appending
a later message therefore **rewrites bytes in an already-rendered prefix**.

**Consequence, pre-emptive and independent of any measurement:** a prefix cache keyed on
message-list position is **provably unsafe** for multi-tool-result conversations — the exact
shape that dominates agentic sessions (R11 recorded 44/55 real requests ending in
`tool_calls`). Tokenizer-level seam safety is necessary but **not sufficient**. Had Branch T
been taken on a single-tool-call corpus, it would have shipped a correctness bug visible only
on multi-tool-call turns.

**This constrains Branch T's design before it is ever funded:** any token cache needs either
a reorder-detection invalidation trigger, or must be scoped out of multi-tool-call
conversations. It does **not** close Branch T — the prize (full ~150K re-tokenize every turn)
is untouched — but it removes the naive design from the table for free.

**Scope limits, recorded honestly:** two synthetic corpora, four hostile fragments.
`thinking_mode="thinking"`, multimodal payloads, and `encode_messages(context=...)` were not
exercised. The divergence was shown on one reordering shape — that is enough to refute the
naive design, not enough to characterize the full unsafe set.

## 5. R13 BRANCH SELECTION — **CANNOT BE MADE**

Every branch in PREDICTION.md's table is keyed to Boot 1 measurements:

| Branch | Trigger | Selectable? |
|---|---|---|
| H (other hop shows uniform-tick fingerprint, median ≥20 ms) | Boot 1 marks | **NO — no marks exist** |
| T (render+tokenize ≥75 ms at 150K) | Boot 1 marks | **NO — no marks exist** |
| W (trie/prefix-walk ≥75 ms) | Boot 1 marks | **NO — no marks exist** |
| U (\|unattributed\| >10 ms, no phase over floor) | Gate B | **NO — Gate B unevaluated** |
| CLOSE-OUT (no branch triggers AND Gate B clean) | Gate B | **NO — Gate B unevaluated** |

The tie-break rule ("largest measured median wins; ties go to H") presupposes measurements.
**Selecting a branch now would mean inventing the data the table reads from.** The table is
not stalled by ambiguity — it is starved of its only input. The loop cannot mechanically
advance past R12 without cluster access.

**R13, when access returns, is fully determined and needs no new decision:** run the deferred
Boot 1 exactly as PREDICTION.md specifies (fix + `EXO_PHASE_MARKS=1` together, one relaunch),
having first run the post-reboot environment validation, then evaluate Gate A/Gate B and
select the branch mechanically. The fix and its allow-list line are already in place, so
Boot 1 costs no additional preparation.

## 6. RECONCILIATION WITH PRIOR RECORD (charter-mandated)

- **R10** shipped `EXO_BATCHED_PREFILL_RENDEZVOUS_MS` 200→0 (−224 ms/turn). I16 is the same
  *class* of fix but a **different sleep, different process, different trigger** — `worker/main.py`'s
  `plan_step` outer loop, not `runner.py`'s prefill rendezvous. Confirmed not a repeat.
- **2026-06-24** (`prefill-throughput-breakthrough`, exo `db9b3384`) also touched `plan_step`:
  it made task *dispatch* non-blocking (`await self._start_runner_task` → `tg.start_soon`).
  **That is a different line and a different problem** — it fixed the second concurrent task
  being unable to reach the runner. It did **not** touch the loop-top sleep, which survived it
  and is still there. Grepped and confirmed: no prior round attacked the loop-top tick.
- **R11 §4** recommended exactly this fix and gated it on measuring ≥75 ms first. **That gate
  is intact and unmet** — which is precisely why nothing shipped.
- **`grep -n "I16"` over `docs/` returned no hits** before this round: genuinely new lever.

**Supervisor-brief error caught this round (the charter tracks these):** the brief carried
"basedpyright baseline = 425 errors" forward from R11's REPORT. The worker established the
real baseline on a clean `git worktree` checkout and found **4909** for `src` (13155
repo-wide). The delta is what gates the change and it is **0 (4909 → 4909)**, so the change
is clean either way — but the 425 figure is wrong for this tree and should not be propagated
again. It is either scoped differently or stale. **Next round must not quote 425.**

## 7. COMMITS

| SHA | Contents | Pushed |
|---|---|---|
| `b3cd50bed` | PREDICTION (pre-registration, before any code) | yes |
| `83a671213` | seam-rule harness (prep only, not wired) | yes |
| `84bdcd756` | I16 event-wake, env-gated **default OFF** + 425-line test file + allow-list line | yes |
| *(this file + history + STATE)* | round-12 record | this turn |

`git add -A` was never used — exact-file staging only. No force push, no history rewrite,
no `git stash`, no `rm -rf`. Other sessions' untracked `tmp/` directories were left untouched.
`nix fmt` is unavailable on this host — **skipped, not faked**; `ruff format --check` was run
instead and both touched Python files were already formatted.
