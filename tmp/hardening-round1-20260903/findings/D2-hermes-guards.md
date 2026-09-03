# D2 — hermes-agent regression-guard teeth verification

**Repo used:** `/Users/adam.durham/repos/hermes-agent` (dev clone with `.venv`/pytest,
branch `main`, remote `origin` = `adurham/hermes-agent`). This is the checkout
where the two commits under test (`fb394a378b` canonical serializer,
`bdc9b6f1fc` pad-strip) already sit at `HEAD~1`/`HEAD~2`, and where CI actually
runs (`.github/workflows/tests.yml`).

A second on-disk copy exists at `/Users/adam.durham/.hermes/hermes-agent` — the
**live running install** (an editable pip install with no dev deps/pytest;
confirmed via 3 live `hermes` processes with that path in their argv). Its
`origin/HEAD` is *ahead* of the dev clone by 2 unrelated commits
(`5cd6452577`, `ce575ff7de`, swarm-board work) but **contains** both commits
under test — the dev clone's HEAD is an ancestor of the live install's HEAD.
The live install has no pytest, so all test execution below used the dev
clone. Repo state confirmed clean (`git status --porcelain` empty) before and
after every mutation; no commits, no staging, no FORK.md edits were made.

## Summary table

| Win | Guard | Has teeth? (verified how) | Runs in CI? |
|---|---|---|---|
| Canonical serializer + golden bytes | `tests/agent/test_exo_canonical_serializer.py` (29 tests) | **YES.** Reordering `_CANONICAL_MESSAGE_KEY_ORDER` (swapped `content`/`reasoning_content`) failed 4 tests: `test_exact_bytes[multi_turn]`, `test_exact_bytes[reasoning_turn]`, `test_key_order_constant_is_frozen`, `test_wire_transport_output_matches_serializer_bytes`. Reverted; suite back to 29/29 green. | **YES** — `scripts/run_tests_parallel.py` discovers all of `tests/` by default except `tests/e2e/`, `tests/integration/`, `tests/docker/` (each has its own dedicated CI job). `tests/agent/` is not in that skip list, so it runs in `.github/workflows/tests.yml`'s `Run tests` step (`scripts/run_tests.sh`). |
| Pad-strip (cached_tokens 0→351) | `tests/run_agent/test_exo_reasoning_pad_omission.py` (16 tests) — **NOT** the serializer suite | **GUARDED**, by a *dedicated* test file, not the serializer tests. See full findings below. | **YES** — `tests/run_agent/` is likewise not in the discovery skip list. |

## WIN 1 — canonical serializer + golden bytes

**(a) Command / counts / runtime**
```
cd /Users/adam.durham/repos/hermes-agent
.venv/bin/python -m pytest tests/agent/test_exo_canonical_serializer.py -v
```
Result: **29 passed in 0.80s** (single run; re-confirmed at 0.59s and 0.73s
across other invocations in this session — no flakiness observed).

**(b) Count and specific test**
29 tests collected, confirmed by `pytest -v` listing (6 stability, 8 golden
fixture, 4 drift-detector, 11 non-exo fail-safe = 29). `TestDriftDetector::
test_reordered_fields_produce_different_bytes` exists and **PASSED**.

**(c) Teeth proof**
- Pristine `agent/exo_canonical_serializer.py` saved to `/tmp` before mutation.
- Mutation: swapped the order of `"content"` and `"reasoning_content"` in
  `_CANONICAL_MESSAGE_KEY_ORDER` (role, reasoning_content, content, ... instead
  of role, content, reasoning_content, ...).
- Result: **4 tests failed** — `TestGoldenFixtures::test_exact_bytes[multi_turn]`,
  `TestGoldenFixtures::test_exact_bytes[reasoning_turn]`,
  `TestGoldenFixtures::test_key_order_constant_is_frozen`,
  `TestGoldenFixtures::test_wire_transport_output_matches_serializer_bytes`
  (25 passed, 4 failed total).
- Restored file from the saved pristine copy (`md5` matched pre/post:
  `b7515cc52978abc0f27f813554e8411c`). Re-ran suite: **29 passed in 0.59s.**
  `git status --porcelain` empty, `git diff --stat` empty.

**(d) CI coverage**
`.github/workflows/tests.yml` runs `scripts/run_tests.sh`, which delegates to
`scripts/run_tests_parallel.py`. That script's default discovery root is
`tests/` with an explicit skip list (quoted from the source):

```
# Directories to skip during discovery — these suites require real
# external services (a model gateway, a docker daemon with a prebuilt
# image, etc.) and are run in their own dedicated CI jobs:
#
#   tests/e2e/         — .github/workflows/tests.yml :: e2e job
#   tests/integration/ — historical; legacy --ignore flags
#   tests/docker/      — .github/workflows/docker.yml ::
```
`tests/agent/` (and `tests/run_agent/`) are **not** in that skip set, so both
run in the default `Run tests` step of the `Tests` workflow (`workflow_call`,
invoked from `.github/workflows/ci.yaml`). **Confirmed: runs in CI.**

## WIN 2 — pad-strip

**(a) Implementation location**
`agent/message_sanitization.py::omits_reasoning_pad_for_provider()` (single
source-of-truth predicate, `provider in {"exo", "custom:exo"}`), consumed at
three call sites: `agent/chat_completion_helpers.py::build_assistant_message`
(the build path), `agent/message_sanitization.py::apply_reasoning_content_policy`
(replay-copy path), and `agent/agent_runtime_helpers.py::reapply_reasoning_echo_for_provider`
(reapply/fallback path). It is *also* duplicated as a last-touch invariant
inside `agent/exo_canonical_serializer.py::_canonical_message` (strips any
whitespace-only `reasoning_content` when the same predicate matches) — but
that is defense-in-depth at the wire, not the primary fix.

**(b) Existing coverage — is it the serializer tests, or something else?**
The serializer's own golden fixtures include one case, `pad_stripped_turn`,
which asserts that a `reasoning_content: " "` value is omitted from the
canonical wire bytes — but that only exercises the *serializer's* duplicate
strip logic, not the three real fix sites (build/copy/reapply). Those three
sites are exercised by a **separate, dedicated file**:
`tests/run_agent/test_exo_reasoning_pad_omission.py` (16 tests, one class per
call site: `TestExoOmitsPadOnBuildPath`, `TestExoOmitsPadOnReplayPath`,
`TestExoOmitsPadOnReapplyPath`, plus `TestOllamaCloudKeepsPad` and
`TestStrictAndUnknownProvidersUnchanged` fail-safe checks).

**(c) Teeth proof (two mutations, both reverted)**

*Mutation 1 — revert the shared predicate entirely* (`omits_reasoning_pad_for_provider`
made to `return False` unconditionally, simulating "the whole fix is gone"):
```
cd /Users/adam.durham/repos/hermes-agent
.venv/bin/python -m pytest tests/agent/test_exo_canonical_serializer.py tests/run_agent/test_exo_reasoning_pad_omission.py -v
```
Result: **19 failed, 26 passed** — failures spread across BOTH files: 14 in
`test_exo_canonical_serializer.py` (stability + golden-fixture + non-exo
fail-safe tests, because the predicate itself also gates the serializer's
no-op fast path) and 5 in `test_exo_reasoning_pad_omission.py`
(`test_exo_empty_reasoning_build_omits_key`, `test_exo_absent_reasoning_build_omits_key`,
`test_exo_empty_reasoning_copy_omits_key`, `test_exo_space_pad_copy_omitted`,
`test_exo_reapply_does_not_repad`). Restored file (md5 matched:
`6c5e77b2b8be8cabd0d391bd2f6f2ad7`), re-ran: 45/45 passed.

*Mutation 2 — narrower: revert only the `build_assistant_message` call site*
(the single-space pad line in `chat_completion_helpers.py`, restoring the
pre-fix unconditional `msg["reasoning_content"] = reasoning_text or " "`,
leaving the shared predicate and the other two call sites untouched):
```
.venv/bin/python -m pytest tests/agent/test_exo_canonical_serializer.py tests/run_agent/test_exo_reasoning_pad_omission.py -q
```
Result: **2 failed, 43 passed** — `test_exo_empty_reasoning_build_omits_key`
and `test_exo_absent_reasoning_build_omits_key` failed. Critically, **zero
serializer-suite tests failed** for this narrower, more realistic single-site
regression — confirming the serializer tests do NOT independently catch a
build-path regression; only the dedicated pad-omission test file does.
Restored file (md5 matched: `bbea6518270f8f77867507d0fdb7a88a`), re-ran:
45/45 passed.

**Verdict: GUARDED.** Pad-strip is NOT covered by the serializer tests in any
way that would matter for a real regression at the actual fix sites — it is
covered by its own dedicated suite, `tests/run_agent/test_exo_reasoning_pad_omission.py`,
which does have teeth (demonstrated against both a full-predicate revert and
a single-call-site revert) and does run in CI (same discovery mechanism as
Win 1, `tests/run_agent/` is not in the skip list).

**(d)** N/A — pad-strip is guarded, not unguarded; no new test needs writing.

## Restore discipline — final state

```
$ git status --porcelain
(empty)
$ git diff --stat
(empty)
```
All three mutated files (`agent/exo_canonical_serializer.py`,
`agent/message_sanitization.py`, `agent/chat_completion_helpers.py`) were
restored from `/tmp` pristine copies with `md5` verified byte-identical
before deleting the temp copies. Repo left exactly as found. No commits, no
staged files, no FORK.md changes.

## Acceptance assertions

1. Clone used: `/Users/adam.durham/repos/hermes-agent` (dev clone; confirmed
   live install at `/Users/adam.durham/.hermes/hermes-agent` is a separate,
   ahead-by-2-commits checkout with no pytest — used for CI-parity testing
   instead). **PASS**
2. `cd /Users/adam.durham/repos/hermes-agent && .venv/bin/python -m pytest
   tests/agent/test_exo_canonical_serializer.py -v` → 29 passed, 0 failed,
   ~0.6–0.8s. Count is 29. **PASS**
3. `test_reordered_fields_produce_different_bytes` exists and passes: **yes**. **PASS**
4. Teeth proof: reordering `_CANONICAL_MESSAGE_KEY_ORDER` failed
   `test_exact_bytes[multi_turn]`, `test_exact_bytes[reasoning_turn]`,
   `test_key_order_constant_is_frozen`, `test_wire_transport_output_matches_serializer_bytes`
   (4 failed / 25 passed). **PASS**
5. Post-restore green + clean status: 29/29 passed; `git status --porcelain`
   empty (pasted above). **PASS**
6. Pad-strip verdict: **GUARDED** — by `tests/run_agent/test_exo_reasoning_pad_omission.py`,
   not the serializer suite. Evidence: full-predicate revert → 19 failed;
   narrow single-call-site revert → 2 failed (both in the dedicated
   pad-omission file, zero in the serializer suite). **PASS**
7. CI: **yes** for both `tests/agent/` and `tests/run_agent/`, via
   `.github/workflows/tests.yml` → `scripts/run_tests.sh` →
   `scripts/run_tests_parallel.py`, whose only discovery skips are
   `tests/e2e/`, `tests/integration/`, `tests/docker/` (quoted above).
   Neither `agent/` nor `run_agent/` is skipped. **PASS**
8. Final `git diff --stat`: empty. **PASS**
