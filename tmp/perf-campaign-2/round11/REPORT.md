# CAMPAIGN 2 / ROUND 11 — REPORT
## Decomposing the ~0.55 s per-request FIXED cost (READ + INSTRUMENT, no ship)

**Status: PARTIAL — Task 0 and Task 1 COMPLETE. Task 2 (instrumented boot) NOT RUN — deferred.**
The round was cut short by an operator request to reboot the control host. The instrumentation
code is written, gated, verified inert, and committed; **no instrumented boot was ever started**,
so the cluster was never taken off production config at any point in this round.

Pre-registration: `tmp/perf-campaign-2/round11/PREDICTION.md`, committed `76294c3d4`
**before** any number was computed or any source file was read.

---

## 0. CLUSTER STATE (verified, end of round)

Verified on **real PIDs**, not asserted:

| Check | Result |
|---|---|
| API | `http://192.168.86.201:52415/v1/models` → **HTTP 200** |
| `EXO_BATCHED_PREFILL_RENDEZVOUS_MS` | **0** (R10's ship, intact) on PIDs 16063/16064/16065/16075 |
| `EXO_SPECULATIVE_GAMMA` | **3** |
| Batched inference | `EXO_DSV4_BATCHED_PREFILL=1`, `MLX_GEMV_BATCH_INVARIANT=1`, `MLX_STEEL_BATCH_INVARIANT=1`, `EXO_DSV4_VERIFY_BATCH=1` |
| `EXO_PHASE_MARKS` on running PIDs | **ABSENT** — instrumentation is not live |
| Probe vars (`*_PROBE`, degen) | **ABSENT** |

**The cluster is HEALTHY on production config and was never relaunched this round.**
Relaunch budget: 2 authorized, **0 used**.

---

## 1. TASK 0 — regression of the 55 residuals on context size (COMPLETE)

Script `task0_regression.py`, results `task0_regression.json`, writeup `TASK0-REGRESSION.md`
(commit `93655896d`). Interpreter `/usr/bin/python3`.

### Validation gate passed first
The residual was **recomputed from raw fields**, not read from a stored column:
`residual = wall_client − (prompt_tokens − cached_tokens)/prompt_tps − completion_tokens/generation_tps`.
Recomputed median/min/max **0.9432 / 0.7515 / 1.3223** vs published **0.94 / 0.75 / 1.32** —
max deviation 0.003 s. The identity reproduces; the regression is on a residual we can rebuild.

### Results

| Fit | n | Slope (µs/tok) | 95 % CI | r² | Intercept (s) |
|---|---|---|---|---|---|
| **(A) PRIMARY** residual − transit vs prompt_tokens (no cold) | 54 | **0.68** | **[−0.45, 1.82]** | **0.027** | **0.681** |
| (A) with cold outlier | 55 | 0.86 | [−0.24, 1.97] | 0.044 | 0.653 |
| (B) raw residual vs prompt_tokens (no cold) | 54 | 1.59 | [0.28, 2.90] | 0.102 | 0.739 |
| (B) with cold outlier | 55 | 1.67 | [0.41, 2.93] | 0.118 | 0.726 |

Theil–Sen agrees with OLS in sign and magnitude but does not tighten the interval.

### VERDICT: **MIXED — the naive regression cannot rank the mechanisms.**

The 95 % CI on the primary slope **spans both** the flat band and the pre-registered
1–2 µs/token band. It excludes neither hypothesis. What *is* robust:

> **A large, stable intercept of 0.65–0.78 s persists across every fit variant.**
> A real, substantial, **context-INDEPENDENT** residual floor exists. That floor — not the
> slope — is the thing worth attacking.

### Collinearity (pre-registered as a rejection criterion)
**Pearson r(prompt_tokens, cached_tokens) = 0.997.** As pre-registered, the trie-walk
(∝ cached) and tokenization (∝ prompt) hypotheses are **NOT separately identifiable** on this
dataset. Any claim to have told them apart from these 55 points must be rejected.

### Prediction scored: 3 of 4 sub-claims correct, and **the one that failed is the important one**

| Pre-registered sub-claim | Outcome |
|---|---|
| Raw-fit slope in 0.5–2.0 µs/tok | **CORRECT** (1.59) |
| Raw-fit r² < 0.35 | **CORRECT** (0.102) |
| Intercept 0.4–0.8 s | **CORRECT** (0.739) |
| **Ex-transit fit r² rises above 0.4** | **WRONG** — r²(A) = 0.027, *lower* than r²(B) = 0.102 |

PREDICTION.md §1 named this exact outcome as its own falsification criterion: *"If r² does
**not** rise, that is real evidence the residual floor is dominated by fixed IPC/polling ticks
rather than context-scaled work."* **That criterion was met.** Removing the transit noise did
not reveal a context-scaled signal underneath — it removed variance that was *correlated with*
the weak apparent slope. This is honest evidence tilting toward a **fixed tick budget**, and it
is exactly what the code read then found.

---

## 2. TASK 1 — the c=1 request path, end to end (COMPLETE)

Full document: `CODE-READ.md` (commit `4f2e555a0`), 204 line cites, machine-validated for
existence and range. **The PM independently re-verified every load-bearing cite below by
direct grep/sed** — they are not taken on the worker's word.

### (a) `server_received_ts` is stamped in the RUNNER, not the API — the "transit" is not network

`src/exo/worker/runner/runner.py:563` is the **only** emitter of `received chat request` in the
tree (PM-verified: `grep -rn` returns exactly one non-binary hit).

**Already done before the stamp:** HTTP body fully read → json parsed → pydantic validated
(`src/exo/api/main.py:1129-1131`) → all ~55 messages walked and `model_dump`ed
(`chat_completions.py:144-147`) → command gossiped (`api/main.py:1087` → `:2646-2651`) →
master indexed it (`master/main.py:268-279`, `:646-671`) → worker's **100 ms poll** picked it up
→ mp.Queue IPC into the runner process.

**After the stamp:** template render → tokenization → trie walk → KV restore → prefill → decode → SSE.

**Consequence for the study:** the 0.191 s median "client→server transit" is **not LAN latency**.
It is client serialization + a full gossip round-trip + a poll tick. Reviewer item #7 is
**confirmed for HTTP/json/pydantic** (they hide inside transit) and **refuted for tokenization**
(which is after the stamp, and therefore genuinely inside the residual).

### (b) Poll ticks on the c=1 critical path — **the R10-shaped find**

| Site | Tick | On the c=1 path? |
|---|---|---|
| **`src/exo/worker/main.py:195`** — `await anyio.sleep(0.1)` at the **top** of `plan_step`'s `while True` | **100 ms** | **YES — 100 ms worst case, ~50 ms expected, on EVERY request** |
| `src/exo/worker/runner/runner.py:580-604` — R10 rendezvous | 0 (shipped `start_cluster.sh:145`) | dead branch |

PM-verified by reading `worker/main.py:193-196`: the sleep is the **first** statement inside the
loop, so the task must wait for the next tick after landing in state. Nothing wakes it early.
exo's own source already documents this: *"a plain 100ms poll loop, confirmed by reading it —
NOT event-triggered"* (`src/exo/api/main.py:850-851`).

**This is a live, unfixed, never-instrumented, size-independent tick — structurally identical to
the rendezvous sleep R10 removed for −224 ms/turn.** ~25 other sleeps were enumerated and
separated into background/idle/error-only lists.

Also flagged, not yet quantified: every hop is JSON-gossiped with `PublishPolicy.Always`, and the
API only accepts events echoed back from the **master** — a master on node 2 doubles the hops.

### (c) Tokenization: full re-tokenize AND full re-render, every single turn

- One unconditional `tokenizer.encode()` over the whole prompt (`batch_generate.py:2194-2195` → `cache.py:2293`).
- **No tokenization-level cache anywhere** — `lru_cache` / `@cache` / `_token_cache` all return zero hits.
- HF **fast (Rust)** tokenizer, single-string `encode`, no `encode_batch`.
- **Node 2 tokenizes independently** — no multi-MB token array crosses the wire. Good.
- DSv4 bypasses Jinja for a vendored pure-Python encoder (`deepseek_v4_encoding.py:598`) that
  re-merges and re-sorts all messages per turn.
- Trie keys on **token ids → numpy**, per-edge compare is **vectorised** — so reviewer item #2's
  "Python per-token walk over 150 K ids = 50–150 ms" is **structurally overstated**.

### (d) KV restore is **LAZY** — the pre-registered TRAP bites, and it kills the #2 candidate

`_materialize_cache_to_depth` (`cache.py:1705-1787`) uses `_detached_copy` (a documented
"near-free lazy alias", `cache.py:150-160`), `mx.concatenate`, and slicing — **no `mx.eval`**.

**PM-verified directly:** all four `mx.eval` occurrences in `cache.py` are at lines 189, 253
(docstring prose), 267, 293. An `awk` scan of lines 1157–1790 — spanning both `get_kv_cache`
and `_materialize_cache_to_depth` — returns **zero** `mx.eval` occurrences.

> **Restore cost is deferred into the first prefill eval → it lands in `prefill_uncached`,
> NOT in the residual. Chasing KV restore as a residual term is chasing a phantom.**

Exception: non-sliceable layers take a per-layer `deepcopy` (`cache.py:1762`, PM-verified),
which *is* eager Python. Restore is a per-layer loop, ≈ `L × (2E + 6) + S × deepcopy` Python ops.

### (e) `_save_prefix_cache` runs **inside `submit()`** — pre-first-token, not a tail cost

PM-verified at `batch_generate.py:2586-2601`: the call sits under `with T("submit.save_prefix_cache")`,
**after** prefill but **before** the sampler is constructed and before the task is registered as
active. Decode has not started. Reviewer item #5's escape clause ("unless `done` waits on cache
commit") is **refuted as stated** — but the cost is real and lands in exactly the pre-first-token
window the study charges to residual. It must be labelled `cache_commit_pre_first_token`, never
`cache_committed`, or the phase ordering will read backwards.

### (f) `generation_tps` is wall-clock — first-step warm-up is already inside `decode`

PM-verified: the vendored mlx-lm `BatchGenerator` has **no `_stats` attribute**, so
`_mlx_gen_elapsed_seconds` (`batch_generate.py:471-484`) always falls through to
`return time.perf_counter()`. `generation_tps` therefore divides by **wall time from end-of-submit
to done**, which **includes** the first decode step, graph warm-up, first Metal alloc, and the
deferred-restore eval. **Reviewer item #4 is refuted** — that warm-up is not a residual target.

### Reviewer's ranked prediction, scored against the code read

| # | Predicted | Verdict from the read |
|---|---|---|
| 1 | Tokenization 150–400 ms | **PLAUSIBLE, UNMEASURED** — confirmed to run in full every turn, after the stamp, inside the residual |
| 2 | Trie + KV restore 30–300 ms | **LARGELY REFUTED** — restore is lazy (pays in prefill); trie compare is vectorised, not per-token Python |
| 3 | IPC / polling ticks 20–200 ms | **CONFIRMED, PROMOTED TO #1 CANDIDATE** — a real 100 ms tick found at `worker/main.py:195` |
| 4 | First decode / warm-up 30–150 ms | **REFUTED** — already inside `generation_tps`'s wall-clock denominator |
| 5 | Tail / cache commit 5–100 ms | **PARTLY REFUTED** — commit is real but pre-first-token, not tail |
| 6 | HTTP body + json + pydantic 15–60 ms | **HIDDEN IN TRANSIT**, not in the residual (per (a)) |
| 7 | LAN/SSE < 10 ms; "transit is not network" | **CONFIRMED, and stronger than predicted** — transit is a gossip round-trip plus a poll tick |

**Converging evidence.** Task 0 (independently, before the code was read) said the floor is
context-**independent**. Task 1 found a **size-independent 100 ms poll tick** on every request.
Two independent methods point at the same mechanism.

---

## 3. TASK 2 — **NOT RUN. DEFERRED.** Resume instructions

**Nothing was measured. No number below this line exists yet.** The instrumentation is written
and committed but has never executed on hardware. Do not cite it as a result.

### What is already done and committed (`39fd9e0bb`, local only)

- `src/exo/worker/engines/mlx/phase_marks.py` (new) — runner-side recorder, **deltas only**
- `src/exo/api/phase_marks.py` (new) — API-side recorder, CommandId-keyed, independent total span
- Marks wired at: `runner.py` b1 · `batch_generator.py` b2 · `batch_generate.py` b3/b6/b7/b8/b9/b10/b11 ·
  `cache.py` b4/b5 · `api/main.py` a1–a3 · `chat_completions.py` a4–a7 (**both** the TokenChunk
  and ToolCallChunk finish branches — 44/55 of real requests end in `tool_calls`)
- `api/types/api.py` — `GenerationStats.phase_marks_ms` / `.api_phase_marks_ms`, optional, default `None`
- `start_cluster.sh:1611` — allow-list entry (PM-verified present):
  `[ -n "${EXO_PHASE_MARKS:-}" ] && EXO_ENV="$EXO_ENV EXO_PHASE_MARKS=$EXO_PHASE_MARKS"`

### Inertness when unset — PM-verified, not asserted

`_MARKS_ENABLED: Final[bool]` is read **once at module import** (`api/phase_marks.py:40`,
`engines/mlx/phase_marks.py:40`) — never `os.environ` in a hot path. Every public entry point
begins with `if not _MARKS_ENABLED: return` (api :71, :83, :92, :101, :125; runner :65, :77, :90).
**With `EXO_PHASE_MARKS` unset the production path executes one boolean check of a module
constant and nothing else.** The var is confirmed absent from the currently running PIDs.

Gates: ruff 0→0; basedpyright **425→425** errors, identical error set (baseline via `git worktree`,
never `git stash`). `nix fmt` unavailable on this host — skipped, not faked.

### Exact resume path (next session)

```bash
# 1. Instrumented boot — ONE relaunch. The env var MUST be on the command line;
#    start_cluster.sh:1611 forwards it to both nodes.
cd ~/repos/exo
EXO_PHASE_MARKS=1 ./start_cluster.sh

# 2. MANDATORY GATE before spending the boot (R4 / Ask-A lesson: a var that does not
#    reach the runner PIDs silently zeroes the entire run).
ssh adam.durham@192.168.86.201 'for p in $(pgrep -f "python.*exo"); do ps eww $p | tr " " "\n" | grep EXO_PHASE_MARKS; done'
#    Expect EXO_PHASE_MARKS=1 on the REAL runner PIDs. If absent: STOP, do not run the workload.

# 3. Start the study's EXISTING capture path (no new harness).
python3 tmp/real-usage-capture-20260902/phase2/passive_capture_proxy.py   # 127.0.0.1:52416 -> .201:52415

# 4. Replay the c=1 workload through the proxy (>=20 requests, 90-150K depth,
#    mostly cache hits, several ending in tool_calls).
/usr/bin/python3 tmp/perf-campaign-2/round11/replay_c1.py
#    NOTE (R10 lesson): pin /usr/bin/python3 — Homebrew python3 lacks httpx.

# 5. Analyze. Prints per-phase median + min/max (RANGES, never means), the derived
#    dispatch_and_ipc_gap, and the PASS/FAIL closure check.
/usr/bin/python3 tmp/perf-campaign-2/round11/analyze_marks.py

# 6. RESTORE — second authorized relaunch. Then re-verify RV=0, gamma=3, BI=1,
#    EXO_PHASE_MARKS ABSENT, API 200, clean logs, on real PIDs.
./start_cluster.sh
```

### Gates that are already pre-registered and must NOT be renegotiated after seeing numbers
- **Closure check:** `(stream_closed − handler_entered) − Σ(adjacent API deltas)`; **PASS if
  |median| ≤ 10 ms**. Whatever gap remains **IS the finding** (unattributed = polling/IPC ticks).
- **Actionability floor: 75 ms** median. Phases in [30, 75) ms are reported and parked below-floor.
- **Requests returning no marks must be 0.** Any non-zero count means the allow-list (G4) or the
  tool-call SSE path (G6) failed — the boot is invalid, do not reinterpret it.
- **Never add `mx.eval()` at a mark** (G1). Marks are Python-visible wall only; lazy cost
  legitimately lands in the next eval'd phase.
- **No cross-node clock arithmetic** (G2) — enforced by construction, since only deltas ship.
- Expected fields on the done event: `stats.phase_marks_ms` (runner) and `stats.api_phase_marks_ms` (API).

### Known scope gaps, recorded honestly
- `recv_headers` / `body_read_done` are **out of scope** — they require hypercorn/ASGI-level hooks;
  the first exo frame is already post-validation.
- `first_decode_done` needs a per-`_EngineTask` first-response flag (small addition, not shipped).
- `b5 kv_restored` is labelled **`lazy_no_eval`** — it measures Python-visible wall only, and per
  §2(d) the real GPU cost is in prefill. Do not misread it as the true restore cost.

---

## 4. R12 RECOMMENDATION — the quality-free fix to take next

### **Remove the 100 ms `plan_step` poll tick at `src/exo/worker/main.py:195`.**

**Mechanism.** `plan_step`'s `while True` opens with `await anyio.sleep(0.1)`. A dispatched task
lands in state via the event applier and then waits for the next tick before the worker acts on
it. Expected added latency ~50 ms per request, worst case 100 ms, **size-independent**, **paid on
every turn**.

**Why it is QUALITY-FREE BY CONSTRUCTION.** It is pure scheduling. It changes *when* the worker
notices work already committed to state — it changes no token, no numeric, no cache content, and
no MLX evaluation order. Per PREDICTION.md §5 this is the "polling-tick removal: YES" class. It
does **not** force earlier MLX evaluation and does **not** touch MTP draft warm state.

**Precedent.** This is structurally the same fix as R10's rendezvous sleep (200→0 ms), which
shipped a measured **−224 ms/turn** with byte-identity PASS on both arms.

**Two independent lines of evidence converge on it:** Task 0 says the floor is
context-independent (large stable intercept, no slope signal after de-noising); Task 1 found a
size-independent 100 ms tick on the critical path.

**Shape of the R12 fix (event-triggered wake, not a shorter sleep).** Shortening the tick trades
latency for CPU spin and is the weaker option. The correct fix signals an `anyio.Event` when a
relevant task is applied to state and has `plan_step` wait on it with the 100 ms sleep as a
**fallback timeout**, preserving today's behaviour if the signal is ever missed.

**Honest caveat that gates the ship.** The 100 ms figure is **derived from reading the code, not
measured**. R12 must run the deferred Task 2 boot **first** and confirm the tick appears in the
measured `dispatch_and_ipc_gap` at ≥ 75 ms before shipping anything. Do not ship on a code read.

### Second candidate, correctly ranked below the first
**Tokenization prefix cache** (full ~150 K re-tokenize every turn, confirmed at
`batch_generate.py:2194-2195`, no cache anywhere). Larger potential prize, but **not quality-free
without the seam rule**, and Task 0 proved it is **not separable** from the trie hypothesis on
existing data (r = 0.997). It needs Task 2's direct measurement before it can be ranked.
The full seam rule is recorded in PREDICTION.md §6 and is binding if this is ever taken: safe
seams only immediately after a special/added token; BOS exactly once; suffix with
`add_special_tokens=False`; check `tokenizer.json` for a normalizer; shadow-assert
`cached + tok(suffix) == tok(full)` for the first N requests. Strictly better: key the trie on a
byte-hash of the serialized prefix so the cached prefix is never tokenized at all.

### Explicitly NOT proposed
KV-restore copy elimination — §2(d) proved the restore is lazy, so its cost is not in the
residual. First-decode warm-up — §2(f) proved it is already inside `generation_tps`.

---

## 5. RECONCILIATION WITH THE RECORD

- **Real-usage study §3** (`tmp/real-usage-capture-20260902/REPORT.md:55-75`): the identity
  `wall = prefill_uncached + decode + residual` on all 55 requests, residual [0.75, 1.32] s
  median 0.94 s, transit [0.077, 0.394] s median 0.191 s. This round **reproduced** that residual
  from raw fields to within 0.003 s before analysing it, and **corrected the interpretation** of
  "transit": it is a gossip round-trip plus a poll tick, not network.
- **R7 / R9**: raw TTFT failed as a metric because forcing MLX evaluation reintroduced an
  arm-dependent in-prefill term. This is why G1 (never `mx.eval` at a mark) is a hard gate here.
- **R10**: shipped `EXO_BATCHED_PREFILL_RENDEZVOUS_MS` 200→0 for a measured −224 ms/turn, pooled
  short gap 224.4 ms inside the pre-registered [150, 250] band. Verified still live this round
  (RV=0 on real PIDs). The R12 recommendation is the same class of fix.
- Only gates that actually exist are named. `gh` requires `--repo adurham/exo`.

## 6. COMMITS (local only — nothing pushed)

| SHA | Contents |
|---|---|
| `76294c3d4` | PRE-REGISTRATION — written before any number or source read |
| `93655896d` | Task 0 regression: script, JSON, writeup |
| `4f2e555a0` | Task 1 CODE-READ — 204 verified line cites |
| `39fd9e0bb` | Phase instrumentation (env-gated, inert when unset) + replay/analysis scripts |
| *(this file)* | Round 11 REPORT |

`git push` was never run. `git add -A` and `git stash` were never used.
