# CAMPAIGN 2 / ROUND 11 — PRE-REGISTRATION

**Written 2026-09-04, BEFORE running the Task 0 regression and BEFORE reading any
request-path source.** Timestamped by the commit that introduces this file. Nothing below
may be edited after the first number is looked at; corrections go in REPORT.md as
scored outcomes, not as edits here.

---

## 0. What this round is and is not

- This round produces a **decomposition of the ~0.55 s unexplained per-request fixed cost**,
  not a ship. No production default changes this round.
- Target defined by the real-usage study (`tmp/real-usage-capture-20260902/REPORT.md` §3,
  `partition_verified.json`): identity `wall = prefill_uncached + decode + residual` holds on
  all 55 real requests, residual ∈ [0.75, 1.32] s, **median 0.94 s**, never decomposed.
  `server_received_ts − client_started_ts` ∈ [0.077, 0.394] s, **median 0.191 s**.
  R10 removed ~0.2 s of it (rendezvous sleep 200→0 ms). **~0.55 s/request remains unexplained.**

---

## 1. Task 0 pre-registration — the slope verdict I expect

The brief's decision rule: slope ≈ 1–2 µs/token ⇒ O(context) work dominates
(tokenization / trie walk / KV restore); flat + high jitter ⇒ IPC / polling ticks dominate.

**My pre-registered expectation: the naive regression will be UNDERPOWERED and will NOT
cleanly discriminate.** Specifically I predict:

| Quantity | Pre-registered prediction |
|---|---|
| Slope of `residual` on `prompt_tokens` | positive, **0.5–2.0 µs/token** |
| r² of that regression | **low, < 0.35** |
| Intercept | **0.4–0.8 s** (large, i.e. a real context-independent floor exists) |
| Verdict | **MIXED** — both an O(context) term and a fixed term are present; the naive fit alone cannot rank them |

**Why I expect low r² even if O(context) work is real.** Residual is computed as
`wall − prefill_uncached − decode`, so it *contains* the client→server transit term, which
the study measured at [0.077, 0.394] s — a **0.32 s context-independent spread** injected
straight into a variable whose total observed spread is only 0.57 s. The rendezvous sleep
adds a further constant ~0.2 s. Together these can bury a genuine 0.1–0.3 s context-driven
term under noise of comparable size. A low r² here is therefore **NOT evidence against
O(context) work** and must not be reported as such.

**Sharper test I am pre-registering as the primary one** (report both; this is the
tie-breaker): regress
`residual_ex_transit = residual − (server_received_ts − client_started_ts)`
on `prompt_tokens`. This removes the single largest context-independent noise source.
Pre-registered prediction: **r² rises materially** (I expect > 0.4) and the slope stays in
the same 0.5–2.0 µs/token band. If r² does **not** rise, that is real evidence the residual
floor is dominated by fixed IPC/polling ticks rather than context-scaled work.

**Collinearity caveat, pre-registered:** at 95 % prefix-cache hit `cached_tokens ≈
prompt_tokens`, so their slopes are **not separately identifiable** on this dataset.
Any claim that the trie walk (∝ cached) is distinguishable from tokenization (∝ prompt)
purely from these 55 points is to be rejected. Report the correlation coefficient between
the two regressors and say so explicitly.

**Exclusions, fixed now:** the 1 cold request (`prefix_cache_hit = none`, prompt 92,594,
prefill 222 s) is a leverage outlier and is excluded from the primary fit; report the fit
with and without it. The 2 auxiliary rows (prompt 18/17, null client fields) are excluded —
they were already excluded from the study's partition.

---

## 2. Task 1 pre-registration — what I expect the code read to find

Pre-registered so the read can falsify me, ranked most→least confident:

1. **There are additional hidden sleeps/poll ticks besides the rendezvous.** The rendezvous
   was found exactly this way; I assume it was not unique. Confidence: high.
2. **`server_received_ts` is stamped at the FastAPI handler entry, i.e. AFTER uvicorn/h11
   has read and buffered the full multi-MB body**, which would mean body receive is *already*
   inside the 0.191 s "transit" and NOT inside the residual-after-transit. Confidence: medium.
   If instead it is stamped after json/pydantic validation or after tokenization, then #6/#1
   of the reviewer's list are hiding inside "transit" and the study's split needs the caveat.
3. **The full prompt is re-tokenized every turn** (no tokenization-level prefix cache), and
   the chat template is re-rendered every turn. Confidence: high — this is the default in
   every serving stack I know of that keys its prefix cache on token ids.
4. **KV restore is lazy (no `mx.eval`)**, so its cost lands inside `prefill_uncached`, not in
   the residual. Confidence: medium. If true, the reviewer's item #2 is **largely not in the
   target** and must be de-ranked — this is the single most important thing the read can settle.
5. **`done` does not wait on cache commit** (commit is fire-and-forget or after the final SSE).
   Confidence: low — genuinely unsure, and if wrong it is a large finding.

---

## 3. Task 2 pre-registration — gates fixed BEFORE any instrumented number is seen

### 3.1 Closure check (the primary validity gate)
For each request, with all marks taken in the **API process** (single host, single clock):

```
closure_gap = (stream_closed − recv_headers) − Σ(all adjacent API-phase deltas)
```

- **PASS** if |closure_gap| ≤ 10 ms at the median across requests.
- Runner marks are joined to the API timeline **only** via the shared barrier events
  (`task_dispatched` → `task_received`, `first_token_emitted` → `first_token_from_runner`).
  **Raw runner `perf_counter` values are never subtracted from API `perf_counter` values.**
  Node-2 marks are reported as node-2-internal deltas only.
- The residual gap between the API-visible span and the sum of *attributed* phases (i.e.
  the part that falls inside a barrier interval but outside any named runner phase) **IS
  THE FINDING** — it is the unattributed IPC/polling/queueing tick budget. It is reported
  as a first-class line item, not swept up.

### 3.2 Actionability floor — fixed now, before numbers
**75 ms.** A measured phase must show a **median ≥ 75 ms** across the replay to be
proposed as an R12 target. Phases in [30, 75) ms are reported and explicitly parked as
below-floor. Phases < 30 ms are noted only in the table. Rationale: single-boot
between-boot variance on this cluster is large (documented ~6 tok/s decode); 75 ms is the
midpoint of the brief's 50–100 ms band and I am choosing it before seeing anything.

### 3.3 Hard rules restated as gates (violation ⇒ discard the boot, do not reinterpret)
- **G1. NEVER `mx.eval()` at a mark.** Marks measure Python-visible wall only, and are
  labelled as such. Lazy MLX cost legitimately lands in the next eval'd phase; that is the
  correct accounting, not an error to be "fixed". (R7/R9 lesson: forcing evaluation
  reintroduced the arm-dependent in-prefill term and destroyed raw TTFT as a metric.)
- **G2. `perf_counter` is comparable across processes on ONE host, never across nodes.**
  Cross-node joins go through barrier events only.
- **G3. Instrumentation must be a no-op when its env var is unset.** Proof required:
  either zero non-comment diff on the unset path, or the env check is the first statement
  in every touched function. To be demonstrated in the report, not asserted.
- **G4. The gating env var MUST be in the `start_cluster.sh` allow-list** (R4 / Ask-A
  lesson: a var not in the allow-list is silently dropped and the run measures nothing),
  and its arrival at the **real runner PIDs** must be verified with `ps eww`, not assumed.
- **G5. No new harness.** Replay goes through the study's existing capture path.
- **G6. Ranges, never means.** Per-phase median + min/max across requests.

### 3.4 Relaunch budget — pre-registered
Exactly **two** relaunches authorized: (1) the instrumented boot, (2) the restore to
production. Cluster must end HEALTHY on production config verified on real PIDs
(RENDEZVOUS=0, γ=3, BI=1, no probe vars left) + API 200 + clean logs.

### 3.5 Workload — pre-registered
≥20 requests at ~90–150 K prompt depth, c=1, mostly prefix-cache hits, several terminating
in `tool_calls` (the user's real mix is 44/55 tool_calls). Replayed through the existing
study capture path.

---

## 4. Reviewer's ranked expectation — recorded verbatim for end-of-round scoring

The consult's pre-registered ranking of where the ~0.55 s lives:

1. Tokenization + chat-template render of the full ~150 K prompt every turn — **150–400 ms**
2. Prefix-trie walk + KV restore — **30–300 ms** (TRAP: lazy MLX restore is free in Python
   and paid inside prefill, i.e. NOT in the residual)
3. IPC / event-loop polling ticks between API process and runners — **20–200 ms**, size-independent
4. First decode step / graph warm-up — **30–150 ms**
5. Tail: stop detect, tool-call parse, cache commit, stream close — **5–100 ms**
6. HTTP body receive + json + pydantic on a multi-MB body — **15–60 ms**
7. LAN / SSE / httpx — **< 10 ms**

Scored in REPORT.md §SCORING against measured values. A rank is scored **correct** if the
measured median falls inside the predicted band AND its rank position is within ±1.

---

## 5. Quality-free-by-construction rule, fixed now

A candidate R12 fix may be proposed **only** if it is quality-free *by construction*:

| Candidate class | Quality-free? |
|---|---|
| Removing / shortening a polling tick or sleep | **YES** — pure scheduling, no numerics |
| Eliminating a pure copy in KV restore | **YES** *if provably a pure copy* (no aliasing/mutation hazard) |
| Tokenization prefix cache | **YES ONLY** under the seam rule in §6 |
| Anything that forces earlier MLX evaluation | **NO — do not propose** |
| Anything touching MTP draft warm state | **NO — do not propose** |

---

## 6. Tokenization-cache seam rule (recorded whether or not tokenization wins)

BPE merges never cross pre-tokenizer splits, so `tok(prefix) + tok(suffix) == tok(prefix+suffix)`
**iff the seam is a pre-tokenizer boundary.** Arbitrary message boundaries are **NOT**
automatically safe — whitespace-run lookahead `\s+(?!\S)` and digit chunking `\p{N}{1,3}`
both straddle naive splits.

**SAFE BY CONSTRUCTION:** a seam placed *immediately after a special/added token* (the chat
template's role / tool delimiters). Added tokens are matched **before** pre-tokenization, so
they are a hard boundary.

Additional requirements: BOS emitted exactly once; suffix encoded with
`add_special_tokens=False`; check `tokenizer.json` for a normalizer (NFC across a seam is not
always compositional).

**Ship-time guard (mandatory if this fix is ever taken):** shadow-assert
`cached + tok(suffix) == tok(full)` for the first N requests and log mismatches.

**Strictly better variant:** if the prefix trie can key on a **byte-hash of the serialized
prefix**, the cached prefix is never tokenized at all and the seam problem disappears.

---

## 7. Reconciliation obligations for REPORT.md

The report must cite, not re-derive: real-usage study §3 (the identity and the transit
window), R7/R9 (why raw TTFT failed as a metric), R10 (the shipped rendezvous 200→0 and
its −224 ms). Only gates that actually exist may be named. `gh` requires `--repo adurham/exo`.
