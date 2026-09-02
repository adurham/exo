# Prefix-cache deep dive — why 0 full hits with 54/57 partial?

Read-only, single-pass, 2026-09-02. No code changes, no commits, no inference
calls to the cluster. Every number below is either MEASURED (from
`tmp/real-usage-capture-20260902/phase1/requests.jsonl`, itself derived from the
API event log's own `usage`/`stats` blocks) or DERIVED (arithmetic on those
measurements, labeled as such). Inferences are labeled INFERENCE.

---

## TL;DR — the verdict

**All three pre-registered hypotheses are refuted. The cache is not broken; the
metric label is misleading.** `"partial"` is the *correct and healthy* steady
state for a growing agentic conversation, and `"exact"` is structurally
unreachable in one. The cache served **97.60% of all prompt tokens** in the
session and reused **99.9986% (median) of everything it held** turn-to-turn.

The real prefill cost is **not** a hit-rate problem. It splits into:

| Bucket | Tokens | % of all prefill | Fixable? |
|---|---:|---:|---|
| A. Cold start after a runner relaunch | 92,594 | 49.0% | Yes — trie doesn't survive process death |
| B. Model's own completions re-prefilled next turn | 41,414 | 21.9% | Yes — decode KV is discarded |
| C. New tool results + chat framing | 55,002 | 29.1% | No — genuinely new tokens |
| **Total uncached prefill** | **189,010** | 100% | (of 7,883,591 presented) |

---

## 1. The cache architecture as it exists

### Key derivation — exact token ids, no synthetic key

`KVPrefixCache` (`src/exo/worker/engines/mlx/cache.py:690`) is a **radix trie
keyed on raw token ids**. `_longest_prefix_match` (`cache.py:1605`) walks the
trie token-by-token; the primitive is `get_prefix_length` (`cache.py:2324`):

```python
equal = mx.equal(prompt[:n], cached_prompt[:n]).astype(mx.int32)
prefix_mask = mx.cumprod(equal)   # 1 until first mismatch, then 0 forever
return int(mx.sum(prefix_mask).item())
```

There is **no hash, no timestamp, no session id, no template-version tag, no
tool-schema ordering** anywhere in the key path. I grepped the whole key
derivation; the key is the token sequence itself. **Hypothesis 1 is refuted at
the source level, not merely by data fit.**

A permanently-on integrity check (`cache.py:1246-1280`, deliberately *not* gated
behind `EXO_PREFIX_CACHE_DIAG`) re-verifies on **every hit** that the donor
leaf's own stored tokens actually share the query's prefix, logging
`[PREFIX_CACHE_INTEGRITY_VIOLATION]` otherwise. No such line appears in this
session's data path.

### Granularity — token-level, not chunk-level

Matching is arbitrary-length and token-granular. Trie edges are split at
divergence points (`add_kv_cache` → `_insert_path`), and a continuing
conversation takes the `_extend_leaf_suffix` fast path (`cache.py:1035`), which
attaches only `[old_depth, new_length)` as a new edge — **the shared prefix is
never re-sliced or re-copied**. There is no 2048-boundary quantization of the
match. **Hypothesis 2 is refuted.**

One real granularity constraint exists but does not bind here: DSv4 has
non-sliceable layers, so the restore position is snapshot-bounded via
`_resolve_restore_position` → `_find_nearest_snapshot` (`cache.py:370`), with
`_SNAPSHOT_RETENTION = 8` chunk-boundary snapshots kept per leaf. That would
cap reuse if a turn's divergence fell far below the leaf tip — it did not, on
any of the 54 turns (see §2).

### Partial-hit semantics — and the source of "0 full hits"

```python
# cache.py:1284
is_exact = match_length >= max_length - 1
```

`max_length` is the length of the **incoming** prompt; `match_length` is bounded
by the length of the **stored** prompt. The metric is then classified in
`generator/batch_generate.py:4601-4605`:

```python
if state.is_exact_hit:            prefix_cache_kind = "exact"
elif state.prefix_hit_length > 0: prefix_cache_kind = "partial"
else:                             prefix_cache_kind = "none"
```

**Therefore `"exact"` requires the incoming prompt to be no more than 1 token
longer than something already cached** — i.e. a retry, a regeneration, or a
re-submission. In an append-only agent loop, every turn appends the assistant's
reply plus a tool result, so the prompt always grows. **"0 full hits" is
correct-by-design, not a defect.**

Measured, this session: per-turn prompt growth was **min 77, median 1,324, max
11,535 tokens**; turns with growth ≤ 1: **0 of 54**. A full hit was arithmetically
impossible on every single turn.

### The reported `cached_tokens` is real compute saved, not an optimistic match length

Worth stating explicitly, because the distinction decides whether the 97.6%
figure means anything. `batch_generate.py:2403` computes

```python
local_hit_length = len(all_prompt_tokens) - len(remaining_tokens)
```

and `get_kv_cache` returns `remaining = prompt_tokens[restore_pos:]`
(`cache.py:1354`), where `restore_pos` is the **snapshot-bounded** position, not
the raw `match_length`. So `cached_tokens` counts tokens whose KV was genuinely
materialized and skipped — the saving is real.

### Eviction

LRU with an explicit guard: `_active_leaf_id` makes the in-flight session
structurally un-evictable (`cache.py:~735`), mirroring vLLM/sglang's ref-count
"touch". The comment cites prior forensics where a 142,384-token leaf was evicted
under a transient memory peak and re-prefilled the next turn. **No eviction event
appears in this session** — the only miss at depth is the cold start (§3).

### Lifetime — the load-bearing structural fact

`KVPrefixCache` is constructed once per runner process (`builder.py:156`) and is
**purely in-memory**. I grepped for any serialize / persist / save / to-disk path
for the trie: **none exists**. When the runner process dies, the entire trie —
every materialized KV tensor for a 146K-token conversation — dies with it.

---

## 2. Turn-to-turn reuse from requests.jsonl (n=55 main-chat turns, call_seq 33–87)

### The headline invariant

For **54 of 54** consecutive turn pairs:

```
len(prompt(n-1)) - cached_tokens(n) == 2      ← exactly 2, every single turn
```

Not approximately 2. Not a distribution centered on 2. The constant 2, 54 times
out of 54. Expressed as efficiency, `cached(n) / len(prompt(n-1))`:

```
min 99.9978%   median 99.9986%   max 99.9989%
```

**The trie gave back essentially everything it held, every turn.** This is the
single most decisive number in the report, and it refutes Hypothesis 3
outright — literal prefix overlap is not low, it is near-total.

### Session aggregates

```
total prompt tokens presented    : 7,883,591
served from cache                : 7,694,581   (97.60%)
actually prefilled               :   189,010   ( 2.40%)
per-warm-turn uncached           : median 1,326   mean 1,785   min 79   max 11,537
cached % per warm turn           : median 99.07%  worst 89.84%
```

### Decomposition of the 189,010 uncached tokens

**A — Cold start: 92,594 tok (49.0%).** Request `call_seq=33` reports
`cached_tokens=0`, `prefix_cache_hit="none"`, and prefilled 92,594 tokens at
416.8 tok/s ≈ **222 s of wall time in a single request**.

This was a **runner relaunch, not an eviction or a key miss** — established
independently of this analysis by the phase-1 capture: instance `25ae372c`
served the *same client session's* first 32 calls and was superseded by a
LoadModel at 12:37:19, after which instance `339f04f8` served `call_seq=33`
onward. The client conversation continued seamlessly; the server-side trie did
not. The 92,594 tokens were fully re-prefilled from scratch despite having been
resident in the previous process moments earlier.

**B — The model's own completions, re-prefilled: 41,414 tok (21.9%).** The leaf
stores `all_prompt_tokens` — the **prompt only**. KV produced during decode is
never inserted into the trie. Next turn, the assistant's reply arrives as new
prompt text and is prefilled from scratch.

Evidence this is what's happening (DERIVED): `uncached(n) − completion_tokens(n-1)`
is **positive on all 54 pairs**, median residual 446 tokens. On the
largest-completion turns the ratio `uncached(n) / completion(n-1)` sits at
**1.017–1.069**:

```
seq 82→83: completion 6,480  uncached_next 6,927  ratio 1.069
seq 54→55: completion 2,903  uncached_next 3,042  ratio 1.048
seq 71→72: completion 1,687  uncached_next 1,716  ratio 1.017
seq 33→34: completion 1,982  uncached_next 2,025  ratio 1.022
```

A 6,480-token completion producing 6,927 uncached tokens next turn means the
model re-read almost exactly its own output, plus a little framing.

**C — Genuinely new content: 55,002 tok (29.1%).** Tool results and chat framing.
Irreducible by any caching strategy.

### Cost in wall time (DERIVED, using each request's own server-measured `prompt_tps`)

```
est. total prefill wall   : ~600 s   (cold 222 s + warm 378 s)
A (cold start)            : ~222 s
B (own completions)       : ~181 s   at the median 229 tok/s
A + B                     : ~403 s of ~600 s  =  67% of all prefill wall time
total client wall, 55 calls: 1,929 s → A+B ≈ 20.9% of all in-call wall time
```

### Tool-call correlation — checked, and it is not a miss driver

Tool-heavy turns do **not** correlate with cache misses: all 44 `tool_calls`
turns and all 12 `stop` turns land in the same 99%+ reuse band. Tool results
enlarge bucket C (new tokens), they never invalidate the prefix. The two largest
uncached warm turns (11,537 and 8,572 tokens, `call_seq` 42/43) are `read_file`/
`search_files` turns injecting large file contents — new content appended, prefix
intact.

### All three `"none"` results are accounted for

| Row | prompt_tok | Explanation |
|---|---:|---|
| `call_seq=33` | 92,594 | Cold start, first call after the runner relaunch |
| aux | 18 | Separate tiny helper call, not the 146K chat thread |
| aux | 17 | Separate tiny helper call, not the 146K chat thread |

**There is zero mid-session `"none"` at depth.** No unexplained miss exists — an
important negative, since a miss at ~146K with a live trie *would* have indicated
a real defect.

---

## 3. Verdict: which hypothesis fits

**None of them. The investigation's premise is wrong.**

| # | Hypothesis | Status | Basis |
|---|---|---|---|
| 1 | Brittle cache key | **REFUTED** | Key is exact token ids; no timestamp/session/template/schema component exists in the key path (`cache.py:2324`, `_longest_prefix_match`) |
| 2 | Granularity / chunk-boundary invalidation | **REFUTED** | Matching is token-granular and arbitrary-length; suffix extension never re-slices the shared prefix (`cache.py:1035`) |
| 3 | Client prepends fresh context; low literal overlap | **REFUTED** | Overlap is 97.60% session-wide and 99.9986% median turn-to-turn |

### The actual mechanism

`is_exact = match_length >= max_length - 1` (`cache.py:1284`) can only fire when
the incoming prompt is no longer than what's cached. An agentic loop appends
every turn (median +1,324 tokens), so `is_exact` is **structurally unreachable**
and every healthy turn is *correctly* labeled `"partial"`. The 37/40 (here 54/57)
"partial" reading was never the anomaly — **it is the signature of a cache
working as designed.** The reference-model review's framing — "0-full-with-N-partial
is the structural anomaly to explain" — was reasonable a priori but does not
survive contact with the code: those two labels do not mean what the framing
assumed.

Note also a minor label looseness: the `− 1` slack means a prompt growing by
*exactly* 1 token would be reported `"exact"` while not literally being one. Moot
here (0 of 54 turns grew by ≤ 1), and it is a naming issue, not a correctness one.

### The one thing I did not fully close: why exactly 2

The `-2` is **MEASURED** and rock-steady. Its decomposition is **INFERENCE**, and
I am flagging it rather than claiming it:

- **−1 is confirmed**: prefill always runs on `prompt_tokens[:-1]`
  (`batch_generate.py:2573`), holding back one token to feed the generator. The
  KV cache is therefore one token shallower than the stored token array.
- **The second token is not conclusively pinned.** Two candidates, and they are
  *not* additive — exactly one is right:
  (a) the DSv4 generation-prompt tail. `vendor/deepseek_v4_encoding.py:452` appends
      `ASSISTANT_SP_TOKEN + thinking_start_token` = `"<｜Assistant｜>" + "<think>"`,
      which I tokenized offline with the real tokenizer to **exactly 2 ids
      `[128804, 128821]`**;
  (b) the snapshot-bounded restore position landing one short of the tip.
- Suggestively, `batch_generate.py:2613` does `last_tokens = prompt_tokens[-2:]`
  — a 2-token handoff to decode — but I did not trace it end-to-end to the
  observed shortfall and will not assert it.

**A chunk-boundary explanation is refuted by the data**: a snapshot-spacing effect
would vary with how far the turn diverged, yet the shortfall is a *constant* 2
across growths spanning 77 → 11,535 tokens.

**Cost of the ambiguity: 2 tokens/turn ≈ 108 tokens across the whole session, out
of 7,883,591.** Negligible; worth one grep to close for tidiness, not worth a
work item.

---

## 4. Pre-registered fix proposal (design only — NOT implemented)

Gates and outcome bands are registered **before** any measurement, per house
methodology. Both proposals are **KV-lifetime** changes. **Neither touches the
cache key — the key is correct and must not be modified.**

### Ranking, and an honest caveat on it

By measured share of this session's prefill work, A (49%) > B (22%). But A's
49% comes from a **single relaunch event**, so its amortized value depends
entirely on relaunch frequency — which this data cannot establish (one session,
one relaunch). **B is the safer first build**: its 22% is spread evenly across
all 54 turns, it is fully in-process, and it needs no cross-node serialization.
Recommend **B first, A second**, with A's priority revisited after measuring
relaunch frequency from the log archive.

---

### Fix B (recommended first) — retain decode-produced KV in the trie

**Target:** bucket B, 41,414 tok (21.9% of prefill work, ~181 s this session).

**Mechanism.** The leaf stores prompt-only tokens; the KV for the model's own
completion is computed during decode, then discarded, then recomputed as prompt
next turn. Extend the leaf with the generated token ids and their already-resident
decode KV at end-of-turn, so turn *n+1*'s prefix match extends through turn *n*'s
reply.

**Pre-registered risk (must be tested, not assumed).** This only pays off if the
completion's *generated token ids* are identical to the ids produced when that
same text is *re-encoded through the chat template* next turn. Boundary tokens
around the assistant message, `<think>`/`</think>` marker handling, and
`_strip_v4_thinking_markers` (`utils_mlx.py:1489`) are all live divergence risks.
The measured ratios (1.017–1.069) suggest most of the completion re-tokenizes
stably, but **the match should be expected to cover most-but-not-all** of it.

**Experiment.** Offline first, no cluster time: for each of the 54 turn pairs,
tokenize turn *n*'s prompt and diff against turn *n-1*'s stored tokens + generated
ids. Measure the length of the agreeing run.

**Outcome bands, registered in advance:**
- **≥ 80%** of completion tokens re-tokenize identically → **build it.** Expected
  ~18% cut in warm-turn prefill.
- **40–80%** → build only with a token-level agreement check at insert
  (insert the agreeing prefix, discard the tail).
- **< 40%** → **abandon.** Re-tokenization is too unstable; say so and stop.

**Gate:** a needle probe at 100K must return `needle_hit=true`, `bos_spam=false`,
and bit-identical output vs. the cache-off path on a fixed prompt. Any output
divergence is an automatic NO-GO regardless of the speedup — this is a
correctness-sensitive path (cf. the `strict_snapshot` history at
`cache.py:~1380`, where a wrong-position restore produced `' his his his'`
degeneration).

---

### Fix A — survive a runner relaunch

**Target:** bucket A, 92,594 tok (49.0% of prefill work, ~222 s this session).

**Mechanism.** The trie is in-memory and dies with the process
(`builder.py:156`); no persistence path exists. A client conversation outlives the
runner, so a relaunch forces a full re-prefill of a 92K-token prompt that was
resident seconds earlier.

**Pre-registered feasibility check — do this BEFORE any implementation.** 146K
tokens of KV per node is multiple GB even with MLA compression. Serializing and
restoring that across a 2-node TP relaunch is real engineering, and the write
itself may cost more than the re-prefill it saves.

**Measure first, in this order:**
1. Relaunch frequency from the log archive (how often do instances actually
   cycle?). This sets the entire amortized value and is cheap to get.
2. Bytes-on-disk for one 146K-token leaf, and measured write/read wall time.

**Outcome bands, registered in advance:**
- Restore wall time **< 25%** of the equivalent re-prefill (~222 s → target
  < 55 s), **and** relaunches occur more than once per ~10 sessions → **build it.**
- Restore **25–60%** of re-prefill → build **only** if relaunch frequency is high;
  otherwise shelve.
- Restore **> 60%**, or serialization cost measurably degrades steady-state
  turns → **abandon persistence.** Pivot to the cheaper adjacent goal: **make the
  runner stop relaunching**, or hand the trie to a supervisor process that
  outlives the runner.

**Gate:** a restored trie must pass the same `[PREFIX_CACHE_INTEGRITY_VIOLATION]`
check on first hit, plus a needle probe. A restored-but-subtly-wrong KV cache is
far worse than a cold prefill.

---

### What NOT to do

- **Do not change the cache key.** It is exact token ids and is working at
  99.9986% efficiency. Any hashing/normalization can only make it worse.
- **Do not chase the 2-token shortfall as a performance item.** ~108 tokens per
  session.
- **Do not "fix" the 0-full-hits metric by loosening `is_exact`.** That would
  paint the dashboard green while changing nothing real. If the label is worth
  changing at all, the honest move is to **report the hit *ratio*
  (`cached/prompt`) rather than the three-way categorical** — a 99.07% median
  would then read as the success it is. Cosmetic, not a perf fix.
- **Do not re-derive throughput from these token counts.** Per house
  methodology, prefill tok/s must use offline tokenization for the numerator and
  wall clock for the denominator. The counts here are used **as counts**, and the
  one place a rate appears (§2 wall-time estimates) uses each request's own
  server-measured `prompt_tps` and is labeled DERIVED.

---

## 5. Answer to the framing question

The external review held that "a 20% cache-hit improvement beats a 20%
prefill_tps improvement in this regime." That is arithmetically sound but
**inapplicable here: there is no 20% of hit rate left to win.** The cache is at
97.60% session-wide and 99.9986% turn-to-turn; the remaining headroom on the
*hit-rate* axis is ~0.001%.

The recoverable work is real but it is **not hit-rate** — it is **KV lifetime**:
KV that was correctly computed and then thrown away, either by process death (A)
or by never being stored at all (B). Together A+B are ~67% of prefill wall time
in this session, but they are won by **retaining KV**, not by matching prefixes
better.

For bucket C (29.1%, genuinely new tokens) the lever does revert to prefill speed
and fixed overhead — the existing `exo-dsv4-prefill-tuning` line of work, where
the GPUs are already measured 96.6–97.0% saturated and the remaining gap is
kernel efficiency.

---

## Appendix — provenance

- **Data:** `tmp/real-usage-capture-20260902/phase1/requests.jsonl`, 57 records,
  every field provenance-wrapped. 55 main-chat rows (`call_seq` 33–87) + 2 aux.
  Server-side `usage`/`stats` from the API event log; client wall from Hermes
  `state.db`.
- **Code read (exo fork, HEAD at time of read):**
  `src/exo/worker/engines/mlx/cache.py` (`KVPrefixCache` 690; `is_exact` 1284;
  `get_kv_cache` 1165; `_extend_leaf_suffix` 1035; `get_prefix_length` 2324;
  `_find_nearest_snapshot` 370),
  `src/exo/worker/engines/mlx/generator/batch_generate.py` (classification
  4601-4605; `cached_tokens` 4631; `local_hit_length` 2403; prefill `[:-1]` 2573;
  `_save_prefix_cache` 5263),
  `src/exo/worker/engines/mlx/vendor/deepseek_v4_encoding.py` (generation tail 452),
  `src/exo/worker/engines/mlx/builder.py:156`.
- **Offline tokenizer probe** (read-only, no cluster contact): confirmed
  `"<｜Assistant｜>" + "<think>"` → `[128804, 128821]`, exactly 2 ids.
- **Constraints honored:** no code changes, no commits, no writes to the repo
  source tree (`git status` shows only pre-existing unrelated `tmp/` artifacts),
  no inference requests to the cluster.
- **Reviewed** against `exo-cluster-debugging/references/perf-hypothesis-discipline.md`:
  guard-reachability checked at source (`is_exact`), fast-path/slow-path sharing
  checked, curve shape interrogated (a *constant* 2 across 77→11,535 growth rules
  out chunk-boundary effects), and an external second opinion was taken on the
  verdict before writing — it raised three gaps (restore-position semantics, the
  unaccounted `"none"` rows, and the non-additive `-2` candidates), all three of
  which are closed above.
