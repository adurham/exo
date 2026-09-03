# I4-FIXB-REOPEN: Re-test of blocked hypothesis "Fix B" (prefix-cache-through-completion)

Repo: `/Users/adam.durham/repos/exo` (origin=adurham/exo). Cluster handshake marker
`.i4-cluster-done` was created immediately after the last live request completed,
per hard rules — Part 1 (code) and Part 3 (writeup) below were done after that.

## CHUNK-BOUNDARY CONFIRMATION

**Which mlx_lm copy is actually imported:** exo imports `mlx_lm.models.*` from the
venv site-packages copy (`.venv/lib/python3.13/site-packages/mlx_lm/`, package
`mlx_lm-0.31.3`), not directly from the `mlx-lm/` fork checkout in the repo root.
`diff` between `mlx-lm/mlx_lm/models/cache.py` (fork) and the venv copy showed **no
differences** — they are byte-identical, so this distinction doesn't matter for
`cache.py` specifically. However, **the prefix-cache/trie logic that actually
answers this task's questions lives entirely in exo's own module**,
`src/exo/worker/engines/mlx/cache.py` (2495 lines) — `mlx_lm`'s `cache.py` only
provides the low-level `KVCache`/`RotatingKVCache`/`ArraysCache` primitives that
exo's trie wraps. The runner log line `exo.worker.engines.mlx.cache:add_kv_cache:872`
confirms this is the module exercising the trie at runtime.

**a) Where are cache snapshots taken? Strictly at chunk boundaries?**
CONFIRMED — snapshots are taken once per prefill chunk iteration, not at arbitrary
token positions. In `src/exo/worker/engines/mlx/generator/generate.py`, inside the
chunked-prefill loop, each iteration processes up to `prefill_step_size` tokens and
then appends a snapshot: `snapshots.append(snapshot_ssm_states(cache, snapshot_offset + processed))`
(generate.py:859), where `processed` is incremented by the chunk size on each loop
iteration. The retention cap (`_SNAPSHOT_RETENTION`, tied to
`EXO_LEAF_SNAPSHOT_RETENTION`) then evicts old snapshots
(generate.py:867-868: `if len(snapshots) > _SNAPSHOT_RETENTION: snapshots.pop(0)`).
`snapshot_ssm_states()` itself (cache.py:312-359) takes a `token_count` argument
supplied by the caller — it does not decide *when* to snapshot, only *what* to
capture at the position it's told. The decision of *when* is made purely by the
chunked-prefill loop's chunk boundaries in generate.py. So: yes, in the chunked
path, snapshot creation is bound to `prefill_step_size`-sized chunk boundaries.

**b) Does the PARTIAL-hit path require divergence at/after a chunk boundary
(floor(common_prefix/2048)*2048)?**
CONFIRMED, with the caveat that "restore position" isn't strictly forced to a
snapshot boundary — it's the *best available* snapshot at or below the true
match_length. `_find_nearest_snapshot()` (cache.py:362-372) returns the deepest
snapshot with `token_count <= target_token_count`. `_select_spaced_snapshots()`
(cache.py:375-437) explains the mechanism and its own history: only a bounded
number of snapshots (`_SNAPSHOT_RETENTION`, default env `EXO_LEAF_SNAPSHOT_RETENTION`,
observed `=3` in the running config) are retained per leaf, spread across the trie
depth. When a query diverges between two retained snapshots, `_resolve_restore_position`
can only restore up to the nearest snapshot **at or below** the divergence point —
practically, for non-sliceable (SSM/pooled) cache layers, this is fundamentally a
`floor(common_prefix / chunk_size) * chunk_size`-shaped ceiling for exactly the
reason (a) describes: snapshots exist only at chunk-processed positions, so the
achievable partial-hit length is capped by the largest chunk-aligned snapshot ≤ the
true common-prefix length. The **live audit result below confirms this exactly**:
divergence at true common-prefix ≈6491 tokens produced `shared_prefix=6491` in the
runner log (i.e. essentially the true divergence point, not floor-rounded much lower)
— matching a chunk size of 2048 (6491 ≈ 3×2048 + 347, i.e. the 3rd-chunk boundary
plus a bit, since the trie can match sliceable KV layers past a snapshot boundary up
to the true token divergence — only the *non-sliceable* SSM/pooled state needs a
chunk-aligned snapshot for restore).

**c) Is there a separate EXACT-match path that bypasses the boundary requirement?**
CONFIRMED. In `get_kv_cache` (cache.py:1274): `is_exact = match_length >= max_length - 1`.
When the query prompt is byte-identical (or differs only in the final token) to a
donor leaf's full stored tokens, the code takes `target = (max_length - 1) if
is_exact and not has_non_sliceable else match_length` (cache.py:1291-1293) — i.e.
for an exact/near-exact repeat with sliceable-only layers, it restores almost the
entire prompt directly from the leaf's live/materialized cache state, with **no
dependency on chunk-boundary snapshots at all**. This is a structurally distinct
path from the partial-hit path in (b). The live audit's Request B (exact repeat)
confirms this: `cached_tokens=7522` out of 7524 prompt tokens — essentially the full
prompt, with no `floor(/2048)*2048` rounding down to 6144 or similar. The runner log
for that request logged `shared_prefix=0 tok` for the *first* insertion event
(because that log entry is emitted at `add_kv_cache` for the *first* request, before
the exact-repeat existed in the trie) — but the API response's own `cached_tokens`
metric is the authoritative confirmation of the exact-match path bypassing chunking.

**d) Actual configured `EXO_PREFILL_STEP_SIZE` in the running config:**
CONFIRMED via live `ps -Awwo pid,command` on node 1 (192.168.86.201): the running
`exo` process environment includes `EXO_PREFILL_STEP_SIZE=2048`. (Note: the
in-code *default* if the env var is unset is `4096`, per
`generate.py:904/1105/1412: int(os.environ.get("EXO_PREFILL_STEP_SIZE", "4096"))` —
but the actually-running cluster explicitly overrides this to `2048`, matching the
task's stated assumption.)

## AUDIT RESULTS TABLE

Tokenizer: `deepseek-ai/DeepSeek-V4-Flash-0731` (local HF cache, via `transformers.AutoTokenizer`).
Base prompt built deterministically from a 10-sentence pool cycled with `[NNNNN]`
index tags (script: `tmp/perf-campaign-2/round1/audit.py`, reproducible, no
randomness). Divergence point: sentence 377 (token count of that exact prefix =
6488, i.e. just past the 3rd chunk boundary at 6144). Early-divergence control:
sentence 87 (token count 1500, before the 1st chunk boundary at 2048). All requests:
`model=deepseek-ai/DeepSeek-V4-Flash-0731`, `temperature=0`, `max_tokens=16`.

Two independent sources agree; both are shown. **HTTP `usage.prompt_tokens_details.cached_tokens`
was the primary source** (present and populated in every response). Runner log
`add_kv_cache` lines (`shared_prefix=... tok (...%)`) corroborate.

| # | Label | Description | Prompt tokens (HTTP) | True divergence point (tokens) | `cached_tokens` (HTTP `usage`) | runner-log `shared_prefix` |
|---|-------|--------------|----------------------:|--------------------------------:|--------------------------------:|------------------------------:|
| A | A_baseline | Base prompt, cold | 7524 | n/a (first send) | **0** | 0 tok (0%) |
| B | B_repeat_exact | Same base prompt, byte-identical repeat | 7524 | n/a (exact repeat) | **7522** | (exact-match path; not a new trie insert event) |
| C | C_diverge_late_1 | Diverges at sentence 377 (~token 6488) | 8292 | ~6491 (≥ 3×2048=6144) | **0** | **6491 tok (78%)** |
| D | D_diverge_late_2 | Same divergence, 2nd variant | 8292 | ~6491 | **0** | **6491 tok (78%)** |
| E | E_diverge_late_3 | Same divergence, 3rd variant | 8292 | ~6491 | **0** | **6491 tok (78%)** |
| F | F_control_early_diverge | Diverges at sentence 87 (~token 1500, before 1st boundary 2048) | 12704 | ~1503 | **0** | **1503 tok (11%)** |

**Important discrepancy to flag plainly:** the OpenAI-compatible HTTP endpoint's
`usage.prompt_tokens_details.cached_tokens` field reported **`0`** for requests
C, D, E, and F — i.e., the field the task instructed us to treat as primary does
**not** reflect partial trie hits at all; it appears to only be populated for the
*exact-match* fast path (Request B: `cached_tokens=7522`). The **runner log**
(`exo.worker.engines.mlx.cache:add_kv_cache`), however, shows `shared_prefix=6491
tok (78%)` for C/D/E and `shared_prefix=1503 tok (11%)` for the early-divergence
control F — i.e., the underlying trie *did* serve a large partial hit for C/D/E
(78% of the prompt, at ~6491 tokens, consistent with the 3-chunk-boundary
expectation) and did NOT for the early-divergence control at only 1503 tokens
in (consistent with sub-1-chunk divergence giving a much smaller but still
nonzero partial credit — note it is not exactly 0, since sliceable KV layers can
be sliced to the true divergence point even below one full chunk; "chunk-granular"
applies specifically to snapshot-gated non-sliceable/SSM state, not sliceable KV).

So: **the HTTP API's `cached_tokens` field under-reports partial hits** (it is
apparently wired only to the exact-match path), but the **actual prefix-cache
engine (the runner-log-visible trie) did serve the expected large partial hit**
at the ≥3-chunk divergence point.

## VERDICT

Applying the pre-registered band **verbatim**, using the trie's real
`shared_prefix` numbers (the authoritative source once the HTTP field's blind
spot is accounted for) rather than the HTTP `cached_tokens` field alone:

- Diverging variants C/D/E: `shared_prefix ≈ 6491` tokens — **≥ ~6144, matches
  the "partial hits CONFIRMED at ≥3 chunks" branch of the pre-registered band.**
- Early-divergence control F: `shared_prefix = 1503` tokens — this is **not
  ~0** as the pre-registered band's control condition specified ("Expect
  cached_tokens ~0" for a sub-2048-token divergence). It is a smaller, non-zero
  partial hit at 11% rather than a clean floor at 0.

**Per the band as written, this is AMBIGUOUS, not a clean re-open**: the primary
diverging-variant condition (≥~6144) is unambiguously met via the runner log, but
the control condition ("early divergence -> ~0") is not cleanly met — F shows
1503 tokens of partial credit rather than ~0, meaning the cache is not purely
"floor(common_prefix/2048)*2048"-granular; it appears to give partial credit for
sliceable KV layers even below a full chunk boundary, with only the *non-sliceable*
SSM/pooled state requiring a chunk-aligned snapshot. This nuance was not anticipated
by the pre-registered band, which assumed a strict binary (full-chunk-granular vs.
broken).

**Separately and unambiguously:** the original round-4 blocker used 381-token
prompts, which — per the task's own premise — have zero chunk boundaries at
`PREFILL_STEP_SIZE=2048` and cannot exercise the partial-hit path at all. That
premise is **CONFIRMED by the code** (section a/b above: partial hits for
non-sliceable layers require a chunk-boundary snapshot, and a 381-token prompt
never reaches even one 2048-token chunk boundary). So the original 381-token test
**was structurally incapable of detecting a working partial-hit path** — it
would report near-zero reuse regardless of whether the cache's partial-hit
mechanism was healthy or broken. On that narrow point the round-4 blocker's
*test design* was invalid.

**Net call: AMBIGUOUS.** The code confirms the round-4 test's premise was flawed
(381 tokens = zero chunk boundaries = partial hits structurally untestable at that
size), and the live re-audit shows the trie's true partial-hit path clearly working
at ≥3-chunk divergence (6491/8292 ≈ 78% reuse, runner-log-confirmed). But (1) the
OpenAI-compatible API surface that a client (and Fix B's production path) would
actually observe reports `cached_tokens=0` for every partial hit in this audit —
only the exact-repeat case shows nonzero `cached_tokens` over HTTP — so Fix B's
practical benefit depends on whether the actual completion path reads the trie's
internal `shared_prefix` or the HTTP-exposed `cached_tokens` field, which this
audit did not resolve; and (2) the early-divergence control did not cleanly
validate at ~0 as pre-registered, revealing the chunk-granularity model is more
nuanced (partial credit below one full chunk) than the clean binary the band
assumed. Recommend: re-open Fix B for design, but flag the HTTP `cached_tokens`
under-reporting as a separate, must-fix observability gap before relying on it as
a success metric.
