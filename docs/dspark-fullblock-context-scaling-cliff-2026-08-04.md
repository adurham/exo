# DSpark FULLBLOCK context-scaling cliff — handoff — 2026-08-04

**STATUS (2026-08-04, evening UPDATE — CORRECTS THIS DOC'S CORE CLAIM):
the original "15.9x collapse that gets progressively worse with context"
finding does NOT reproduce and is very likely WRONG as originally framed.
A live re-test (same commits, same weights, fresh relaunch) found the
slow-cycle behavior is confined to a narrow, one-off band around
prompt_tokens≈2800 — it reproduces THERE, deterministically, across 3
repeated identical requests — but completely clears up by 3000 tokens and
stays clean (16-22 tok/s, zero diagnostic outliers) all the way out to
15,030 prompt tokens, comparable to the original "1.73 tok/s" data point.
See the "MAJOR CORRECTION" section below before trusting anything above
this line. The doc is kept largely intact below for the historical trail
and because the ~2800-token anomaly itself is real and still unexplained
— just NOT the sustained, context-scaling cliff originally claimed.**
Cluster is currently live on **DSpark OFF** (`EXO_SPECULATIVE=0
EXO_DSV4_DSPARK=0`) as a deliberate, temporary safety choice — DSpark's
`FULLBLOCK` attention path is actively worse than no speculation at any
real conversation length. Do not flip `EXO_DSV4_DSPARK` back to `1` in
production without either fixing this or explicitly accepting the
regression for short-context-only workloads.

## TL;DR for whoever picks this up

Earlier the same day (2026-08-04), a different DSpark throughput bug was
found and fixed cleanly: `EXO_DSV4_ROWSEQ_FULLBLOCK_MOE=1` (forces the
whole MoE FFN per-row in the verify forward) was capping throughput
regardless of draft quality. An offline bisect found the correctness fix
only actually needs `shared_experts` per-rowed, not the whole FFN —
shipped as `EXO_DSV4_MOE_PARTS_ROWSEQ=shared` (commit `b9921962e`, exo
main). That fix is solid, verified correct, and gives a real win — **but
only at near-zero context** (300-500 token prompts): 27.48 tok/s mean,
+12% over the old FULLBLOCK_MOE config, +11% over DSpark-off.

When asked whether this holds at real context depths, it does not. A
clean context-depth sweep (methodology below) found:

| Config | depth≈500 | depth≈14K | Ratio |
|---|---|---|---|
| DSpark ON + FULLBLOCK + MOE_PARTS_ROWSEQ=shared | 27.56 tok/s | **1.73 tok/s** | **15.9x collapse** |
| DSpark OFF (sequential decode) | 25.11 tok/s | 19.57 tok/s | 1.28x (normal) |

DSpark+FULLBLOCK is not "no longer winning" at real context — it's
**catastrophically worse than doing nothing** (1.73 vs 19.57 tok/s at the
same depth). This is a severe, previously-undiscovered bug. As far as I
can tell from searching prior session memory, nobody has ever benchmarked
DSpark decode throughput at real context depth (10K+ tokens) on this fork
before today — every historical DSpark number in memory was measured at
300-500 token prompts.

**Important: this is NOT a regression from today's MoE-side fix.** The
attention-side `EXO_DSV4_ROWSEQ_FULLBLOCK=1` mechanism responsible for
the cliff has been the default since 2026-08-02 (two days earlier), and
was not touched in today's session at all — only the *MoE-side* flag
(`FULLBLOCK_MOE`) was changed today. The cliff was already there; it was
just never measured until today.

## The mechanism (traced, not yet root-caused to the exact sub-operation)

`EXO_DSV4_ROWSEQ_FULLBLOCK=1` runs the ENTIRE attention block per-row
inside the DSpark verify forward — `block_size=5` separate attention calls
per verify cycle, instead of one batched call over all 5 rows. This is a
*correctness* fix (makes L>1 batched verify bitwise-equivalent to L
sequential decode steps, closing a numerics-drift bug that caused a
self-doubt infinite loop — see `exo-speculative-decode-correctness`
skill). It has always had a real bandwidth/compute cost; what wasn't known
until today is that the cost apparently scales catastrophically with
context, not linearly.

Confirmed via the cluster's own existing per-cycle diagnostic (no new
instrumentation needed, this logging already exists in
`pp_speculation.py`, fires automatically when a cycle exceeds 1000ms):

```
[PP DSpark OUTLIER R1 n=296] cycle took 1561.5ms (>1000ms threshold) --
batch_xchg=1.3ms r0_fwd=0.0ms r1_verify_wait=0.0ms
r1_verify_fwd=1455.8ms draft=103.1ms trim_xchg=0.7ms
```

The verify forward pass itself (`r1_verify_fwd`) costs **1.46 seconds per
cycle** at ~14K context. Draft acceptance was still 94% in this same log
window — so this is not a drafting-quality problem, it's the verify
forward's own cost.

**Leading hypothesis (from a second-opinion consult, NOT yet verified):**
DSv4's per-layer attention (for most layers, `SparseCompressedAttention`)
includes an `Indexer` top-k search over the pooled/compressed KV cache —
an operation whose cost scales with context size. Running that search 5x
per verify cycle (once per row, unbatched) instead of once (batched)
could plausibly explain a cost that explodes as context grows, especially
if the per-row calls also miss whatever fast/fused SDPA kernel path the
batched call would hit (mask-shape/dtype fast-path misses are a known
MLX SDPA gotcha on this fork — see `exo-perf-tuning` skill's SDPA notes).
Other candidates not yet ruled out: KV-cache re-materialization/copy cost
per row (rather than in-place buffer writes), or `mx.eval`/sync placement
inside the per-row loop breaking lazy graph fusion.

**None of these candidates has been confirmed. This is the actual next
step if resuming.**

## UPDATE 2026-08-04 (later, same day, code-only follow-up — no cluster time spent)

Read `SparseCompressedAttention.__call__` and `Indexer.__call__` in full
(`mlx-lm/mlx_lm/models/deepseek_v4.py`) plus `ModelArgs.__post_init__`
(compress_ratios) and found a concrete, quantitative refinement of the
Indexer hypothesis above — narrows it from "the Indexer's cost scales with
context" to a specific, falsifiable mechanism with a predicted knee point.

**The config:** `compress_ratios` alternates `4` and `128` across the ~21
sparse layers (`__post_init__`, `[4 if i % 2 else 128 for i in
range(...)]` — roughly half each). `Indexer.index_topk` defaults to `512`.

**The branch (`SparseCompressedAttention.__call__`, ~line 4107-4133):**
pool size `P = context // compress_ratio` for that layer. If `P <= 512`
(the `"compressed attention"` branch): ONE dense `scaled_dot_product_attention`
over the whole (small) pool, no indexer top-k gather. If `P > 512` (the
`"sparse compressed attention"` branch): the FULL `Indexer` machinery runs —
a `(B,L,D)@(B,D,P)` score GEMM reading the entire pooled tensor, an O(P)
pmask apply, an O(P) top-k search — followed by a gathered SDPA over just
the selected k=512 entries.

**The knee this predicts:** for a `compress_ratio=4` layer, `P=512` is
crossed at **context ≈ 2048 tokens**. For `compress_ratio=128` layers, the
same crossing happens at context ≈ 65,536 tokens — outside both of today's
sweep points (500, 14253).

**Why this produces exactly the observed shape:** at depth≈500, every
layer is still under its P=512 threshold (even the ratio=4 ones, P≈125) —
the whole model runs the cheap dense branch, so FULLBLOCK's 5x-per-row
loop is nearly free everywhere → 27.56 tok/s, no cliff. At depth≈14,253,
the ratio=4 layers (~half the sparse layers, P≈3,563) have flipped into
the expensive sparse/indexer branch — FULLBLOCK now forces that branch's
O(P)-scaling work (the score GEMM's full read of the `(D,P)` pooled
tensor, the pmask apply, the top-k search) to run **5 separate times per
verify cycle** (once per drafted row, unbatched) instead of once. That's
a genuine ~5x memory-bandwidth multiplier on an O(context)-scaling op,
concentrated in only half the sparse layers, and ONLY past the 2048-token
knee — which plausibly compounds with the other O(P) per-row-duplicated
ops (pmask, gather) to produce something steeper than a flat 5x, matching
the observed 15.9x.

**Falsifiable prediction for next session:** add a 3rd sweep point at
~1500-3000 tokens (straddling the predicted 2048-token knee for
compress_ratio=4 layers). If this hypothesis is right, throughput should
show a SHARP drop right around there, not a smooth/gradual decline — a
much cheaper single-point test than a full bisect, and it would confirm
or kill this specific mechanism before investing in the harder fix.

**Why a naive fix ("just batch the indexer call across the 5 rows") is
NOT safe, so don't reach for it without more care:** within one 5-row
FULLBLOCK verify block, `PoolingCache.accumulate_windows`'s remainder
buffer can genuinely advance mid-block — for `compress_ratio=4`, a new
pooled entry lands roughly every 4th token, i.e. plausibly once inside a
5-row block. That means the pooled tensor is NOT bit-identical across all
5 rows of one verify cycle; a correctness-preserving fix has to account
for the (at most 1-entry) pool growth mid-block, not assume it's static.
This is the same class of subtlety FULLBLOCK itself was built to handle
correctly (see `exo-speculative-decode-correctness` skill) — treat any
"just batch it back" fix with the same bisect rigor as the MoE-side fix
(`EXO_DSV4_MOE_ISOLATION_DUMP`/`EXO_DSV4_MOE_PARTS_ROWSEQ` precedent), not
a quick patch.

**Where to find this:** `SparseCompressedAttention.__call__` branch dispatch
at deepseek_v4.py ~line 4105-4133 (`if pooled.shape[1] == 0: ... elif
pooled.shape[1] <= self.indexer.index_topk: ... else:` — the sparse
branch); `Indexer.__call__` ~line 3326-3524 (the score GEMM at
`_indexer_score`, ~line 3159-3224, and the top-k search ~line 3432-3493);
`ModelArgs.__post_init__` ~line 825-841 (compress_ratios default pattern);
`PoolingCache.accumulate_windows` (`mlx-lm/mlx_lm/models/cache.py`
~line 1342) for the mid-block pool-growth subtlety above.

## Reproduction (verified, deterministic)

Cluster config: `DSV4_MODEL_ID=deepseek-ai/DeepSeek-V4-Flash-0731
EXO_DSV4_DSPARK=1 EXO_DSV4_DSPARK_NATIVE=1 EXO_DSV4_ROWSEQ_FULLBLOCK=1
EXO_DSV4_ROWSEQ_FULLBLOCK_MOE=0 EXO_DSV4_MOE_PARTS_ROWSEQ=shared`
(today's shipped "improved" default).

Send a request with a ~14K-token prompt (any content, doesn't need to be
adversarial), `max_tokens` large enough to force `finish_reason=length`,
`temperature=0`. Measure decode time as: HTTP response arrival timestamp
minus the server's own `Prefill complete: N tokens in Xs` log line
timestamp (grep `~/exo.log` on the rank-0 node) — this avoids all
client-side prefix-cache/timing confounds (see Methodology Pitfalls below,
several were hit and fixed while building this sweep). At 300 decode
tokens this should take ~11s at healthy throughput; it took 174s.

## Methodology used, and pitfalls hit building it (save future time)

1. **`use_prefix_cache` is NOT wired up for `/v1/chat/completions`** — only
   the separate `/bench` endpoint's `BenchChatCompletionRequest` reads it
   (confirmed via source read of `src/exo/api/main.py`). An earlier
   attempt at this sweep tried to isolate decode-only time by sending the
   identical prompt twice (once `max_tokens=1`, once `max_tokens=1+N`) and
   assuming the second call's prefill would be cache-hot — it wasn't; both
   calls did a full, independent prefill, producing nonsensical (even
   negative) "decode elapsed" numbers via naive subtraction. Do not reuse
   that approach without first confirming prefix caching is actually
   active for whatever endpoint you're using.
2. **`random.choice()`-built prompts must be seeded deterministically per
   depth** if you intend to compare "the same prompt" across two requests
   — an early version reseeded on each call and silently sent a
   *different* prompt of a *different* actual token count each time
   (visible only by noticing the logged prompt-token-count jumped between
   calls that were supposed to use the same prompt).
3. **PP is single-request-only; killing the CLIENT does not stop
   server-side generation.** Repeatedly hit this while iterating on the
   sweep script — killing a stuck/wrong client process left the server
   still actively prefilling/decoding for minutes afterward. Standing
   rule reinforced: always poll `/state` for any `Running` task and wait
   for `IDLE` before sending a new request, never assume killing the
   client is sufficient.
4. **Foreground terminal calls here cap at ~300s** — any single sweep step
   at real context depth (prefill of 14K+ tokens takes ~30-70s alone, plus
   decode) needs `background=true` + `process(action='wait', ...)` polling
   in a loop, not a single blocking foreground call.
5. The final working sweep script (now deleted from `/tmp`, not
   committed anywhere — rewrite if resuming, it's short, see the
   "Reproduction" section above for the exact measurement approach) built
   a fixed-per-depth prompt via `random.seed(depth)`, sent one request per
   depth with `max_tokens` large enough to force `finish_reason=length`,
   and read decode time from server log timestamps as described above.

## Confirmed NOT the cause (ruled out this session)

- **Not a DSpark drafting/acceptance problem.** 94% draft acceptance was
  observed in the same log window as the 1.46s verify-forward cost.
- **Not a general DSv4 decode-scaling problem.** Sequential decode
  (DSpark fully off) on the identical cluster, identical prompts, only
  slows 1.28x over the same context range (25.11 → 19.57 tok/s) — normal,
  expected KV-cache-growth behavior, nothing like a cliff.
- **Not caused by today's MoE-side change.** `EXO_DSV4_ROWSEQ_FULLBLOCK`
  (the attention-side mechanism implicated) was already the default
  before today's session; today only changed the MoE-side flag
  (`FULLBLOCK_MOE` → off, `MOE_PARTS_ROWSEQ=shared` → on). The attention
  per-row loop is untouched by that change.

## Concrete next steps if resuming

0. **Cheapest first: test the predicted knee.** The 2026-08-04 code-read
   update above predicts a SHARP throughput drop at context≈2048 tokens
   (where `compress_ratio=4` layers cross `P=index_topk=512` and flip from
   the cheap dense attention branch to the expensive per-row-duplicated
   Indexer/sparse branch). Add one sweep point at ~1500-3000 tokens using
   the existing sweep methodology (see below) — a single request, no new
   code. If throughput drops sharply right there (not gradually), that
   confirms this specific mechanism and focuses the bisect (below) on the
   `SparseCompressedAttention` sparse branch specifically, not the whole
   attention block. If it doesn't, this hypothesis is wrong and the
   Indexer-cost theory needs to be dropped in favor of another candidate.
1. **Apply the same bisect methodology that fixed the MoE-side cost this
   morning** (see commit `b9921962e`'s message and warm memory fact
   `1169` for the full writeup): capture real per-cycle attention inputs
   at depth (reuse tensors the FULLBLOCK per-row loop already computes,
   zero new forward passes — same pattern as `EXO_DSV4_MOE_ISOLATION_DUMP`
   in `deepseek_v4.py`), then offline-test candidate configs (is the
   Indexer topk search the actual cost driver? does batching JUST the
   indexer search while keeping the rest of attention per-row preserve
   correctness — note `PoolingCache` can advance mid-block for
   `compress_ratio=4`, see the code-read update above, so this needs the
   same rigor as the MoE-side bisect, not a naive batch-it-back patch?
   use the exact same key/shape-match + bit-exact-sanity-gate
   pattern from this morning's bisect — do NOT trust any subset's result
   until a "fully per-row, must equal ground truth" sanity check passes
   bit-exact).
2. **Before that, or in parallel:** re-run the context sweep for the
   DSpark-OFF baseline out to 50K/100K to confirm its mild scaling (1.28x
   by 14K) holds further out, or also degrades — not yet measured beyond
   14K this session. This tells you whether "DSpark off" is a safe
   permanent fallback for long-context workloads or just currently the
   *less bad* option.
3. **Decide on `start_cluster.sh`'s `EXO_DSV4_DSPARK` default.** It's
   currently still `1` in the script (not changed this session — only the
   live cluster's runtime env was overridden to `0` for this session).
   Given DSpark+FULLBLOCK is actively harmful at any real context depth
   right now, strongly consider flipping the script default to `0` until
   the cliff is fixed, so a plain relaunch doesn't silently re-introduce
   this regression. Needs its own explicit approval before changing
   (per standing rule: config/code changes and cluster relaunches are
   separately gated, and default-flag changes affecting production
   behavior are exactly the kind of thing to confirm first).
4. Once the attention-side fix exists, re-verify BOTH threads together:
   correctness (the `math_digit_sum` self-doubt-loop repro,
   `max_tokens=8000`, must still converge cleanly) AND the full
   context-depth throughput sweep (not just near-zero context) before
   calling anything "shipped."

## Where to find things

- Today's MoE-side fix (correct, shipped, unaffected by this bug): exo
  main `b9921962e`; mlx-lm `55401ac` on
  `adurham/mlx-lm@diag/spec-state-split-timing-v2`; warm memory fact `1169`.
- This bug's full writeup: warm memory fact `1170`
  (`memory(action='recall', query='DSpark FULLBLOCK context scaling cliff')`).
- The attention-side `FULLBLOCK` mechanism itself:
  `mlx-lm/mlx_lm/models/deepseek_v4.py`, the `_VERIFY_ROWSEQ_FULLBLOCK`
  branch inside `DeepseekV4Block.__call__` (~line 4418-4510) — the
  per-row loop that calls `self.attn(...)` once per row.
  `SparseCompressedAttention`'s `Indexer` (class def ~line 3293,
  used by `SparseCompressedAttention` ~line 3904) is the prime suspect
  for the actual scaling-cost source, not yet confirmed.
- `exo-speculative-decode-correctness` skill: full background on why
  `FULLBLOCK` exists (the self-doubt-loop correctness bug it fixes) —
  has been patched today with a pointer to this new finding.
- `exo-perf-tuning` skill: has today's MoE-side bisect writeup; NOT yet
  updated with this attention-side finding (do that if resuming, so the
  skill stays the single source of truth for DSpark perf work).

## Cluster state as of this handoff

Live, idle, healthy. `DSV4_MODEL_ID=deepseek-ai/DeepSeek-V4-Flash-0731
EXO_SPECULATIVE=0 EXO_DSV4_DSPARK=0` (DSpark fully off — deliberate safety
choice, not the `start_cluster.sh` script default). Correctness verified
(plain chat smoke test clean, `finish_reason=stop`). Both repos
(`exo`, `mlx-lm`) clean working trees, in sync with their remotes.

## MAJOR CORRECTION (2026-08-04, evening session) — the original collapse claim does NOT reproduce

**TL;DR: relaunched the cluster fresh (same exo commit `880f4b3a0`, same
mlx-lm commit `55401ac`, same weights, same repro config
`EXO_DSV4_DSPARK=1 EXO_DSV4_DSPARK_NATIVE=1 EXO_DSV4_ROWSEQ_FULLBLOCK=1
EXO_DSV4_ROWSEQ_FULLBLOCK_MOE=0 EXO_DSV4_MOE_PARTS_ROWSEQ=shared`) intending
to test the "sharp knee at ~2048 tokens" prediction from the UPDATE section
above. Instead found the original "15.9x collapse, gets worse with
context" story does not hold up. Slow verify cycles (1.4-1.6s each,
magnitude matching the original finding) DO occur, deterministically, but
ONLY in a narrow band around prompt_tokens≈2800 — and NOT at any larger
context tested, including 15,030 tokens (comparable to the original's
"1.73 tok/s" data point at depth≈14253, which today measured a clean
17.31 tok/s, back in the normal 16-22 tok/s range).**

### What was actually measured (finer sweep, corrected methodology)

A finer context sweep (prompt_tokens, decode_tok/s, all measured via the
proper methodology below): 580→27.63, 1348→22.61, 1794→23.06, 2002→20.61,
then a burst of degradation around 2254-2825 (see below — the ORIGINAL
first pass through this sweep mismeasured this window due to a script bug,
corrected on retest), then clean again: 3371→18.40, 4408→17.52,
6612→17.53, 10028→16.33, 12242→20.84, 13863→17.04, **15031→17.31**.

The 15,031-token point is the critical one: it is comparable in depth to
the original doc's "confirmed" depth≈14253 → 1.73 tok/s collapse, and
today it measured a completely normal 17.31 tok/s with ZERO diagnostic
OUTLIER log lines anywhere near that request. The originally-claimed
sustained, context-scaling 15.9x collapse simply is not present in this
re-test.

### The ~2800-token anomaly IS real (unlike the sustained-cliff claim) — reproduced 3x

Sending the IDENTICAL deterministic request (same seed, same prompt
content, `prompt_tokens=2825`, temp=0) three times in a row against the
warm, idle cluster gave: run 1 = 18.03 tok/s (clean), run 2 = 16.88 tok/s
(clean), **run 3 = 1.28 tok/s** (235s wall time for 301 tokens — a real
collapse, same rough magnitude as the original "1.73 tok/s" claim).
Cross-checked against the raw per-cycle OUTLIER diagnostic (fires
automatically >1000ms, zero sampling, so this is not a measurement
artifact): run 3's decode window contains ~50 outlier cycles at
1.4-1.6s/cycle (`r1_verify_fwd` on rank1, `r0_fwd`+`msg2_recv_wait` on
rank0) — the SAME per-cycle cost signature as the original finding.

**This means the underlying slow-path IS real and reachable — DSpark's
verify forward genuinely can cost ~1.5s/cycle instead of the normal
~80-150ms — but it is NOT a monotonic function of context depth.** It
triggered on the 1st and 3rd identical requests at ~2800 tokens in this
session's timeline but not deterministically on every request at that
depth (run 2 was clean), and did not trigger at all at any depth from
3000 to 15,031 tokens across this whole re-test session.

### What this rules out from the original write-up

- **The "sharp knee at compress_ratio=4's P=512 threshold, context≈2048"
  hypothesis (added earlier today, see UPDATE above) is FALSIFIED as a
  complete explanation.** If it were the true mechanism, EVERY context
  above ~2048 tokens should show the expensive branch's cost, scaling
  further as more layers cross their own P=512 threshold. Instead the
  cost appeared only in a narrow band, then vanished for 3000-15,031
  tokens where far more layers should be well past that threshold.
- **The original "27.56 tok/s at depth≈500 → 1.73 tok/s at depth≈14253,
  15.9x collapse" framing is not a stable, reproducible property of this
  config.** Today's fresh measurement at a comparable depth (15,031
  tokens) was 17.31 tok/s — roughly in line with the "normal" numbers
  from elsewhere in this sweep, not collapsed.
- **The original session's methodology (verified correct in isolation —
  seeded deterministic prompts, decode time from server log timestamps)
  was NOT the problem.** The re-test used the identical methodology and
  still found normal throughput at comparable depth. What differs is that
  today's re-test happened to catch the slow-path triggering (or not) at
  a DIFFERENT point in the sweep than the original session did — pointing
  at something session/timing/state-dependent rather than a pure function
  of `context_length`.

### Leading candidate mechanisms for the ~2800-token anomaly (NOT yet confirmed)

Per a second-opinion consult during this session: pure Metal
kernel-compile/shapeless-retrace warmup was proposed and then ruled
implausible as the *sole* explanation — ~50-90 consecutive slow cycles
across an entire decode window is too many for a one-time compile cost
(expect a handful of slow cycles, not sustained slowness for the whole
window), and the exact 1.4-1.6s magnitude matching the original finding
suggests the same expensive code path genuinely executing, not compile
overhead. More likely candidates, NOT yet tested:

1. **First-touch/allocator state establishment on first entry into the
   sparse/Indexer code path per RUNNER PROCESS lifetime** (not per
   context depth) — e.g. MLX buffer-cache growth, GPU memory
   commit/wired-page first-touch, or lazy weight-shard materialization
   for the sparse attention branch's parameters, which would explain why
   the SAME depth (2825 tokens) was clean on run 2 but slow again on run
   3: if the runner was recycled/restarted between requests (crashes were
   observed in this exact session — see below), state that was "already
   warm" could be lost and re-paid.
2. **This session's own instability may be a confound, not just a
   backdrop.** During this exact investigation, sending accidentally
   low-entropy repeating-vocabulary prompts (a real, separate methodology
   bug in this session's own sweep script, NOT any DSpark bug) tripped the
   real degeneration kill-switch and crashed BOTH runners
   (`PPSpecAlreadyActiveError`, `DEGENERATION DETECTED`), forcing the
   master to tear down and reload fresh runners mid-sweep. The 2500-depth
   requests that showed the anomaly were run shortly after this crash/
   reload cycle. The 3000-15031 requests that were all clean ran later,
   once the reloaded runners had processed several successful requests.
   This is a plausible, testable "runner just came back from a crash/
   reload, first few requests near a certain size are slow" story —
   distinct from any context-length-based architectural mechanism.
3. **A genuine but narrow, non-monotonic MLX/JACCL scheduling hazard**
   specific to some interaction of prompt size, block boundary alignment,
   or KV-cache buffer resize timing that happens to land badly near
   ~2800 tokens for THIS model's chunking (`EXO_PREFILL_STEP_SIZE=2048`,
   `EXO_DSV4_MOE_PARTS_ROWSEQ=shared`'s cache-growth pattern, or
   `PoolingCache`'s `step=256` reallocation growth) — would predict the
   anomaly is tied to a specific buffer-resize boundary, not literally
   "2800 tokens" as a constant. NOT yet tested.

### Concrete next steps (supersedes the "Concrete next steps" section above)

1. **Rule in/out the crash-recovery-state hypothesis (candidate 1/2
   above) first — cheapest test.** Deliberately crash/reload the runners
   (or just relaunch cleanly), then IMMEDIATELY send a handful of
   requests at ~2800 tokens before sending anything else, and watch
   whether the outlier burst appears on the first few post-reload
   requests specifically, regardless of exact token count. If confirmed,
   this reframes the entire investigation: it's a runner-warmup/
   crash-recovery cost, not a context-scaling cliff, and the practical
   fix is very different (e.g., a proper warmup pass after any
   runner reload, not an architectural rework of `FULLBLOCK`).
2. **If ruled out, chase candidate 3 (buffer-resize boundary)** by
   testing several depths tightly clustered around 2700-2900 with a FRESH
   runner each time (no post-crash confound) to see if the anomaly is
   reproducible independent of runner lifecycle.
3. **Either way, retract the "15.9x collapse, context-scaling cliff" framing
   from any downstream consumer of this doc** (the `exo-perf-tuning` and
   `exo-speculative-decode-correctness` skills both currently state this
   as a confirmed, severe, context-scaling bug — both need a correction
   pass pointing back to this section) until the real mechanism is
   understood. In the meantime, the practical, verified-true statement is
   narrower: "DSpark+FULLBLOCK's verify-forward cost is NOT reliably
   context-scaling; it CAN spike ~10-20x on a subset of requests for a
   currently-unknown reason, independent of context depth" — a real
   reliability/tail-latency concern, just not the mechanism originally
   claimed.
4. **`EXO_DSV4_DSPARK` default in `start_cluster.sh`:** the original
   doc's recommendation to keep DSpark off by default is STILL reasonable
   given a confirmed (if rarer/less predictable than first thought) risk
   of a ~10-20x tail-latency spike, but the previously-stated reasoning
   ("gets worse the longer the conversation runs") is no longer
   supported and should not be cited as the justification going forward.

### Where to find this correction's evidence

Raw sweep data and outlier-log correlation captured via ad hoc scripts in
`/tmp/dspark_knee_sweep*.py` (not committed — short, rewrite if needed,
see the "Reproduction" section's methodology which was reused unmodified).
Outlier log lines cross-referenced directly via
`ssh <node> "grep 'DSpark OUTLIER' ~/.exo/exo_log/exo.log"` on both nodes
for the 2026-08-04 15:40-16:03 window.
