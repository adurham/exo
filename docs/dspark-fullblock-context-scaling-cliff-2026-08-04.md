# DSpark FULLBLOCK context-scaling cliff — handoff — 2026-08-04

**STATUS: BUG FOUND AND CONFIRMED. NOT ROOT-CAUSED. NOT FIXED.**
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

1. **Apply the same bisect methodology that fixed the MoE-side cost this
   morning** (see commit `b9921962e`'s message and warm memory fact
   `1169` for the full writeup): capture real per-cycle attention inputs
   at depth (reuse tensors the FULLBLOCK per-row loop already computes,
   zero new forward passes — same pattern as `EXO_DSV4_MOE_ISOLATION_DUMP`
   in `deepseek_v4.py`), then offline-test candidate configs (is the
   Indexer topk search the actual cost driver? does batching JUST the
   indexer search while keeping the rest of attention per-row preserve
   correctness? use the exact same key/shape-match + bit-exact-sanity-gate
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
