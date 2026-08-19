moe.all_sum confirmed as the dominant TP prefill cost: ~61-64% of wall time (2026-08-19)
============================================================================================

Summary
-------

Using the existing NOP-ablation infrastructure (`/tmp/dsv4_nop_targets`,
file-toggle, no relaunch needed -- `mx.distributed.all_sum` becomes an
identity pass-through when "all_sum" is in the target set), measured the
real cost of the per-layer TP all_sum collective on the live 2-node
cluster at standing config (EXO_PREFILL_STEP_SIZE=2048, SEQ_SPLIT=1).

**Result, confirmed at TWO depths with warm-vs-warm comparisons (ruling
out a cold-start/JIT-warmup confound):**

| depth | baseline (all_sum ON) | NOP (all_sum OFF) | speedup | all_sum share of wall time |
|---|---|---|---|---|
| 12,066-12,069 tok | 162.5 tok/s | 427.5 tok/s | 2.63x | 62.0% |
| 38,066-38,067 tok | 167.3 tok/s | 430.3 tok/s | 2.57x | 61.1% |

Both NOP runs completed prefill cleanly (12069/12069 and 38067/38067
tokens respectively) before crashing during DECODE (HTTP 500, "Runner
shutdown... signal=9" pattern) -- expected, since NOP'ing all_sum
breaks cross-rank correctness by design (this is a diagnostic-only mode,
documented in deepseek_v4.py's own comment: "Output is GARBAGE -- bench
tok/s only. Quality intentionally broken."). The baseline runs completed
end-to-end with correct secret-code recall, confirming they're a valid
reference.

This is a MUCH larger effect than a prior stale skill note claimed
("Comms cost is significant (~12%) at low context (50K) but effectively
zero at high context (300K+)") -- that note is either measuring a
different quantity (e.g. total comms including all_gather, at a
different chunk-size config, or predates a later architectural change)
or is simply wrong. Do not trust it going forward without re-verification
at the SAME depths this session tested.

Method
------

1. Confirmed cluster healthy at standing config before starting.
2. Ran a genuine cold-start baseline at 12K tokens (all_sum active) --
   152.7 tok/s. Suspicious of cold-start confound (JIT/compile warmup
   on first request), so re-ran a WARM baseline (3rd request on the
   already-warmed cluster) at the same depth -- got 162.5 tok/s,
   confirming the cold number was NOT badly confounded (both landed in
   the same ballpark), but used the warm number as the fair comparison
   point going forward.
3. Set `/tmp/dsv4_nop_targets` containing "all_sum" on BOTH nodes (must
   be set on both -- it's a per-process file read, cached 1s).
4. Ran a genuinely FRESH prompt (different secret code, so no KV-cache
   hit) at the same token depth -- confirmed `cached_tokens: 0` in the
   response to rule out a cache-hit artifact (caught and corrected one
   false-positive cache-hit result during this session before landing
   on the real fresh-prefill numbers).
5. Repeated the whole baseline/NOP pair at 38K tokens to confirm the
   effect isn't a shallow-context-only artifact -- it holds, within 1
   percentage point of the 12K result.
6. Cleared `/tmp/dsv4_nop_targets` on both nodes immediately after each
   NOP test -- this flag must NEVER be left set, since it silently
   produces wrong (garbage) output.

Why this wasn't found earlier tonight
-----------------------------------------

Every other investigation tonight targeted the MoE GEMM kernel and
attention SDPA kernel -- genuinely real, correctly-measured effects, but
smaller (28% MoE-kernel gap, ~2x SDPA scaling with L). Nobody had
isolated the TP collective itself with a controlled NOP ablation at a
REPRESENTATIVE prefill-shape depth this session -- the skill's stale
comms-cost note was accepted without re-verification, which in
retrospect was exactly the kind of un-re-checked inherited claim that
turned out wrong multiple times tonight (the 82 TFLOP/s figure, the
July tile sweep, the "4096 breaks quality" claim). This is the same
lesson again: re-verify old comms-cost claims, don't inherit them.

What this does NOT mean
---------------------------

`all_sum` cannot simply be deleted -- it's the mechanism that reduces
each TP rank's sharded partial MoE output into the correct full result.
Removing it breaks correctness (confirmed: NOP runs crash on decode).
The real question is WHY it costs this much and whether the cost is
reducible without breaking correctness -- e.g. payload size, dtype
(fp32-vs-bf16 downcast already exists per `_collective_fp32_safe` in
deepseek_v4.py), transport/RDMA efficiency, per-layer call count (43
per forward pass), or whether it's genuinely bandwidth-bound on the
actual Thunderbolt 5 RDMA link and closer to a hard ceiling than the
GEMM-kernel work was.

Next step
------------

Consult on a concrete plan to attribute WHERE inside all_sum's cost
lives (payload size vs transport efficiency vs per-call fixed overhead
vs genuine RDMA bandwidth ceiling) before proposing any fix, per the
standing "measure before guessing" discipline from tonight's other
threads.
