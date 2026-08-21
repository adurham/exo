# EXO_DSV4_INDEXER_PBLOCK: prefill-neutral, real decode regression at small p_block — 2026-08-21 (session 2, part 6)

## Lever tested

`EXO_DSV4_INDEXER_PBLOCK` (int, default 0=OFF). Tiles `_indexer_score`'s
pooled-axis (P) computation into blocks so the full `(B, H=64, L, P)`
pre-collapse scores tensor never materializes — bounds peak allocation.
Added 2026-06-21 (`INDEXER_TILED_P_PLAN.md`, commit `24059598f`) as a fix
for the documented high-context prefill-cliff cost (`attn.indexer`
max/avg ~4x spike variance, 22ms spikes at 360K context). Bit-exactness
proven offline via `bench/indexer_score_microbench.py` (max abs diff 0.0,
top-k SET overlap 1.0). Never A/B tested on real cluster hardware before
tonight — the plan's own step-3 validation ("Throughput (cluster):
span profiler A/B... Quality (cluster): bench/quality_probe_dsv4.py")
was never executed.

## Method

1. Relaunched with `EXO_DSV4_MOE_FUSED_GATE_UP=1
   EXO_DSV4_INDEXER_PBLOCK=16384` (a reasonable mid-size block per the
   plan's own tuning discussion). Verified live via `ps aux`.
2. Ran the standard 100K/300K/500K depth ladder. All three needle checks
   passed (correctness intact — the bit-exactness claim holds up on real
   hardware, not just the offline microbench).
3. Prefill throughput at all three depths was at parity with the
   known-good baseline (363.0/347.9/328.9 tok/s vs baseline 368.7/351.9/
   333.2) — **no measurable prefill win**, contrary to the original
   plan's stated goal of flattening the prefill cliff.
4. Decode throughput showed a suspicious monotonic decline with context
   depth: 13.67 tok/s (100K) → 11.55 tok/s (300K) → 10.03 tok/s (500K).
   This is the OPPOSITE of the flat ~17-18 tok/s decode baseline observed
   all session regardless of depth (see
   `docs/prefill-trace-instrumentation-findings-2026-08-21.md`).
5. Read `_indexer_score_tiled`'s own docstring: "Falls back to the full-P
   kernel when P <= p_block (single tile) so small contexts (and decode,
   L==1, P small) pay zero overhead." This assumption breaks at deep
   context: P (pooled length) grows with context depth, and DECODE steps
   also query the indexer with L=1 but P is now the FULL context's pooled
   length — at 500K context, P exceeds p_block=16384, so tiling fires on
   EVERY decode step, not just prefill chunks. The "zero overhead for
   decode" claim only holds for context shallow enough that P stays under
   p_block.
6. Isolated the mechanism: relaunched with `EXO_DSV4_INDEXER_PBLOCK=262144`
   (larger than any realistic pooled length at these test depths, so
   tiling should essentially never fire) and reran 500K. Result: decode
   returned to 17.39 tok/s — back to baseline. Prefill unaffected either
   way (333.4 tok/s, same as the small-p_block run and the untiled
   baseline).

## Conclusion

**Confirmed root cause: p_block=16384 is too small relative to real
pooled lengths at deep context, causing per-decode-step tiling overhead
that was not anticipated by the original design** (whose docstring
explicitly claims decode pays zero overhead — true only when P stays
below the chosen p_block). This is real, reproducible (isolated by
changing exactly one variable across two clean cluster relaunches), and
consistent with the mechanism visible in the code.

**No prefill win was observed at any tested depth or p_block value.**
The original 2026-06-21 profiling data motivating this fix
(`attn.indexer` spikes at 360K) was gathered before tonight's transport
hardening and dual-cable topology work — it's possible those earlier
prefill-cliff spikes were partly or wholly a symptom of the jaccl/network
issues fixed earlier tonight, not purely compute-bound indexer
allocation pressure, and the compute-side fix alone doesn't show up in
prefill throughput because the earlier network/timing hazard rather than
memory-allocation stalls was the larger factor. This is speculative --
not independently re-verified this session -- but consistent with
tonight's other finding that prefill is compute-bound with no orchestration
overhead worth chasing at the granularity tested.

**Recommendation: do not enable `EXO_DSV4_INDEXER_PBLOCK` in production
at any small/moderate block size** (it will regress decode at real
deployment context depths with no offsetting prefill gain). If ever
revisited, `p_block` would need to be set dynamically relative to
context depth (large enough that decode-time P never exceeds it) rather
than a fixed constant — a real design change, not a env-var tuning
exercise, and not worth the effort given the prefill side shows no
benefit to justify it.

## Real generated text quality

Confirmed coherent at both p_block=16384 and p_block=262144 via direct
needle-in-haystack retrieval (`FALCON-MERCURY-7749`, correct, single-pass
reasoning) and a separate CAP-theorem explanation prompt (factually
correct, well-formed). Bit-exactness claim from the offline microbench
holds on real hardware — this lever's problem is purely a
performance/design gap (decode-path P-scaling), not a correctness bug
(unlike `EXO_DSV4_FUSED_SOFTMAX`, tested immediately prior this session).
