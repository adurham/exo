
=== OPT-4 RESULT (2026-06-13) — GOAL EXCEEDED ===
Two-level chunking: EXO_PREFILL_STEP_SIZE=256 super-chunk (proj/MoE weight-amortization) + tile sparse SDPA gathered tensor at EXO_DSV4_SPARSE_SDPA_TILE=128 so it never blows up. mlx-lm 7fd9289 / exo 4c8af728.
Prefill: 317.8 tok/s at chunk 256 WITH profiler on (drags ~5%) — clean number expected ~330+. Quality PASS. vs 236 baseline = +35%. CROSSED 300 GOAL.
Progression: 236 -> 280 (OPT-3 seq-split 21 layers) -> 290 (OPT-3b +20 compressed) -> 317.8 (OPT-4 two-level chunk 256, profiled).
Profiler at 256 confirms design: ffn experts=96% of ffn (weight-amortized), attn balanced (proj_qkv 24.5/indexer 30.2/sdpa 24.8/out_proj 17.4), no blowup. Raw 256 WITHOUT tiling was 120 tok/s — the tile is what makes the big chunk viable.
TODO: clean measure (profiler off, ~330+); set chunk 256 default; sweep 384/512 for more; verify decode unchanged.

=== CORRECTED VERIFIED RESULTS (2026-06-13, fixed wall-cross-checked harness /tmp/measure_clean.py) ===
The earlier 236->280->290->317 numbers came from a BUGGY harness (fire_and_measure scraped the LAST "Prefill complete" log line via grep|tail-1, which caught stray MTP/probe/co-host prefill lines — phantom tok/s, incl the bogus 317 "crossed 300"). Root cause + fix in measure_clean.py: match the log line to the response usage.prompt_tokens AND cross-check log-tps vs wall-tps (flag >30% disagree).

TRUE verified numbers at ~25K ctx (3 distinct prompts each, all wall-consistent within 1%):
- Baseline (seq-split OFF, chunk 128): ~258 tok/s (252/261/261)
- seq-split ON (OPT-3/3b, chunk 128): ~285 tok/s (274/288/288) = +10-11% REAL WIN, shipped default
- OPT-4 chunk 256: ~140 tok/s (118/172/123) = REGRESSION, reverted (exo ff1d3f42)

VERDICT: sequence-split attention is a verified +10% prefill win (258->285), quality-clean, decode-safe, default-on. NOT +35%/300 (that was measurement error). OPT-4 two-level chunking is NOT a win on this hardware. Chunk default reverted to 128.
LESSON: never trust a single un-replicated number or a log-scrape tok/s; always cross-check against wall time and replicate with distinct (cache-busting) prompts.

=== seq-split v2 VERIFIED (2026-06-13, fixed harness) ===
v2 shards proj_qkv (main wq_b) + the indexer score GEMM/pmask/topk to the per-node row band (compressor+pool+kv stay full/coherent). mlx-lm d8e2751 / exo 5e055c0d, default-on.
VERIFIED: 296/298/298 tok/s (3 distinct prompts, log_tps==wall_tps within 0.5%). Quality intact (needle recall "vault code is 7741" correct through filler; thinking mode).
FULL VERIFIED PROGRESSION: 258 (baseline, no split) -> 285 (v1: sdpa+o_proj) -> 298 (v2: +proj_qkv+indexer) = +15.5%. 298 = 99.3% of 300 goal.
NEXT: Fable idea #4 (8-bit indexer score-GEMM pooled-K operand) for the last ~2 tok/s to cross 300.

=== FINAL (2026-06-13): shipped 298 tok/s, full precision, +15.5% verified ===
DECISION: do NOT pursue quantization levers (Fable idea #4 8-bit indexer score GEMM, KV-cache bits, etc.) — keep everything bf16/fp16 for now. Quality integrity over the last 0.7% to 300. The shipped seq-split wins (v1+v2) are precision-EXACT (split query rows, all_gather back = bit-identical), not approximations, so they carry zero quality risk. fence=8 tested = NOISE (298-299, same as fence=4), NOT shipped (kept committed fence=4 default). Shipped config: EXO_COMPUTE_DTYPE=bf16, EXO_KV_CACHE_BITS=0, chunk 128, seq-split v2 on. mlx-lm d8e2751 / exo 5e055c0d. 298/300 = 99.3%; not cleanly crossed but a strong honest verified win.
