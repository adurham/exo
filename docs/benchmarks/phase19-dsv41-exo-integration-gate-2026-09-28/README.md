# Phase 19 -- exo integration plan + the prefill blocker

Date: 2026-09-28. Fable consult on integration architecture, then the gate it
named first. Production stopped cleanly for the run and restored.

## Integration architecture (consult verdict, adopted)

Dedicated Engine + custom load branch for DSv4.1; do NOT force it through
mlx_generate / ExoBatchGenerator / KVPrefixCache / tensor_auto_parallel. The
model has its own cache, its own TP inside exl3_build, is not loadable via
mlx_lm load_model, and has no c>=2 / prefix cache / logprobs. Reuse exo's
tokenizer, chat template, tool parser, chunk plumbing and model cards.

Gates before flipping the start_cluster.sh default: greedy parity harness;
100K end-to-end Hermes session latency vs production; memory peak with
vision + long context; sampling in the spec path; cancel/rollback stress;
Adam signs off on the numbers.

## First gate measured: long-prompt prefill -- BLOCKER

| prompt | time | prefill tok/s | peak GB/rank |
|---:|---:|---:|---:|
| 2,048 | 206 s | 9.9 (includes first-run compile/warmup) | 108.3 |
| 8,192 | 111 s | 73.7 | 108.6 |
| 16,384 | -- | crashed: Metal GPU Timeout (527 errors) despite MTL_DISABLE_TIMEOUT | -- |

Production V4 prefills ~430 tok/s at 100K and reuses cached prefixes across
turns. DSv4.1 today: ~74 tok/s at best, crashes above 8K, no prefix cache. A
100K Hermes turn would take ~23 min even if it did not crash. DSv4.1 cannot
be the default until prefill is fixed (speed + stability) and session KV
reuse exists.

Likely causes, unverified: the reference sparse attention / indexer build
[chunk x window+compressed] gather tensors per chunk (grows with context) --
one command buffer then runs past the GPU watchdog; EXL3 prefill-shape paths
(R=512) were never tuned here.
