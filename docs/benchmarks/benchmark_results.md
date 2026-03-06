# Exo Benchmark Results

**Date:** 2026-02-25
**Model:** `mlx-community/MiniMax-M2.5-5bit`
**Cluster:** 3 nodes (2x Mac Studio M4 Max 128GB + MacBook Pro M4 36GB)
**Sharding:** Pipeline, MLX RDMA (MlxJaccl)

## Upstream Baseline (stock exo, FAST_SYNCH off)

| Size | Prompt Tokens | TTFT | Prefill (t/s) | Decode (t/s) | Output Tokens | Status |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| **Short** | 104 | **1.50s** | 42.5 | **38.23** | 1,949 | Pass |
| **Medium** | 1,754 | **5.83s** | 294.5 | **37.52** | 2,020 | Pass |
| **Large** | 6,595 | **15.75s** | 309.3 | **32.77** | 2,012 | Pass |
| **XL** | 13,734 | **26.75s** | 268.0 | **28.33** | 2,017 | Pass |

> FAST_SYNCH=on was also tested but deadlocked at ~1,800 tokens with a short prompt.

## Fork Results (latest, pipeline parallel 7/28/27 split)

| Context | Decode (t/s) | Notes |
|---------|:---:|-------|
| ~50 tokens | 44 | Exceeds 20 t/s target |
| ~5K | 26.5-29 | KV quant vs fp16 |
| ~20K | 19.6-20.9 | Borderline |
| ~34K | ~21 | fp16 only |
| ~50K | 12.5 | KV quant, below target |
| Prefill | 317-529 | Varies by context size |

## GPU Fence Fix Stress Test (3-node Hybrid TP+PP)

| Context | Cache Hit | Decode Speed | Status |
|---------|-----------|-------------|--------|
| 16K (cold) | 0% | 33ms/step (30 tok/s) | Pass |
| 50K | 78% | 46ms/step (22 tok/s) | Pass |
| 99K | 98% | 64ms/step (16 tok/s) | Pass |
| **104K** | **99%** | **67ms/step (15 tok/s)** | Pass |

5,795 total decode steps over 16 minutes. Zero deadlocks.

### Hardware at Peak (104K context)

| Node | GPU Power | Free RAM |
|------|-----------|----------|
| Mac Studio M4-1 (128GB) | 31W | ~5.8GB |
| Mac Studio M4-2 (128GB) | 56W | ~5.4GB |
| MacBook Pro M4 (36GB) | 0W (idle between PP stages) | ~4.2GB |
