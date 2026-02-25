# Exo A/B Benchmark Results

**Date:** 2026-02-25  
**Model:** `mlx-community/MiniMax-M2.5-5bit`  
**Cluster:** 3 nodes (2× Mac Studio M4 Max 128GB + MacBook Pro M4 36GB)  
**Sharding:** Pipeline, MLX RDMA (MlxJaccl)  
**FAST_SYNCH:** off  

---

## B-Side — Upstream (stock exo)

| Size | Prompt Tokens | TTFT | Prefill (t/s) | Decode (t/s) | Output Tokens | Status |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| **Short** | 104 | **1.50s** | 42.5 | **38.23** | 1,949 | ✅ |
| **Medium** | 1,754 | **5.83s** | 294.5 | **37.52** | 2,020 | ✅ |
| **Large** | 6,595 | **15.75s** | 309.3 | **32.77** | 2,012 | ✅ |
| **XL** | 13,734 | **26.75s** | 268.0 | **28.33** | 2,017 | ✅ |

> **Note:** FAST_SYNCH=on was also tested but **deadlocked at ~1,800 tokens** with a short prompt. All further testing uses FAST_SYNCH=off.

---

## A-Side — Fork (custom exo)

*Pending — results will be added after fork benchmark run.*

---

## Comparison

*Will be populated after both sides complete.*

| Size | Upstream TTFT | Fork TTFT | Δ TTFT | Upstream Decode | Fork Decode | Δ Decode |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Short | 1.50s | — | — | 38.23 t/s | — | — |
| Medium | 5.83s | — | — | 37.52 t/s | — | — |
| Large | 15.75s | — | — | 32.77 t/s | — | — |
| XL | 26.75s | — | — | 28.33 t/s | — | — |
