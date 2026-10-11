# Task B raw results summary

Harness: `bench/p20_pmp_taskB_opclass.py` (staged `/tmp/p20_pmp_B/` on studio2).
Runs under `bench/phase20_guard.py watch` (idle-guarded). Production
`gemv_metal.py` md5 649d2d6a0a68c8d6eab2a8a5394b7ab9 identical BEFORE/AFTER.

Real EXL3 experts (production `EXL3_MM_SEG=v19d`) + production dense
(`DSV41_DENSE=affine6`, `DSV41_DENSE_POLICY=layers.*.ffn.shared_experts.*=q8g64`),
TP2 rank0/world2 shard shapes, chunk 2048. all_sum stubbed to identity (no RDMA).
Attribution = eval-fenced module wrappers with a running gpu_time counter
(phase-22 methodology). `mx.metal.gpu_time_ns()` reset/read verified on-node
(one 4096^3 bf16 matmul: gpu 12.67 ms < wall 13.81 ms).

## Per-layer closure (unbracketed wall vs Σ(bracketed))

ctx 16384:
L0  ratio0        busy 133.2  wall 112.0  Sbrk 110.7  gap_wall  -1.2%
L1  ratio0+engram busy 205.1  wall 585.3  Sbrk 176.4  gap_wall -69.9%  (SSD-bound)
L2  ratio2 kv+idx busy 239.2  wall 229.1  Sbrk 217.4  gap_wall  -5.1%
L3  ratio2 cons   busy 149.6  wall 129.6  Sbrk 128.3  gap_wall  -1.0%
L20 ratio1 kv+idx busy 253.4  wall 244.8  Sbrk 233.1  gap_wall  -4.8%
L21 ratio1 cons   busy 129.8  wall 127.4  Sbrk 126.0  gap_wall  -1.1%  (MoE-down outlier)
L24 ratio1 idxsrc busy 239.3  wall 247.1  Sbrk 236.4  gap_wall  -4.3%

ctx 100352:
L0  busy 114.6 wall 112.2 Sbrk  92.0 gap_wall -18.0%
L1  busy 206.8 wall 584.1 Sbrk 154.1 gap_wall -73.6%
L2  busy 350.2 wall 342.1 Sbrk 346.8 gap_wall  +1.4%
L3  busy 133.0 wall 130.7 Sbrk 111.5 gap_wall -14.7%
L20 busy 475.4 wall 471.4 Sbrk 439.6 gap_wall  -6.8%
L21 busy 147.8 wall 128.5 Sbrk 127.4 gap_wall  -0.9%
L24 busy 478.7 wall 473.2 Sbrk 474.6 gap_wall  +0.3%

## Weighted 40-layer fold (layer counts 1/1/2/15/1/15/4; L14~=L2)

ctx 16384: busy 6458.1 wall 6472.7 Sbrk 5933.3 gap_wall -8.3%
  MoE(down) 23.42% | sdpa 15.90% | indexer 13.48% | MoE(up) 9.90% |
  dense attn-o 8.76% | MoE(gate) 7.36% | HC 6.23% | dense qkv 5.20% |
  shared-expert 3.94% | MoE switch-total(glue) 2.18% | norms 2.13% |
  dense other 0.92% | routing gate 0.49% | compression 0.05% | engram 0.00%

ctx 100352: busy 7974.2 wall 7975.4 Sbrk 7207.6 gap_wall -9.6%
  indexer 31.10% | MoE(down) 19.10% | sdpa 13.59% | dense attn-o 7.21% |
  MoE(up) 7.17% | HC 5.16% | dense qkv 4.28% | MoE switch-total(glue) 3.25% |
  shared-expert 3.24% | MoE(gate) 2.95% | norms 1.68% | dense other 0.73% |
  routing gate 0.45% | compression 0.08% | engram 0.01%

## all_sum census
2 calls/forward, 41.9 MB/forward per layer (attn [1,2048,5120] + MoE [2048,5120] bf16).
Analytic: MoE-only 839 MB/chunk = 84-168 ms; attn+MoE 1678 MB/chunk = 168-336 ms
(@ 10/5 GB/s raw/eff + 40x20us latency). By-difference (16K): 8380-6473 = ~1907 ms.
RANGE ~0.08-1.9 s/chunk (~1-24%).

## Files
- `B_16k.json`, `B_100k.json` — full raw blobs (per-layer tables, env, bounds).
- `B_test.json` — 1-layer ctx4096 real sanity run.
- `run_remote.sh` — node-side runner (args ctx layers reps out).
- `analyze_B.py` — fold/aggregation helper.
