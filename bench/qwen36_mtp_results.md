# Qwen3.6-35B-A3B — MTP Speculative Decoding A/B

Measured impact of multi-token-prediction (MTP) self-speculative decoding on
`mlx-community/Qwen3.6-35B-A3B-8bit`, tensor-parallel across two Mac Studio
M4 Max (128 GB) nodes over RDMA/Thunderbolt 5 (MlxJaccl).

## Background

Qwen3.6 is a hybrid-attention MoE: 30 DeltaNet/SSM layers + 10 full-attention
layers, with a fused/stacked MoE expert layout (`experts.gate_up_proj`,
`experts.down_proj`) plus a shared expert. Out of the box on this exo fork,
MTP **failed to initialize** and the model silently ran pure autoregressive.
Three bugs had to be fixed before MTP worked end-to-end (commits on `main`):

| commit | fix |
|--------|-----|
| `73c5d474` | `EXO_KV_CACHE_BITS=0` sentinel leaked into `QuantizedKVCache(bits=0)` → ZeroDivision on the first full-attention layer |
| `25c1e1fb` | MTP MoE loader only understood the legacy per-expert layout; added a fused/stacked branch (splits `gate_up` at midpoint → `switch_mlp.*`, same transform as `mlx_lm.qwen3_5_moe.Model.sanitize`) |
| `b9fe0bfe` | MTP instantiated `type(layer.mlp)` which under tensor-parallel is the `ShardedMoE` wrapper, not the raw block → `'TextModelArgs' object has no attribute '__call__'` at forward. Unwrap via `.original_layer` |

(A fourth fix, `6d0e8a74`, repaired the master `place_instance` call that was
missing `node_backends`/`node_rdma_ctl`, unblocking programmatic placement.)

## Toggle

MTP is gated by `EXO_SPECULATIVE` (read at runner spawn from the exo process
env). `start_cluster.sh` defaults it to `1` when DSV4 is enabled; override at
launch:

```bash
EXO_SPECULATIVE=0 DSV4_ENABLED=0 ./start_cluster.sh   # MTP OFF (baseline)
EXO_SPECULATIVE=1 DSV4_ENABLED=0 ./start_cluster.sh   # MTP ON
```

There is no per-request or per-instance toggle — changing it requires a
cluster restart so the runner re-reads the env.

## Method

Identical prompt and methodology for both arms. Generation-TPS computed
`(n_tokens - 1) / (last_token_time - first_token_time)`, matching
`METHODOLOGY.md`. Quality gate per repo rule: needle-in-haystack recall +
special-token (BOS) spam scan — throughput is never reported without
confirming the model emitted real, coherent text.

- Short context: `bench/mtp_ab_probe.py` (~50-token prompt, 200 max tokens, 10 iters)
- 100K context:  `bench/mtp_longctx_probe.py` (~100K-token needle prompt, 200 max tokens, 3 iters)

## Results (2026-06-04)

### Short context (~50 tokens, 10 iters)

| metric | MTP-OFF | MTP-ON | delta |
|--------|--------:|-------:|------:|
| decode mean   | 89.94 t/s | 117.90 t/s | **+31.1%** |
| decode median | 89.63 t/s | 118.43 t/s | +32.1% |
| decode min/max | 88.89 / 91.12 | 112.64 / 121.52 | — |
| decode stddev | 0.825 | 2.908 | — |

### 100K context (3 iters)

| metric | MTP-OFF | MTP-ON | delta |
|--------|--------:|-------:|------:|
| decode mean   | 73.66 t/s | 103.52 t/s | **+40.5%** |
| decode median | 73.59 t/s | 102.22 t/s | +38.9% |
| decode min/max | 73.54 / 73.84 | 101.94 / 106.40 | — |
| decode stddev | 0.131 | 2.044 | — |
| needle recall | 3/3 | 3/3 | quality preserved |
| BOS-spam | 0/3 | 0/3 | clean |
| prefill ttft (one-time) | ~76 s | ~76 s | MTP doesn't touch prefill |

### Takeaways

- MTP yields **+31% short / +40% at 100K** decode throughput. The win grows
  with context because long-context decode is KV-bandwidth-bound, and
  speculation amortizes multiple tokens per verify pass.
- Zero distribution overlap at both sizes (MTP-ON min always beats MTP-OFF
  max) — the gain is real, not bistability/luck.
- Quality verified end-to-end: needle recalled from mid-document, coherent
  output, no special-token leakage. Not the throughput-clean / quality-dead trap.
- MTP-ON stddev is higher (speculative throughput varies with per-prompt
  draft acceptance rate); MTP-OFF is metronomic.

Hardware: 2× Mac Studio M4 Max 128 GB, RDMA over TB5, KV cache bf16
(`EXO_KV_CACHE_BITS=0`), γ=2.

## Future model bringup — recurring bug classes

Qwen3.6 was the **first hybrid-attention / fused-MoE model** run on this fork
(30 DeltaNet/SSM + 10 full-attention layers; fused `experts.gate_up_proj` /
`experts.down_proj` + a shared expert). That architecture is what exposed all
four bugs above — DSV4 (pure MQA, MLA caches) never tripped them. When bringing
up the **next** hybrid or fused-MoE model, expect these classes to recur and
check them first:

1. **KV-quant sentinel leak.** Any model with plain `KVCache` full-attention
   layers will hit the `EXO_KV_CACHE_BITS=0` path. The env path must map `0 →
   None` (bf16), else `QuantizedKVCache(bits=0)` → ZeroDivision on the first
   such layer. (`cache.py:_effective_kv_cache_bits`)

2. **MTP MoE weight layout.** The MTP loader must handle the model's expert
   layout. Fused/stacked (`experts.gate_up_proj` [E,2I,H] + `experts.down_proj`)
   needs a midpoint gate_up split → `switch_mlp.*`, mirroring the main model's
   `sanitize()`. Legacy per-expert (`experts.N.*`) is the other branch.
   (`mtp_module.py`)

3. **Tensor-parallel layer wrappers.** Under TP, `layer.mlp` is a
   `CustomMlxLayer`/`ShardedMoE` wrapper, not the raw block. Any code doing
   `type(layer.X)(args)` to clone a submodule must unwrap via `.original_layer`
   first, or it stores `args` as the wrapped layer and crashes at forward
   (`'TextModelArgs' object has no attribute '__call__'`). (`mtp_module.py`)

4. **Placement plumbing.** Programmatic `/place_instance` requires the master to
   pass `node_backends` (and `node_rdma_ctl` for RDMA). The dashboard
   `create_instance` path doesn't exercise this, so a regression here is easy to
   miss. (`master/main.py`)

Quick verification recipe for a new model: load it, `grep -iE 'MTP speculative
decoding enabled|Falling back|crashed with critical' ~/exo.log`, then run
`bench/mtp_longctx_probe.py` to confirm decode-tps + needle recall + zero
BOS-spam before quoting any throughput.
