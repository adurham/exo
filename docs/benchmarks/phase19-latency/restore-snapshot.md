# Phase-19 restore snapshot (NO relaunch performed; env never changed)

Captured 2026-10-07 ~12:00 CDT, both nodes idle.

## Deployed commit (both nodes)
```
m4-1: f0840af1c50ecfdc85236a68b60a4c341805f8ce
m4-2: f0840af1c50ecfdc85236a68b60a4c341805f8ce
```

## start_cluster.sh invocation
```
cd ~/repos/exo && ./start_cluster.sh   (run from the LAPTOP; EXO_TARGET_BRANCH default)
```

## Runner env (node m4-1, exo PID 32954) — full ps-eww dump
Saved to raw/node1-env.txt. Salient knobs:
```
DSV41_INDEXER_ROW_BF16=1
EXO_DSV4_DSPARK_NATIVE=1
EXO_DSV4_DSPARK_TP_SHARD=1
EXO_DSV4_DSPARK=1
EXO_DSV4_MTP_ACCEPT_LOGPROBS=1
EXO_DSV4_MTP_C2_MAX_CTX=1
EXO_DSV4_MTP_DEDICATED=0
EXO_DSV4_MTP_EAGLE_K=8
EXO_DSV4_MTP_MAX_CTX=0
EXO_DSV4_MTP_TIE_REVERIFY=0
EXO_DSV4_MTP_TIEBREAK_EPS=0.5
EXO_DSV4_MTP_TIEBREAK_FIX=0
EXO_DSV4_MTP=1
EXO_KV_CACHE_BITS=0
EXO_PROFILER_LEVEL=1
EXO_SPECULATIVE_GAMMA=3
EXO_SPECULATIVE=1
MLX_JACCL_ACK_RETRANSMIT_US=500000
MLX_JACCL_ACK_SYNC_PRE=1
MLX_JACCL_RECONNECT_FRESH=1
MLX_JACCL_RELIABLE_DATA=1
MLX_JACCL_RELIABLE_IDLE_US=0
MLX_JACCL_RELIABLE_INFLIGHT=8
MLX_JACCL_RELIABLE_MAX_SZ=2
MLX_JACCL_RELIABLE_OPTIMISTIC=1
MLX_JACCL_SHARDING_MODE=Tensor
```

## Restore verification
```
NO relaunch was performed in this campaign; the cluster env is byte-identical to
the pre-campaign snapshot. Diff = empty by construction (nothing was changed).
Marker evidence: today's ~/exo.log contains 261 '[DSV41]' lines and 0 'MTP-PROF'
lines -> the live process runs the dsv41 engine, not the legacy MTP path.
```
