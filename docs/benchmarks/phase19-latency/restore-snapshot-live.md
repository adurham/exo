# Phase-19 gamma-depth LIVE run — restore verification (2026-10-07)

The live gamma matrix required deploying an instrumented branch, so the cluster was
relaunched **twice** (budget: 2): once to deploy `deploy/next14-gamma`, once to restore
production. This file records the restore.

## Final deployed state (both nodes)

| item | m4-1 | m4-2 |
|---|---|---|
| `git rev-parse HEAD` | `f0840af1c50ecfdc85236a68b60a4c341805f8ce` | `f0840af1c50ecfdc85236a68b60a4c341805f8ce` |
| branch | `deploy/next13` | `deploy/next13` |
| mlx / mlx-lm submodule | `603f16eb7` / `6cc9c1e` | `603f16eb7` / `6cc9c1e` |
| runner PIDs (start) | 64939/64952 (15:26:10) | 74660/74672 (15:26:11) |

Restore relaunch: `EXO_TARGET_BRANCH=deploy/next13 ./start_cluster.sh` → "Nodes synchronized
on commit f0840af1c" → "READY (2/2)".

## Env parity vs the pre-campaign B0 snapshot

Both runners' `ps eww` env diffed against `raw/b0-live-env-m4-{1,2}-ps-eww.txt` (captured
before the deploy relaunch). **101 non-network env vars; 1 delta:**
`DSV41_INDEXER_ROW_BF16=1` explicit in B0, ABSENT after restore.

**This delta is a semantic no-op.** `mlx-lm/mlx_lm/models/deepseek_v41/indexer.py:111`

```python
_ROW_DTYPE = (mx.float32 if os.environ.get("DSV41_INDEXER_ROW_BF16", "1") == "0" else bf16)
```

The default when the var is unset is `"1"` ⇒ **bf16 — identical to explicitly setting it to
1.** The launcher only forwards it when set (`start_cluster.sh:2670`), and every production
relaunch this session (all with it unset) has run the same bf16 score row. So the running
dtype is unchanged; only the explicit-vs-default provenance differs. Not worth a third
relaunch (which the campaign budget forbids anyway).

## Smoke test

```
POST /v1/chat/completions  {"max_tokens":16,"temperature":0}
-> HTTP 200, usage 11 -> 16, finish=length (reasoning-only, greedy)
-> dsv41 engine live: "[DSV41] prefill controls: ... score_row_bytes=1 ..." @ 15:28:47
Launcher readiness jq: 2  (READY 2/2), runners 0744dd23 / ab870203 both RunnerReady
```

Cluster is back on production `deploy/next13 @ f0840af1c`, both runners Ready, serving.
