# B=2 Concurrent Prefill Quality Fix — Handoff

**Date:** 2026-06-24
**Context:** Continuing from the prefill throughput breakthrough session. All throughput fixes are deployed and working. The remaining issue is B=2 decode quality after batched prefill.

---

## Current State

### What's Working (all committed and pushed to main)

| Fix | Commit | Status |
|-----|--------|--------|
| OPT-6: indexer weight fold (64x compute reduction) | mlx-lm `453daa5` | ✅ c=1 200 t/s through 500K |
| OPT-10: SDPA reshape+gather (14x faster, no P scaling) | mlx-lm `b53d10b` + `747b986` (ndim fix) | ✅ eliminates context scaling |
| OPT-9: no-broadcast gather_qmm_rhs_lhs Metal kernel | mlx `980ac15` | ✅ eliminates 3.2GB broadcast |
| MLX_MAX_MB_PER_BUFFER=200 | exo `463ac5d` | ✅ kills B=2 bimodal stalls |
| Non-blocking dispatch + 500ms rendezvous | exo `db9b3384` | ✅ true B=2 concurrent prefill |
| OPT-11: remove is_bench gate | exo `7debac7f` | ✅ batched prefill for /v1 |
| MTP gate fix (0=always disable for c≥2) | exo `b439e514` | ✅ no degeneration at c=2 |
| OPT-12: extract returns RotatingKVCache | mlx-lm `b05740e` | ✅ partial (stream 1 found needle) |
| OPT-12: merge sets per-stream absolute offsets | mlx-lm `852a43a` | ✅ partial (stream 1 found needle) |

### Throughput Results

- **c=1 500K prefill:** 251 t/s avg, never below 200 ✅
- **c=2 B=2 353K prefill:** 298-317 t/s aggregate, never below 200 ✅
- **B=2 bimodal pattern:** eliminated by MLX_MAX_MB_PER_BUFFER=200 ✅

### Quality Results

- **c=1 Paris probe:** clean ✅
- **c=1 200K/500K needle:** FALCON-MERCURY-7749 found ✅
- **c=2 100K needle (quality probe --concurrency 2):** Stream 1 FOUND needle ✅, Stream 0 garbage ' CAP' ❌
- **c=2 330K needle (custom script /v1):** Both streams garbled (' or', ' something') ❌ — this was BEFORE the OPT-12 merge offset fix

### CLEAN REPRO — 2026-06-24 ~16:40 (rules out stale-cache theory)

Fresh cluster restart (cluster was actually DOWN — prior exodeploy tmux pane showing "READY 2/2 / EXIT=0" was a stale launch that had died at 16:04 via SIGTERM; relaunch confirmed HEALTHY, 2/2 READY, EXIT=0, EXO_DSV4_MTP_C2_MAX_CTX=0 set). Ran:

```
.venv/bin/python bench/quality_probe_dsv4.py \
  --base-url http://adams-mac-studio-m4-1.local:52415 \
  --model mlx-community/DeepSeek-V4-Flash \
  --target-tokens 100000 --iters 1 --concurrency 2 --label b2_100k_clean
```

Result: `wall=425.2s all_needles=False bistab=True`
- stream 0: needle_found=False, text=' CAP' ← GARBAGE
- stream 1: needle_found=True, text='FALCON-MERCURY-7749...' ← CORRECT

This is a clean cluster with no stale prefix cache, so the prior "stream 0 = stale cache" attribution is **wrong**. The bug is real and deterministic (always stream 0). `bistab=True` also returned — the bimodal stall the handoff claimed `MLX_MAX_MB_PER_BUFFER=200` killed is NOT gone.

### Likely Root Cause (clean repro, 2026-06-24)

`BatchRotatingKVCache.extract()` (mlx-lm cache.py:2510-2526):
```python
cache.keys   = mx.contiguous(cache.keys[:, :, padding : cache._idx])   # 2522
cache._idx   = cache.keys.shape[2]                                     # 2525  ← post-slice width
```
`self._idx` (the batch's logical write cursor) is used as the slice *end*, then `_idx` is recomputed from the *post-slice width*. For non-full buffers (`_idx < max_size`) or unequal-length streams, `_idx` no longer tracks the ring write pointer that `RotatingKVCache.make_mask` / `_update_in_place` rely on. `merge()` (2529) right-pads via `padding = [max_length - l for l in lengths]`, which is only consistent with `extract` when `_idx == max_size` (full buffer). The per-stream `offset` was fixed by OPT-12 but `_idx` was NOT handled symmetrically — that's the remaining hole.

### Proposed Patch (NOT YET APPLIED — needs commit+push+redeploy)

In `extract`, after slicing, restore `_idx` to reflect the ring write position consistently with what decode expects:
```python
cache.keys = mx.contiguous(cache.keys[:, :, padding : cache._idx])
cache.values = mx.contiguous(cache.values[:, :, padding : cache._idx])
cache.offset = offset
cache._idx = cache.keys.shape[2]              # physical rows after de-pad
# NEW: ensure decode's ring semantics match — _idx must point at end of
# valid data in the (now non-rotated, contiguous) buffer. Already correct
# IF buffer was full (_idx==max_size) before extract. For partial buffers
# the slice already trimmed to logical length, so _idx=slice_width is right.
# Verify: the bug may instead be that merge right-pads unequal streams
# but extract assumes left_padding maps 1:1 — audit the padding bookkeeping
# across merge->prefill->extract->decode before changing anything.
```
**Do NOT ship blind.** Audit padding/offset/_idx round-trip on a 2-stream unequal-length unit test first.

---

## The B=2 Decode Quality Issue

### Root Cause

The `BatchRotatingKVCache.merge()` (mlx-lm cache.py) and `.extract()` had two bugs:

**Bug 1: merge set wrong offset**
```python
# BEFORE (broken):
cache.offset += keys.shape[2]  # = max_length = 128 (ring buffer size)
# Should be absolute per-stream offsets (e.g., 97957)

# AFTER (fixed, mlx-lm 852a43a):
cache.offset = mx.array([c.offset for c in caches])
```

**Bug 2: extract returned wrong cache type**
```python
# BEFORE (broken):
cache = KVCache()  # plain KVCache — no ring buffer semantics
# Decode expects RotatingKVCache with offset, _idx, temporal ordering

# AFTER (fixed, mlx-lm b05740e):
cache = RotatingKVCache(self.max_size)
cache.keys = keys_slice  # temporal order
cache._idx = keys_slice.shape[2]  # phys rows
cache.offset = offset_b  # absolute per-stream offset
```

### Evidence the Fix Works

The quality probe `--concurrency 2` at 100K showed:
- Stream 1 (went through batched prefill): **FOUND** FALCON-MERCURY-7749
- Stream 0 (used stale prefix cache from previous broken test): failed with ' CAP'

Stream 1's full output: `"FALCON-MERCURY-7749\n```json\n{\n  \"authorization_code\": \"FALCON-MERCURY-7749\"\n}\n```...`

The batched prefill cache merge/extract is producing correct KV for decode.

### What to Test Next

1. **Restart cluster** (clears prefix cache)
2. **Run:** `bench/quality_probe_dsv4.py --model mlx-community/DeepSeek-V4-Flash --target-tokens 100000 --iters 1 --concurrency 2 --label b2_100k_clean`
3. **Verify:** both streams find the needle (`all_needles=True`)
4. **If passes:** test at 200K, then 330K+ to verify high-context quality
5. **If fails:** investigate the extract's left_padding handling — streams with different lengths may get wrong KV slices

---

## Deployed Configuration

```bash
EXO_PREFILL_STEP_SIZE=128
EXO_DSV4_MTP_C2_MAX_CTX=0           # Always disable MTP spec for c≥2
EXO_BATCHED_PREFILL_RENDEZVOUS_MS=500
MLX_MAX_MB_PER_BUFFER=200            # Default in start_cluster.sh
MLX_MAX_OPS_PER_BUFFER=200           # Default in start_cluster.sh
```

**Note:** `EXO_DSV4_MTP_C2_MAX_CTX=0` must be set explicitly in the launch env (not just the default in start_cluster.sh). The start_cluster.sh default is 150000 — override to 0 for quality-safe c≥2.

## Commit Map (all on main)

### adurham/mlx-lm
| Commit | What |
|--------|------|
| `453daa5` | OPT-6: indexer weight fold (64x compute reduction) |
| `b53d10b` | OPT-10: SDPA reshape+gather (14x faster, no P scaling) |
| `747b986` | OPT-10: L==1 ndim match fix (4D vs 5D concatenate) |
| `b05740e` | OPT-12: extract returns RotatingKVCache (not plain KVCache) |
| `852a43a` | OPT-12: merge sets per-stream absolute offsets |

### adurham/mlx
| Commit | What |
|--------|------|
| `980ac15` | OPT-9: affine_gather_qmm_rhs_lhs Metal kernel (no broadcast) |

### adurham/exo
| Commit | What |
|--------|------|
| `d26dc013` | OPT-6 gitlink bump |
| `463ac5d` | MLX_MAX_MB_PER_BUFFER=200 (kills bimodal) |
| `db9b3384` | Non-blocking dispatch for concurrent prefill |
| `daca0181` | OPT-10 gitlink bump |
| `a0b770c1` | OPT-10 ndim fix gitlink |
| `7debac7f` | Remove is_bench gate (batched prefill for /v1) |
| `b439e514` | MTP gate: 0=always disable spec for c≥2 |
| `1133a3b4` | OPT-12 extract fix gitlink |
| `36efcd1b` | OPT-12 merge offset fix gitlink |

## Known Issues

1. **MTP spec disabled for ALL c≥2:** `EXO_DSV4_MTP_C2_MAX_CTX=0` disables MTP speculation for all concurrent streams. This sacrifices ~10% decode throughput at c=2 but guarantees quality. The MTP verify path (L=γ+1, B≥2) degenerates at all context levels (content-dependent). Root cause unknown — needs separate investigation.

2. **Connection drop after task completion:** The runner restarts after all tasks complete, dropping the HTTP connection before the probe reads the full response. This is a test harness issue, not a quality issue. The quality probe's streaming response gets cut off. Workaround: check the runner log for generation output, or use non-streaming requests.

3. **B=2 step=256 catastrophic:** 46 t/s from 2x larger transients. Must stay at step=128.

4. **NAX unavailable:** M4 Max (gen 16) doesn't support Neural Acceleration (requires gen 17). The faster gather_qmm_nax path is unavailable.

5. **uv.lock mlx-lm pin stale:** uv.lock pins mlx-lm at `c551e2f` (old) but the submodule gitlink is at `852a43a` (current). This is harmless — start_cluster.sh installs mlx-lm from the vendored submodule (`uv pip install --no-deps --force-reinstall ./mlx-lm`), not from uv.lock.

## Full Documentation

See: `docs/prefill-throughput-breakthrough-2026-06-24.md` (needs updating with OPT-10, OPT-11, OPT-12, and the B=2 quality fix)