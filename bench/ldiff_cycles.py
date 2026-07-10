#!/usr/bin/env python3
"""Cycle-level ldiff — multi-cycle MTP verify chains vs sequential decode.

Single-pass ldiff proved ONE rowseq verify forward bitwise == sequential.
This drives CHAINS of verify cycles with the exact serving cache-mutation
semantics (dsv4_mtp.py B=1 linear path) and bitwise-compares every row's
logits against a pure sequential run over the same fixed token stream:

  accept3  — full acceptance: feed [t,t,t] chunks, no rollback (tests
             deferred-bump / pool handoff ACROSS cycles).
  reject0  — zero acceptance: verify [t, J, J] then serving rollback
             (regime a: trim(2); regime b on pool flush: restore_meta +
             trim(3) + commit-forward [t]); advance 1/cycle.
  reject1  — one accepted: verify [t0, t1, J], rollback 1 (regime a
             trim(1); regime b restore + trim(3) + commit [t0,t1]);
             advance 2/cycle.

Real quant recipe (experts mxfp4 g32, rest affine8 g64). Requires
EXO_DSV4_VERIFY_ROWSEQ=1 EXO_DSV4_VERIFY_ROWSEQ_MIN_CTX=0 for the serving
verify path on L=3 forwards.

Run (on a node):
  cd ~/repos/exo && EXO_DSV4_VERIFY_ROWSEQ=1 EXO_DSV4_VERIFY_ROWSEQ_MIN_CTX=0 \
    .venv/bin/python ~/scratch/ldiff_cycles.py <PROMPT_L> [N_LAYERS] [T_TOKENS]
"""
import json
import os
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import deepseek_v4 as dsv4
from mlx_lm.models.cache import BatchPoolingCache, CacheList, PoolingCache

# LDIFF_BATCH_CACHE: "0" plain classes; "all" (or "1") merge everything to
# Batch* classes like serving insert; "ring" only RotatingKVCache ->
# BatchRotatingKVCache; "pool" only PoolingCache -> BatchPoolingCache.
BATCH_CACHE = os.environ.get("LDIFF_BATCH_CACHE", "0")
if BATCH_CACHE == "1":
    BATCH_CACHE = "all"


CFG = Path.home() / ".exo/models/mlx-community--DeepSeek-V4-Flash/config.json"
PROMPT_L = int(sys.argv[1]) if len(sys.argv) > 1 else 48
N_LAYERS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
T_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 48
GAMMA = 2  # serving EXO_SPECULATIVE_GAMMA

cfg = json.load(open(CFG))
cfg["num_hidden_layers"] = N_LAYERS
cfg.pop("quantization", None)
cfg.pop("quantization_config", None)
args = dsv4.ModelArgs.from_dict(cfg)
print(
    f"PROMPT_L={PROMPT_L} N_LAYERS={N_LAYERS} T={T_TOKENS} gamma={GAMMA} "
    f"ROWSEQ={os.environ.get('EXO_DSV4_VERIFY_ROWSEQ')} BATCH_CACHE={BATCH_CACHE!r} "
    f"MIN_CTX={os.environ.get('EXO_DSV4_VERIFY_ROWSEQ_MIN_CTX')}"
)

mx.random.seed(11)
model = dsv4.Model(args)
model.set_dtype(mx.bfloat16)
mx.eval(model.parameters())
nn.quantize(
    model, group_size=32, bits=4, mode="mxfp4",
    class_predicate=lambda p, m: "switch_mlp" in p and hasattr(m, "to_quantized"),
)
nn.quantize(
    model, group_size=64, bits=8,
    class_predicate=lambda p, m: "switch_mlp" not in p and hasattr(m, "to_quantized"),
)
mx.eval(model.parameters())
print("quantized: experts mxfp4 g32, rest affine8 g64")

# LDIFF_FENCE=1: serving-parity per-layer fence — set a SINGLETON distributed
# group as every ffn's sharding_group so DeepseekV4MoE.__call__ takes the
# TP branch (sum_gradients + all_sum identity at 1 rank) and executes its
# per-layer mx.eval(y). This is the graph-segmentation difference between
# serving (TP) and the plain harness; suspected to make bf16 hc ops
# M-dependent (kernel/fusion selection differs by eval boundaries).
if os.environ.get("LDIFF_FENCE", "0") == "1":
    # Sentinel group + identity collectives: avoids touching any real
    # distributed backend while driving the exact TP code path (the
    # sharding_group is only None-checked; all_sum/sum_gradients are
    # value-identity at 1 rank anyway).
    dsv4.sum_gradients = lambda g: (lambda x: x)
    mx.distributed.all_sum = lambda y, group=None: y
    _sentinel = object()
    for _l in model.model.layers:
        _l.ffn.sharding_group = _sentinel
    print(f"[fence] sentinel sharding_group + identity collectives on {len(model.model.layers)} ffns")


VOCAB_CAP = min(args.vocab_size, 50000)
prompt = mx.random.randint(0, VOCAB_CAP, (1, PROMPT_L))
stream_toks = mx.random.randint(0, VOCAB_CAP, (T_TOKENS,))
mx.eval(prompt, stream_toks)
T = [int(x) for x in stream_toks.tolist()]
JUNK = [(t + 1237) % VOCAB_CAP for t in T]


def prefill():
    cache = model.make_cache()
    step = 4096
    for s in range(0, PROMPT_L, step):
        mx.eval(model(prompt[:, s : s + step], cache=cache))
    if BATCH_CACHE == "all":
        # Exactly what serving does at insert: merge to the Batch* classes
        # (BatchRotatingKVCache / BatchPoolingCache) with history.
        from mlx_lm.generate import _merge_caches
        cache = _merge_caches([list(cache)])
    elif BATCH_CACHE in ("ring", "pool", "both"):
        from mlx_lm.models.cache import RotatingKVCache

        def maybe(sub):
            if BATCH_CACHE in ("ring", "both") and isinstance(sub, RotatingKVCache):
                return type(sub).merge([sub])
            if BATCH_CACHE in ("pool", "both") and isinstance(sub, PoolingCache):
                return type(sub).merge([sub])
            return sub

        cache = [
            CacheList(*(maybe(sub) for sub in c.caches))
            if isinstance(c, CacheList)
            else maybe(c)
            for c in cache
        ]
    return cache


POOL_CLASSES = (PoolingCache, BatchPoolingCache)


def pools_of(cache):
    pools = []
    for c in cache:
        if isinstance(c, CacheList):
            for sub in c.caches:
                if isinstance(sub, POOL_CLASSES):
                    pools.append(sub)
        elif isinstance(c, POOL_CLASSES):
            pools.append(c)
    return pools


def may_flush(pc, vlen):
    rem = pc.remainder
    if isinstance(rem, list):
        rem = max(rem) if rem else 0
    return rem + vlen >= pc.ratio


def flushed_since(pc, snap):
    if hasattr(pc, "_pool_lengths"):  # BatchPoolingCache
        cur = [int(l) + int(b) for l, b in zip(pc._pool_lengths, pc._pending_bumps)]
        pre = [int(l) + int(b) for l, b in zip(snap[0], snap[5])]
        return cur != pre
    return (pc._pool_offset + pc._pending_offset_bump) != (snap[0] + snap[1])


def trim_all(cache, n):
    for c in cache:
        if hasattr(c, "trim"):
            c.trim(n)
        elif hasattr(c, "offset"):
            c.offset -= n


def ring_state(cache):
    out = []
    from mlx_lm.models.cache import BatchRotatingKVCache, RotatingKVCache
    for c in cache:
        subs = c.caches if hasattr(c, "caches") else [c]
        for sub in subs:
            if isinstance(sub, (RotatingKVCache, BatchRotatingKVCache)):
                d = {}
                for f in ("_offset", "_idx", "rotated", "keep"):
                    v = getattr(sub, f, None)
                    d[f] = v
                off = getattr(sub, "offset", None)
                try:
                    d["offset"] = (
                        int(mx.max(off)) if hasattr(off, "shape") else int(off)
                    )
                except Exception:
                    d["offset"] = None
                out.append(d)
    return out


def pool_state(pools):
    """Structural + content summary of every pool, for state diffing."""
    out = []
    for pc in pools:
        d = {}
        for f in ("remainder", "_processed"):
            v = getattr(pc, f, None)
            d[f] = list(v) if isinstance(v, list) else v
        # (length, pending) SPLIT is commit_pending-equivalent state: a
        # flush landing on the LAST committed row is pending in sequential
        # but already applied in a cache-level-rollback timeline (a rejected
        # row's commit_pending ran during verify). The TOTAL is what every
        # reader sees (commit_pending runs at the top of every Compressor
        # call before any read), so compare totals. pooled_width stays
        # strict — it caught the real padded-restore bug (13 vs 12).
        pl = getattr(pc, "_pool_lengths", None)
        pb = getattr(pc, "_pending_bumps", None)
        if pl is not None:
            d["pool_total"] = [int(l) + int(p) for l, p in zip(pl, pb)]
        else:
            d["pool_total"] = (
                getattr(pc, "_pool_offset", 0)
                + getattr(pc, "_pending_offset_bump", 0)
            )
        import hashlib
        import numpy as np

        def _h(arr):
            mx.eval(arr)
            return hashlib.md5(
                np.asarray(arr.astype(mx.float32)).tobytes()
            ).hexdigest()[:10]

        # Widths are first-class state: a padded-but-restored pool tensor
        # is wider than sequential's and changes SDPA K-length even when
        # the extra rows are masked. Plain pool: `pooled` slices to
        # _pool_offset, which is split-sensitive (applied vs pending bump)
        # — report the TOTAL-visible width instead. Batch pool: raw storage
        # width (strict — catches the padded-restore bug).
        if hasattr(pc, "_pool_storage"):
            d["pooled_width"] = (
                None
                if pc._pool_storage is None
                else min(
                    pc._pool_offset + pc._pending_offset_bump,
                    pc._pool_storage.shape[1],
                )
            )
        else:
            d["pooled_width"] = None if pc.pooled is None else pc.pooled.shape[1]
        # Hash only the LIVE regions (stale rows beyond remainder /
        # pool_lengths are semantically dead and legitimately differ).
        rem = d.get("remainder")
        rem0 = (rem[0] if isinstance(rem, list) else rem) or 0
        for name in ("buf_kv", "buf_gate"):
            a = getattr(pc, name, None)
            d[name] = None if a is None or rem0 == 0 else _h(a[:, :rem0])
        # Live pooled region hashed to the TOTAL (incl. a deferred-written
        # entry — both timelines wrote the same px value there).
        pt = d["pool_total"]
        pl0 = (pt[0] if isinstance(pt, list) else pt) or 0
        # Plain pool: the deferred slot sits beyond the `pooled` view's
        # _pool_offset slice — hash the raw storage. Batch pool: `pooled`
        # IS the raw storage attribute.
        a = getattr(pc, "_pool_storage", None)
        if a is None:
            a = getattr(pc, "pooled", None)
        if a is not None:
            pl0 = min(pl0, a.shape[1])
        d["pooled_live"] = None if a is None or pl0 == 0 else _h(a[:, :pl0])
        out.append(d)
    return out


def seq_reference():
    """Pure sequential: logits row per position (the MTP-off generator)."""
    cache = prefill()
    pools = pools_of(cache)
    rows = []
    states = {}
    for k, t in enumerate(T):
        lg = model(mx.array([t], dtype=mx.int32).reshape(1, 1), cache=cache)
        mx.eval(lg)
        rows.append(lg[:, 0, :])
        states[k + 1] = pool_state(pools) + ring_state(cache)
    return rows, states


def cycle_run(mode):
    """Verify-cycle chain with serving rollback semantics.

    Returns list of (seq_position, logits_row, regime_b_fired_this_cycle).
    """
    cache = prefill()
    pools = pools_of(cache)
    out = []
    states = {}
    i = 0
    while i < T_TOKENS:
        if mode == "accept3":
            toks = T[i : i + GAMMA + 1]
            if len(toks) < GAMMA + 1:
                break
            n_acc = GAMMA
        elif mode == "reject0":
            toks = [T[i], JUNK[i], (JUNK[i] + 331) % VOCAB_CAP]
            n_acc = 0
        elif mode == "reject1":
            if i + 1 >= T_TOKENS:
                break
            toks = [T[i], T[i + 1], JUNK[i]]
            n_acc = 1
        else:
            raise ValueError(mode)

        verify_len = GAMMA + 1
        spec_state = os.environ.get("LDIFF_SPEC_STATE", "0") == "1"
        cache_rb = os.environ.get("LDIFF_CACHE_ROLLBACK", "0") == "1"
        if spec_state:
            snaps = [pc.save_meta() for pc in pools]
            ring_caches = []
            ring_snaps = []
            for c in cache:
                subs = c.caches if hasattr(c, "caches") else [c]
                for sub in subs:
                    if hasattr(sub, "save_spec_state"):
                        ring_caches.append(sub)
                        ring_snaps.append(sub.save_spec_state())
            if cache_rb:
                for pc in pools:
                    pc.arm_spec_stash()
                for rc in ring_caches:
                    rc.arm_spec_stash()
        else:
            snaps = [
                pc.save_meta() if may_flush(pc, verify_len) else None
                for pc in pools
            ]
        vin = mx.array(toks, dtype=mx.int32).reshape(1, -1)
        vlogits = model(vin, cache=cache)
        mx.eval(vlogits)
        if spec_state and cache_rb:
            for pc in pools:
                pc.disarm_spec_stash()
            for rc in ring_caches:
                rc.disarm_spec_stash()

        rollback = GAMMA - n_acc
        regime_b = False
        if rollback > 0 and spec_state and cache_rb:
            # Cache-level exact undo (EXO_DSV4_SPEC_CACHE_ROLLBACK twin):
            # rings restore+re-push committed rows; pools trim or restore+
            # re-accumulate per flush attribution. NO commit-forward.
            regime_b = True
            keep = verify_len - rollback
            ok_rings = all(
                rc.spec_pushed_rows() == verify_len for rc in ring_caches
            )
            ok_pools = all(
                pc.spec_can_rollback(psnap, keep, verify_len)
                for pc, psnap in zip(pools, snaps)
            )
            if not (ok_rings and ok_pools):
                raise RuntimeError(
                    f"cache-level rollback refused: rings={ok_rings} "
                    f"pools={ok_pools} (i={i} mode={mode})"
                )
            for rc, rsnap in zip(ring_caches, ring_snaps):
                rc.rollback_spec_write(rsnap, keep)
            for pc, psnap in zip(pools, snaps):
                pc.spec_rollback(psnap, keep)
        elif rollback > 0 and spec_state:
            # Unified faithful rollback: wholesale restore + commit-forward.
            # LDIFF_SPEC_PART bisect: all|rings|pools — which component uses
            # the wholesale restore (the other falls back to trim(G+1)).
            part = os.environ.get("LDIFF_SPEC_PART", "all")
            regime_b = True
            if part in ("all", "rings"):
                for rc, rsnap in zip(ring_caches, ring_snaps):
                    rc.restore_spec_state(rsnap)
            else:
                for c in cache:
                    subs = c.caches if hasattr(c, "caches") else [c]
                    for sub in subs:
                        if hasattr(sub, "save_spec_state") and hasattr(sub, "trim"):
                            sub.trim(GAMMA + 1)
            if part in ("all", "pools"):
                for pc, psnap in zip(pools, snaps):
                    if psnap is not None:
                        pc.restore_meta(psnap)
            else:
                for pc in pools:
                    pc.trim(GAMMA + 1)
            commit = mx.array(toks[: n_acc + 1], dtype=mx.int32).reshape(1, -1)
            _cl = model(commit, cache=cache)
            mx.eval(_cl)
            del _cl
        elif rollback > 0:
            pool_flushed = any(
                snap is not None and flushed_since(pc, snap)
                for pc, snap in zip(pools, snaps)
            )
            if not pool_flushed:
                trim_all(cache, rollback)
            else:
                regime_b = True
                if os.environ.get("LDIFF_RESTORE_AFTER_TRIM", "0") == "1":
                    # FIX CANDIDATE: trim first (rewinds ring KV + the pools
                    # that were NOT snapshotted), THEN restore snapshotted
                    # pools — so the blanket CacheList.trim can no longer
                    # double-roll-back a pool that restore_meta already
                    # rewound to its pre-verify state.
                    trim_all(cache, GAMMA + 1)
                    for pc, snap in zip(pools, snaps):
                        if snap is not None:
                            pc.restore_meta(snap)
                else:
                    # serving order (dsv4_mtp.py): restore THEN blanket trim
                    for pc, snap in zip(pools, snaps):
                        if snap is not None:
                            pc.restore_meta(snap)
                    trim_all(cache, GAMMA + 1)
                commit = mx.array(toks[: n_acc + 1], dtype=mx.int32).reshape(1, -1)
                _cl = model(commit, cache=cache)
                mx.eval(_cl)
                del _cl

        for j in range(n_acc + 1):
            if i + j < T_TOKENS:
                out.append((i + j, vlogits[:, j, :], regime_b))
        i += n_acc + 1
        states[i] = pool_state(pools) + ring_state(cache)
    return out, states


# ── per-layer capture (LDIFF_LAYER_TRACE=1): localize the first drifting
# layer/row inside the first drifting cycle ─────────────────────────────
LAYER_TRACE = os.environ.get("LDIFF_LAYER_TRACE", "0") == "1"
_captures = []
_BLOCK_CLS = type(model.model.pipeline_layers[0])
_ORIG_BLOCK_CALL = _BLOCK_CLS.__call__


def _install_capture():
    def wrapped(self, h, mask, cache, input_ids):
        out = _ORIG_BLOCK_CALL(self, h, mask, cache, input_ids)
        _captures.append(out)
        return out
    _BLOCK_CLS.__call__ = wrapped


def _uninstall_capture():
    _BLOCK_CLS.__call__ = _ORIG_BLOCK_CALL


def seq_layers_for(positions):
    """Per-layer block outputs for given seq positions (fresh prefill)."""
    cache = prefill()
    # advance to min(positions) first
    per_pos = {}
    for k, t in enumerate(T[: max(positions) + 1]):
        want = k in positions
        if want:
            _captures.clear()
            _install_capture()
        lg = model(mx.array([t], dtype=mx.int32).reshape(1, 1), cache=cache)
        mx.eval(lg)
        if want:
            _uninstall_capture()
            per_pos[k] = [c for c in _captures[:N_LAYERS]]
            for c in per_pos[k]:
                mx.eval(c)
    return per_pos


if os.environ.get("LDIFF_SEQ_MASK_NONE", "0") == "1":
    # Test: make the sequential reference pass mask=None at L=1 (like the
    # rowseq rows do) instead of the batch-ring array mask. If the drift
    # vanishes, the None-vs-array mask asymmetry is the whole story.
    _orig_cam = dsv4.create_attention_mask

    def _cam(h, cache, window_size=None, return_array=False):
        if h.shape[1] == 1:
            return None
        return _orig_cam(
            h, cache, window_size=window_size, return_array=return_array
        )

    dsv4.create_attention_mask = _cam
    print("[seq-mask-none] sequential L=1 masks forced to None")

print("\nsequential reference...")
ref, ref_states = seq_reference()

for mode in ("accept3", "reject0", "reject1"):
    print(f"\n[{mode}]")
    rows, cyc_states = cycle_run(mode)
    # Pool-state diff at matching committed-token counts: name the first
    # unfaithful field (the thing the next fix must repair).
    state_diff = None
    for cnt in sorted(cyc_states):
        if cnt not in ref_states:
            continue
        for pi, (a, b) in enumerate(zip(cyc_states[cnt], ref_states[cnt])):
            for f in a:
                if a[f] != b[f]:
                    state_diff = (cnt, pi, f, a[f], b[f])
                    break
            if state_diff:
                break
        if state_diff:
            break
    if state_diff:
        cnt, pi, f, av, bv = state_diff
        print(f"  first pool-state diff: committed={cnt} pool#{pi} field={f} cyc={av} seq={bv}")
    else:
        print("  pool states: IDENTICAL at every matching committed count")
    first_drift = None
    n_regime_b = sum(1 for (_, _, rb) in rows if rb)
    worst = 0.0
    for pos, row, rb in rows:
        eq = bool(mx.all(row == ref[pos]).item())
        if not eq:
            d = float(
                mx.abs(row.astype(mx.float32) - ref[pos].astype(mx.float32)).max()
            )
            worst = max(worst, d)
            if first_drift is None:
                first_drift = (pos, d, rb)
    print(f"  rows compared: {len(rows)}  regime-b cycles: {n_regime_b}")
    if first_drift is None:
        print("  VERDICT: BITWISE CLEAN")
    else:
        pos, d, rb = first_drift
        print(
            f"  VERDICT: DRIFT — first at seq position {pos} "
            f"(abs {PROMPT_L + pos}), max|dlogit|={d:.6f}, "
            f"regime_b_that_cycle={rb}, worst overall={worst:.6f}"
        )
if os.environ.get("LDIFF_TRACE_COMMIT", "0") == "1":
    # Unified-mode microscope: at cycle 0 of reject0, trace the COMMIT
    # forward's per-layer outputs against sequential position 0.
    seq_caps = seq_layers_for({0})
    cache = prefill()
    pools = pools_of(cache)
    snaps = [pc.save_meta() for pc in pools]
    ring_caches, ring_snaps = [], []
    for c in cache:
        subs = c.caches if hasattr(c, "caches") else [c]
        for sub in subs:
            if hasattr(sub, "save_spec_state"):
                ring_caches.append(sub)
                ring_snaps.append(sub.save_spec_state())
    toks = [T[0], JUNK[0], (JUNK[0] + 331) % VOCAB_CAP]
    vin = mx.array(toks, dtype=mx.int32).reshape(1, -1)
    mx.eval(model(vin, cache=cache))
    for rc, rsnap in zip(ring_caches, ring_snaps):
        rc.restore_spec_state(rsnap)
    for pc, psnap in zip(pools, snaps):
        pc.restore_meta(psnap)
    _captures.clear()
    _install_capture()
    commit = mx.array(toks[:1], dtype=mx.int32).reshape(1, -1)
    mx.eval(model(commit, cache=cache))
    _uninstall_capture()
    commit_caps = [c for c in _captures[:N_LAYERS]]
    print("[trace-commit] commit forward vs sequential t0, per layer:")
    for li in range(N_LAYERS):
        a = commit_caps[li]
        b = seq_caps[0][li]
        eq = bool(mx.all(a == b).item())
        d = float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max())
        print(f"  layer {li} (ratio {args.compress_ratios[li]}): {'EQ' if eq else f'DIFF {d:.3e}'}")
    import sys as _sys
    _sys.exit(0)

if LAYER_TRACE:
    # accept3, cycle containing the first drifting position: find it first
    rows, _ = cycle_run("accept3")
    first = next((pos for pos, row, _ in rows if not bool(mx.all(row == ref[pos]).item())), None)
    if first is None:
        print("\n[layer-trace] accept3 clean; nothing to trace")
    else:
        cyc_idx = first // (GAMMA + 1)
        base = cyc_idx * (GAMMA + 1)
        print(f"\n[layer-trace] first drifting position {first} -> cycle {cyc_idx} (rows {base}..{base+GAMMA})")
        seq_caps = seq_layers_for(set(range(base, base + GAMMA + 1)))
        # rerun cycles, capturing the target cycle's layers
        cache = prefill()
        i = 0
        while i < T_TOKENS:
            toks = T[i : i + GAMMA + 1]
            if len(toks) < GAMMA + 1:
                break
            capture = i == base
            if capture:
                _captures.clear()
                _install_capture()
            vin = mx.array(toks, dtype=mx.int32).reshape(1, -1)
            vlogits = model(vin, cache=cache)
            mx.eval(vlogits)
            if capture:
                _uninstall_capture()
                cyc_caps = [c for c in _captures[:N_LAYERS]]
                for c in cyc_caps:
                    mx.eval(c)
                break
            i += GAMMA + 1
        for li in range(N_LAYERS):
            cl = cyc_caps[li]
            msg = []
            for j in range(GAMMA + 1):
                pos = base + j
                if pos not in seq_caps:
                    continue
                sl = seq_caps[pos][li]
                a = cl[:, j : j + 1]
                b = sl[:, 0:1] if sl.shape[1] == 1 else sl[:, -1:]
                eq = bool(mx.all(a == b).item())
                d = float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max())
                msg.append(f"row{j}={'EQ' if eq else f'DIFF {d:.2e}'}")
            print(f"  layer {li} (ratio {args.compress_ratios[li]}): " + "  ".join(msg))

print("\ndone.")
