"""Shared helpers for the p20 MoE partition work (bit-exact harness + probes).

Synthetic-weight reproduction of the dsv41 per-rank segmented-GEMM launches
(E=384, k=3, gu: K=5120 N=1152 tpe=72; dn: K=1152 N=5120 tpe=320) with the
real routing mix (phase-10 trace, layer 0, 2048 tok x top-6, seed 20261010 --
identical to bench/p20_moe_autopsy.py real_chunk_idx) plus the autopsy's
uniform/skew E=192 controls and adversarial ragged mixes.

Kernel numerics depend only on (trellis bits, x, blk_tab), so random trellis
words exercise every decode path; the real checkpoint is used on studio2.
"""
import json
import os

import numpy as np

KK = 6


def pack(ids):
    tokens = (ids.size + KK - 1) // KK
    pad = tokens * KK - ids.size
    if pad:
        ids = np.concatenate([ids, np.zeros(pad, np.int32)])
    return ids.reshape(1, tokens, KK)


def real_chunk_ids(trace_path, layer=0, n_tokens=2048, seed=20261010):
    d = json.load(open(trace_path))
    rows = [np.asarray(r[1], np.int32) for r in d["records"] if int(r[0]) == layer]
    rng = np.random.RandomState(seed)
    sel = rng.randint(0, len(rows), size=n_tokens)
    return np.stack([rows[i] for i in sel]).reshape(-1).astype(np.int32)


def uniform_ids(E, N, seed=2026):
    rng = np.random.RandomState(seed)
    ids = np.repeat(np.arange(E), int(np.ceil(N / E)))[:N].astype(np.int32)
    rng.shuffle(ids)
    return ids


def skew_ids(E, N, seed=2026, alpha=0.9, hot_frac=0.35):
    rng = np.random.RandomState(seed)
    nhot = max(1, int(round(E * hot_frac)))
    w = np.zeros(E)
    w[:nhot] = 1.0 / (np.arange(1, nhot + 1) ** alpha)
    w[nhot:] = w[nhot - 1] * 0.25 if nhot < E else 0.0
    w = w / w.sum()
    return rng.choice(E, size=N, replace=True, p=w).astype(np.int32)


def counts_ids(counts, seed=0):
    """Flat expert ids with exact per-expert counts (shuffled)."""
    rng = np.random.RandomState(seed)
    ids = np.concatenate(
        [np.full(int(c), e, np.int32) for e, c in enumerate(counts) if c > 0]
    )
    rng.shuffle(ids)
    return ids


def adversarial_mixes(E=384):
    """name -> per-expert counts. Covers tail-only experts, len=1, all-full,
    lengths straddling 8/64-row boundaries, max-length + tiny, long runs."""
    rng = np.random.RandomState(7)
    mixes = {}
    c = np.zeros(E, np.int64); c[: E // 2] = rng.randint(1, 8, size=E // 2)
    mixes["tail_only_1to7"] = c
    c = np.zeros(E, np.int64); c[:] = 1
    mixes["all_len1"] = c
    c = np.zeros(E, np.int64); c[:96] = 64
    mixes["all_full_64"] = c
    c = np.zeros(E, np.int64)
    edges = [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 39, 40, 41, 55, 56, 57, 63, 64,
             65, 71, 72, 73, 127, 128, 129, 136, 191, 192, 193]
    for i, v in enumerate(edges * 4):
        c[(i * 37) % E] += v
    mixes["boundary_lens"] = c
    c = np.zeros(E, np.int64); c[::2] = 64; c[1::2] = rng.randint(1, 4, size=E // 2)
    mixes["max64_plus_tiny"] = c
    c = np.zeros(E, np.int64); c[[0, 5, 383]] = [1000, 517, 3]; c[100:140] = rng.randint(1, 70, 40)
    mixes["long_runs_sparse"] = c
    c = rng.randint(0, 80, size=E).astype(np.int64)
    c[rng.rand(E) < 0.3] = 0
    mixes["random_ragged_a"] = c
    c = rng.geometric(0.03, size=E).astype(np.int64)
    c[rng.rand(E) < 0.1] = 0
    mixes["random_geometric_b"] = c
    c = np.zeros(E, np.int64); c[E - 1] = 5
    mixes["single_last_expert"] = c
    c = np.zeros(E, np.int64); c[0] = 1
    mixes["single_row"] = c
    return mixes


def synth_layer(mx, E, k=3, D=5120, H=1152, seed=0):
    """Random trellis (u16) for stacked gu (2E*tpe_gu tiles) and dn."""
    P = k * 256 // 16
    gt = H // 16
    dt = D // 16
    key = mx.random.key(seed)
    k1, k2 = mx.random.split(key)
    gu = mx.random.randint(0, 1 << 16, (D // 16, 2 * E * gt, P), key=k1).astype(mx.uint16)
    dn = mx.random.randint(0, 1 << 16, (H // 16, E * dt, P), key=k2).astype(mx.uint16)
    mx.eval(gu, dn)
    return {"gu": gu, "dn": dn, "gt": gt, "dt": dt, "D": D, "H": H, "E": E, "k": k}


def launch_inputs(mx, MOE, ids, E, D, seed=1):
    """Sorted ids + blk_tab exactly as EXL3SwitchGLU._prefill builds them."""
    N = int(ids.size)
    flat = mx.array(ids.astype(np.uint32))
    order = mx.argsort(flat)
    sidx = flat[order]
    nb_max = N // 64 + E + 1
    tab, nbr = MOE._seg_table_fn(E, nb_max, 64)(sidx)
    x = (mx.random.normal((N + 64, D), key=mx.random.key(seed)) * 0.5).astype(mx.float16)
    mx.eval(sidx, tab, nbr, x)
    return {"N": N, "sidx": sidx, "tab": tab, "nbr": nbr, "x": x, "nb_max": nb_max}


def env_trace_path(root):
    return os.environ.get(
        "MB_TRACE",
        os.path.join(root, "docs/benchmarks/phase10-planb-gate-2026-09-28/raw/p30_exl3_trace_clamp.json"),
    )
