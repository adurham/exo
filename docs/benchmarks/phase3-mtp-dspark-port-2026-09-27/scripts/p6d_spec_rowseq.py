#!/usr/bin/env python3
"""DSpark speculative loop, with the verify forward selectable:

  --verify rowseq : process the gamma+1 verify rows as SEPARATE 1-token
                    forwards. Each row is then computationally identical to a
                    plain decode step, so token-exactness against the greedy
                    baseline is a real, falsifiable expectation.
  --verify chunk  : process all gamma+1 rows in ONE forward (the naive/cheap
                    shape, and the one whose per-token cost we measured).

WHY BOTH. p6b/p6c showed the port's forward result depends on how the input is
split into chunks -- layer 0 (a pure-window layer: no compressor, no indexer)
already differs by ~2 bf16 ulps at the final position, and that amplifies into a
different argmax within two steps. So a naive multi-row verify CANNOT be
token-identical to greedy decode; that is a numerical property of the body, not
a bookkeeping bug. Production encodes the same knowledge: its batched verify
path is gated behind EXO_DSV4_VERIFY_BATCH_MIN_CTX and left OFF at short context
precisely to "preserve byte-identity at short ctx where the base decode is
deterministic."

That makes rowseq the correct instrument for validating the accept/rollback/
ctx-feed logic: with identical chunk shapes, ANY token difference is a
bookkeeping bug, so a clean match is meaningful. chunk mode is then run to
document (not to validate) the numerics gap.

BLOCK CONVENTION: ``draft()`` computes block_ids = [anchor, noise x (bs-1)] and
returns bs tokens [d1..d_bs]; position 0 of the block IS the anchor, so
d_tokens[k] is the token at anchor+k+1. Verify input = [anchor, d1..d_bs]
(bs+1 tokens); acceptance compares the target argmax at row k with d_tokens[k].

ROLLBACK: ring and pooled comp_kv self-heal (a position lives at slot p%window
and is rewritten by whichever chunk commits it; a group straddling the rollback
target is recomputed by the next chunk before it is read). The compressor's
open-group carry must be REBUILT from (pre-round carry rows) + (the verify
chunk's own stashed raw rows), because rows for newly committed positions may
exist only in the verify chunk.

Usage:
  python3 p6d_spec_rowseq.py <MODEL_DIR> [--n 48] [--verify rowseq|chunk]
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import time

import mlx.core as mx

REPO = os.path.expanduser("~/repos/ref/deepseek-v41-mlx")
sys.path.insert(0, REPO)
from deepseek_v41_mlx.load import load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model_dir")
ap.add_argument("--n", type=int, default=48)
ap.add_argument("--verify", choices=("rowseq", "chunk"), default="rowseq")
ap.add_argument("--width", type=int, default=5, help="draft width (<= block_size)")
a = ap.parse_args()

try:
    mx.set_wired_limit(int(float(os.environ.get("V41_WIRED_GB", "100")) * 1e9))
except Exception as e:  # noqa: BLE001
    print("[warn] wired:", e)

PATH = os.path.expanduser(a.model_dir)
model, args = load(PATH)
TAPS = list(args.dspark_target_layer_ids)
BS = args.dspark_block_size
GAMMA = min(a.width, BS)
bsz = 1
print(f"=== {args.n_layers} layers, {args.n_mtp_layers} draft stages, block={BS}, "
      f"width={GAMMA}, verify={a.verify}, taps={TAPS} ===")

try:
    from deepseek_v41_mlx.generate import load_tokenizer
    tok = load_tokenizer(PATH)
    prompt_ids = tok("The history of the Roman Empire")["input_ids"]
except Exception as e:  # noqa: BLE001
    print(f"[warn] tokenizer unavailable ({type(e).__name__}) -- using ids")
    prompt_ids = [43281, 108435, 109597, 105148, 56193, 56712, 100489, 87240]
print(f"  prompt: {len(prompt_ids)} tokens")
print()

MAXSEQ = len(prompt_ids) + a.n + 64


def snap_spec(cache, start_pos):
    st = []
    for lc in cache.layers:
        cs = lc.comp_state
        if cs is None:
            st.append(None)
            continue
        m = start_pos % lc.ratio
        st.append((cs.kv_state[:bsz, :m], cs.score_state[:bsz, :m], m) if m
                  else (None, None, 0))
    return start_pos, st


def rebuild_carry(cs, ratio, O, target, saved, stash_kv, stash_sc):
    """Rebuild the open-group carry to hold positions [ratio*(t//r), target)."""
    s = ratio * (target // ratio)
    need = target - s
    if need == 0:
        return
    saved_kv, saved_sc, m = saved
    pk, ps = [], []
    if s < O:
        lo = max(s, O - m)
        pk.append(saved_kv[:, lo - (O - m): m])
        ps.append(saved_sc[:, lo - (O - m): m])
    lo2 = max(s, O)
    pk.append(stash_kv[:, lo2 - O: target - O])
    ps.append(stash_sc[:, lo2 - O: target - O])
    nk = mx.concatenate(pk, axis=1)
    ns = mx.concatenate(ps, axis=1)
    mx.eval(nk, ns)
    if nk.shape[1] != need:
        raise RuntimeError(f"carry rebuild {nk.shape[1]} != {need}")
    cs.kv_state[:, :need] = nk
    cs.score_state[:, :need] = ns


def rollback_spec(cache, snap, target, stashes):
    off0, st = snap
    cache.offset = target
    for li, (lc, saved) in enumerate(zip(cache.layers, st)):
        cs = lc.comp_state
        if cs is None:
            continue
        kv, sc = stashes[li]
        if kv is None:
            raise RuntimeError("no stashed chunk rows for a compressor layer")
        rebuild_carry(cs, lc.ratio, off0, target, saved, kv, sc)


def plain_greedy(n_new):
    cache = model.make_cache(bsz=1, max_seq_len=MAXSEQ)
    lg = model(mx.array([prompt_ids]), cache, last_logit_only=True)
    t = mx.argmax(lg[:, -1], axis=-1)
    mx.eval(t)
    out = [int(t[0])]
    t0 = time.perf_counter()
    for _ in range(n_new - 1):
        lg = model(t[:, None], cache, last_logit_only=True)
        t = mx.argmax(lg[:, -1], axis=-1)
        mx.eval(t)
        out.append(int(t[0]))
    dt = time.perf_counter() - t0
    return out, dt


def verify_chunk(ids):
    """One multi-row forward. Returns (logits, taps, stashes)."""
    logits, taps = model(ids, cache_cur, return_taps=True)
    mx.eval(logits, *taps.values())
    stashes = []
    for lc in cache_cur.layers:
        cs = lc.comp_state
        stashes.append((cs.chunk_kv, cs.chunk_score) if cs is not None else (None, None))
    return logits, taps, stashes


def verify_rowseq(ids):
    """gamma+1 separate 1-token forwards; concatenates logits, taps, stashes."""
    lg_parts, tap_parts = [], {L: [] for L in TAPS}
    stash_kv = {i: [] for i in range(len(cache_cur.layers))}
    stash_sc = {i: [] for i in range(len(cache_cur.layers))}
    for j in range(ids.shape[1]):
        lg, tp = model(ids[:, j:j + 1], cache_cur, return_taps=True)
        mx.eval(lg, *tp.values())
        lg_parts.append(lg)
        for L in TAPS:
            tap_parts[L].append(tp[L])
        for i, lc in enumerate(cache_cur.layers):
            cs = lc.comp_state
            if cs is not None:
                if cs.chunk_kv is None:
                    raise RuntimeError("compressor produced no chunk rows")
                stash_kv[i].append(cs.chunk_kv)
                stash_sc[i].append(cs.chunk_score)
    logits = mx.concatenate(lg_parts, axis=1)
    taps = {L: mx.concatenate(tap_parts[L], axis=1) for L in TAPS}
    stashes = []
    for i, lc in enumerate(cache_cur.layers):
        if lc.comp_state is None:
            stashes.append((None, None))
        else:
            stashes.append((mx.concatenate(stash_kv[i], axis=1),
                            mx.concatenate(stash_sc[i], axis=1)))
    mx.eval(logits, *taps.values())
    return logits, taps, stashes


cache_cur = None

def spec_greedy(n_new):
    global cache_cur
    cache = model.make_cache(bsz=1, max_seq_len=MAXSEQ)
    cache_cur = cache
    lg = model(mx.array([prompt_ids]), cache, last_logit_only=True)
    t = mx.argmax(lg[:, -1], axis=-1)
    mx.eval(t)
    out = [int(t[0])]
    pos = cache.offset

    head = model.mtp
    dsc = head.make_cache(bsz=1)
    hist = []

    t0 = time.perf_counter()
    while len(out) < n_new:
        d_tokens, _conf = head.draft(t, model.embed, model.head, dsc, width=GAMMA)
        mx.eval(d_tokens)

        verify_ids = mx.concatenate(
            [t.reshape(1, 1).astype(d_tokens.dtype), d_tokens], axis=1)
        snap = snap_spec(cache, pos)
        if a.verify == "chunk":
            logits, taps, stashes = verify_chunk(verify_ids)
        else:
            logits, taps, stashes = verify_rowseq(verify_ids)

        targ = mx.argmax(logits, axis=-1)
        mx.eval(targ)
        n_acc = 0
        for k in range(GAMMA):
            if int(targ[0, k]) == int(d_tokens[0, k]):
                n_acc += 1
            else:
                break
        bonus = int(targ[0, n_acc])
        hist.append(n_acc)

        for k in range(n_acc):
            out.append(int(d_tokens[0, k]))
        out.append(bonus)

        target = pos + n_acc + 1
        rollback_spec(cache, snap, target, stashes)
        pos = target

        ctx_cat = mx.concatenate([taps[L] for L in TAPS], axis=-1)
        head.append_ctx(ctx_cat[:, :n_acc + 1], dsc)

        t = mx.array([bonus])
    dt = time.perf_counter() - t0
    return out[:n_new], dt, hist


print("=== baseline: plain greedy ===")
base, dt_base = plain_greedy(a.n)
print(f"  {len(base)} tokens in {dt_base:.3f}s -> {len(base)/dt_base:.2f} tok/s")
print(f"  first 12: {base[:12]}")
print()

print(f"=== speculative (DSpark, verify={a.verify}) ===")
spec, dt_spec, hist = spec_greedy(a.n)
print(f"  {len(spec)} tokens in {dt_spec:.3f}s -> {len(spec)/dt_spec:.2f} tok/s")
print(f"  first 12: {spec[:12]}")
print()

print("=== VERDICT ===")
if spec == base:
    print(f"  tokens IDENTICAL to plain greedy ({len(base)} tokens)  OK")
else:
    print("  tokens DIVERGE")
    for i, (x, y) in enumerate(zip(spec, base)):
        if x != y:
            print(f"    first divergence at index {i}: spec={x} base={y}")
            break
print()
c = collections.Counter(hist)
print(f"=== acceptance (4-layer body: NOT representative) ===")
print(f"  rounds: {len(hist)}  mean accepted: {sum(hist)/len(hist):.2f} / {GAMMA}")
for k in sorted(c):
    print(f"    n_acc={k}: {c[k]:3d}  ({c[k]/len(hist):5.1%})")
print()
print(f"  speedup here: {dt_base/dt_spec:.2f}x")
print()
print("SPEC-DONE")
