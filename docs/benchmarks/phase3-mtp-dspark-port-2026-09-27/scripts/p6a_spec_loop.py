#!/usr/bin/env python3
"""DSpark speculative decode round loop for the V4.1 port.

Mirrors production's cycle (dsv4_mtp.py / pp_speculation.py):

    anchor t ──► draft(block) ──► verify forward [anchor, d1..d_bs]
                                        │
                    accept = #leading rows where target argmax agrees with draft
                                        │
              commit n_acc drafts + 1 bonus (target's own argmax at row n_acc)
                                        │
                  rollback cache to pos+n_acc+1, feed taps[:, :n_acc+1] as ctx

BLOCK CONVENTION (this is where the first attempt went wrong): ``draft()``
computes ``block_ids = [anchor, noise x (bs-1)]`` and returns ``bs`` tokens
``[d1..d_bs]`` -- position 0 of the block IS the anchor, so ``d_tokens[k]`` is
the token at ``anchor+k+1``. The verify input is therefore
``[anchor, d1, ..., d_bs]`` (bs+1 tokens), and acceptance compares the target's
argmax at row k against ``d_tokens[k]``.

The DECISIVE correctness property, and the point of this script: speculative
decode must emit the SAME tokens as plain greedy decode. Acceptance rate only
changes speed, never output. A 4-layer body drafts poorly, so we cannot measure
acceptance here -- but a mismatch would mean the accept/rollback bookkeeping is
wrong, which is exactly what a reduced model CAN prove.

The port's cache keeps positions p at slot p%window, and attention reads the
window BEFORE writing its own rows, so stale rows left by rejected drafts are
overwritten by the real write for that position before any read can reach them.
The one piece of genuinely cross-call state is the compressor's OPEN-GROUP
carry (kv_state/score_state): it must be rebuilt, not merely restored, because
the rows for newly committed positions may exist only in the verify chunk.

Usage:
  python3 p6a_spec_loop.py <MODEL_DIR> [--n 48] [--no-rollback] [--no-mtp]
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
ap.add_argument("--n", type=int, default=48, help="tokens to generate")
ap.add_argument("--prompt", type=str, default="The history of the Roman Empire")
ap.add_argument("--no-rollback", action="store_true",
                help="negative control: skip the cache rollback")
ap.add_argument("--no-mtp", action="store_true",
                help="baseline only (plain greedy, one token at a time)")
ap.add_argument("--width", type=int, default=None,
                help="draft width (default: block_size)")
a = ap.parse_args()

try:
    mx.set_wired_limit(int(float(os.environ.get("V41_WIRED_GB", "100")) * 1e9))
except Exception as e:  # noqa: BLE001
    print("[warn] wired:", e)

PATH = os.path.expanduser(a.model_dir)
model, args = load(PATH)
TAPS = list(args.dspark_target_layer_ids)
BS = args.dspark_block_size
GAMMA = a.width or BS
print(f"=== {args.n_layers} body layers, {args.n_mtp_layers} draft stages, "
      f"block={BS}, width={GAMMA}, taps={TAPS} ===")

try:
    from deepseek_v41_mlx.generate import load_tokenizer
    tok = load_tokenizer(PATH)
    prompt_ids = tok(a.prompt)["input_ids"]
except Exception as e:  # noqa: BLE001
    print(f"[warn] tokenizer unavailable ({type(e).__name__}: {e}) -- using ids")
    tok = None
    prompt_ids = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
print(f"  prompt: {len(prompt_ids)} tokens")
print()

MAXSEQ = len(prompt_ids) + a.n + 64
bsz = 1


def snap_spec(cache, start_pos):
    """Pre-verify state needed to roll back: offset + open-group carry rows."""
    st = []
    for lc in cache.layers:
        cs = lc.comp_state
        if cs is None:
            st.append(None)
            continue
        m = start_pos % lc.ratio                 # carried rows of the open group
        if m:
            st.append((cs.kv_state[:bsz, :m], cs.score_state[:bsz, :m], m))
        else:
            st.append((None, None, 0))
    return start_pos, st


def rebuild_carry(cs, ratio, O, target, saved):
    """Rebuild the open-group carry to hold positions [ratio*(t//r), target).

    Restoring the pre-verify snapshot is NOT enough: when the verify chunk is
    what first wrote the rows for the newly committed positions, those rows
    exist ONLY in that chunk. So splice saved carry rows (positions < O) with
    the chunk's own stashed rows (positions >= O).
    """
    s = ratio * (target // ratio)
    need = target - s
    if need == 0:
        return                                # a full group completed: nothing live
    saved_kv, saved_sc, m = saved
    parts_kv, parts_sc = [], []
    if s < O:                                 # rows before the chunk -> from the save
        lo = max(s, O - m)
        parts_kv.append(saved_kv[:, lo - (O - m): m])
        parts_sc.append(saved_sc[:, lo - (O - m): m])
    lo2 = max(s, O)                           # rows inside the chunk -> from the stash
    parts_kv.append(cs.chunk_kv[:, lo2 - O: target - O])
    parts_sc.append(cs.chunk_score[:, lo2 - O: target - O])
    new_kv = mx.concatenate(parts_kv, axis=1)
    new_sc = mx.concatenate(parts_sc, axis=1)
    mx.eval(new_kv, new_sc)
    if new_kv.shape[1] != need:
        raise RuntimeError(f"carry rebuild shape {new_kv.shape[1]} != {need}")
    cs.kv_state[:, :need] = new_kv
    cs.score_state[:, :need] = new_sc


def rollback_spec(cache, snap, target):
    off0, st = snap
    cache.offset = target
    for lc, saved in zip(cache.layers, st):
        cs = lc.comp_state
        if cs is None:
            continue
        if cs.chunk_kv is None:
            raise RuntimeError("compressor did not stash its chunk rows; "
                               "was the port patched?")
        rebuild_carry(cs, lc.ratio, off0, target, saved)


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


def spec_greedy(n_new):
    cache = model.make_cache(bsz=1, max_seq_len=MAXSEQ)
    lg = model(mx.array([prompt_ids]), cache, last_logit_only=True)
    t = mx.argmax(lg[:, -1], axis=-1)
    mx.eval(t)
    out = [int(t[0])]
    pos = cache.offset

    head = model.mtp
    dsc = head.make_cache(bsz=1)
    n_acc_hist = []

    t0 = time.perf_counter()
    while len(out) < n_new:
        # 1. draft: returns [d1 .. d_gamma]
        d_tokens, _conf = head.draft(t, model.embed, model.head, dsc, width=GAMMA)
        mx.eval(d_tokens)

        # 2. verify input = [anchor, d1 .. d_gamma]  (gamma + 1 tokens)
        verify_ids = mx.concatenate(
            [t.reshape(1, 1).astype(d_tokens.dtype), d_tokens], axis=1)
        snap = snap_spec(cache, pos)
        logits, taps = model(verify_ids, cache, return_taps=True)
        mx.eval(logits, *taps.values())

        # 3. accept: leading agreement between the target's argmax and the drafts
        targ = mx.argmax(logits, axis=-1)          # [1, gamma+1]
        mx.eval(targ)
        n_acc = 0
        for k in range(GAMMA):
            if int(targ[0, k]) == int(d_tokens[0, k]):
                n_acc += 1
            else:
                break
        bonus = int(targ[0, n_acc])
        n_acc_hist.append(n_acc)

        # 4. commit n_acc drafts + the bonus
        for k in range(n_acc):
            out.append(int(d_tokens[0, k]))
        out.append(bonus)

        # 5. rollback: keep [pos, pos+n_acc], drop the rejected tail
        target = pos + n_acc + 1
        if not a.no_rollback:
            rollback_spec(cache, snap, target)
        pos = target

        # 6. ctx feed: the ACCEPTED prefix's tapped hiddens (production: ~2945)
        ctx_cat = mx.concatenate([taps[i] for i in TAPS], axis=-1)
        head.append_ctx(ctx_cat[:, :n_acc + 1], dsc)

        t = mx.array([bonus])
    dt = time.perf_counter() - t0
    return out[:n_new], dt, n_acc_hist


print("=== baseline: plain greedy ===")
base, dt_base = plain_greedy(a.n)
print(f"  {len(base)} tokens in {dt_base:.3f}s -> {len(base)/dt_base:.2f} tok/s")
print(f"  first 12: {base[:12]}")
print()

if a.no_mtp:
    print("(--no-mtp: stopping after the baseline)")
    sys.exit(0)

print("=== speculative (DSpark) ===")
spec, dt_spec, hist = spec_greedy(a.n)
print(f"  {len(spec)} tokens in {dt_spec:.3f}s -> {len(spec)/dt_spec:.2f} tok/s")
print(f"  first 12: {spec[:12]}")
print()

print("=== VERDICT ===")
if spec == base:
    print(f"  tokens IDENTICAL to plain greedy  ({len(base)} tokens)  OK")
else:
    print("  tokens DIVERGE  FAIL")
    for i, (x, y) in enumerate(zip(spec, base)):
        if x != y:
            print(f"    first divergence at index {i}: spec={x} base={y}")
            print(f"    spec[{max(0,i-4)}:{i+4}] = {spec[max(0,i-4):i+4]}")
            print(f"    base[{max(0,i-4)}:{i+4}] = {base[max(0,i-4):i+4]}")
            break
print()
c = collections.Counter(hist)
print("=== acceptance (4-layer body: NOT representative) ===")
print(f"  rounds: {len(hist)}  mean accepted: {sum(hist)/len(hist):.2f} / {GAMMA}")
for k in sorted(c):
    print(f"    n_acc={k}: {c[k]:3d}  ({c[k]/len(hist):5.1%})")
print()
print(f"  speedup here: {dt_base/dt_spec:.2f}x")
print("  (a 4-layer body drafts near-randomly, so acceptance is low by")
print("   construction; the identical-token result is the real finding)")
print()
print("SPEC-LOOP-DONE")
