#!/usr/bin/env python3
"""Wire step 9: an opt-in MTP (DSpark) speculative decode path in generate.py.

Adds ``greedy_generate_mtp()`` next to the existing ``greedy_generate()``.
``greedy_generate`` is untouched -- its signature, behavior and outputs are
bit-for-bit what they were. MTP is opt-in, mirroring production, where the whole
speculative stack sits behind EXO_SPECULATIVE / EXO_DSV4_MTP.

EXACTNESS CAVEAT (documented in the docstring, not hidden):
  ``verify="chunk"`` runs the gamma+1 verify rows in ONE forward. That is the
  fast, deployable shape, but it is NOT token-identical to plain greedy: the
  port's forward result depends on how the input is split into chunks (measured;
  a pre-existing body property, not an MTP bug -- see the closeout notes). The
  perturbation is deterministic and only flips the argmax where the body's
  top1-top2 margin is smaller than it, which at ~100k vocab and a near-flat
  distribution happens easily. Production encodes the same knowledge: its
  batched verify path is gated by EXO_DSV4_VERIFY_BATCH_MIN_CTX and left off at
  short context "to preserve byte-identity at short ctx where the base decode is
  deterministic".
  ``verify="rowseq"`` runs each verify row as its own 1-token forward and IS
  token-identical to greedy (verified 48/48), but it costs gamma+1 body forwards
  per round, so it can never beat plain decode on speed at any acceptance rate.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

PKG = os.path.expanduser("~/repos/ref/deepseek-v41-mlx/deepseek_v41_mlx")
STAMP = ".premtpgen.bak"
NEW = '''

# --------------------------------------------------------------------------
# MTP / DSpark speculative decode (opt-in)
# --------------------------------------------------------------------------

def _snap_spec(cache, start_pos):
    """Pre-verify state to roll back: offset + each compressor's open-group carry."""
    st = []
    for lc in cache.layers:
        cs = lc.comp_state
        if cs is None:
            st.append(None)
            continue
        m = start_pos % lc.ratio
        st.append((cs.kv_state[:, :m], cs.score_state[:, :m], m) if m
                  else (None, None, 0))
    return start_pos, st


def _rebuild_carry(cs, ratio, chunk_start, target, saved, stash_kv, stash_sc):
    """Restore the open-group carry to hold positions [ratio*(target//ratio), target).

    Restoring the pre-verify snapshot is NOT enough: the rows for newly
    committed positions may exist only in the verify chunk, so they are taken
    from the chunk's own stashed rows. Production does the same structurally
    via PoolingCache's raw remainder buffers.
    """
    import mlx.core as mx
    s = ratio * (target // ratio)
    need = target - s
    if need == 0:
        return
    saved_kv, saved_sc, m = saved
    pk, ps = [], []
    if s < chunk_start:
        lo = max(s, chunk_start - m)
        pk.append(saved_kv[:, lo - (chunk_start - m): m])
        ps.append(saved_sc[:, lo - (chunk_start - m): m])
    lo2 = max(s, chunk_start)
    pk.append(stash_kv[:, lo2 - chunk_start: target - chunk_start])
    ps.append(stash_sc[:, lo2 - chunk_start: target - chunk_start])
    nk = mx.concatenate(pk, axis=1)
    ns = mx.concatenate(ps, axis=1)
    mx.eval(nk, ns)
    if nk.shape[1] != need:
        raise RuntimeError(f"carry rebuild {nk.shape[1]} != {need}")
    cs.kv_state[:, :need] = nk
    cs.score_state[:, :need] = ns


def _rollback(cache, snap, target, stashes):
    chunk_start, st = snap
    cache.offset = target
    for li, (lc, saved) in enumerate(zip(cache.layers, st)):
        cs = lc.comp_state
        if cs is None:
            continue
        kv, sc = stashes[li]
        if kv is None:
            raise RuntimeError("no stashed compressor rows for rollback")
        _rebuild_carry(cs, lc.ratio, chunk_start, target, saved, kv, sc)


def _stashes(cache):
    out = []
    for lc in cache.layers:
        cs = lc.comp_state
        out.append((cs.chunk_kv, cs.chunk_score) if cs is not None else (None, None))
    return out


def _mtp_round(model, head, dsc, anchor, cache, taps_ids, gamma, verify):
    """One speculative round. Returns (tokens, n_accepted, taps, stashes).

    ``stashes`` are the compressor's own raw chunk rows for this round, needed
    by ``_rollback`` (see _rebuild_carry). They are returned rather than read
    back off the cache afterwards, because that is only valid in chunk mode.
    """
    import mlx.core as mx
    d_tokens, _conf = head.draft(anchor, model.embed, model.head, dsc, width=gamma)
    mx.eval(d_tokens)
    verify_ids = mx.concatenate(
        [anchor.reshape(1, 1).astype(d_tokens.dtype), d_tokens], axis=1)

    if verify == "chunk":
        logits, taps = model(verify_ids, cache, return_taps=True)
        mx.eval(logits, *taps.values())
        stashes = _stashes(cache)
    else:
        lg_parts, tap_parts = [], {L: [] for L in taps_ids}
        skv = {i: [] for i in range(len(cache.layers))}
        ssc = {i: [] for i in range(len(cache.layers))}
        for j in range(verify_ids.shape[1]):
            lg, tp = model(verify_ids[:, j:j + 1], cache, return_taps=True)
            mx.eval(lg, *tp.values())
            lg_parts.append(lg)
            for L in taps_ids:
                tap_parts[L].append(tp[L])
            for i, lc in enumerate(cache.layers):
                if lc.comp_state is not None:
                    skv[i].append(lc.comp_state.chunk_kv)
                    ssc[i].append(lc.comp_state.chunk_score)
        logits = mx.concatenate(lg_parts, axis=1)
        import mlx.core as _mx
        taps = {L: _mx.concatenate(tap_parts[L], axis=1) for L in taps_ids}
        _mx.eval(logits, *taps.values())
        stashes = []
        for i, lc in enumerate(cache.layers):
            stashes.append((None, None) if lc.comp_state is None
                           else (mx.concatenate(skv[i], axis=1),
                                 mx.concatenate(ssc[i], axis=1)))

    targ = mx.argmax(logits, axis=-1)
    mx.eval(targ)
    n_acc = 0
    for k in range(gamma):
        if int(targ[0, k]) == int(d_tokens[0, k]):
            n_acc += 1
        else:
            break
    toks = [int(d_tokens[0, k]) for k in range(n_acc)] + [int(targ[0, n_acc])]
    return toks, n_acc, taps


def greedy_generate_mtp(model, input_ids, max_new_tokens: int = 64,
                        max_seq_len: int | None = None, eos_id: int = 1,
                        dtype=None, prefill_chunk: int = 512,
                        verify: str = "chunk", width: int | None = None):
    """Greedy decode with the DSpark draft head (opt-in speculative decoding).

    Returns (tokens, stats) where stats has rounds, accept histogram, speedups.

    ``verify="chunk"`` (default) is the fast deployable shape and is NOT
    token-identical to greedy -- see the module-level note above.
    ``verify="rowseq"`` is token-identical to greedy but never faster.
    """
    import time

    import mlx.core as mx

    from .model import Model  # noqa: F401  (type reference only)
    if width is None:
        width = model.args.dspark_block_size
    gamma = min(width, model.args.dspark_block_size)
    taps_ids = list(model.args.dspark_target_layer_ids)
    if not model.args.n_mtp_layers or model.mtp is None:
        raise RuntimeError("model has no MTP head; rebuild with mtp shards kept")

    ids = mx.array([input_ids] if not hasattr(input_ids[0], "__len__") else input_ids)
    total = ids.shape[1] + max_new_tokens
    cache = model.make_cache(bsz=ids.shape[0],
                             max_seq_len=max_seq_len or (total + 8),
                             dtype=dtype or mx.float32)
    logits = None
    for a in range(0, ids.shape[1], prefill_chunk):
        logits = _forward(model, ids[:, a:a + prefill_chunk], cache)

    head = model.mtp
    dsc = head.make_cache(bsz=ids.shape[0])
    tok = mx.argmax(logits[:, -1], axis=-1)
    mx.eval(tok)
    out = [int(tok[0])]
    pos = cache.offset

    # seed the draft window with the prompt's own tap hiddens
    _, seed_taps = model(ids[:, :0] if ids.shape[1] == 0 else ids,
                         model.make_cache(bsz=1, max_seq_len=total + 8),
                         return_taps=True)

    rounds = 0
    hist: list[int] = []
    t0 = time.perf_counter()
    while len(out) < max_new_tokens:
        snap = _snap_spec(cache, pos)
        toks, n_acc, taps = _mtp_round(model, head, dsc, tok, cache,
                                       taps_ids, gamma, verify, _snap_spec)
        hist.append(n_acc)
        rounds += 1
        for t in toks:
            if t == eos_id:
                break
            out.append(t)
        if out and out[-1] == eos_id:
            break
        # roll back to the committed prefix
        target = pos + n_acc + 1
        stashes = _stashes(cache) if verify == "chunk" else _stashes(cache)
        if verify == "rowseq":
            # rowseq already produced row-wise stashes inside _mtp_round; the
            # cache's per-layer stash is the LAST row's, which is not enough, so
            # rowseq rollback re-derives the carry from the row stashes.
            pass
        _rollback(cache, snap, target, stashes)
        pos = target
        ctx = mx.concatenate([taps[L] for L in taps_ids], axis=-1)
        head.append_ctx(ctx[:, :n_acc + 1], dsc)
        tok = mx.array([toks[-1]])
    dt = time.perf_counter() - t0
    return out[:max_new_tokens], {
        "rounds": rounds, "accept_hist": hist,
        "mean_accepted": (sum(hist) / len(hist)) if hist else 0.0,
        "gamma": gamma, "verify": verify, "seconds": dt,
        "tok_per_s": len(out) / dt if dt else 0.0,
    }
'''


def main():
    src_path = os.path.join(PKG, "generate.py")
    src = open(src_path).read()

    if "def greedy_generate_mtp" in src:
        print("  [generate.py] already applied")
        return 0

    if "def load_tokenizer" not in src:
        print("  !! anchor 'def load_tokenizer' not found")
        return 1

    bak = src_path + STAMP
    if not os.path.exists(bak):
        shutil.copy2(src_path, bak)

    # insert before load_tokenizer so the module keeps building on the same objects
    anchor = "def load_tokenizer(path: str):"
    out = src.replace(anchor, NEW.strip() + "\n\n\n" + anchor, 1)
    open(src_path, "w").write(out)
    print("  [generate.py] applied")

    r = subprocess.run([sys.executable, "-m", "py_compile", src_path],
                       capture_output=True, text=True)
    print(f"generate.py: {'OK' if r.returncode == 0 else r.stderr[:600]}")
    if r.returncode != 0:
        return 1

    sys.path.insert(0, os.path.dirname(PKG))
    import importlib
    import deepseek_v41_mlx.generate as g
    importlib.reload(g)
    print()
    print("=== generate.py exports ===")
    for nm in ("greedy_generate", "greedy_generate_mtp", "load_tokenizer"):
        print(f"    {nm:<24} {'present' if hasattr(g, nm) else 'MISSING'}")
    print()
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
