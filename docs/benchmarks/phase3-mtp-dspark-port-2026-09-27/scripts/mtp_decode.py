"""MTP / DSpark speculative decode for the DeepSeek-V4.1 port.

This is the round loop validated by p6d_spec_rowseq.py: with ``verify="rowseq"``
it produced 48/48 tokens IDENTICAL to plain greedy decode over 47 rounds, which
is the property that proves the accept / rollback / ctx-feed bookkeeping.

A new module rather than an edit to ``generate.py``: nothing existing changes.

CYCLE (mirrors production's dsv4_mtp.py / pp_speculation.py)
------------------------------------------------------------
    anchor t --draft(block)--> verify forward over [anchor, d1..d_gamma]
                                    |
              accept = #leading rows where the target argmax == the draft
                                    |
          commit n_acc drafts + 1 bonus (the target's own argmax at row n_acc)
                                    |
        roll back the cache to pos+n_acc+1, feed taps[:, :n_acc+1] as ctx-KV

BLOCK CONVENTION. ``head.draft()`` builds ``block_ids = [anchor, noise x (bs-1)]``
and returns ``bs`` tokens ``[d1..d_bs]``; block position 0 IS the anchor, so
``d_tokens[k]`` is the token at ``anchor+k+1``. The verify input is therefore
``[anchor, d1..d_bs]`` (bs+1 tokens), and acceptance compares the target's
argmax at row k against ``d_tokens[k]``.

ROLLBACK. The window ring and the pooled ``comp_kv`` need no repair: position p
lives at slot ``p % window`` and is rewritten by whichever chunk commits it, and
any pooled group straddling the rollback target is recomputed by the next chunk
before it is read. The compressor's OPEN-GROUP carry (``kv_state`` /
``score_state``) does need repair, and it must be REBUILT rather than merely
restored -- the rows for newly committed positions may exist only in the verify
chunk. The compressor stashes its own chunk rows (``chunk_kv`` / ``chunk_score``
/ ``chunk_start``) and :func:`rebuild_carry` splices saved carry rows (positions
< chunk start) with chunk rows (positions >= chunk start). Production does the
same thing structurally, via PoolingCache's ``buf_kv`` / ``buf_gate`` raw
remainder buffers.

THE TWO VERIFY MODES, AND WHY BOTH EXIST
----------------------------------------
``verify="chunk"`` -- the gamma+1 verify rows in ONE forward. This is the fast,
deployable shape (the per-token cost that makes MTP worth it). It is NOT
token-identical to plain greedy: the port's forward result depends on how the
input is split into chunks. That is a PRE-EXISTING property of the body, not of
MTP -- measured, and localized to layer 0, a pure-window layer with no
compressor and no indexer (compress_ratio=0), which already differs by ~2 bf16
ulps between splits. Those ulps amplify through the stack and flip the argmax
wherever the body's top1-top2 margin is smaller than the perturbation, which on
a near-flat distribution is common. Production encodes the identical knowledge:
its batched verify path is gated behind ``EXO_DSV4_VERIFY_BATCH_MIN_CTX`` and
left OFF at short context, explicitly "to preserve byte-identity at short ctx
where the base decode is deterministic".

``verify="rowseq"`` -- each verify row as its own 1-token forward, so every row
is computationally identical to a plain decode step. This IS token-identical to
greedy under the same chunking. It is the validation instrument. It can never
beat plain decode on speed: a round costs (gamma+1) body forwards plus the
draft, against gamma+1 tokens in the best case, so it wins only if
``(1 + a*gamma) * T_body > (gamma + 1) * T_body + T_draft``, i.e. only if
``a > 1`` -- impossible.
"""

from __future__ import annotations

import time

import mlx.core as mx

from .model import Model


def snap_spec(cache, start_pos: int):
    """Pre-verify state needed to roll back: offset + each open-group carry."""
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


def rebuild_carry(cs, ratio: int, chunk_start: int, target: int, saved, stash):
    """Restore the open-group carry to hold positions [ratio*(target//ratio), target).

    ``saved`` is the (kv, score, m) carried into the round; ``stash`` is the
    (kv, score) the compressor stored for this round's own rows. Splicing them
    is what makes the carry correct even when the verified positions exist only
    in the verify chunk.
    """
    s = ratio * (target // ratio)
    need = target - s
    if need == 0:
        return
    saved_kv, saved_sc, m = saved
    stash_kv, stash_sc = stash
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
        raise RuntimeError(f"carry rebuild produced {nk.shape[1]} rows, need {need}")
    cs.kv_state[:, :need] = nk
    cs.score_state[:, :need] = ns


def rollback_spec(cache, snap, target: int, stashes) -> None:
    """Drop the rejected tail and repair the compressor carries."""
    chunk_start, st = snap
    cache.offset = target
    for li, (lc, saved) in enumerate(zip(cache.layers, st)):
        cs = lc.comp_state
        if cs is None:
            continue
        if stashes[li][0] is None:
            raise RuntimeError("no stashed compressor rows; cannot roll back")
        rebuild_carry(cs, lc.ratio, chunk_start, target, saved, stashes[li])


def _stashes(cache):
    out = []
    for lc in cache.layers:
        cs = lc.comp_state
        out.append((cs.chunk_kv, cs.chunk_score) if cs is not None else (None, None))
    return out


def mtp_round(model, head, dsc, anchor, cache, taps_ids, gamma: int,
              verify: str = "rowseq"):
    """One speculative round.

    Returns ``(tokens, n_accepted, taps, stashes)`` where ``tokens`` is the
    committed run (n_accepted drafts followed by the bonus token).
    """
    d_tokens, _conf = head.draft(anchor, model.embed, model.head, dsc, width=gamma)
    mx.eval(d_tokens)
    verify_ids = mx.concatenate(
        [anchor.reshape(1, 1).astype(d_tokens.dtype), d_tokens], axis=1)

    if verify == "chunk":
        logits, taps = model(verify_ids, cache, return_taps=True)
        mx.eval(logits, *taps.values())
        stashes = _stashes(cache)
    elif verify == "rowseq":
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
        taps = {L: mx.concatenate(tap_parts[L], axis=1) for L in taps_ids}
        mx.eval(logits, *taps.values())
        stashes = []
        for i, lc in enumerate(cache.layers):
            stashes.append(
                (None, None) if lc.comp_state is None
                else (mx.concatenate(skv[i], axis=1), mx.concatenate(ssc[i], axis=1)))
    else:
        raise ValueError(f"verify must be 'chunk' or 'rowseq', got {verify!r}")

    targ = mx.argmax(logits, axis=-1)
    mx.eval(targ)
    n_acc = 0
    for k in range(gamma):
        if int(targ[0, k]) == int(d_tokens[0, k]):
            n_acc += 1
        else:
            break
    toks = [int(d_tokens[0, k]) for k in range(n_acc)] + [int(targ[0, n_acc])]
    return toks, n_acc, taps, stashes


def speculative_generate(model: Model, input_ids, max_new_tokens: int = 64,
                         max_seq_len: int | None = None, eos_id: int = 1,
                         verify: str = "rowseq", width: int | None = None,
                         prefill_chunk: int = 512):
    """Greedy speculative decode. Returns ``(tokens, stats)``.

    ``verify="rowseq"`` (default here) is token-identical to plain greedy.
    ``verify="chunk"`` is the fast shape and is NOT token-identical -- see the
    module docstring before using it for anything that must match greedy.
    """
    if not getattr(model.args, "n_mtp_layers", 0) or model.mtp is None:
        raise RuntimeError("model has no MTP head (rebuild with the mtp shards kept)")

    gamma = min(width or model.args.dspark_block_size, model.args.dspark_block_size)
    taps_ids = list(model.args.dspark_target_layer_ids)
    ids = mx.array([input_ids] if not hasattr(input_ids[0], "__len__") else input_ids)
    total = ids.shape[1] + max_new_tokens
    msl = max_seq_len or (total + 8)

    cache = model.make_cache(bsz=ids.shape[0], max_seq_len=msl)
    logits = None
    for a in range(0, ids.shape[1], prefill_chunk):
        logits = model(ids[:, a:a + prefill_chunk], cache, last_logit_only=True)
        mx.eval(logits)

    head = model.mtp
    dsc = head.make_cache(bsz=ids.shape[0])

    # seed the draft window with the prompt's tap hiddens (production feeds the
    # accepted prefix's hiddens every round; the prompt is round zero's prefix)
    seed_cache = model.make_cache(bsz=ids.shape[0], max_seq_len=msl)
    _lg, seed_taps = model(ids, seed_cache, return_taps=True)
    mx.eval(_lg, *seed_taps.values())
    head.append_ctx(mx.concatenate([seed_taps[L] for L in taps_ids], axis=-1), dsc)

    tok = mx.argmax(logits[:, -1], axis=-1)
    mx.eval(tok)
    out = [int(tok[0])]
    pos = cache.offset
    hist: list[int] = []

    t0 = time.perf_counter()
    while len(out) < max_new_tokens:
        snap = snap_spec(cache, pos)
        toks, n_acc, taps, stashes = mtp_round(
            model, head, dsc, tok, cache, taps_ids, gamma, verify)
        hist.append(n_acc)
        stop = False
        for t in toks:
            if t == eos_id:
                stop = True
                break
            out.append(t)
        if stop:
            break
        target = pos + n_acc + 1
        rollback_spec(cache, snap, target, stashes)
        pos = target
        head.append_ctx(
            mx.concatenate([taps[L] for L in taps_ids], axis=-1)[:, :n_acc + 1], dsc)
        tok = mx.array([toks[-1]])
    dt = time.perf_counter() - t0
    n = len(out)
    return out[:max_new_tokens], {
        "rounds": len(hist),
        "accept_hist": hist,
        "mean_accepted": (sum(hist) / len(hist)) if hist else 0.0,
        "gamma": gamma,
        "verify": verify,
        "seconds": dt,
        "tok_per_s": (n / dt) if dt else 0.0,
    }
