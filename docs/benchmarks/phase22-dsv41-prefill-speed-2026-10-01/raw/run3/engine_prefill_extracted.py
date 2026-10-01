def engine_prefill(
    model: Any,
    ids: Any,
    cache: Any,
    *,
    chunk: int | None = None,
    long_chunk: int | None = None,
    long_threshold: int | None = None,
    last_logit_only: bool = True,
    argmax: bool = False,
    return_taps: bool = False,
    taps_out: Any = None,
    progress: Any = None,
    **rest: Any,
) -> Any:
    """The engine's prefill loop, as a driver for ``SessionCache``.

    Same signature and chunk policy as ``prefill.prefill``/``chunked_prefill`` so
    ``SessionCache`` can take it as ``prefill_fn``, but with the engine's own
    shape: every chunk is one ``last_logit_only`` forward (intermediate chunks
    need no full-row head projection), one sync each, with no eval fences, no
    periodic pool clears and no post-prefill decode-prime probe. A reused-prefix
    turn and a cold turn run this SAME loop, so session reuse cannot change the
    tokens it produces.

    ``taps_out`` collects the per-chunk DSpark taps (the draft window's context
    feed); ``progress`` is ``fn(chunks, rows_done, elapsed_s)`` and doubles as
    the engine's cancellation point.
    """
    del rest  # tolerated, unused driver keywords (forward compatibility)
    if isinstance(ids, mx.array):
        ids_mx = ids if ids.ndim == 2 else ids[None]
    else:
        ids_mx = mx.array(np.asarray(ids, dtype=np.int64)[None])
    total = int(ids_mx.shape[1])
    if total == 0:
        raise ValueError("engine_prefill: empty ids")

    base = 512 if chunk is None else int(chunk)
    long_step = 128 if long_chunk is None else int(long_chunk)
    threshold = 10**9 if long_threshold is None else int(long_threshold)
    want_taps = bool(return_taps or taps_out is not None)

    t0 = time.perf_counter()
    out = None
    done = 0
    nchunks = 0
    last_taps = None
    while done < total:
        step = long_step if int(cache.offset) >= threshold else base
        stop = min(done + step, total)
        piece = ids_mx[:, done:stop]
        last = stop == total
        res = model(
            piece,
            cache,
            last_logit_only=True if not last else last_logit_only,
            return_taps=want_taps,
            argmax=argmax if last else False,
        )
        handle, taps = res if isinstance(res, tuple) else (res, None)
        if taps is not None and taps_out is not None:
            taps_out.append(taps)
            last_taps = taps
        mx.eval(handle, *(taps.values() if taps else []))
        if last:
            out = handle
        done = stop
        nchunks += 1
        if progress is not None:
            progress(nchunks, done, time.perf_counter() - t0)
    if return_taps:
        return out, (last_taps if last_taps is not None else {})
    return out