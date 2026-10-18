"""Sampling for the dsv41 engine.

Greedy is the DEFAULT and stays byte-for-byte what it was: a request with no
temperature (or temperature 0) never enters this module's sampling path.

For temperature > 0 the engine samples HONESTLY through speculative decoding.
The DSpark head proposes tokens with no distribution of its own (its draft is an
argmax), so the draft distribution q is one-hot -- which collapses the standard
speculative-sampling rejection rule (Leviathan et al. 2023; Chen et al. 2023) to:

    accept draft x_i with probability p_i(x_i)

and, at the first rejection, sample from the residual p_i with x_i's mass
removed and renormalized. That is exactly the rejection-sampling scheme with
q = one-hot(x_i): the residual (p - q)+ is p minus the draft's point mass, so
the accepted-plus-residual mixture reconstructs p exactly. When every draft is
accepted, the bonus token is drawn from p at the last row.

RANK DETERMINISM. Both TP ranks hold the same full-vocab logits row (the ban
path already relies on this), so the only thing that could diverge is the RNG.
Every draw therefore comes from an explicit key split from the turn's seed in a
FIXED order -- acceptance uniforms, then the residual draw or the bonus draw.
Same seed + same rows => same tokens on both ranks, with no collective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

NEG_INF = float("-inf")


@dataclass(frozen=True)
class Sampling:
    """Resolved sampling parameters for one request."""

    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    #: Seed for the turn's RNG. Both TP ranks MUST resolve the same value (the
    #: engine derives it from the task id when the request carries none).
    seed: int = 0

    @property
    def greedy(self) -> bool:
        """True when this request must take the untouched argmax path."""
        return not (self.temperature is not None and float(self.temperature) > 0.0)


def resolve_sampling(
    params: Any,
    *,
    default_temperature: float | None = None,
    default_top_p: float | None = None,
    default_top_k: int | None = None,
    default_min_p: float | None = None,
    seed: int = 0,
) -> Sampling:
    """Resolve a request's sampling knobs (request value, else engine default).

    A negative temperature is a request error upstream; here it clamps to 0 so a
    malformed value can never silently enable the sampling path.
    """
    temperature = getattr(params, "temperature", None)
    if temperature is None:
        temperature = default_temperature
    temperature = 0.0 if temperature is None else max(0.0, float(temperature))

    top_p = getattr(params, "top_p", None)
    if top_p is None:
        top_p = default_top_p
    top_p = None if top_p is None else float(top_p)
    if top_p is not None and not (0.0 < top_p <= 1.0):
        top_p = None

    top_k = getattr(params, "top_k", None)
    if top_k is None:
        top_k = default_top_k
    top_k = None if top_k is None else int(top_k)
    if top_k is not None and top_k <= 0:
        top_k = None

    min_p = default_min_p
    min_p = None if min_p is None else float(min_p)
    if min_p is not None and not (0.0 < min_p < 1.0):
        min_p = None

    return Sampling(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        seed=int(seed),
    )


def apply_sampling(logits: mx.array, sampling: Sampling) -> mx.array:
    """Temperature + top-k/top-p/min-p, row-wise, in logit space.

    Stateless: the same rows always produce the same result, which is what makes
    the two TP ranks agree without communicating. Top-k applies first (a hard
    cap), then top-p (nucleus), then min-p -- the order the OpenAI-style stacks
    use and the order mlx-lm's own sampler applies them.
    """
    from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p

    out = logits
    t = float(sampling.temperature)
    if t > 0.0 and t != 1.0:
        out = out / t
    if sampling.top_k is not None:
        out = apply_top_k(out, int(sampling.top_k))
    if sampling.top_p is not None and float(sampling.top_p) < 1.0:
        out = apply_top_p(out, float(sampling.top_p))
    if sampling.min_p is not None:
        out = apply_min_p(out, float(sampling.min_p))
    return out


def _apply_penalties(rows: mx.array, per_row: list[dict[int, float]] | None) -> mx.array:
    """Add the escape's per-row penalties (-inf = hard ban) to raw logits."""
    if not per_row:
        return rows
    out = rows
    for i, penalties in enumerate(per_row[: out.shape[0]]):
        if not penalties:
            continue
        idx = mx.array([int(j) for j in penalties], dtype=mx.int32)
        vals = mx.array([float(v) for v in penalties.values()], dtype=out.dtype)
        out = out.at[i, idx].add(vals)
    return out


def _fully_banned_fallback(rows: mx.array, raw: mx.array) -> mx.array:
    """A row whose every candidate was banned would NaN the softmax: use the raw row."""
    dead = mx.max(rows, axis=-1, keepdims=True) == mx.array(NEG_INF)
    return mx.where(dead, raw, rows)


def _categorical(logits: mx.array, key: Any) -> int:
    """Draw an id from ``softmax(logits)`` via an explicit inverse-CDF.

    An explicit inverse-CDF draw: a couple of ops, deterministic for a given key,
    and it treats ``-inf`` (banned) entries as zero mass. Equivalent to
    ``mx.random.categorical`` but with the normalisation visible, which is what the
    rank-parity argument depends on.
    """
    z = logits - mx.max(logits)
    weights = mx.exp(z)
    total = mx.sum(weights)
    u = mx.random.uniform(shape=(1,), key=key) * total
    cdf = mx.cumsum(weights)
    idx = int(mx.sum((cdf < u).astype(mx.int32)).item())
    return min(idx, int(logits.shape[-1]) - 1)


def _row_draw(row_t: mx.array, key: Any) -> int:
    return _categorical(row_t, key)


def sample_one(
    raw_logits: mx.array,
    sampling: Sampling,
    *,
    banned: dict[int, float] | None = None,
    key: Any | None = None,
    step: int = 0,
) -> int:
    """Sample ONE token from a single-row logits array (non-speculative path).

    Order is load-bearing: escape penalties (logit space) -> temperature ->
    top-k/top-p/min-p. A fully banned row falls back to its raw logits rather
    than sampling from an all ``-inf`` vector.
    """
    if sampling.greedy:
        raise ValueError("sample_one is for the sampling path only")
    key = (
        mx.random.key((int(sampling.seed) + int(step) * 0x9E3779B1) % (2**32))
        if key is None
        else key
    )
    raw = raw_logits if raw_logits.ndim > 1 else raw_logits.reshape(1, -1)
    rows = _apply_penalties(raw, [banned] if banned else None)
    rows = _fully_banned_fallback(rows, raw)
    rows = apply_sampling(rows, sampling)
    return _row_draw(rows[0], key)


def sample_prefix(
    raw_logits: mx.array,
    draft: list[int],
    *,
    sampling: Sampling,
    banned_per_row: list[dict[int, float]] | None = None,
    key: Any | None = None,
    step: int = 0,
) -> tuple[list[int], int]:
    """Rejection-sample a draft prefix; returns ``(committed, accepted_count)``.

    ``raw_logits`` is the ``[1 + gamma, vocab]`` verify batch: row i predicts
    ``draft[i]`` and row ``gamma`` predicts the bonus token. Penalties land
    before sampling, then temperature/truncation, then the rejection rule
    (accept with probability p(draft); on rejection draw from the residual p
    with the rejected token's mass removed and renormalized).
    """
    gamma = len(draft)
    if gamma == 0:
        return [], 0
    # The step is folded into the key: re-deriving it from the seed alone every
    # round would replay the SAME accept uniforms, correlating accept/reject
    # decisions across rounds (a bias the per-seed tests cannot see). ``step`` is
    # the round counter, identical on every TP rank.
    key = (
        mx.random.key((int(sampling.seed) + int(step) * 0x9E3779B1) % (2**32))
        if key is None
        else key
    )
    k_accept, k_res, k_bonus = mx.random.split(key, 3)

    raw = raw_logits
    rows = _apply_penalties(raw, banned_per_row)
    rows = _fully_banned_fallback(rows, raw)
    rows = apply_sampling(rows, sampling)

    logp = rows - mx.logsumexp(rows, axis=-1, keepdims=True)
    draft_idx = mx.array([int(t) for t in draft], dtype=mx.int32)
    p_draft = mx.exp(
        mx.take_along_axis(logp[:gamma], draft_idx[:, None], axis=-1).squeeze(-1)
    )
    u = mx.random.uniform(shape=(gamma,), key=k_accept)
    accepted = int(mx.cumprod((u < p_draft).astype(mx.int32)).sum().item())

    if accepted >= gamma:
        token = _row_draw(rows[gamma], k_bonus)
        return list(draft) + [token], gamma

    # Rejection: draw from p with the rejected draft token's mass removed. A peaked
    # row under a tight top-p/min-p can leave the draft as the SOLE survivor; its
    # residual then has no mass at all, and "rejecting" would sample noise (or NaN).
    # Committing the draft is the exact answer there (p(draft) was ~1).
    rejected = int(draft[accepted])
    residual = mx.where(
        mx.arange(rows.shape[-1]) == rejected,
        mx.array(NEG_INF, dtype=rows.dtype),
        rows[accepted],
    )
    _r = residual.astype(mx.float32)
    if not float(mx.sum(mx.exp(_r - mx.max(_r))).item()) > 0.0:
        return list(draft[: accepted + 1]), accepted + 1
    token = _row_draw(residual, k_res)
    return list(draft[:accepted]) + [int(token)], accepted


def sampled_logprobs(
    raw_logits: mx.array,
    committed: list[int],
    sampling: Sampling,
    *,
    banned_per_row: list[dict[int, float]] | None,
    k: int,
) -> list[tuple[float, list[int], list[float]]]:
    """``(selected_logprob, top_k_ids, top_k_logprobs)`` for SAMPLED tokens.

    Under sampling the committed id is not the argmax, so the model's own
    argmax-based logprob path cannot be used: read the value off the transformed
    row at the committed id and take the top-k over the same row.
    """
    if not k:
        return []
    rows = _apply_penalties(raw_logits, banned_per_row)
    rows = _fully_banned_fallback(rows, raw_logits)
    rows = apply_sampling(rows, sampling)
    logp = rows - mx.logsumexp(rows, axis=-1, keepdims=True)
    out: list[tuple[float, list[int], list[float]]] = []
    for i, token in enumerate(committed[: rows.shape[0]]):
        row = logp[i]
        sel = float(row[int(token)].item()) if False else row[int(token)]
        top_idx = mx.argsort(row)[-int(k):][::-1]
        top_val = row[top_idx]
        mx.eval(sel, top_idx, top_val)
        out.append((float(sel.item()), [int(v) for v in top_idx.tolist()], [float(v) for v in top_val.tolist()]))
    return out
