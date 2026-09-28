"""DSv4.1 thinking-marker detection for exo's output pipeline.

THE PROBLEM. DeepSeek-V4.1 does not spell its reasoning markers as plain text in
the vocabulary: it has single special tokens
``<｜begin▁of▁thinking｜>`` (id 128821) and ``<｜end▁of▁thinking｜>``, with
FULLWIDTH vertical bars (U+FF5C). ``mlx_lm.tokenizer_utils._infer_thinking``
only recognises the ASCII spellings (``" thinking"``/``"</think>"``,
longcat, XTML), so exo's ``TokenizerWrapper`` reports
``has_thinking=False`` for this tokenizer. That is not cosmetic: with no think
markers, exo's output pipeline never routes reasoning into
``reasoning_content`` AND cannot tell that a continuation prompt already ends
inside a reasoning block. Both are properties exo's DSv4 (V4) serving has and
DSv4.1 must match, so the engine resolves the markers itself.

HOW. Three ways were possible:

1. Change ``_infer_thinking`` in the mlx-lm fork.  Rejected for now: it is a
   submodule we do not own in this stream, and the change would affect every
   model exo serves.
2. Wrap the tokenizer for this model on the exo side.  The wrap has to live
   *inside* the TokenizerWrapper instance anyway, because everything downstream
   reads ``tokenizer.think_start``/``has_thinking`` off that object.
3. Populate the wrapper's marker state directly, which is what this module does.

The values written here are exactly what ``_infer_thinking`` would have produced
had it known the DSv4.1 spelling: the marker STRINGS (never token ids). That
matters -- the streamed text arrives with ``skip_special_tokens=1`` (the default
in ``stream_generate``), so the markers are decoded literal text; matching on
strings is what makes ``parse_thinking_models`` work, and matching on ids (as
``fix_unmatched_think_end_tokens`` does for the prompt) is a separate, correct
use that keeps working because ids 128821/128822 are real vocab entries.

FAILURE MODE IF mlx-lm CHANGES. ``TokenizerWrapper.__init__`` sets ``_think_start``
et al. as plain attributes (not read-only properties), so this patch is a
straight attribute write; if the attribute is ever renamed the guard below raises
at load time instead of silently leaving ``has_thinking=False``. Setting
``EXO_DSV41_THINK_MARKERS=0`` disables the patch entirely.
"""

from __future__ import annotations

import os

from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.worker.runner.bootstrap import logger

#: DSv4.1's reasoning delimiters, from the checkpoint's own vocab: both are
#: single added tokens (a real tool call likewise emits DSML as one token).
THINK_START = "<think>"
THINK_END = "</think>"

_MARKER_ENV = "EXO_DSV41_THINK_MARKERS"


def _token_ids(tokenizer: object, marker: str) -> tuple[int, ...] | None:
    """Vocab id(s) for ``marker`` in ``tokenizer``, or None when absent."""
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if get_vocab is None:
        return None
    try:
        vocab: dict[str, int] = get_vocab()
    except Exception:  # noqa: BLE001 -- marker detection must never break a load
        return None
    tid = vocab.get(marker)
    return None if tid is None else (tid,)


def ensure_thinking_markers(tokenizer: TokenizerWrapper) -> bool:
    """Teach ``tokenizer`` its DSv4.1 reasoning markers. True when applied.

    Idempotent, and a no-op when another code path already resolved markers
    (e.g. a future mlx-lm that knows this spelling, or the harness that built
    the tokenizer): in that case the existing value is left untouched.
    """
    if os.environ.get(_MARKER_ENV, "1") != "1":
        logger.warning(
            f"[DSV41] {_MARKER_ENV}=0: reasoning markers left unresolved — "
            "reasoning will stream as content and mid-reasoning prefill "
            "continuations will be misclassified."
        )
        return False

    if tokenizer.has_thinking:
        logger.info(
            f"[DSV41] tokenizer already reports thinking markers "
            f"({tokenizer.think_start!r} <-> {tokenizer.think_end!r})"
        )
        return False

    hf = getattr(tokenizer, "_tokenizer", tokenizer)
    start_tokens = _token_ids(hf, THINK_START)
    end_tokens = _token_ids(hf, THINK_END)
    if start_tokens is None or end_tokens is None:
        logger.warning(
            "[DSV41] DSv4.1 thinking markers are absent from this tokenizer's "
            "vocab; leaving has_thinking=False (reasoning will stream as "
            "content). Expected tokens: "
            f"{THINK_START!r}/{THINK_END!r}"
        )
        return False

    for attr in ("_think_start", "_think_end", "_think_start_tokens", "_think_end_tokens"):
        if not hasattr(tokenizer, attr):
            raise RuntimeError(
                f"mlx-lm TokenizerWrapper no longer exposes {attr}; the DSv4.1 "
                "thinking-marker patch needs to be re-pointed at the new "
                "attribute (see exo/worker/engines/mlx/dsv41/thinking.py)."
            )
    tokenizer._think_start = THINK_START
    tokenizer._think_end = THINK_END
    tokenizer._think_start_tokens = start_tokens
    tokenizer._think_end_tokens = end_tokens
    logger.info(
        f"[DSV41] reasoning markers resolved by token id: "
        f"{THINK_START!r} (id {start_tokens[0]}) <-> {THINK_END!r} "
        f"(id {end_tokens[0]}). exo's thinking parser will route reasoning into "
        "reasoning_content."
    )
    return True
