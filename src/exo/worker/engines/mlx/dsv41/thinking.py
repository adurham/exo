"""DSv4.1 thinking-marker resolution for exo's output pipeline.

WHAT THE CHECKPOINT ACTUALLY HAS (verified 2026-09-29 against
``~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`` on
macstudio-m4-1, by loading its tokenizer through this repo's mlx-lm fork):

* `` thinking``  -- a single added token, id **128821** (utf-8 ``3c7468696e6b3e``)
* `` response`` -- a single added token, id **128822** (utf-8 ``3c2f7468696e6b3e``)
* the checkpoint's own ``chat_template.jinja`` emits them literally as
  ``think_open`` / ``think_close`` around the reasoning block.

Both are plain ASCII, and ``mlx_lm.tokenizer_utils._infer_thinking`` does
recognise exactly this pair, so ``TokenizerWrapper`` reports
``has_thinking=True`` for this tokenizer and exo's standard thinking pipeline
(``parse_thinking_models`` -> ``reasoning_content``) works unmodified. That is
the happy path and it is what the tests pin.

WHY THIS MODULE STILL EXISTS. It is the single place that (a) asserts the
sentinel strings and their vocab ids are the ones this engine's parsers expect,
and (b) installs them when a tokenizer/wrapper does NOT report them -- which
is the failure mode to defend against rather than a description of today. The
markers matter beyond cosmetics: without them exo cannot route reasoning into
``reasoning_content``, and it cannot tell that a continuation prompt already
ends inside a reasoning block -- both properties DSv4 (V4) serving has and
DSv4.1 must match.

HOW THE INSTALL WORKS. Everything downstream reads ``tokenizer.think_start`` /
``think_end`` / ``has_thinking`` off the ``TokenizerWrapper`` instance, and
``TokenizerWrapper.__init__`` sets ``_think_start`` et al. as plain attributes,
so the install is a straight attribute write. The values written are the marker
STRINGS (never token ids) because streamed text arrives with
``skip_special_tokens=1`` (``stream_generate``'s default): the markers are
decoded literal text by then, which is what makes ``parse_thinking_models``
work. The token ids are recorded alongside them for the prompt-side helpers
(``fix_unmatched_think_end_tokens``) that do match by id.

FAILURE MODES. If mlx-lm ever renames those attributes the guard below raises
at load time instead of silently leaving reasoning unparsed. If the markers are
absent from the vocab entirely, the install is a no-op with a loud warning.
``EXO_DSV41_THINK_MARKERS=0`` disables the whole thing.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from exo.worker.runner.bootstrap import logger

if TYPE_CHECKING:  # MLX only, and only for the annotations below
    from mlx_lm.tokenizer_utils import TokenizerWrapper

#: DSv4.1's reasoning delimiters, byte-for-byte as they appear in the
#: checkpoint's tokenizer (ids 128821 / 128822) and chat template.
#:
#: Spelled by concatenation on purpose: a "<" immediately followed by a letter
#: is eaten by the source-writing toolchain used on this tree (an earlier
#: revision of this file silently landed as " think"). The byte-exact test in
#: tests/test_dsv41_sentinels.py is the guard.
THINK_START = "<" + "think" + ">"
THINK_END = "<" + "/think" + ">"

#: The vocab ids above, pinned so a checkpoint swap cannot silently keep
#: serving with markers that no longer match.
THINK_START_ID = 128821
THINK_END_ID = 128822

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
    (today's mlx-lm does for this tokenizer, see the module docstring): the
    existing value is left untouched. Returns True only when THIS call wrote
    the markers.
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
            f"({tokenizer.think_start!r} <-> {tokenizer.think_end!r}); "
            "leaving them as resolved."
        )
        return False

    hf = getattr(tokenizer, "_tokenizer", tokenizer)
    start_tokens = _token_ids(hf, THINK_START)
    end_tokens = _token_ids(hf, THINK_END)
    if start_tokens is None or end_tokens is None:
        logger.warning(
            "[DSV41] DSv4.1 thinking markers are absent from this tokenizer's "
            "vocab; leaving has_thinking=False (reasoning will stream as "
            f"content). Expected tokens: {THINK_START!r}/{THINK_END!r}"
        )
        return False

    for attr in (
        "_think_start",
        "_think_end",
        "_think_start_tokens",
        "_think_end_tokens",
    ):
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


def markers_match_checkpoint(tokenizer: TokenizerWrapper) -> bool:
    """True when the tokenizer's resolved markers are this checkpoint's pair.

    Cheap consistency check for load-time logging and for tests: it compares
    the STRINGS (what the parsers match) and, when the wrapper exposes them,
    the single-token ids (what the prompt-side helpers match).
    """
    if not tokenizer.has_thinking:
        return False
    if tokenizer.think_start != THINK_START or tokenizer.think_end != THINK_END:
        return False
    start_ids = getattr(tokenizer, "think_start_tokens", None)
    end_ids = getattr(tokenizer, "think_end_tokens", None)
    if start_ids is not None and tuple(start_ids) != (THINK_START_ID,):
        return False
    return end_ids is None or tuple(end_ids) == (THINK_END_ID,)
