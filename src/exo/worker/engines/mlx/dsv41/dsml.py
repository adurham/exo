"""DeepSeek-V4.1's DSML dialect, expressed with exo's existing DSML machinery.

WHY THIS EXISTS. V4.1 renamed the tool-call sentinel AND dropped the underscore
in the tag names:

    V4 (exo production, ``deepseek_v4_encoding.py``)   V4.1 (this checkpoint)
    <｜DSML｜tool_calls>                          <｜DSML｜ calls>
    <｜DSML｜invoke name="x">                    <｜DSML｜ invoke name="x">
    <｜DSML｜parameter name="k" string="true">v  <｜DSML｜ parameter name="k" string="true">v

``mlx_lm.chat_templates.deepseek_v32.dsml_token`` (which exo's vendor module
rebuilds from) is the V4 spelling, so ``parse_deepseek_v4`` cannot see a V4.1
block at all: the wrapper marker never matches. The checkpoint's own
``chat_template.jinja`` agrees with the reference encoder (``dsml_token = "｜DSML｜"``,
``tool_calls_block_name = " calls"``, ``tool_call_tag_name = "｜DSML｜"``,
``tool_parameter_tag_name = "｜DSML｜"``) -- note the leading space in each tag name.

RATHER THAN FORKING THE PARSER, this module normalizes a V4.1 block into the V4
spelling and hands it to exo's own ``parse_dsml_output``. That keeps ONE
implementation of the subtle parts (typed-parameter decoding, the
``string="true|false"`` semantics, tag-garble repair, the
``ToolCallItem`` shape) and limits this module to the dialect translation plus
the two sentinel-keyed regexes that are genuinely V4-specific
(``strip_dsml_markers`` and the orphan-content stripper). The streaming
skeleton itself is exo's ``_parse_dsml_stream``, unchanged: it is already
parameterized on the wrapper markers, the body parser and the special-token ids.
"""

from __future__ import annotations

import re
from collections.abc import Generator

from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.api.types import ToolCallItem
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
    ToolCallResponse,
)
from exo.worker.engines.mlx.vendor.dsml_encoding import parse_dsml_output
from exo.worker.runner.bootstrap import logger

#: V4.1's sentinel token (``｜DSML｜``, one added token in the checkpoint vocab) and
#: the V4 sentinel it is translated to. Both are checked against the checkpoint
#: tokenizer at import time by ``resolve_dsml_v41_ids``.
DSML_V41 = "\uff5cDSML\uff5c"
DSML_V4 = "\uff5cDSML\uff5c"

#: Wrapper markers as the model actually emits them (space included).
CALLS_START_V41 = f"<{DSML_V41} calls>"
CALLS_END_V41 = f"</{DSML_V41} calls>"

_CALLS_START_V4 = f"<{DSML_V4}tool_calls>"
_CALLS_END_V4 = f"</{DSML_V4}tool_calls>"

#: Translate V4.1 tag names back to the V4 spelling parse_dsml_output expects.
#: Order matters: the two-part names first, then the bare sentinel.
_V41_TO_V4 = (
    (f"{DSML_V41} invoke", f"{DSML_V4}invoke"),
    (f"{DSML_V41} parameter", f"{DSML_V4}parameter"),
    (f"{DSML_V41} calls", f"{DSML_V4}tool_calls"),
    (DSML_V41, DSML_V4),
)

#: Orphan-tag stripper for emitted content, keyed on the V4.1 spelling.
#: ``<｜DSML｜ invoke ...>`` / ``<｜DSML｜ calls>`` / `</｜DSML｜ parameter>` and the bare
#: sentinel. Shaped like ``dsml_encoding._DSML_TAG_PATTERN`` plus the optional
#: space that V4.1 puts between sentinel and name.
_DSML_V41_TAG = re.compile(rf"</?{re.escape(DSML_V41)}\s?\w+(?:\s+[^>]*)?>")
_DSML_V41_ORPHAN = re.compile(rf"(?:<\s*/?\s*)?{re.escape(DSML_V41)}")


def normalize_dsml_v41(text: str) -> str:
    """Rewrite a V4.1 DSML block into the V4 spelling (for parsing only)."""
    for old, new in _V41_TO_V4:
        text = text.replace(old, new)
    return text


def parse_dsml_v41_body(text: str) -> list[ToolCallItem] | None:
    """Body parser for the V4.1 dialect, using exo's V4 parser underneath."""
    return parse_dsml_output(normalize_dsml_v41(text))


def strip_dsml_v41(text: str) -> str:
    """Strip V4.1 DSML control tokens from text (never leaks the sentinel)."""
    return _DSML_V41_ORPHAN.sub("", _DSML_V41_TAG.sub("", text))


def resolve_dsml_v41_ids(tokenizer: TokenizerWrapper) -> frozenset[int]:
    """Vocab id of the V4.1 sentinel, for real-vs-quoted detection.

    Same contract as ``model_output_parsers._resolve_dsml_special_token_ids``:
    an EMPTY set means "cannot resolve" and makes the stream parser fall back to
    text-only detection, so a tokenizer without the marker never regresses below
    today's behaviour. The V4 sentinel is deliberately NOT included here -- a V4
    tool call emitted by this checkpoint is not something the checkpoint was
    trained to do, and including it would weaken the detector.
    """
    hf = getattr(tokenizer, "_tokenizer", tokenizer)
    try:
        vocab: dict[str, int] = hf.get_vocab()
    except Exception:  # noqa: BLE001 -- detection must never break a request
        return frozenset()
    ids: set[int] = set()
    for candidate in (DSML_V41, f"<{DSML_V41} calls>", f"<{DSML_V41} invoke>"):
        tid = vocab.get(candidate)
        if tid is not None:
            ids.add(tid)
    if not ids:
        logger.warning(
            "[DSV41] tokenizer has no ｜DSML｜ sentinel; DSML tool-call detection "
            "falls back to text-only matching"
        )
    return frozenset(ids)


def strip_orphan_dsml_v41(
    stream: Generator[GenerationResponse | ToolCallResponse | None],
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    """V4.1 analogue of ``_strip_orphan_dsml_from_content``.

    Enforces the standing invariant that the DSML sentinel never reaches
    displayed text, for closing tags the stream parser never committed to (the
    model bailing out of a repetition loop inside a reasoning block emits
    ``</｜DSML｜ invoke></｜DSML｜ calls>`` with no opener -- observed with the V4
    sentinel on production DSv4, and structurally identical here).
    """
    for item in stream:
        if (
            isinstance(item, GenerationResponse)
            and item.text
            and DSML_V41 in item.text
        ):
            cleaned = strip_dsml_v41(item.text)
            if cleaned == item.text:
                yield item
            elif cleaned:
                yield item.model_copy(update={"text": cleaned})
            # else: the chunk was entirely orphan markers -- drop it
        else:
            yield item
