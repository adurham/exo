"""The DSv4.1 sentinel strings, pinned to the checkpoint they came from.

Provenance: read from ``~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw``
on macstudio-m4-1 on 2026-09-29 --

* ``tokenizer.json`` adds the sentinel as a single token, id **128825**,
  utf-8 ``efbd9c44534d4cefbd9c``;
* ``chat_template.jinja`` sets ``dsml = "\uff5cDSML\uff5c"`` and spells the tags
  ``"<" ~ dsml ~ " calls>"`` / ``"<" ~ dsml ~ " invoke name=..."`` /
  ``"<" ~ dsml ~ " parameter name=..."`` -- i.e. a SPACE after the sentinel;
* the thinking markers are ``\u003cthink\u003e`` / ``\u003c/think\u003e``, ids
  128821 / 128822, plain ASCII.

These tests exist so a future edit to the parser cannot quietly change what
"the model actually emits": the literals below are asserted byte-for-byte
against both the module constants and the checkpoint's own values.
"""

from __future__ import annotations

from exo.worker.engines.mlx.dsv41 import dsml, thinking

#: The checkpoint's values, spelled out independently of the modules under
#: test (utf-8 hex + codepoints, so a renderer that "helpfully" turns a
#: fullwidth bar into a fullwidth bar cannot hide a mismatch).
SENTINEL_UTF8_HEX = "efbd9c44534d4cefbd9c"
SENTINEL_CODEPOINTS = (0xFF5C, 0x44, 0x53, 0x4D, 0x4C, 0xFF5C)

THINK_START = "<" + "think" + ">"
THINK_END = "<" + "/think" + ">"
THINK_START_UTF8_HEX = "3c7468696e6b3e"
THINK_END_UTF8_HEX = "3c2f7468696e6b3e"


def test_sentinel_is_byte_identical_to_the_checkpoint_tokenizer():
    assert dsml.DSML_V41.encode("utf-8").hex() == SENTINEL_UTF8_HEX
    assert tuple(ord(c) for c in dsml.DSML_V41) == SENTINEL_CODEPOINTS
    # V4 and V4.1 use the same sentinel; only the tag names differ.
    assert dsml.DSML_V4 == dsml.DSML_V41


def test_wrapper_and_tag_spellings_match_the_chat_template():
    sentinel = dsml.DSML_V41
    assert dsml.CALLS_START_V41 == f"<{sentinel} calls>"
    assert dsml.CALLS_END_V41 == f"</{sentinel} calls>"
    # The V4.1 template writes the space INSIDE the tag, so the invoke and
    # parameter tags carry it too. Pin the exact strings the model emits and
    # the exact V4 spellings they normalize to.
    assert dsml._V41_TO_V4[0] == (f"{sentinel} invoke", f"{sentinel}invoke")
    assert dsml._V41_TO_V4[1] == (f"{sentinel} parameter", f"{sentinel}parameter")
    assert dsml._V41_TO_V4[2] == (f"{sentinel} calls", f"{sentinel}tool_calls")


def test_thinking_markers_are_byte_identical_to_the_checkpoint_tokenizer():
    assert thinking.THINK_START.encode("utf-8").hex() == THINK_START_UTF8_HEX
    assert thinking.THINK_END.encode("utf-8").hex() == THINK_END_UTF8_HEX
    assert thinking.THINK_START == THINK_START
    assert thinking.THINK_END == THINK_END


def test_thinking_marker_ids_are_the_checkpoints_own():
    assert thinking.THINK_START_ID == 128821
    assert thinking.THINK_END_ID == 128822


def test_dsml_v41_declares_the_v41_tag_names_not_the_v4_ones():
    """A regression guard for the whole point of this module.

    ``parse_deepseek_v4`` looks for ``<|DSML|tool_calls>``; if the V4.1
    constants ever drift back to the V4 spelling, the engine would silently
    stop recognising tool calls (they would stream out as raw text).

    The V4.1 wrapper is spelled with a space after the sentinel AND without the
    ``tool_`` prefix, so the two are related but not equal.
    """
    sentinel = dsml.DSML_V41
    assert dsml.CALLS_START_V41 != dsml._CALLS_START_V4
    assert dsml.CALLS_START_V41 == f"<{sentinel} calls>"
    assert (
        dsml.CALLS_START_V41.replace(f"{sentinel} ", sentinel).replace(
            "calls", "tool_calls"
        )
        == dsml._CALLS_START_V4
    )
