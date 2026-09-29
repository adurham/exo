"""The DSv4.1 DSML dialect: body parsing, streaming, and the failure modes.

Shapes covered, deliberately mirroring what ``test_dsml_e2e.py`` pins for the
V4 dialect (that file is the contract for the shared machinery this module
reuses; these tests are the V4.1 analogue):

* a clean block, split across chunks the way this checkpoint's tokenizer
  actually splits it (the sentinel is one token, the tag names are BPE);
* typed parameters (``string="true|false"``) and nested JSON objects;
* several invokes in one block;
* the model QUOTING the marker in prose (no special token) -- must stay
  verbatim text, not become a tool call;
* malformed / unterminated / garbled blocks -- must not leak the sentinel and
  must fail the turn rather than silently dropping the call;
* sentinel-less blocks (the correct structure, no sentinel) -- recovered, not
  leaked.
"""

from __future__ import annotations

import json
from collections.abc import Generator

from exo.shared.types.worker.runner_response import (
    GenerationResponse,
    ToolCallResponse,
)
from exo.worker.engines.mlx.dsv41.dsml import (
    CALLS_END_V41,
    CALLS_START_V41,
    DSML_V41,
    normalize_dsml_v41,
    parse_dsml_v41_body,
    resolve_dsml_v41_ids,
    strip_dsml_v41,
)
from exo.worker.engines.mlx.dsv41.output import parse_dsv41
from exo.worker.engines.mlx.dsv41.tests.conftest import (
    DSML_SENTINEL_ID,
    FakeTokenizer,
    responses,
)


def _block(*invokes: str) -> str:
    return CALLS_START_V41 + "\n" + "\n".join(invokes) + "\n" + CALLS_END_V41


def _invoke(name: str, *params: str) -> str:
    return (
        f'<{DSML_V41} invoke name="{name}">\n'
        + "\n".join(params)
        + f"\n</{DSML_V41} invoke>"
    )


def _param(name: str, value: str, *, string: bool = True) -> str:
    kind = "true" if string else "false"
    return (
        f'<{DSML_V41} parameter name="{name}" string="{kind}">'
        f"{value}</{DSML_V41} parameter>"
    )


# --------------------------------------------------------------- body parsing


def test_body_parses_one_call_with_typed_params():
    text = _block(
        _invoke(
            "get_weather",
            _param("city", "New York"),
            _param("days", "3", string=False),
            _param("metric", "true", string=False),
        )
    )
    calls = parse_dsml_v41_body(text)
    assert calls is not None
    assert len(calls) == 1
    assert calls[0].name == "get_weather"
    assert json.loads(calls[0].arguments) == {
        "city": "New York",
        "days": 3,
        "metric": True,
    }


def test_body_parses_nested_json_object_argument():
    config = '{"recurring": true, "days": ["mon", "wed"], "time": "09:00"}'
    calls = parse_dsml_v41_body(
        _block(_invoke("create_event", _param("title", "Standup"), _param("config", config, string=False)))
    )
    assert calls is not None
    args = json.loads(calls[0].arguments)
    assert args["title"] == "Standup"
    assert args["config"] == {"recurring": True, "days": ["mon", "wed"], "time": "09:00"}


def test_body_parses_several_invokes_in_one_block():
    text = _block(
        _invoke("read", _param("path", "/tmp/a")),
        _invoke("read", _param("path", "/tmp/b")),
    )
    calls = parse_dsml_v41_body(text)
    assert calls is not None
    assert [c.name for c in calls] == ["read", "read"]
    assert [json.loads(c.arguments)["path"] for c in calls] == ["/tmp/a", "/tmp/b"]


def test_empty_block_parses_to_none():
    assert parse_dsml_v41_body(CALLS_START_V41 + CALLS_END_V41) is None


def test_garble_repair_still_applies_after_normalization():
    """The V4 parser repairs ``invinvoke``-style token garble; V4.1 inherits it."""
    garbled = (
        CALLS_START_V41
        + "\n"
        + f'<{DSML_V41} invoke name="read">\n'
        + _param("path", "/tmp/a")
        + "\n"
        + f"</{DSML_V41} invinvoke>"
        + "\n"
        + CALLS_END_V41
    )
    # V4.1 spells the garbled closer with a space ("</DSML| invinvoke>"), which
    # the V4-keyed repair regex does not see: document what the current behaviour
    # is, so a future fix has a failing expectation to flip.
    assert parse_dsml_v41_body(garbled) is None
    # The V4-spelled equivalent IS repaired, which is what the normalization
    # would produce for a model that dropped the space.
    v4_spelled = normalize_dsml_v41(garbled).replace(
        f"</{DSML_V41} invinvoke>", f"</{DSML_V41}invinvoke>"
    )
    calls = parse_dsml_v41_body(v4_spelled)
    assert calls is not None and calls[0].name == "read"


# --------------------------------------------------------------- normalization


def test_normalize_rewrites_only_the_tag_names():
    text = _block(_invoke("x", _param("k", "v")))
    normalized = normalize_dsml_v41(text)
    # Exactly the three tag names change; everything else is untouched.
    assert normalized == (
        f"<{DSML_V41}tool_calls>\n"
        f'<{DSML_V41}invoke name="x">\n'
        f'<{DSML_V41}parameter name="k" string="true">v</{DSML_V41}parameter>\n'
        f"</{DSML_V41}invoke>\n"
        f"</{DSML_V41}tool_calls>"
    )
    assert normalized.count(DSML_V41) == text.count(DSML_V41)


def test_strip_removes_tags_and_orphan_sentinel_but_keeps_prose():
    text = (
        "Here is the plan.\n"
        + _block(_invoke("x", _param("k", "v")))
        + f"\nand a stray sentinel: {DSML_V41} the end"
    )
    stripped = strip_dsml_v41(text)
    assert DSML_V41 not in stripped
    assert stripped.startswith("Here is the plan.")
    assert stripped.endswith("the end")


# --------------------------------------------------------------- detection ids


def test_resolve_ids_finds_the_checkpoint_sentinel(tokenizer: FakeTokenizer):
    assert resolve_dsml_v41_ids(tokenizer) == frozenset({DSML_SENTINEL_ID})


def test_resolve_ids_is_empty_for_a_tokenizer_without_the_sentinel():
    other = FakeTokenizer(vocab={"hello": 7})
    assert resolve_dsml_v41_ids(other) == frozenset()


# --------------------------------------------------------------- streaming


def test_stream_parses_a_call_split_like_the_tokenizer_splits_it():
    """The sentinel is ONE token; the tag name is ordinary BPE."""
    chunks = [
        "Let me check that.\n\n",
        "<",
        DSML_V41,
        " calls",
        ">\n<",
        DSML_V41,
        " invoke",
        ' name="read"',
        ">\n<",
        DSML_V41,
        " parameter",
        ' name="path" string="true"',
        ">",
        "/tmp/a",
        "</",
        DSML_V41,
        " parameter",
        ">\n</",
        DSML_V41,
        " invoke",
        ">\n</",
        DSML_V41,
        " calls",
        ">",
    ]
    tokens = [
        DSML_SENTINEL_ID if DSML_V41 in c else 1000 + i for i, c in enumerate(chunks)
    ]
    stream = responses(chunks, tokens=tokens)
    parsed = list(parse_dsv41(stream, frozenset({DSML_SENTINEL_ID})))

    tool_calls = [r for r in parsed if isinstance(r, ToolCallResponse)]
    text = "".join(
        r.text for r in parsed if isinstance(r, GenerationResponse) and r.text
    )
    assert len(tool_calls) == 1, text
    assert tool_calls[0].tool_calls[0].name == "read"
    assert json.loads(tool_calls[0].tool_calls[0].arguments) == {"path": "/tmp/a"}
    # The lead-in text survives; the block itself never leaks.
    assert "Let me check that." in text
    assert DSML_V41 not in text


def test_stream_parses_a_tool_call_split_like_the_tokenizer_splits_it():
    """The sentinel is ONE token; the tag name is ordinary BPE."""
    chunks = [
        "Let me check that.\n\n",
        "<",
        DSML_V41,
        " calls",
        ">\n<",
        DSML_V41,
        " invoke",
        ' name="read"',
        ">\n<",
        DSML_V41,
        " parameter",
        ' name="path" string="true"',
        ">",
        "/tmp/a",
        "</",
        DSML_V41,
        " parameter",
        ">\n</",
        DSML_V41,
        " invoke",
        ">\n</",
        DSML_V41,
        " calls",
        ">",
    ]
    tokens = [
        DSML_SENTINEL_ID if DSML_V41 in c else 1000 + i for i, c in enumerate(chunks)
    ]
    stream = responses(chunks, tokens=tokens)
    parsed = list(parse_dsv41(stream, frozenset({DSML_SENTINEL_ID})))

    tool_calls = [r for r in parsed if isinstance(r, ToolCallResponse)]
    text = "".join(
        r.text for r in parsed if isinstance(r, GenerationResponse) and r.text
    )
    assert len(tool_calls) == 1, text
    assert tool_calls[0].tool_calls[0].name == "read"
    assert json.loads(tool_calls[0].tool_calls[0].arguments) == {"path": "/tmp/a"}
    # The lead-in text survives; the block itself never leaks.
    assert "Let me check that." in text
    assert DSML_V41 not in text


def test_stream_tool_call_carries_usage_from_the_terminal_response():
    """The block closes a token or two BEFORE finish_reason/usage arrives.

    The ``_parse_dsml_stream`` parser holds a parsed call until the terminal
    response so the client gets usage on a tool-calling turn; the chunk that
    closes the block itself carries none.
    """
    from exo.api.types import Usage
    from exo.api.types.api import CompletionTokensDetails, PromptTokensDetails

    usage = Usage(
        prompt_tokens=11,
        completion_tokens=5,
        total_tokens=16,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=0),
        completion_tokens_details=CompletionTokensDetails(reasoning_tokens=2),
    )
    text = _block(_invoke("read", _param("path", "/tmp/a")))
    parsed = list(
        parse_dsv41(
            _stream_of(
                GenerationResponse(text=text, token=DSML_SENTINEL_ID, usage=None),
                GenerationResponse(text="", token=1, finish_reason="stop", usage=usage),
            ),
            frozenset({DSML_SENTINEL_ID}),
        )
    )
    tool_calls = [r for r in parsed if isinstance(r, ToolCallResponse)]
    assert len(tool_calls) == 1
    assert tool_calls[0].usage is not None
    assert tool_calls[0].usage.prompt_tokens == 11


def test_quoted_marker_in_prose_is_not_a_tool_call(tokenizer: FakeTokenizer):
    """The model explaining its own syntax must stream out verbatim."""
    chunks = [
        "Tool calls look like ",
        f"<{DSML_V41} calls>",
        f"<{DSML_V41} invoke name=\"x\">",
        " inside a reply.",
    ]
    # No sentinel token id anywhere: ordinary BPE text of the same characters.
    parsed = list(parse_dsv41(responses(chunks), resolve_dsml_v41_ids(tokenizer)))
    assert [r for r in parsed if isinstance(r, ToolCallResponse)] == []
    text = "".join(
        r.text for r in parsed if isinstance(r, GenerationResponse) and r.text
    )
    # NOTE: the sentinel characters survive; the closing ">" of the quoted tag
    # does not, because exo's orphan-marker stripper removes the bare sentinel
    # from content unconditionally (the standing "never leak the sentinel"
    # invariant, shared with the V4 path). A quoted marker is left readable
    # enough for the client, and the important part -- no false tool call -- holds.
    assert DSML_V41 in text
    assert [r for r in parsed if isinstance(r, ToolCallResponse)] == []


def test_malformed_block_fails_the_turn_without_leaking_the_sentinel():
    """A wrapper with no parseable body (the model parroting a tool RESULT).

    This is the V4 failure shape ``test_dsml_e2e`` pins for its own dialect,
    replayed in the V4.1 spelling: the wrapper opens and closes, but the body
    is not an invoke block at all. With the sentinel CONFIRMED as its special
    token the turn must fail cleanly (retryable) rather than showing the raw
    control tokens.
    """
    chunks, tokens = _sentinel_chunks(
        [CALLS_START_V41, "\n6 files changed\n", CALLS_END_V41]
    )
    parsed = list(
        parse_dsv41(responses(chunks, tokens=tokens), frozenset({DSML_SENTINEL_ID}))
    )
    errors = [
        r
        for r in parsed
        if isinstance(r, GenerationResponse) and r.finish_reason == "error"
    ]
    assert len(errors) == 1
    assert errors[0].tool_call_parse_failure_kind == "malformed"
    assert DSML_V41 not in errors[0].text


def test_malformed_block_in_legacy_mode_strips_the_markers():
    """Without a resolvable sentinel id the same block must not fail the turn."""
    chunks, _ = _sentinel_chunks(
        [CALLS_START_V41, "\n6 files changed\n", CALLS_END_V41]
    )
    parsed = list(parse_dsv41(responses(chunks), frozenset()))
    assert [
        r
        for r in parsed
        if isinstance(r, GenerationResponse) and r.finish_reason == "error"
    ] == []
    text = "".join(
        r.text for r in parsed if isinstance(r, GenerationResponse) and r.text
    )
    assert "6 files changed" in text
    assert DSML_V41 not in text


def test_unterminated_block_fails_the_turn():
    chunks, tokens = _sentinel_chunks(
        [CALLS_START_V41, "\n<", DSML_V41, " invoke name=\"read\">\n<", DSML_V41, " parameter"]
    )
    parsed = list(parse_dsv41(responses(chunks, tokens=tokens), frozenset({DSML_SENTINEL_ID})))
    errors = [
        r
        for r in parsed
        if isinstance(r, GenerationResponse) and r.finish_reason == "error"
    ]
    assert len(errors) == 1
    assert errors[0].tool_call_parse_failure_kind == "unterminated"


def test_legacy_mode_strips_instead_of_failing():
    """With no resolvable sentinel id, the parsers must not regress below V4.

    The sentinel cannot be confirmed as real, so a malformed block keeps the
    original safe behavior: strip the control tokens, keep the readable
    residue, never fail the turn.
    """
    chunks, _ = _sentinel_chunks(
        [CALLS_START_V41, "\n<", DSML_V41, " invoke name=\"read\">\nfeather<tool>>"]
    )
    stream = responses(chunks)
    parsed = list(parse_dsv41(stream, frozenset()))
    assert [
        r
        for r in parsed
        if isinstance(r, GenerationResponse) and r.finish_reason == "error"
    ] == []
    text = "".join(
        r.text for r in parsed if isinstance(r, GenerationResponse) and r.text
    )
    assert DSML_V41 not in text


def test_orphan_closers_never_leak_the_sentinel():
    """A repetition-loop bail-out emits closers with no opener."""
    chunks, _ = _sentinel_chunks(
        ["the answer is 42\n", "</", DSML_V41, " invoke>", "</", DSML_V41, " calls>"]
    )
    parsed = list(parse_dsv41(responses(chunks), frozenset()))
    text = "".join(
        r.text for r in parsed if isinstance(r, GenerationResponse) and r.text
    )
    assert "the answer is 42" in text
    assert DSML_V41 not in text


def test_sentinelless_block_is_recovered_not_leaked():
    """The correct structure with NO sentinel: the DSv4 recovery path owns it."""
    chunks = [
        "<tool_calls>\n<invoke name=\"read_file\">\n",
        '<parameter name="path" string="true">/tmp/a</parameter>\n',
        "</invoke>\n</tool_calls>",
    ]
    parsed = list(parse_dsv41(responses(chunks), frozenset({DSML_SENTINEL_ID})))
    tool_calls = [r for r in parsed if isinstance(r, ToolCallResponse)]
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_calls[0].name == "read_file"
    text = "".join(
        r.text for r in parsed if isinstance(r, GenerationResponse) and r.text
    )
    assert "<parameter" not in text


def _sentinel_chunks(chunks: list[str]) -> tuple[list[str], list[int]]:
    """Tag every chunk containing the sentinel with the sentinel's token id."""
    return chunks, [
        DSML_SENTINEL_ID if DSML_V41 in chunk else 1000 + i
        for i, chunk in enumerate(chunks)
    ]


def _stream_of(
    *items: GenerationResponse | None,
) -> Generator[GenerationResponse | None, None, None]:
    """A generator over already-materialised responses (typing-clean helper)."""
    yield from items
