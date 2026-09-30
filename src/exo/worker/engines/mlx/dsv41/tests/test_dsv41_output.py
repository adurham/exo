"""The DSv4.1 output pipeline: thinking split, tool calls, chunk mapping.

These tests run the REAL pipeline the engine uses (``dsv41_output_parser``)
over stream shapes the production path actually produces: a terminal
finish_reason on a separate empty chunk, and a bare ``None`` between every pair
of real responses (see ``test_dsml_e2e`` for why both matter).

Two behaviours here are easy to assume WRONG, so both are pinned explicitly
rather than left to a docstring:

* a REAL tool call is only recognised when the sentinel arrives as its vocab
  token (``DSML_SENTINEL_ID``); the SAME text without that token is quoted
  prose, and prose is never turned into a tool call;
* a malformed block on a CONFIRMED-real sentinel fails the turn
  (``finish_reason="error"``) instead of leaking control tokens as content,
  while the same block in legacy mode (no resolvable sentinel id) strips the
  markers and keeps the readable residue.
"""

from __future__ import annotations

import json
from collections.abc import Generator

from exo.shared.types.chunks import ErrorChunk, TokenChunk, ToolCallChunk
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
    ToolCallResponse,
)
from exo.worker.engines.mlx.dsv41.dsml import CALLS_END_V41, CALLS_START_V41, DSML_V41
from exo.worker.engines.mlx.dsv41.output import dsv41_output_parser, parse_dsv41
from exo.worker.engines.mlx.dsv41.tests.conftest import (
    DSML_SENTINEL_ID,
    THINK_END,
    THINK_END_ID,
    THINK_START,
    FakeTokenizer,
    model_id,
)
from exo.worker.runner.llm_inference.model_output_parsers import parse_thinking_models

MODEL = model_id()

#: Chunk text for the fake detokenizer (see conftest.DEFAULT_CHUNKS).
T_ONE = 2  # "hi"
T_TWO = 3  # "there"
T_THREE = 4  # " and"
T_FOUR = 5  # " bye"


def _block(name: str = "read", param: str = "path", value: str = "/tmp/a") -> str:
    return (
        CALLS_START_V41
        + "\n"
        + f'<{DSML_V41} invoke name="{name}">\n'
        + f'<{DSML_V41} parameter name="{param}" string="true">{value}</{DSML_V41} parameter>\n'
        + f"</{DSML_V41} invoke>\n"
        + CALLS_END_V41
    )


def _parse(
    texts: list[str], tokenizer: FakeTokenizer, *, tokens=None, prompt: str = ""
):
    """Run the pipeline over a stream built from text chunks (+ their token ids)."""
    return list(
        filter(
            None, dsv41_output_parser(_stream(texts, tokens), tokenizer, prompt, MODEL)
        )
    )


def _parse_ids(
    ids: list[int],
    tokenizer: FakeTokenizer,
    *,
    prompt: str = "",
    text_of=None,
    extra_text: dict[int, str] | None = None,
):
    """Run the pipeline over a stream of token ids, taking text from ``text_of``.

    ``text_of`` defaults to the detokenizer's table plus the thinking markers,
    i.e. the same id -> text mapping a real detokenizer would produce.
    ``extra_text`` adds per-test ids (e.g. a whole DSML block on one token).
    """
    from exo.worker.engines.mlx.dsv41.tests.conftest import (
        DEFAULT_CHUNKS,
        THINK_END_ID,
        THINK_START_ID,
    )

    table = dict(DEFAULT_CHUNKS)
    table[THINK_END_ID] = THINK_END
    table[THINK_START_ID] = THINK_START
    if text_of is not None:
        table.update(text_of)
    if extra_text is not None:
        table.update(extra_text)
    return _parse([table.get(i, "") for i in ids], tokenizer, tokens=ids, prompt=prompt)


def _stream(
    texts: list[str], tokens: list[int] | None = None
) -> Generator[GenerationResponse | None]:
    """Production-shaped stream: None between responses, terminal chunk separate."""
    ids = tokens if tokens is not None else list(range(200, 200 + len(texts)))
    for i, text in enumerate(texts):
        yield GenerationResponse(text=text, token=ids[i], usage=None)
        yield None
    # The terminal response carries usage -- that is where reasoning_tokens is
    # patched in (the mid-stream responses deliberately have none).
    yield GenerationResponse(
        text="", token=1, finish_reason="stop", usage=_terminal_usage()
    )


def _terminal_usage():
    from exo.api.types import Usage
    from exo.api.types.api import CompletionTokensDetails, PromptTokensDetails

    return Usage(
        prompt_tokens=3,
        completion_tokens=4,
        total_tokens=7,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=0),
        completion_tokens_details=CompletionTokensDetails(reasoning_tokens=0),
    )


def _text(chunks) -> str:
    return "".join(c.text for c in chunks if isinstance(c, TokenChunk))


def _reasoning(chunks) -> str:
    return "".join(
        c.text for c in chunks if isinstance(c, TokenChunk) and c.is_thinking
    )


def _content(chunks) -> str:
    return "".join(
        c.text for c in chunks if isinstance(c, TokenChunk) and not c.is_thinking
    )


# --------------------------------------------------------------- thinking split


def test_plain_answer_is_all_content():
    chunks = _parse_ids([T_ONE, T_TWO, T_THREE, T_FOUR], FakeTokenizer())
    assert [c for c in chunks if isinstance(c, ToolCallChunk)] == []
    assert _text(chunks) == "hithere and bye"
    assert _reasoning(chunks) == ""
    assert all(not c.is_thinking for c in chunks if isinstance(c, TokenChunk))


def test_prompt_inside_reasoning_routes_text_to_reasoning_content():
    """The prompt ends on the open marker, so the stream starts in reasoning."""
    chunks = _parse_ids(
        [T_ONE, T_TWO, THINK_END_ID, T_THREE],
        FakeTokenizer(),
        prompt="<|Assistant|>" + THINK_START,
    )
    assert _reasoning(chunks) == "hithere"
    assert _content(chunks) == " and"
    assert THINK_START not in _text(chunks)
    assert THINK_END not in _text(chunks)


def test_prompt_outside_reasoning_makes_the_same_text_content():
    """The same stream with a prompt that is NOT inside reasoning: all content.

    This is the ``starts_in_thinking`` contract, and it is why the engine
    passes the rendered prompt into the parser.
    """
    chunks = _parse_ids(
        [T_ONE, T_TWO, THINK_END_ID, T_THREE],
        FakeTokenizer(),
        prompt="<|Assistant|>",
    )
    assert _reasoning(chunks) == ""
    assert "hithere" in _content(chunks)
    assert THINK_END not in _text(chunks)


def test_reasoning_tokens_are_counted_into_usage():
    """The terminal chunk's usage carries reasoning_tokens (counted, not guessed)."""
    chunks = _parse_ids(
        [T_ONE, T_TWO, THINK_END_ID, T_THREE],
        FakeTokenizer(),
        prompt="<|Assistant|>" + THINK_START,
    )
    terminal = [c for c in chunks if c.usage is not None]
    assert terminal, "no chunk carried usage"
    usage = terminal[-1].usage
    assert usage is not None
    assert usage.completion_tokens_details.reasoning_tokens == 2


def test_tokenizer_without_markers_never_thinking_splits():
    """Without resolvable markers the whole stream is content; nothing is lost."""
    chunks = _parse_ids([T_ONE, T_TWO], FakeTokenizer(has_thinking=False))
    assert _text(chunks) == "hithere"
    assert _reasoning(chunks) == ""


# --------------------------------------------------------------- tool calls


def test_tool_call_becomes_a_tool_call_chunk_with_parsed_arguments():
    chunks = _parse([_block()], FakeTokenizer(), tokens=[DSML_SENTINEL_ID])
    calls = [c for c in chunks if isinstance(c, ToolCallChunk)]
    assert len(calls) == 1
    assert calls[0].finish_reason == "tool_calls"
    assert calls[0].tool_calls[0].name == "read"
    assert json.loads(calls[0].tool_calls[0].arguments) == {"path": "/tmp/a"}
    assert DSML_V41 not in _text(chunks)


def test_reasoning_then_tool_call_in_one_turn():
    """A stream that starts inside reasoning and ends in a real tool call.

    The block arrives as ONE token carrying the sentinel's vocab id, which is
    what makes it a genuine call (not quoted prose).
    """
    chunks = _parse_ids(
        [T_ONE, T_TWO, THINK_END_ID, DSML_SENTINEL_ID],
        FakeTokenizer(),
        prompt="<|Assistant|>" + THINK_START,
        extra_text={DSML_SENTINEL_ID: _block("ping", "host", "a")},
    )
    assert _reasoning(chunks) == "hithere"
    assert [c.tool_calls[0].name for c in chunks if isinstance(c, ToolCallChunk)] == [
        "ping"
    ]


def test_quoted_block_without_the_special_token_is_never_a_tool_call():
    """Same characters, ordinary token ids: prose, not a call (the V4 lesson)."""
    chunks = _parse([_block()], FakeTokenizer())
    assert [c for c in chunks if isinstance(c, ToolCallChunk)] == []
    assert [c for c in chunks if isinstance(c, ErrorChunk)] == []


# --------------------------------------------------------------- failures


def test_malformed_confirmed_block_fails_the_turn():
    """Sentinel confirmed but the body is not a call: clean, retryable failure."""
    block = CALLS_START_V41 + "\n6 files changed\n" + CALLS_END_V41
    chunks = _parse([block], FakeTokenizer(), tokens=[DSML_SENTINEL_ID])
    errors = [c for c in chunks if isinstance(c, ErrorChunk)]
    assert len(errors) == 1
    assert errors[0].tool_call_parse_failure_kind == "malformed"
    # The failure message must not itself carry the sentinel.
    assert DSML_V41 not in errors[0].error_message


def test_malformed_block_in_legacy_mode_strips_instead_of_failing():
    """No resolvable sentinel id: keep the residue, never fail the turn."""
    block = CALLS_START_V41 + "\n6 files changed\n" + CALLS_END_V41
    chunks = _parse([block], FakeTokenizer(vocab={"hello": 7}))
    assert [c for c in chunks if isinstance(c, ErrorChunk)] == []
    assert "6 files changed" in _text(chunks)
    assert DSML_V41 not in _text(chunks)


# --------------------------------------------------------------- raw parsers


def test_parse_thinking_models_stray_close_marker_is_swallowed():
    """A bare close marker with no opener never leaks (V4 lesson, V4.1 markers)."""
    parsed = list(
        parse_thinking_models(
            _stream(["hi", THINK_END, "there"]),
            THINK_START,
            THINK_END,
            starts_in_thinking=False,
        )
    )
    text = "".join(
        r.text for r in parsed if isinstance(r, GenerationResponse) and r.text
    )
    assert THINK_END not in text
    assert text == "hithere"


def test_tool_call_response_is_emitted_once_with_usage_attached():
    from exo.api.types import Usage
    from exo.api.types.api import CompletionTokensDetails, PromptTokensDetails

    usage = Usage(
        prompt_tokens=3,
        completion_tokens=1,
        total_tokens=4,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=0),
        completion_tokens_details=CompletionTokensDetails(reasoning_tokens=0),
    )
    parsed = list(
        parse_dsv41(
            _materialised(
                GenerationResponse(text=_block(), token=DSML_SENTINEL_ID, usage=None),
                GenerationResponse(text="", token=1, finish_reason="stop", usage=usage),
            ),
            frozenset({DSML_SENTINEL_ID}),
        )
    )
    calls = [r for r in parsed if isinstance(r, ToolCallResponse)]
    assert len(calls) == 1
    assert calls[0].usage == usage


def _materialised(*items: GenerationResponse | None):
    """A generator over already-built responses (typing-clean helper)."""

    def gen():
        yield from items

    return gen()
