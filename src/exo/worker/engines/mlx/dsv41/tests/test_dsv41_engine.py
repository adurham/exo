"""The DSv4.1 engine driven end to end against a mock model.

This is the "load/shard/dispatch wiring against a mock engine" half of the
stream's acceptance: no checkpoint, no GPU, no mlx-lm fork -- a scripted model
and a fake tokenizer drive the REAL engine, the REAL output pipeline (thinking
split + V4.1 DSML) and the REAL agreement bookkeeping, and the test asserts on
the chunks a client would see.

What this covers that a parser unit test cannot:

* the engine's step()/submit() lifecycle (agreement -> queue -> drain ->
  FinishedResponse) with a single rank;
* the fenced prefill + first-token path, then rounds to EOS / max_tokens /
  stop sequence;
* a reasoning turn, and a turn whose reasoning ends in a tool call;
* the refusals (images, prefix cache, logprobs) firing rather than silently
  degrading;
* the speculative round's accept/rollback contract (never emit an unverified
  draft), using the fork's real ``spec`` helpers.

NOTE on the prompt: the engine renders it with exo's vendored DeepSeek-V4
encoder (the model id contains ``deepseek-v4``), which ends the assistant
header with `` thinking`` when thinking is on -- i.e. the stream STARTS inside
reasoning, exactly like production. The scripts below start with reasoning
tokens for that reason.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import pytest

from exo.shared.types.chunks import ErrorChunk, TokenChunk, ToolCallChunk
from exo.shared.types.common import CommandId
from exo.shared.types.tasks import TaskId, TextGeneration
from exo.shared.types.text_generation import (
    Base64Image,
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runner_response import FinishedResponse
from exo.worker.engines.mlx.dsv41.dsml import CALLS_END_V41, CALLS_START_V41, DSML_V41
from exo.worker.engines.mlx.dsv41.engine import Dsv41Engine
from exo.worker.engines.mlx.dsv41.errors import Dsv41UnsupportedFeature
from exo.worker.engines.mlx.dsv41.load import Dsv41Loaded
from exo.worker.engines.mlx.dsv41.rounds import _one_round, _spec_policy
from exo.worker.engines.mlx.dsv41.tests.conftest import (
    DSML_SENTINEL_ID,
    MX_ON_CPU,
    THINK_END,
    THINK_START,
    FakeTokenizer,
    model_id,
)

MODEL = model_id()


def test_suite_never_takes_the_gpu():
    """The whole DSv4.1 suite must run without a Metal device.

    Enforced here, not just documented: it is the difference between a test run
    that is safe to launch on a serving node and one that steals the GPU and
    stalls behind the production job. The pin is installed by conftest.py before
    exo's engine modules are imported (a Metal stream created at import time
    would keep the process on the GPU no matter what a later call says).
    """
    import mlx.core as mx

    assert MX_ON_CPU is True, "MLX default device could not be pinned to the CPU"
    assert mx.default_device() == mx.cpu


#: Token ids the scripted model emits, with the text each detokenizes to.
THINK_TEXT = {30: THINK_START, 31: "why", 32: " because", 33: THINK_END}
ANSWER_TEXT = {34: "hi", 35: " there", 36: " bye"}
TOOL_CALL_BLOCK = (
    CALLS_START_V41
    + "\n"
    + f'<{DSML_V41} invoke name="read">\n'
    + f'<{DSML_V41} parameter name="path" string="true">/tmp/a</{DSML_V41} parameter>\n'
    + f"</{DSML_V41} invoke>\n"
    + CALLS_END_V41
)
TOOL_TEXT = {DSML_SENTINEL_ID: TOOL_CALL_BLOCK}

#: Script for a reasoning turn that ends in a real tool call.
TOOL_SCRIPT = [31, 33, DSML_SENTINEL_ID, 1]


def _sentinel_tokens(chunks: list[str]) -> list[int]:
    """Tag the chunk(s) carrying the sentinel with its real vocab id."""
    return [
        DSML_SENTINEL_ID if DSML_V41 in chunk else 1000 + i
        for i, chunk in enumerate(chunks)
    ]


class ScriptedModel:
    """A model that answers from a PRE-BUILT logits table, one row per position.

    Position semantics match the real body and are the reason the expectations
    below are what they are:

    * a logits forward with ``last_logit_only`` names ONE token -- the argmax
      for the position AFTER everything the caller has fed the cache, i.e.
      table index ``cache.offset - prompt_tokens``. So the prefill's LAST chunk
      names ``script[0]`` regardless of how many chunks the prompt was split
      into, and each greedy round names the next entry;
    * a verify forward (``argmax=True``) returns one token per row, the table
      entry after that row's own position, so a draft of the wrong length or
      content is caught by the accept/rollback comparison.
    """

    def __init__(self, script: list[int], *, prompt_tokens: int = 3) -> None:
        self.script = list(script)
        #: How many prompt tokens the fake tokenizer produces. Round-level tests
        #: (which call ``_one_round`` without a prefill) set this to 0.
        self.prompt_tokens = prompt_tokens
        #: The body's embedding/head modules; the draft head takes them as
        #: arguments, so the mock has to carry stand-ins.
        self.embed = object()
        self.head = object()
        self.calls: list[tuple[int, int]] = []  # (rows, cols) per forward
        self.argmax_calls: list[list[int]] = []
        self.logits: list[mx.array] = [_one_hot(t) for t in self.script]
        self.last_offset = 0
        self.args = _Args()

    def make_cache(self, bsz: int = 1, max_seq_len: int | None = None, **_: Any):
        return MockCache(max_seq_len or 0)

    def __call__(
        self,
        input_ids: mx.array,
        cache: MockCache,
        last_logit_only: bool = False,
        return_taps: bool = False,
        argmax: bool = False,
    ):
        del last_logit_only
        rows = int(input_ids.shape[0])
        cache.offset += rows
        self.last_offset = cache.offset
        self.calls.append((rows, rows))
        if argmax:
            ids = self._ids_for(input_ids, cache.offset)
            self.argmax_calls.append(ids)
            out = mx.array([ids], dtype=mx.int32)
            return (out, self._taps(rows)) if return_taps else out
        logits = self.logits[self._index(cache.offset)]
        return (logits, self._taps(1)) if return_taps else logits

    def _index(self, offset: int) -> int:
        """Table index for the token the position ``offset`` predicts."""
        return max(0, min(offset - self.prompt_tokens, len(self.script) - 1))

    def _ids_for(self, input_ids: mx.array, offset: int) -> list[int]:
        """One token per verify row: the table entry after that row's position.

        The verify feed STARTS at ``offset - rows`` (the engine hands it the
        anchor plus its drafts), so row ``r`` predicts the table entry at
        ``offset - rows + r + 1``. That is what makes a wrong draft visible to
        the comparison instead of silently mapping onto the table's last row.
        """
        first = offset - int(input_ids.shape[0])
        return [
            self.script[self._index(first + row + 1)]
            for row in range(int(input_ids.shape[0]))
        ]

    def _taps(self, rows: int) -> dict[int, mx.array]:
        return {0: mx.zeros((1, rows, 4), dtype=mx.float32)}


class _Args:
    dspark_target_layer_ids: tuple[int, ...] = (0,)
    max_seq_len: int = 4096


class MockCache:
    """Cache stand-in: the rounds only need ``offset`` (and ``layers`` for spec)."""

    def __init__(self, max_seq_len: int) -> None:
        self.max_seq_len = max_seq_len
        self.offset = 0
        self.layers: list[Any] = []


class MockHead:
    """DSpark head stand-in: drafts the table's next tokens, or lies on purpose."""

    def __init__(self, model: ScriptedModel, *, lie_at: int | None = None) -> None:
        self.model = model
        self.lie_at = lie_at
        self.appended: list[int] = []

    def make_cache(self, bsz: int = 1):
        return object()

    def append_ctx(self, taps: mx.array, dsc: Any) -> None:
        self.appended.append(int(taps.shape[1]))

    def draft(
        self, anchor: mx.array, embed: Any, head_lin: Any, dsc: Any, *, width: int
    ):
        del anchor, embed, head_lin, dsc
        start = self.model._index(self.model.last_offset) + 1
        ids = list(self.model.script[start : start + width])
        if self.lie_at is not None and self.lie_at < len(ids):
            ids[self.lie_at] = 9999  # a token the target will not confirm
        return mx.array([ids], dtype=mx.int32)


class _Sender:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def send(self, item: Any) -> None:
        self.events.append(item)


class _Receiver:
    def __init__(self) -> None:
        self.items: list[TaskId] = []

    def collect(self) -> list[TaskId]:
        out, self.items = self.items, []
        return out


@dataclass
class _EngineStub:
    """Just enough engine for the round helpers (they touch ``_draft_windows``)."""

    _draft_windows: dict[int, Any] = field(default_factory=dict)


def _engine(
    script: list[int],
    *,
    text_of: dict[int, str],
    head: Any | None = None,
    max_output_tokens: int | None = None,
    stop: str | None = None,
    images: list[Base64Image] | None = None,
    use_prefix_cache: bool = False,
    logprobs: bool = False,
    tools: list[dict[str, Any]] | None = None,
) -> tuple[Dsv41Engine, ScriptedModel, FakeTokenizer, list[Any]]:
    tokenizer = FakeTokenizer(text_of=text_of)
    model = ScriptedModel(script)
    loaded = Dsv41Loaded(
        model=model,
        tokenizer=tokenizer,
        args=model.args,
        model_path=Path("/nonexistent-checkpoint"),
        built_layers=list(range(40)),
        full_stack=True,
        rank=0,
        world=1,
        load_seconds=0.0,
        head=head,
    )
    sender = _Sender()
    engine = Dsv41Engine(
        loaded=loaded,
        model_id=MODEL,
        group=None,
        cancel_receiver=_Receiver(),  # type: ignore[arg-type]
        event_sender=sender,  # type: ignore[arg-type]
        device_rank=0,
        speculative=head is not None,
        prefill_chunk_size=2,
    )
    params = TextGenerationTaskParams(
        model=MODEL,
        input=[InputMessage(role="user", content=InputMessageContent("hello"))],
        max_output_tokens=max_output_tokens,
        temperature=0.0,
        stop=stop,
        tools=tools,
        images=images or [],
        use_prefix_cache=use_prefix_cache,
        logprobs=logprobs,
    )
    task = TextGeneration(
        task_id=TaskId("task-1"),
        command_id=CommandId("cmd-1"),
        task_params=params,
        instance_id=InstanceId("inst-1"),
    )
    engine.submit(task)
    return engine, model, tokenizer, sender.events


def _drain(engine: Dsv41Engine, *, limit: int = 500) -> list[Any]:
    """Drive step() until it is empty; returns the chunks that came out."""
    out: list[Any] = []
    finished = False
    for _ in range(limit):
        step = list(engine.step())
        if not step:
            break
        for _task_id, item in step:
            if isinstance(item, FinishedResponse):
                finished = True
            else:
                out.append(item)
    assert finished, "engine never reported FinishedResponse"
    return out


def _text_of(chunks: list[Any]) -> str:
    return "".join(c.text for c in chunks if isinstance(c, TokenChunk))


# --------------------------------------------------------------- plain answers


def test_plain_turn_streams_text_and_finishes():
    engine, model, _tokenizer, _events = _engine(
        [34, 35, 36, 1], text_of=ANSWER_TEXT, max_output_tokens=8
    )
    chunks = _drain(engine)

    assert _text_of(chunks) == "hi there bye"
    # The prefill was fenced into chunks of 2: (2 rows) then (1 row). The
    # prefill's last chunk names exactly ONE token (the model's
    # ``last_logit_only`` contract), and that token is script[0] -- the fence
    # does not skip a scripted token per chunk.
    prefill = [rows for rows, cols in model.calls if cols > 1]
    assert prefill == [2, 1]


def test_prefill_is_fenced_and_names_the_first_scripted_token():
    engine, model, _tokenizer, _events = _engine(
        [34, 35, 36, 1], text_of=ANSWER_TEXT, max_output_tokens=8
    )
    chunks = _drain(engine)
    # The first token emitted is script[0], not script[n_chunks - 1].
    assert _text_of(chunks)[:2] == "hi"
    assert model.argmax_calls == []  # no verifies without a head


def test_generation_stops_on_eos_before_max_tokens():
    engine, _model, _tokenizer, _events = _engine(
        [34, 1, 35, 36], text_of=ANSWER_TEXT, max_output_tokens=8
    )
    chunks = _drain(engine)
    assert _text_of(chunks) == "hi"
    assert any(getattr(c, "finish_reason", None) == "stop" for c in chunks)


def test_max_output_tokens_stops_the_turn_with_length():
    engine, _model, _tokenizer, _events = _engine(
        [34, 35, 36, 34, 35], text_of=ANSWER_TEXT, max_output_tokens=2
    )
    chunks = _drain(engine)
    # Two tokens from the first round, then the cap truncates the second round's
    # two drafted tokens down to zero: max_output_tokens is a hard cap.
    assert _text_of(chunks) == "hi there"
    assert any(getattr(c, "finish_reason", None) == "length" for c in chunks)


def test_max_output_tokens_caps_a_single_token_round():
    engine, _model, _tokenizer, _events = _engine(
        [34, 35, 36, 1], text_of=ANSWER_TEXT, max_output_tokens=1
    )
    chunks = _drain(engine)
    assert _text_of(chunks) == "hi"
    assert any(getattr(c, "finish_reason", None) == "length" for c in chunks)


def test_stop_sequence_truncates_and_stops_the_turn():
    """A stop string spanning chunks ends the turn; the text is trimmed back.

    PINNED BEHAVIOUR (measured, and worth knowing): the engine finds the stop
    string in the ACCUMULATED text but trims only the CURRENT chunk, so when
    the sequence spans two chunks the residue can include its first characters
    ("hi there" here rather than "hi "). The turn does stop, and the full stop
    string never reaches the client -- but the trim is approximate and that is
    the engine's code (engine.py), not this test's.
    """
    engine, _model, _tokenizer, _events = _engine(
        [34, 35, 36, 1], text_of=ANSWER_TEXT, max_output_tokens=8, stop="there by"
    )
    chunks = _drain(engine)
    assert _text_of(chunks) == "hi there"
    assert "there by" not in _text_of(chunks)
    assert any(getattr(c, "finish_reason", None) == "stop" for c in chunks)


def test_usage_is_attached_to_the_terminal_chunk_only():
    engine, _model, _tokenizer, _events = _engine(
        [34, 35, 1], text_of=ANSWER_TEXT, max_output_tokens=8
    )
    chunks = _drain(engine)
    with_usage = [
        c for c in chunks if isinstance(c, TokenChunk) and c.usage is not None
    ]
    assert len(with_usage) == 1
    usage = with_usage[0].usage
    assert usage is not None
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (
        3,
        2,
        5,
    )


# --------------------------------------------------------------- reasoning


def test_reasoning_is_routed_and_then_content_follows():
    """The prompt ends inside reasoning, so the stream starts there."""
    engine, _model, _tokenizer, _events = _engine(
        [31, 32, 33, 34, 35, 1],
        text_of={**THINK_TEXT, **ANSWER_TEXT},
        max_output_tokens=16,
    )
    chunks = _drain(engine)
    thinking = "".join(
        c.text for c in chunks if isinstance(c, TokenChunk) and c.is_thinking
    )
    content = "".join(
        c.text for c in chunks if isinstance(c, TokenChunk) and not c.is_thinking
    )
    assert thinking == "why because"
    assert content == "hi there"
    assert THINK_START not in content and THINK_END not in content
    terminal = [c for c in chunks if isinstance(c, TokenChunk) and c.usage is not None]
    assert terminal[-1].usage is not None
    assert terminal[-1].usage.completion_tokens_details.reasoning_tokens == 2


def test_stray_open_marker_inside_reasoning_never_leaks():
    """A repeated ``open marker`` while already thinking is swallowed, not shown."""
    engine, _model, _tokenizer, _events = _engine(
        [31, 30, 32, 33, 34, 1],
        text_of={**THINK_TEXT, **ANSWER_TEXT},
        max_output_tokens=16,
    )
    chunks = _drain(engine)
    shown = _text_of(chunks)
    assert THINK_START not in shown
    assert shown == "why becausehi"


# --------------------------------------------------------------- tool calls


def test_tool_call_turn_emits_one_tool_call_chunk_with_usage():
    engine, _model, _tokenizer, _events = _engine(
        [31, 33, DSML_SENTINEL_ID, 1],
        text_of={**THINK_TEXT, **TOOL_TEXT},
        max_output_tokens=16,
        tools=[{"type": "function", "function": {"name": "read", "parameters": {}}}],
    )
    chunks = _drain(engine)
    calls = [c for c in chunks if isinstance(c, ToolCallChunk)]
    assert len(calls) == 1
    assert calls[0].tool_calls[0].name == "read"
    assert json.loads(calls[0].tool_calls[0].arguments) == {"path": "/tmp/a"}
    assert calls[0].usage is not None
    assert _text_of(chunks).strip() == "why"
    assert DSML_V41 not in _text_of(chunks)


# --------------------------------------------------------------- refusals


def test_images_are_refused_loudly_when_no_vision_processor_is_attached():
    engine, _model, _tokenizer, events = _engine(
        [34, 1],
        text_of=ANSWER_TEXT,
        images=[Base64Image("iVBORw0KGgo=")],
    )
    with pytest.raises(Dsv41UnsupportedFeature, match="image"):
        _drain(engine)
    assert any(isinstance(e.chunk, ErrorChunk) for e in events)


def test_prefix_cache_request_is_refused():
    engine, _model, _tokenizer, _events = _engine(
        [34, 1], text_of=ANSWER_TEXT, use_prefix_cache=True
    )
    with pytest.raises(Dsv41UnsupportedFeature, match="prefix-cache"):
        _drain(engine)


def test_logprobs_request_is_refused():
    engine, _model, _tokenizer, _events = _engine(
        [34, 1], text_of=ANSWER_TEXT, logprobs=True
    )
    with pytest.raises(Dsv41UnsupportedFeature, match="logprobs"):
        _drain(engine)


# --------------------------------------------------------------- spec rounds


def test_speculative_round_commits_only_target_confirmed_tokens():
    """Full acceptance: the whole draft plus the target's bonus token.

    The draft window is pre-seeded so the round takes the SPECULATIVE path
    (a fresh engine's first round primes instead -- see the priming test).
    """
    model = ScriptedModel([34, 35, 36, 34, 35, 1], prompt_tokens=0)
    head = MockHead(model)
    cache = MockCache(64)
    engine = _EngineStub(_draft_windows={0: object()})

    committed, _ms, accepted, gamma = _one_round(
        engine,  # type: ignore[arg-type]
        model=model,
        cache=cache,
        token=model.script[0],
        head=head,
        policy=_spec_policy(3),
    )
    assert gamma == 3
    # The anchor row is fed first, so the first prediction is script[1]; the
    # draft of script[1:4] is fully confirmed and the round also commits the
    # target's own next token (script[4]).
    assert accepted == 3
    assert committed == model.script[1:5]
    # The cache lands on the committed position (rollback, not append): the
    # verify fed 1 + gamma rows and the round accepted all gamma drafts.
    assert cache.offset == 1 + gamma
    assert head.appended == [gamma + 1]  # n_acc + 1 taps fed to the draft


def test_speculative_round_rejects_a_wrong_draft_and_takes_the_target_token():
    model = ScriptedModel([34, 35, 36, 34, 35, 1], prompt_tokens=0)
    head = MockHead(model, lie_at=1)  # the second drafted token is wrong
    cache = MockCache(64)
    engine = _EngineStub(_draft_windows={0: object()})

    committed, _ms, accepted, gamma = _one_round(
        engine,  # type: ignore[arg-type]
        model=model,
        cache=cache,
        token=model.script[0],
        head=head,
        policy=_spec_policy(3),
    )
    assert gamma == 3
    # The lie at draft position 1 breaks acceptance after one token.
    assert accepted == 1
    assert committed == model.script[1:3]
    assert 9999 not in committed
    assert cache.offset == 2


def test_first_round_with_a_head_primes_the_draft_window():
    """Round 1 with a head steps plainly and keeps the taps for round 2."""
    model = ScriptedModel([34, 35, 36, 1], prompt_tokens=0)
    head = MockHead(model)
    engine = _EngineStub()

    committed, _ms, accepted, gamma = _one_round(
        engine,  # type: ignore[arg-type]
        model=model,
        cache=MockCache(64),
        token=model.script[0],
        head=head,
        policy=_spec_policy(3),
    )
    # The primed round commits the anchor's own next token, one row of taps is
    # fed to the draft window, and no draft was consulted yet.
    assert committed == [model.script[1]]
    assert (accepted, gamma) == (1, 1)
    assert engine._draft_windows  # primed for the next round
    assert head.appended == [1]


def test_greedy_round_without_a_head_is_one_forward_one_token():
    model = ScriptedModel([34, 35, 36, 1], prompt_tokens=0)
    committed, _ms, accepted, gamma = _one_round(
        _EngineStub(),  # type: ignore[arg-type]
        model=model,
        cache=MockCache(64),
        token=model.script[0],
        head=None,
        policy=None,
    )
    assert committed == [model.script[1]]
    assert (accepted, gamma) == (1, 1)


def _one_hot(token: int) -> mx.array:
    """A logits row whose argmax is ``token`` (otherwise flat).

    The row is sized to fit the token, because the script may contain real
    checkpoint ids (e.g. the DSML sentinel, 128825) and not just small ones.
    """
    row = [0.0] * (token + 1)
    row[token] = 1.0
    return mx.array([row])
