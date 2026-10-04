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
from typing import Any, cast

import mlx.core as mx
import pytest
from mlx_lm.models.deepseek_v41.cache import CapacityError as _ScCapacityError

from exo.shared.types.chunks import (
    ErrorChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.common import CommandId
from exo.shared.types.events import ChunkGenerated
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
from exo.worker.engines.mlx.dsv41.errors import (
    Dsv41InvalidRequest,
    reclassify_input_error,
)
from exo.worker.engines.mlx.dsv41.load import Dsv41Loaded
from exo.worker.engines.mlx.dsv41.rounds import (
    _cache_capacity,
    _one_round,
    _spec_policy,
)
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
        #: Cache offset the round started from (set to 1 when a round feeds the
        #: anchor as its first row). The draft must answer from here, and the
        #: verify feed answers from the rows it is handed.
        self.round_entry_offset = 0
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
        logprobs: int = 0,
    ):
        del last_logit_only
        rows, fed = int(input_ids.shape[0]), int(input_ids.shape[-1])
        cache.offset += fed
        self.last_offset = cache.offset
        self.calls.append((rows, fed))
        if argmax:
            ids = self._ids_for(input_ids, cache.offset)
            self.argmax_calls.append(ids)
            out = mx.array([ids], dtype=mx.int32)
            if logprobs:
                k = int(logprobs)
                lp = {
                    "selected": mx.zeros((1, fed)),
                    "top_ids": mx.array([[[t] * k for t in ids]], dtype=mx.int32),
                    "top_logprobs": mx.zeros((1, fed, k)),
                }
                return (out, self._taps(fed), lp) if return_taps else (out, lp)
            return (out, self._taps(fed)) if return_taps else out
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
        rows = int(input_ids.shape[-1])
        first = offset - rows
        return [
            self.script[self._index(first + row + 1)] for row in range(rows)
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
        # The real ``ModelCache`` grows on demand; the mock mirrors the surface
        # so the engine's ensure wiring is exercised (growth is a no-op here).
        self.capacity = max_seq_len

    def ensure_capacity(self, required_tokens: int) -> None:
        if int(required_tokens) > self.max_seq_len:
            raise _ScCapacityError(
                f"cache holds {self.max_seq_len} tokens; a request at "
                f"{required_tokens} exceeds it")


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
        # The draft is what the model predicts next from the current position:
        # the table entry after the offset the round was started from.
        start = self.model._index(self.model.round_entry_offset)
        ids = list(self.model.script[start : start + width])
        if self.lie_at is not None and self.lie_at < len(ids):
            ids[self.lie_at] = 9999  # a token the target will not confirm
        # Same shape as the real DSparkHead.draft: (tokens, confidence).
        return mx.array([ids], dtype=mx.int32), mx.ones((1, len(ids)))


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


def _submit(
    engine: Dsv41Engine,
    task_id: str,
    *,
    content: str,
    max_output_tokens: int | None = None,
    images: list[Base64Image] | None = None,
) -> None:
    """Queue one more request on an engine built by ``_engine``."""
    params = TextGenerationTaskParams(
        model=MODEL,
        input=[InputMessage(role="user", content=InputMessageContent(content))],
        max_output_tokens=max_output_tokens,
        temperature=0.0,
        images=images or [],
    )
    engine.submit(
        TextGeneration(
            task_id=TaskId(task_id),
            command_id=CommandId(f"cmd-{task_id}"),
            task_params=params,
            instance_id=InstanceId("inst-1"),
        )
    )


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
    content: str = "hello",
    device_rank: int = 0,
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
        device_rank=device_rank,
        speculative=head is not None,
        prefill_chunk_size=2,
    )
    params = TextGenerationTaskParams(
        model=MODEL,
        input=[InputMessage(role="user", content=InputMessageContent(content))],
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


def _drain_pairs(
    engine: Dsv41Engine, *, limit: int = 500
) -> list[tuple[TaskId, Any]]:
    """Drive step() until it is empty, KEEPING the task id of every response.

    ``_drain`` discards the ids because single-request tests never need them;
    the invalid-request tests do, to tell a failed request's terminal from the
    next request's stream.
    """
    out: list[tuple[TaskId, Any]] = []
    for _ in range(limit):
        step = list(engine.step())
        if not step:
            break
        out.extend(step)
    return out


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
    prefill, fed = [], 0
    for _rows, cols in model.calls:
        if fed >= model.prompt_tokens:
            break
        prefill.append(cols)
        fed += cols
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
    _drain(engine)  # the request fails alone; the runner keeps serving
    errs = [e.chunk for e in events if isinstance(e.chunk, ErrorChunk)]
    assert errs and "image" in (errs[0].error_message or "")


def test_prefix_cache_request_is_served():
    # Prefix reuse is served by the conversation sessions (rounds._refuse_unsupported
    # docstring), so a use_prefix_cache request must complete, not be refused.
    engine, _model, _tokenizer, _events = _engine(
        [34, 1], text_of=ANSWER_TEXT, use_prefix_cache=True, max_output_tokens=8
    )
    chunks = _drain(engine)
    assert _text_of(chunks) == "hi"
    assert any(getattr(c, "finish_reason", None) == "stop" for c in chunks)


def test_logprobs_request_is_served():
    """The exo dashboard sends logprobs=true + top_logprobs=5 on every chat
    request; refusing it crashed the runner (seen live). Each token chunk now
    carries its own log-prob and the top alternatives."""
    engine, _model, _tokenizer, _events = _engine(
        [34, 35, 36, 1], text_of=ANSWER_TEXT, logprobs=True, max_output_tokens=8
    )
    chunks = _drain(engine)
    assert _text_of(chunks) == "hi there bye"
    toks = [c for c in chunks if isinstance(c, TokenChunk) and c.text]
    assert toks and all(c.logprob is not None for c in toks)
    # one-hot rows: the greedy token holds almost all the mass
    assert all(c.logprob <= 0.0 for c in toks)


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

    model.round_entry_offset = 1  # the round feeds the anchor as its first row
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

    model.round_entry_offset = 1  # the round feeds the anchor as its first row
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


def test_tokens_stream_before_the_turn_finishes():
    """The first token chunk leaves step() before the remaining decode rounds
    run (the engine used to run every round before emitting anything)."""
    engine, model, _tokenizer, _events = _engine(
        [34, 35, 36, 34, 35, 36, 1], text_of=ANSWER_TEXT, max_output_tokens=8
    )
    first_chunk_calls = None
    for _ in range(50):
        out = list(engine.step())
        if any(isinstance(c, TokenChunk) and c.text for _t, c in out):
            first_chunk_calls = len(model.calls)
            break
    assert first_chunk_calls is not None
    _drain(engine)
    assert first_chunk_calls < len(model.calls), (first_chunk_calls, len(model.calls))


def test_speculative_round_reports_one_logprob_per_committed_token():
    model = ScriptedModel([34, 35, 36, 34, 35, 1], prompt_tokens=0)
    head = MockHead(model, lie_at=1)
    cache = MockCache(64)
    engine = _EngineStub(_draft_windows={0: object()})
    model.round_entry_offset = 1
    lps: list[object] = []
    committed, _ms, accepted, _gamma = _one_round(
        engine,  # type: ignore[arg-type]
        model=model, cache=cache, token=model.script[0], head=head,
        policy=_spec_policy(3), logprobs=2, lp_out=lps,
    )
    assert accepted == 1 and len(committed) == 2
    assert len(lps) == len(committed)
    assert [entry[1][0] for entry in lps] == committed  # type: ignore[index]


def test_a_refused_request_fails_alone_and_the_engine_keeps_serving():
    """A refusal used to escape step() and crash the runner (a ~2 min reload)."""
    engine, _model, _tokenizer, events = _engine(
        [34, 1], text_of=ANSWER_TEXT, max_output_tokens=100_000
    )
    chunks = _drain(engine)  # FinishedResponse, no exception
    assert not any(isinstance(c, TokenChunk) for c in chunks)
    assert any(isinstance(e.chunk, ErrorChunk) for e in events)


def test_cache_capacity_refusal_is_client_classifiable():
    """The prompt-exceeds-cache refusal must be classifiable by OpenAI-style
    clients: standard ``maximum context length is {N} tokens`` wording plus the
    canonical ``context_length_exceeded`` structured code, while keeping the
    exact same condition and capacity source (the fake model's max_seq_len)."""
    engine, _model, _tokenizer, events = _engine(
        [34, 1], text_of=ANSWER_TEXT, max_output_tokens=100_000
    )
    # Same capacity source as the refusal itself (``_cache_capacity``): the
    # fake checkpoint's ``max_seq_len``, since no ``max_kv_tokens`` is set.
    capacity = _cache_capacity(engine)
    assert capacity == 4096  # the fixture's _Args.max_seq_len
    _drain(engine)

    errs = [e.chunk for e in events if isinstance(e.chunk, ErrorChunk)]
    assert errs, "expected a context-length refusal"
    message = errs[0].error_message or ""
    # Standard OpenAI wording, with the real capacity, for clients that only
    # pattern-match the message.
    assert f"maximum context length is {capacity} tokens" in message
    # The prompt / max-output breakdown is preserved (semantics unchanged).
    assert "3 prompt tokens" in message
    assert "100000 max_output_tokens" in message
    # And the structured code, for clients that read ``error.code``.
    assert errs[0].error_code == "context_length_exceeded"


def test_no_max_tokens_fits_the_cache_instead_of_refusing():
    """The dashboard sends no max_tokens; the 32K default must not exceed the
    instance cache and get the request refused."""
    engine, _model, _tokenizer, events = _engine([34, 35, 36, 1], text_of=ANSWER_TEXT)
    chunks = _drain(engine)
    assert _text_of(chunks) == "hi there bye"
    assert not any(isinstance(e.chunk, ErrorChunk) for e in events)


# ------------------------------------------------- invalid request (no crash)


#: The literal image placeholder text, fullwidth bars U+FF5C, spelled by
#: CONCATENATION like conftest's markers (the toolchain eats a "<" that is
#: immediately followed by a letter). Production crashed on exactly this text
#: typed into a user message: the vendored encoder refuses it, and the refusal
#: used to escape step() and take the runner down with it.
IMAGE_TEXT = "<" + "\uff5cdeepseek_image\uff5c>"


def _errors_for(events: list[Any], command_id: str) -> list[ErrorChunk]:
    return [
        e.chunk
        for e in events
        if isinstance(e.chunk, ErrorChunk)
        and e.command_id == CommandId(command_id)
    ]


def test_image_placeholder_text_in_a_message_fails_the_request_alone():
    """A user message containing the literal placeholder token must fail THIS
    request (error chunk with the reason, engine._active cleared) instead of
    raising out of step() and crashing the runner."""
    from exo.worker.engines.mlx.dsv41.vision import IMAGE_PLACEHOLDER

    assert IMAGE_TEXT == IMAGE_PLACEHOLDER  # pin: this is the encoder's token
    engine, _model, _tokenizer, events = _engine(
        [34, 1], text_of=ANSWER_TEXT, content=f"please explain {IMAGE_TEXT} here"
    )
    pairs = _drain_pairs(engine)  # no exception

    assert engine._active is None
    assert not any(isinstance(item, TokenChunk) for _tid, item in pairs)
    assert [tid for tid, item in pairs if isinstance(item, FinishedResponse)] == [
        TaskId("task-1")
    ]
    errs = _errors_for(events, "cmd-1")
    assert errs and "image special token" in (errs[0].error_message or "")


def test_invalid_request_does_not_block_the_next_queued_request():
    """The bad request fails; the NEXT queued (valid) request still completes."""
    engine, _model, _tokenizer, events = _engine(
        [34, 1], text_of=ANSWER_TEXT, content=f"a {IMAGE_TEXT} b", max_output_tokens=8
    )
    _submit(engine, "task-2", content="hello", max_output_tokens=8)
    pairs = _drain_pairs(engine)

    finished = {tid for tid, item in pairs if isinstance(item, FinishedResponse)}
    assert finished == {TaskId("task-1"), TaskId("task-2")}
    good_text = "".join(
        item.text
        for tid, item in pairs
        if tid == TaskId("task-2") and isinstance(item, TokenChunk)
    )
    assert good_text == "hi"
    assert not any(
        tid == TaskId("task-1") and isinstance(item, TokenChunk)
        for tid, item in pairs
    ), "the failed request must not start streaming tokens"
    assert engine._active is None
    assert _errors_for(events, "cmd-task-2") == []
    assert _errors_for(events, "cmd-1")


def test_invalid_next_task_inside_the_stop_handler_does_not_crash_the_step():
    """After a turn finishes, step()'s handler starts the next queued task; that
    next task being invalid must fail it THERE (same step) rather than raising
    through the handler and crashing the loop."""
    engine, _model, _tokenizer, events = _engine(
        [34, 1], text_of=ANSWER_TEXT, max_output_tokens=8
    )
    _submit(engine, "task-2", content=f"x {IMAGE_TEXT}", max_output_tokens=8)

    steps: list[list[tuple[TaskId, Any]]] = []
    for _ in range(50):
        step = list(engine.step())
        if not step:
            break
        steps.append(step)

    task1_final = [
        i for i, step in enumerate(steps) if (TaskId("task-1"), FinishedResponse()) in step
    ]
    assert task1_final, "task-1 never finished"
    final_step = steps[task1_final[0]]
    # The bad task-2 was failed by the very step that closed task-1: the handler
    # called the activation helper, which skipped it without raising.
    assert (TaskId("task-2"), FinishedResponse()) in final_step
    assert engine._active is None
    assert _errors_for(events, "cmd-task-2")


def test_invalid_next_task_inside_the_refusal_handler_does_not_crash_the_step():
    """Same, from the OTHER handler: a refused request (max_tokens over cache)
    fails, and the invalid next task must be skipped in the same step."""
    engine, _model, _tokenizer, events = _engine(
        [34, 1], text_of=ANSWER_TEXT, max_output_tokens=100_000
    )
    _submit(engine, "task-2", content=f"y {IMAGE_TEXT}", max_output_tokens=8)

    steps: list[list[tuple[TaskId, Any]]] = []
    for _ in range(50):
        step = list(engine.step())
        if not step:
            break
        steps.append(step)

    task1_final = [
        i for i, step in enumerate(steps) if (TaskId("task-1"), FinishedResponse()) in step
    ]
    assert task1_final, "task-1 never reported its refusal"
    # The refusal's handler started the next request; task-2 is invalid, so it
    # must be failed in that SAME step instead of raising through the handler.
    assert (TaskId("task-2"), FinishedResponse()) in steps[task1_final[0]]
    assert engine._active is None
    assert _errors_for(events, "cmd-1")  # the refusal's reason
    assert _errors_for(events, "cmd-task-2")


def test_an_internal_render_error_still_crashes_loudly(monkeypatch: Any):
    """A render failure that is NOT request input is an engine bug: it must
    still propagate out of step() (so the supervisor re-creates the runner),
    with the client told why -- exactly the pre-fix behaviour."""
    engine, _model, _tokenizer, events = _engine([34, 1], text_of=ANSWER_TEXT)

    def broken(_params: Any) -> str:
        raise ValueError("internal render bookkeeping broke")

    monkeypatch.setattr(engine, "_render_prompt", broken)
    with pytest.raises(ValueError, match="internal render bookkeeping"):
        list(engine.step())
    assert any(isinstance(e.chunk, ErrorChunk) for e in events)


def test_reclassify_input_error_leaves_an_internal_value_error_alone():
    """The classifier is marker-based: unknown ValueErrors pass through, so the
    engine's internal checks can never be swallowed as 'invalid request'."""
    with pytest.raises(ValueError, match="resize solve overflowed"):
        try:
            raise ValueError("resize solve overflowed the token budget")
        except ValueError as e:
            reclassify_input_error(e)
            raise
    with pytest.raises(Dsv41InvalidRequest, match="image special token"):
        try:
            raise ValueError("Message content contains image special token 'x'.")
        except ValueError as e:
            reclassify_input_error(e)
            raise


def test_undecodable_image_payload_is_an_invalid_request():
    """An image payload the REQUEST sent that cannot be decoded (bad base64) is
    reclassified at the ``prompt_tokens_for_request`` seam -- the request fails,
    it does not crash the runner."""
    from exo.worker.engines.mlx.dsv41.vision import (
        Dsv41Vision,
        prompt_tokens_for_request,
    )

    class _Cfg:
        #: Flat ``vision_*`` bag, the shape ``_as_preprocess_config`` accepts.
        image_token_id = 129264
        vision_patch_size = 14
        vision_downsample_ratio = 2
        vision_max_n_token = 4096
        vision_min_pixels = 1
        vision_max_wh_ratio = None

    class _EncodedPrompt:
        def encode(self, _text: str) -> list[int]:
            return [10, 129264, 11]  # exactly one placeholder, as rendered

    vision = Dsv41Vision(
        tower=object(),
        cfg=_Cfg(),
        placeholder=IMAGE_TEXT,
        image_token_id=129264,
        text_dim=8,
        n_vit_layers=1,
    )
    with pytest.raises(Dsv41InvalidRequest, match="base64"):
        prompt_tokens_for_request(vision, "x", [Base64Image("A")], _EncodedPrompt())


def test_image_list_without_matching_message_blocks_is_an_invalid_request():
    """The client sent an image but its messages contain no image block: the
    renderer refuses at ``render_prompt`` and the request fails alone while the
    next queued request completes."""
    from exo.worker.engines.mlx.dsv41.vision import IMAGE_PLACEHOLDER

    class _Vision:
        placeholder = IMAGE_PLACEHOLDER

    # task-1 carries an image but its message text has no image block; the tower
    # is attached before step() runs so the renderer actually takes the vision
    # path. task-2 (valid, queued behind) proves the failed skip.
    engine, _model, _tokenizer, events = _engine(
        [34, 1], text_of=ANSWER_TEXT, max_output_tokens=8,
        images=[Base64Image("A")],
    )
    engine.vision_processor = _Vision()  # type: ignore[assignment]
    _submit(engine, "task-2", content="hello", max_output_tokens=8)

    pairs = _drain_pairs(engine)
    assert engine._active is None
    finished = {tid for tid, item in pairs if isinstance(item, FinishedResponse)}
    assert finished == {TaskId("task-1"), TaskId("task-2")}
    errs = _errors_for(events, "cmd-1")
    assert errs and "must agree" in (errs[0].error_message or "")
    good_text = "".join(
        item.text
        for tid, item in pairs
        if tid == TaskId("task-2") and isinstance(item, TokenChunk)
    )
    assert good_text == "hi"


def test_image_expansion_failure_inside_the_generator_fails_the_request():
    """The image expansion runs LAZILY inside the generator (``_generate`` ->
    ``prompt_tokens_for_request``); its request-input failure must be handled by
    step() exactly like an eager one -- error chunk, engine cleared, no crash.
    """
    from exo.worker.engines.mlx.dsv41.vision import (
        IMAGE_PLACEHOLDER,
        Dsv41Vision,
    )

    class _Cfg:
        #: Flat ``vision_*`` bag, the shape ``_as_preprocess_config`` accepts.
        image_token_id = 129264
        vision_patch_size = 14
        vision_downsample_ratio = 2
        vision_max_n_token = 4096
        vision_min_pixels = 1
        vision_max_wh_ratio = None

    tokenizer = FakeTokenizer(text_of=ANSWER_TEXT)
    model = ScriptedModel([34, 1])
    sender = _Sender()
    engine = Dsv41Engine(
        loaded=Dsv41Loaded(
            model=model,
            tokenizer=tokenizer,
            args=model.args,
            model_path=Path("/nonexistent-checkpoint"),
            built_layers=list(range(40)),
            full_stack=True,
            rank=0,
            world=1,
            load_seconds=0.0,
            head=None,
        ),
        model_id=MODEL,
        group=None,
        cancel_receiver=_Receiver(),  # type: ignore[arg-type]
        event_sender=sender,  # type: ignore[arg-type]
        device_rank=0,
        speculative=False,
        prefill_chunk_size=2,
    )
    engine.vision_processor = Dsv41Vision(
        tower=object(),
        cfg=_Cfg(),
        placeholder=IMAGE_PLACEHOLDER,
        image_token_id=129264,
        text_dim=8,
        n_vit_layers=1,
    )
    # An image rides the request, but the checkpoint-shaped tokenizer's encode
    # returns no placeholder id for the rendered prompt -- the image processor's
    # own count check ("Found N image tokens but got M images") fires.
    params = TextGenerationTaskParams(
        model=MODEL,
        input=[InputMessage(role="user", content=InputMessageContent("see this"))],
        max_output_tokens=8,
        temperature=0.0,
        images=[Base64Image("A")],
        chat_template_messages=[
            {"role": "user", "content": [{"type": "image", "url": "exo-image:0"}]}
        ],
    )
    engine.submit(
        TextGeneration(
            task_id=TaskId("task-1"),
            command_id=CommandId("cmd-1"),
            task_params=params,
            instance_id=InstanceId("inst-1"),
        )
    )
    chunks = _drain(engine)  # no exception
    assert engine._active is None
    assert not any(isinstance(c, TokenChunk) for c in chunks)
    errs = _errors_for(sender.events, "cmd-1")
    assert errs and "image tokens" in (errs[0].error_message or "")


# ------------------------------------------------- prefill progress / heartbeat


def test_long_prefill_emits_progress_chunks():
    """A prefill split into >=2 chunks reports progress to the client.

    The engine's prefill-progress hook (``_session_progress``) fires once per
    prefill chunk and, on rank 0, sends a ``PrefillProgressChunk``. That hook is
    wired through ``Dsv41Sessions`` -> ``Conversation`` -> mlx-lm's
    ``SessionCache``; before the wiring existed the hook was defined but never
    reached, so any prefill longer than the supervisor's 45 s hang-watchdog
    window emitted NO events and the healthy runner was SIGKILLed mid-prefill.

    ``FakeTokenizer.encode`` always yields 3 prompt tokens and the engine's
    prefill chunk size here is 2, so the fence splits into 2 chunks (2 + 1) and
    the hook runs twice.
    """
    engine, _model, _tokenizer, events = _engine(
        [34, 1], text_of=ANSWER_TEXT, max_output_tokens=8
    )
    _drain(engine)

    emitted = [e.chunk for e in cast("list[ChunkGenerated]", events)]
    progress = [c for c in emitted if isinstance(c, PrefillProgressChunk)]
    assert progress, "a multi-chunk prefill must emit at least one progress chunk"
    assert progress[-1].processed_tokens == 3  # 2 + 1 rows over two chunks
    assert progress[-1].total_tokens == 3
    assert progress[-1].total_tokens > 0


def test_nonzero_rank_prefill_heartbeats():
    """A non-zero rank heartbeats during prefill but emits no client chunks.

    Only rank 0 emits client-visible prefill chunks (the c>=2 dedup guard); the
    other ranks must instead call the runner-installed liveness heartbeat or the
    supervisor's hang watchdog SIGKILLs them while they are healthy and busy.
    """
    engine, _model, _tokenizer, events = _engine(
        [34, 1], text_of=ANSWER_TEXT, max_output_tokens=8, device_rank=1
    )
    hits: list[int] = []
    engine.heartbeat = lambda: hits.append(1)
    _drain(engine)

    assert len(hits) >= 1, "rank!=0 must heartbeat during prefill"
    emitted = [e.chunk for e in cast("list[ChunkGenerated]", events)]
    assert not any(isinstance(c, PrefillProgressChunk) for c in emitted), (
        "rank!=0 must not emit client-visible prefill progress chunks"
    )
