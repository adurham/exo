import re
from collections.abc import Callable, Generator, Iterator
from functools import cache
from typing import Any

from mlx_lm.models.deepseek_v4 import Model as DeepseekV4Model
from mlx_lm.models.deepseek_v32 import Model as DeepseekV32Model
from mlx_lm.models.gpt_oss import Model as GptOssModel
from mlx_lm.tokenizer_utils import TokenizerWrapper
from openai_harmony import (  # pyright: ignore[reportMissingTypeStubs]
    HarmonyEncodingName,
    HarmonyError,  # pyright: ignore[reportUnknownVariableType]
    Role,
    StreamableParser,
    load_harmony_encoding,
)

from exo.api.types import ToolCallItem, ToolCallParseFailureKind
from exo.shared.types.chunks import (
    ErrorChunk,
    GenerationChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.common import ModelId
from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse
from exo.worker.engines.mlx.types import Model
from exo.worker.engines.mlx.utils_mlx import (
    detect_thinking_prompt_suffix,
)
from exo.worker.engines.mlx.vendor.dsml_encoding import (
    parse_dsml_output,
    parse_sentinelless_tool_call,
    strip_dsml_markers,
)
from exo.worker.runner.bootstrap import logger
from exo.worker.runner.llm_inference.tool_parsers import ToolParser


@cache
def get_gpt_oss_encoding():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return encoding


def count_reasoning_tokens(
    responses: Generator[GenerationResponse | ToolCallResponse | None],
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    """Count tokens with is_thinking=True and patch the total into Usage on the final response."""
    reasoning_tokens = 0
    for response in responses:
        if response is None:
            yield None
            continue
        if isinstance(response, GenerationResponse) and response.is_thinking:
            reasoning_tokens += 1
        if response.usage is not None and reasoning_tokens > 0:
            response = response.model_copy(
                update={
                    "usage": response.usage.model_copy(
                        update={
                            "completion_tokens_details": response.usage.completion_tokens_details.model_copy(
                                update={"reasoning_tokens": reasoning_tokens}
                            )
                        }
                    )
                }
            )
        yield response


def apply_all_parsers(
    receiver: Generator[GenerationResponse | None],
    prompt: str,
    tool_parser: ToolParser | None,
    tokenizer: TokenizerWrapper,
    model_type: type[Model],
    model_id: ModelId,
    tools: list[dict[str, Any]] | None,
) -> Iterator[GenerationChunk | None]:
    generator = receiver

    normalized_id = model_id.normalize().lower()
    if issubclass(model_type, GptOssModel):
        generator = parse_gpt_oss(generator)
    elif issubclass(model_type, DeepseekV32Model) and "deepseek" in normalized_id:
        if tokenizer.has_thinking:
            generator = parse_thinking_models(
                generator,
                tokenizer.think_start,
                tokenizer.think_end,
                starts_in_thinking=detect_thinking_prompt_suffix(prompt, tokenizer),
            )
        generator = parse_deepseek_v32(generator)
    elif issubclass(model_type, DeepseekV4Model) and "deepseek-v4" in normalized_id:
        if tokenizer.has_thinking:
            generator = parse_thinking_models(
                generator,
                tokenizer.think_start,
                tokenizer.think_end,
                starts_in_thinking=detect_thinking_prompt_suffix(prompt, tokenizer),
            )
        generator = parse_deepseek_v4(
            generator, _resolve_dsml_special_token_ids(tokenizer)
        )
    else:
        if tokenizer.has_thinking:
            generator = parse_thinking_models(
                generator,
                tokenizer.think_start,
                tokenizer.think_end,
                starts_in_thinking=detect_thinking_prompt_suffix(prompt, tokenizer),
            )

        if tool_parser:
            generator = parse_tool_calls(generator, tool_parser, tools)

    generator = count_reasoning_tokens(generator)

    return map(lambda r: map_responses_to_chunks(r, model_id), generator)


def map_responses_to_chunks(
    response: GenerationResponse | ToolCallResponse | None, model_id: ModelId
) -> GenerationChunk | None:
    match response:
        case None:
            return None
        case GenerationResponse():
            if response.finish_reason == "error":
                return ErrorChunk(
                    error_message=response.text,
                    model=model_id,
                    tool_call_parse_failure_kind=(
                        response.tool_call_parse_failure_kind
                    ),
                )
            else:
                finish_reason = response.finish_reason
                assert finish_reason not in (
                    "error",
                    "tool_calls",
                    "function_call",
                )
                return TokenChunk(
                    model=model_id,
                    text=response.text,
                    token_id=response.token,
                    usage=response.usage,
                    finish_reason=finish_reason,
                    stats=response.stats,
                    logprob=response.logprob,
                    top_logprobs=response.top_logprobs,
                    is_thinking=response.is_thinking,
                )
        case ToolCallResponse():
            return ToolCallChunk(
                tool_calls=response.tool_calls,
                model=model_id,
                usage=response.usage,
                stats=response.stats,
            )


def parse_gpt_oss(
    responses: Generator[GenerationResponse | None],
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    encoding = get_gpt_oss_encoding()
    stream = StreamableParser(encoding, role=Role.ASSISTANT)
    current_tool_name: str | None = None
    tool_arg_parts: list[str] = []

    for response in responses:
        if response is None:
            yield None
            continue
        try:
            stream.process(response.token)
        except HarmonyError:
            logger.error("Encountered critical Harmony Error, returning early")
            return

        delta = stream.last_content_delta
        ch = stream.current_channel
        recipient = stream.current_recipient

        # Debug: log every token with state
        logger.debug(
            f"parse_gpt_oss token={response.token} text={response.text!r} "
            f"recipient={recipient!r} ch={ch!r} delta={delta!r} "
            f"state={stream.state} current_tool={current_tool_name!r}"
        )

        if recipient != current_tool_name:
            if current_tool_name is not None:
                prefix = "functions."
                if current_tool_name.startswith(prefix):
                    current_tool_name = current_tool_name[len(prefix) :]
                logger.info(
                    f"parse_gpt_oss yielding tool call: name={current_tool_name!r}"
                )
                yield ToolCallResponse(
                    tool_calls=[
                        ToolCallItem(
                            name=current_tool_name,
                            arguments="".join(tool_arg_parts).strip(),
                        )
                    ],
                    usage=response.usage,
                )
                tool_arg_parts = []
            current_tool_name = recipient

        # If inside a tool call, accumulate arguments
        if current_tool_name is not None:
            if delta:
                tool_arg_parts.append(delta)
            if response.finish_reason is not None:
                yield response.model_copy(update={"text": "".join(tool_arg_parts)})
                tool_arg_parts = []
            continue

        if delta:
            yield response.model_copy(
                update={"text": delta, "is_thinking": ch == "analysis"}
            )

        if response.finish_reason is not None:
            yield response


def parse_deepseek_v32(
    responses: Generator[GenerationResponse | None],
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    """Parse DeepSeek V3.2 DSML tool calls from the generation stream.

    Uses accumulated-text matching (not per-token marker checks) because
    DSML markers like <｜DSML｜function_calls> may span multiple tokens.
    Thinking tag handling is delegated to parse_thinking_models, which
    wraps this parser in apply_all_parsers.
    """
    from exo.worker.engines.mlx.vendor.dsml_encoding import (
        TOOL_CALLS_END,
        TOOL_CALLS_START,
        parse_dsml_output,
    )

    return _parse_dsml_stream(
        responses, TOOL_CALLS_START, TOOL_CALLS_END, parse_dsml_output
    )


@cache
def _resolve_dsml_special_token_ids(tokenizer: TokenizerWrapper) -> frozenset[int]:
    """Resolve the vocab ids of DSv4's DSML control tokens.

    A REAL tool call emits the ``｜DSML｜`` sentinel as a dedicated special
    vocab token. When the model merely QUOTES the marker inside reasoning /
    content prose (e.g. explaining ``<｜DSML｜tool_calls>`` in an answer), those
    same characters are produced as ordinary BPE text tokens whose ids differ
    from the special-token id. Matching on the special-token *id* — not the
    decoded string — is therefore the only reliable way to tell a genuine
    tool-call block from prose that happens to contain the marker text. (This
    is the same lesson as the fused-think-delimiter fix: detect by token, not
    by substring.)

    Returns the set of valid single-token ids among the candidate DSML
    markers. Empty set when the tokenizer can't resolve any (older/stub
    tokenizers) — callers fall back to text-only detection so behavior never
    regresses below today's.
    """
    dsml_token = "｜DSML｜"
    candidates = (
        dsml_token,
        f"<{dsml_token}tool_calls>",
        f"</{dsml_token}tool_calls>",
        f"<{dsml_token}tool_call>",
    )
    ids: set[int] = set()
    hf = getattr(tokenizer, "_tokenizer", tokenizer)
    vocab: dict[str, int] = {}
    try:
        vocab = hf.get_vocab()
    except Exception:  # pragma: no cover - defensive
        vocab = {}
    for cand in candidates:
        tid: int | None = None
        # Only accept a candidate that is a SINGLE dedicated vocab token —
        # i.e. present in the vocab as one entry. A multi-token BPE split of
        # the literal characters (what prose produces) must NOT qualify.
        if cand in vocab:
            tid = vocab[cand]
        else:
            try:
                resolved = hf.convert_tokens_to_ids(cand)
            except Exception:  # pragma: no cover - defensive
                resolved = None
            unk = getattr(hf, "unk_token_id", None)
            if isinstance(resolved, int) and resolved >= 0 and resolved != unk:
                tid = resolved
        if tid is not None:
            ids.add(tid)
    return frozenset(ids)


def parse_deepseek_v4(
    responses: Generator[GenerationResponse | None],
    dsml_special_token_ids: frozenset[int] = frozenset(),
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    # DSv4-Flash wraps its tool calls in <｜DSML｜tool_calls> ... </｜DSML｜tool_calls>
    # (verified empirically from raw model output — distinct from V3.2 which uses
    # the function_calls wrapper). The inner body uses <｜DSML｜invoke name="..."> /
    # <｜DSML｜parameter ...> tags, which parse_dsml_output already handles. Only the
    # wrapper marker differs, so define it explicitly here rather than importing the
    # V3.2 function_calls constants.
    #
    # ``dsml_special_token_ids`` gates real-vs-quoted detection: a tool-call
    # block is only recognized when the DSML sentinel arrives as its special
    # vocab token, so the model quoting the marker in prose no longer triggers
    # a false tool-call parse (which stripped markers and leaked the rest of
    # the reasoning into content). Empty set → text-only fallback (legacy).
    dsml_token = "｜DSML｜"
    start = f"<{dsml_token}tool_calls>"
    end = f"</{dsml_token}tool_calls>"
    stream = _parse_dsml_stream(
        responses, start, end, parse_dsml_output, dsml_special_token_ids
    )
    stream = _strip_orphan_dsml_from_content(stream)
    return _recover_or_fail_sentinelless_tool_call(stream)


# A tool-call block emitted in the WRONG dialect: the correct invoke/parameter
# STRUCTURE but with the ``｜DSML｜`` sentinel dropped from every tag, e.g.
# ``<tool_called><invoke name="memory"><parameter name="action" string="true">``.
# DSv4 occasionally does this instead of the real ``<｜DSML｜invoke …>`` form
# (observed live 2026-06-15: a memory() call leaked as raw tags into content).
# The DSML parser only recognizes sentinel-bearing tags, so this slips through
# and the raw tags leak as content.
#
# The DISTINCTIVE, prose-unlikely signature is the parameter tag
# ``<parameter name="…" string="true|false">`` — that exact attribute pairing
# does not occur in natural prose (verified against the session corpus: 0 false
# positives). The model wraps it in one of several openers, and which opener it
# emits VARIES (it's degenerate output):
#   * ``<invoke name="…">``                 (2026-06-15, msg 70581)
#   * ``<tool_calls>`` / ``<tool_call>`` wrapper with NO invoke tag
#     (2026-06-16, msg 70765 — the model skipped the invoke line entirely;
#      the original both-tags-required gate MISSED this and it leaked)
#   * ``<tool_called>`` wrapper
# So we trigger on the parameter signature PLUS any recognized sentinel-less
# opener. The parameter tag alone is the false-positive guard; the opener set
# is permissive because the opener is exactly what the model garbles.
_SENTINELLESS_PARAM = re.compile(
    r"<parameter\s+name=\"[^\"]+\"\s+string=\"(?:true|false)\"\s*>"
)
# The pure Claude/minimax dialect omits the ``string=`` annotation entirely:
# ``<parameter name="x">value</parameter>``. On its own this attribute-less tag
# is too prose-likely to be a safe trigger, so it is NEVER the false-positive
# guard by itself — it only counts when it appears INSIDE a confirmed
# ``<invoke name="…">…</invoke>`` block (see _is_sentinelless_tool_call). That
# invoke wrapper is the prose-unlikely signature for this shape.
_SENTINELLESS_PARAM_PLAIN = re.compile(
    r"<parameter\s+name=\"[^\"]+\"\s*>.*?</parameter>",
    re.DOTALL,
)
_SENTINELLESS_INVOKE_BLOCK = re.compile(
    r"<invoke\s+name=\"[^\"]+\">.*?</invoke>",
    re.DOTALL,
)
# Any sentinel-less tool-call OPENER: an invoke tag, or a bare tool-call
# wrapper (``<tool_call>``, ``<tool_calls>``, ``<tool_called>``) — with or
# without a closing ``>`` (the model sometimes omits it, as in msg 70765).
_SENTINELLESS_OPENER = re.compile(
    r"<(?:invoke\s+name=\"[^\"]+\"\s*>|tool_calls?\b|tool_called\b)"
)


def _is_sentinelless_tool_call(text: str) -> bool:
    """True when text contains a sentinel-less tool-call block (a tool-call
    opener + a ``<parameter …>`` tag, no ｜DSML｜ sentinel).

    Two shapes are recognized, both with the ｜DSML｜ sentinel ABSENT:
      1. Any opener (``<invoke>``, ``<tool_call(s)>``, ``<tool_called>``) plus a
         typed ``<parameter … string="true|false">`` tag. The typed parameter is
         the prose-unlikely false-positive guard (msg 70581 / 70765).
      2. A complete ``<invoke name="…">…</invoke>`` block containing a plain
         attribute-less ``<parameter name="…">…</parameter>`` — the pure
         Claude/minimax dialect (msg 95278, 2026-06-29). Here the invoke wrapper
         is the prose-unlikely guard, so the plain parameter never triggers on
         its own in free text.
    """
    if "｜DSML｜" in text:
        return False  # real/quoted DSML is handled elsewhere
    # Shape 1: typed-param signature + any opener.
    if _SENTINELLESS_PARAM.search(text) and _SENTINELLESS_OPENER.search(text):
        return True
    # Shape 2: a full invoke block that itself contains a plain parameter tag.
    for invoke_match in _SENTINELLESS_INVOKE_BLOCK.finditer(text):
        if _SENTINELLESS_PARAM_PLAIN.search(invoke_match.group(0)):
            return True
    return False


def _recover_or_fail_sentinelless_tool_call(
    stream: Generator[GenerationResponse | ToolCallResponse | None],
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    """Recover a tool call the model emitted WITHOUT the ｜DSML｜ sentinel; fail
    cleanly only if it can't be parsed.

    DSv4 sometimes emits the right tool-call structure in the wrong dialect —
    ``<tool_called>/<invoke name=…>/<parameter name=… [string=…]>`` with no
    ``｜DSML｜`` prefix on any tag. The DSML parser can't recognize these (by
    design — it gates on the special-token sentinel), so they otherwise leak as
    raw tags into displayed content AND the tool call is lost.

    We buffer NON-thinking content (reasoning passes straight through); once the
    buffered content confirms the sentinel-less signature we, at the terminal
    response, attempt ``parse_sentinelless_tool_call``:
      * parseable  -> emit a real ``ToolCallResponse`` so the tool runs (the raw
        tags are dropped, never shown as content);
      * unparseable -> clean-fail the turn (finish_reason="error" -> 500 ->
        hermes retries), the pre-recovery behavior for a truly corrupt block.
    If the signature never confirms, the buffered content is flushed verbatim so
    legitimate prose is never held back or dropped.
    """
    buffer: list[GenerationResponse] = []
    buffered_text = ""
    triggered = False

    for item in stream:
        # Reasoning tokens and non-text items are never tool calls — pass through,
        # but first flush any pending content buffer to preserve ordering.
        if not isinstance(item, GenerationResponse) or item.is_thinking or not item.text:
            if buffer and not triggered:
                yield from buffer
                buffer = []
                buffered_text = ""
            yield item
            continue

        buffer.append(item)
        buffered_text += item.text
        if not triggered and _is_sentinelless_tool_call(buffered_text):
            triggered = True

        # On the terminal response, decide: recover (parseable) -> emit a real
        # tool call; else clean-fail (confirmed signature but unparseable); else
        # flush (signature never confirmed).
        if item.finish_reason is not None:
            if triggered:
                # ── DEGEN PROBE: a sentinel-less / garbled tool-call leak is a
                # DEGENERATION SHAPE (the model garbling its own structure —
                # observed 2026-06-16 msg 70765: char-corruption `adad.durham`
                # + a hallucinated tool-result fused into a <parameter> value),
                # distinct from the repeat-token-loop the kill-switch sees and
                # INVISIBLE to it. Emit a probe record so the next run captures
                # BOTH failure shapes and we can test whether malformed-structure
                # failures cluster with the same conditions (BS>1 cache swap,
                # long context) as repeat-loops — i.e. share one root cause.
                # Lazy import + only on the (rare) trigger path = zero hot cost.
                try:
                    from exo.worker.engines.mlx.speculative.dsv4_mtp import (
                        _DEGEN_LAST_TRANSITION,
                        _DEGEN_PROBE_ENABLED,
                        _degen_probe_write,
                    )
                    if _DEGEN_PROBE_ENABLED:
                        import time as _t
                        # Most-recent swap across any active uid (parser layer
                        # has no uid; timestamp-correlate against bs_transition).
                        _swaps: list[dict[str, Any]] = [
                            s for s in _DEGEN_LAST_TRANSITION.values()
                            if "wall_ns" in s
                        ]
                        _last_swap: dict[str, Any] | None = (
                            max(_swaps, key=lambda s: int(s["wall_ns"]))
                            if _swaps else None
                        )
                        _ms = (
                            (_t.perf_counter_ns() - int(_last_swap["wall_ns"]))
                            / 1e6
                            if _last_swap else None
                        )
                        _degen_probe_write({
                            "event": "malformed_toolcall_cleanfail",
                            "shape": "sentinelless_toolcall",
                            "buffered_tail": buffered_text[-200:],
                            "ms_since_any_swap": _ms,
                            "last_swap": _last_swap,
                            "wall_ns": _t.perf_counter_ns(),
                        })
                except Exception:
                    pass
                # ── RECOVER-FIRST ───────────────────────────────────────────
                # DSv4 emitted the right tool-call STRUCTURE in the wrong
                # (sentinel-less) dialect. Rather than burn a turn on a retry
                # that just re-rolls the same degenerate draw, parse the bare
                # <invoke>/<parameter> block into a real tool call so the tool
                # actually runs. The raw tags are NEVER shown as content either
                # way (buffer is dropped). Only if the block can't be parsed do
                # we clean-fail (retryable 500) as before — preserving the old
                # behavior for a truly corrupt block.
                recovered = parse_sentinelless_tool_call(buffered_text)
                if recovered is not None:
                    logger.info(
                        "Recovered sentinel-less tool call "
                        "(%d call(s)) from wrong-dialect output: %s",
                        len(recovered),
                        [tc.name for tc in recovered],
                    )
                    yield ToolCallResponse(
                        tool_calls=recovered,
                        usage=item.usage,
                        stats=item.stats,
                    )
                else:
                    yield item.model_copy(
                        update={
                            "text": (
                                "DSv4 emitted a tool call without the ｜DSML｜ tool-call "
                                "markers (wrong dialect: bare <invoke>/<parameter> tags) "
                                "and the block could not be parsed for recovery. "
                                "Failing the turn so it can be retried."
                            ),
                            "finish_reason": "error",
                            "tool_call_parse_failure_kind": "sentinelless",
                        }
                    )
            else:
                yield from buffer
            buffer = []
            buffered_text = ""
            triggered = False
            continue

    # Stream ended with no terminal response while buffering.
    if buffer and not triggered:
        yield from buffer


def _strip_orphan_dsml_from_content(
    stream: Generator[GenerationResponse | ToolCallResponse | None],
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    """Final safety net: strip ORPHAN DSML control tokens from emitted content.

    The stream parser only routes text through a tool-call block when it sees a
    real ``<｜DSML｜tool_calls>`` opening wrapper. But DSv4 sometimes emits STRAY
    DSML *closing* tags with no matching open — most often when it falls into a
    repetition loop inside a ``<think>`` block and emits
    ``</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>`` to bail out (observed
    live 2026-06-15: a vision-summary turn looped, then dribbled orphan close
    tags into the reasoning display). Those orphans never form a parseable block,
    so they pass through verbatim and the raw ``｜DSML｜`` sentinel leaks into the
    user-visible reasoning/content stream.

    This stage enforces the parser's standing invariant — the ``｜DSML｜`` special
    token must NEVER survive into displayed text — by stripping orphan markers
    from any GenerationResponse whose text still contains the sentinel after
    tool-call parsing. It runs on already-classified output, so it cannot affect
    tool-call detection. ToolCallResponses pass through untouched. Tokens that
    are wholly the sentinel collapse to empty text and are dropped to avoid
    emitting blank chunks.
    """
    dsml_token = "｜DSML｜"
    for item in stream:
        if isinstance(item, GenerationResponse) and item.text and dsml_token in item.text:
            cleaned = strip_dsml_markers(item.text)
            if cleaned == item.text:
                yield item
            elif cleaned:
                yield item.model_copy(update={"text": cleaned})
            # else: text was entirely orphan markers — drop the empty chunk
        else:
            yield item


def _parse_dsml_stream(
    responses: Generator[GenerationResponse | None],
    tool_calls_start: str,
    tool_calls_end: str,
    parse_body: Callable[[str], list[ToolCallItem] | None],
    dsml_special_token_ids: frozenset[int] = frozenset(),
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    accumulated = ""
    in_tool_call = False
    # Whether the DSML sentinel arrived as its special vocab token in the
    # CURRENT detection window. A real tool call emits ``｜DSML｜`` as a
    # dedicated token id; the model quoting the marker in prose emits ordinary
    # text tokens. We only commit to tool-call parsing when this is True, so a
    # quoted marker in reasoning/content no longer triggers a false parse that
    # stripped the markers and leaked the rest of the turn into ``content``.
    # When ``dsml_special_token_ids`` is empty (tokenizer couldn't resolve the
    # ids, e.g. a stub) we fall back to the legacy text-only behavior so the
    # change never regresses below today's. Reset whenever the window resets.
    dsml_special_seen = False
    # Tokens buffered while we detect the start of a DSML block
    pending_buffer: list[GenerationResponse] = []
    # Text accumulated during a tool call block
    tool_call_text = ""
    # A parsed ToolCallResponse held back until the terminal (is_done) response
    # so usage/stats can be attached. Usage is only populated on the is_done
    # response (see generator.generate's ``if is_done:`` gate), but the DSML
    # close marker ``</｜DSML｜tool_calls>`` typically arrives a token or two
    # BEFORE finish_reason/EOS. Emitting the ToolCallResponse the instant the
    # block closes therefore loses the token usage (usage=None) — the downstream
    # chat-completions adapter then returns on the tool-call chunk before the
    # usage-bearing response is ever consumed, so the client sees a tool call
    # with no usage. Holding the tool call until the terminal response keeps the
    # usage attached. (When the close marker and finish_reason land on the SAME
    # response, the finish-branch below already has usage — no deferral needed.)
    pending_tool_call: ToolCallResponse | None = None

    def _tool_call_error(
        response: GenerationResponse,
        reason: str,
        *,
        failure_kind: ToolCallParseFailureKind,
    ) -> GenerationResponse:
        # Fail the generation cleanly with finish_reason="error" (->
        # ErrorChunk -> 500 on the OpenAI stream). The hermes client classifies
        # that as a retryable upstream error and re-runs the turn, giving the
        # model another draw. Nothing tool-call-shaped is ever shown to the user
        # as content, and the (broken) call is not silently dropped.
        # ``failure_kind`` is carried to the master so the failure is counted by
        # kind in exo_tool_call_parse_failures_total.
        return response.model_copy(
            update={
                "text": reason,
                "finish_reason": "error",
                "tool_call_parse_failure_kind": failure_kind,
            }
        )

    def _try_parse_tool_call(
        text: str, response: GenerationResponse, *, confirmed_real: bool
    ) -> ToolCallResponse | GenerationResponse:
        parsed = parse_body(text)
        if parsed is not None:
            return ToolCallResponse(
                tool_calls=parsed, usage=response.usage, stats=response.stats
            )
        logger.warning(f"DSML tool call parsing failed for: {text}")
        if confirmed_real:
            # A COMPLETE tool-call block whose DSML sentinel was CONFIRMED as
            # its special vocab token (token-id detection) but that still fails
            # to parse means the model garbled the block's interior — most often
            # the ``<｜DSML｜invoke name="...">`` opening tag (observed 2026-06-15:
            # DSv4 emitted ``feather_rpc_<tool>>`` in place of the invoke tag at
            # the model's recommended temperature=1.0, a rare stochastic
            # generation hiccup). The old behavior — strip the markers and yield
            # the residue as CONTENT — surfaced raw tool-call innards to the user
            # as a garbled message AND silently dropped the tool call. Instead,
            # fail the generation cleanly so the turn is retried (see
            # _tool_call_error).
            return _tool_call_error(
                response,
                "DSv4 emitted a malformed tool-call block that could not "
                "be parsed (the tool_calls wrapper was present but its "
                "invoke body was corrupt). Failing the turn so it can be "
                "retried.",
                failure_kind="malformed",
            )
        # Legacy fallback (no special-token id resolved, so real-vs-quoted can't
        # be told apart): preserve the original safe behavior — strip the DSML
        # control tokens and yield the readable residue as content, so a quoted
        # mention never becomes a spurious turn failure and raw ``｜DSML｜`` tokens
        # never leak to the user.
        return response.model_copy(update={"text": strip_dsml_markers(text)})

    for response in responses:
        if response is None:
            yield None
            continue

        # Token-level real-vs-quoted detection. The DSML sentinel is a special
        # vocab token only when the model actually invokes a tool; quoting the
        # marker text in prose produces ordinary BPE tokens. Track whether the
        # special token appeared in the current window.
        if dsml_special_token_ids and response.token in dsml_special_token_ids:
            dsml_special_seen = True
        # Legacy fallback: with no resolved ids, trust the text marker (today's
        # behavior). Otherwise a marker is only "real" once the special token
        # has been seen.
        dsml_is_real = (not dsml_special_token_ids) or dsml_special_seen
        # CONFIRMED real means we positively saw the special token (not the
        # legacy text-only fallback). Only then do we fail a malformed block as
        # an error; in legacy mode we keep the safe strip-to-content behavior.
        dsml_confirmed_real = bool(dsml_special_token_ids) and dsml_special_seen

        # Safety: if a tool call is held but the next response is NOT the
        # terminal one (some model emitted trailing content after the block),
        # flush the held call first to preserve ordering. Usage stays None in
        # that rare case — correctness of order beats the token count.
        if pending_tool_call is not None and response.finish_reason is None:
            yield pending_tool_call
            pending_tool_call = None

        if response.finish_reason is not None:
            yield from pending_buffer
            pending_buffer.clear()
            if pending_tool_call is not None:
                # Terminal response carries usage/stats (built only on is_done).
                # Attach them to the tool call we held back when its DSML block
                # closed a token or two earlier, then emit. This is what makes
                # the client see token usage on a tool-calling turn.
                yield pending_tool_call.model_copy(
                    update={"usage": response.usage, "stats": response.stats}
                )
                pending_tool_call = None
                break
            if in_tool_call:
                tool_call_text += response.text
                if tool_calls_end in tool_call_text:
                    yield _try_parse_tool_call(
                        tool_call_text, response, confirmed_real=dsml_confirmed_real
                    )
                elif dsml_confirmed_real:
                    # Stream ended mid-tool-call with NO closing marker, on a
                    # CONFIRMED-real block (special token seen). The model began
                    # a genuine tool call then stopped before closing it
                    # (observed 2026-06-15, finish_reason=stop: DSv4 emitted
                    # ``<command>\n<timeout>`` with the closing tags never
                    # written, after a failure-spiral / hung prior tool call).
                    # Stripping markers here LEAKS the bare command (+ trailing
                    # param value) as content AND drops the call — the exact
                    # symptom. Fail cleanly so the turn is retried instead.
                    yield _tool_call_error(
                        response,
                        "DSv4 began a tool-call block but the stream ended "
                        "before it was closed (unterminated invoke). Failing "
                        "the turn so it can be retried.",
                        failure_kind="unterminated",
                    )
                else:
                    # Legacy fallback (no special-token id): can't be sure it was
                    # a real call, so preserve the original behavior — strip the
                    # DSML control tokens so an unterminated wrapper doesn't leak
                    # raw ``<｜DSML｜...>`` tokens into displayed content.
                    yield response.model_copy(
                        update={"text": strip_dsml_markers(tool_call_text)}
                    )
            elif (
                tool_calls_start in response.text
                and tool_calls_end in response.text
                and dsml_is_real
            ):
                dsml_start = response.text.index(tool_calls_start)
                before = response.text[:dsml_start]
                if before:
                    yield response.model_copy(update={"text": before})
                yield _try_parse_tool_call(
                    response.text[dsml_start:], response,
                    confirmed_real=dsml_confirmed_real,
                )
            else:
                # No real tool-call block (or the marker is quoted prose):
                # emit verbatim as content. Quoted ``<｜DSML｜…>`` text is left
                # intact — it is the model's own readable output, not a leaked
                # control token, so it must NOT be stripped.
                yield response
            break

        if in_tool_call:
            tool_call_text += response.text
            if tool_calls_end in tool_call_text:
                result = _try_parse_tool_call(
                    tool_call_text, response, confirmed_real=dsml_confirmed_real
                )
                in_tool_call = False
                tool_call_text = ""
                if isinstance(result, ToolCallResponse):
                    # Hold until the terminal response so usage can be attached
                    # (this mid-stream response has usage=None).
                    pending_tool_call = result
                else:
                    # Parse failed → content residue; no usage to wait for.
                    yield result
            continue

        accumulated += response.text

        if tool_calls_start in accumulated:
            # The marker TEXT has assembled. Commit to tool-call parsing only
            # when it's a REAL block (special token seen, or legacy fallback).
            # If it's quoted prose (special token never arrived), emit the
            # buffered text + this response verbatim as content — do NOT strip
            # or reroute the model's own readable output.
            if not dsml_is_real:
                yield from pending_buffer
                pending_buffer.clear()
                accumulated = ""
                yield response
                continue
            start_idx = accumulated.index(tool_calls_start)
            pre_text = accumulated[:start_idx]
            # Flush pending buffer tokens that contributed text before the marker
            if pre_text:
                for buf_resp in pending_buffer:
                    if not pre_text:
                        break
                    chunk = buf_resp.text
                    if len(chunk) <= len(pre_text):
                        yield buf_resp
                        pre_text = pre_text[len(chunk) :]
                    else:
                        yield buf_resp.model_copy(update={"text": pre_text})
                        pre_text = ""
            pending_buffer = []
            tool_call_text = accumulated[start_idx:]
            accumulated = ""

            if tool_calls_end in tool_call_text:
                result = _try_parse_tool_call(
                    tool_call_text, response, confirmed_real=dsml_confirmed_real
                )
                tool_call_text = ""
                dsml_special_seen = False
                if isinstance(result, ToolCallResponse):
                    # Hold until the terminal response so usage can be attached.
                    pending_tool_call = result
                else:
                    yield result
            else:
                in_tool_call = True
            continue

        # Buffer on a partial marker match. This is intentionally NOT gated on
        # dsml_is_real: the special-token id and the marker's leading ``<`` can
        # arrive on the SAME or ADJACENT chunks, so we must keep accumulating
        # the candidate marker across chunks regardless. The real-vs-quoted
        # decision is made above, once the full marker text has assembled. A
        # quoted marker that never becomes "real" is flushed verbatim there.
        if _could_be_marker_prefix(accumulated, tool_calls_start):
            pending_buffer.append(response)
            continue

        # No partial match — flush all pending tokens and the current one
        yield from pending_buffer
        pending_buffer.clear()
        accumulated = ""
        yield response

    # Flush any remaining pending buffer at generator end
    yield from pending_buffer
    # If the stream ended without a terminal (finish_reason) response while a
    # tool call was held, emit it now so the call is never dropped. Usage stays
    # None in this degenerate case (no is_done response ever arrived).
    if pending_tool_call is not None:
        yield pending_tool_call


def _could_be_marker_prefix(text: str, marker: str) -> bool:
    max_check = len(marker)
    tail = text[-max_check:] if len(text) > max_check else text
    for i in range(len(tail)):
        suffix = tail[i:]
        if marker.startswith(suffix):
            return True
    return False


def parse_thinking_models(
    responses: Generator[GenerationResponse | None],
    think_start: str | None,
    think_end: str | None,
    starts_in_thinking: bool = True,
) -> Generator[GenerationResponse | None]:
    """Route thinking tokens via is_thinking flag.

    Swallows think tag tokens, sets is_thinking on all others.
    Always yields tokens with finish_reason to avoid hanging the chunk stream.
    """
    is_thinking = starts_in_thinking
    accumulated = ""
    pending_buffer: list[GenerationResponse] = []

    def drain_pending(_is_thinking: bool):
        for buffered in pending_buffer:
            yield buffered.model_copy(update={"is_thinking": _is_thinking})
        pending_buffer.clear()

    for response in responses:
        if response is None:
            yield None
            continue

        accumulated += response.text

        if response.finish_reason is not None:
            yield from drain_pending(is_thinking)
            yield response.model_copy(update={"is_thinking": False})
            continue

        # Fast path: the delimiter arrives as its own clean chunk (or the
        # accumulation is exactly the delimiter). Preserved verbatim so the
        # existing clean-token behavior and its regression tests are untouched.
        if accumulated == think_start and not is_thinking:
            is_thinking = True
            accumulated = ""
            pending_buffer.clear()
            continue
        if accumulated == think_end and is_thinking:
            is_thinking = False
            accumulated = ""
            pending_buffer.clear()
            continue

        # Fused / embedded delimiter handling.
        #
        # The mlx-lm streaming detokenizer yields `last_segment` deltas that can
        # carry MORE than one token's text, so a think delimiter may arrive
        # glued to neighbouring text in a single chunk (e.g. "code.</think>def")
        # or the accumulation may overshoot the exact match when the delimiter
        # spans chunks (e.g. "</" then "think>def" -> "</think>def"). The
        # exact-equality checks above miss both cases, which previously leaked
        # the delimiter + the post-delimiter answer into the wrong stream
        # (reasoning text + code dumped into `content`). Detect the currently
        # active delimiter as a SUBSTRING and split around it. Loop so multiple
        # delimiters fused into one chunk are all handled.
        active = think_end if is_thinking else think_start
        if active and active in accumulated:
            # Any prefix-buffered tokens are mirrored in `accumulated` (both are
            # appended in lockstep), so their text is already represented in the
            # split below. Discard the parallel copy rather than draining it, to
            # avoid double-emitting delimiter fragments that were buffered as a
            # clean prefix (e.g. "</" then "think>answer").
            pending_buffer.clear()
            while True:
                active = think_end if is_thinking else think_start
                if not active or active not in accumulated:
                    break
                idx = accumulated.index(active)
                pre = accumulated[:idx]
                if pre:
                    yield response.model_copy(
                        update={"text": pre, "is_thinking": is_thinking}
                    )
                # Swallow the delimiter and flip the thinking state.
                is_thinking = not is_thinking
                accumulated = accumulated[idx + len(active):]
            # Handle the remainder after the last delimiter. If it is a clean
            # (incomplete) prefix of the next possible delimiter, keep it
            # buffered and wait for more; otherwise emit it with the new flag.
            if accumulated:
                nxt = think_end if is_thinking else think_start
                is_clean_prefix = bool(
                    nxt
                    and len(accumulated) < len(nxt)
                    and accumulated == nxt[: len(accumulated)]
                )
                if is_clean_prefix:
                    pending_buffer.append(
                        response.model_copy(update={"text": accumulated})
                    )
                else:
                    yield response.model_copy(
                        update={"text": accumulated, "is_thinking": is_thinking}
                    )
                    accumulated = ""
            continue

        if (think_start and accumulated == think_start[: len(accumulated)]) or (
            think_end and accumulated == think_end[: len(accumulated)]
        ):
            pending_buffer.append(response)
            continue

        accumulated = ""

        yield from drain_pending(is_thinking)
        yield response.model_copy(update={"is_thinking": is_thinking})


def parse_tool_calls(
    responses: Generator[GenerationResponse | None],
    tool_parser: ToolParser,
    tools: list[dict[str, Any]] | None,
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    in_tool_call = False
    tool_call_text_parts: list[str] = []
    accumulated_tool_calls: list[ToolCallItem] = []

    for response in responses:
        if response is None:
            yield None
            continue

        if not in_tool_call and response.text.startswith(tool_parser.start_parsing):
            in_tool_call = True

        if (
            not in_tool_call
            and accumulated_tool_calls
            and (response.stats is not None or response.finish_reason is not None)
        ):
            yield ToolCallResponse(
                tool_calls=accumulated_tool_calls,
                usage=response.usage,
                stats=response.stats,
            )
            accumulated_tool_calls.clear()
            continue

        if not in_tool_call:
            yield response
            continue

        tool_call_text_parts.append(response.text)
        if response.text.endswith(tool_parser.end_parsing):
            # parse the actual tool calls from the tool call text
            combined = "".join(tool_call_text_parts)
            parsed = tool_parser.parse(combined.strip(), tools=tools)
            logger.info(f"parsed {tool_call_text_parts=} into {parsed=}")
            in_tool_call = False
            tool_call_text_parts = []

            if parsed is None:
                logger.warning(f"tool call parsing failed for text {combined}")
                yield response.model_copy(
                    update={
                        "text": combined,
                        "token": 0,
                        "finish_reason": "error",
                        "tool_call_parse_failure_kind": "malformed",
                    }
                )
                break

            accumulated_tool_calls.extend(parsed)
            if accumulated_tool_calls and (
                response.finish_reason is not None or response.stats is not None
            ):
                yield ToolCallResponse(
                    tool_calls=accumulated_tool_calls,
                    usage=response.usage,
                    stats=response.stats,
                )
                accumulated_tool_calls.clear()
            continue

        if response.finish_reason is not None:
            logger.info(
                "tool call parsing interrupted, yield partial tool call as text"
            )
            response = response.model_copy(
                update={
                    "text": "".join(tool_call_text_parts),
                    "token": 0,
                    "finish_reason": "error",
                    "tool_call_parse_failure_kind": "unterminated",
                }
            )
            yield response

    if not accumulated_tool_calls:
        logger.warning("Tool calls should have all been emitted but were not")
