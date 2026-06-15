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

from exo.api.types import ToolCallItem
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
    return _parse_dsml_stream(
        responses, start, end, parse_dsml_output, dsml_special_token_ids
    )


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

    def _try_parse_tool_call(
        text: str, response: GenerationResponse
    ) -> ToolCallResponse | GenerationResponse:
        parsed = parse_body(text)
        if parsed is not None:
            return ToolCallResponse(
                tool_calls=parsed, usage=response.usage, stats=response.stats
            )
        logger.warning(f"DSML tool call parsing failed for: {text}")
        # Parsing failed (malformed block — e.g. the model opened a tool_calls
        # wrapper but emitted no valid invoke body). Strip the DSML control
        # tokens before yielding the residue as content; otherwise the raw
        # ``<｜DSML｜...>`` special tokens leak verbatim to the user.
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
                yield (
                    _try_parse_tool_call(tool_call_text, response)
                    if tool_calls_end in tool_call_text
                    # Stream ended mid-block (no closing marker). Strip the DSML
                    # control tokens so the unterminated wrapper doesn't leak
                    # raw ``<｜DSML｜...>`` tokens into displayed content.
                    else response.model_copy(
                        update={"text": strip_dsml_markers(tool_call_text)}
                    )
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
                yield _try_parse_tool_call(response.text[dsml_start:], response)
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
                result = _try_parse_tool_call(tool_call_text, response)
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
                result = _try_parse_tool_call(tool_call_text, response)
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
                    update={"text": combined, "token": 0, "finish_reason": "error"}
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
                }
            )
            yield response

    if not accumulated_tool_calls:
        logger.warning("Tool calls should have all been emitted but were not")
