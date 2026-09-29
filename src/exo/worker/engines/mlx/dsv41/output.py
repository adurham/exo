"""DSv4.1 output parsing for exo's chunk plumbing.

Reuses exo's DSML pieces unchanged and adds only the dispatch that the generic
``apply_all_parsers`` cannot express.

WHY A LOCAL DISPATCH. ``model_output_parsers.apply_all_parsers`` picks the V4
parser with ``issubclass(model_type, DeepseekV4Model)``. DSv4.1's model class is
``mlx_lm.models.deepseek_v41.Model``, which is NOT a subclass (deliberately:
making it one would drag this engine's model under
``mlx_lm.models.deepseek_v4``'s isinstance checks all over
``auto_parallel``/``batch_generate``/``pp_speculation``). Rather than edit a
shared file for one call site, the engine builds the same pipeline here, in the
same order, from the same pieces -- with two V4.1 substitutions:

    parse_thinking_models -> parse_dsv41 (V4.1) -> count_reasoning_tokens
                          -> map_responses_to_chunks
                          ^
                          +-- same shape as ``parse_deepseek_v4`` (exo's
                              ``_parse_dsml_stream`` + orphan stripping +
                              sentinel-less recovery), but parameterized on the
                              V4.1 wrapper markers ``<|DSML| calls>`` /
                              ``</|DSML| calls>`` and the V4.1 body parser.

The sentinel-less recovery is kept for DSv4.1 on purpose: it keys on the
``<parameter name="…" string="true|false">`` signature and the bare
``|DSML|`` sentinel, and BOTH of those are spelled identically in the two
dialects, so the protection (and the clean-fail) is exactly as valid here. What
it must not do is run the V4 parser: that is what ``parse_deepseek_v4`` would
do, and its wrapper marker never matches a V4.1 block.
"""

from __future__ import annotations

from collections.abc import Generator, Iterator

from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import GenerationChunk
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
    ToolCallResponse,
)
from exo.worker.engines.mlx.dsv41.dsml import (
    CALLS_END_V41,
    CALLS_START_V41,
    parse_dsml_v41_body,
    resolve_dsml_v41_ids,
    strip_orphan_dsml_v41,
)
from exo.worker.runner.llm_inference.model_output_parsers import (
    _parse_dsml_stream,
    _recover_or_fail_sentinelless_tool_call,
    count_reasoning_tokens,
    map_responses_to_chunks,
    parse_thinking_models,
)


def parse_dsv41(
    responses: Generator[GenerationResponse | None],
    dsml_special_token_ids: frozenset[int] = frozenset(),
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    """Parse a DeepSeek-V4.1 DSML tool-call block out of the token stream.

    The V4.1 analogue of ``model_output_parsers.parse_deepseek_v4``: same
    skeleton, V4.1 wrapper markers and body parser. ``dsml_special_token_ids``
    gates real-vs-quoted detection exactly as it does for V4 -- a tool call is
    only recognized when the sentinel arrived as its dedicated vocab token
    (id 128825 here), so the model quoting ``<|DSML| calls>`` in prose is left
    as readable content rather than stripped or rerouted. An empty set keeps
    the text-only fallback for tokenizers that cannot resolve the id.
    """
    stream = _parse_dsml_stream(
        responses,
        CALLS_START_V41,
        CALLS_END_V41,
        parse_dsml_v41_body,
        dsml_special_token_ids,
    )
    stream = strip_orphan_dsml_v41(stream)
    return _recover_or_fail_sentinelless_tool_call(stream)


def dsv41_output_parser(
    responses: Generator[GenerationResponse | None],
    tokenizer: TokenizerWrapper,
    prompt: str,
    model_id: ModelId,
) -> Iterator[GenerationChunk | None]:
    """Full output pipeline for DSv4.1: thinking split, DSML tool calls, chunks.

    ``prompt`` is the rendered prompt; it decides the initial state of the
    thinking split (a prefill that ends on `` thinking`` starts inside reasoning).
    """
    generator: Generator[GenerationResponse | ToolCallResponse | None] = responses
    if tokenizer.has_thinking:
        generator = parse_thinking_models(
            generator,
            tokenizer.think_start,
            tokenizer.think_end,
            starts_in_thinking=_starts_in_thinking(prompt, tokenizer),
        )
    generator = parse_dsv41(generator, resolve_dsml_v41_ids(tokenizer))
    generator = count_reasoning_tokens(generator)
    return (map_responses_to_chunks(r, model_id) for r in generator)


def _starts_in_thinking(prompt: str, tokenizer: TokenizerWrapper) -> bool:
    """Mirror of ``detect_thinking_prompt_suffix`` for the DSv4.1 marker.

    Kept local rather than importing the helper so the two callers cannot drift
    apart silently: this is the same one-line predicate exo's generic path uses,
    and the tests pin both spellings.
    """
    think_token = tokenizer.think_start
    return think_token is not None and prompt.rstrip().endswith(think_token)
