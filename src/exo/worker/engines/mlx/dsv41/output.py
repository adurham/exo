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
same order, from the same pieces:

    parse_thinking_models -> parse_deepseek_v4 -> count_reasoning_tokens
                          -> map_responses_to_chunks

so DSv4.1's chunks are byte-for-byte what the V4 path would have produced.
"""

from __future__ import annotations

from collections.abc import Generator, Iterator

from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import GenerationChunk
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.runner.llm_inference.model_output_parsers import (
    _resolve_dsml_special_token_ids,
    count_reasoning_tokens,
    map_responses_to_chunks,
    parse_deepseek_v4,
    parse_thinking_models,
)


def dsv41_output_parser(
    responses: Generator[GenerationResponse | None],
    tokenizer: TokenizerWrapper,
    prompt: str,
    model_id: ModelId,
) -> Iterator[GenerationChunk | None]:
    """Full output pipeline for DSv4.1: thinking split, DSML tool calls, chunks.

    ``prompt`` is the rendered prompt; it decides the initial state of the
    thinking split (a prefill that ends on ``<think>`` starts inside reasoning).
    """
    generator: Generator[GenerationResponse | None] = responses
    if tokenizer.has_thinking:
        generator = parse_thinking_models(
            generator,
            tokenizer.think_start,
            tokenizer.think_end,
            starts_in_thinking=_starts_in_thinking(prompt, tokenizer),
        )
    generator = parse_deepseek_v4(generator, _resolve_dsml_special_token_ids(tokenizer))
    generator = count_reasoning_tokens(generator)
    return map(lambda r: map_responses_to_chunks(r, model_id), generator)


def _starts_in_thinking(prompt: str, tokenizer: TokenizerWrapper) -> bool:
    """Mirror of ``detect_thinking_prompt_suffix`` for the DSv4.1 marker.

    Kept local rather than importing the helper so the two callers cannot drift
    apart silently: this is the same one-line predicate exo's generic path uses,
    and the tests pin both spellings.
    """
    think_token = tokenizer.think_start
    return think_token is not None and prompt.rstrip().endswith(think_token)
