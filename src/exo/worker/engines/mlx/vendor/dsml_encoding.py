import json
import re
from typing import Any

from mlx_lm.chat_templates import deepseek_v32

from exo.api.types import ToolCallItem

BOS_TOKEN: str = deepseek_v32.bos_token
EOS_TOKEN: str = deepseek_v32.eos_token
DSML_TOKEN: str = deepseek_v32.dsml_token
THINKING_START: str = deepseek_v32.thinking_start_token
THINKING_END: str = deepseek_v32.thinking_end_token
USER_TOKEN = "<\uff5cUser\uff5c>"
ASSISTANT_TOKEN = "<\uff5cAssistant\uff5c>"
TOOL_CALLS_START = f"<{DSML_TOKEN}function_calls>"
TOOL_CALLS_END = f"</{DSML_TOKEN}function_calls>"
_ORPHAN_THINK_END = ASSISTANT_TOKEN + THINKING_END
_FIXED_THINK_BLOCK = ASSISTANT_TOKEN + THINKING_START + "\n" + THINKING_END
_FUNCTION_RESULTS_CLOSE = "</function_results>"
_ORPHAN_TOOL_RESULT_SUFFIX = _FUNCTION_RESULTS_CLOSE + "\n\n" + THINKING_END
_EMPTY_THINK_BLOCKS = (
    THINKING_START + "\n\n" + THINKING_END,
    THINKING_START + "\n" + THINKING_END,
    THINKING_START + THINKING_END,
)


def encode_messages(
    messages: list[dict[str, Any]],
    thinking_mode: str = "thinking",
    context: list[dict[str, Any]] | None = None,
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    tools: Any = None,  # pyright: ignore[reportAny]
) -> str:
    # V3.2 (like V4) is `tool_conditional`: when tools are in play, prior-turn
    # reasoning_content must be retained so multi-step tool chains stay
    # coherent.
    effective_drop_thinking = drop_thinking
    if tools:
        effective_drop_thinking = False
    prompt: str = deepseek_v32.encode_messages(
        messages,
        thinking_mode=thinking_mode,
        context=context,
        drop_thinking=effective_drop_thinking,
        add_default_bos_token=add_default_bos_token,
        tools=tools,
    )
    prompt = prompt.replace(_ORPHAN_TOOL_RESULT_SUFFIX, _FUNCTION_RESULTS_CLOSE)
    prompt = prompt.replace(_ORPHAN_THINK_END, _FIXED_THINK_BLOCK)
    for empty in _EMPTY_THINK_BLOCKS:
        prompt = prompt.replace(empty, "")
    return prompt


_INVOKE_PATTERN = re.compile(
    rf"<{re.escape(DSML_TOKEN)}invoke\s+name=\"([^\"]+)\">"
    rf"(.*?)"
    rf"</{re.escape(DSML_TOKEN)}invoke>",
    re.DOTALL,
)

_PARAM_PATTERN = re.compile(
    rf"<{re.escape(DSML_TOKEN)}parameter\s+name=\"([^\"]+)\"\s+string=\"(true|false)\">"
    rf"(.*?)"
    rf"</{re.escape(DSML_TOKEN)}parameter>",
    re.DOTALL,
)


# ── DSML structural-tag garble repair ────────────────────────────────────────
#
# DSv4 occasionally emits a token-id divergence on a rigid DSML structural tag
# at temp=1.0 — the model samples a neighbor token where the target is
# ~deterministic, producing a doubled/typo'd tag name. Observed 2026-06-18:
# the closing ``</｜DSML｜invoke>`` tag was emitted as ``</｜DSML｜invinvoke>``
# (token ``inv`` split into ``in`` + ``v``, yielding ``invinvoke``). The block
# is otherwise structurally complete and the tool call is fully recoverable,
# but the strict ``_INVOKE_PATTERN`` regex requires exactly ``invoke`` to
# close, so it returns None and the whole turn fails.
#
# This is NOT MTP-specific — it reproduces under the plain main sampler
# (MTP-off). The root cause is the model emitting a known-garbled structural
# tag; the fix is parser-side: recognize the known garble shapes and repair
# them to the canonical tag before the strict regex runs.
#
# The repair is deliberately conservative: a garbled tag is only repaired when
# the canonical name is unambiguously recoverable. For closing tags
# (``</｜DSML｜<name>>``) we repair when ``<name>`` is NOT already a known
# structural name but ENDS with one — e.g. ``invinvoke`` ends with ``invoke``
# and is not itself a known name, so it repairs to ``invoke``. This can't
# misfire on prose because the ``｜DSML｜`` sentinel only appears in real
# tool-call blocks (the parser only runs on blocks whose sentinel was
# confirmed-real via special-token id).

_DSML_STRUCTURAL_NAMES = ("function_calls", "invoke", "parameter")


def _repair_dsml_tag_garbles(text: str) -> str:
    """Repair known DSML structural-tag token-garble shapes.

    Scans for malformed ``<｜DSML｜...>`` / ``</｜DSML｜...>`` tags whose name is
    a doubled/typo'd version of a known structural tag name (``invoke``,
    ``parameter``, ``function_calls``) and normalizes them to the canonical
    tag. Only repairs when the canonical name is recoverable as a suffix of
    the garbled name (and the garbled name is not itself a valid structural
    name) — this prevents misfiring on legitimate content.

    Returns the text with repaired tags. Tags that don't match a known garble
    shape are left untouched (the strict regex will then fail on them as
    before, preserving the existing fail-the-turn behavior for truly broken
    blocks).
    """
    # Match any DSML tag (open or close) with a word-like name. Group 1 is the
    # leading ``</?`` and group 2 is the tag name (everything between the
    # sentinel and the closing ``>`` or the first attribute).
    tag_pattern = re.compile(
        rf"(</?){re.escape(DSML_TOKEN)}(\w+)([^>]*>)"
    )

    def _repair_one(match: re.Match[str]) -> str:
        slash = match.group(1)
        name = match.group(2)
        rest = match.group(3)
        if name in _DSML_STRUCTURAL_NAMES:
            return match.group(0)
        # Try suffix-recovery: does the garbled name END with a known name?
        for canonical in _DSML_STRUCTURAL_NAMES:
            if name.endswith(canonical) and len(name) > len(canonical):
                return f"{slash}{DSML_TOKEN}{canonical}{rest}"
        # No known garble shape — leave it so the strict regex fails loudly.
        return match.group(0)

    return tag_pattern.sub(_repair_one, text)


# Matches a well-formed DSML control tag: <｜DSML｜name ...> or </｜DSML｜name>.
# The tag name must be word-like (\w+) and the only thing before the closing
# '>' is optional whitespace-led attributes. This deliberately does NOT match
# a stray ``<｜DSML｜`` glued to arbitrary prose (e.g. the model parroting
# ``<｜DSML｜_cli.py | 6 files changed…``) — that residue is readable text and
# is preserved; only its leaked sentinel is removed by _DSML_ORPHAN_PATTERN.
_DSML_TAG_PATTERN = re.compile(
    rf"</?{re.escape(DSML_TOKEN)}\w+(?:\s+[^>]*)?>"
)
# Matches an orphaned DSML sentinel left behind when a malformed block emitted
# ``｜DSML｜`` without forming a valid tag. The leading ``<`` / ``</`` is
# optional so a bare sentinel (no angle bracket) is stripped too — the
# invariant is that the ``｜DSML｜`` special token never survives.
_DSML_ORPHAN_PATTERN = re.compile(rf"(?:<\s*/?\s*)?{re.escape(DSML_TOKEN)}")


def strip_dsml_markers(text: str) -> str:
    """Strip DSML control-token markup from text.

    Safety net for the case where a tool-call block fails to parse (a
    generation-quality hiccup: the model opens a ``tool_calls`` wrapper but
    emits something other than a valid ``invoke`` body). Without this, the raw
    ``<｜DSML｜...>`` special tokens leak verbatim into user-visible content via
    the chat-completions ``content`` field.

    Removes well-formed ``<｜DSML｜name ...>`` tags first, then any
    orphaned/unclosed ``｜DSML｜`` sentinels. The invariant is that the
    ``｜DSML｜`` special token never survives into displayed text; residual
    prose around a malformed block is preserved rather than silently dropped.
    """
    text = _DSML_TAG_PATTERN.sub("", text)
    text = _DSML_ORPHAN_PATTERN.sub("", text)
    return text


def parse_dsml_output(text: str) -> list[ToolCallItem] | None:
    """Parse DSML function_calls block from model output text.

    Args:
        text: The text containing the DSML function_calls block
              (including the start/end markers).

    Returns:
        List of ToolCallItem, or None if parsing fails.
    """
    # Repair known DSML structural-tag token-garbles (e.g. ``</｜DSML｜invinvoke>``
    # → ``</｜DSML｜invoke>``) before the strict regex runs. See
    # ``_repair_dsml_tag_garbles`` for the root cause and observed shapes.
    text = _repair_dsml_tag_garbles(text)

    tool_calls: list[ToolCallItem] = []

    for invoke_match in _INVOKE_PATTERN.finditer(text):
        func_name = invoke_match.group(1)
        invoke_body = invoke_match.group(2)

        args: dict[str, Any] = {}
        for param_match in _PARAM_PATTERN.finditer(invoke_body):
            param_name = param_match.group(1)
            is_string = param_match.group(2) == "true"
            param_value = param_match.group(3)

            if is_string:
                args[param_name] = param_value
            else:
                try:
                    args[param_name] = json.loads(param_value)
                except (json.JSONDecodeError, ValueError):
                    args[param_name] = param_value

        tool_calls.append(
            ToolCallItem(
                name=func_name,
                arguments=json.dumps(args),
            )
        )

    return tool_calls if tool_calls else None
