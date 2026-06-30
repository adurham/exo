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


def _within_edit_distance_one(a: str, b: str) -> bool:
    """True iff ``a`` is reachable from ``b`` by at most one single-character
    edit (substitution, insertion, or deletion).

    Used to recover a canonical DSML structural tag name from a garble where
    the model sampled ONE neighbor token in the rigid tag region — e.g.
    ``invode`` (substitution k→d of ``invoke``) or ``paramter`` (deletion of
    ``parameter``). Bounded/early-exit; never builds the full Levenshtein
    matrix. Equal strings are distance 0 (also "within one").
    """
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        # Same length → only a substitution can connect them; count mismatches.
        mismatches = sum(1 for ca, cb in zip(a, b, strict=True) if ca != cb)
        return mismatches == 1
    # Lengths differ by exactly one → check single insertion/deletion. Walk the
    # shorter against the longer, allowing exactly one skip in the longer.
    shorter, longer = (a, b) if la < lb else (b, a)
    i = j = 0
    skipped = False
    while i < len(shorter) and j < len(longer):
        if shorter[i] == longer[j]:
            i += 1
            j += 1
            continue
        if skipped:
            return False
        skipped = True
        j += 1  # skip one char in the longer string
    return True


def _recover_structural_name(name: str) -> str | None:
    """Recover the canonical DSML structural tag name from a garbled ``name``.

    Returns the canonical name (``invoke`` / ``parameter`` / ``function_calls``)
    when ``name`` is an unambiguous garble of exactly one of them, else None.
    Recovery shapes, in priority order:

      1. Suffix recovery — ``name`` is longer than a canonical and ENDS with it
         (e.g. ``invinvoke`` → ``invoke``: the ``inv`` token split into
         ``in`` + ``v``, doubling the prefix). Observed 2026-06-18.
      2. Single-edit recovery — ``name`` is within one character edit of a
         canonical (substitution/insertion/deletion), e.g. ``invode`` →
         ``invoke`` (the model sampled a neighbor token at temp=1.0 in the
         rigid tag region). Observed 2026-06-20 (MTP-off, both nodes).

    This mirrors what the upstream DeepSeek stacks do (vLLM / sglang /
    HF ``encoding_v32`` all normalize known tool-call tag garbles before the
    strict parse) rather than failing the whole turn on a one-token slip. It is
    safe because the ``｜DSML｜`` sentinel is a dedicated special vocab token that
    only appears inside genuine tool-call blocks — a garbled name carrying the
    sentinel is known-structural-intent, so recovery cannot misfire on prose.
    Returns None when no canonical is unambiguously recoverable, preserving the
    fail-the-turn behavior for truly broken blocks.
    """
    if name in _DSML_STRUCTURAL_NAMES:
        return None  # not garbled
    # 1. Suffix recovery (handles doubled-prefix garbles like ``invinvoke``).
    for canonical in _DSML_STRUCTURAL_NAMES:
        if name.endswith(canonical) and len(name) > len(canonical):
            return canonical
    # 2. Single-edit recovery (substitution / insertion / deletion). Require a
    # UNIQUE canonical match so an ambiguous garble is never silently
    # mis-repaired (the structural names are far apart, so this is the common
    # case, but guard against it regardless).
    edit_matches = [
        canonical
        for canonical in _DSML_STRUCTURAL_NAMES
        if _within_edit_distance_one(name, canonical)
    ]
    if len(edit_matches) == 1:
        return edit_matches[0]
    return None


def _repair_dsml_tag_garbles(text: str) -> str:
    """Repair known DSML structural-tag token-garble shapes.

    Scans for malformed ``<｜DSML｜...>`` / ``</｜DSML｜...>`` tags whose name is a
    recoverable garble of a known structural tag name (``invoke``,
    ``parameter``, ``function_calls``) and normalizes them to the canonical
    tag. See ``_recover_structural_name`` for the recovery shapes (suffix
    doubling + single-character edit) and the safety argument.

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
        canonical = _recover_structural_name(name)
        if canonical is None:
            # Not a recoverable garble — leave it so the strict regex fails
            # loudly (preserves fail-the-turn for truly broken blocks).
            return match.group(0)
        return f"{slash}{DSML_TOKEN}{canonical}{rest}"

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


# ── Sentinel-less (wrong-dialect) tool-call recovery ─────────────────────────
#
# DSv4 occasionally emits the CORRECT invoke/parameter STRUCTURE but drops the
# ``｜DSML｜`` sentinel from every tag — the bare Claude/minimax dialect, e.g.
# (observed live 2026-06-29, msg 95278; the read_file call dropped + leaked):
#
#     <tool_call>
#     <invoke name="read_file">
#     <parameter name="limit" string="false">15</parameter>
#     <parameter name="path" string="true">~/.hermes/config.yaml</parameter>
#     </invoke>
#
# The strict DSML parser only matches sentinel-bearing tags, so this slips
# through: the tags leak into ``content`` AND the tool call is lost. Two
# parameter shapes occur — DSv4's own ``string="true|false"`` annotation, and
# the pure Claude/minimax form ``<parameter name="x">value</parameter>`` with
# NO attribute (test corpus: mlx-lm minimax_m2). We recover both.
#
# Safety: this parser is only invoked from the recovery stage AFTER the
# sentinel-less SIGNATURE (a tool-call opener + a ``<parameter name=…>`` tag,
# with no ``｜DSML｜`` anywhere) has already been confirmed — see
# ``_is_sentinelless_tool_call`` in model_output_parsers.py. It is never run on
# free prose, so the permissive tag matching here cannot misfire on ordinary
# text. Returns None when nothing parses, so the caller can clean-fail (the
# pre-recovery behavior) for a truly corrupt block.

_BARE_INVOKE_PATTERN = re.compile(
    r"<invoke\s+name=\"([^\"]+)\">(.*?)</invoke>",
    re.DOTALL,
)
# Parameter with DSv4's explicit ``string="true|false"`` type annotation.
_BARE_PARAM_TYPED_PATTERN = re.compile(
    r"<parameter\s+name=\"([^\"]+)\"\s+string=\"(true|false)\"\s*>(.*?)</parameter>",
    re.DOTALL,
)
# Parameter in the pure Claude/minimax dialect: no type attribute at all.
_BARE_PARAM_PLAIN_PATTERN = re.compile(
    r"<parameter\s+name=\"([^\"]+)\"\s*>(.*?)</parameter>",
    re.DOTALL,
)


def parse_sentinelless_tool_call(text: str) -> list[ToolCallItem] | None:
    """Recover a tool call emitted in the wrong (sentinel-less) dialect.

    Parses bare ``<invoke name="…">`` / ``<parameter name="…" [string="…"]>``
    structure that carries no ``｜DSML｜`` sentinel. Handles both the
    ``string="true|false"``-annotated form DSv4 emits and the attribute-less
    Claude/minimax form. ``string="true"`` keeps the verbatim text; otherwise
    the value is ``json.loads``-decoded (so ``15`` → int) with a raw-string
    fallback for non-JSON values like file paths — mirroring the typed-path
    logic in ``parse_dsml_output``. Returns the recovered tool calls, or None
    when the block does not yield a parseable invoke (caller then clean-fails).

    See the module-level note above for the safety argument (only called on a
    pre-confirmed sentinel-less signature, never on free prose).
    """
    if DSML_TOKEN in text:
        # A real/quoted DSML block — not our job; the sentinel parser owns it.
        return None

    tool_calls: list[ToolCallItem] = []
    for invoke_match in _BARE_INVOKE_PATTERN.finditer(text):
        func_name = invoke_match.group(1)
        invoke_body = invoke_match.group(2)

        args: dict[str, Any] = {}
        # Typed params first (consume the explicit-annotation shape), then any
        # remaining plain params. A typed match also matches the plain regex,
        # so track spans to avoid double-counting the same tag.
        typed_spans: list[tuple[int, int]] = []
        for pm in _BARE_PARAM_TYPED_PATTERN.finditer(invoke_body):
            typed_spans.append(pm.span())
            if pm.group(2) == "true":
                args[pm.group(1)] = pm.group(3)
            else:
                try:
                    args[pm.group(1)] = json.loads(pm.group(3))
                except (json.JSONDecodeError, ValueError):
                    args[pm.group(1)] = pm.group(3)
        for pm in _BARE_PARAM_PLAIN_PATTERN.finditer(invoke_body):
            if any(s <= pm.start() < e for s, e in typed_spans):
                continue  # already captured as a typed param
            if pm.group(1) in args:
                continue
            try:
                args[pm.group(1)] = json.loads(pm.group(2))
            except (json.JSONDecodeError, ValueError):
                args[pm.group(1)] = pm.group(2)

        tool_calls.append(
            ToolCallItem(name=func_name, arguments=json.dumps(args))
        )

    return tool_calls if tool_calls else None
