"""Corpora for the seam-rule harness.

1. ROUND_TRIP_MESSAGE_CORPUS -- message lists rendered through the real
   vendored DSv4 encoder (encode_messages), used for round-trip identity
   at genuine added-token seams (property 1) and for the template
   position-invariance check (property 3).

2. ADVERSARIAL_HOSTILE_STRINGS -- raw hostile text fragments (not run through
   the DSv4 encoder) used to build synthetic "prefix + hostile + suffix"
   strings so we control exactly where the seam lands relative to the
   hostile content (property 2). Each is exercised at both a candidate-safe
   seam (immediately after a real added token) and a candidate-unsafe seam
   (a derived mid-token split, see seam_core.find_midtoken_seam).
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 1. Message-list corpus for round-trip identity + position-invariance
# ---------------------------------------------------------------------------

# Plain multi-turn conversation, no tool calls: the "easy" case where
# position-invariance is expected to hold (each render should be a strict
# byte-prefix of the next, since encode_messages just appends turns).
PLAIN_MULTITURN: list[dict] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 is 4."},
    {"role": "user", "content": "And 3+3?"},
    {"role": "assistant", "content": "3+3 is 6."},
    {"role": "user", "content": "Thanks!"},
]

# Tool-call conversation with TWO tool results arriving as separate messages
# in the incoming message list -- this is the shape that triggers
# merge_tool_messages()/sort_tool_results_by_call_order() re-merging and
# re-sorting (deepseek_v4_encoding.py ~L500-591), per the pre-registration.
TOOLCALL_MULTI_RESULT: list[dict] = [
    {"role": "system", "content": "You can call tools."},
    {"role": "user", "content": "What's the weather in NYC and SF?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "call_a", "function": {"name": "weather", "arguments": '{"city":"NYC"}'}},
            {"id": "call_b", "function": {"name": "weather", "arguments": '{"city":"SF"}'}},
        ],
    },
    # Results arrive OUT OF CALL ORDER (b before a) and as two SEPARATE
    # messages -- both conditions the merge/sort logic actively normalizes.
    {"role": "tool", "tool_call_id": "call_b", "content": "SF: 60F foggy"},
    {"role": "tool", "tool_call_id": "call_a", "content": "NYC: 75F sunny"},
    {"role": "assistant", "content": "NYC is 75F and sunny; SF is 60F and foggy."},
]

ROUND_TRIP_MESSAGE_CORPUS: dict[str, list[dict]] = {
    "plain_multiturn": PLAIN_MULTITURN,
    "toolcall_multi_result": TOOLCALL_MULTI_RESULT,
}


# ---------------------------------------------------------------------------
# 2. Adversarial hostile-seam corpus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HostileCase:
    name: str
    description: str
    hostile_text: str


ADVERSARIAL_HOSTILE_STRINGS: list[HostileCase] = [
    HostileCase(
        name="combining_chars",
        description="Base letter immediately followed by combining diacritics "
        "(e\\u0301\\u0302\\u0303 -- e + acute + circumflex + tilde), "
        "split across the seam.",
        hostile_text="e\u0301\u0302\u0303llo world",
    ),
    HostileCase(
        name="emoji_zwj",
        description="Multi-codepoint ZWJ emoji sequence (family: man, woman, "
        "girl, boy joined by ZWJ) plus a plain emoji, split across the seam.",
        hostile_text="\U0001f468\u200d\U0001f469\u200d\U0001f467\u200d\U0001f466 \U0001f600",
    ),
    HostileCase(
        name="digit_run",
        description="Long run of digits (potential BPE digit-chunking boundary "
        "effects), split across the seam.",
        hostile_text="1234567890123456789",
    ),
    HostileCase(
        name="whitespace_run",
        description="Mixed whitespace run (spaces, tabs, newlines), split "
        "across the seam.",
        hostile_text="   \t\t\n\n   ",
    ),
]
