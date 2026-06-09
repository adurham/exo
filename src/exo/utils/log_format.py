"""Helpers for keeping log lines readable.

Some objects logged at INFO carry very large payloads — most notably
``TextGeneration`` commands/tasks, whose ``tools`` array is the full
OpenAI tool-schema list a client sends (Hermes ships ~40-200 tools,
hundreds of KB). Logging the raw repr floods the log and buries real
lifecycle/error lines, making incidents hard to diagnose. Cap the repr;
the head retains the type + key identifying fields, which is what's useful.
"""

from __future__ import annotations

DEFAULT_LOG_MAX_CHARS = 600


def truncate_for_log(value: object, max_chars: int = DEFAULT_LOG_MAX_CHARS) -> str:
    """Return ``repr(value)`` capped at ``max_chars`` with a truncation note."""
    text = repr(value)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}… <{len(text)} chars truncated>"
