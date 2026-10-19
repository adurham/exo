"""Opt-in command-delivery tracing (``EXO_CMD_TRACE=1``).

Why this exists: a >1 MB ``TextGeneration`` (the 1M-token row) is published by the API and
never appears in the master's command processor, with no error on either node. The generic
"the cluster didn't accept the request" after 55 s cannot distinguish two very different
faults: the publish call never completing on the API side, or the command being lost between
the publish and the master's subscription.

With the switch on, the API logs a line before and after the send, and the master logs a line
as each command reaches its processor — all three carry the same command id, byte size and
digest, so the two ends can be matched without guessing. Default off: one env lookup per
command, no serialisation.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

ENV_VAR = "EXO_CMD_TRACE"

_OFF = {"", "0", "false", "no", "off"}


def enabled() -> bool:
    """True when EXO_CMD_TRACE is set to anything but an explicit off value."""
    return os.environ.get(ENV_VAR, "0").strip().lower() not in _OFF


def meta(command: Any) -> str:
    """Stable identity + size for one command, identical on both ends.

    Size is the JSON serialisation the command object produces, not the wire encoding; it is
    the same computation on both sides, which is what makes the two log lines comparable.
    """
    try:
        payload = command.model_dump_json()
    except Exception:  # pragma: no cover - defensive: never let tracing break delivery
        payload = repr(command)
    raw = payload.encode("utf-8", "replace")
    return "kind=%s id=%s bytes=%d digest=%s" % (
        type(command).__name__,
        getattr(command, "command_id", "?"),
        len(raw),
        hashlib.blake2b(raw, digest_size=8).hexdigest(),
    )
