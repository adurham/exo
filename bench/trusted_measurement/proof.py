"""Unforgeable in-process proof tokens.

Several envelope fields (content-correctness result, runtime-mode markers)
were historically expressible as a plain ``bool`` that a caller could set to
``True`` without the underlying check ever running. That is exactly how the
cancellation-test harness reported green for a whole session while cancelling
already-finished requests.

The fix is that those fields are not booleans: they are *proof tokens* which
can only be minted by the code that actually performed the observation, inside
this process, in this session. A token is an HMAC over the observation's
canonical payload, keyed by a secret generated fresh at import time and never
serialised. A record is only valid if its token re-verifies against the payload
it claims to describe, so a hand-written literal cannot pass.

Because the key is per-process, a token minted in an earlier run (a stale check
copied out of a previous session's JSON) also fails verification. That is
deliberate: point 1 of the design requires the content check to have run in the
SAME process as the measurement.
"""

from __future__ import annotations

import hmac
import json
import secrets
from hashlib import sha256
from typing import Final, final

__all__ = [
    "SESSION_ID",
    "ProofToken",
    "mint_proof",
    "verify_proof",
]

_SESSION_SECRET: Final[bytes] = secrets.token_bytes(32)

SESSION_ID: Final[str] = sha256(_SESSION_SECRET).hexdigest()[:16]
"""Public, non-secret identifier for this process's proof session."""


def _canonical_payload(domain: str, payload: dict[str, object]) -> bytes:
    return json.dumps(
        {"domain": domain, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        default=repr,
    ).encode("utf-8")


@final
class ProofToken:
    """An opaque, per-process, HMAC-backed attestation of an observation.

    Construct only via :func:`mint_proof`. Equality and ``repr`` deliberately
    avoid leaking the digest so it cannot be lifted out of a log and pasted
    into a hand-built record.
    """

    __slots__ = ("_digest", "_domain", "_session_id")

    def __init__(self, domain: str, digest: str, session_id: str) -> None:
        self._domain: Final[str] = domain
        self._digest: Final[str] = digest
        self._session_id: Final[str] = session_id

    @property
    def domain(self) -> str:
        return self._domain

    @property
    def session_id(self) -> str:
        return self._session_id

    def matches(self, domain: str, payload: dict[str, object]) -> bool:
        if domain != self._domain or self._session_id != SESSION_ID:
            return False
        expected = hmac.new(
            _SESSION_SECRET, _canonical_payload(domain, payload), sha256
        ).hexdigest()
        return hmac.compare_digest(expected, self._digest)

    def __repr__(self) -> str:
        return f"ProofToken(domain={self._domain!r}, session={self._session_id!r})"


def mint_proof(domain: str, payload: dict[str, object]) -> ProofToken:
    """Mint a proof token for an observation that actually happened here."""
    digest = hmac.new(
        _SESSION_SECRET, _canonical_payload(domain, payload), sha256
    ).hexdigest()
    return ProofToken(domain=domain, digest=digest, session_id=SESSION_ID)


def verify_proof(token: ProofToken, domain: str, payload: dict[str, object]) -> bool:
    """Return whether ``token`` really attests ``payload`` in this process."""
    return token.matches(domain, payload)
