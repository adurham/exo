"""Runtime-mode markers (design point 3).

Config says what you asked for. Markers say what actually ran. Every time those
two were assumed equal on this cluster, a config was set but the code path was
never reached, and the resulting numbers described the wrong system.

A probe declares which modes it *requires* to have executed. The measured code
path (or an instrumented shim around it) calls :meth:`RuntimeModeRecorder.emit`
from inside the process while the measurement is running. At validation time,
the record must contain a marker for every required mode, each carrying a proof
token minted at emit time - so "I set the env var" can never stand in for
"the path executed".
"""

from __future__ import annotations

import threading
from typing import final

from pydantic import BaseModel, ConfigDict, Field

from trusted_measurement.proof import ProofToken, mint_proof, verify_proof

__all__ = [
    "RUNTIME_MODE_DOMAIN",
    "RuntimeModeMarker",
    "RuntimeModeRecorder",
]

RUNTIME_MODE_DOMAIN = "runtime_mode"


@final
class RuntimeModeMarker(BaseModel):
    """Evidence that a specific code path executed, emitted from inside it."""

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    mode: str
    detail: str
    observed_count: int = Field(ge=1)
    proof: ProofToken

    def payload(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "detail": self.detail,
            "observed_count": self.observed_count,
        }

    def proof_is_valid(self) -> bool:
        return verify_proof(self.proof, RUNTIME_MODE_DOMAIN, self.payload())


@final
class RuntimeModeRecorder:
    """Thread-safe collector for in-process runtime-mode markers.

    Pass one of these into the measured path. It is intentionally cheap so it
    can live on a hot path (an ``emit`` per layer is fine; counts are folded).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[tuple[str, str], int] = {}

    def emit(self, mode: str, detail: str = "", count: int = 1) -> None:
        """Record that ``mode`` executed. Safe to call from any thread."""
        if count < 1:
            raise ValueError("count must be >= 1")
        with self._lock:
            key = (mode, detail)
            self._counts[key] = self._counts.get(key, 0) + count

    def observed_modes(self) -> frozenset[str]:
        with self._lock:
            return frozenset(mode for mode, _ in self._counts)

    def markers(self) -> tuple[RuntimeModeMarker, ...]:
        """Freeze what was observed into proof-carrying markers."""
        with self._lock:
            snapshot = dict(self._counts)
        markers: list[RuntimeModeMarker] = []
        for (mode, detail), count in sorted(snapshot.items()):
            payload: dict[str, object] = {
                "mode": mode,
                "detail": detail,
                "observed_count": count,
            }
            markers.append(
                RuntimeModeMarker(
                    mode=mode,
                    detail=detail,
                    observed_count=count,
                    proof=mint_proof(RUNTIME_MODE_DOMAIN, payload),
                )
            )
        return tuple(markers)

    def reset(self) -> None:
        with self._lock:
            self._counts.clear()
