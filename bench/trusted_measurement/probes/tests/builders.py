"""Shared fakes for the Phase 2 probe regression suite."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import final

from trusted_measurement.fingerprint import Fingerprint, LinkTopology
from trusted_measurement.runtime_mode import RuntimeModeMarker, RuntimeModeRecorder

__all__ = [
    "WordTokenizer",
    "fake_command_runner",
    "markers_for",
    "probe_fingerprint",
]


@final
class WordTokenizer:
    """A real, deterministic tokenizer: one token per whitespace-separated word.

    Satisfies :class:`~trusted_measurement.token_truth.Tokenizer` and lets the
    prefill probe's token arithmetic be exercised for real offline.
    """

    def encode(self, text: str, /) -> list[int]:
        return [hash(word) % 50257 for word in text.split()]


def fake_command_runner(
    *, exo_commit: str = "a" * 40, mlx_commit: str = "b" * 40, links: int = 6
):
    """A ``CommandRunner`` that answers git/system_profiler without shelling out."""

    def runner(command: Sequence[str]) -> str:
        parts = list(command)
        if parts[0] == "git" and parts[-1] == "HEAD":
            return (exo_commit if "mlx" not in parts[2] else mlx_commit) + "\n"
        if parts[0] == "git":
            return ""
        return json.dumps(
            {
                "SPThunderboltDataType": [
                    {
                        "_name": "bus",
                        "_items": [
                            {"_name": f"link {index}", "receptacle_1_tag": {}}
                            for index in range(links)
                        ],
                    }
                ]
            }
        )

    return runner


def probe_fingerprint(**overrides: object) -> Fingerprint:
    fields: dict[str, object] = dict(
        exo_commit="a" * 40,
        mlx_commit="b" * 40,
        exo_dirty=False,
        mlx_dirty=False,
        registered_env={"MLX_JACCL_SHARDING_MODE": "tp"},
        link_topology=LinkTopology(
            thunderbolt_link_count=6,
            link_descriptors=("link-0",),
            source="system_profiler",
        ),
        hostname="test-host",
    )
    fields.update(overrides)
    return Fingerprint(**fields)  # pyright: ignore[reportArgumentType]


def markers_for(*modes: str) -> tuple[RuntimeModeMarker, ...]:
    recorder = RuntimeModeRecorder()
    for mode in modes:
        recorder.emit(mode, detail="probe test", count=3)
    return recorder.markers()
