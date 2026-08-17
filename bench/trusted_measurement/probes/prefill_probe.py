"""Prefill throughput probe scaffold (Phase 2).

REAL TODAY (typed, tested, no cluster required)
-----------------------------------------------
* :func:`build_prefill_prompt` - depth-matrix-aware prompt construction: builds
  a needle-in-haystack prompt and grows it until it tokenizes to within
  tolerance of a target token count, using a **real** offline tokenizer. The
  needle survives the growth, so the same prompt serves as both the timing
  workload and the content-correctness check.
* :func:`prefill_throughput_tokens_per_second` - throughput arithmetic on the
  *offline* token count, never the server's claim.
* :func:`build_prefill_record` - full record construction including
  :func:`~trusted_measurement.token_truth.cross_check_token_count`, which
  raises when the server's ``prompt_tokens`` disagrees with real tokenization
  (the 1.42x inflation bug class).
* :class:`FakePrefillClient` - a complete fake satisfying :class:`PrefillClient`,
  used by the regression tests.

INTERFACE STUB (needs the live cluster)
---------------------------------------
* :class:`PrefillClient` - the one network call. The real implementation POSTs
  a chat-completion request to the cluster API, times to first token, and
  returns the server's reported prompt-token count and the completion text.
  Nothing else in this module talks to the network.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from typing import Protocol, final, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from trusted_measurement.content_check import (
    ContentCheckResult,
    NeedleHaystack,
    build_needle_haystack,
    run_needle_check,
)
from trusted_measurement.depth_matrix import ContextDepth, DepthMatrixCell, ThermalState
from trusted_measurement.fingerprint import (
    CommandRunner,
    Fingerprint,
    capture_fingerprint,
)
from trusted_measurement.record import MeasurementRecord
from trusted_measurement.replication import (
    DEFAULT_DISPERSION_THRESHOLD,
    DEFAULT_MINIMUM_REPLICATES,
    aggregate_replicates,
)
from trusted_measurement.runtime_mode import RuntimeModeMarker
from trusted_measurement.token_truth import (
    TokenGroundTruth,
    Tokenizer,
    cross_check_token_count,
)

__all__ = [
    "PREFILL_REQUIRED_RUNTIME_MODES",
    "FakePrefillClient",
    "PrefillClient",
    "PrefillProbeConfig",
    "PrefillPrompt",
    "PrefillResponse",
    "build_argument_parser",
    "build_prefill_prompt",
    "build_prefill_record",
    "main",
    "prefill_throughput_tokens_per_second",
]

PREFILL_REQUIRED_RUNTIME_MODES: tuple[str, ...] = ("batched_prefill",)
"""TODO(phase-3, needs coordination): emitted from the prefill path itself."""


@final
class PrefillPrompt(BaseModel):
    """A prompt built to a target token count, with its real token count."""

    model_config = ConfigDict(frozen=True, strict=True)

    text: str = Field(min_length=1)
    needle: str = Field(min_length=1)
    offline_token_count: int = Field(gt=0)
    target_token_count: int = Field(gt=0)
    filler_sentences: int = Field(gt=0)

    @property
    def relative_error(self) -> float:
        return abs(self.offline_token_count - self.target_token_count) / (
            self.target_token_count
        )

    def haystack(self) -> NeedleHaystack:
        """The needle/haystack view used for the content-correctness check."""
        return NeedleHaystack(
            prompt=self.text,
            needle=self.needle,
            question="What is the secret access code? Reply with the digits only.",
        )


@final
class PrefillResponse(BaseModel):
    """What one prefill request returned."""

    model_config = ConfigDict(frozen=True, strict=True)

    completion_text: str
    server_reported_prompt_tokens: int = Field(ge=0)
    time_to_first_token_seconds: float = Field(gt=0.0)


@runtime_checkable
class PrefillClient(Protocol):
    """INTERFACE STUB - the single network call this probe makes.

    A real implementation posts ``prompt`` to the cluster, measures wall-clock
    time to first token, and returns the server's reported prompt token count
    verbatim (never corrected - the whole point is to cross-check it).
    """

    def run_prefill(self, prompt: str) -> PrefillResponse: ...


@final
class FakePrefillClient:
    """Deterministic fake client for offline tests.

    Answers the needle correctly and reports a prompt-token count derived from
    the injected tokenizer, optionally scaled by ``token_count_scale`` so tests
    can reproduce the server-inflation bug class.
    """

    def __init__(
        self,
        *,
        tokenizer: Tokenizer,
        needle: str,
        seconds_per_thousand_tokens: float = 1.0,
        token_count_scale: float = 1.0,
        completion_override: str | None = None,
    ) -> None:
        self._tokenizer: Tokenizer = tokenizer
        self._needle: str = needle
        self._seconds_per_thousand: float = seconds_per_thousand_tokens
        self._scale: float = token_count_scale
        self._completion_override: str | None = completion_override
        self.calls: int = 0

    def run_prefill(self, prompt: str) -> PrefillResponse:
        self.calls += 1
        token_count = len(self._tokenizer.encode(prompt))
        completion = (
            f"The secret access code is {self._needle}."
            if self._completion_override is None
            else self._completion_override
        )
        return PrefillResponse(
            completion_text=completion,
            server_reported_prompt_tokens=int(round(token_count * self._scale)),
            time_to_first_token_seconds=max(
                1e-6, token_count / 1000.0 * self._seconds_per_thousand
            ),
        )


@final
class PrefillProbeConfig(BaseModel):
    """Which cells to measure and how large the prompt is at each."""

    model_config = ConfigDict(frozen=True, strict=True)

    target_tokens_by_depth: Mapping[ContextDepth, int]
    thermal_states: tuple[ThermalState, ...] = Field(min_length=1)
    replicates_per_cell: int = Field(ge=1)
    seed: int = 20260817
    token_tolerance_fraction: float = Field(gt=0.0, le=0.5, default=0.02)

    def cells(self) -> tuple[tuple[ContextDepth, ThermalState], ...]:
        return tuple(
            (depth, thermal)
            for depth in self.target_tokens_by_depth
            for thermal in self.thermal_states
        )


def build_prefill_prompt(
    *,
    tokenizer: Tokenizer,
    target_token_count: int,
    seed: int,
    tolerance_fraction: float = 0.02,
    max_iterations: int = 40,
) -> PrefillPrompt:
    """Build a needle prompt that really tokenizes to ~``target_token_count``.

    Uses measured tokens-per-filler-sentence rather than a character heuristic,
    so the result is correct for any tokenizer. Raises if it cannot converge
    within ``tolerance_fraction`` - an approximate prompt size silently
    corrupts every tokens-per-second number computed from it.
    """
    if target_token_count <= 0:
        raise ValueError("target_token_count must be positive")
    filler = 1
    best: PrefillPrompt | None = None
    for _ in range(max_iterations):
        haystack = build_needle_haystack(filler_sentences=filler, seed=seed)
        count = len(tokenizer.encode(haystack.prompt))
        candidate = PrefillPrompt(
            text=haystack.prompt,
            needle=haystack.needle,
            offline_token_count=count,
            target_token_count=target_token_count,
            filler_sentences=filler,
        )
        if best is None or candidate.relative_error < best.relative_error:
            best = candidate
        if candidate.relative_error <= tolerance_fraction:
            return candidate
        tokens_per_sentence = count / filler
        if tokens_per_sentence <= 0.0:
            raise ValueError("tokenizer produced no tokens for the filler text")
        next_filler = max(1, int(round(target_token_count / tokens_per_sentence)))
        filler = filler + 1 if next_filler == filler else next_filler
    assert best is not None
    raise ValueError(
        f"could not build a prompt within {tolerance_fraction:.1%} of "
        f"{target_token_count} tokens (best was {best.offline_token_count})"
    )


def prefill_throughput_tokens_per_second(
    *, offline_token_count: int, time_to_first_token_seconds: float
) -> float:
    """Prefill tok/s, computed on the offline count only."""
    if time_to_first_token_seconds <= 0.0:
        raise ValueError("time to first token must be positive")
    if offline_token_count <= 0:
        raise ValueError("offline token count must be positive")
    return offline_token_count / time_to_first_token_seconds


def _measure_cell(
    *,
    client: PrefillClient,
    prompt: PrefillPrompt,
    tokenizer: Tokenizer,
    tokenizer_name: str,
    replicates: int,
) -> tuple[tuple[float, ...], ContentCheckResult, TokenGroundTruth]:
    throughputs: list[float] = []
    content_check: ContentCheckResult | None = None
    ground_truth: TokenGroundTruth | None = None
    for _ in range(replicates):
        response = client.run_prefill(prompt.text)
        ground_truth = cross_check_token_count(
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            prompt=prompt.text,
            server_reported_token_count=response.server_reported_prompt_tokens,
        )
        content_check = run_needle_check(prompt.haystack(), response.completion_text)
        throughputs.append(
            prefill_throughput_tokens_per_second(
                offline_token_count=prompt.offline_token_count,
                time_to_first_token_seconds=response.time_to_first_token_seconds,
            )
        )
    assert content_check is not None
    assert ground_truth is not None
    return tuple(throughputs), content_check, ground_truth


def build_prefill_record(
    *,
    client: PrefillClient,
    tokenizer: Tokenizer,
    tokenizer_name: str,
    config: PrefillProbeConfig,
    runtime_mode_markers: Sequence[RuntimeModeMarker],
    canary_session_certified: bool,
    exo_repo: str,
    mlx_repo: str,
    command_runner: CommandRunner,
    environ: Mapping[str, str],
    fingerprint: Fingerprint | None = None,
    claims_default_safe: bool = False,
    notes: str = "",
) -> MeasurementRecord:
    """Run the full depth-matrix prefill sweep and build one record."""
    cells: list[DepthMatrixCell] = []
    replicates: list[float] = []
    content_check: ContentCheckResult | None = None
    ground_truth: TokenGroundTruth | None = None
    for depth, thermal in config.cells():
        prompt = build_prefill_prompt(
            tokenizer=tokenizer,
            target_token_count=config.target_tokens_by_depth[depth],
            seed=config.seed,
            tolerance_fraction=config.token_tolerance_fraction,
        )
        cell_values, content_check, ground_truth = _measure_cell(
            client=client,
            prompt=prompt,
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            replicates=config.replicates_per_cell,
        )
        replicates.extend(cell_values)
        cells.append(
            DepthMatrixCell(
                context_depth=depth,
                thermal_state=thermal,
                prompt_tokens=prompt.offline_token_count,
            )
        )
    if content_check is None or ground_truth is None:
        raise ValueError("prefill config produced no measured cells")
    value = aggregate_replicates(
        metric_name="prefill_throughput",
        unit="tok/s",
        replicates=tuple(replicates),
        minimum_replicates=DEFAULT_MINIMUM_REPLICATES,
        dispersion_threshold=DEFAULT_DISPERSION_THRESHOLD,
    )
    captured = (
        capture_fingerprint(exo_repo, mlx_repo, runner=command_runner, environ=environ)
        if fingerprint is None
        else fingerprint
    )
    return MeasurementRecord(
        probe_name="prefill_throughput",
        value=value,
        content_check=content_check,
        fingerprint=captured,
        runtime_mode_markers=tuple(runtime_mode_markers),
        required_runtime_modes=PREFILL_REQUIRED_RUNTIME_MODES,
        depth_matrix_cells=tuple(cells),
        claims_default_safe=claims_default_safe,
        token_ground_truth=ground_truth,
        prompt_size_dependent=True,
        canary_session_certified=canary_session_certified,
        notes=notes,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prefill_probe")
    _ = parser.add_argument("--exo-repo", default=".")
    _ = parser.add_argument("--mlx-repo", default="./mlx")
    _ = parser.add_argument("--base-url", default="http://localhost:52415")
    _ = parser.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
    _ = parser.add_argument("--shallow-tokens", type=int, default=2048)
    _ = parser.add_argument("--mid-tokens", type=int, default=32768)
    _ = parser.add_argument("--near-max-tokens", type=int, default=131072)
    _ = parser.add_argument(
        "--replicates", type=int, default=DEFAULT_MINIMUM_REPLICATES
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Refuses to run: no live PrefillClient exists yet."""
    _ = build_argument_parser().parse_args(None if argv is None else list(argv))
    raise NotImplementedError(
        "prefill_probe has no live PrefillClient yet. Implement PrefillClient "
        "against the cluster API and pass it to build_prefill_record."
    )
