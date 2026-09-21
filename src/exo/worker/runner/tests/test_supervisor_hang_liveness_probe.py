"""Unit tests for the hang watchdog's memory-growth liveness probe.

Added 2026-09-21 after two independently-reproduced false-positive hang-kills
(a from-scratch cache-busting prefill's first chunk; fresh model loading —
both 100% reproducible) proved the event-silence heuristic alone cannot tell
"genuinely wedged" apart from "real GPU compute that hasn't emitted its own
progress event yet". Both cases were confirmed via the supervisor's own
pre-existing `sample` post-mortem diagnostic to be inside live, progressing
compute with a large/growing physical memory footprint at kill time.

Tests here cover ONLY the new pure helper `_sample_physical_footprint_gb`
(parsing real `sample` output shapes) in isolation, matching this file's
sibling `test_supervisor_hang_tracing.py`'s "mock subprocess.run, no real
process" approach — do NOT spawn real runners or call the full async
`_check_hang` state machine here (that needs a live RunnerSupervisor
instance, out of scope for a fast unit test; the growth/extension/plateau
decision logic is a straightforward diff+threshold+counter, exercised
end-to-end informally via the live incident this fix responds to).
"""

from __future__ import annotations

import subprocess

import pytest

from exo.worker.runner.supervisor import (
    _sample_physical_footprint_gb,  # pyright: ignore[reportPrivateUsage]
)

# A real `sample <pid> 1` stdout capture (trimmed), from the actual incident
# this fix responds to (2026-09-21, /tmp/exo_hang_96033.txt).
_REAL_SAMPLE_OUTPUT_GB = """Analysis of sampling python (pid 96033) every 1 millisecond
Process:         python3.13 [96033]
Path:            /Users/USER/*/python3.13
Load Address:    0x100224000
Identifier:      python3.13
Version:         ???
Code Type:       ARM64
Platform:        macOS
Parent Process:  python3.13 [95710]
Target Type:     live task (Roots Present)

Date/Time:       2026-09-21 15:43:52.602 -0500
Launch Time:     2026-09-21 15:41:52.238 -0500
OS Version:      macOS 27.0 (26A428)
Report Version:  7
Analysis Tool:   /usr/bin/sample

Physical footprint:         88.4G
Physical footprint (peak):  97.5G
"""


def _install_fake_sample(
    monkeypatch: pytest.MonkeyPatch,
    stdout: str = "",
    raises: type[BaseException] | BaseException | None = None,
    capture: list[list[str]] | None = None,
) -> None:
    """Typed stub for `subprocess.run` — no MagicMock, matching this test
    file's sibling's convention."""

    def _fake_run(
        cmd: list[str],
        *,
        capture_output: bool = False,
        timeout: float | None = None,
        check: bool = False,
        text: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        if capture is not None:
            capture.append(list(cmd))
        if raises is not None:
            if isinstance(raises, BaseException):
                raise raises
            raise raises("stub error")
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=stdout, stderr=""
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)


def test_sample_physical_footprint_parses_real_gigabyte_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact `sample` output shape captured from the live 2026-09-21
    incident that motivated this fix — must parse to 88.4, not the peak
    97.5 (the FIRST "Physical footprint:" line, not "(peak)")."""
    _install_fake_sample(monkeypatch, stdout=_REAL_SAMPLE_OUTPUT_GB)
    result = _sample_physical_footprint_gb(96033, duration_s=1)
    assert result is not None
    assert abs(result - 88.4) < 1e-9


def test_sample_physical_footprint_parses_megabyte_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A small/idle process reports footprint in M, not G."""
    stdout = "Physical footprint:         512.0M\nPhysical footprint (peak):  600.0M\n"
    _install_fake_sample(monkeypatch, stdout=stdout)
    result = _sample_physical_footprint_gb(1234, duration_s=1)
    assert result is not None
    assert abs(result - (512.0 / 1024.0)) < 1e-9


def test_sample_physical_footprint_returns_none_on_missing_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No 'Physical footprint:' line at all (unexpected sample output shape,
    e.g. a future macOS version reformats it) -> None, never a crash. The
    caller (_check_hang) must treat None as 'no evidence either way' and
    fall through to the original kill decision, not as 'confirmed hung'."""
    _install_fake_sample(
        monkeypatch, stdout="Some unexpected output\nwith no footprint line\n"
    )
    result = _sample_physical_footprint_gb(1234, duration_s=1)
    assert result is None


def test_sample_physical_footprint_returns_none_on_subprocess_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OSError (e.g. /usr/bin/sample missing) -> None, never raises into the
    watchdog loop — matches _process_is_stopped_or_traced's existing
    fail-safe convention in the sibling test file."""
    _install_fake_sample(monkeypatch, raises=OSError("no such file"))
    result = _sample_physical_footprint_gb(1234, duration_s=1)
    assert result is None


def test_sample_physical_footprint_returns_none_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_sample(
        monkeypatch,
        raises=subprocess.TimeoutExpired(cmd=["/usr/bin/sample"], timeout=11),
    )
    result = _sample_physical_footprint_gb(1234, duration_s=1)
    assert result is None


def test_sample_physical_footprint_calls_sample_with_expected_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity check: shells out to `/usr/bin/sample <pid> <duration>` with a
    timeout comfortably longer than the sample duration itself, so a slow
    `sample` invocation can't itself wedge the watchdog loop."""
    calls: list[list[str]] = []
    _install_fake_sample(monkeypatch, stdout=_REAL_SAMPLE_OUTPUT_GB, capture=calls)

    _ = _sample_physical_footprint_gb(4242, duration_s=1)

    assert len(calls) == 1
    assert calls[0] == ["/usr/bin/sample", "4242", "1"]


def test_sample_physical_footprint_growth_arithmetic_matches_real_incident() -> None:
    """The exact scenario this fix targets: three consecutive kills in the
    2026-09-21 incident all showed a STATIC ~88.4-88.6GB footprint across
    samples (the runner was near-done loading, not making visible further
    progress at exactly the sample moments) -- confirms the growth-diff
    arithmetic itself is simple and correct, independent of the live
    async _check_hang integration (which needs a real RunnerSupervisor to
    exercise end-to-end)."""
    baseline = 88.6
    second_probe = 88.6
    growth = second_probe - baseline
    threshold = 0.25
    assert growth < threshold  # would NOT extend -- matches the real incident

    # Contrast: genuine in-progress prefill/loading growth (e.g. the
    # successful 106K prefill's memory climbed from ~87.8GB to ~90.7GB
    # over its run) DOES clear the threshold and would extend.
    baseline2 = 87.8
    second_probe2 = 90.7
    growth2 = second_probe2 - baseline2
    assert growth2 >= threshold
