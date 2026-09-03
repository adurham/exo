"""Unit tests for the supervisor's tracing-aware hang watchdog helper.

The helper `_process_is_stopped_or_traced` was introduced 2026-08-22 to prevent
`_check_hang` from SIGKILLing a runner that is deliberately paused by
`xcrun xctrace record --attach <pid>` for a Metal System Trace capture (the
P2 profiling-capture killer that dropped both nodes' captures that night).

We test the helper in isolation by mocking `subprocess.run` so no real ps call
happens; do NOT spawn real runners here (per the task's testing guardrail).
"""

from __future__ import annotations

import subprocess

import pytest

from exo.worker.runner.supervisor import (
    _process_is_stopped_or_traced,  # pyright: ignore[reportPrivateUsage]
)


def _fake_completed(
    stdout: bytes, returncode: int = 0
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess(
        args=["/bin/ps"], returncode=returncode, stdout=stdout, stderr=b""
    )


def _install_fake_ps(
    monkeypatch: pytest.MonkeyPatch,
    stdout: bytes = b"",
    returncode: int = 0,
    raises: type[BaseException] | BaseException | None = None,
    capture: list[list[str]] | None = None,
    capture_kwargs: list[dict[str, object]] | None = None,
) -> None:
    """Install a typed stub for `subprocess.run` — no MagicMock (Any leaks)."""

    def _fake_run(
        cmd: list[str],
        *,
        capture_output: bool = False,
        timeout: float | None = None,
        check: bool = False,
    ) -> subprocess.CompletedProcess[bytes]:
        if capture is not None:
            capture.append(list(cmd))
        if capture_kwargs is not None:
            capture_kwargs.append(
                {
                    "capture_output": capture_output,
                    "timeout": timeout,
                    "check": check,
                }
            )
        if raises is not None:
            if isinstance(raises, BaseException):
                raise raises
            raise raises("stub error")
        return _fake_completed(stdout, returncode=returncode)

    monkeypatch.setattr(subprocess, "run", _fake_run)


def test_process_is_stopped_or_traced_true_for_stopped_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ps -o state=` returning 'T' -> STOPPED/TRACED, defer the kill."""
    _install_fake_ps(monkeypatch, stdout=b"T\n")
    assert _process_is_stopped_or_traced(12345) is True


def test_process_is_stopped_or_traced_true_for_stopped_state_with_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """macOS often includes flag chars after the primary state, e.g. 'T+'."""
    _install_fake_ps(monkeypatch, stdout=b"T+\n")
    assert _process_is_stopped_or_traced(12345) is True


def test_process_is_stopped_or_traced_false_for_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`R` = running -> do NOT defer; caller proceeds with normal kill path."""
    _install_fake_ps(monkeypatch, stdout=b"R+\n")
    assert _process_is_stopped_or_traced(12345) is False


def test_process_is_stopped_or_traced_false_for_sleeping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuinely-hung native-code runner reports 'S' (sleeping in kernel);
    kill must still fire — this is the correctness-preserving case."""
    _install_fake_ps(monkeypatch, stdout=b"S\n")
    assert _process_is_stopped_or_traced(12345) is False


def test_process_is_stopped_or_traced_false_when_ps_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ps` returned non-zero (process already gone) -> conservative False so
    the caller falls back to the existing kill decision."""
    _install_fake_ps(monkeypatch, stdout=b"", returncode=1)
    assert _process_is_stopped_or_traced(99999) is False


def test_process_is_stopped_or_traced_false_on_subprocess_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OSError / timeout -> False (never let the helper raise into the loop)."""
    _install_fake_ps(monkeypatch, raises=OSError("no such file"))
    assert _process_is_stopped_or_traced(12345) is False


def test_process_is_stopped_or_traced_false_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_ps(
        monkeypatch,
        raises=subprocess.TimeoutExpired(cmd=["/bin/ps"], timeout=2),
    )
    assert _process_is_stopped_or_traced(12345) is False


def test_process_is_stopped_or_traced_calls_ps_with_expected_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity check: the shell-out uses `/bin/ps -o state= -p <pid>` with a
    short timeout so it can't wedge the watchdog loop."""
    calls: list[list[str]] = []
    kwargs: list[dict[str, object]] = []
    _install_fake_ps(monkeypatch, stdout=b"R\n", capture=calls, capture_kwargs=kwargs)

    _ = _process_is_stopped_or_traced(4242)

    assert len(calls) == 1
    assert calls[0] == ["/bin/ps", "-o", "state=", "-p", "4242"]
    assert kwargs[0]["timeout"] == 2
    assert kwargs[0]["capture_output"] is True
    assert kwargs[0]["check"] is False
