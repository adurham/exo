"""Regression test for the 2026-09-21 hang-watchdog liveness-probe bug.

The liveness probe (supervisor.py's ``_check_hang``, see
``docs/incidents/hang-watchdog-false-positive-and-vision-regression-2026-09-21.md``)
was shipped with a bug caught the SAME DAY on a live cluster: an extension
granted at the first probe tick did not actually hold off the kill for the
promised ``HANG_PROBE_INTERVAL_SECONDS`` -- it protected only the exact tick
it was granted on, and the very next ``_watch_runner`` tick (5s later, per
its ``anyio.sleep(5)`` loop) fell through to an unconditional kill anyway.

Live evidence (m4-1, 2026-09-21 18:01): a real 106K-token cache-busting
prefill logged "silent for 46s ... extending 20s for a growth check", then
was SIGKILLed 5 seconds later at "no event for 54s" -- 15s short of the
promised extension.

This test reproduces that EXACT tick pattern against a real ``RunnerSupervisor``
instance (not a mock) with a controllable fake clock, and asserts the runner
survives past the point where the bug would have killed it.
"""

from __future__ import annotations

import subprocess
from typing import cast

import pytest

from exo.shared.models.model_cards import ModelId
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.events import Event
from exo.shared.types.tasks import Task, TaskId, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import BoundInstance, InstanceId
from exo.shared.types.worker.runners import RunnerId
from exo.utils.async_process import AsyncProcess
from exo.utils.channels import channel, mp_channel
from exo.worker.runner import supervisor as supervisor_module
from exo.worker.runner.bootstrap import RunnerTerminationError
from exo.worker.runner.supervisor import RunnerStdioHandler, RunnerSupervisor
from exo.worker.tests.unittests.conftest import get_bound_mlx_ring_instance


class _AliveProcess:
    """A fake AsyncProcess that reports alive with an OBVIOUSLY-FAKE PID.

    9999999 is deliberately not a real PID on this machine. The kill path
    this test exercises (``os.kill`` + a real ``subprocess.run(["/usr/bin/
    sample", ...])``) is monkeypatched out in every test below BEFORE
    ``_check_hang`` can reach a kill decision -- this fake PID is a second,
    independent guard: if a future edit removes one of those patches, the
    test fails loudly (ProcessLookupError / sample erroring on a bogus PID)
    instead of silently sending a REAL SIGKILL to whatever process happens
    to own this PID, which is exactly what happened during this file's own
    development (an earlier draft used os.getpid() and killed the pytest
    worker running it -- see the incident doc's test history).
    """

    _FAKE_PID = 9999999

    def __init__(self, pid: int | None = None) -> None:
        rx1, _ = channel[bytes]()
        rx2, _ = channel[bytes]()
        self.stdout = rx1
        self.stderr = rx2
        self._pid = pid if pid is not None else self._FAKE_PID

    exitcode = None

    def is_alive(self) -> bool:
        return True

    @property
    def pid(self) -> int:
        return self._pid


async def _make_supervisor() -> RunnerSupervisor:
    event_sender, _event_receiver = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event | RunnerTerminationError]()

    bound_instance: BoundInstance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-hangprobe"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-hangprobe"),
        node_id=NodeId("node-hangprobe"),
    )

    proc = cast(AsyncProcess, cast(object, _AliveProcess()))
    handler = await RunnerStdioHandler.create(
        stdout_rx=proc.stdout, stderr_rx=proc.stderr
    )
    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=proc,
        _runner_stdio_handler=handler,
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )

    task = TextGeneration(
        task_id=TaskId("task-hangprobe"),
        instance_id=bound_instance.instance.instance_id,
        command_id=CommandId("cmd-hangprobe"),
        task_params=TextGenerationTaskParams(
            model=bound_instance.bound_shard.model_card.model_id,
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
            stream=True,
        ),
    )
    supervisor.in_progress[task.task_id] = task
    return supervisor


class _FakeClock:
    """A monotonic-like clock advanced explicitly by the test, so the exact
    real-incident tick pattern (0s, 46s, 51s, 54s, ...) can be reproduced
    deterministically without real sleeps."""

    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def now(self) -> float:
        return self.t

    def advance_to(self, t: float) -> None:
        assert t >= self.t
        self.t = t


def _fake_subprocess_run(
    *args: object, **kwargs: object
) -> subprocess.CompletedProcess[bytes]:
    """Typed stand-in for subprocess.run -- see _neutralize_real_kill_side_effects."""
    return subprocess.CompletedProcess(args=(), returncode=0, stdout=b"", stderr=b"")


def _fake_os_kill(pid: int, sig: int) -> None:
    """Typed stand-in for os.kill -- see _neutralize_real_kill_side_effects."""
    return None


def _fake_not_stopped_or_traced(pid: int) -> bool:
    """Typed stand-in for _process_is_stopped_or_traced, always False."""
    return False


def _neutralize_real_kill_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test that can reach `_check_hang`'s actual kill branch MUST
    call this first. That branch does two REAL, PID-targeting side effects
    after `_hang_killed = True`: a real `subprocess.run(["/usr/bin/sample",
    str(pid), ...])` and a real `os.kill(pid, signal.SIGKILL)`. Even with
    the obviously-fake `_AliveProcess._FAKE_PID`, letting the real `sample`
    binary run against a nonexistent PID is slow/flaky, and skipping this
    is one accidental fixture change away from repeating this file's own
    development incident (an early draft used a REAL pid and SIGKILLed the
    pytest worker running the test -- see _AliveProcess's docstring)."""
    monkeypatch.setattr(supervisor_module.subprocess, "run", _fake_subprocess_run)
    monkeypatch.setattr(supervisor_module.os, "kill", _fake_os_kill)


@pytest.mark.anyio
async def test_extension_holds_off_kill_across_multiple_ticks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE regression test: reproduces the exact live-incident tick pattern.

    Real incident timeline (HANG_TIMEOUT_SECONDS=45, HANG_PROBE_INTERVAL=20):
      t=46s: first tick past timeout -> probe -> baseline recorded, "extending
             20s" logged (buggy code: sets a THIS-TICK-ONLY allowance)
      t=51s: next _watch_runner tick (5s poll interval) -- BUG: fell through
             to kill here in the shipped-then-fixed version. FIXED version
             must return early (deadline is t=66s, we're at t=51s).
      t=54s: another tick -- BUGGY CODE KILLED HERE. Fixed version must still
             defer (54 < 66).
      t=66s: deadline reached -- a probe is now DUE. With a growing
             footprint, this extends again rather than killing.
    """
    supervisor = await _make_supervisor()
    clock = _FakeClock(start=0.0)
    monkeypatch.setattr(supervisor_module.time, "monotonic", clock.now)
    supervisor._last_event_monotonic = 0.0  # pyright: ignore[reportPrivateUsage]

    # Deterministic, growing footprint: first probe returns 90.0GB (baseline),
    # second returns 91.0GB (+1GB, comfortably over the 0.25GB threshold).
    footprint_calls: list[int] = []
    footprint_sequence = [90.0, 91.0]

    def _fake_sample(pid: int, duration_s: int) -> float | None:
        footprint_calls.append(len(footprint_calls))
        idx = min(len(footprint_calls) - 1, len(footprint_sequence) - 1)
        return footprint_sequence[idx]

    monkeypatch.setattr(
        supervisor_module, "_sample_physical_footprint_gb", _fake_sample
    )
    monkeypatch.setattr(
        supervisor_module, "_process_is_stopped_or_traced", _fake_not_stopped_or_traced
    )

    assert supervisor_module.HANG_TIMEOUT_SECONDS == 45.0, (
        "test assumes the documented default; update the timeline in this "
        "test's docstring if the default ever changes"
    )

    # t=46s: past the 45s timeout -> first probe -> baseline + extend.
    clock.advance_to(46.0)
    await supervisor._check_hang()  # pyright: ignore[reportPrivateUsage]
    assert not supervisor._hang_killed, (  # pyright: ignore[reportPrivateUsage]
        "must not kill on the tick that establishes the baseline probe"
    )
    assert len(footprint_calls) == 1

    # t=51s: mid-extension-window tick (the exact +5s _watch_runner poll
    # cadence). THE BUG: shipped code killed here. Fixed code must defer
    # WITHOUT even re-probing (51 < deadline of 66).
    clock.advance_to(51.0)
    await supervisor._check_hang()  # pyright: ignore[reportPrivateUsage]
    assert not supervisor._hang_killed, (  # pyright: ignore[reportPrivateUsage]
        "REGRESSION: killed mid-extension-window, exactly the live-incident bug "
        "(baseline probe granted a 20s extension at t=46s, but the kill fired "
        "anyway at t=51s -- 15s early)"
    )
    assert len(footprint_calls) == 1, "must not re-probe before the deadline"

    # t=54s: the EXACT second at which the live incident's SIGKILL fired.
    clock.advance_to(54.0)
    await supervisor._check_hang()  # pyright: ignore[reportPrivateUsage]
    assert not supervisor._hang_killed, (  # pyright: ignore[reportPrivateUsage]
        "REGRESSION: this is the exact tick (t=54s, 'no event for 54s') the "
        "live incident's SIGKILL fired on, 15s before the promised 20s "
        "extension (granted at t=46s) had elapsed"
    )
    assert len(footprint_calls) == 1, "must not re-probe before the deadline"

    # t=66s: deadline reached (46 + 20). A new probe is due; footprint grew
    # by 1.0GB (>= the 0.25GB threshold) -> extend again, not kill.
    clock.advance_to(66.0)
    await supervisor._check_hang()  # pyright: ignore[reportPrivateUsage]
    assert not supervisor._hang_killed, (  # pyright: ignore[reportPrivateUsage]
        "growing footprint at the deadline must extend again, not kill"
    )
    assert len(footprint_calls) == 2
    assert supervisor._hang_probe_extensions_used == 2  # pyright: ignore[reportPrivateUsage]


@pytest.mark.anyio
async def test_plateaued_footprint_kills_at_next_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other side of the same fix: a GENUINELY wedged runner (static
    footprint across probes) must still be killed -- at the deadline, not
    before it and not never."""
    supervisor = await _make_supervisor()
    clock = _FakeClock(start=0.0)
    monkeypatch.setattr(supervisor_module.time, "monotonic", clock.now)
    supervisor._last_event_monotonic = 0.0  # pyright: ignore[reportPrivateUsage]
    _neutralize_real_kill_side_effects(monkeypatch)

    # Static footprint across every probe -- a genuine wedge.
    def _fake_sample(pid: int, duration_s: int) -> float | None:
        return 50.0

    monkeypatch.setattr(
        supervisor_module, "_sample_physical_footprint_gb", _fake_sample
    )
    monkeypatch.setattr(
        supervisor_module, "_process_is_stopped_or_traced", _fake_not_stopped_or_traced
    )

    clock.advance_to(46.0)
    await supervisor._check_hang()  # pyright: ignore[reportPrivateUsage]
    assert not supervisor._hang_killed  # pyright: ignore[reportPrivateUsage]

    # Mid-window tick: still deferred (this is the fix under test).
    clock.advance_to(51.0)
    await supervisor._check_hang()  # pyright: ignore[reportPrivateUsage]
    assert not supervisor._hang_killed  # pyright: ignore[reportPrivateUsage]

    # At the deadline: footprint has NOT grown -> kill.
    clock.advance_to(66.0)
    await supervisor._check_hang()  # pyright: ignore[reportPrivateUsage]
    assert supervisor._hang_killed, (  # pyright: ignore[reportPrivateUsage]
        "a genuinely static footprint at the probe deadline must still kill "
        "-- the fix must not weaken real-wedge detection"
    )


@pytest.mark.anyio
async def test_extension_budget_exhaustion_still_kills(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A runner that keeps growing forever cannot stall the watchdog forever
    -- HANG_PROBE_MAX_EXTENSIONS must eventually be exhausted and kill."""
    supervisor = await _make_supervisor()
    clock = _FakeClock(start=0.0)
    monkeypatch.setattr(supervisor_module.time, "monotonic", clock.now)
    supervisor._last_event_monotonic = 0.0  # pyright: ignore[reportPrivateUsage]
    _neutralize_real_kill_side_effects(monkeypatch)

    growth_counter = {"n": 0.0}

    def _fake_sample(pid: int, duration_s: int) -> float | None:
        growth_counter["n"] += 10.0  # always grows -- never plateaus
        return growth_counter["n"]

    monkeypatch.setattr(
        supervisor_module, "_sample_physical_footprint_gb", _fake_sample
    )
    monkeypatch.setattr(
        supervisor_module, "_process_is_stopped_or_traced", _fake_not_stopped_or_traced
    )
    monkeypatch.setattr(supervisor_module, "HANG_PROBE_MAX_EXTENSIONS", 2)

    t = 46.0
    clock.advance_to(t)
    await supervisor._check_hang()  # pyright: ignore[reportPrivateUsage]
    assert not supervisor._hang_killed  # pyright: ignore[reportPrivateUsage]

    t += supervisor_module.HANG_PROBE_INTERVAL_SECONDS
    clock.advance_to(t)
    await supervisor._check_hang()  # pyright: ignore[reportPrivateUsage]
    assert not supervisor._hang_killed, (  # pyright: ignore[reportPrivateUsage]
        "2nd extension (budget=2) should still be granted"
    )

    t += supervisor_module.HANG_PROBE_INTERVAL_SECONDS
    clock.advance_to(t)
    await supervisor._check_hang()  # pyright: ignore[reportPrivateUsage]
    assert supervisor._hang_killed, (  # pyright: ignore[reportPrivateUsage]
        "budget exhausted (used=2/2) -- must kill even though footprint is "
        "still growing, so a slow-leak process cannot stall the watchdog "
        "forever"
    )

# ── Native-setup discriminator (2026-09-23) ──────────────────────────────
# Third false-positive class: a runner blocked in JACCL side-channel /
# TCP all-gather setup has a FLAT footprint (it consumes no memory while
# blocked) but is genuinely waiting on its peer, not wedged. The footprint
# probe alone therefore misreads it as a hang. These tests pin both
# directions: the new check must EXTEND when the stack shows native setup,
# and must NOT weaken the existing real-wedge kill.


def _fake_sample_bytes_native_setup(*_a, **_k):
    """subprocess.run stand-in whose stdout is BYTES containing the real
    2026-09-23 call chain (the shape that was wrongly killed)."""
    dump = (
        b"Analysis of sampling python (pid 32526) every 1 millisecond\n"
        b"Physical footprint:         387.3M\n"
        b"Call graph:\n"
        b"    2186 Thread_1   DispatchQueue_1: com.apple.main-thread  (serial)\n"
        b"    + 2186 mlx::core::distributed::init(bool, std::string const&)\n"
        b"    +   2186 mlx::core::distributed::jaccl::init(bool)\n"
        b"    +     2186 jaccl::init(jaccl::Config const&, bool)\n"
        b"    +       2186 jaccl::Config::get_side_channel() const\n"
        b"    +         2186 jaccl::TCPAllGather::TCPAllGather(int, int, char const*)\n"
    )
    return subprocess.CompletedProcess(args=(), returncode=0, stdout=dump, stderr=b"")


def test_native_setup_symbols_are_detected():
    """The detector recognises the exact 2026-09-23 stack."""
    assert supervisor_module._NATIVE_BLOCKED_SYMBOLS  # pyright: ignore[reportPrivateUsage]
    dump_hits = [
        s
        for s in supervisor_module._NATIVE_BLOCKED_SYMBOLS  # pyright: ignore[reportPrivateUsage]
        if s in (
            "mlx::core::distributed::init\n"
            "jaccl::init(jaccl::Config const&, bool)\n"
            "jaccl::Config::get_side_channel() const\n"
            "jaccl::TCPAllGather::TCPAllGather(int, int, char const*)\n"
        )
    ]
    assert len(dump_hits) >= 2, "the real dump must arm the detector"


def test_bytes_stdout_is_failsafe_not_crash(monkeypatch: pytest.MonkeyPatch):
    """Bytes stdout must be decoded, never raise, and a dump WITHOUT the
    native-setup symbols must return False (fail-safe)."""
    neutral = subprocess.CompletedProcess(
        args=(),
        returncode=0,
        stdout=b"Physical footprint: 87.0G\ncom.apple.main-thread\n"
        b"mlx::core::eval()\nEvent::wait()\n",
        stderr=b"",
    )
    monkeypatch.setattr(
        supervisor_module.subprocess, "run", lambda *a, **k: neutral
    )
    assert (
        supervisor_module._sample_is_blocked_in_native_setup(  # pyright: ignore[reportPrivateUsage]
            12345, 1
        )
        is False
    ), "a normal compute stack must NOT be reported as native-blocked"


def test_bytes_stdout_with_native_setup_returns_true(
    monkeypatch: pytest.MonkeyPatch,
):
    """The positive direction: bytes output containing the side-channel
    chain must be detected as native-blocked."""
    monkeypatch.setattr(
        supervisor_module.subprocess, "run", _fake_sample_bytes_native_setup
    )
    assert (  # pyright: ignore[reportPrivateUsage]
        supervisor_module._sample_is_blocked_in_native_setup(12345, 1) is True
    )


def test_unparseable_output_is_failsafe(monkeypatch: pytest.MonkeyPatch):
    """Garbage/empty output must return False, so a genuine wedge is still
    killed -- this function must never be able to suppress a real kill."""
    for junk in (b"", b"\xff\xfe", "", None):
        empty = subprocess.CompletedProcess(
            args=(), returncode=0, stdout=junk, stderr=b""
        )
        monkeypatch.setattr(
            supervisor_module.subprocess, "run", lambda *a, **k: empty
        )
        assert (  # pyright: ignore[reportPrivateUsage]
            supervisor_module._sample_is_blocked_in_native_setup(1, 1) is False
        ), f"junk={junk!r} must fail safe"
