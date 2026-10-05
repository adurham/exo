"""Stack-class hang-guard tests (design doc "Fix B", 2026-10-05).

The growth probe cannot discriminate at the hardware ceiling: a healthy
deep-context prefill pinned at ~physical size shows a flat footprint and was
SIGKILLed as hung (2026-10-04 soak-2 postmortem). This layer adds a stack
classifier, a SPIN metric (CPU-time delta via ps), and an at-ceiling
condition, gated by ``EXO_RUNNER_HANG_STACK_MODE`` (off|shadow|arm, default
shadow).

Coverage here:
  * pure helpers (cputime parsing, verdict table, stack classifier) against
    the REAL golden fixtures in ~/.hermes/cache/scratch/hang-dumps/:
      - incident_20261004_47545.txt  -> gpu/blocked/at-ceiling (extend-class)
      - synthetic_spin_<pid>.txt     -> a real busy-loop sample (no gpu image,
                                        no native signature) => unknown/spin
  * the full decision matrix against a real RunnerSupervisor with a fake
    clock, for all three modes, incl. budget monotonicity and the
    shadow-changes-no-kill-behavior assertion.

The fixtures are LOCAL reads only (no cluster contact). If the golden dump is
absent the fixture-dependent tests skip rather than fail — the synthetic spin
dump is copied into this directory so the spin-side test always runs.
"""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path
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
from exo.worker.runner.supervisor import (
    RunnerStdioHandler,
    RunnerSupervisor,
    _decide_plateau_verdict,  # pyright: ignore[reportPrivateUsage]
    _parse_cputime_seconds,  # pyright: ignore[reportPrivateUsage]
    _sample_stack_class,  # pyright: ignore[reportPrivateUsage]
)
from exo.worker.tests.unittests.conftest import get_bound_mlx_ring_instance

_DUMPS_DIR = Path.home() / ".hermes/cache/scratch/hang-dumps"
_INCIDENT_DUMP = _DUMPS_DIR / "incident_20261004_47545.txt"
# Copied next to this test so the spin-side classification runs everywhere.
_SPIN_DUMP = Path(__file__).with_name("fixtures") / "synthetic_spin.txt"


def _completed(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=(), returncode=0, stdout=stdout, stderr="")


def _install_stack_sample(
    monkeypatch: pytest.MonkeyPatch, stdout_by_pid: dict[int, str]
) -> None:
    """Route a fake `subprocess.run` by the sampled PID to a dump's contents.

    Only `sample` calls are faked; ps/sysctl calls (different argv) get an
    empty successful result, which those helpers already treat as
    "no evidence".
    """

    def _fake_run(
        cmd: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if cmd[:1] == ["/usr/bin/sample"] and len(cmd) >= 2:
            pid = int(cmd[1])
            return _completed(stdout_by_pid.get(pid, ""))
        return _completed("")

    monkeypatch.setattr(supervisor_module.subprocess, "run", _fake_run)


# ── Pure helpers ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0:00.00", 0.0),
        ("0:12.34", 12.34),
        ("1:02.50", 62.5),
        ("2:04:00.00", 7440.0),
        ("3-04:05:06.00", 3 * 86400 + 4 * 3600 + 5 * 60 + 6),
        ("", None),
        ("garbage", None),
        ("1:2:3:4", None),
    ],
)
def test_parse_cputime_seconds(text: str, expected: float | None) -> None:
    got = _parse_cputime_seconds(text)
    if expected is None:
        assert got is None
    else:
        assert got is not None
        assert abs(got - expected) < 1e-9


def test_decide_plateau_verdict_full_matrix() -> None:
    """Every row of the design doc's armed decision table."""
    # growth -> extend regardless of anything else.
    assert (
        _decide_plateau_verdict(
            growth_gb=0.30,
            spin=False,
            stack_class="unknown",
            at_ceiling=False,
            sample_failed_once=False,
        )
        == "extend"
    )
    # flat + spin -> kill fast (even with a gpu stack at the ceiling).
    assert (
        _decide_plateau_verdict(
            growth_gb=0.0,
            spin=True,
            stack_class="gpu",
            at_ceiling=True,
            sample_failed_once=False,
        )
        == "kill"
    )
    # flat + blocked + gpu + at-ceiling -> extend.
    assert (
        _decide_plateau_verdict(
            growth_gb=0.0,
            spin=False,
            stack_class="gpu",
            at_ceiling=True,
            sample_failed_once=False,
        )
        == "extend"
    )
    # flat + blocked + gpu + NOT at-ceiling -> kill.
    assert (
        _decide_plateau_verdict(
            growth_gb=0.0,
            spin=False,
            stack_class="gpu",
            at_ceiling=False,
            sample_failed_once=False,
        )
        == "kill"
    )
    # flat + native-setup -> extend (existing path).
    assert (
        _decide_plateau_verdict(
            growth_gb=0.0,
            spin=False,
            stack_class="native-setup",
            at_ceiling=False,
            sample_failed_once=False,
        )
        == "extend"
    )
    # flat + unknown -> kill.
    assert (
        _decide_plateau_verdict(
            growth_gb=0.0,
            spin=False,
            stack_class="unknown",
            at_ceiling=False,
            sample_failed_once=False,
        )
        == "kill"
    )
    # classifier/sample failure -> extend once, then kill on repeat.
    assert (
        _decide_plateau_verdict(
            growth_gb=0.0,
            spin=None,
            stack_class=None,
            at_ceiling=False,
            sample_failed_once=False,
        )
        == "failure-extend"
    )
    assert (
        _decide_plateau_verdict(
            growth_gb=0.0,
            spin=None,
            stack_class=None,
            at_ceiling=False,
            sample_failed_once=True,
        )
        == "kill"
    )


# ── Golden fixtures (real captures) ──────────────────────────────────────


@pytest.mark.skipif(
    not _INCIDENT_DUMP.exists(), reason="golden incident dump not present"
)
def test_incident_dump_classifies_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The healthy at-ceiling prefill dump (2026-10-04) must classify GPU:
    main thread mlx::core::eval -> eval_impl -> Scheduler::wait_for_one /
    MetalAllocator, workers blocked, AGXMetal*/IOGPU images present."""
    stdout = _INCIDENT_DUMP.read_text(encoding="utf-8", errors="replace")
    _install_stack_sample(monkeypatch, {47545: stdout})
    assert _sample_stack_class(47545, 2) == "gpu"


@pytest.mark.skipif(
    not _INCIDENT_DUMP.exists(), reason="golden incident dump not present"
)
def test_incident_dump_is_not_native_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The existing native-setup signature MUST NOT match the gpu dump (it
    would otherwise be misread as a peer-wait)."""
    stdout = _INCIDENT_DUMP.read_text(encoding="utf-8", errors="replace")
    native = [
        s
        for s in supervisor_module._NATIVE_BLOCKED_SYMBOLS  # pyright: ignore[reportPrivateUsage]
        if s in stdout
    ]
    assert len(native) < 2
    # And the full classifier agrees.
    _install_stack_sample(monkeypatch, {47545: stdout})
    assert _sample_stack_class(47545, 2) != "native-setup"


def test_synthetic_spin_dump_classifies_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real busy-loop process (CPU burner) sampled by /usr/bin/sample has
    neither mlx/Metal images nor the native-setup chain => "unknown". The
    spin axis, not the stack, is what makes it kill-class (see the async
    decision-table test)."""
    assert _SPIN_DUMP.exists(), "synthetic spin fixture missing"
    stdout = _SPIN_DUMP.read_text(encoding="utf-8", errors="replace")
    _install_stack_sample(monkeypatch, {4242: stdout})
    assert _sample_stack_class(4242, 2) == "unknown"


def test_sample_stack_class_failure_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sample subprocess failure is a CLASSIFIER FAILURE (None), distinct
    from a successful "unknown" sample — the decision table treats them
    differently (extend once vs kill)."""

    def _raise(cmd: list[str], **_k: object) -> subprocess.CompletedProcess[str]:
        raise OSError("no /usr/bin/sample")

    monkeypatch.setattr(supervisor_module.subprocess, "run", _raise)
    assert _sample_stack_class(1, 2) is None


def test_sample_stack_class_empty_output_is_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stack_sample(monkeypatch, {1: ""})
    assert _sample_stack_class(1, 2) is None


def test_stack_mode_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    """EXO_RUNNER_HANG_STACK_MODE is read at module import; anything outside
    off|shadow|arm falls back to shadow (the safe default)."""
    cases = {
        "off": "off",
        "shadow": "shadow",
        "arm": "arm",
        "ARM": "arm",
        " shadow ": "shadow",
        "bogus": "shadow",
        "": "shadow",
    }
    for raw, expected in cases.items():
        monkeypatch.setenv("EXO_RUNNER_HANG_STACK_MODE", raw)
        _ = importlib.reload(supervisor_module)
        assert expected == supervisor_module.HANG_STACK_MODE, (
            raw,
            supervisor_module.HANG_STACK_MODE,
        )
    # Unset -> default shadow.
    monkeypatch.delenv("EXO_RUNNER_HANG_STACK_MODE", raising=False)
    _ = importlib.reload(supervisor_module)
    assert supervisor_module.HANG_STACK_MODE == "shadow"


# ── Full decision matrix against a real RunnerSupervisor ─────────────────


class _AliveProcess:
    """Fake AsyncProcess reporting alive with an obviously-fake PID (see the
    sibling hang-probe test file for why a fake PID is a safety guard)."""

    _FAKE_PID = 9999999

    def __init__(self) -> None:
        rx1, _ = channel[bytes]()
        rx2, _ = channel[bytes]()
        self.stdout = rx1
        self.stderr = rx2

    exitcode = None

    def is_alive(self) -> bool:
        return True

    @property
    def pid(self) -> int:
        return self._FAKE_PID


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def now(self) -> float:
        return self.t

    def advance_to(self, t: float) -> None:
        assert t >= self.t
        self.t = t


def _install_stack(monkeypatch: pytest.MonkeyPatch, cls: str | None) -> None:
    """Install a typed fake for the stack classifier returning `cls`."""

    def _stack(_pid: int, _duration_s: int = 2) -> str | None:
        return cls

    monkeypatch.setattr(supervisor_module, "_sample_stack_class", _stack)


def _install_cpu(monkeypatch: pytest.MonkeyPatch, seq: list[float] | float) -> None:
    """Install a typed fake ps-cputime reader. A single float repeats; a list
    is consumed in order (clamping at the last value)."""
    values = seq if isinstance(seq, list) else [seq]
    state = {"n": 0}

    def _ps(_pid: int) -> float:
        idx = min(state["n"], len(values) - 1)
        state["n"] += 1
        return values[idx]

    monkeypatch.setattr(supervisor_module, "_read_ps_cputime_seconds", _ps)


def _install_memsize(monkeypatch: pytest.MonkeyPatch, nbytes: int | None) -> None:
    def _memsize() -> int | None:
        return nbytes

    monkeypatch.setattr(supervisor_module, "_host_memsize_bytes", _memsize)


def _install_footprint_seq(monkeypatch: pytest.MonkeyPatch, seq: list[float]) -> None:
    state = {"n": 0}

    def _fp(_pid: int, _duration_s: int) -> float:
        idx = min(state["n"], len(seq) - 1)
        state["n"] += 1
        return seq[idx]

    monkeypatch.setattr(supervisor_module, "_sample_physical_footprint_gb", _fp)


async def _make_supervisor() -> RunnerSupervisor:
    event_sender, _event_receiver = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event | RunnerTerminationError]()

    bound_instance: BoundInstance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-stackguard"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-stackguard"),
        node_id=NodeId("node-stackguard"),
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
        task_id=TaskId("task-stackguard"),
        instance_id=bound_instance.instance.instance_id,
        command_id=CommandId("cmd-stackguard"),
        task_params=TextGenerationTaskParams(
            model=bound_instance.bound_shard.model_card.model_id,
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
            stream=True,
        ),
    )
    supervisor.in_progress[task.task_id] = task
    return supervisor


def _neutralize_kill(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace the real kill side effects; count kills instead."""
    kills: list[int] = []

    def _fake_subprocess_run(
        *_a: object, **_k: object
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            args=(), returncode=0, stdout=b"", stderr=b""
        )

    def _fake_kill(pid: int, _sig: int) -> None:
        kills.append(pid)

    def _not_stopped(_pid: int) -> bool:
        return False

    monkeypatch.setattr(supervisor_module.subprocess, "run", _fake_subprocess_run)
    monkeypatch.setattr(supervisor_module.os, "kill", _fake_kill)
    monkeypatch.setattr(
        supervisor_module, "_process_is_stopped_or_traced", _not_stopped
    )
    return kills


def _set_mode(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.setattr(supervisor_module, "HANG_STACK_MODE", mode)


async def _drive_to_plateau(
    monkeypatch: pytest.MonkeyPatch,
    supervisor: RunnerSupervisor,
    clock: _FakeClock,
    footprint: float = 50.0,
) -> None:
    """t=46 baseline (extends once), then t=66: footprint flat."""
    _install_footprint_seq(monkeypatch, [footprint])
    supervisor._last_event_monotonic = 0.0  # pyright: ignore[reportPrivateUsage]

    clock.advance_to(46.0)
    await supervisor._check_hang()  # pyright: ignore[reportPrivateUsage]
    assert not supervisor._hang_killed  # pyright: ignore[reportPrivateUsage]
    clock.advance_to(66.0)
    await supervisor._check_hang()  # pyright: ignore[reportPrivateUsage]


@pytest.mark.anyio
async def test_shadow_plateau_kills_like_today(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SHADOW changes no kill behavior: a plateau with unknown stack (no
    native signature) still kills, exactly as today."""
    supervisor = await _make_supervisor()
    clock = _FakeClock()
    monkeypatch.setattr(supervisor_module.time, "monotonic", clock.now)
    kills = _neutralize_kill(monkeypatch)
    _set_mode(monkeypatch, "shadow")
    _install_stack(monkeypatch, "unknown")
    _install_cpu(monkeypatch, 0.0)
    _install_memsize(monkeypatch, 137 * 1024**3)

    await _drive_to_plateau(monkeypatch, supervisor, clock)

    assert supervisor._hang_killed  # pyright: ignore[reportPrivateUsage]
    assert kills, "shadow must still SIGKILL an unknown flat plateau"


@pytest.mark.anyio
async def test_shadow_preserves_native_setup_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SHADOW preserves today's kill path: flat + native-setup still EXTENDS
    (this is the incumbent behavior shadow is required to leave unchanged)."""
    supervisor = await _make_supervisor()
    clock = _FakeClock()
    monkeypatch.setattr(supervisor_module.time, "monotonic", clock.now)
    kills = _neutralize_kill(monkeypatch)
    _set_mode(monkeypatch, "shadow")
    _install_stack(monkeypatch, "native-setup")
    _install_cpu(monkeypatch, 0.0)
    _install_memsize(monkeypatch, 137 * 1024**3)

    await _drive_to_plateau(monkeypatch, supervisor, clock)

    assert not supervisor._hang_killed  # pyright: ignore[reportPrivateUsage]
    assert not kills
    assert supervisor._hang_probe_extensions_used == 2  # pyright: ignore[reportPrivateUsage]


@pytest.mark.anyio
async def test_off_plateau_kills_without_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OFF skips sampling entirely: a flat plateau kills, and neither the
    stack classifier nor ps is consulted."""
    supervisor = await _make_supervisor()
    clock = _FakeClock()
    monkeypatch.setattr(supervisor_module.time, "monotonic", clock.now)
    _neutralize_kill(monkeypatch)
    _set_mode(monkeypatch, "off")

    called = {"stack": 0, "ps": 0}

    def _stack(*_a: object, **_k: object) -> str:
        called["stack"] += 1
        return "gpu"

    def _ps(_pid: int) -> float:
        called["ps"] += 1
        return 0.0

    monkeypatch.setattr(supervisor_module, "_sample_stack_class", _stack)
    monkeypatch.setattr(supervisor_module, "_read_ps_cputime_seconds", _ps)

    await _drive_to_plateau(monkeypatch, supervisor, clock)

    assert supervisor._hang_killed  # pyright: ignore[reportPrivateUsage]
    assert called == {"stack": 0, "ps": 0}, "off must not sample at all"


@pytest.mark.anyio
async def test_arm_spin_kills_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ARM: flat + SPIN kills fast, even with a gpu stack at the ceiling."""
    supervisor = await _make_supervisor()
    clock = _FakeClock()
    monkeypatch.setattr(supervisor_module.time, "monotonic", clock.now)
    kills = _neutralize_kill(monkeypatch)
    _set_mode(monkeypatch, "arm")
    # Baseline at t=46 reads cputime 0.0; plateau at t=66 reads 15.0 over a
    # 20s wall -> 0.75 >= 0.5 => spin.
    seq = [0.0, 15.0]
    calls = {"n": 0}

    def _ps(_pid: int) -> float:
        idx = min(calls["n"], len(seq) - 1)
        calls["n"] += 1
        return seq[idx]

    monkeypatch.setattr(supervisor_module, "_read_ps_cputime_seconds", _ps)
    _install_stack(monkeypatch, "gpu")
    _install_memsize(monkeypatch, 137 * 1024**3)

    await _drive_to_plateau(monkeypatch, supervisor, clock, footprint=136.5)

    assert supervisor._hang_killed  # pyright: ignore[reportPrivateUsage]
    assert kills


@pytest.mark.anyio
async def test_arm_gpu_at_ceiling_extends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ARM: flat + blocked + gpu + at-ceiling extends (the incident case)."""
    supervisor = await _make_supervisor()
    clock = _FakeClock()
    monkeypatch.setattr(supervisor_module.time, "monotonic", clock.now)
    kills = _neutralize_kill(monkeypatch)
    _set_mode(monkeypatch, "arm")
    _install_cpu(monkeypatch, 0.0)
    _install_stack(monkeypatch, "gpu")
    _install_memsize(monkeypatch, 137 * 1024**3)
    _install_footprint_seq(monkeypatch, [136.5])

    await _drive_to_plateau(monkeypatch, supervisor, clock, footprint=136.5)

    assert not supervisor._hang_killed  # pyright: ignore[reportPrivateUsage]
    assert not kills
    assert supervisor._hang_probe_extensions_used == 2  # pyright: ignore[reportPrivateUsage]


@pytest.mark.anyio
async def test_arm_gpu_not_at_ceiling_kills(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ARM: flat + gpu but NOT at the ceiling kills (conservative first ship)."""
    supervisor = await _make_supervisor()
    clock = _FakeClock()
    monkeypatch.setattr(supervisor_module.time, "monotonic", clock.now)
    kills = _neutralize_kill(monkeypatch)
    _set_mode(monkeypatch, "arm")
    _install_cpu(monkeypatch, 0.0)
    _install_stack(monkeypatch, "gpu")
    _install_memsize(monkeypatch, 137 * 1024**3)
    _install_footprint_seq(monkeypatch, [50.0])

    await _drive_to_plateau(monkeypatch, supervisor, clock)

    assert supervisor._hang_killed  # pyright: ignore[reportPrivateUsage]
    assert kills


@pytest.mark.anyio
async def test_arm_native_setup_extends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ARM: flat + native-setup extends (existing path, now table-driven)."""
    supervisor = await _make_supervisor()
    clock = _FakeClock()
    monkeypatch.setattr(supervisor_module.time, "monotonic", clock.now)
    kills = _neutralize_kill(monkeypatch)
    _set_mode(monkeypatch, "arm")
    _install_cpu(monkeypatch, 0.0)
    _install_stack(monkeypatch, "native-setup")
    _install_memsize(monkeypatch, 137 * 1024**3)

    await _drive_to_plateau(monkeypatch, supervisor, clock)

    assert not supervisor._hang_killed  # pyright: ignore[reportPrivateUsage]
    assert not kills


@pytest.mark.anyio
async def test_arm_sample_failure_extends_once_then_kills(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A persistent classifier/sample failure gets exactly ONE extension,
    then kills — bounded by the shared budget, never unbounded."""
    supervisor = await _make_supervisor()
    clock = _FakeClock()
    monkeypatch.setattr(supervisor_module.time, "monotonic", clock.now)
    kills = _neutralize_kill(monkeypatch)
    _set_mode(monkeypatch, "arm")
    _install_cpu(monkeypatch, 0.0)
    # The classifier always fails (None) -> failure-extend, then kill.
    _install_stack(monkeypatch, None)
    _install_memsize(monkeypatch, 137 * 1024**3)

    await _drive_to_plateau(monkeypatch, supervisor, clock)
    # t=66: first failure -> one extension, still alive.
    assert not supervisor._hang_killed  # pyright: ignore[reportPrivateUsage]
    assert supervisor._hang_probe_sample_failed_once  # pyright: ignore[reportPrivateUsage]
    assert supervisor._hang_probe_extensions_used == 2  # pyright: ignore[reportPrivateUsage]

    # Next probe tick: failure repeats -> kill.
    clock.advance_to(86.0)
    await supervisor._check_hang()  # pyright: ignore[reportPrivateUsage]
    assert supervisor._hang_killed  # pyright: ignore[reportPrivateUsage]
    assert kills


@pytest.mark.anyio
async def test_budget_monotonic_across_reclassifications(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SAME monotonic budget spans every extend reason: a gpu+ceiling
    extension must not reset the counter when the next tick reclassifies to
    native-setup. Budget is capped at 2 for a fast test."""
    supervisor = await _make_supervisor()
    clock = _FakeClock()
    monkeypatch.setattr(supervisor_module.time, "monotonic", clock.now)
    kills = _neutralize_kill(monkeypatch)
    _set_mode(monkeypatch, "arm")
    monkeypatch.setattr(supervisor_module, "HANG_PROBE_MAX_EXTENSIONS", 2)
    _install_cpu(monkeypatch, 0.0)
    _install_memsize(monkeypatch, 137 * 1024**3)

    classes = ["gpu", "native-setup", "native-setup"]
    seen = {"n": 0}

    def _stack(_pid: int, _d: int = 2) -> str:
        idx = min(seen["n"], len(classes) - 1)
        seen["n"] += 1
        return classes[idx]

    monkeypatch.setattr(supervisor_module, "_sample_stack_class", _stack)

    await _drive_to_plateau(monkeypatch, supervisor, clock, footprint=136.5)
    # gpu+ceiling extended (budget 2/2 now).
    assert not supervisor._hang_killed  # pyright: ignore[reportPrivateUsage]
    assert supervisor._hang_probe_extensions_used == 2  # pyright: ignore[reportPrivateUsage]

    # Even though the next tick reclassifies to native-setup (which normally
    # extends), the budget is already spent -> kill. Reclassification did not
    # reset it.
    clock.advance_to(86.0)
    await supervisor._check_hang()  # pyright: ignore[reportPrivateUsage]
    assert supervisor._hang_killed  # pyright: ignore[reportPrivateUsage]
    assert kills


@pytest.mark.anyio
async def test_growth_path_extend_unchanged_in_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The growth path is untouched by the guard: a growing footprint still
    extends (and never consults the classifier)."""
    supervisor = await _make_supervisor()
    clock = _FakeClock()
    monkeypatch.setattr(supervisor_module.time, "monotonic", clock.now)
    kills = _neutralize_kill(monkeypatch)
    _set_mode(monkeypatch, "arm")
    seq = [50.0, 51.0]  # +1GB >> 0.25 threshold
    seen = {"n": 0}

    def _fp(_pid: int, _d: int) -> float:
        idx = min(seen["n"], len(seq) - 1)
        seen["n"] += 1
        return seq[idx]

    consulted = {"n": 0}

    def _stack(_pid: int, _d: int = 2) -> str:
        consulted["n"] += 1
        return "unknown"

    monkeypatch.setattr(supervisor_module, "_sample_physical_footprint_gb", _fp)
    monkeypatch.setattr(supervisor_module, "_sample_stack_class", _stack)

    supervisor._last_event_monotonic = 0.0  # pyright: ignore[reportPrivateUsage]
    clock.advance_to(46.0)
    await supervisor._check_hang()  # pyright: ignore[reportPrivateUsage]
    clock.advance_to(66.0)
    await supervisor._check_hang()  # pyright: ignore[reportPrivateUsage]

    assert not supervisor._hang_killed  # pyright: ignore[reportPrivateUsage]
    assert not kills
    assert consulted["n"] == 0, "growth must not consult the classifier"
