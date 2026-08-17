"""Runtime-mode markers: config implying a path is not proof it executed."""

from __future__ import annotations

import threading

from trusted_measurement.runtime_mode import RuntimeModeRecorder
from trusted_measurement.tests.builders import valid_record


def test_record_invalid_when_required_marker_never_emitted() -> None:
    """Config says TP; nothing emitted a marker; the record must not validate."""
    record = valid_record(
        runtime_mode_markers=(),
        required_runtime_modes=("tp_allreduce",),
    )
    violations = record.validate_envelope()
    assert any(
        "emitted no in-process marker" in violation.reason for violation in violations
    )
    assert not record.is_trusted


def test_wrong_path_marker_does_not_satisfy_requirement() -> None:
    """Chunked-prefill required, plain-prefill executed: still red."""
    recorder = RuntimeModeRecorder()
    recorder.emit("plain_prefill", detail="fallback path")
    record = valid_record(
        runtime_mode_markers=recorder.markers(),
        required_runtime_modes=("chunked_prefill",),
    )
    violations = record.validate_envelope()
    assert any("chunked_prefill" in violation.reason for violation in violations)


def test_emitted_marker_satisfies_requirement() -> None:
    recorder = RuntimeModeRecorder()
    recorder.emit("chunked_prefill", detail="4 chunks", count=4)
    record = valid_record(
        runtime_mode_markers=recorder.markers(),
        required_runtime_modes=("chunked_prefill",),
    )
    assert record.is_trusted


def test_marker_counts_fold_and_carry_valid_proofs() -> None:
    recorder = RuntimeModeRecorder()
    for _ in range(61):
        recorder.emit("tp_allreduce", detail="2 ops/layer", count=2)
    markers = recorder.markers()
    assert len(markers) == 1
    assert markers[0].observed_count == 122
    assert markers[0].proof_is_valid()


def test_recorder_is_thread_safe() -> None:
    recorder = RuntimeModeRecorder()

    def worker() -> None:
        for _ in range(500):
            recorder.emit("tp_allreduce")

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    markers = recorder.markers()
    assert markers[0].observed_count == 2000


def test_reset_clears_observations() -> None:
    recorder = RuntimeModeRecorder()
    recorder.emit("tp_allreduce")
    recorder.reset()
    assert recorder.markers() == ()
    assert recorder.observed_modes() == frozenset()
