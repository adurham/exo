from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.common import CommandId, NodeId, SessionId, SystemId
from exo.shared.types.events import ChunkGenerated, LocalForwarderEvent
from exo.worker.runner.diagnostics import (
    KnownRunnerDiagnostic,
    RunnerMetalGpuTimeout,
    RunnerRingSocketReceivingError,
)


def test_error_chunk_with_diagnostics_roundtrip() -> None:
    """LocalForwarderEvent → JSON → LocalForwarderEvent with runner diagnostics.

    Regression test: TaggedModel's wrap-validator re-validates the parsed
    JSON payload in Python mode, where a strict ``tuple[str, ...]`` field
    rejected the JSON array (a ``list``). Every node receiving an event
    that carried a diagnostic (e.g. the GPU-timeout ErrorChunk emitted
    when a degenerating stream is killed) then died on an unhandled
    ValidationError in the router — taking the whole cluster down.
    """
    diagnostics: list[KnownRunnerDiagnostic] = [
        RunnerMetalGpuTimeout(
            message=(
                "[METAL] Command buffer execution failed: Caused GPU Timeout "
                "Error (00000002:kIOGPUCommandBufferCallbackErrorTimeout)"
            ),
            evidence=("stderr line 1", "stderr line 2"),
        ),
        RunnerRingSocketReceivingError(
            message="[ring] Receiving from socket 7 failed with errno 54",
            evidence=(),
            error_number=54,
            error_name="ECONNRESET",
            error_description="Connection reset by peer",
        ),
    ]
    event = LocalForwarderEvent(
        origin_idx=1,
        origin=SystemId("test-system"),
        session=SessionId(master_node_id=NodeId("node-a"), election_clock=1),
        event=ChunkGenerated(
            command_id=CommandId("02684512-c859-4ffa-8771-2aaaaaaaaaaa"),
            chunk=ErrorChunk(
                model=ModelId("mlx-community/DeepSeek-V4-Flash"),
                error_message="Runner shutdown before completing command",
                diagnostics=diagnostics,
            ),
        ),
    )

    restored = LocalForwarderEvent.model_validate_json(event.model_dump_json())

    assert restored == event
    assert restored.model_dump_json() == event.model_dump_json()
