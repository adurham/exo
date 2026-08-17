"""Domain probes built on the Phase 1 trusted-measurement core (Phase 2).

Each probe in this package is a *scaffold*: every piece of logic that can be
exercised without the live cluster is real, typed and regression-tested, and
every piece that genuinely needs live hardware (an HTTP call to a runner, a
jaccl log stream, a stack sample of a wedged process) sits behind an injectable
`Protocol` with a fake implementation used by the tests.

The split is deliberate and is documented per module under the headings
"REAL TODAY" and "INTERFACE STUB". A future task wires real implementations to
the stub protocols; nothing in this package needs to change for that.
"""

from __future__ import annotations

from trusted_measurement.probes.decode_probe import (
    DecodeProbeConfig,
    DecodeStreamSample,
    DiagnosticAction,
    RecordingDiagnosticAction,
    StallEvent,
    StallWatchdog,
    WatchdogPolicy,
    build_decode_record,
    decode_latencies_milliseconds,
    gaps_from_timestamps,
)
from trusted_measurement.probes.jaccl_probe import (
    JacclLogSource,
    JacclTransportSample,
    RoundTripHistogram,
    StaticJacclLogSource,
    build_jaccl_record,
    jaccl_arm_environments,
    parse_jaccl_transport_sample,
    summarise_samples,
)
from trusted_measurement.probes.prefill_probe import (
    PrefillClient,
    PrefillProbeConfig,
    PrefillPrompt,
    PrefillResponse,
    build_prefill_prompt,
    build_prefill_record,
    prefill_throughput_tokens_per_second,
)

__all__ = [
    "DecodeProbeConfig",
    "DecodeStreamSample",
    "DiagnosticAction",
    "JacclLogSource",
    "JacclTransportSample",
    "PrefillClient",
    "PrefillProbeConfig",
    "PrefillPrompt",
    "PrefillResponse",
    "RecordingDiagnosticAction",
    "RoundTripHistogram",
    "StallEvent",
    "StallWatchdog",
    "StaticJacclLogSource",
    "WatchdogPolicy",
    "build_decode_record",
    "build_jaccl_record",
    "build_prefill_prompt",
    "build_prefill_record",
    "decode_latencies_milliseconds",
    "gaps_from_timestamps",
    "jaccl_arm_environments",
    "parse_jaccl_transport_sample",
    "prefill_throughput_tokens_per_second",
    "summarise_samples",
]
