# pyright: reportAny=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Real 2-PROCESS regression test for
``exchange_prefill_peer_layer_count`` (2026-08-07 real production
incident): the original point-to-point send/recv_like implementation
deadlocked on real 2-node hardware because both ranks call this
function with the SAME source order (send then recv), so both post a
blocking send before either posts a matching recv -- confirmed via
real cluster logs, both ranks hit jaccl's own 15-second drain deadline
at the identical millisecond. The caught timeout's in-flight send then
corrupted a LATER, unrelated recv_metaframe call on the peer rank
(``MetaFrame protocol version mismatch: received 22, this build
expects 3`` -- 22 being the sending rank's real layer count, not
garbage).

HONEST LIMITATION (confirmed empirically, not assumed): this test does
NOT reproduce the original deadlock -- verified by running it against
a git-stashed reversion back to the point-to-point send/recv
implementation, which ALSO passes cleanly under this harness's
localhost ring transport. The ring backend's TCP-loopback send() does
not share jaccl's specific RDMA reliable-send drain-deadline behavior,
so the send/send race this incident's root cause depends on genuinely
does not manifest here. This test therefore does NOT serve as the
regression gate for the original incident (only a real 2-node jaccl
cluster run can do that -- see the design doc's real-hardware
verification entry for this fix). What this test DOES prove: the
scatter+``all_sum`` mechanism is logically correct under real,
genuinely-independent-process MLX distributed transport -- two real OS
processes, each running the exact production function with realistic
uneven layer counts (22 vs 21, matching the real cluster's own
confirmed asymmetric split), each correctly learning the OTHER rank's
value, not its own value and not corrupted/misaligned data.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_WORKER_SCRIPT = str(
    Path(__file__).parent / "_pp_exchange_prefill_peer_layer_count_subprocess_worker.py"
)
_PYTHON = sys.executable
_TIMEOUT_SECONDS = 30.0


def _find_two_free_ports() -> tuple[int, int]:
    import socket

    ports: list[int] = []
    socks: list[socket.socket] = []
    try:
        for _ in range(2):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", 0))
            ports.append(s.getsockname()[1])
            socks.append(s)
    finally:
        for s in socks:
            s.close()
    return ports[0], ports[1]


@pytest.mark.slow
def test_exchange_prefill_peer_layer_count_two_real_independent_processes() -> None:
    """Real 2-process, genuinely-independent-event-loop proof that the
    scatter+all_sum fix correctly and symmetrically exchanges each
    rank's real (uneven) local layer count with no deadlock -- the
    exact scenario (both ranks calling this function with identical
    source order, real transport) that deadlocked the original
    point-to-point implementation on real 2-node hardware.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        hostfile_path = os.path.join(tmpdir, "hostfile.json")
        port0, port1 = _find_two_free_ports()
        with open(hostfile_path, "w") as f:
            f.write(json.dumps([[f"127.0.0.1:{port0}"], [f"127.0.0.1:{port1}"]]))

        out_paths = [
            os.path.join(tmpdir, "rank0.json"),
            os.path.join(tmpdir, "rank1.json"),
        ]
        # Real, confirmed-asymmetric layer counts from the actual
        # production cluster (43-layer DSv4-Flash, 2 nodes, allocated
        # by memory weight not an even split) -- not arbitrary test
        # values.
        layer_counts = [22, 21]
        procs: list[subprocess.Popen[bytes]] = []
        for rank in (0, 1):
            env = dict(os.environ)
            env["MLX_HOSTFILE"] = hostfile_path
            env["MLX_RANK"] = str(rank)
            procs.append(
                subprocess.Popen(
                    [
                        _PYTHON,
                        _WORKER_SCRIPT,
                        str(rank),
                        out_paths[rank],
                        str(layer_counts[rank]),
                    ],
                    cwd=str(Path(__file__).parents[6]),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
            )

        outputs: list[bytes] = []
        timed_out = False
        for proc in procs:
            try:
                stdout, _ = proc.communicate(timeout=_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.kill()
                stdout, _ = proc.communicate()
            outputs.append(stdout or b"")
        if timed_out:
            raise RuntimeError(
                f"DEADLOCK -- a rank did not finish within "
                f"{_TIMEOUT_SECONDS}s. This would reproduce the exact "
                f"2026-08-07 real production incident (both ranks "
                f"blocking in send() before either posts a matching "
                f"recv()). Output:\n"
                + b"\n---\n".join(outputs).decode(errors="replace")
            )

        results: list[dict[str, object]] = []
        for rank, (proc, out_path, stdout) in enumerate(
            zip(procs, out_paths, outputs, strict=True)
        ):
            if not os.path.exists(out_path):
                raise RuntimeError(
                    f"rank {rank} never wrote its result file "
                    f"(exit code {proc.returncode}). stdout/stderr:\n"
                    f"{stdout.decode(errors='replace')}"
                )
            with open(out_path) as f:
                results.append(json.load(f))

        rank0_result, rank1_result = results
        for rank, result in ((0, rank0_result), (1, rank1_result)):
            assert result.get("ok"), (
                f"rank={rank} reported failure: {result.get('error')}\n"
                f"{result.get('traceback', '')}"
            )
        # Rank 0 (local=22) must learn rank 1's real value (21), and
        # vice versa -- not its own value, not garbage, not the OTHER
        # protocol's data (the exact corruption mode the original bug
        # produced).
        assert rank0_result.get("peer_layer_count") == 21, (
            f"rank 0 should learn rank 1's real layer count (21), got "
            f"{rank0_result.get('peer_layer_count')!r}"
        )
        assert rank1_result.get("peer_layer_count") == 22, (
            f"rank 1 should learn rank 0's real layer count (22), got "
            f"{rank1_result.get('peer_layer_count')!r}"
        )
