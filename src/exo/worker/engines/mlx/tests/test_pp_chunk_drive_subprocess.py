# pyright: reportAny=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Real 2-PROCESS, GENUINELY-INDEPENDENT-EVENT-LOOP regression test for
Phase 2's chunk-drive live-wiring (2026-08-07): closes the gap
``test_pp_admission_race_subprocess.py``'s own module docstring
identifies for a DIFFERENT hazard (its own N=2 admission race), now
applied to the chunk-drive registration ordering hazards fixed the
same session as this test (see
``docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md``'s
2026-08-07 entries for Hazard 1/Hazard 2/priority-order-guard).

WHY A NEW TEST RATHER THAN EXTENDING THE EXISTING IN-PROCESS ONES:
``test_batch_generate_chunked_prefill_live_wiring.py`` and
``test_pp_batched_decode_glue_chunk_drive.py`` both drive BOTH ranks'
state from a single Python thread. Per this exact codebase's own
hard-won precedent (the N=2 admission race was invisible to a
lockstep 2-process test and only reproduced once a genuinely-
independent-per-rank-event-loop subprocess harness was built --
``test_pp_admission_race_subprocess.py``), a single-driver test
structurally cannot exercise what happens when each rank's decision
to call ``tick()`` -- and therefore each rank's own reactive
``register_prefill_session()`` call for the NEXT chunk boundary --
happens on an independently-jittered schedule with no cross-rank
coordination on WHEN.

WHAT THIS TEST PROVES THE IN-PROCESS TESTS COULD NOT:

1. Hazard 2's "``tick()`` is the only recv site, so a peer cannot
   observe the next chunk's first advance before this rank's own
   registration for it" argument is a STRUCTURAL, code-level
   invariant -- but whether the two ranks' independently-scheduled
   loops actually CONVERGE on driving the same chunk boundary at all
   (not just whether one rank's own registration precedes its own
   next recv) depends on inter-loop scheduling this test's genuinely
   independent per-process jitter can exercise and a single-threaded
   driver cannot.
2. The priority-order guard (no new prefill granted while a chunk-
   drive is active) is stressed under REAL independent scheduling,
   with contention DETERMINISTICALLY injected (a competing request
   enqueued the instant each rank's own local view confirms the
   chunk-drive is active) rather than relying on random jitter alone
   to open the window -- see the worker script's own docstring for
   why probabilistic contention alone would be insufficient.
3. A real deadlock (one rank blocked in a collective the other never
   issues) is a HANG under this harness -- exactly the observable,
   reportable-as-failure signature a single-threaded in-process test
   cannot produce at all, since nothing there ever genuinely blocks
   waiting on a peer process.

Reuses ``test_pp_admission_race_subprocess.py``'s PROVEN
infrastructure verbatim (two real OS subprocesses, real
``mx.distributed.init(backend="ring")`` transport over a 2-entry
hostfile on two genuinely free localhost ports, JSON result files read
and asserted by the parent).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_WORKER_SCRIPT = str(Path(__file__).parent / "_pp_chunk_drive_subprocess_worker.py")
_PYTHON = sys.executable

# Deliberately generous relative to the tiny model's real work: a
# genuine deadlock is a HANG, and this timeout is what converts that
# hang into a reportable failure rather than a stuck CI job.
_TIMEOUT_SECONDS = 90.0

# Independent seeds tried per test invocation -- same count as the
# admission-race test's own precedent (a seed where both ranks happen
# to schedule identically proves nothing; several are needed before
# "no divergence observed" is a meaningful statement).
_SEEDS = (11, 23, 37, 41, 59)


def _find_two_free_ports() -> tuple[int, int]:
    """See ``test_pp_batched_decode_glue_subprocess.py``'s
    identically-named helper for the full rationale (real ports, not
    hardcoded constants, to avoid TIME_WAIT collisions across
    sequential runs)."""
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


def _run_chunk_drive_round(seed: int) -> tuple[dict[str, object], ...]:
    """Spawn the two genuinely-independent rank processes for one seed
    and return both result dicts. Raises on timeout (a real deadlock)
    or on a missing result file (a hard crash) -- both of which ARE a
    real hazard reproducing, not infrastructure noise."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hostfile_path = os.path.join(tmpdir, "hostfile.json")
        port0, port1 = _find_two_free_ports()
        with open(hostfile_path, "w") as f:
            f.write(json.dumps([[f"127.0.0.1:{port0}"], [f"127.0.0.1:{port1}"]]))

        out_paths = [
            os.path.join(tmpdir, "rank0.json"),
            os.path.join(tmpdir, "rank1.json"),
        ]
        procs: list[subprocess.Popen[bytes]] = []
        for rank in (0, 1):
            env = dict(os.environ)
            env["MLX_HOSTFILE"] = hostfile_path
            env["MLX_RANK"] = str(rank)
            procs.append(
                subprocess.Popen(
                    [_PYTHON, _WORKER_SCRIPT, str(rank), out_paths[rank], str(seed)],
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
                f"seed={seed}: DEADLOCK -- a rank did not finish within "
                f"{_TIMEOUT_SECONDS}s. This is a real chunk-drive-ordering "
                f"hazard reproducing: the two independent loops' calls to "
                f"register_prefill_session()/tick() desynced and blocked on "
                f"each other. Output:\n"
                + b"\n---\n".join(outputs).decode(errors="replace")
            )

        results: list[dict[str, object]] = []
        for rank, (proc, out_path, stdout) in enumerate(
            zip(procs, out_paths, outputs, strict=True)
        ):
            if not os.path.exists(out_path):
                raise RuntimeError(
                    f"seed={seed}: rank {rank} never wrote its result file "
                    f"(exit code {proc.returncode}). stdout/stderr:\n"
                    f"{stdout.decode(errors='replace')}"
                )
            with open(out_path) as f:
                results.append(json.load(f))
        return results[0], results[1]


@pytest.mark.slow
def test_independent_per_rank_event_loops_keep_chunk_drive_registration_synchronized() -> (
    None
):
    """THE regression gate for Phase 2's chunk-drive live-wiring
    hazards, under real independent per-process scheduling (see
    module docstring for the full rationale).

    Runs, for each of several independent seeds, two real OS
    processes over real MLX ring transport, each executing its OWN
    event loop with its OWN jittered schedule -- one ordinary
    already-admitted request (A) decodes throughout; a chunked
    request (D) is admitted mid-stream and driven across
    ``_N_CHUNKS`` real chunk boundaries, with each rank registering
    each chunk's session purely reactively to its OWN ``tick()``
    calls; a THIRD request (E) is enqueued the instant each rank's own
    local view confirms D's drive is active, deterministically
    stressing the priority-order guard.

    Passes only if every seed completes cleanly on both ranks with
    zero priority-order-guard violations and zero cross-rank chunk-
    index/advance-sequence desyncs (both of which the real glue's own
    fail-loud ``GlueError``/``RuntimeError`` guards would raise,
    surfacing here as a reported failure, not a silent pass). Any
    hang, crash, or protocol error is a real hazard reproducing.
    """
    failures: list[str] = []
    for seed in _SEEDS:
        try:
            rank0_result, rank1_result = _run_chunk_drive_round(seed)
        except RuntimeError as e:
            failures.append(f"seed={seed}: {e}")
            continue
        for rank, result in ((0, rank0_result), (1, rank1_result)):
            if not result.get("ok"):
                failures.append(
                    f"seed={seed} rank={rank} reported failure: "
                    f"{result.get('error')}\ntrace={result.get('trace')}\n"
                    f"{result.get('traceback', '')}"
                )
        if rank0_result.get("ok") and rank1_result.get("ok"):
            # Both ranks survived -- confirm the loops really did run
            # the full scenario (a silently-empty run would "pass"
            # without ever exercising the seam).
            assert rank0_result.get("tokens_a"), (
                f"seed={seed}: rank 0 completed but produced no decode "
                f"tokens for request A -- the scenario did not actually "
                f"run, so this round proves nothing about the hazards"
            )
            assert rank0_result.get("tokens_e"), (
                f"seed={seed}: rank 0 completed but produced no decode "
                f"tokens for request E -- the priority-order-guard "
                f"contention scenario did not actually complete"
            )

    if failures:
        raise AssertionError(
            "Chunk-drive registration-ordering hazard reproduced by "
            "genuinely-independent per-rank event loops (this is the "
            "hazard the 2026-08-07 live-wiring fix is supposed to close "
            "-- see this module's docstring):\n\n" + "\n\n".join(failures)
        )
