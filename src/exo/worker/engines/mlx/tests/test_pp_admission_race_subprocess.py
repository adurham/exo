# pyright: reportAny=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Real 2-PROCESS, GENUINELY-INDEPENDENT-EVENT-LOOP regression test for
the Phase-1 batched-decode admission race documented in
``docs/batched-decode-n2-admission-handoff-2026-08-05.md`` (Concrete
next step #2) and ``docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md``
Section 15 ("The N=2 concurrent-admission race").

WHY A NEW TEST RATHER THAN EXTENDING THE EXISTING ONE (quoting the
handoff doc's own statement of the gap): the existing subprocess
harness ``test_pp_batched_decode_glue_subprocess.py`` "drives BOTH
ranks' glue.tick()/glue.enqueue_admission()/glue.stage_local_cache()
calls explicitly, in lockstep, from ONE test driver script. It never
exercises two genuinely independent runner.py-equivalent event loops
polling their own queues on their own schedule -- so it structurally
cannot reproduce a race that only exists because of that
independence."

This test keeps that harness's PROVEN infrastructure verbatim (two
real OS subprocesses, real ``mx.distributed.init(backend="ring")``
transport over a 2-entry hostfile on two genuinely free localhost
ports, JSON result files read and asserted by the parent) and changes
ONLY what each subprocess's main loop does: each rank now runs its own
``runner.py``-shaped loop (see ``handle_generation_tasks()``), draining
its own local work queue and calling ``tick()`` on its OWN
independently-jittered schedule, with NO cross-rank synchronization on
WHEN it makes either decision.

WHAT THE TEST ASSERTS AND WHY THAT IS THE RIGHT ASSERTION:

Both ranks perform the SAME total work -- exactly one real
single-request prefill through the REAL Phase-0.5 metaframe layers,
and exactly ``_ITERATIONS - 1`` real ``tick()`` calls. Only the
ORDER differs, and only because each rank chose independently. Under a
CORRECT design (the handoff doc's next-step #1: an in-band,
rank-0-decides / rank-1-reacts admission signal folded into the
existing decode-step wire traffic) that ordering difference must be
harmless, because rank 1 would never independently decide to switch
from decode-mode collectives to prefill-mode collectives.

Under TODAY'S (unfixed) code there is no such signal, so whenever the
two ranks' independent schedules disagree about which iteration is the
prefill iteration, rank 0 issues one wire shape while rank 1 issues
another:

  * prefill  -> ``pp_metaframe``'s 6-int32 metaframe header followed by
    a ``batch_axis=0`` activation tensor
    (``MetaFramedPipelineFirstLayer``/``LastLayer``, reached through
    the ``Batched*`` subclasses' documented outside-``batch_step_scope``
    fallback -- i.e. exactly how a real runner reaches prefill once
    batched-decode layers are installed at model-load time; design doc
    Section 15, Attempt 1).
  * decode   -> ``pp_scheduler_wire``'s 5-int32 control header
    (``StepMessage``) followed by a ``batch_axis=1`` metaframe
    (``Rank0BatchedDecodeGlue.tick()`` /
    ``Rank1BatchedDecodeGlue.tick()``).

Mismatched shapes on the same link is precisely the local, observable
equivalent of the real cluster's
``[jaccl] reliable_all_reduce_v2 deadline``: either a protocol-level
fail-stop (``SchedulerWireProtocolError`` / a metaframe version or
``batch_axis`` mismatch) or an outright hang until the subprocess
timeout fires. Both are FAILURES of this test, and both are the race.

This test WAS marked xfail(strict=False) while the admission race was
unfixed -- see docs/batched-decode-n2-admission-handoff-2026-08-05.md
and design doc Section 15 for the pre-fix behavior this test used to
reproduce (a real jaccl-deadline-equivalent local desync: mismatched
wire shapes, [Event::wait] timeouts, or SchedulerWireProtocolError
version mismatches, non-deterministically depending on which seed's
independent per-rank schedule happened to diverge). The marker is now
REMOVED (2026-08-06): after the PrefillMessage-based in-band admission
signal shipped (pp_scheduler_wire.py MSG_KIND_PREFILL,
Rank0BatchedDecodeGlue.enqueue_prefill()/tick()-returns-PrefillGrant,
Rank1BatchedDecodeGlue's matching reactive branch, and
ExoBatchGenerator's submit()/step() integration in batch_generate.py),
this test genuinely XPASSED across all 5 independent seeds (verified
real run: 1 xpassed in 15.27s, then re-verified as a plain PASSED with
the marker removed) -- the fix closes the race under real,
genuinely-independent per-rank scheduling pressure, not just in a
lockstep test. This test now serves as the PERMANENT regression gate
for the fix: if a future change reopens the race, this test fails for
real (no longer an expected XFAIL to hide behind).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_WORKER_SCRIPT = str(Path(__file__).parent / "_pp_admission_race_subprocess_worker.py")
_PYTHON = sys.executable

# Deliberately generous relative to the tiny model's real work: a
# genuine deadlock is a HANG, and this timeout is what converts that
# hang into a reportable failure rather than a stuck CI job.
_TIMEOUT_SECONDS = 90.0

# Independent seeds tried per test invocation. Each seed produces a
# different pair of per-rank scheduling RNGs, so a seed where both
# ranks happen to agree proves nothing -- several are needed before
# "no divergence observed" is a meaningful statement.
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


def _run_admission_race_round(seed: int) -> tuple[dict[str, object], ...]:
    """Spawn the two genuinely-independent rank processes for one seed
    and return both result dicts. Raises on timeout (a real deadlock)
    or on a missing result file (a hard crash) -- both of which ARE the
    race, not infrastructure noise."""
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
                f"{_TIMEOUT_SECONDS}s. This is the admission race: the two "
                f"independent loops issued mismatched wire operations and "
                f"blocked on each other. Output:\n"
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
def test_independent_per_rank_event_loops_do_not_desynchronize_the_wire() -> None:
    """THE regression gate for the admission race (marker history: see
    module docstring's "This test WAS marked xfail..." paragraph).

    Runs, for each of several independent seeds, two real OS processes
    over real MLX ring transport, each executing its OWN
    ``runner.py``-shaped event loop with its OWN jittered schedule and
    its OWN local work queue -- never synchronized with the peer on
    WHEN it drains the queue (prefill) versus WHEN it advances decode
    (``tick()``). Both ranks do identical TOTAL work; only the order is
    free to diverge.

    Passes only if every seed completes cleanly on both ranks. Any
    hang, crash, or protocol error is the race reproducing.
    """
    failures: list[str] = []
    for seed in _SEEDS:
        try:
            rank0_result, rank1_result = _run_admission_race_round(seed)
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
                f"tokens -- the scenario did not actually run, so this "
                f"round proves nothing about the race"
            )

    if failures:
        raise AssertionError(
            "Admission race reproduced by genuinely-independent per-rank "
            "event loops (this is the documented, UNFIXED gap -- see this "
            "module's docstring):\n\n" + "\n\n".join(failures)
        )
