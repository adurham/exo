# pyright: reportAny=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false
"""Real 2-PROCESS correctness test for the batched-decode session --
the genuinely separate-OS-process test the consult review (2026-08-05)
flagged as in-scope even without cluster access, and which incidentally
also gives real DSv4-Flash correctness testing (design doc Risk #1)
a working harness for the first time this session, since MLX's
cross-thread multi-output-@mx.compile bug (warm memory fact 1202)
does not apply across real process boundaries.

Uses MLX's own real `ring` distributed backend directly (no cluster,
no RDMA hardware -- two real localhost TCP connections), NOT the
`mlx.launch` CLI (which shells out via ssh-like plumbing even for
127.0.0.1 and is fiddly with venv/PATH resolution for local dev; see
warm memory fact 1202 for the exact env-var-based invocation this
test uses instead: MLX_HOSTFILE + MLX_RANK per subprocess).

Each rank runs `_pp_subprocess_worker.py` as a genuinely separate
`subprocess.Popen` Python process and writes its result to a JSON
file; this test reads both files after both processes exit and
asserts on the results. Golden reference (llama case) is computed
in-process here for comparison -- the exact same serial-plain-forward
methodology this whole session's suite uses.

Slower and heavier than the rest of this session's tests (real
process spawn + real model construction + real TCP loopback) --
kept as a small, targeted set (not run for every scenario this
session's in-process/threaded suite already covers) specifically for
what ONLY a genuine process boundary can prove: no cross-thread MLX
compile hazards, and no Python-object-sharing shortcuts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.utils
import pytest

_WORKER_SCRIPT = str(Path(__file__).parent / "_pp_subprocess_worker.py")
_PYTHON = sys.executable
_TIMEOUT_SECONDS = 60.0


def _find_two_free_ports() -> tuple[int, int]:
    """Find two genuinely free TCP loopback ports for this test run --
    real ports, not hardcoded constants, since a prior run's ring
    listener can linger in TIME_WAIT and cause the SAME hardcoded
    port to spuriously fail group formation on the very next run
    (confirmed while developing this test: identical hardcoded ports
    across sequential pytest invocations intermittently produced
    ``group.size()==1`` instead of ``2`` -- MLX's ring backend falling
    back to its singleton-group behavior when the expected peer
    connection didn't complete in time)."""
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


def _run_two_rank_subprocess_test(
    model_kind: str, seed: int
) -> tuple[dict[str, object], dict[str, object]]:
    """Launch two real Python subprocesses (ranks 0 and 1) running
    ``_pp_subprocess_worker.py`` over a real MLX ring backend
    (localhost TCP), wait for both to exit, and return their two
    result dicts (order: rank0, rank1)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hostfile_path = os.path.join(tmpdir, "hostfile.json")
        port0, port1 = _find_two_free_ports()
        hostfile_content = json.dumps([[f"127.0.0.1:{port0}"], [f"127.0.0.1:{port1}"]])
        with open(hostfile_path, "w") as f:
            f.write(hostfile_content)

        out_paths = [
            os.path.join(tmpdir, "rank0.json"),
            os.path.join(tmpdir, "rank1.json"),
        ]
        procs: list[subprocess.Popen[bytes]] = []
        for rank in (0, 1):
            env = dict(os.environ)
            env["MLX_HOSTFILE"] = hostfile_path
            env["MLX_RANK"] = str(rank)
            proc = subprocess.Popen(
                [
                    _PYTHON,
                    _WORKER_SCRIPT,
                    str(rank),
                    model_kind,
                    out_paths[rank],
                    str(seed),
                ],
                cwd=str(Path(__file__).parents[6]),  # repo root (src/exo/... -> repo)
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            procs.append(proc)

        outputs: list[bytes] = []
        for proc in procs:
            try:
                stdout, _ = proc.communicate(timeout=_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired as e:
                proc.kill()
                stdout, _ = proc.communicate()
                outputs.append(stdout or b"")
                raise RuntimeError(
                    f"_run_two_rank_subprocess_test: subprocess timed out "
                    f"after {_TIMEOUT_SECONDS}s. Output so far:\n"
                    + b"\n---\n".join(outputs).decode(errors="replace")
                ) from e
            outputs.append(stdout or b"")

        results: list[dict[str, object]] = []
        for rank, (proc, out_path, stdout) in enumerate(
            zip(procs, out_paths, outputs, strict=True)
        ):
            if not os.path.exists(out_path):
                raise RuntimeError(
                    f"_run_two_rank_subprocess_test: rank {rank} never wrote "
                    f"its result file (exit code {proc.returncode}). "
                    f"stdout/stderr:\n{stdout.decode(errors='replace')}"
                )
            with open(out_path) as f:
                result = json.load(f)
            if not result.get("ok"):
                raise RuntimeError(
                    f"_run_two_rank_subprocess_test: rank {rank} reported "
                    f"failure: {result.get('error')}\n"
                    f"{result.get('traceback', '')}\n"
                    f"stdout/stderr:\n{stdout.decode(errors='replace')}"
                )
            results.append(result)

        return results[0], results[1]


def _golden_llama_tokens(seed: int, n_decode_steps: int) -> list[int]:
    """Independently reproduces exactly what rank 0's worker-script
    branch does for prefill+decode on a PLAIN (unsharded) model --
    the golden reference this test compares the real 2-process
    result against, matching this session's established methodology."""
    from mlx_lm.models.llama import Model as LlamaModel
    from mlx_lm.models.llama import ModelArgs

    args = ModelArgs(
        model_type="llama",
        hidden_size=256,
        num_hidden_layers=4,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=4096,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    mx.random.seed(seed)
    model = LlamaModel(args)
    params = model.parameters()
    new_params = mlx.utils.tree_map(
        lambda p: mx.random.normal(shape=p.shape, dtype=p.dtype)
        if isinstance(p, mx.array)
        else p,
        params,
    )
    model.update(new_params)
    mx.eval(model.parameters())

    mx.random.seed(seed + 1000)
    prompt = mx.random.randint(0, args.vocab_size, shape=(6,))
    cache = model.make_cache()
    logits = model(prompt[None, :], cache=cache)
    mx.eval(logits)
    next_token = int(mx.argmax(logits[0, -1]).item())
    tokens = [next_token]
    for _ in range(n_decode_steps - 1):
        logits = model(mx.array([[next_token]]), cache=cache)
        mx.eval(logits)
        next_token = int(mx.argmax(logits[0, -1]).item())
        tokens.append(next_token)
    return tokens


@pytest.mark.slow
def test_batched_decode_over_real_2process_transport_matches_plain_forward() -> None:
    """THE real-process checkpoint: the batched-decode session driving
    a SINGLE request (rank 0 samples, rank 1 mirrors) across two
    genuinely separate OS processes connected by MLX's real ring
    backend -- proving the whole stack (session, wire protocol,
    batched metaframe layers) works with ZERO in-process shortcuts
    of any kind (no shared Python objects, no shared threads/locks,
    no shared MLX compile-decoration context)."""
    seed = 555
    golden_tokens = _golden_llama_tokens(seed, n_decode_steps=5)

    rank0_result, rank1_result = _run_two_rank_subprocess_test("llama", seed)

    assert rank0_result["tokens"] == golden_tokens
    # rank 1 has no tokens of its own to check (it never samples --
    # matches RankOneMirrorSession's zero-decision-logic design) but
    # its own "ok" flag being True means its forward passes all
    # completed without any ProtocolViolationError/shape mismatch.
    assert rank1_result["ok"] is True


@pytest.mark.slow
def test_batched_decode_over_real_2process_transport_with_real_dsv4() -> None:
    """The DSv4-specific real-process checkpoint (design doc Risk #1):
    the SAME batched-decode session driving a real (small) DSv4-Flash
    model -- including its HyperConnection multi-output @mx.compile'd
    Sinkhorn function, the exact mechanism that crashed this session's
    in-process 2-THREAD harness (warm memory fact 1202) -- across two
    genuinely separate OS processes. This does not independently
    verify DSv4 token-for-token correctness against a golden reference
    (that would require a much larger, slower model + a separate
    single-process DSv4 golden run); it verifies the full batched-PP
    STACK runs DSv4 without crashing or raising a protocol violation,
    which is the thing that was previously impossible to test at all
    for this specific model architecture."""
    seed = 777
    rank0_result, rank1_result = _run_two_rank_subprocess_test("dsv4", seed)

    assert rank0_result["ok"] is True
    assert rank1_result["ok"] is True
    assert isinstance(rank0_result["tokens"], list)
    assert len(rank0_result["tokens"]) == 5
