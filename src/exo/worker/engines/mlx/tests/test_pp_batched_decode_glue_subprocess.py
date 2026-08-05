# pyright: reportAny=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false
"""Real 2-PROCESS correctness test for Rank0BatchedDecodeGlue/
Rank1BatchedDecodeGlue -- the piggyback-admission glue layer that
closes the last real gap found before ExoBatchGenerator.submit()/
step() could safely dispatch into BatchedDecodeSession/
RankOneMirrorSession (see docs/hybrid-pp-prefill-tp-decode-design-
2026-08-04.md Section 9 for the full design history: single-writer
rule, piggyback-onto-step pattern, reactive admission detection).

Drives a real submit()/step()-SHAPED lifecycle (enqueue_admission
mimics submit(), tick() mimics step()'s single call site) across two
genuine OS processes connected via MLX's real ring backend -- the
exact harness pattern established by
test_pp_batched_decode_subprocess.py, extended one layer up (glue,
not just session) and one scenario further (mid-stream admission of
a SECOND request while the first is already decoding, plus a real
eviction via complete_request()).
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

_WORKER_SCRIPT = str(Path(__file__).parent / "_pp_glue_subprocess_worker.py")
_PYTHON = sys.executable
_TIMEOUT_SECONDS = 60.0


def _find_two_free_ports() -> tuple[int, int]:
    """See test_pp_batched_decode_subprocess.py's identically-named
    helper for the full rationale (real ports, not hardcoded
    constants, to avoid TIME_WAIT collisions across sequential runs)."""
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


def _run_two_rank_glue_subprocess_test(
    seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
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
                [_PYTHON, _WORKER_SCRIPT, str(rank), out_paths[rank], str(seed)],
                cwd=str(Path(__file__).parents[6]),
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
                    f"_run_two_rank_glue_subprocess_test: subprocess timed "
                    f"out after {_TIMEOUT_SECONDS}s. Output so far:\n"
                    + b"\n---\n".join(outputs).decode(errors="replace")
                ) from e
            outputs.append(stdout or b"")

        results: list[dict[str, object]] = []
        for rank, (proc, out_path, stdout) in enumerate(
            zip(procs, out_paths, outputs, strict=True)
        ):
            if not os.path.exists(out_path):
                raise RuntimeError(
                    f"_run_two_rank_glue_subprocess_test: rank {rank} never "
                    f"wrote its result file (exit code {proc.returncode}). "
                    f"stdout/stderr:\n{stdout.decode(errors='replace')}"
                )
            with open(out_path) as f:
                result = json.load(f)
            if not result.get("ok"):
                raise RuntimeError(
                    f"_run_two_rank_glue_subprocess_test: rank {rank} "
                    f"reported failure: {result.get('error')}\n"
                    f"{result.get('traceback', '')}\n"
                    f"stdout/stderr:\n{stdout.decode(errors='replace')}"
                )
            results.append(result)

        return results[0], results[1]


def _golden_two_request_tokens(
    seed: int,
) -> tuple[list[int], list[int]]:
    """Independently reproduces exactly what the worker script's rank
    0 branch does for BOTH requests via two separate serial plain
    forwards -- the golden reference this test compares the real
    2-process glue-layer result against."""
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

    def _decode(prompt: mx.array, n_steps: int) -> list[int]:
        cache = model.make_cache()
        logits = model(prompt[None, :], cache=cache)
        mx.eval(logits)
        next_token = int(mx.argmax(logits[0, -1]).item())
        tokens = [next_token]
        for _ in range(n_steps - 1):
            logits = model(mx.array([[next_token]]), cache=cache)
            mx.eval(logits)
            next_token = int(mx.argmax(logits[0, -1]).item())
            tokens.append(next_token)
        return tokens

    mx.random.seed(seed + 1000)
    prompt_a = mx.random.randint(0, args.vocab_size, shape=(5,))
    golden_a = _decode(prompt_a, n_steps=4)

    mx.random.seed(seed + 2000)
    prompt_b = mx.random.randint(0, args.vocab_size, shape=(4,))
    golden_b = _decode(prompt_b, n_steps=5)

    return golden_a, golden_b


@pytest.mark.slow
def test_glue_layer_over_real_2process_transport_matches_plain_forwards() -> None:
    """THE real-process checkpoint for the glue layer itself: a real
    submit()/step()-shaped lifecycle (enqueue upfront, tick to admit,
    tick to decode, enqueue MID-STREAM, tick to admit the second
    request alongside the first's ongoing decode, tick both together,
    complete_request to evict the first via a real eviction
    round-trip, tick the survivor solo) across two genuine OS
    processes -- proving the single-writer piggyback design actually
    works end-to-end, not just its two component sessions in
    isolation (already proven by
    test_pp_batched_decode_subprocess.py)."""
    seed = 999
    golden_a, golden_b = _golden_two_request_tokens(seed)

    rank0_result, rank1_result = _run_two_rank_glue_subprocess_test(seed)

    assert rank0_result["tokens_a"] == golden_a
    assert rank0_result["tokens_b"] == golden_b
    assert rank1_result["ok"] is True
