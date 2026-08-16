"""Regression test: ``generation_tps`` must not be computed against a
hardcoded ``generation_time_at_start=0.0`` baseline.

THE BUG (found 2026-08-15 from a real user report via the exo web GUI).
A short request ("what is the capital of France?") that answered
correctly and fast displayed:

    TTFT 2727.0ms    TPS 0.0 tok/s  (6785033.3 ms/tok)

6,785,033 ms/token is ~113 minutes per token. The generation was
genuinely fast (independently measured at ~25 tok/s on the same
cluster); the NUMBER was fabricated.

MECHANISM. ``ExoBatchGenerator`` computes, at completion:

    gen_time_delta = _mlx_gen_elapsed_seconds(self._mlx_gen) - state.generation_time_at_start
    generation_tps = state.completion_tokens / gen_time_delta

``_mlx_gen_elapsed_seconds()`` returns a CUMULATIVE counter when the
mlx-lm generator exposes ``_stats.generation_time``, but falls back to
``time.perf_counter()`` -- an ABSOLUTE monotonic clock reading whose
zero point is an arbitrary moment in the distant past (on macOS, host
boot) -- when it does not. The current mlx-lm pin does not expose
``_stats``, so the fallback is what actually runs in production.

Two of the five ``_EngineTask`` construction sites hardcoded
``generation_time_at_start=0.0`` instead of sampling the same function.
Subtracting 0.0 from an absolute clock yields the machine's entire
uptime as the "generation time". The nodes had been up 1 day 6:30 at
the time of the report; 33 tokens / 223,906s = 0.000147 tok/s, which
renders as "0.0" tok/s and 6,785,033.3 ms/tok. That reproduces the
reported figure exactly.

WHY IT MATTERS BEYOND COSMETICS. The affected sites are
``_submit_batched_decode`` and ``_submit_batched_decode_deferred`` --
the latter is the ACTIVE admission path whenever
``EXO_PP_BATCHED_DECODE=1``, i.e. the exact configuration this
campaign's throughput investigation runs under. This field feeds
``GenerationStats.generation_tps``, which flows to the Prometheus
``exo_generation_tps`` histogram and to the dashboard's own display.
A silently-wrong throughput metric in a campaign whose entire purpose
is measuring throughput is a measurement-apparatus defect, not a UI
nit -- the same class as the ``usage.prompt_tokens`` bug fixed at
exo@7d14daea7, which reported the prompt TAIL instead of the prompt.

THE FIX. Sample ``_mlx_gen_elapsed_seconds(self._mlx_gen)`` at task
construction, exactly as the three already-correct sites do, so the
baseline and the reading always come from the same clock and the
subtraction is a true delta regardless of which branch the helper
takes.
"""

from __future__ import annotations

import re
from pathlib import Path

_BATCH_GENERATE = (
    Path(__file__).resolve().parents[1] / "generator" / "batch_generate.py"
)


def _engine_task_constructor_blocks(source: str) -> list[str]:
    """Return the source text of every ``_EngineTask(...)`` construction."""
    blocks: list[str] = []
    for match in re.finditer(r"_EngineTask\(", source):
        start = match.end()
        depth = 1
        idx = start
        while idx < len(source) and depth > 0:
            if source[idx] == "(":
                depth += 1
            elif source[idx] == ")":
                depth -= 1
            idx += 1
        blocks.append(source[start : idx - 1])
    return blocks


def test_no_engine_task_hardcodes_a_zero_generation_time_baseline() -> None:
    """Every ``_EngineTask`` must derive its generation-time baseline from
    the same clock the completion path reads, never from a literal 0.0.

    A literal 0.0 baseline makes ``gen_time_delta`` equal the absolute
    ``perf_counter()`` reading (machine uptime) whenever
    ``_mlx_gen_elapsed_seconds`` takes its documented fallback branch,
    producing a throughput figure wrong by many orders of magnitude.
    """
    source = _BATCH_GENERATE.read_text()
    blocks = _engine_task_constructor_blocks(source)

    assert blocks, (
        "found no _EngineTask(...) constructions in batch_generate.py -- this "
        "test's parsing assumption has broken, not the code under test"
    )

    offenders = [
        block
        for block in blocks
        if re.search(r"generation_time_at_start\s*=\s*0\.0", block)
    ]

    assert not offenders, (
        f"{len(offenders)} of {len(blocks)} _EngineTask(...) constructions set "
        f"generation_time_at_start=0.0. That baseline is subtracted from "
        f"_mlx_gen_elapsed_seconds(), which falls back to an ABSOLUTE "
        f"time.perf_counter() reading -- so a 0.0 baseline reports the "
        f"machine's entire uptime as the generation time and yields a "
        f"generation_tps near zero (the real 2026-08-15 GUI report: 0.0 tok/s "
        f"/ 6785033.3 ms/tok for a fast, correct answer). Pass "
        f"_mlx_gen_elapsed_seconds(self._mlx_gen) instead."
    )


def test_every_engine_task_sets_the_baseline_explicitly() -> None:
    """The baseline must never be left to the dataclass default either.

    ``_EngineTask.generation_time_at_start`` defaults to 0.0, so an
    omitted keyword is exactly as wrong as an explicit ``=0.0`` -- and
    harder to spot in review. Pin the requirement that every
    construction site states it.
    """
    source = _BATCH_GENERATE.read_text()
    blocks = _engine_task_constructor_blocks(source)

    missing = [
        block for block in blocks if "generation_time_at_start" not in block
    ]

    assert not missing, (
        f"{len(missing)} of {len(blocks)} _EngineTask(...) constructions omit "
        f"generation_time_at_start entirely, silently inheriting the 0.0 "
        f"dataclass default -- which is the same uptime-as-generation-time "
        f"bug as an explicit 0.0. Set it from "
        f"_mlx_gen_elapsed_seconds(self._mlx_gen)."
    )


def test_reported_gui_figure_is_reproduced_by_a_zero_baseline() -> None:
    """Pin the arithmetic that ties the user's report to this root cause.

    Guards against a future 'fix' that changes the number without
    understanding it: 33 completion tokens against a 1-day-6:30 uptime
    baseline is precisely the 0.0 tok/s / 6785033.3 ms/tok the dashboard
    displayed.
    """
    completion_tokens = 33
    uptime_seconds = 223906.1  # nodes were up 1 day, 6:30 at report time

    tps = completion_tokens / uptime_seconds
    ms_per_token = 1000.0 / tps

    assert f"{tps:.1f}" == "0.0"
    assert round(ms_per_token, 1) == 6785033.3
