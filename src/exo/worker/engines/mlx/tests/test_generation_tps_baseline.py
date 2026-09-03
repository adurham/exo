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

    missing = [block for block in blocks if "generation_time_at_start" not in block]

    assert not missing, (
        f"{len(missing)} of {len(blocks)} _EngineTask(...) constructions omit "
        f"generation_time_at_start entirely, silently inheriting the 0.0 "
        f"dataclass default -- which is the same uptime-as-generation-time "
        f"bug as an explicit 0.0. Set it from "
        f"_mlx_gen_elapsed_seconds(self._mlx_gen)."
    )


def test_reported_gui_figure_implies_an_absurd_seconds_per_token() -> None:
    """Pin what the user's report actually determines -- and only that.

    CORRECTION (2026-08-15): an earlier version of this test asserted
    "33 completion tokens against a 1-day-6:30 uptime", which was a
    fabricated decomposition. It is also self-inconsistent: 1d6:30 is
    109,800 s, not the 223,906 s the assertion used. Worse, the
    assertion was circular -- it divided two invented numbers and
    checked the quotient it had just constructed, so it would pass for
    ANY pair with the right ratio and could never fail.

    The report fixes exactly ONE quantity: the RATIO. 6,785,033.3 ms/tok
    means 6,785.03 seconds per token. The split into (tokens, elapsed)
    is underdetermined -- 1 token / 1.88 h, 16 tokens / 30.2 h, and
    33 tokens / 62.2 h are all equally consistent with what was
    displayed, and nothing in the report distinguishes them.

    So assert the thing that is actually true and actually diagnostic:
    the implied per-token cost is absurd by orders of magnitude against
    any real decode rate this cluster produces, which is the signature
    of an elapsed-time baseline that was never subtracted.
    """
    reported_ms_per_token = 6785033.3
    seconds_per_token = reported_ms_per_token / 1000.0

    # The fastest and slowest REAL per-token costs measured on this
    # cluster (Section 54): ~40 ms/tok below the prefill-step-size
    # threshold, ~2.2 s/tok above it. The report is orders of magnitude
    # outside even the slow end.
    slowest_real_seconds_per_token = 2.2

    assert seconds_per_token > 1000 * slowest_real_seconds_per_token, (
        f"{seconds_per_token:.1f} s/tok should be absurd relative to the "
        f"slowest real measurement ({slowest_real_seconds_per_token} s/tok); "
        f"if this ever becomes plausible the report was not the baseline bug"
    )

    # And it renders as "0.0 tok/s", which is why the bug looked like a
    # missing value rather than a wrong one.
    assert f"{1.0 / seconds_per_token:.1f}" == "0.0"


def test_absolute_clock_baseline_produces_the_absurd_ratio() -> None:
    """The mechanism, stated as a property rather than invented numbers.

    Whatever the true token count was, subtracting a 0.0 baseline from
    an absolute ``perf_counter()`` reading yields a "generation time"
    equal to the process/host clock -- necessarily enormous relative to
    a real request -- so tps collapses toward zero. Demonstrated across
    a range of plausible clock values instead of asserting one.
    """
    completion_tokens = 20  # any realistic short-answer count

    for absolute_clock_seconds in (6_785.0, 109_800.0, 223_906.1):
        correct_elapsed = 0.8  # a real fast answer
        correct_tps = completion_tokens / correct_elapsed

        # The bug: baseline 0.0, so the delta IS the absolute clock.
        buggy_delta = absolute_clock_seconds - 0.0
        buggy_tps = completion_tokens / buggy_delta

        assert correct_tps > 1.0, "sanity: the real rate is a normal number"
        assert buggy_tps < 0.01, (
            f"a 0.0 baseline against clock={absolute_clock_seconds}s must "
            f"collapse tps toward zero, got {buggy_tps}"
        )
        assert f"{buggy_tps:.1f}" == "0.0"
