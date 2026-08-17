"""Prefill probe: real tokenization, prompt sizing, throughput, record."""

from __future__ import annotations

import math

import pytest

from trusted_measurement.depth_matrix import ALL_DEPTH_CELLS
from trusted_measurement.probes.prefill_probe import (
    PREFILL_REQUIRED_RUNTIME_MODES,
    FakePrefillClient,
    PrefillClient,
    PrefillProbeConfig,
    build_argument_parser,
    build_prefill_prompt,
    build_prefill_record,
    main,
    prefill_throughput_tokens_per_second,
)
from trusted_measurement.probes.tests.builders import (
    WordTokenizer,
    fake_command_runner,
    markers_for,
)
from trusted_measurement.token_truth import TokenCountMismatchError

# ------------------------------------------------------------ prompt sizing


def test_prompt_hits_the_target_token_count_for_real() -> None:
    tokenizer = WordTokenizer()
    prompt = build_prefill_prompt(tokenizer=tokenizer, target_token_count=2000, seed=11)
    real_count = len(tokenizer.encode(prompt.text))
    assert prompt.offline_token_count == real_count
    assert prompt.relative_error <= 0.02
    assert prompt.needle in prompt.text


def test_prompt_sizing_works_across_several_depths() -> None:
    tokenizer = WordTokenizer()
    for target in (200, 1000, 5000, 20000):
        prompt = build_prefill_prompt(
            tokenizer=tokenizer, target_token_count=target, seed=3
        )
        assert prompt.relative_error <= 0.02, (target, prompt.offline_token_count)


def test_prompt_sizing_rejects_a_nonpositive_target() -> None:
    with pytest.raises(ValueError):
        _ = build_prefill_prompt(
            tokenizer=WordTokenizer(), target_token_count=0, seed=1
        )


def test_prompt_sizing_gives_up_loudly_when_it_cannot_converge() -> None:
    with pytest.raises(ValueError, match="could not build a prompt"):
        _ = build_prefill_prompt(
            tokenizer=WordTokenizer(),
            target_token_count=3,
            seed=1,
            tolerance_fraction=0.001,
        )


def test_haystack_view_preserves_the_needle() -> None:
    prompt = build_prefill_prompt(
        tokenizer=WordTokenizer(), target_token_count=500, seed=5
    )
    haystack = prompt.haystack()
    assert haystack.needle == prompt.needle
    assert haystack.prompt == prompt.text


# --------------------------------------------------------------- throughput


def test_throughput_uses_the_offline_count() -> None:
    assert math.isclose(
        prefill_throughput_tokens_per_second(
            offline_token_count=1000, time_to_first_token_seconds=4.0
        ),
        250.0,
        rel_tol=1e-9,
    )


def test_throughput_rejects_degenerate_inputs() -> None:
    with pytest.raises(ValueError):
        _ = prefill_throughput_tokens_per_second(
            offline_token_count=100, time_to_first_token_seconds=0.0
        )
    with pytest.raises(ValueError):
        _ = prefill_throughput_tokens_per_second(
            offline_token_count=0, time_to_first_token_seconds=1.0
        )


# ------------------------------------------------------------ fake client


def test_fake_client_satisfies_the_protocol() -> None:
    client = FakePrefillClient(tokenizer=WordTokenizer(), needle="12345")
    assert isinstance(client, PrefillClient)


# ------------------------------------------------------------------ record


def _config(replicates: int = 3) -> PrefillProbeConfig:
    return PrefillProbeConfig(
        target_tokens_by_depth={"shallow": 500, "mid": 2000, "near_max_context": 8000},
        thermal_states=("cold_start", "warm"),
        replicates_per_cell=replicates,
    )


def _build(client: PrefillClient, **overrides: object):
    kwargs: dict[str, object] = dict(
        client=client,
        tokenizer=WordTokenizer(),
        tokenizer_name="word_tokenizer",
        config=_config(),
        runtime_mode_markers=markers_for(*PREFILL_REQUIRED_RUNTIME_MODES),
        canary_session_certified=True,
        exo_repo="/exo",
        mlx_repo="/mlx",
        command_runner=fake_command_runner(),
        environ={"EXO_PREFILL_STEP_SIZE": "2048"},
        claims_default_safe=True,
    )
    kwargs.update(overrides)
    return build_prefill_record(**kwargs)  # pyright: ignore[reportArgumentType]


def test_full_matrix_sweep_produces_a_trusted_record() -> None:
    tokenizer = WordTokenizer()
    prompt = build_prefill_prompt(
        tokenizer=tokenizer, target_token_count=500, seed=20260817
    )
    client = FakePrefillClient(tokenizer=tokenizer, needle=prompt.needle)
    record = _build(client)
    assert record.validate_envelope() == ()
    assert record.prompt_size_dependent is True
    assert record.token_ground_truth is not None
    assert record.token_ground_truth.agrees
    # 3 depths x 2 thermal states x 3 replicates.
    assert len(record.value.replicates) == 18
    assert client.calls == 18
    assert {cell.label() for cell in record.depth_matrix_cells} == {
        f"{depth}/{thermal}" for depth, thermal in ALL_DEPTH_CELLS
    }


def test_server_inflated_token_count_fails_the_run() -> None:
    tokenizer = WordTokenizer()
    prompt = build_prefill_prompt(
        tokenizer=tokenizer, target_token_count=500, seed=20260817
    )
    client = FakePrefillClient(
        tokenizer=tokenizer, needle=prompt.needle, token_count_scale=1.42
    )
    with pytest.raises(TokenCountMismatchError):
        _ = _build(client)


def test_wrong_completion_makes_the_record_untrusted() -> None:
    tokenizer = WordTokenizer()
    client = FakePrefillClient(
        tokenizer=tokenizer,
        needle="0000000",
        completion_override="I could not find the code.",
    )
    record = _build(client)
    violations = record.validate_envelope()
    assert any(violation.field == "content_check" for violation in violations)


def test_missing_runtime_marker_makes_the_record_untrusted() -> None:
    tokenizer = WordTokenizer()
    prompt = build_prefill_prompt(
        tokenizer=tokenizer, target_token_count=500, seed=20260817
    )
    client = FakePrefillClient(tokenizer=tokenizer, needle=prompt.needle)
    record = _build(client, runtime_mode_markers=markers_for("something_else"))
    assert any(
        "batched_prefill" in str(violation) for violation in record.validate_envelope()
    )


def test_empty_configuration_is_rejected() -> None:
    tokenizer = WordTokenizer()
    client = FakePrefillClient(tokenizer=tokenizer, needle="1")
    empty = PrefillProbeConfig(
        target_tokens_by_depth={},
        thermal_states=("warm",),
        replicates_per_cell=3,
    )
    with pytest.raises(ValueError, match="no measured cells"):
        _ = _build(client, config=empty)


# --------------------------------------------------------------------- CLI


def test_argument_parser_defaults() -> None:
    arguments = build_argument_parser().parse_args([])
    assert arguments.mid_tokens == 32768  # pyright: ignore[reportAny]


def test_main_is_an_honest_stub() -> None:
    with pytest.raises(NotImplementedError, match="PrefillClient"):
        _ = main([])
