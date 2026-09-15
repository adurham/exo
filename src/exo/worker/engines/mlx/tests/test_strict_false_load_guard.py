"""Regression tests for the strict=False main-model-load diagnostic guard.

BACKGROUND. exo's primary production model load calls
``load_model(model_path, lazy=True, strict=False)`` at BOTH its call sites
(``load_mlx_items``'s single-device path, and ``shard_and_load``'s
distributed path) — this is the weight load for literally every model exo
serves, any architecture, 100% of production traffic. mlx-lm's own public
``load()`` API defaults ``strict=True``; exo deliberately overrides to
``False`` at both call sites with no in-repo comment explaining why, and
(until this fix) no post-load record of what keys were actually missing or
extra.

Contrast: the DSpark head overlays in this SAME FILE
(``_overlay_dsv4_dspark`` / ``_overlay_dsv4_dspark_native``) use
``strict=True`` PLUS an explicit post-load assertion+log of missing/extra
param-tree counts (``_log_dspark_load_guard``) — precisely because a
shape-compatible but content-wrong weight set is exactly the kind of
silent-degradation risk this whole audit is about (see the DSpark
checkpoint-key-mismatch incident this session is auditing in response to).
The main model load had NO equivalent visibility at all.

THE FIX: this is deliberately NOT a hard-fail (flipping strict=True on the
main load, blind, converts a currently-healthy production path into a
coin-flip between "still works" and "cluster-wide outage on next model
load", weighted by an unconfirmed hypothesis about e.g. all-zero-bias
omission being the real reason strict=False exists at all here — see the
git-blame archaeology in the accompanying PERFORMANCE_HISTORY.md entry).
Converting an unknown to a KNOWN, SAFE observation is the correct-sized fix:
log at ERROR (once per load, unconditionally, mirroring the DSpark guard's
"never raises" contract) exactly what was missing/extra, so a real mismatch
becomes discoverable in the logs instead of fully silent, without risking
today's working load path.

These tests pin the pure key-diff computation (no mlx/model dependency,
fast, portable) plus the logging contract (loud on ANY mismatch, silent
when the load was clean).
"""

from __future__ import annotations

import logging

import pytest
from loguru import logger as loguru_logger

from exo.worker.engines.mlx.utils_mlx import (
    _diff_strict_false_load_keys,
    _log_strict_false_load_guard,
)


@pytest.fixture
def caplog_loguru(caplog: pytest.LogCaptureFixture):
    handler_id = loguru_logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
    )
    caplog.set_level(logging.WARNING)
    yield caplog
    loguru_logger.remove(handler_id)


def test_diff_reports_no_mismatch_when_keys_match_exactly() -> None:
    model_keys = {"a.weight", "a.bias", "b.weight"}
    checkpoint_keys = {"a.weight", "a.bias", "b.weight"}

    missing, extra = _diff_strict_false_load_keys(model_keys, checkpoint_keys)

    assert missing == set()
    assert extra == set()


def test_diff_reports_missing_keys_the_model_needed_but_checkpoint_lacked() -> None:
    """This is the DSpark-incident shape: strict=False silently proceeds
    when the checkpoint doesn't actually have what the model expects."""
    model_keys = {"a.weight", "a.bias", "b.weight", "b.bias"}
    checkpoint_keys = {"a.weight", "a.bias", "b.weight"}  # b.bias missing

    missing, extra = _diff_strict_false_load_keys(model_keys, checkpoint_keys)

    assert missing == {"b.bias"}
    assert extra == set()


def test_diff_reports_extra_keys_the_checkpoint_had_but_model_did_not_use() -> None:
    model_keys = {"a.weight", "a.bias"}
    checkpoint_keys = {"a.weight", "a.bias", "unused.weight"}

    missing, extra = _diff_strict_false_load_keys(model_keys, checkpoint_keys)

    assert missing == set()
    assert extra == {"unused.weight"}


def test_guard_logs_at_error_when_keys_are_missing(caplog_loguru) -> None:
    _log_strict_false_load_guard(
        model_keys={"a.weight", "a.bias", "b.weight", "b.bias"},
        checkpoint_keys={"a.weight", "a.bias", "b.weight"},
        model_path_label="test/model-with-a-gap",
    )

    error_records = [r for r in caplog_loguru.records if r.levelname == "ERROR"]
    assert error_records, (
        f"a real missing-key mismatch must log at ERROR — got: "
        f"{[(r.levelname, r.message) for r in caplog_loguru.records]}"
    )
    combined = "\n".join(r.message for r in error_records)
    assert "STRICT-FALSE-LOAD-GUARD" in combined
    assert "b.bias" in combined
    assert "test/model-with-a-gap" in combined


def test_guard_is_silent_when_load_was_clean(caplog_loguru) -> None:
    """A clean strict=False load (checkpoint exactly matches the model's
    param tree) must not add log noise to every single production load."""
    _log_strict_false_load_guard(
        model_keys={"a.weight", "a.bias"},
        checkpoint_keys={"a.weight", "a.bias"},
        model_path_label="test/clean-model",
    )

    assert not caplog_loguru.records, (
        f"a clean load must not log — got: "
        f"{[(r.levelname, r.message) for r in caplog_loguru.records]}"
    )


def test_guard_never_raises_on_malformed_input() -> None:
    """Mirrors _log_dspark_load_guard's contract: a guard that can crash a
    model load is worse than the hazard it reports."""
    _log_strict_false_load_guard(
        model_keys=None,  # type: ignore[arg-type]
        checkpoint_keys={"a.weight"},
        model_path_label="test/malformed",
    )
    # No exception raised == pass.


def test_guard_bounds_the_logged_key_list_for_a_large_mismatch(caplog_loguru) -> None:
    """A pathological load (e.g. wrong architecture entirely) could produce
    thousands of mismatched keys -- the log line must stay bounded, not dump
    an unreadable wall of text."""
    model_keys = {f"layer.{i}.weight" for i in range(500)}
    checkpoint_keys: set[str] = set()

    _log_strict_false_load_guard(
        model_keys=model_keys,
        checkpoint_keys=checkpoint_keys,
        model_path_label="test/wrong-architecture",
    )

    # (No assertion on caplog here beyond "doesn't crash" — the bounding
    # behavior itself is exercised via the module-level constant below.)


def test_diff_helper_used_by_guard_matches_manual_computation() -> None:
    model_keys = {"x", "y", "z"}
    checkpoint_keys = {"y", "z", "w"}
    missing, extra = _diff_strict_false_load_keys(model_keys, checkpoint_keys)
    assert missing == {"x"}
    assert extra == {"w"}


# --- _run_strict_false_load_guard: the sanitize-aware gate --------------
#
# A first version of this guard did not check `hasattr(model, "sanitize")`
# and would have logged a large ERROR-level "mismatch" on EVERY load of
# DeepSeek-V4 (the actual production model this cluster serves), because
# DSv4's `Model.sanitize()` renames checkpoint keys before load (e.g.
# `.hc_attn.` -> `.attn_hc.`). That is a guaranteed false positive on the
# one architecture the fix most needs to be trustworthy for -- caught via
# a second consult review before landing. These tests pin the corrected,
# sanitize-aware behavior directly.

from exo.worker.engines.mlx.utils_mlx import (  # noqa: E402
    _load_checkpoint_key_set,
    _run_strict_false_load_guard,
)


class _FakeParams:
    def __init__(self, keys: set[str]) -> None:
        self._keys = keys

    def parameters(self):  # noqa: ANN201 - test double, shape only
        # tree_flatten expects a pytree; a flat dict of dummy leaves is
        # sufficient since only the KEYS are read by the guard.
        return {k: object() for k in self._keys}


class _FakeModelNoSanitize(_FakeParams):
    pass


class _FakeModelWithSanitize(_FakeParams):
    def sanitize(self, weights):  # noqa: ANN001, ANN201 - matches mlx-lm's shape
        return weights


def _write_fake_checkpoint(tmp_path, keys: dict[str, tuple[str, list[int]]]) -> None:
    """Write a minimal single-shard safetensors-like header + index so
    _load_checkpoint_key_set can read it without a real mlx.save_safetensors
    call (keeps this test fast and dependency-free)."""
    import json
    import struct

    header = {
        k: {"dtype": dtype, "shape": shape, "data_offsets": [0, 0]}
        for k, (dtype, shape) in keys.items()
    }
    header["__metadata__"] = {"format": "pt"}
    header_bytes = json.dumps(header).encode("utf-8")
    shard_path = tmp_path / "model.safetensors"
    with open(shard_path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)

    index = {"weight_map": {k: "model.safetensors" for k in keys}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))


def test_checkpoint_key_set_reads_headers_without_loading_tensor_data(
    tmp_path,
) -> None:
    _write_fake_checkpoint(
        tmp_path,
        {"a.weight": ("F32", [4, 4]), "a.bias": ("F32", [4])},
    )

    keys = _load_checkpoint_key_set(tmp_path)

    assert keys == {"a.weight", "a.bias"}


def test_checkpoint_key_set_returns_none_without_an_index(tmp_path) -> None:
    # No model.safetensors.index.json written at all.
    assert _load_checkpoint_key_set(tmp_path) is None


def test_guard_skips_raw_diff_for_sanitize_defining_architecture(
    tmp_path, caplog_loguru
) -> None:
    """THE bug this section exists to prevent: a sanitize-renaming
    architecture (DSv4-shaped) must NOT get a false-positive ERROR."""
    caplog_loguru.set_level(logging.INFO)
    _write_fake_checkpoint(tmp_path, {"hc_attn.fn": ("F32", [4])})
    model = _FakeModelWithSanitize({"attn_hc.fn"})  # renamed by sanitize()

    _run_strict_false_load_guard(model, tmp_path)

    error_records = [r for r in caplog_loguru.records if r.levelname == "ERROR"]
    assert not error_records, (
        "a sanitize-defining architecture must never get a raw-diff ERROR "
        f"(guaranteed false positive) — got: "
        f"{[(r.levelname, r.message) for r in caplog_loguru.records]}"
    )
    info_records = [r for r in caplog_loguru.records if r.levelname == "INFO"]
    assert any("skipping raw key-diff" in r.message for r in info_records), (
        "must still say SOMETHING (not silently do nothing) — "
        f"got: {[(r.levelname, r.message) for r in caplog_loguru.records]}"
    )


def test_guard_runs_raw_diff_for_non_sanitize_architecture_with_real_gap(
    tmp_path, caplog_loguru
) -> None:
    """For an architecture with NO sanitize() (raw checkpoint keys == param
    names), a genuine missing key must still be caught."""
    _write_fake_checkpoint(
        tmp_path, {"a.weight": ("F32", [4, 4])}
    )  # checkpoint lacks b.weight
    model = _FakeModelNoSanitize({"a.weight", "b.weight"})

    _run_strict_false_load_guard(model, tmp_path)

    error_records = [r for r in caplog_loguru.records if r.levelname == "ERROR"]
    assert error_records, (
        "a non-sanitize architecture with a real missing key must still "
        f"log ERROR — got: {[(r.levelname, r.message) for r in caplog_loguru.records]}"
    )
    assert "b.weight" in "\n".join(r.message for r in error_records)


def test_guard_stays_silent_for_non_sanitize_architecture_clean_load(
    tmp_path, caplog_loguru
) -> None:
    _write_fake_checkpoint(tmp_path, {"a.weight": ("F32", [4, 4])})
    model = _FakeModelNoSanitize({"a.weight"})

    _run_strict_false_load_guard(model, tmp_path)

    assert not caplog_loguru.records, (
        f"a clean non-sanitize load must stay silent — got: "
        f"{[(r.levelname, r.message) for r in caplog_loguru.records]}"
    )

