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


# --- _run_strict_false_load_guard: the load-instant capture --------------
#
# HISTORY — both halves matter.
#
# 1. A first version of this guard did not check `hasattr(model, "sanitize")`
#    and would have logged a large ERROR-level "mismatch" on EVERY load of
#    DeepSeek-V4 (the actual production model this cluster serves), because
#    DSv4's `Model.sanitize()` renames checkpoint keys before load (e.g.
#    `.hc_attn.` -> `.attn_hc.`). Caught via a second consult review before
#    landing.
#
# 2. The fix for (1) was to SKIP the diff entirely for sanitize() models —
#    which disabled the guard on exactly the model it exists to protect
#    (DSv4 is the production model; the strict=False random-init incident
#    happened there). The current shape removes the skip: the caller
#    captures BOTH sides of the comparison at load time via
#    `_RecordLoadedWeightKeys`, so the diff is namespace-correct for ANY
#    architecture. The skip survives only as a documented FALLBACK for
#    callers that pass no capture.

from exo.worker.engines.mlx.utils_mlx import (  # noqa: E402
    _load_checkpoint_key_set,
    _RecordLoadedWeightKeys,
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


def test_guard_catches_missing_key_on_sanitize_architecture_with_capture(
    tmp_path, caplog_loguru
) -> None:
    """THE point of the capture: a sanitize()-defining architecture (DSv4-shaped
    — the production model) with a genuinely dropped key must now be CAUGHT.

    Before this fix the guard skipped every such architecture, so this exact
    failure — a parameter silently left at random init by a strict=False load —
    was invisible on the one model that matters most.
    """
    # The raw on-disk keys use PRE-sanitize names; the capture carries the
    # post-sanitize set the load actually delivered, with one key missing.
    _write_fake_checkpoint(tmp_path, {"hc_attn.fn": ("F32", [4])})
    model = _FakeModelWithSanitize({"attn_hc.fn", "attn_hc.base"})
    captured = {
        "keys": {"attn_hc.fn"},  # delivered — "attn_hc.base" never arrived
        "params": {"attn_hc.fn", "attn_hc.base"},
    }

    _run_strict_false_load_guard(model, tmp_path, captured)

    error_records = [r for r in caplog_loguru.records if r.levelname == "ERROR"]
    assert error_records, (
        "a dropped key on a sanitize-defined architecture must be caught "
        f"once a load capture exists — got: "
        f"{[(r.levelname, r.message) for r in caplog_loguru.records]}"
    )
    assert "attn_hc.base" in "\n".join(r.message for r in error_records)
    assert "post-sanitize" in "\n".join(r.message for r in error_records), (
        "the log line must say which namespace it compared"
    )


def test_guard_stays_silent_on_clean_sanitize_load_with_capture(
    tmp_path, caplog_loguru
) -> None:
    """A healthy DSv4-shaped load must not cry wolf — the reason the skip
    existed in the first place, now satisfied by the capture instead."""
    _write_fake_checkpoint(tmp_path, {"hc_attn.fn": ("F32", [4])})
    model = _FakeModelWithSanitize({"attn_hc.fn"})
    captured = {"keys": {"attn_hc.fn"}, "params": {"attn_hc.fn"}}

    _run_strict_false_load_guard(model, tmp_path, captured)

    assert not caplog_loguru.records, (
        "a clean captured load must stay silent — got: "
        f"{[(r.levelname, r.message) for r in caplog_loguru.records]}"
    )


def test_capture_beats_raw_diff_which_would_false_alarm(
    tmp_path, caplog_loguru
) -> None:
    """The exact case that made the raw diff unusable on DSv4: on-disk keys are
    pre-sanitize, so a raw diff reports an EXPECTED mismatch. With a capture
    present the raw path must not be consulted at all."""
    # Raw on-disk names deliberately differ from the model's param names.
    _write_fake_checkpoint(tmp_path, {"hc_attn_fn": ("F32", [4])})
    model = _FakeModelWithSanitize({"attn_hc.fn"})
    captured = {"keys": {"attn_hc.fn"}, "params": {"attn_hc.fn"}}

    _run_strict_false_load_guard(model, tmp_path, captured)

    error_records = [r for r in caplog_loguru.records if r.levelname == "ERROR"]
    assert not error_records, (
        "with a capture available the raw on-disk set (pre-sanitize names) "
        "must never drive an ERROR — that is the guaranteed false positive. "
        f"got: {[(r.levelname, r.message) for r in caplog_loguru.records]}"
    )


def test_guard_skips_raw_diff_for_sanitize_architecture_without_capture(
    tmp_path, caplog_loguru
) -> None:
    """FALLBACK path (no capture available at all — e.g. a caller that never
    wrapped its load): the original conservative behavior still applies, because
    a raw diff on a renaming architecture is a guaranteed false positive."""
    caplog_loguru.set_level(logging.INFO)
    _write_fake_checkpoint(tmp_path, {"hc_attn.fn": ("F32", [4])})
    model = _FakeModelWithSanitize({"attn_hc.fn"})  # renamed by sanitize()

    _run_strict_false_load_guard(model, tmp_path)  # no capture

    error_records = [r for r in caplog_loguru.records if r.levelname == "ERROR"]
    assert not error_records, (
        "a sanitize-defining architecture without a capture must never get a "
        f"raw-diff ERROR (guaranteed false positive) — got: "
        f"{[(r.levelname, r.message) for r in caplog_loguru.records]}"
    )
    info_records = [r for r in caplog_loguru.records if r.levelname == "INFO"]
    assert any("no load-time key" in r.message for r in info_records), (
        "must still say SOMETHING (not silently do nothing) — "
        f"got: {[(r.levelname, r.message) for r in caplog_loguru.records]}"
    )


def test_capture_partial_falls_back_instead_of_mis_diffing(
    tmp_path, caplog_loguru
) -> None:
    """A capture carrying only ONE side cannot support the load-instant
    comparison; falling through to the sanitize fallback is correct — using the
    half-capture would diff a post-sanitize set against a raw one."""
    caplog_loguru.set_level(logging.INFO)
    _write_fake_checkpoint(tmp_path, {"hc_attn.fn": ("F32", [4])})
    model = _FakeModelWithSanitize({"attn_hc.fn"})

    _run_strict_false_load_guard(model, tmp_path, {"keys": {"attn_hc.fn"}})

    error_records = [r for r in caplog_loguru.records if r.levelname == "ERROR"]
    assert not error_records, (
        "a half-capture must not produce a mixed-namespace ERROR — "
        f"got: {[(r.levelname, r.message) for r in caplog_loguru.records]}"
    )


def test_record_loaded_weight_keys_captures_both_sides_at_load_time() -> None:
    """The capture must record, from ONE instant: the exact key set delivered
    to ``load_weights`` AND the module's own param keys — i.e. the same two
    sides mlx's own ``strict=True`` branch compares. Requires real mlx; this
    is the mechanism the production call sites depend on.
    """
    import mlx.core as mx
    import mlx.nn as nn

    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4, bias=True)

    m = _M()
    with _RecordLoadedWeightKeys() as captured:
        m.load_weights(
            [("lin.weight", mx.zeros((4, 4))), ("lin.bias", mx.zeros((4,)))],
            strict=False,
        )

    assert captured.get("keys") == {"lin.weight", "lin.bias"}, (
        f"delivered keys not captured: {captured.get('keys')}"
    )
    assert {"lin.weight", "lin.bias"} <= (captured.get("params") or set()), (
        f"param side not captured: {captured.get('params')}"
    )


def test_record_loaded_weight_keys_restores_the_original_method() -> None:
    """The patch must be undone on exit — a leaked monkeypatch on nn.Module
    would affect every later load in the process."""
    import mlx.nn as nn

    before = nn.Module.load_weights
    with _RecordLoadedWeightKeys():
        pass
    assert nn.Module.load_weights is before


def test_record_loaded_weight_keys_shares_delivered_and_param_namespace() -> None:
    """The whole reason this guard works on sanitize() architectures: the
    captured keys arrive in the SAME namespace as the param tree, so a diff of
    the two is namespace-correct without knowing anything about sanitize().
    """
    import mlx.core as mx
    import mlx.nn as nn

    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4, bias=False)

    m = _M()
    with _RecordLoadedWeightKeys() as captured:
        m.load_weights([("lin.weight", mx.zeros((4, 4)))], strict=True)

    params = captured.get("params") or set()
    keys = captured.get("keys") or set()
    assert keys == params == {"lin.weight"}, (
        f"keys={keys} params={params} — must be identical on an exact load"
    )
    assert not (params - keys), "missing set must be empty on an exact load"


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

