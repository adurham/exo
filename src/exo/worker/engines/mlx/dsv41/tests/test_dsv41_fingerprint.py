"""CPU-only tests for the DSv4.1 park build fingerprint + refusal policy (§G2).

No GPU, no checkpoint, no cluster, no network. Everything is synthetic: tiny
fake ``config.json`` / weight / tokenizer files under ``tmp_path``, and hand-
built :class:`BuildFingerprint` records for the policy table. Pins:

* ``compute_fingerprint`` is deterministic, JSON-safe, never raises on missing
  inputs, and its ``model_fp`` / ``tokenizer_fp`` CHANGE when a file inside the
  model or tokenizer path is edited (the whole point of the fingerprint);
* ``layout_fingerprint`` is deterministic and sensitive to each axis;
* ``check_manifest`` returns ``(True, None)`` on a match, ``fp_mismatch`` per
  changed KV-affecting field, ``legacy`` for a missing/ malformed fingerprint,
  ``mlx_drift`` for an MLX-only change, and honours ``allow_mlx_drift`` /
  ``allow_stale`` exactly (stale overrides everything; drift overrides only
  MLX);
* the env flag readers default to False;
* the CI guard: :func:`allowlist_problems` is empty for the real import graph
  (allowlist completeness -- plan risk R1), and it DOES detect a planted
  offender while ignoring test/support scaffolding.
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from exo.worker.engines.mlx.dsv41 import fingerprint as fp

# ---------------------------------------------------------------------------
# fixtures: tiny fake model + tokenizer dirs
# ---------------------------------------------------------------------------


def _fake_model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps({"model_type": "deepseek_v41", "hidden_size": 64, "n_layers": 8})
    )
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00001.safetensors"}})
    )
    (d / "model-00001-of-00001.safetensors").write_bytes(b"\x00" * 128)
    (d / "quantize_config.json").write_text(json.dumps({"bits": 2.9}))
    # A file that must NOT matter (not config/index/weight): README noise.
    (d / "README.md").write_text("hello")
    return d


def _fake_tok_dir(tmp_path: Path) -> Path:
    d = tmp_path / "tokenizer"
    d.mkdir()
    (d / "tokenizer.json").write_text(json.dumps({"v": 1}))
    (d / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 1}))
    (d / "chat_template.jinja").write_text("{{ messages }} ANSWER:")
    return d


def _fp(**over: str) -> fp.BuildFingerprint:
    """A reference fingerprint with an optional single-field override."""
    base = fp.BuildFingerprint(
        model_fp="model",
        tokenizer_fp="tok",
        kv_code_fp="kv",
        mlx_version="0.1",
        layout_fp="layout",
    )
    return replace(base, **over) if over else base


# ---------------------------------------------------------------------------
# compute_fingerprint
# ---------------------------------------------------------------------------


def test_compute_fingerprint_deterministic_and_json_safe(tmp_path: Path) -> None:
    model, tok = _fake_model_dir(tmp_path), _fake_tok_dir(tmp_path)
    a = fp.compute_fingerprint(model_path=model, tokenizer=tok)
    b = fp.compute_fingerprint(model_path=model, tokenizer=tok)
    assert a == b
    fields = a.as_dict()
    assert set(fields) == {
        "model_fp",
        "tokenizer_fp",
        "kv_code_fp",
        "mlx_version",
        "layout_fp",
    }
    assert all(isinstance(v, str) and v for v in fields.values())
    # JSON-safe: the manifest stores this directly.
    assert json.loads(json.dumps(fields)) == fields


def test_compute_fingerprint_never_raises_without_inputs() -> None:
    got = fp.compute_fingerprint()  # nothing at all
    assert isinstance(got, fp.BuildFingerprint)
    assert all(isinstance(v, str) for v in got.as_dict().values())
    # A nonexistent path degrades, it does not raise.
    missing = fp.compute_fingerprint(
        model_path="/no/such/model", tokenizer="/no/such/tok"
    )
    assert isinstance(missing, fp.BuildFingerprint)
    assert missing.model_fp != got.model_fp  # a missing path is not "no path"


def test_model_config_edit_changes_model_fp(tmp_path: Path) -> None:
    model, tok = _fake_model_dir(tmp_path), _fake_tok_dir(tmp_path)
    before = fp.compute_fingerprint(model_path=model, tokenizer=tok)
    (model / "config.json").write_text(
        json.dumps({"model_type": "deepseek_v41", "hidden_size": 64, "n_layers": 9})
    )
    after = fp.compute_fingerprint(model_path=model, tokenizer=tok)
    assert before.model_fp != after.model_fp
    # Only the model hash moves.
    assert before.tokenizer_fp == after.tokenizer_fp
    assert before.kv_code_fp == after.kv_code_fp
    assert before.layout_fp == after.layout_fp


def test_weight_tensor_size_change_changes_model_fp(tmp_path: Path) -> None:
    model, tok = _fake_model_dir(tmp_path), _fake_tok_dir(tmp_path)
    before = fp.compute_fingerprint(model_path=model, tokenizer=tok)
    shard = model / "model-00001-of-00001.safetensors"
    shard.write_bytes(shard.read_bytes() + b"\x01" * 16)  # size changes
    after = fp.compute_fingerprint(model_path=model, tokenizer=tok)
    assert before.model_fp != after.model_fp


def test_unrelated_file_in_model_dir_does_not_change_model_fp(tmp_path: Path) -> None:
    model, tok = _fake_model_dir(tmp_path), _fake_tok_dir(tmp_path)
    before = fp.compute_fingerprint(model_path=model, tokenizer=tok)
    (model / "README.md").write_text("goodbye")  # not config/index/weight
    after = fp.compute_fingerprint(model_path=model, tokenizer=tok)
    assert before.model_fp == after.model_fp


def test_tokenizer_file_edit_changes_tokenizer_fp(tmp_path: Path) -> None:
    model, tok = _fake_model_dir(tmp_path), _fake_tok_dir(tmp_path)
    before = fp.compute_fingerprint(model_path=model, tokenizer=tok)
    (tok / "chat_template.jinja").write_text("{{ messages }} CHANGED:")
    after = fp.compute_fingerprint(model_path=model, tokenizer=tok)
    assert before.tokenizer_fp != after.tokenizer_fp
    assert before.model_fp == after.model_fp
    assert before.kv_code_fp == after.kv_code_fp


# ---------------------------------------------------------------------------
# layout_fingerprint
# ---------------------------------------------------------------------------

_LAYOUT_BASE: dict[str, Any] = dict(
    world_size=2, rank=0, layers=[1, 2, 3], dtype="bfloat16", window=128, bsz=1
)


def test_layout_fingerprint_deterministic() -> None:
    assert fp.layout_fingerprint(**_LAYOUT_BASE) == fp.layout_fingerprint(
        **_LAYOUT_BASE
    )


@pytest.mark.parametrize(
    "override",
    [
        {"world_size": 1},
        {"rank": 1},
        {"layers": [1, 2, 4]},
        {"dtype": "float16"},
        {"window": 256},
        {"bsz": 2},
    ],
)
def test_layout_fingerprint_sensitive_to_each_axis(override: dict[str, Any]) -> None:
    base = fp.layout_fingerprint(**_LAYOUT_BASE)
    assert fp.layout_fingerprint(**{**_LAYOUT_BASE, **override}) != base


def test_layout_fingerprint_layer_order_insensitive_but_none_differs() -> None:
    shuffled = {**_LAYOUT_BASE, "layers": [3, 1, 2, 2]}  # order + dupes
    assert fp.layout_fingerprint(**shuffled) == fp.layout_fingerprint(**_LAYOUT_BASE)
    full = fp.layout_fingerprint(**{**_LAYOUT_BASE, "layers": None})
    assert full != fp.layout_fingerprint(**_LAYOUT_BASE)


# ---------------------------------------------------------------------------
# check_manifest
# ---------------------------------------------------------------------------


def test_check_manifest_match() -> None:
    cur = _fp()
    ok, reason = fp.check_manifest({fp.FINGERPRINT_FIELD: cur.as_dict()}, cur)
    assert ok is True and reason is None


@pytest.mark.parametrize(
    "field",
    [
        "model_fp",
        "tokenizer_fp",
        "kv_code_fp",
        "layout_fp",
    ],
)
def test_check_manifest_each_field_mismatch(field: str) -> None:
    cur = _fp()
    man = {fp.FINGERPRINT_FIELD: _fp(**{field: "OTHER"}).as_dict()}
    assert fp.check_manifest(man, cur) == (False, "fp_mismatch")


def test_check_manifest_missing_fingerprint_is_legacy() -> None:
    cur = _fp()
    assert fp.check_manifest({}, cur) == (False, "legacy")
    assert fp.check_manifest({"magic": "DSV41-PARK", "version": 1}, cur) == (
        False,
        "legacy",
    )


def test_check_manifest_malformed_fingerprint_is_legacy() -> None:
    cur = _fp()
    for bad in ("nope", 123, {"model_fp": "model"}, []):
        assert fp.check_manifest({fp.FINGERPRINT_FIELD: bad}, cur) == (False, "legacy")


def test_check_manifest_non_mapping_is_legacy() -> None:
    assert fp.check_manifest(None, _fp()) == (False, "legacy")  # type: ignore[arg-type]


def test_check_manifest_mlx_only_mismatch_is_drift() -> None:
    cur = _fp()
    man = {fp.FINGERPRINT_FIELD: _fp(mlx_version="9.9").as_dict()}
    assert fp.check_manifest(man, cur) == (False, "mlx_drift")


def test_check_manifest_allow_mlx_drift_overrides_mlx_only() -> None:
    cur = _fp()
    drift = {fp.FINGERPRINT_FIELD: _fp(mlx_version="9.9").as_dict()}
    assert fp.check_manifest(drift, cur, allow_mlx_drift=True) == (True, None)
    # ...but it does NOT override a KV-affecting mismatch.
    mismatch = {
        fp.FINGERPRINT_FIELD: _fp(model_fp="OTHER", mlx_version="9.9").as_dict()
    }
    assert fp.check_manifest(mismatch, cur, allow_mlx_drift=True) == (
        False,
        "fp_mismatch",
    )


def test_check_manifest_fp_mismatch_beats_mlx_drift() -> None:
    cur = _fp()
    man = {fp.FINGERPRINT_FIELD: _fp(kv_code_fp="OTHER", mlx_version="9.9").as_dict()}
    assert fp.check_manifest(man, cur) == (False, "fp_mismatch")


@pytest.mark.parametrize(
    "man",
    [
        {},  # legacy
        {fp.FINGERPRINT_FIELD: "garbage"},  # malformed -> legacy
        {fp.FINGERPRINT_FIELD: _fp(mlx_version="9.9").as_dict()},  # mlx_drift
        {fp.FINGERPRINT_FIELD: _fp(tokenizer_fp="OTHER").as_dict()},  # fp_mismatch
        {fp.FINGERPRINT_FIELD: _fp().as_dict()},  # would-be match
    ],
)
def test_check_manifest_allow_stale_overrides_everything(man: dict[str, Any]) -> None:
    assert fp.check_manifest(man, _fp(), allow_stale=True) == (True, None)


def test_check_manifest_ignores_magic_and_version() -> None:
    # park.py owns the format version; check_manifest must not look at it.
    cur = _fp()
    man = {"magic": "OTHER", "version": 999, fp.FINGERPRINT_FIELD: cur.as_dict()}
    assert fp.check_manifest(man, cur) == (True, None)


# ---------------------------------------------------------------------------
# env flag readers
# ---------------------------------------------------------------------------


def test_env_flag_readers_default_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(fp.ENV_ALLOW_STALE, raising=False)
    monkeypatch.delenv(fp.ENV_ALLOW_MLX_DRIFT, raising=False)
    assert fp.allow_stale() is False
    assert fp.allow_mlx_drift() is False


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1", True),
        ("true", True),
        ("YES", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
    ],
)
def test_env_flag_readers_values(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool
) -> None:
    monkeypatch.setenv(fp.ENV_ALLOW_STALE, raw)
    monkeypatch.setenv(fp.ENV_ALLOW_MLX_DRIFT, raw)
    assert fp.allow_stale() is expected
    assert fp.allow_mlx_drift() is expected


# ---------------------------------------------------------------------------
# BuildFingerprint record
# ---------------------------------------------------------------------------


def test_buildfingerprint_from_dict_roundtrip() -> None:
    cur = _fp()
    assert fp.BuildFingerprint.from_dict(cur.as_dict()) == cur


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "x",
        {},
        {"model_fp": "model"},
        {
            "model_fp": 1,
            "tokenizer_fp": "t",
            "kv_code_fp": "k",
            "mlx_version": "0",
            "layout_fp": "l",
        },
    ],
)
def test_buildfingerprint_from_dict_rejects_malformed(bad: Any) -> None:
    assert fp.BuildFingerprint.from_dict(bad) is None


# ---------------------------------------------------------------------------
# allowlist guard
# ---------------------------------------------------------------------------


def test_allowlist_sets_disjoint_and_cover_the_families() -> None:
    assert fp.KV_ALLOWLIST, "allowlist must not be empty"
    assert fp.KV_ALLOWLIST.isdisjoint(fp.KV_NEUTRAL)
    # Spot-check that both families are represented.
    assert any(n.startswith("mlx_lm.models.deepseek_v41.") for n in fp.KV_ALLOWLIST)
    assert any(n.startswith("exo.worker.engines.mlx.dsv41.") for n in fp.KV_ALLOWLIST)


def test_allowlist_problems_is_empty() -> None:
    """The CI guard: every imported KV-family module is allowlisted or neutral."""
    assert fp.allowlist_problems() == []


def test_allowlist_guard_detects_planted_offender(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planted = {
        "exo.worker.engines.mlx.dsv41.zz_planted_offender",
        "mlx_lm.models.deepseek_v41.zz_planted_offender",
        "exo.worker.engines.mlx.disaggregated.zz_planted_offender",
    }
    for name in planted:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    # A test-shaped module in the family must be ignored (a new test cannot
    # fail the guard).
    ignored = "exo.worker.engines.mlx.dsv41.tests.test_zz_planted_offender"
    monkeypatch.setitem(sys.modules, ignored, types.ModuleType(ignored))

    problems = fp.allowlist_problems()
    for name in planted:
        assert name in problems, f"guard missed {name}"
    assert ignored not in problems
