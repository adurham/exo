"""Regression test: the DSpark load guard must not report FAIL on a correctly
loaded QUANTIZED head.

The bug (observed live on Vision-Exp, 2026-09-27):
    [DSPARK-GUARD] param_tree=118/84 missing=0 extra=34 param_tree_assert=FAIL

``_log_dspark_load_guard`` diffed the ALREADY-QUANTIZED attached module against a
FRESHLY-CONSTRUCTED, UNQUANTIZED ``DeepseekV4DSparkModule``. nn.quantize adds a
``.scales`` key per quantized projection, so the loaded side legitimately had
more keys; the guard counted them as ``extra`` and flipped the assert to FAIL on
every load, forever -- while ``missing=0`` (did every needed parameter arrive?)
was passing. 118 - 84 == 34 == the overlay's own "25 mxfp8 + 9 mxfp4" tally.

These tests exercise the normalization directly, with no mlx/model dependency,
so they run on the Linux gateway box.
"""

from __future__ import annotations


def _norm_quant(keys, _quant_suffixes=(".scales", ".biases")):
    """Mirror of utils_mlx._log_dspark_load_guard's inner normalizer.

    A quantized ``X.weight`` appears as ``X.weight`` + ``X.scales``, so a
    dropped quantization suffix maps to ``X.weight`` -- NOT to a bare ``X``.
    """
    out, n = set(), 0
    for k in keys:
        for sfx in _quant_suffixes:
            if k.endswith(sfx):
                out.add(k[: -len(sfx)] + ".weight")
                n += 1
                break
        else:
            out.add(k)
    return out, n


def _tree_ok(loaded_keys, expected_keys):
    """Mirror of the guard's comparison, post-fix."""
    loaded, n_quant = _norm_quant(loaded_keys)
    expected, _ = _norm_quant(expected_keys)
    missing = expected - loaded
    extra = loaded - expected
    return (not missing and not extra), missing, extra, n_quant


# A small but structurally faithful DSpark head: stage-scoped projections on the
# unquantized side, each gaining a `.scales` sibling on the quantized side.
_UNQUANTIZED = {
    "main_proj.weight",
    "main_norm.weight",
    "norm.weight",
    "markov_embed.weight",
    "markov_head.weight",
    "confidence_proj.weight",
    "stages.0.attn.wkv.weight",
    "stages.0.attn.wo_a.weight",
    "stages.0.attn.wo_b.weight",
    "stages.0.attn.wq_a.weight",
    "stages.0.attn.wq_b.weight",
    "stages.0.attn.kv_norm.weight",
    "stages.0.attn.q_norm.weight",
    "stages.0.ffn.shared_experts.down_proj.weight",
    "stages.0.ffn.shared_experts.gate_proj.weight",
    "stages.0.ffn.shared_experts.up_proj.weight",
    "stages.0.ffn.experts.gate_proj.weight",
    "stages.0.ffn.experts.up_proj.weight",
    "stages.0.ffn.experts.down_proj.weight",
    "stages.0.norm.weight",
    "stages.1.attn.wkv.weight",
    "stages.1.ffn.shared_experts.gate_proj.weight",
    "stages.2.attn.wkv.weight",
    "stages.2.ffn.shared_experts.down_proj.weight",
}

# which of those get quantized (and thus gain `.scales`)
_QUANTIZED = {
    k for k in _UNQUANTIZED
    if k.endswith(".weight")
    and not k.endswith(("_norm.weight", ".norm.weight"))
    and "experts." not in k or k.endswith(("gate_proj.weight", "down_proj.weight"))
}


def _quantized_side():
    """The loaded side: every quantized projection carries `.weight` + `.scales`."""
    out = set(_UNQUANTIZED)
    for k in _UNQUANTIZED:
        if k in _QUANTIZED and (".attn." in k or "shared_experts" in k
                                or "main_proj" in k or "markov" in k
                                or "confidence" in k):
            out.add(k[: -len(".weight")] + ".scales")
    return out


def test_quantized_head_is_not_flagged_as_fail():
    """THE REGRESSION: a fully-loaded quantized head must PASS."""
    loaded = _quantized_side()
    ok, missing, extra, n_quant = _tree_ok(loaded, _UNQUANTIZED)
    assert n_quant > 0, "fixture must actually exercise the quant path"
    assert not missing, f"no parameter should be missing, got {sorted(missing)}"
    assert not extra, f"quant artifacts must not be reported as extra: {sorted(extra)}"
    assert ok, "a correctly-loaded quantized head must NOT report FAIL"


def test_all_reported_extras_are_quantization_keys():
    """The bug's signature: every 'extra' key ends in a quantization suffix."""
    loaded = _quantized_side()
    raw_extra = loaded - _UNQUANTIZED
    assert raw_extra, "fixture must produce extras for the pre-fix comparison"
    for k in raw_extra:
        assert k.endswith((".scales", ".biases")), (
            f"unexpected non-quant extra key {k!r} — the normalization would"
            " hide a genuine mismatch")


def test_genuine_missing_key_still_fails():
    """Negative control: a real missing parameter must still FAIL."""
    loaded = _quantized_side()
    expected = set(_UNQUANTIZED) | {"stages.0.attn.wq_z.weight"}
    ok, missing, extra, _ = _tree_ok(loaded, expected)
    assert not ok, "a genuinely missing parameter must still FAIL"
    assert missing == {"stages.0.attn.wq_z.weight"}


def test_genuine_extra_key_still_fails():
    """Negative control: a real unexpected parameter must still FAIL."""
    loaded = _quantized_side() | {"stages.0.attn.wq_z.weight"}
    ok, missing, extra, _ = _tree_ok(loaded, _UNQUANTIZED)
    assert not ok, "a genuinely extra parameter must still FAIL"
    assert extra == {"stages.0.attn.wq_z.weight"}


def test_scales_maps_to_weight_not_bare_stem():
    """`X.scales` must normalize to `X.weight`, not to a bare `X`.

    Getting this wrong makes the fix silently useless (bare stems never match
    the `X.weight` keys on the expected side, so the extras survive).
    """
    loaded, n = _norm_quant({"stages.0.attn.wkv.scales"})
    assert loaded == {"stages.0.attn.wkv.weight"}, loaded
    assert n == 1


def test_biases_suffix_also_normalized():
    loaded, n = _norm_quant({"stages.0.ffn.experts.gate_proj.biases"})
    assert loaded == {"stages.0.ffn.experts.gate_proj.weight"}, loaded
    assert n == 1


def test_non_quant_keys_pass_through_unchanged():
    keys = {"main_norm.weight", "norm.weight", "stages.0.attn.attn_sink"}
    loaded, n = _norm_quant(keys)
    assert loaded == keys
    assert n == 0
