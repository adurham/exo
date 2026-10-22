# Copyright © 2026 Adam Durham (hermes-gw)
"""Coverage and A8 checks against the published pack's own headers.

These run offline: ``data/oq2.5e_text_index.json`` is the text stack's
safetensors header snapshot of
``mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-ngram-mtp`` (dtype, shape and
byte length per tensor -- no weights), so the structural claims below are made
against all 460 declared modules, not just a downloaded layer.

What is pinned:

* every ``quantized_modules`` key has exactly ``{weight, scales}`` or
  ``{weight, scales, biases}`` -- biases iff affine;
* no stray ``.scales``/``.biases`` tensor exists for an undeclared module;
* every *undeclared* ``.weight`` of a Linear is plain BF16 (the engram table is
  the one fp8 exception, and the loader never reads it);
* the packed geometry of each module agrees with its declared bits/group
  (``packed_cols * 32 / bits == scale_cols * group_size``);
* ``quantize_input`` is a pure runtime flag (the A8 question): 408 modules are
  flagged and not one of them stores an input-scale/smoothing tensor;
* MTP is quantized (mxfp4 experts, mxfp8 dense) with no biases anywhere.
"""

from __future__ import annotations

import json
import os
import unittest
from collections import Counter

from mlx_lm.models.deepseek_v41 import oq_build as oq

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class PackCoverageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(DATA, "oq2.5e_text_index.json")) as fh:
            cls.d = json.load(fh)
        cls.tensors = cls.d["tensors"]          # name -> [dtype, shape, nbytes]
        cls.quant = cls.d["omlx"]["quantized_modules"]
        cls.keys = set(cls.tensors)

    def _siblings(self, module: str) -> set:
        return {k[len(module) + 1:] for k in self.keys
                if k.startswith(module + ".") and "." not in k[len(module) + 1:]}

    def test_quantized_module_key_sets(self):
        want_affine = {"weight", "scales", "biases"}
        want_mxfp = {"weight", "scales"}
        bad = {}
        for name, qi in self.quant.items():
            want = want_affine if qi["mode"] == "affine" else want_mxfp
            got = self._siblings(name)
            if got != want:
                bad[name] = sorted(got)
        self.assertFalse(bad, f"{len(bad)} modules with a wrong key set: "
                              f"{list(bad.items())[:5]}")
        self.assertEqual(len(self.quant), 460)
        self.assertEqual(sum(1 for v in self.quant.values()
                             if v.get("quantize_input")), 408)

    def test_no_stray_scale_or_bias_tensors(self):
        # the two engram tables are declared in ``engram_tables``, not in
        # ``quantized_modules`` -- they are the only such exception
        declared = set(self.quant) | {
            k.replace("language_model.", "") for k in self.d["omlx"]["engram_tables"]}
        stray = []
        for k in self.keys:
            tail = k.rsplit(".", 1)[-1]
            if tail not in ("scales", "biases"):
                continue
            if k[: -len(tail) - 1] not in declared:
                stray.append(k)
        self.assertFalse(stray, f"{len(stray)} scale/bias tensors belong to no "
                                f"declared module: {stray[:5]}")

    def test_undeclared_linears_are_plain_bf16(self):
        plain = [k for k in self.keys
                 if k.endswith(".weight") and k[: -len(".weight")] not in self.quant]
        self.assertTrue(plain)
        offenders = {}
        for k in plain:
            if k.endswith("engram.embed.weight"):
                # the 384M-row fp8 table: read from the native release, never
                # from this pack (its two shards are 94 GiB each)
                self.assertEqual(self.tensors[k][0], "F8_E4M3")
                continue
            if self.tensors[k][0] != "BF16":
                offenders[k] = self.tensors[k][0]
        self.assertFalse(offenders,
                         f"undeclared weights that are not bf16: {offenders}")
        # spot-check the ones that matter: wo_a is one stacked grouped tensor
        for k in ("layers.0.attn.wo_a.weight", "layers.0.ffn.gate.weight",
                  "head.weight", "embed.weight"):
            self.assertEqual(self.tensors[k][0], "BF16", k)
        self.assertEqual(self.tensors["layers.0.attn.wo_a.weight"][1],
                         [8192, 4096])

    def test_packed_geometry_matches_metadata(self):
        """Two independent encodings (packed cols, scale cols) must agree."""
        bad = {}
        for name, qi in self.quant.items():
            w = self.tensors.get(f"{name}.weight")
            s = self.tensors.get(f"{name}.scales")
            if w is None or s is None:
                bad[name] = "missing"
                continue
            bits, group, _mode = oq._qparams(qi)
            in_elems = w[1][-1] * 32 // bits
            if s[1][-1] * group != in_elems:
                bad[name] = (f"packed {tuple(w[1])} -> {in_elems} in, scales "
                             f"{tuple(s[1])} -> {s[1][-1] * group} in")
            if s[1][:-1] != w[1][:-1]:
                bad[name] = f"rows {w[1][:-1]} vs {s[1][:-1]}"
        self.assertFalse(bad, f"{len(bad)} geometry mismatches: "
                              f"{list(bad.items())[:5]}")

    def test_a8_flag_is_runtime_only(self):
        rep = oq.a8_report(self.keys, self.quant)
        self.assertEqual(rep["quantize_input_modules"], 408)
        self.assertFalse(rep["key_set_mismatches"],
                         f"{len(rep['key_set_mismatches'])} flagged modules "
                         f"store more than weight/scales: "
                         f"{list(rep['key_set_mismatches'].items())[:5]}")
        self.assertFalse(rep["suspicious_tensors"],
                         f"smoothing/input-scale tensors present: "
                         f"{rep['suspicious_tensors'][:5]}")
        self.assertTrue(rep["runtime_only"])

    def test_mode_histogram(self):
        hist = Counter((v["mode"], v["bits"]) for v in self.quant.values())
        # body: affine g64 at 8-bit (dense) / 2-3 bit (routed experts)
        body = Counter((v["mode"], v["bits"]) for k, v in self.quant.items()
                       if k.startswith("layers."))
        self.assertEqual(set(body), {("affine", 8), ("affine", 2), ("affine", 3),
                                     ("mxfp8", 8)})
        # affine entries state group 64; the mxfp ones omit it and imply 32
        for name, qi in self.quant.items():
            _bits, group, mode = oq._qparams(qi)
            self.assertEqual(group, 64 if mode == "affine" else 32, name)
            if mode != "affine":
                self.assertIsNone(qi.get("group_size"), name)

    def test_mtp_is_quantized(self):
        mtp = {k: v for k, v in self.quant.items() if k.startswith("mtp.")}
        self.assertEqual(len(mtp), 31)          # 10 per stage + main_proj
        self.assertEqual({v["mode"] for v in mtp.values()}, {"mxfp4", "mxfp8"})
        for k, v in mtp.items():
            bits, group, mode = oq._qparams(v)
            self.assertEqual(group, 32, k)          # implied, not declared
            if ".ffn.experts." in k:
                self.assertEqual((mode, bits), ("mxfp4", 4), k)
            else:
                self.assertEqual((mode, bits), ("mxfp8", 8), k)
        # nothing quantized in mtp carries biases
        self.assertFalse([k for k in self.keys
                          if k.startswith("mtp.") and k.endswith(".biases")])
        # the markov / confidence heads and the draft norm are plain
        for k in ("mtp.2.markov_head.head.weight", "mtp.2.markov_head.embed.weight",
                  "mtp.2.confidence_head.proj.weight", "mtp.2.norm.weight"):
            self.assertEqual(self.tensors[k][0], "BF16", k)
        self.assertTrue(self.d["omlx"]["preserve_mtp"])

    def test_engram_tables_are_declared_and_not_read(self):
        et = self.d["omlx"]["engram_tables"]
        self.assertEqual(sorted(et), ["language_model.layers.1.engram.embed",
                                      "language_model.layers.14.engram.embed"])
        for name, meta in et.items():
            self.assertEqual((meta["mode"], meta["group_size"]), ("mxfp8", 32))
        self.assertTrue(self.d["omlx"]["engram_in_index"])
        # both tables live in the 94 GiB shards: reading them from this pack is
        # what the loader must never do (it uses LazyEngramTable on the native
        # release instead), and OqCheckpoint rejects fp8 outright.
        self.assertEqual(self.tensors["layers.1.engram.embed.weight"][0], "F8_E4M3")
        self.assertNotIn("F8_E4M3", oq._DTYPES)
        self.assertNotIn("F8_E8M0", oq._DTYPES)

    def test_total_size_sanity(self):
        total = self.d["total_text_bytes"] / 2 ** 30
        self.assertGreater(total, 300)      # the published pack is ~394 GiB
        self.assertLess(total, 420)


if __name__ == "__main__":
    unittest.main()
