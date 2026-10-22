# Copyright © 2026 Adam Durham (hermes-gw)
"""The engine's pack-format auto-detection (no model card field).

``load_dsv41`` must route an ``*-oQ*`` pack to ``oq_build`` and an exllamav3
pack to ``exl3_build`` from the checkpoint's own ``config.json``. These tests
pin that decision and the fact that both builders expose the same three entry
points, so nothing downstream of the build has to know which format loaded.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path


class PackKindTest(unittest.TestCase):
    def _dir(self, config: dict | None) -> Path:
        d = Path(tempfile.mkdtemp(prefix="oqkind_"))
        if config is not None:
            (d / "config.json").write_text(json.dumps(config))
        return d

    def test_detects_oq_from_the_config_block(self):
        from exo.worker.engines.mlx.dsv41.load import pack_kind

        self.assertEqual(pack_kind(self._dir({"omlx_deepseek_v41": {"version": 1}})),
                         "oq")

    def test_defaults_to_exl3(self):
        from exo.worker.engines.mlx.dsv41.load import pack_kind

        self.assertEqual(pack_kind(self._dir({"text_config": {}})), "exl3")
        # a missing or unreadable config must not raise at detection time
        self.assertEqual(pack_kind(self._dir(None)), "exl3")

    def test_builders_expose_the_same_entry_points(self):
        from exo.worker.engines.mlx.dsv41.load import _build_module

        ours = _build_module(self._dir({"omlx_deepseek_v41": {}}))
        theirs = _build_module(self._dir({}))
        for name in ("build_model", "build_block", "build_mtp"):
            self.assertTrue(callable(getattr(ours, name)), f"oq_build.{name}")
            self.assertTrue(callable(getattr(theirs, name)), f"exl3_build.{name}")
        self.assertEqual(ours.__name__.rsplit(".", 1)[-1], "oq_build")
        self.assertEqual(theirs.__name__.rsplit(".", 1)[-1], "exl3_build")

    def test_oq_checkpoint_refuses_a_non_oq_pack(self):
        from mlx_lm.models.deepseek_v41.oq_build import OqCheckpoint

        d = self._dir({"text_config": {}})
        (d / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {}}))
        with self.assertRaises(ValueError):
            OqCheckpoint(str(d))


if __name__ == "__main__":
    unittest.main()
