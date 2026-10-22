# Copyright © 2026 Adam Durham (hermes-gw)
"""Fixtures for the DeepSeek-V4.1 native-MLX (oQ) loader tests.

Run from a worktree root:

    PYTHONPATH=$PWD/src:$PWD/mlx-lm .venv/bin/python -m pytest tests/dsv41_oq -q

The numeric tests against the published pack need a local *subset* of it (the
full pack is 393 GiB). Point ``EXO_DSV41_OQ_PACK`` at such a directory; the
default is ``~/.hermes/cache/scratch/oqmini``. Without it those tests skip --
the arithmetic, coverage and TP tests run anywhere.
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(ROOT, "mlx-lm"), os.path.join(ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@pytest.fixture(scope="session")
def synth_pack(tmp_path_factory) -> str:
    """A tiny oQ pack in the published layout (no download needed)."""
    import synth

    root = str(tmp_path_factory.mktemp("oqsynth"))
    return synth.write_synth_pack(root)


@pytest.fixture(scope="session")
def real_pack() -> str:
    """A downloaded subset of the published pack, or skip."""
    import synth

    path = synth.real_pack()
    if path is None:
        pytest.skip(
            "no local subset of mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-"
            f"ngram-mtp (set {synth.PACK_ENV} or fetch to {synth.DEFAULT_PACK})")
    return path


@pytest.fixture(scope="session")
def full_index() -> dict:
    """The published pack's text-stack headers: dtype/shape/nbytes per tensor."""
    import json

    with open(os.path.join(DATA, "oq2.5e_text_index.json")) as fh:
        return json.load(fh)
