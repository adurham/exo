#!/usr/bin/env python3
"""pytest wrapper for the Q2-Gamma offline self-test.

The repo-root conftest has a "landmine" guard that refuses to run from a worktree
whose ``mlx-lm/`` is empty, so bench tests must run with ``--noconftest``::

    cd <worktree>
    PYTHONPATH=bench /Users/adam.durham/repos/exo/.venv/bin/python -m pytest \\
        --noconftest bench/phase20_tests/test_phase20_q2_gamma.py -q -p no:cacheprovider

Each check from ``selftest.all_checks()`` becomes one parametrised test id, so a
failure names the exact gate that broke.
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
Q2DIR = os.path.abspath(os.path.join(HERE, "..", "phase20_q2_gamma"))
if Q2DIR not in sys.path:
    sys.path.insert(0, Q2DIR)

import selftest as ST  # noqa: E402

_CHECKS = ST.all_checks()
_IDS = [c[0] for c in _CHECKS]
_PARAMS = [(c[0], c[1], (c[2] if len(c) > 2 else "")) for c in _CHECKS]


@pytest.mark.parametrize("name,ok,detail", _PARAMS, ids=_IDS)
def test_q2_gamma_check(name, ok, detail):
    assert ok, f"{name}: {detail}"
