"""Repo-root pytest guard against the shared-venv `.pth` import landmine.

BACKGROUND (Task B2): on the Mac Studio cluster nodes, the shared venv
(``/Users/adam.durham/repos/exo/.venv``) carries site-packages `.pth` files
(``exo.pth`` -> ``/Users/adam.durham/repos/exo/src``,
``_editable_impl_exo_bench.pth``, ``_editable_impl_exo_tools.pth``) installed
by `uv sync` against the LIVE production checkout at
``/Users/adam.durham/repos/exo``. `.pth` entries get appended to `sys.path`
at interpreter startup, unconditionally -- they are not affected by
`PYTHONPATH` ordering the way a normal path entry is, and they are always
active in that venv regardless of which repo checkout you actually intend to
test. So `import exo` (and, symmetrically, `import mlx_lm` if this repo's own
`mlx-lm/` is not placed first on `PYTHONPATH`) can silently resolve to that
OLD checkout instead of whatever worktree/scratch copy a test dispatch is
actually meant to exercise. A green run against the wrong checkout is not a
passing test -- it is a silently WRONG one, and the failure mode is
invisible: nothing crashes, nothing warns, the test output just quietly
describes different code than the diff under review.

THE FIX: this conftest.py is git-tracked, so it ships to every future
checkout/worktree and cannot be forgotten the way a PYTHONPATH convention
can. It runs once per pytest session (`pytest_configure`, effectively at
collection time) and asserts that `exo` and `mlx_lm` -- whichever is actually
importable in the current environment -- resolve to files INSIDE this repo
root. If either resolves outside it, this raises immediately with a message
naming the resolved path and the exact PYTHONPATH fix, converting a
silent-wrong-results landmine into a loud, immediate, first-line failure for
every future test dispatch (both the `src/exo/**/tests/` trees and
`mlx-lm/tests/` -- pytest walks up from each collected test file's directory
looking for `conftest.py`, so this one root file covers both without needing
a second copy under `mlx-lm/`).

Deliberately does NOT touch the Studios' shared venv (leaving `exo.pth` etc.
alone): venv state is untracked and would silently regress on the next
`uv sync`, whereas this file travels with the repo.

PROOF THIS FIRES (paste from a real run):
  WRONG PYTHONPATH (only mlx-lm on it, so `exo` resolves via the shared
  venv's exo.pth to the OTHER, unrelated checkout):
    PYTHONPATH=<this-repo>/mlx-lm <shared-venv>/bin/python -m pytest \\
      mlx-lm/tests/test_deepseek_v4_gate.py -q
    -> RuntimeError at pytest_configure, naming the wrong resolved path.

  CORRECT PYTHONPATH (this repo's own src/ placed first):
    PYTHONPATH=<this-repo>/src:<this-repo>/mlx-lm <shared-venv>/bin/python \\
      -m pytest mlx-lm/tests/test_deepseek_v4_gate.py -q
    -> guard passes silently, tests run against the intended checkout.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent


def _assert_resolves_inside_repo(module_name: str) -> None:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        # Not importable at all in this environment -- nothing to guard.
        # (E.g. a bare mlx-lm checkout with no `exo` package anywhere on
        # sys.path.) A test that genuinely needs it will raise its own,
        # unambiguous ImportError later; that is not this landmine.
        return

    module_file = getattr(module, "__file__", None)
    if module_file is None:
        # Namespace package or similar -- no single file to check against;
        # do not block collection over something this check can't verify.
        return

    module_path = Path(module_file).resolve()
    try:
        module_path.relative_to(_REPO_ROOT)
    except ValueError:
        raise pytest.UsageError(
            f"TEST-INFRASTRUCTURE LANDMINE: `{module_name}` resolved to "
            f"{module_path}, which is OUTSIDE this repo checkout "
            f"({_REPO_ROOT}). On the Mac Studio cluster nodes this almost "
            "always means the shared venv's site-packages `.pth` entries "
            "(e.g. exo.pth pointing at the LIVE production checkout at "
            "/Users/adam.durham/repos/exo/src) won the import race because "
            f"this repo's own src/ and/or mlx-lm/ were not placed FIRST on "
            "PYTHONPATH. Continuing would silently test a DIFFERENT "
            "checkout than the one you intend to exercise -- a false pass "
            "or false failure with no indication anything is wrong. Fix: "
            "put this repo's own src/ and mlx-lm/ first on PYTHONPATH, e.g."
            ":\n"
            f"  PYTHONPATH={_REPO_ROOT / 'src'}:{_REPO_ROOT / 'mlx-lm'} "
            "<venv>/bin/python -m pytest ..."
        ) from None


def pytest_configure(config: object) -> None:  # noqa: ARG001 (pytest hook signature)
    _assert_resolves_inside_repo("exo")
    _assert_resolves_inside_repo("mlx_lm")

