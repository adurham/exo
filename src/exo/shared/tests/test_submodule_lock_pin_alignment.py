"""Guard: every git submodule's committed gitlink must equal its ``uv.lock`` pin.

Why this file exists
====================
exo vendors ``mlx`` and ``mlx-lm`` as git submodules of forks carrying local
patches. There are TWO independent install paths, and they resolve the fork
from DIFFERENT places:

* **Deploy** (``start_cluster.sh``) runs ``uv sync ... --inexact
  --no-install-package mlx``, deliberately excluding mlx from the lock-driven
  install, then force-reinstalls ``./mlx`` and ``./mlx-lm`` from the LOCAL
  SUBMODULE CHECKOUT. The submodule is authoritative for what the cluster runs.
* **CI** (``.github/workflows/pipeline.yml``) builds ``.#exo-test-env``, which
  uv2nix resolves straight from ``uv.lock``. The lock pin is authoritative for
  what CI tests.

When the two disagree, CI validates code the cluster does not run -- a silent,
green-looking blind spot.

Why drift is the DEFAULT, not an accident
-----------------------------------------
A plain ``uv lock`` regeneration does **not** advance a git-URL pin's SHA.
Exo commit ``8a04cf492`` regenerated the lock and rewrote only the dev-date
suffix (``dev20260821+1c591e10`` -> ``dev20260823+1c591e10``) while the SHA
stayed stuck. Advancing a git pin requires an explicit
``uv lock --upgrade-package <name>``. Bumping a submodule gitlink therefore
leaves the lock behind unless someone remembers a second, non-obvious command.

This has already happened twice:

* round 2 (``b5ee1a113``): the ``mlx-lm`` pin was nine commits stale -- behind
  the very fix a mirrored regression guard was written to protect.
* round 3 (``9991126f9``): the ``mlx`` pin was two commits stale.

Both were found by hand. This file is the mechanical enforcement so there is
no round 4.

**If this test fails, the fix is:**
``uv lock --upgrade-package mlx`` (and/or ``mlx-lm``), then commit ``uv.lock``.

What is compared, and why
=========================
Three candidate sources of truth exist and they are NOT equivalent:

(a) the **committed gitlink** -- ``git ls-tree HEAD <path>``. What a fresh
    checkout resolves to; it is what is actually recorded in the repository.
(b) the **working-tree submodule HEAD** -- ``git -C <path> rev-parse HEAD``.
    What a developer (or the deploy path) has checked out right now. May
    legitimately differ from (a) mid-development, and does not exist at all
    when the submodule is not initialized.
(c) the **uv.lock pin**.

``test_committed_gitlink_matches_uv_lock_pin`` asserts **(a) == (c)**. That is
the comparison that belongs in CI:

* It is the repository's own content, so it is meaningful on any checkout,
  reviewable in a diff, and deterministic.
* ``actions/checkout@v4`` in ``pipeline.yml`` has **no ``submodules:`` key**,
  so submodule working trees are NOT populated on the runner. ``git ls-tree
  HEAD`` reads the committed tree object and works regardless; ``git -C mlx
  rev-parse HEAD`` would not. (a) is therefore the only viable comparison in
  CI, and there (a) == (b) by construction anyway on a pristine checkout.

``test_initialized_submodule_head_matches_committed_gitlink`` treats **(b) vs
(a)** as a SEPARATE, clearly-labelled condition rather than conflating it with
the lock check. A dirty working tree is a local-workspace fact, not a defect in
the committed repository, and it is the deploy path -- not CI -- that compiles
(b). That test additionally asserts each initialized submodule's working tree
is CLEAN: ``start_cluster.sh`` rsyncs working-tree *contents*, so HEAD matching
the gitlink is necessary but not sufficient to know what actually ships.

Parsing hazards this file deliberately avoids
=============================================
1. ``git submodule status`` output carries a ONE-CHARACTER STATUS FLAG in
   column 1 (space = clean, ``+`` = checkout differs from gitlink, ``-`` = not
   initialized, ``U`` = conflicts). Naive slicing eats the leading space and
   silently truncates the first character of the SHA -- turning
   ``e40a416b2085...`` into ``40a416b2085...``, a SHA that does not exist. That
   exact mistake manufactured a fictitious "divergent branches" conclusion
   during the round-3 investigation. This file never parses that command; it
   uses fixed-format plumbing (``git ls-tree -z``, ``git rev-parse``,
   ``git config -z --get-regexp``) exclusively.
2. The lock pin appears at 8+ separate locations for ``mlx`` alone, as a mix of
   full ``source = { git = "...#<sha>" }`` entries and abbreviated
   ``version = "0.32.1.dev...+<sha8>"`` local-version segments. A guard that
   checked only the first occurrence would pass while a partial skew hid in the
   rest. Every occurrence is collected and required to agree.

No-silent-skip policy
=====================
These tests never call ``pytest.skip``. A guard that skips when git is missing,
when ``.git`` is absent, when ``uv.lock`` cannot be found, or when a subprocess
fails is worse than no guard at all: it shows green forever while everyone
believes they are protected. Every such condition is a ``pytest.fail`` with a
diagnostic. The only state that is legitimately not-an-error is an
*uninitialized submodule working tree* in the (b)-vs-(a) test -- and that is
handled by narrowing that one comparison's subject, not by skipping the test,
so its enumeration and gitlink assertions still execute and still have teeth.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import final

import pytest

# Submodules that are intentionally NOT pinned in uv.lock (e.g. a vendored
# native library that is not a Python distribution). Adding a name here is a
# deliberate, reviewable opt-out: leaving a new submodule OUT of this set makes
# the guard fail loudly rather than let it escape unnoticed.
SUBMODULES_WITHOUT_LOCK_PINS: frozenset[str] = frozenset()

# Submodules that must always be present. Enumeration is dynamic (see
# `discover_submodules`) so a THIRD submodule is covered automatically, but a
# dynamic enumeration has one nasty failure mode: if it silently returns
# nothing, every loop body is skipped and the test passes vacuously. This floor
# makes that impossible. It is a minimum, not an equality -- new submodules do
# not need to be added here to be guarded.
REQUIRED_SUBMODULE_PATHS: frozenset[str] = frozenset({"mlx", "mlx-lm"})

FULL_SHA_PATTERN = re.compile(r"\A[0-9a-f]{40}\Z")
VERSION_ASSIGNMENT_PATTERN = re.compile(r'version = "([^"]+)"')
# A PEP 440 local-version segment such as "+e40a416b" (uv derives it from the
# resolved commit). Trailing segments after another "+" are not expected but are
# tolerated by anchoring only on the first hex run.
LOCAL_VERSION_SHA_PATTERN = re.compile(r"\+([0-9a-f]{7,40})\Z")

GIT_TIMEOUT_SECONDS = 60


@final
@dataclass(frozen=True)
class Submodule:
    """A submodule as declared in ``.gitmodules``."""

    path: str
    url: str


@final
@dataclass(frozen=True)
class LockPinOccurrence:
    """One place in ``uv.lock`` that pins a submodule's git URL to a commit."""

    line_number: int
    sha: str
    version: str | None


def find_repository_root() -> Path:
    """Locate the repo root by walking up from this file.

    Anchored on ``__file__`` rather than the working directory: CI invokes
    pytest as ``python -m pytest src ...`` from the checkout root, but a
    developer may invoke it from anywhere, and ``--import-mode=importlib``
    does not change ``__file__``.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / ".gitmodules").is_file() and (candidate / "uv.lock").is_file():
            return candidate
    pytest.fail(
        "Could not locate the exo repository root: no ancestor of "
        f"{here} contains both .gitmodules and uv.lock. This guard cannot "
        "verify submodule/uv.lock alignment without them, and refuses to pass "
        "vacuously. Ancestors searched: "
        + ", ".join(str(parent) for parent in here.parents)
    )


def run_git(repository_root: Path, arguments: list[str], purpose: str) -> str:
    """Run a git plumbing command, or fail the test with a diagnostic.

    Never skips. A missing/broken git is a broken guard, and a broken guard
    must be loud.
    """
    command = ["git", *arguments]
    try:
        completed = subprocess.run(
            command,
            cwd=repository_root,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError:
        pytest.fail(
            f"`git` executable not found while trying to {purpose}. This guard "
            "compares the committed submodule gitlinks against uv.lock and "
            "cannot do so without git. Failing loudly instead of skipping: a "
            "skip here would make submodule/lock drift invisible forever."
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"`{' '.join(command)}` timed out after {GIT_TIMEOUT_SECONDS}s while "
            f"trying to {purpose}."
        )
    if completed.returncode != 0:
        pytest.fail(
            f"`{' '.join(command)}` failed with exit code "
            f"{completed.returncode} while trying to {purpose}.\n"
            f"stdout: {completed.stdout!r}\nstderr: {completed.stderr!r}\n"
            "Failing loudly instead of skipping."
        )
    return completed.stdout


def discover_submodules(repository_root: Path) -> tuple[Submodule, ...]:
    """Enumerate submodules from ``.gitmodules`` dynamically.

    Uses ``git config -z --get-regexp`` (NUL-delimited, key on the first line,
    value on the rest) rather than an ad-hoc INI parse or, worse, slicing
    ``git submodule status``.
    """
    raw = run_git(
        repository_root,
        [
            "config",
            "--file",
            ".gitmodules",
            "-z",
            "--get-regexp",
            r"^submodule\..*\.(path|url)$",
        ],
        "enumerate submodules from .gitmodules",
    )

    paths: dict[str, str] = {}
    urls: dict[str, str] = {}
    for record in raw.split("\0"):
        if not record:
            continue
        key, _, value = record.partition("\n")
        # key looks like: submodule.<name>.path / submodule.<name>.url
        if key.startswith("submodule.") and key.endswith(".path"):
            paths[key[len("submodule.") : -len(".path")]] = value
        elif key.startswith("submodule.") and key.endswith(".url"):
            urls[key[len("submodule.") : -len(".url")]] = value

    submodules = tuple(
        Submodule(path=paths[name], url=urls[name])
        for name in sorted(paths)
        if name in urls
    )

    discovered_paths = {submodule.path for submodule in submodules}
    missing = REQUIRED_SUBMODULE_PATHS - discovered_paths
    if missing:
        pytest.fail(
            "Dynamic submodule enumeration did not find the submodules this "
            f"guard is known to protect: {sorted(missing)}. Discovered: "
            f"{sorted(discovered_paths)}. Either .gitmodules changed (update "
            "REQUIRED_SUBMODULE_PATHS deliberately) or the enumeration broke -- "
            "either way the guard would otherwise loop over nothing and pass "
            "vacuously, so it fails instead."
        )
    return submodules


def read_committed_gitlink(repository_root: Path, submodule_path: str) -> str:
    """Return source-of-truth (a): the gitlink SHA recorded in ``HEAD``'s tree.

    ``git ls-tree -z HEAD -- <path>`` emits ``<mode> SP <type> SP <sha> TAB
    <name> NUL``. Splitting on the TAB first and only then on whitespace keeps
    the SHA field intact regardless of what the name contains -- and, unlike
    ``git submodule status``, there is no leading status column to accidentally
    consume.

    This reads the committed tree, NOT the working copy, so it works even when
    submodules were never checked out -- which is exactly CI's situation.
    """
    raw = run_git(
        repository_root,
        ["ls-tree", "-z", "HEAD", "--", submodule_path],
        f"read the committed gitlink for submodule {submodule_path!r}",
    )
    records = [record for record in raw.split("\0") if record]
    if len(records) != 1:
        pytest.fail(
            f"Expected exactly one `git ls-tree HEAD -- {submodule_path}` "
            f"record, got {len(records)}: {records!r}."
        )
    metadata, tab, name = records[0].partition("\t")
    if not tab:
        pytest.fail(
            f"Malformed ls-tree record for {submodule_path!r} (no TAB "
            f"separator): {records[0]!r}"
        )
    fields = metadata.split()
    if len(fields) != 3:
        pytest.fail(
            f"Malformed ls-tree metadata for {submodule_path!r}: "
            f"{metadata!r} (expected '<mode> <type> <sha>')"
        )
    mode, object_type, sha = fields
    if mode != "160000" or object_type != "commit":
        pytest.fail(
            f"{submodule_path!r} is not a gitlink in HEAD: mode={mode!r} "
            f"type={object_type!r} name={name!r}. .gitmodules declares it as a "
            "submodule, so the committed tree disagreeing with .gitmodules is "
            "itself a defect."
        )
    if not FULL_SHA_PATTERN.match(sha):
        pytest.fail(
            f"ls-tree returned a non-SHA gitlink for {submodule_path!r}: "
            f"{sha!r}. (A truncated SHA here is the classic symptom of "
            "column-eating string slicing -- this parser splits on TAB and "
            "whitespace precisely to avoid that.)"
        )
    return sha


def read_working_tree_dirt(repository_root: Path, submodule_path: str) -> tuple[str, ...]:
    """Return porcelain status records for an initialized submodule working tree.

    ``git ls-tree``/``rev-parse`` only prove which COMMIT is checked out. The
    deploy path rsyncs working-tree *contents*, so a submodule can sit exactly
    on its gitlink and still ship uncommitted edits with no git trace of what
    actually ran. ``--porcelain`` is the stable, machine-readable format (v1),
    and ``-z`` makes it NUL-delimited so pathnames are never mangled.
    """
    raw = run_git(
        repository_root,
        ["-C", submodule_path, "status", "--porcelain", "-z"],
        f"check whether submodule {submodule_path!r} has a clean working tree",
    )
    # Records are NUL-terminated. Renames/copies emit an extra NUL-delimited
    # origin path, but for a dirtiness verdict counting non-empty records is
    # sufficient and cannot under-report.
    return tuple(record for record in raw.split("\0") if record)


def read_working_tree_head(repository_root: Path, submodule_path: str) -> str | None:
    """Return source-of-truth (b), or ``None`` when the submodule is not checked out.

    ``None`` is a legitimate state (a fresh clone without ``--recursive``, and
    CI itself, since ``actions/checkout@v4`` is configured without
    ``submodules:``). It is NOT an error, and it is NOT swallowed: the caller
    reports it explicitly.
    """
    submodule_root = repository_root / submodule_path
    # A populated submodule has a `.git` file (gitlink to ../.git/modules/...)
    # or, for older layouts, a `.git` directory. Absence means "never
    # initialized"; an empty directory is git's placeholder for the same thing.
    if not (submodule_root / ".git").exists():
        return None
    sha = run_git(
        repository_root,
        ["-C", submodule_path, "rev-parse", "HEAD"],
        f"read the working-tree HEAD of submodule {submodule_path!r}",
    ).strip()
    if not FULL_SHA_PATTERN.match(sha):
        pytest.fail(
            f"`git -C {submodule_path} rev-parse HEAD` returned a non-SHA: {sha!r}"
        )
    return sha


def collect_lock_pin_occurrences(
    lock_text: str, url: str
) -> tuple[tuple[LockPinOccurrence, ...], int]:
    """Collect EVERY pin of ``url`` in ``uv.lock``.

    Returns ``(pinned_occurrences, unpinned_occurrence_count)``.

    ``uv.lock`` references a git source in three shapes:

    1. ``source = { git = "<url>?branch=main#<40-hex sha>" }`` on its own line
       inside a ``[[package]]`` block.
    2. the same, inline inside a dependency entry:
       ``{ name = "mlx", version = "<v>", source = { git = "<url>...#<sha>" } }``
    3. ``{ name = "mlx", ..., git = "<url>?branch=main" }`` with NO ``#sha`` --
       the requirement spec under ``[package.metadata.requires-dist]``. These
       carry no commit and are counted separately, not treated as skew.

    Shapes 1 and 2 also carry a ``version`` string whose PEP 440 local segment
    (``+<sha8>``) is an abbreviation of the same commit; it is captured here so
    the caller can require it to agree too. For shape 1 the ``version`` sits on
    a preceding line of the same ``[[package]]`` block, so block context is
    tracked. mlx-lm's version has no local segment at all -- that is fine and
    the caller only checks segments that exist.
    """
    escaped = re.escape(url)
    # Optional "?query" (uv writes "?branch=main"), then optionally "#<sha>".
    reference_pattern = re.compile(
        escaped + r'(?P<query>\?[^"#]*)?(?:#(?P<sha>[0-9a-f]{40}))?"'
    )

    occurrences: list[LockPinOccurrence] = []
    unpinned_count = 0
    block_version: str | None = None

    for index, line in enumerate(lock_text.splitlines(), start=1):
        stripped = line.strip()
        if stripped == "[[package]]":
            block_version = None
        elif stripped.startswith("[") and stripped.endswith("]"):
            # Any other table header ends the package's own top-level keys.
            pass
        elif stripped.startswith("version = "):
            match = VERSION_ASSIGNMENT_PATTERN.search(stripped)
            if match is not None:
                block_version = match.group(1)

        for reference in reference_pattern.finditer(line):
            sha = reference.group("sha")
            if sha is None:
                unpinned_count += 1
                continue
            # Prefer a `version = "..."` appearing earlier on the SAME line
            # (shape 2); otherwise fall back to the enclosing block (shape 1).
            inline_version: str | None = None
            for version_match in VERSION_ASSIGNMENT_PATTERN.finditer(line):
                if version_match.end() <= reference.start():
                    inline_version = version_match.group(1)
            occurrences.append(
                LockPinOccurrence(
                    line_number=index,
                    sha=sha,
                    version=inline_version if inline_version is not None else block_version,
                )
            )

    return tuple(occurrences), unpinned_count


def test_committed_gitlink_matches_uv_lock_pin() -> None:
    """(a) committed gitlink == (c) uv.lock pin, for EVERY submodule.

    This is the CI-relevant comparison: CI installs (c) and the cluster runs
    the submodule, so a divergence means CI is validating code the cluster does
    not run.
    """
    repository_root = find_repository_root()
    lock_path = repository_root / "uv.lock"
    lock_text = lock_path.read_text(encoding="utf-8")

    submodules = discover_submodules(repository_root)
    assert submodules, "no submodules discovered (should be unreachable)"

    failures: list[str] = []
    checked_paths: list[str] = []

    for submodule in submodules:
        gitlink_sha = read_committed_gitlink(repository_root, submodule.path)
        occurrences, unpinned_count = collect_lock_pin_occurrences(
            lock_text, submodule.url
        )

        if not occurrences:
            if submodule.path in SUBMODULES_WITHOUT_LOCK_PINS:
                continue
            failures.append(
                f"[{submodule.path}] no commit-pinned occurrence of "
                f"{submodule.url!r} found anywhere in uv.lock "
                f"({unpinned_count} unpinned reference(s) seen). A submodule "
                "that CI never installs from the lock is exactly the blind "
                "spot this guard exists to prevent. If that is intentional, "
                "add its path to SUBMODULES_WITHOUT_LOCK_PINS with a comment "
                "explaining why."
            )
            continue

        checked_paths.append(submodule.path)

        # Trap #2: the pin appears many times. Require ALL of them to agree
        # with the gitlink, not just the first.
        skewed = [
            occurrence for occurrence in occurrences if occurrence.sha != gitlink_sha
        ]
        if skewed:
            detail = ", ".join(
                f"line {occurrence.line_number}: {occurrence.sha}"
                for occurrence in skewed
            )
            failures.append(
                f"[{submodule.path}] committed gitlink is {gitlink_sha}, but "
                f"{len(skewed)} of {len(occurrences)} uv.lock pin(s) disagree "
                f"({detail}). Fix: `uv lock --upgrade-package "
                f"{submodule.path}` then commit uv.lock. NOTE a plain "
                "`uv lock` will NOT advance a git pin's SHA."
            )

        # The abbreviated SHA embedded in the PEP 440 local version segment
        # must also agree -- a partial skew can hide there.
        for occurrence in occurrences:
            if occurrence.version is None:
                continue
            local_match = LOCAL_VERSION_SHA_PATTERN.search(occurrence.version)
            if local_match is None:
                continue
            abbreviated = local_match.group(1)
            if not gitlink_sha.startswith(abbreviated):
                failures.append(
                    f"[{submodule.path}] uv.lock line {occurrence.line_number} "
                    f"carries version {occurrence.version!r}, whose local "
                    f"segment {abbreviated!r} is not a prefix of the committed "
                    f"gitlink {gitlink_sha}."
                )

    assert not failures, (
        "Submodule gitlink / uv.lock pin divergence detected.\n\n"
        + "\n\n".join(failures)
        + "\n\nWhy this matters: start_cluster.sh force-reinstalls ./mlx and "
        "./mlx-lm from the LOCAL SUBMODULE, while CI's nix exo-test-env "
        "installs whatever uv.lock pins. When they diverge, CI validates code "
        "the cluster does not run."
    )

    assert checked_paths, (
        "No submodule was actually checked against uv.lock -- the guard would "
        "have passed vacuously."
    )


def test_initialized_submodule_head_matches_committed_gitlink() -> None:
    """(b) working-tree submodule HEAD == (a) committed gitlink, where (b) exists.

    Deliberately SEPARATE from the uv.lock check. A checked-out submodule that
    differs from the gitlink is a *local workspace* condition -- normal
    mid-development, not a defect in the committed repository -- but it is
    also precisely what ``start_cluster.sh`` compiles and ships to the
    cluster, so it is worth surfacing distinctly rather than conflating with a
    lock mismatch.

    On CI this comparison has no subject: ``actions/checkout@v4`` in
    ``pipeline.yml`` declares no ``submodules:`` key, so the working trees are
    empty. The test does NOT skip -- it still enumerates submodules and reads
    every committed gitlink (both of which fail loudly if git or .git is
    unavailable), and only the HEAD comparison itself is narrowed to the
    submodules that are actually present.
    """
    repository_root = find_repository_root()
    submodules = discover_submodules(repository_root)

    mismatches: list[str] = []
    dirty: list[str] = []
    uninitialized: list[str] = []
    compared: list[str] = []

    for submodule in submodules:
        gitlink_sha = read_committed_gitlink(repository_root, submodule.path)
        working_head = read_working_tree_head(repository_root, submodule.path)
        if working_head is None:
            uninitialized.append(submodule.path)
            continue
        compared.append(submodule.path)
        if working_head != gitlink_sha:
            mismatches.append(
                f"[{submodule.path}] working-tree HEAD {working_head} != "
                f"committed gitlink {gitlink_sha}"
            )
        dirt = read_working_tree_dirt(repository_root, submodule.path)
        if dirt:
            preview = ", ".join(dirt[:5])
            suffix = f" (+{len(dirt) - 5} more)" if len(dirt) > 5 else ""
            dirty.append(
                f"[{submodule.path}] {len(dirt)} uncommitted working-tree "
                f"change(s): {preview}{suffix}"
            )

    print(
        "submodule working-tree check: compared="
        f"{compared or '[]'} uninitialized(not-an-error)={uninitialized or '[]'}"
    )

    assert not mismatches and not dirty, (
        "Submodule working tree does not match what HEAD records:\n"
        + "\n".join([*mismatches, *dirty])
        + "\n\nThis is a WORKING-TREE condition, NOT a uv.lock problem -- the "
        "uv.lock alignment is asserted separately by "
        "test_committed_gitlink_matches_uv_lock_pin.\n"
        "Why it matters: start_cluster.sh rsyncs this working tree to the "
        "nodes and force-reinstalls ./mlx and ./mlx-lm from it, so a deploy "
        "from here would ship code that no commit records -- leaving no git "
        "trace of what the cluster actually ran.\n"
        "Resolve by committing the change (`git -C <path> commit` then "
        "`git add <path>` in exo), or by restoring the pinned state "
        "(`git submodule update --init <path>`)."
    )
