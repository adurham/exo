#!/usr/bin/env python3
"""Static lint for §11 ("How to reproduce") of docs/dsv41-cluster-writeup-2026-10-17/WRITEUP.md.

WHAT THIS PROVES
  The §11 instructions are *well-formed and not stale* at the SHA §11 pins:
  every path §11 names exists in that tree, every CLI flag it relies on is
  accepted by the parser at that SHA, the line/symbol cross-references it makes
  still land on the lines it claims, and §11's hardcoded-constants note matches
  the actual NODES / peer-ID constants in start_cluster.sh at that SHA.

WHAT THIS DOES NOT PROVE -- do not read a PASS as more than it is
  Nothing here is launched. A PASS says nothing about whether that SHA boots,
  whether the hardcoded node IPs / libp2p peer IDs are correct for *any*
  particular pair of Mac Studios, whether two 128 GB nodes can actually hold the
  model, or whether any figure in §9 reproduces. Those need the hardware.

Checks
  1. the SHA pinned in §11's table resolves to a commit in this repo
  2. every path §11 references exists at that SHA (annotated `file:NN-MM`
     references must also be long enough to contain those lines)
  3. every CLI flag §11 uses is accepted: exo argparse definitions are read from
     `git show <sha>:src/exo/main.py` and fed §11's argv (plus the argv
     start_cluster.sh actually launches) without importing exo or launching
     anything; curl flags are checked against the local curl's own option table
  4. scripts/gen-prompts.py runs and its output passes
     scripts/validate-prompts.py (SKIPped, loudly, while those are uncommitted)
  5. §11's hardcoded-constants note is cross-referenced against the actual
     values at the SHA (NODES line, peer constants, wired-limit default,
     submodule gitlinks vs. the SHA table, uv.lock/pyproject pins)

On full success it rewrites a single dated status line inside §11 (replacing any
previous one, so repeated runs do not stack). --check-only never writes and is
what CI uses: it fails when the status line is missing or names a different SHA,
so the claim cannot go stale silently.

Exit codes: 0 all executed checks pass (SKIPs allowed unless
CHECK_S11_REQUIRE_PROMPTS=1), 1 a check failed, 2 the writeup/§11 could not be read.

Stdlib only; Python 3.9+. (Despite the .sh name -- §11's lint is named that way so it
sits next to the other scripts/ helpers -- this is a Python 3 program. Run it as
`scripts/check-s11.sh` or `python3 scripts/check-s11.sh`; both work.)
"""

from __future__ import print_function

import argparse
import datetime
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback

WRITEUP_REL = "docs/dsv41-cluster-writeup-2026-10-17/WRITEUP.md"
STATUS_RE = re.compile(r"^_Static-checked at .*_$")
SECTION_11_RE = re.compile(r"^##\s*11\.\s")
SECTION_12_RE = re.compile(r"^##\s*12\.\s")

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"

results = []  # list of (num, name, status, [detail lines])


def record(num, name, status, details):
    results.append((num, name, status, details or []))


def git(*args, **kwargs):
    """Run git in the repo; return (rc, stdout, stderr)."""
    cmd = ["git"] + list(args)
    p = subprocess.run(
        cmd,
        cwd=kwargs.get("cwd", REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return (
        p.returncode,
        p.stdout.decode("utf-8", "replace"),
        p.stderr.decode("utf-8", "replace"),
    )


def show(sha, path):
    """Return file content at sha:path, or None."""
    rc, out, _ = git("show", "%s:%s" % (sha, path))
    return out if rc == 0 else None


def tree_names(sha):
    rc, out, _ = git("ls-tree", "-r", "--name-only", sha)
    if rc != 0:
        return []
    return [ln for ln in out.splitlines() if ln]


def find_repo_root(start):
    p = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=start,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if p.returncode != 0:
        return None
    return p.stdout.decode().strip()


# ---------------------------------------------------------------- §11 extraction


def slice_section_11(lines):
    """Return (start_idx, end_idx) inclusive/exclusive body bounds of '## 11.'."""
    start = None
    for i, ln in enumerate(lines):
        if SECTION_11_RE.match(ln):
            start = i
            break
    if start is None:
        return None
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if SECTION_12_RE.match(lines[i]):
            end = i
            break
    return (start, end)


def fenced_blocks(lines, start, end):
    """Yield lists of logical command lines from ``` fences in the section."""
    blocks = []
    cur = None
    fence = None
    for i in range(start, end):
        ln = lines[i]
        m = re.match(r"^\s*```(\w*)\s*$", ln)
        if m and fence is None:
            fence = m.group(1) or "text"
            cur = []
            continue
        if fence is not None and re.match(r"^\s*```\s*$", ln):
            blocks.append((fence, cur))
            fence = None
            cur = None
            continue
        if cur is not None:
            cur.append(ln)
    return blocks


def logical_commands(block_lines):
    """Join backslash continuations, drop comments/blank lines, shlex-split."""
    joined = []
    buf = ""
    for ln in block_lines:
        s = ln.rstrip()
        if buf == "" and (s.strip() == "" or s.lstrip().startswith("#")):
            continue
        if s.endswith("\\"):
            buf += s[:-1] + " "
            continue
        buf += s
        joined.append(buf.strip())
        buf = ""
    if buf.strip():
        joined.append(buf.strip())

    import shlex

    out = []
    for cmd in joined:
        try:
            out.append(shlex.split(cmd, comments=True))
        except ValueError:
            out.append(cmd.split())
    return [c for c in out if c]


# ------------------------------------------------------- check 4 helpers (prompts)


def pick_python():
    for cand in ("python3.13", "python3.12", "python3.11", "python3"):
        p = shutil.which(cand)
        if p:
            return p
    return sys.executable


def locate_prompt_script(repo, sha, candidates):
    for rel in candidates:
        if os.path.exists(os.path.join(repo, rel)):
            return rel, "working tree"
    names = tree_names(sha)
    for rel in candidates:
        if rel in names:
            return rel, "pinned SHA only (not checked out)"
    base = [c.split("/")[-1] for c in candidates]
    for n in names:
        if n.split("/")[-1] in base:
            return n, "pinned SHA (path differs from scripts/: %s)" % n
    return None, None


# --------------------------------------------------------------------- check 1


def check_sha(lines, start, end, sha_override):
    body = "\n".join(lines[start:end])
    details = []
    sha = sha_override

    if sha is None:
        # §11.2 table row for exo: | exo | `github.com/<user>/exo` | `main` | `<sha>` |
        for ln in lines[start:end]:
            if re.match(r"^\|\s*exo\s*\|", ln):
                cells = [c.strip() for c in ln.strip().strip("|").split("|")]
                for c in reversed(cells):
                    cand = c.strip("` ")
                    if re.match(r"^[0-9a-f]{40}$", cand):
                        sha = cand
                        break
                if sha:
                    details.append("pinned SHA taken from §11.2 table: %s" % sha)
                break

    if sha is None:
        return FAIL, ["no 40-hex SHA found in the §11.2 table's exo row "
                      "(and no --sha given); §11 no longer states a pin"]

    if not re.match(r"^[0-9a-f]{40}$", sha):
        return FAIL, ["%r is not a full 40-hex commit SHA -- abbreviated SHAs go "
                      "stale on rebase/GC" % sha]

    rc, out, err = git("cat-file", "-e", "%s^{commit}" % sha)
    if rc != 0:
        return FAIL, ["git cat-file -e %s^{commit} failed: %s"
                      % (sha, (err or out).strip().splitlines()[:1])]

    rc, out, _ = git("rev-parse", "--verify", "%s^{commit}" % sha)
    details.append("resolves to commit %s" % out.strip())

    _, subj, _ = git("log", "-1", "--format=%h %ad %s", "--date=short", sha)
    details.append("commit: %s" % subj.strip())
    details.append("(mlx/mlx-lm row SHAs are foreign to this repo and cannot be "
                   "cat-file'd here -- check 5 verifies them against the submodule "
                   "gitlinks and uv.lock pins)")
    return PASS, details


# --------------------------------------------------------------------- check 2

PATH_TOKEN_RE = re.compile(
    r"`(?P<path>[A-Za-z0-9_./-]+\.(?:sh|py|toml|lock|md|json|yml|yaml|cfg|ini))"
    r"(?::(?P<lo>\d+)(?:-(?P<hi>\d+))?)?`"
    r"|`(?P<bpath>[A-Za-z0-9_./-]+\.(?:sh|py|toml|lock))`\s*:(?P<blo>\d+)(?:-(?P<bhi>\d+))?"
    r"|`:\s*(?P<elo>\d+)(?:-(?P<ehi>\d+))?`"
)


def iter_path_refs(body):
    """Yield (path, (lo, hi) or None) in document order.

    One pass, so an elided range like §11.2's "uv.lock:1382-1384 for mlx,
    :1499-1501 for mlx-lm" attaches to the path named just before it.
    """
    last_path = None
    for m in PATH_TOKEN_RE.finditer(body):
        path = m.group("path") or m.group("bpath")
        if path is None:
            if last_path is None:
                continue
            lo, hi = m.group("elo"), m.group("ehi")
            yield last_path, (int(lo), int(hi) if hi else int(lo))
            continue
        last_path = path
        lo = m.group("lo") or m.group("blo")
        hi = m.group("hi") or m.group("bhi")
        yield path, ((int(lo), int(hi) if hi else int(lo)) if lo else None)


def check_paths(sha, lines, start, end):
    details = []
    fails = []
    body_lines = lines[start:end]
    body = "\n".join(body_lines)

    refs = {}  # path -> set of (lo, hi)
    for path, rng in iter_path_refs(body):
        refs.setdefault(path, set())
        if rng:
            refs[path].add(rng)

    # §11's corpus claim ("the repo's own docs/**.md") needs the tree, not a file
    for extra_dir in ("docs",):
        rc, out, _ = git("ls-tree", sha, "--", extra_dir + "/")
        if rc != 0 or not out.strip():
            fails.append("directory %s/ referenced by §11 does not exist at %s"
                         % (extra_dir, sha[:12]))
        else:
            n_md = len([n for n in tree_names(sha)
                        if n.startswith(extra_dir + "/") and n.endswith(".md")])
            details.append("ok  %s/ tree exists (%d .md files -- §11's prompt corpus)"
                           % (extra_dir, n_md))

    names = tree_names(sha)
    name_set = set(names)
    by_base = {}
    for n in names:
        by_base.setdefault(n.split("/")[-1], []).append(n)

    for path in sorted(refs):
        clean = path.lstrip("./")
        exists = clean in name_set
        how = "exact"
        if not exists:
            rc, out, _ = git("ls-tree", sha, "--", clean)
            exists = bool(out.strip())
            how = "ls-tree"
        if not exists:
            base = clean.split("/")[-1]
            cands = by_base.get(base, [])
            if len(cands) == 1:
                clean = cands[0]
                exists = True
                how = "basename -> %s" % clean
            elif len(cands) > 1:
                fails.append("%s: ambiguous, candidates %s" % (path, cands))
                continue
        if not exists:
            fails.append("missing at %s: %s" % (sha[:12], path))
            continue

        content = show(sha, clean)
        nlines = len(content.splitlines()) if content is not None else 0
        rngs = refs[path]
        if not rngs:
            details.append("ok  %s (%s)" % (clean, how))
            continue
        for (lo, hi) in sorted(rngs):
            if hi > nlines:
                fails.append("%s:%d-%d cites lines past EOF (%d lines at %s)"
                             % (clean, lo, hi, nlines, sha[:12]))
            else:
                details.append("ok  %s:%d-%d (%d lines total)" % (clean, lo, hi, nlines))

    details.insert(0, "%d referenced path(s) checked at %s" % (len(refs), sha[:12]))
    if fails:
        return FAIL, fails + details
    return PASS, details


# --------------------------------------------------------------------- check 3

EXO_LAUNCH_FLAGS_RE = re.compile(
    r"-m\s+exo\b(.*?)(?:>>|2>&1|$)"
)


def extract_toplevel_def(src_lines, name):
    """Return the source of a module-level `def name(...)` block, or None."""
    start = None
    for i, ln in enumerate(src_lines):
        if re.match(r"^def\s+%s\s*\(" % re.escape(name), ln):
            start = i
            break
    if start is None:
        return None
    end = len(src_lines)
    for i in range(start + 1, len(src_lines)):
        if re.match(r"^(def|class)\s", src_lines[i]):
            end = i
            break
    return "\n".join(src_lines[start:end]).rstrip()


def load_exo_argparse(sha):
    """Exec the argparse `parse` classmethod at sha without importing exo."""
    src = show(sha, "src/exo/main.py")
    if src is None:
        return None, "src/exo/main.py missing at %s" % sha[:12]

    src_lines = src.splitlines()
    start = None
    for i, ln in enumerate(src_lines):
        if re.match(r"^\s+def parse\(cls\)", ln):
            start = i
            if i > 0 and "classmethod" in src_lines[i - 1]:
                start = i - 1
            break
    if start is None:
        return None, "no `def parse(cls)` found in src/exo/main.py at %s" % sha[:12]

    end = None
    for i in range(start, len(src_lines)):
        if "return cls(**vars(args))" in src_lines[i]:
            end = i
            break
    if end is None:
        return None, "could not delimit parse() body at %s" % sha[:12]

    method_src = "\n".join(src_lines[start:end + 1])
    method_src = method_src.replace("@classmethod", "@classmethod", 1)

    ns = {
        "__name__": "exo_main_extracted",
        "argparse": argparse,
        "os": os,
        "Self": object,          # 3.9 has no typing.Self; annotation only
        "__version__": "0.0.0+static",
    }
    # helpers the parser's defaults call into, taken from the same file/tree
    helpers = []
    from collections.abc import Mapping  # noqa: F401  (annotation in the source)
    ns["Mapping"] = Mapping
    for helper in ("default_namespace", "__init__"):
        chunk = extract_toplevel_def(src_lines, helper)
        if chunk:
            helpers.append(chunk)
    try:
        prelude = ""
        if helpers:
            prelude = "\n\n".join(helpers) + "\n\n"
        # stand-in holder for the parsed namespace: parse() returns
        # cls(**vars(args)) and §11 only cares that argparse ACCEPTS the argv,
        # not what ExoConfig.__init__ does with the values afterwards.
        holder = ("class _Extracted(object):\n"
                  "    def __init__(self, **kwargs):\n"
                  "        self.__dict__.update(kwargs)\n")
        # method_src is already indented four spaces for the class body
        compiled = compile(
            prelude + holder + method_src,
            "<exo/main.py:parse@%s>" % sha[:12],
            "exec",
        )
        exec(compiled, ns)
        return ns["_Extracted"], None
    except Exception as exc:  # pragma: no cover - reported, not swallowed
        return None, "could not exec parse() at %s: %s: %s" % (
            sha[:12], type(exc).__name__, exc)


def parse_argv(parser_cls, argv):
    """Feed argv to the real parser; return (ok, message, namespace)."""
    old = sys.argv
    err = io.StringIO()
    old_err = sys.stderr
    sys.argv = ["exo"] + list(argv)
    sys.stderr = err
    try:
        try:
            ns = parser_cls.parse()
            return True, "", ns
        except SystemExit as se:
            msg = err.getvalue().strip().splitlines()
            return False, "rejected argv %s (exit %s): %s" % (
                argv, se.code, msg[-1] if msg else "no stderr"), None
        except Exception as exc:
            return False, "parse() raised on argv %s: %s: %s" % (
                argv, type(exc).__name__, exc), None
    finally:
        sys.argv = old
        sys.stderr = old_err


def curl_known_flags():
    """Return the set of long/short options the local curl advertises."""
    curl = shutil.which("curl")
    if not curl:
        return None
    try:
        p = subprocess.run([curl, "--help", "all"], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    except Exception:
        return None
    if p.returncode != 0:
        return None
    text = p.stdout.decode("utf-8", "replace")
    flags = set()
    for m in re.finditer(r"(?<![\w-])(--[a-zA-Z0-9][a-zA-Z0-9-]*)", text):
        flags.add(m.group(1))
    for m in re.finditer(r"^\s{0,6}-([a-zA-Z])\s*[, ]", text, re.M):
        flags.add("-" + m.group(1))
    return flags


def check_cli(sha, lines, start, end):
    details = []
    fails = []
    notes = []

    parser_cls, why = load_exo_argparse(sha)
    if parser_cls is None:
        return FAIL, [why]

    # --- argv §11 itself issues to exo (usually none: the launch passes no flags)
    cmds = []
    for lang, block in fenced_blocks(lines, start, end):
        for argv in logical_commands(block):
            cmds.append(argv)

    exo_argvs = []
    for argv in cmds:
        toks = list(argv)
        if toks and toks[0].split("/")[-1] in ("exo", "exo.exe"):
            exo_argvs.append(toks[1:])
            continue
        for i in range(len(toks) - 1):
            if toks[i] == "-m" and toks[i + 1] == "exo":
                exo_argvs.append(toks[i + 2:])
                break

    # --- argv start_cluster.sh launches (what ./start_cluster.sh in §11.3 runs)
    sc = show(sha, "start_cluster.sh") or ""
    launch_argvs = []
    for ln in sc.splitlines():
        if "-m exo" in ln and "python" in ln:
            tail = ln.split("-m exo", 1)[1]
            tail = tail.split(">>")[0].split("2>&1")[0].strip()
            launch_argvs.append(tail.split() if tail else [])
    if not launch_argvs:
        notes.append("no `python -m exo` launch line found in start_cluster.sh at %s"
                     " -- §11.3's launcher may have changed shape" % sha[:12])
    else:
        dedup = []
        for a in launch_argvs:
            if a not in dedup:
                dedup.append(a)
        launch_argvs = dedup
        details.append("launch argv inherited by §11.3 from start_cluster.sh: %s"
                       % " ".join(launch_argvs[0] if launch_argvs[0] else ["<none>"]))
        if not launch_argvs[0]:
            details.append("that launch passes NO flags -- §11 relies entirely on "
                           "argparse defaults for ports/model placement")

    checked = 0
    for argv in exo_argvs:
        ok, msg, _ = parse_argv(parser_cls, argv)
        checked += 1
        if ok:
            details.append("accepted: exo %s" % " ".join(argv) if argv else "accepted: exo <no args>")
        else:
            fails.append(msg)
    for argv in launch_argvs:
        ok, msg, _ = parse_argv(parser_cls, argv)
        checked += 1
        if ok:
            details.append("accepted: python -m exo %s"
                           % (" ".join(argv) if argv else "<no args>"))
        else:
            fails.append("start_cluster.sh launch: " + msg)

    details.append("exo argparse surface exercised on %d argv(s); §11 itself names "
                   "%d exo-level flag(s) (%s)"
                   % (checked, len(exo_argvs),
                      "none -- its launch is flag-free"
                      if not exo_argvs else "listed above"))

    # --- documented defaults §11 states as facts must match the parser
    ok, msg, ns = parse_argv(parser_cls, [])
    if not ok:
        fails.append("bare invocation (no args) rejected: %s" % msg)
    else:
        claims = []
        api_m = re.search(r"port\s+(\d{4,5})", "\n".join(lines[start:end]))
        if api_m:
            claimed = int(api_m.group(1))
            actual = getattr(ns, "api_port", None)
            if actual == claimed:
                details.append("§11's 'port %d' == --api-port default %s"
                               % (claimed, actual))
            else:
                fails.append("§11 says the API listens on %d but --api-port "
                             "default is %s at %s" % (claimed, actual, sha[:12]))
            claims.append(claimed)
        zen_m = re.search(r'EXO_ZENOH_PORT="\$\{EXO_ZENOH_PORT:-(\d+)\}"', sc)
        dis_m = re.search(r'EXO_DISCOVERY_PORT="\$\{EXO_DISCOVERY_PORT:-(\d+)\}"', sc)
        for label, m, attr in (("--zenoh-port", zen_m, "zenoh_port"),
                               ("--discovery-port", dis_m, "discovery_port")):
            if m:
                want = int(m.group(1))
                got = getattr(ns, attr, None)
                if got == want:
                    details.append("start_cluster.sh %s default %d == argparse "
                                   "default %s" % (label, want, got))
                else:
                    fails.append("start_cluster.sh assumes %s %d but the parser's "
                                 "default is %s at %s"
                                 % (label, want, got, sha[:12]))

    # --- curl flags §11 uses, checked against this curl's own option table
    curl_flags = set()
    for argv in cmds:
        if argv and argv[0].split("/")[-1] == "curl":
            for tok in argv[1:]:
                if not tok.startswith("-"):
                    continue
                if tok.startswith("--"):
                    curl_flags.add(tok.split("=")[0])
                elif len(tok) > 2:
                    # bundled shorts (-sS == -s -S); expand each letter
                    for ch in tok[1:]:
                        curl_flags.add("-" + ch)
                else:
                    curl_flags.add(tok)
    known = curl_known_flags()
    if not curl_flags:
        notes.append("no curl invocation found in §11 -- nothing to check there")
    elif known is None:
        notes.append("curl not available; cannot verify curl flags %s"
                     % sorted(curl_flags))
    else:
        bad = sorted(f for f in curl_flags if f not in known)
        if bad:
            fails.append("curl rejects/does not know: %s" % bad)
        else:
            details.append("curl accepts all %d §11 flags: %s"
                           % (len(curl_flags), " ".join(sorted(curl_flags))))

    # --- git flags §11 uses
    git_flags = set()
    for argv in cmds:
        if argv and argv[0] == "git":
            for tok in argv[1:]:
                if tok.startswith("-"):
                    git_flags.add(tok)
    if git_flags:
        try:
            p = subprocess.run(["git", "submodule", "update", "-h"],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            usage = p.stdout.decode("utf-8", "replace")
        except Exception:
            usage = ""
        unverifiable = sorted(f for f in git_flags if f not in usage)
        if unverifiable:
            notes.append("not machine-checked (git usage text does not list them "
                         "here): %s" % unverifiable)
        else:
            details.append("git usage lists all §11 git flags: %s"
                           % " ".join(sorted(git_flags)))

    if fails:
        return FAIL, fails + details
    return PASS, details + notes


# --------------------------------------------------------------------- check 4


def check_prompts(repo, sha):
    details = []
    gen_rel, gen_where = locate_prompt_script(
        repo, sha, ["scripts/gen-prompts.py", "gen-prompts.py"])
    val_rel, val_where = locate_prompt_script(
        repo, sha, ["scripts/validate-prompts.py", "validate-prompts.py"])

    if not gen_rel or not val_rel:
        missing = []
        if not gen_rel:
            missing.append("scripts/gen-prompts.py")
        if not val_rel:
            missing.append("scripts/validate-prompts.py")
        return SKIP, [
            "SKIPPED: %s not committed yet." % " and ".join(missing),
            "§11.5 admits the prompt files and their generator were never in the "
            "repo; the siblings writing them are still in flight. This check "
            "activates automatically once both exist (working tree or pinned SHA).",
        ]

    py = pick_python()
    details.append("generator: %s (%s)" % (gen_rel, gen_where))
    details.append("validator: %s (%s)" % (val_rel, val_where))
    details.append("interpreter: %s" % py)

    def run(argv, cwd):
        try:
            p = subprocess.run(argv, cwd=cwd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=300)
            return p.returncode, p.stdout.decode("utf-8", "replace")
        except subprocess.TimeoutExpired:
            return 124, "timed out after 300s"

    tmp = tempfile.mkdtemp(prefix="check-s11-prompts-")
    try:
        gen_path = os.path.join(repo, gen_rel)
        rc, out = run([py, gen_path, "--help"], repo)
        if rc != 0:
            return FAIL, ["%s --help exits %d: %s"
                          % (gen_rel, rc, out.strip().splitlines()[-1:])]

        outdir = os.path.join(tmp, "out")
        os.makedirs(outdir)
        gen_argv = [py, gen_path]
        if re.search(r"--out\b", out):
            gen_argv += ["--out", outdir]
        elif re.search(r"--output-dir\b", out):
            gen_argv += ["--output-dir", outdir]
        elif re.search(r"--dir\b", out):
            gen_argv += ["--dir", outdir]
        details.append("generator argv: %s" % " ".join(gen_argv[2:] or ["<none>"]))
        rc, out = run(gen_argv, outdir)
        if rc != 0:
            return FAIL, ["%s exits %d: %s"
                          % (gen_rel, rc, out.strip().splitlines()[-3:])]

        produced = []
        for root, _dirs, files in os.walk(tmp):
            for f in files:
                if f.endswith(".json"):
                    produced.append(os.path.join(root, f))
        if not produced:
            return FAIL, ["%s ran but produced no .json under %s"
                          % (gen_rel, tmp), out.strip()[-400:]]
        details.append("generated %d prompt file(s): %s"
                       % (len(produced), ", ".join(sorted(os.path.basename(p)
                                                         for p in produced))))

        val_path = os.path.join(repo, val_rel)
        rc, vout = run([py, val_path] + sorted(produced), repo)
        if rc != 0:
            rc2, vout2 = run([py, val_path, tmp], repo)
            if rc2 != 0:
                return FAIL, ["%s rejects generated output (exit %d): %s"
                              % (val_rel, rc, vout.strip()[-600:])]
            vout = vout2
        details.append("validator accepted the generated files")
        tail = [ln for ln in vout.strip().splitlines() if ln.strip()][-3:]
        for ln in tail:
            details.append("  validate: %s" % ln)
        return PASS, details
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------- check 5


def check_constants(sha, lines, start, end):
    details = []
    fails = []
    body = "\n".join(lines[start:end])

    sc = show(sha, "start_cluster.sh")
    if sc is None:
        return FAIL, ["start_cluster.sh missing at %s" % sha[:12]]
    sc_lines = sc.splitlines()

    # §11 says: "`NODES` :1113, peer constants :1025-1028"
    nodes_m = re.search(r"`NODES`\s*:(\d+)", body)
    peer_m = re.search(r"peer constants\s*:(\d+)-(\d+)", body)
    if not nodes_m:
        fails.append("§11 no longer states a line for NODES -- the note may have "
                     "been reworded; update this check with it")
    if not peer_m:
        fails.append("§11 no longer states a line range for the peer constants")

    if nodes_m:
        want = int(nodes_m.group(1))
        got = sc_lines[want - 1] if want - 1 < len(sc_lines) else ""
        if not re.match(r"^\s*NODES=\(", got):
            # tolerate a small drift: find where NODES really is
            real = [i + 1 for i, ln in enumerate(sc_lines)
                    if re.match(r"^\s*NODES=\(", ln)]
            fails.append("§11 puts NODES at start_cluster.sh:%d but it is at %s"
                         % (want, real or "nowhere"))
        else:
            entries = re.findall(r'"([^"]*)"', got)
            details.append("start_cluster.sh:%d NODES=(%s) -- %d node(s), §11 says two"
                           % (want, ", ".join(entries), len(entries)))
            if len(entries) != 2:
                fails.append("§11's instructions assume two nodes; NODES has %d"
                             % len(entries))

    if peer_m:
        lo, hi = int(peer_m.group(1)), int(peer_m.group(2))
        window = sc_lines[lo - 1:hi]
        if len(window) != hi - lo + 1:
            fails.append("start_cluster.sh has no lines %d-%d at %s"
                         % (lo, hi, sha[:12]))
        else:
            peer_ids = re.findall(r"([A-Z0-9_]*PEER_ID)=\"([^\"]+)\"", "\n".join(window))
            ips = re.findall(r"([A-Z0-9_]*_IP)=\"([^\"]+)\"", "\n".join(window))
            details.append("start_cluster.sh:%d-%d holds %d PEER_ID and %d IP "
                           "constant(s)" % (lo, hi, len(peer_ids), len(ips)))
            if len(peer_ids) != 2 or len(ips) != 2:
                fails.append("§11 says the note covers both nodes' IPs and peer IDs "
                             "but lines %d-%d hold %d peer ID(s) and %d IP(s)"
                             % (lo, hi, len(peer_ids), len(ips)))
            for name, val in peer_ids:
                if not re.match(r"^12D3Koo[A-Za-z0-9]{44,}$", val):
                    fails.append("%s=%r at start_cluster.sh:%d-%d is not a libp2p "
                                 "12D3Koo... peer ID" % (name, val, lo, hi))
                else:
                    details.append("  %s = %s (%d chars, 12D3Koo shape)"
                                   % (name, val, len(val)))
            for name, val in ips:
                if not re.match(r"^\d{1,3}(\.\d{1,3}){3}$", val):
                    fails.append("%s=%r is not a dotted-quad IPv4" % (name, val))
                else:
                    details.append("  %s = %s" % (name, val))

    # §11.1: "sysctl iogpu.wired_limit_mb = 120000 ... reapplied by start_cluster.sh"
    wl = re.search(r"iogpu\.wired_limit_mb\s*=\s*(\d+)", body)
    if wl:
        want = wl.group(1)
        m = re.search(r"DSV4_WIRED_LIMIT_MB:-(\d+)", sc)
        if not m:
            fails.append("§11 claims start_cluster.sh reapplies "
                         "iogpu.wired_limit_mb=%s every launch, but no "
                         "DSV4_WIRED_LIMIT_MB default exists at %s"
                         % (want, sha[:12]))
        elif m.group(1) != want:
            fails.append("§11 says wired_limit_mb=%s but start_cluster.sh defaults "
                         "to %s" % (want, m.group(1)))
        else:
            details.append("wired-limit claim: %s == DSV4_WIRED_LIMIT_MB default "
                           "(%s), reapplied at launch" % (want, m.group(1)))
    else:
        fails.append("§11 no longer states a wired_limit_mb value to cross-check")

    # §11.2's SHA table vs the actual gitlinks and lockfile pins
    table = {}
    for ln in lines[start:end]:
        m = re.match(r"^\|\s*(exo|mlx|mlx-lm)\s*\|", ln)
        if not m:
            continue
        cells = [c.strip().strip("`") for c in ln.strip().strip("|").split("|")]
        for c in reversed(cells):
            if re.match(r"^[0-9a-f]{40}$", c):
                table[m.group(1)] = c
                break
    for sub in ("mlx", "mlx-lm"):
        pinned = table.get(sub)
        if not pinned:
            continue
        rc, out, _ = git("ls-tree", sha, sub)
        m = re.search(r"commit\s+([0-9a-f]{40})", out)
        if not m:
            fails.append("no gitlink for submodule %s at %s" % (sub, sha[:12]))
        elif m.group(1) != pinned:
            fails.append("%s gitlink is %s at %s but §11's table says %s"
                         % (sub, m.group(1), sha[:12], pinned))
        else:
            details.append("%s gitlink == table SHA %s" % (sub, pinned[:8]))

    lock = show(sha, "uv.lock") or ""
    lock_lines = lock.splitlines()
    for sub, lineref in (("mlx", 1384), ("mlx-lm", 1501)):
        pinned = table.get(sub)
        if not pinned:
            continue
        if lineref - 1 >= len(lock_lines):
            fails.append("uv.lock has no line %d at %s" % (lineref, sha[:12]))
            continue
        line = lock_lines[lineref - 1]
        if pinned[:8] not in line:
            # tolerate drift, but say where the pin really is
            real = [i + 1 for i, ln in enumerate(lock_lines) if pinned in ln]
            fails.append("uv.lock:%d does not carry the %s pin %s (found at %s)"
                         % (lineref, sub, pinned[:8], real or "nowhere"))
        else:
            details.append("uv.lock:%d carries %s pin %s" % (lineref, sub, pinned[:8]))

    rb = show(sha, "src/exo/worker/engines/mlx/dsv41/reasoning_budget.py")
    if rb is None:
        fails.append("reasoning_budget.py (cited at §11.7) missing at %s" % sha[:12])
    else:
        rbl = rb.splitlines()
        if len(rbl) >= 92 and "request_value" in rbl[90] and "return int(" in rbl[91]:
            details.append("reasoning_budget.py:91-92 is still the uncapped "
                           "pass-through §11.7 cites")
        else:
            fails.append("reasoning_budget.py:91-92 no longer reads as the "
                         "uncapped pass-through (§11.7's citation drifted)")

    if fails:
        return FAIL, fails + details
    return PASS, details


# ------------------------------------------------------------- status line write


def status_line(sha, today, skipped=None):
    core = (
        "_Static-checked at %s on %s by scripts/check-s11.sh: the paths, CLI flags "
        "and constants §11 names all resolve at that SHA. Static only -- nothing was "
        "launched, so this does not show that the SHA boots on this hardware, that "
        "the hardcoded node/peer constants fit your cluster, or that any §9 figure "
        "reproduces" % (sha[:12], today)
    )
    if skipped:
        core += " (skipped: %s)" % skipped
    return core + "._"


def find_status(lines, start, end):
    for i in range(start, end):
        if STATUS_RE.match(lines[i].strip()):
            return i
    return None


def write_status(path, lines, start, end, sha, today, skipped=None):
    """Replace an existing status line, else append one at the end of §11."""
    line = status_line(sha, today, skipped)
    idx = find_status(lines, start, end)
    if idx is not None:
        lines[idx] = line
        action = "replaced"
    else:
        insert_at = None
        for i in range(end - 1, start, -1):
            s = lines[i].strip()
            if s and s != "---":
                insert_at = i + 1
                break
        if insert_at is None:
            insert_at = end
        if insert_at > 0 and lines[insert_at - 1].strip() != "":
            lines.insert(insert_at, "")
            insert_at += 1
        lines.insert(insert_at, line)
        if insert_at + 1 < len(lines) and lines[insert_at + 1].strip() != "":
            lines.insert(insert_at + 1, "")
        action = "appended"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines).rstrip("\n") + "\n")
    return action, line


# ------------------------------------------------------------------------- main


def main(argv):
    ap = argparse.ArgumentParser(
        prog="check-s11.sh",
        description="Static lint of WRITEUP.md §11 against the SHA §11 pins. "
                    "Proves the instructions are well-formed and current; proves "
                    "nothing about booting or reproducing.")
    ap.add_argument("--sha", default=None,
                    help="commit to check (default: the SHA in §11.2's table)")
    ap.add_argument("--writeup", default=None, help="path to WRITEUP.md")
    ap.add_argument("--check-only", action="store_true",
                    help="never modify the writeup; fail if the §11 status line is "
                         "missing or names a different SHA (for CI)")
    ap.add_argument("--quiet", action="store_true", help="only the verdict lines")
    args = ap.parse_args(argv)

    global REPO_ROOT
    here = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = find_repo_root(here) or find_repo_root(os.getcwd())
    if not REPO_ROOT:
        print("FAIL  not inside a git repository")
        return 2

    path = args.writeup or os.path.join(REPO_ROOT, WRITEUP_REL)
    if not os.path.exists(path):
        print("FAIL  writeup not found: %s" % path)
        return 2
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    lines = text.splitlines()
    bounds = slice_section_11(lines)
    if not bounds:
        print("FAIL  no '## 11.' section in %s" % path)
        return 2
    start, end = bounds

    today = datetime.date.today().isoformat()
    print("check-s11: static lint of %s §11" % os.path.basename(path))
    print("repo: %s" % REPO_ROOT)
    print("section 11: lines %d-%d of %s" % (start + 1, end, os.path.basename(path)))
    print("date: %s" % today)
    print("")

    # 1
    try:
        st, det = check_sha(lines, start, end, args.sha)
    except Exception:
        st, det = FAIL, ["check crashed: " + traceback.format_exc().splitlines()[-1]]
    record(1, "pinned SHA resolves", st, det)

    sha = args.sha
    if sha is None:
        for i in range(start, end):
            if re.match(r"^\|\s*exo\s*\|", lines[i]):
                for c in reversed([x.strip().strip("`") for x in
                                   lines[i].strip().strip("|").split("|")]):
                    if re.match(r"^[0-9a-f]{40}$", c):
                        sha = c
                        break
                break

    if sha and results[0][2] == PASS:
        for num, name, fn in (
            (2, "referenced paths exist at the SHA", lambda: check_paths(sha, lines, start, end)),
            (3, "CLI flags accepted at the SHA", lambda: check_cli(sha, lines, start, end)),
            (4, "prompt generator -> validator", lambda: check_prompts(REPO_ROOT, sha)),
            (5, "hardcoded constants match the SHA", lambda: check_constants(sha, lines, start, end)),
        ):
            try:
                st, det = fn()
            except Exception:
                st, det = FAIL, ["check crashed: "
                                 + traceback.format_exc().splitlines()[-1]]
            record(num, name, st, det)
    else:
        why = ("no SHA to check against" if not sha
               else "check 1 failed -- the pinned SHA does not resolve, so there is "
                    "no tree to lint against")
        for num, name in ((2, "referenced paths exist at the SHA"),
                          (3, "CLI flags accepted at the SHA"),
                          (4, "prompt generator -> validator"),
                          (5, "hardcoded constants match the SHA")):
            record(num, name, SKIP, [why])

    for num, name, st, det in results:
        print("CHECK %d  %-4s  %s" % (num, st, name))
        if not args.quiet:
            for ln in det:
                print("         %s" % ln)
    print("")

    failed = [r for r in results if r[2] == FAIL]
    skipped = [r for r in results if r[2] == SKIP]
    require_prompts = os.environ.get("CHECK_S11_REQUIRE_PROMPTS", "") not in ("", "0")

    if require_prompts and any(r[0] == 4 and r[2] == SKIP for r in results):
        print("FAIL  CHECK_S11_REQUIRE_PROMPTS=1 but check 4 is SKIP")
        failed.append((4, "prompts", FAIL))

    writable = not failed
    skip_note = None
    if skipped:
        d4 = [r for r in skipped if r[0] == 4]
        if d4 and d4[0][3]:
            first = d4[0][3][0]
            first = first.replace("SKIPPED: ", "").rstrip(".")
            skip_note = "check 4: %s" % first
        else:
            skip_note = "checks %s" % ", ".join(str(r[0]) for r in skipped)

    # status line handling
    idx = find_status(lines, start, end)
    if args.check_only:
        if not writable:
            print("NOTE  status line not required: %d check(s) failed, %d skipped"
                  % (len(failed), len(skipped)))
        elif idx is None:
            print("FAIL  §11 has no status line; run scripts/check-s11.sh to write it")
            failed.append((0, "status line", FAIL))
        else:
            have = lines[idx].strip()
            if sha and sha[:12] not in have:
                print("FAIL  §11's status line names a different SHA: %s" % have)
                failed.append((0, "status line", FAIL))
            else:
                print("ok    §11 status line is current: %s" % have)
    else:
        if writable:
            action, line = write_status(path, lines, start, end, sha, today, skip_note)
            print("%s §11 status line (%s):" % (action.upper(), action))
            print("  %s" % line)
        else:
            print("status line NOT written: %d check(s) failed -- §11 must pass "
                  "the static checks before it claims to be static-checked"
                  % len(failed))

    print("")
    if failed:
        print("SUMMARY FAIL  %d of %d checks failed" % (len(failed), len(results)))
        return 1
    if skipped:
        print("SUMMARY PASS  (%d skipped) -- static only: nothing was launched, so "
              "no claim about booting, hardware, or §9 figures" % len(skipped))
        return 0
    print("SUMMARY PASS  all 5 checks -- static only: paths, flags and constants "
          "resolve at %s. Nothing was launched; boot, hardware fit and §9 figures "
          "remain unverified." % sha[:12])
    return 0


REPO_ROOT = os.getcwd()

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
