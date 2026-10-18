#!/usr/bin/env python3
"""Deterministically regenerate the §11 prompt bodies lost from the DSv4.1 writeup.

WHAT THIS IS
  The 2026-10-16/17 capture session built its request bodies in a scratch
  directory (`~/.hermes/cache/scratch/writeup/prompts/`) that was never
  committed, so §11 of docs/dsv41-cluster-writeup-2026-10-17/ cannot be re-run
  from the repo alone. This script rebuilds *equivalent-class* inputs: same
  message structure, same request fields, same corpus family and concatenation
  rule as the writeup records, filled to the exact prompt token counts the
  surviving `[DSV41] obs` log lines report.

WHAT THIS IS NOT
  It is not the original. The originals are RAW-LOST; nothing here makes their
  byte sizes or their exact tokenizations recoverable. See
  docs/dsv41-cluster-writeup-2026-10-17/preserved/RECONSTRUCTION.md.

DETERMINISM
  - corpus: `git ls-tree` of the pinned commit b88d65dd..., docs/**.md plus the
    repo's top-level *.md, sorted, joined with a blank line. No working-tree
    reads, no network, no wall clock.
  - prompt framing: exo's own vendor encoder for DSv4
    (`src/exo/worker/engines/mlx/vendor/deepseek_v4_encoding.py`) read at that
    same commit, so the template tokens are exo's, not a re-implementation.
  - tokenizer: the checkpoint-family tokenizer.json exo loads for this model
    (~/.exo/models/deepseek-ai--DeepSeek-V4-Flash/tokenizer.json), sha256
    pinned below. Its first 8 hex digits (8f9f37ca) are the id the writeup
    quotes for the local tokenizer.
  - `--seed` is recorded and used only to break ties between equally valid
    prefix lengths. Same seed + same commit + same tokenizer => same bytes.

USAGE
  .venv/bin/python scripts/gen-prompts.py            # writes ./prompts
  .venv/bin/python scripts/gen-prompts.py --check    # verify, write nothing

  Requires only the stdlib plus the HF `tokenizers` package (the repo's .venv
  has it; the script re-execs into .venv/bin/python if the current interpreter
  lacks it).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

# --------------------------------------------------------------------------
# Pinned inputs
# --------------------------------------------------------------------------

#: Commit the writeup pins for the capture and for reproduction (§11.3).
PINNED_SHA = "b88d65dd330478e8e1b8a565848af3c863be7a06"

MODEL = "dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"

#: Vendor encoder path inside the repo (stdlib-only module; safe to exec).
ENCODER_PATH = "src/exo/worker/engines/mlx/vendor/deepseek_v4_encoding.py"
#: sha1 of that blob at PINNED_SHA (git hash-object), used as a drift check.
ENCODER_BLOB_SHA1 = "7b785a2d93df39931572f66ab33a4b4bc2b009a0"

DEFAULT_TOKENIZER = Path(
    "~/.exo/models/deepseek-ai--DeepSeek-V4-Flash/tokenizer.json"
).expanduser()
#: Full sha256; the writeup quotes the 8-hex prefix "8f9f37ca".
TOKENIZER_SHA256 = (
    "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf"
)

#: The writeup's own directory was added by the same merge commit that PINNED_SHA
#: points at, i.e. AFTER the capture. Feeding the writeup (which quotes the
#: model's answers) back into the prompt would be self-referential, so it is
#: excluded from the corpus. This is the only exclusion.
CORPUS_EXCLUDE_PREFIX = "docs/dsv41-cluster-writeup-2026-10-17/"

SEED = 20261016

#: The question appended after the corpus. The original wording is NOT preserved
#: anywhere in the repo -- only its shape is (every surviving completion starts
#: "BOTTLENECK: ..." and names the document that quantifies the bottleneck), so
#: this is a reconstruction and is labelled as such.
BOTTLENECK_QUESTION = (
    "Question: across all of the reference material above, what is the single "
    "most frequently discussed performance bottleneck, and which document "
    "quantifies it most directly? Answer in one paragraph. Begin the paragraph "
    "with the word BOTTLENECK: followed by a space, and cite the document paths "
    "you rely on."
)

#: (filename, prompt-token target from the surviving obs lines, max_tokens,
#:  stream, enable_thinking, kind)
BIG_SPECS: list[dict[str, Any]] = [
    # §9.1 / Deliverable 1: server counted 160,004 prompt tokens, stream=false.
    dict(name="prompts_160k.json", target=160_004, max_tokens=200,
         stream=False, thinking=False, corpus=True),
    # Deliverable 4: server counted 1,000,003 prompt tokens.
    dict(name="prompts_1m.json", target=1_000_003, max_tokens=100,
         stream=True, thinking=False, corpus=True),
    # Deliverable 2 depth rows: 1003 / 10004 / 50004 / 100003 / 300003.
    dict(name="depth_d1k.json", target=1_003, max_tokens=200,
         stream=True, thinking=False, corpus=True),
    dict(name="depth_d10k.json", target=10_004, max_tokens=200,
         stream=True, thinking=False, corpus=True),
    dict(name="depth_d50k.json", target=50_004, max_tokens=200,
         stream=True, thinking=False, corpus=True),
    dict(name="depth_d100k.json", target=100_003, max_tokens=200,
         stream=True, thinking=False, corpus=True),
    dict(name="depth_d300k.json", target=300_003, max_tokens=200,
         stream=True, thinking=False, corpus=True),
]

#: Small bodies. guard_banana's body is quoted verbatim in evidence-live.md, so
#: it is reproduced exactly rather than reconstructed. guard_loop and the two
#: limiter bodies survive only as prose + their logged prompt-token counts
#: (48 / 52 / 52), so their wording is a reconstruction tuned to those counts.
SMALL_SPECS: list[dict[str, Any]] = [
    dict(
        name="guard_banana.json",
        target=26,  # "[DSV41] obs first-token ... 26 prompt tokens"
        max_tokens=400,
        stream=True,
        thinking=True,
        content=(
            "Output the word banana exactly 40 times, separated by single "
            "spaces, and nothing else. Do not explain."
        ),
        verbatim=True,
    ),
    dict(
        name="guard_loop.json",
        target=48,  # "... 48 prompt tokens" (negative control, 3b)
        max_tokens=1200,
        stream=True,
        thinking=True,
        content=(
            "Write a 12-section essay about the history of distributed "
            "computing. Begin each section with a marker of the form "
            "[SECTION n] on its own line, where n is the section number. "
            "Do not use other headings."
        ),
        verbatim=False,
    ),
    dict(
        name="limiter_default.json",
        target=52,  # "... 52 prompt tokens" (no reasoning_budget_tokens field)
        max_tokens=800,
        stream=True,
        thinking=True,
        content=(
            "A bat and a ball cost $1.10 in total. The bat costs $1.00 more "
            "than the ball. How much does the ball cost? Think through it step "
            "by step, then give the final answer in one sentence."
        ),
        verbatim=False,
    ),
    dict(
        name="limiter_96.json",
        target=52,  # same prompt, same 52 prompt tokens; only the budget differs
        max_tokens=800,
        stream=True,
        thinking=True,
        reasoning_budget_tokens=96,
        content=(
            "A bat and a ball cost $1.10 in total. The bat costs $1.00 more "
            "than the ball. How much does the ball cost? Think through it step "
            "by step, then give the final answer in one sentence."
        ),
        verbatim=False,
    ),
]


# --------------------------------------------------------------------------
# Environment plumbing
# --------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_tokenizers() -> None:
    """Re-exec into the repo .venv when the ambient interpreter lacks tokenizers."""
    try:
        import tokenizers  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    for cand in (_repo_root() / ".venv/bin/python", Path(sys.executable)):
        if cand.exists() and Path(cand) != Path(sys.executable):
            os.execv(str(cand), [str(cand), *sys.argv])
    sys.exit(
        "gen-prompts: the HF `tokenizers` package is required (stdlib + tokenizer "
        "only). Install it or run with <repo>/.venv/bin/python."
    )


def _git(*args: str, binary: bool = False) -> bytes | str:
    out = subprocess.run(
        ["git", *args], cwd=_repo_root(), check=True, capture_output=True
    ).stdout
    return out if binary else out.decode("utf-8")


def _load_encoder() -> ModuleType:
    """Import exo's DSv4 encoder from the pinned commit, not from the worktree."""
    raw = _git("show", f"{PINNED_SHA}:{ENCODER_PATH}", binary=True)
    assert isinstance(raw, bytes)
    header = b"blob %d\0" % len(raw)
    got = hashlib.sha1(header + raw).hexdigest()
    if got != ENCODER_BLOB_SHA1:
        raise SystemExit(
            f"gen-prompts: encoder blob at {PINNED_SHA} hashes to {got}, "
            f"expected {ENCODER_BLOB_SHA1}"
        )
    mod = ModuleType("dsv41_vendor_encoder")
    mod.__file__ = f"{PINNED_SHA}:{ENCODER_PATH}"
    exec(compile(raw.decode("utf-8"), mod.__file__, "exec"), mod.__dict__)
    return mod


def _load_tokenizer():
    from tokenizers import Tokenizer

    path = Path(os.environ.get("EXO_DSV41_TOKENIZER", DEFAULT_TOKENIZER)).expanduser()
    if not path.exists():
        raise SystemExit(f"gen-prompts: tokenizer not found at {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != TOKENIZER_SHA256:
        raise SystemExit(
            f"gen-prompts: tokenizer {path} sha256={digest} does not match the "
            f"pinned {TOKENIZER_SHA256}; refusing to build a non-reproducible body"
        )
    return Tokenizer.from_file(str(path))


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------


def corpus_member_paths() -> list[str]:
    """docs/**.md + top-level *.md at PINNED_SHA, minus the writeup directory."""
    listing = _git("ls-tree", "-r", "--name-only", PINNED_SHA)
    assert isinstance(listing, str)
    keep = []
    for name in listing.splitlines():
        if not name.endswith(".md"):
            continue
        if name.startswith(CORPUS_EXCLUDE_PREFIX):
            continue
        if name.startswith("docs/") or "/" not in name:
            keep.append(name)
    keep.sort()
    return keep


def build_corpus(placeholder: str) -> tuple[str, list[str], int]:
    """Concatenate the pinned blobs in sorted order, blank-line separated.

    exo's DSv4 encoder refuses message text containing the image placeholder
    token, and one doc at the pinned commit (`docs/DSV4_VISION_PORT_PLAN_PHASE34.md`)
    quotes it while discussing the vision port. The capture session's corpus
    evidently did not contain that literal, so it is replaced with a visible
    ASCII marker; the substitution is counted and recorded in the manifest.
    """
    names = corpus_member_paths()
    if not names:
        raise SystemExit("gen-prompts: corpus is empty at the pinned commit")
    request = "".join(f"{PINNED_SHA}:{n}\n" for n in names).encode()
    raw = subprocess.run(
        ["git", "cat-file", "--batch"], cwd=_repo_root(), input=request,
        check=True, capture_output=True,
    ).stdout
    blobs: list[str] = []
    i = 0
    while i < len(raw):
        nl = raw.index(b"\n", i)
        header = raw[i:nl].split()
        if len(header) < 3:  # "<sha> missing"
            raise SystemExit(f"gen-prompts: {raw[i:nl]!r} missing at {PINNED_SHA}")
        size = int(header[2])
        blobs.append(raw[nl + 1 : nl + 1 + size].decode("utf-8"))
        i = nl + 1 + size + 1
    if len(blobs) != len(names):
        raise SystemExit(
            f"gen-prompts: read {len(blobs)} blobs for {len(names)} names"
        )
    corpus = "\n\n".join(blobs)
    substitutions = corpus.count(placeholder)
    if substitutions:
        corpus = corpus.replace(placeholder, "[image-placeholder-token]")
    return corpus, names, substitutions


# --------------------------------------------------------------------------
# Body construction
# --------------------------------------------------------------------------


class Builder:
    def __init__(self, tok, encoder: ModuleType) -> None:
        self.tok = tok
        self.enc = encoder

    def ntok(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False).ids)

    def render(self, content: str, thinking: bool) -> str:
        """Exactly the string exo tokenizes for a one-message chat request."""
        return self.enc.encode_messages(
            [{"role": "user", "content": content}],
            thinking_mode="thinking" if thinking else "chat",
        )

    def prompt_tokens(self, content: str, thinking: bool) -> int:
        return self.ntok(self.render(content, thinking))

    def fill_corpus_body(
        self, ids: list[int], target: int, thinking: bool, seed: int
    ) -> tuple[str, dict[str, int]]:
        """Pick the corpus prefix whose rendered prompt hits `target` exactly.

        The mapping prefix-length -> prompt-token-count is monotone except at
        BPE merge boundaries: one extra corpus token can move the rendered
        prompt by two, so some targets are unreachable by slicing alone. A short
        deterministic newline run between the corpus and the question restores
        unit granularity (it costs 0-1 tokens per newline). Preference order is
        "no padding, longest prefix" so the body stays as close to a pure
        corpus prefix as the tokenizer allows.
        """
        suffix = "\n\n" + BOTTLENECK_QUESTION
        suffix_tokens = self.ntok(suffix)
        framing = self.prompt_tokens(suffix, thinking) - suffix_tokens
        n = max(0, target - framing - suffix_tokens)

        best: tuple[int, int, str, int] | None = None
        for offset in (0, -1, 1, -2, 2):
            for pad in range(0, 9):
                k = n + offset
                if not 0 <= k <= len(ids):
                    continue
                content = self.tok.decode(ids[:k]) + "\n" * pad + suffix
                if self.prompt_tokens(content, thinking) == target:
                    best = (k, pad, content, target)
                    break
            if best is not None:
                break
        if best is None:
            raise SystemExit(
                f"gen-prompts: cannot hit {target} prompt tokens exactly "
                f"(nearest prefix {n})"
            )

        n_used, pad, content, got = best
        return content, {
            "target_prompt_tokens": target,
            "prompt_tokens": got,
            "content_tokens": self.ntok(content),
            "corpus_prefix_tokens": n_used,
            "tail_newlines": pad,
            "framing_tokens": framing,
            "suffix_tokens": suffix_tokens,
        }

    def build(self, spec: dict[str, Any], ids: list[int] | None, seed: int):
        stats: dict[str, Any]
        if spec.get("corpus"):
            content, stats = self.fill_corpus_body(
                ids or [], spec["target"], spec["thinking"], seed
            )
        else:
            content = spec["content"]
            stats = {
                "target_prompt_tokens": spec["target"],
                "prompt_tokens": self.prompt_tokens(content, spec["thinking"]),
                "content_tokens": self.ntok(content),
                "corpus_prefix_tokens": 0,
                "framing_tokens": 0,
                "suffix_tokens": 0,
            }
        body: dict[str, Any] = {
            "model": MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": spec["max_tokens"],
            "temperature": 0.0,
            "stream": spec["stream"],
            "enable_thinking": spec["thinking"],
        }
        if "reasoning_budget_tokens" in spec:
            body["reasoning_budget_tokens"] = spec["reasoning_budget_tokens"]
        stats["name"] = spec["name"]
        stats["bytes"] = None  # filled after serialisation
        stats["verbatim"] = bool(spec.get("verbatim"))
        return body, stats


def serialise(body: dict[str, Any]) -> bytes:
    # indent=1 matches the quoted guard_banana body in evidence-live.md.
    return (json.dumps(body, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Regenerate the lost DSv4.1 writeup prompt bodies (deterministic)."
    )
    ap.add_argument("--out", default=str(_repo_root() / "prompts"),
                    help="output directory (default: <repo>/prompts)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--only", action="append", default=None,
                    help="build only these filenames (repeatable)")
    ap.add_argument("--check", action="store_true",
                    help="build in memory, compare with what is on disk, write nothing")
    args = ap.parse_args(argv)

    tok = _load_tokenizer()
    encoder = _load_encoder()
    builder = Builder(tok, encoder)

    out = Path(args.out)
    corpus_ids: list[int] | None = None
    corpus_chars = 0
    corpus_files: list[str] = []
    corpus_substitutions = 0
    if any(s.get("corpus") for s in BIG_SPECS):
        corpus, corpus_files, corpus_substitutions = build_corpus(
            getattr(encoder, "IMAGE_PLACEHOLDER", "\0none\0")
        )
        corpus_chars = len(corpus)
        corpus_ids = tok.encode(corpus, add_special_tokens=False).ids

    specs = [*BIG_SPECS, *SMALL_SPECS]
    if args.only:
        wanted = set(args.only)
        specs = [s for s in specs if s["name"] in wanted]
        if not specs:
            raise SystemExit(f"gen-prompts: --only matched nothing in {wanted}")

    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "generator": "scripts/gen-prompts.py",
        "pinned_sha": PINNED_SHA,
        "model": MODEL,
        "seed": args.seed,
        "tokenizer": {"path": str(DEFAULT_TOKENIZER), "sha256": TOKENIZER_SHA256},
        "corpus": {
            "rule": "docs/**.md + top-level *.md at pinned_sha, sorted, "
                    "joined with a blank line",
            "excluded_prefix": CORPUS_EXCLUDE_PREFIX,
            "files": len(corpus_files),
            "chars": corpus_chars,
            "tokens": len(corpus_ids) if corpus_ids is not None else 0,
            "image_placeholder_substitutions": corpus_substitutions,
        },
        "bodies": {},
    }

    rc = 0
    for spec in specs:
        body, stats = builder.build(spec, corpus_ids, args.seed)
        data = serialise(body)
        stats["bytes"] = len(data)
        path = out / spec["name"]
        on_disk = path.read_bytes() if path.exists() else None
        stats["matches_disk"] = on_disk == data
        if args.check:
            if on_disk is None:
                print(f"MISSING {spec['name']}")
                rc = 1
            elif on_disk != data:
                print(f"DRIFT   {spec['name']} ({len(on_disk)} B on disk, "
                      f"{len(data)} B regenerated)")
                rc = 1
            else:
                print(f"OK      {spec['name']}  {stats['prompt_tokens']} tok  "
                      f"{stats['bytes']} B")
        else:
            path.write_bytes(data)
            # Verify the bytes that actually landed, not the ones we intended.
            stats["matches_disk"] = path.read_bytes() == data
            stats["sha256"] = hashlib.sha256(data).hexdigest()
            flag = "" if stats["prompt_tokens"] == spec["target"] else "  <-- MISS"
            print(
                f"wrote {spec['name']:<22} prompt_tokens={stats['prompt_tokens']:>8}"
                f" (target {spec['target']})  content={stats['content_tokens']:>8}"
                f"  bytes={stats['bytes']:>9}{flag}"
            )
        manifest["bodies"][spec["name"]] = stats

    if not args.check:
        (out / "GENERATED-MANIFEST.json").write_text(
            json.dumps(manifest, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {out/'GENERATED-MANIFEST.json'}")
    return rc


if __name__ == "__main__":
    _ensure_tokenizers()
    sys.exit(main(sys.argv[1:]))
