"""Core seam-rule mechanics: safe-seam discovery, tokenization wrappers, and
the round-trip identity assertion.

SEAM RULE (binding, from pre-registration):
  - Safe seams ONLY immediately after a special/added token.
  - BOS emitted exactly once.
  - The suffix MUST be encoded with add_special_tokens=False.
  - A normalizer can invalidate seam safety entirely -- must be checked
    separately (see normalizer_check.py), never assumed absent.
  - Core shadow-assertion:
        cached_prefix_tokens + tok(suffix) == tok(full)   (exact list equality)

Production reality (verified against cache.py:2300-2308, encode_prompt()):
  the ENTIRE prompt (including the literal BOS/special-token text emitted by
  the DSv4 vendored encoder) is passed through
  `tokenizer.encode(prompt, add_special_tokens=False)`. The tokenizer never
  auto-adds BOS; the chat/DSv4 template embeds the BOS token's literal string
  once, and add_special_tokens=False means the tokenizer's own
  automatic-special-token machinery is disabled -- the tokenizer instead
  recognizes the literal special-token substrings via its added-vocab and
  converts them to their token ids as part of normal BPE tokenization. Any
  candidate prefix-cache split must reproduce this exact call shape or it is
  not testing what production does.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokenizer_paths import find_tokenizer_dir


@dataclass(frozen=True)
class LoadedTokenizer:
    tokenizer: Any
    source_dir: Path


def load_tokenizer() -> LoadedTokenizer | None:
    """Load the local fast tokenizer with local_files_only=True. Returns None
    (never raises, never downloads) if no local tokenizer files are found."""
    tok_dir = find_tokenizer_dir()
    if tok_dir is None:
        return None
    from transformers import AutoTokenizer  # local import: optional dependency

    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir), local_files_only=True)
    return LoadedTokenizer(tokenizer=tokenizer, source_dir=tok_dir)


# ---------------------------------------------------------------------------
# Tokenization wrappers -- match production's exact call shape
# ---------------------------------------------------------------------------


def tok_full(tokenizer: Any, text: str) -> list[int]:
    """Ground truth: what production actually does to the whole prompt."""
    return list(tokenizer.encode(text, add_special_tokens=False))


def tok_suffix(tokenizer: Any, text: str) -> list[int]:
    """Seam-rule binding requirement: suffix MUST use add_special_tokens=False.
    Named separately from tok_full so call sites document *why* False is used
    here, even though the value is the same encode() call."""
    return list(tokenizer.encode(text, add_special_tokens=False))


def tok_prefix_for_cache(tokenizer: Any, text: str) -> list[int]:
    """What would be cached from a prior turn: same call shape as tok_full,
    since the prefix is itself a complete, previously-seen prompt string."""
    return list(tokenizer.encode(text, add_special_tokens=False))


def count_bos_occurrences(tokenizer: Any, token_ids: list[int]) -> int:
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        # This tokenizer's config disables automatic BOS (add_bos_token=False
        # for DeepSeek-V4-Flash) but the BOS string is still an added token
        # with a fixed id -- look it up by string instead.
        bos_str = getattr(tokenizer, "bos_token", None)
        if bos_str is None:
            return -1  # cannot determine; caller must report this explicitly
        bos_id = tokenizer.convert_tokens_to_ids(bos_str)
    return sum(1 for t in token_ids if t == bos_id)


# ---------------------------------------------------------------------------
# Seam discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeamCandidate:
    offset: int
    description: str
    is_added_token_boundary: bool


def find_added_token_seams(tokenizer: Any, text: str) -> list[SeamCandidate]:
    """Candidate SAFE seams per the seam rule: character offsets immediately
    after a special/added token, discovered from the tokenizer's own
    offset_mapping (not guessed by string search) so the boundary is
    guaranteed to align with a real token span in tok(full)."""
    added_ids = set(tokenizer.all_special_ids)
    try:
        added_ids |= set(tokenizer.get_added_vocab().values())
    except AttributeError:
        pass

    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    seams: list[SeamCandidate] = []
    for tok_id, (start, end) in zip(ids, offsets, strict=True):
        if tok_id in added_ids and end < len(text):
            seams.append(
                SeamCandidate(
                    offset=end,
                    description=f"after added-token id={tok_id} span=[{start},{end})",
                    is_added_token_boundary=True,
                )
            )
    return seams


def find_midtoken_seam(tokenizer: Any, text: str) -> SeamCandidate | None:
    """Candidate UNSAFE seam: a character offset strictly inside a single
    multi-character token's span in tok(full). This is a genuine, derived
    (not guessed) violation of the seam rule -- splitting here means the
    prefix half never sees the bytes that the real BPE merge process used
    to justify merging into one token, so independent re-encoding of the
    two halves is not guaranteed to reproduce tok(full)."""
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]
    for start, end in offsets:
        if end - start > 1:
            mid = start + (end - start) // 2
            if start < mid < end:
                return SeamCandidate(
                    offset=mid,
                    description=f"mid-token split inside span=[{start},{end})",
                    is_added_token_boundary=False,
                )
    return None


# ---------------------------------------------------------------------------
# Core shadow-assertion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeamCheckResult:
    seam: SeamCandidate
    prefix_text: str
    suffix_text: str
    full_tokens: list[int]
    prefix_tokens: list[int]
    suffix_tokens: list[int]
    reconstructed: list[int]
    matches: bool


def check_seam(tokenizer: Any, full_text: str, seam: SeamCandidate) -> SeamCheckResult:
    prefix_text = full_text[: seam.offset]
    suffix_text = full_text[seam.offset :]

    full_tokens = tok_full(tokenizer, full_text)
    prefix_tokens = tok_prefix_for_cache(tokenizer, prefix_text)
    suffix_tokens = tok_suffix(tokenizer, suffix_text)
    reconstructed = prefix_tokens + suffix_tokens

    return SeamCheckResult(
        seam=seam,
        prefix_text=prefix_text,
        suffix_text=suffix_text,
        full_tokens=full_tokens,
        prefix_tokens=prefix_tokens,
        suffix_tokens=suffix_tokens,
        reconstructed=reconstructed,
        matches=(reconstructed == full_tokens),
    )
