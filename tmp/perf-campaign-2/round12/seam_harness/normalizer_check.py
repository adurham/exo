"""Normalizer detection (seam-rule requirement #4).

Reads tokenizer.json directly (no transformers import needed) and reports
whether a normalizer is configured. A normalizer that mutates text before
BPE (NFC/NFKC folding, lowercasing, strip-accents, replace, etc.) can make
the seam rule unsafe even at an "obviously safe" special-token boundary,
because the normalizer can look across the seam and change bytes on either
side of it in a way that depends on what comes after. This module never
assumes "no normalizer" -- it reads the file and says exactly what it finds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class NormalizerReport:
    def __init__(self, raw: Any, tokenizer_json_path: Path):
        self.raw = raw
        self.tokenizer_json_path = tokenizer_json_path

    @property
    def is_present(self) -> bool:
        if self.raw is None:
            return False
        # HF tokenizers commonly encode "no normalizer" as an explicit
        # {"type": "Sequence", "normalizers": []} -- treat an EMPTY sequence
        # as "present but inert", not absent, and say so explicitly.
        return True

    @property
    def is_empty_sequence(self) -> bool:
        return (
            isinstance(self.raw, dict)
            and self.raw.get("type") == "Sequence"
            and self.raw.get("normalizers") == []
        )

    def summary(self) -> str:
        if self.raw is None:
            return f"No 'normalizer' key in {self.tokenizer_json_path} (null/absent)."
        if self.is_empty_sequence:
            return (
                f"normalizer = Sequence with 0 sub-normalizers in "
                f"{self.tokenizer_json_path} -- present in the schema but "
                f"INERT (identity transform). Seam-safety NOT invalidated by "
                f"the normalizer for this tokenizer."
            )
        kind = self.raw.get("type") if isinstance(self.raw, dict) else type(self.raw).__name__
        return (
            f"normalizer = {kind!r} in {self.tokenizer_json_path} -- "
            f"NON-EMPTY normalizer present. This CAN invalidate seam safety "
            f"(it may rewrite bytes across the seam depending on what "
            f"precedes/follows). Do not assume seams are safe without "
            f"checking whether this specific normalizer is seam-invariant. "
            f"Raw value: {self.raw!r}"
        )


def check_normalizer(tokenizer_json_path: Path) -> NormalizerReport:
    with open(tokenizer_json_path, encoding="utf-8") as f:
        data = json.load(f)
    return NormalizerReport(data.get("normalizer"), tokenizer_json_path)
