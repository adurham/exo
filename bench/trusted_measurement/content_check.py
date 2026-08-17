"""Content-correctness checking (design point 1).

A latency number from a request whose output was garbage is not a measurement,
it is noise. Every trusted record must carry proof that an output-content check
ran *in the same process and request* as the timing it accompanies.

The result type cannot be forged: :class:`ContentCheckResult` is only accepted
by the record model when it carries a proof token minted by :func:`run_needle_check`
(or another checker in this module) over its own payload.
"""

from __future__ import annotations

import random
import re
from typing import final

from pydantic import BaseModel, ConfigDict

from trusted_measurement.proof import ProofToken, mint_proof, verify_proof

__all__ = [
    "CONTENT_CHECK_DOMAIN",
    "ContentCheckResult",
    "NeedleHaystack",
    "build_needle_haystack",
    "run_needle_check",
]

CONTENT_CHECK_DOMAIN = "content_check"


@final
class NeedleHaystack(BaseModel):
    """A prompt with a known unique answer buried in filler text."""

    model_config = ConfigDict(frozen=True, strict=True)

    prompt: str
    needle: str
    question: str


@final
class ContentCheckResult(BaseModel):
    """Outcome of an output-content check, with unforgeable proof it ran."""

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    check_name: str
    passed: bool
    expected: str
    observed_excerpt: str
    proof: ProofToken

    def payload(self) -> dict[str, object]:
        return {
            "check_name": self.check_name,
            "passed": self.passed,
            "expected": self.expected,
            "observed_excerpt": self.observed_excerpt,
        }

    def proof_is_valid(self) -> bool:
        """True only if the proof was minted in this process for this payload."""
        return verify_proof(self.proof, CONTENT_CHECK_DOMAIN, self.payload())


def build_needle_haystack(
    *,
    filler_sentences: int,
    seed: int,
    needle: str | None = None,
) -> NeedleHaystack:
    """Build a deterministic needle-in-haystack prompt of a chosen size."""
    rng = random.Random(seed)  # noqa: S311 - prompt filler, not security
    chosen_needle = f"{rng.randrange(10**7, 10**8)}" if needle is None else needle
    lines = [
        f"Fact {index}: the calibration constant for unit "
        f"{rng.randrange(1000, 9999)} is {rng.randrange(100, 999)}."
        for index in range(filler_sentences)
    ]
    insert_at = rng.randrange(0, max(1, len(lines)))
    lines.insert(insert_at, f"The secret access code is {chosen_needle}.")
    question = "What is the secret access code? Reply with the digits only."
    prompt = "\n".join(lines) + "\n\n" + question
    return NeedleHaystack(prompt=prompt, needle=chosen_needle, question=question)


def run_needle_check(
    haystack: NeedleHaystack, completion: str, *, excerpt_chars: int = 200
) -> ContentCheckResult:
    """Run the needle check against a real completion and mint its proof."""
    normalised = re.sub(r"[^0-9A-Za-z]", "", completion)
    target = re.sub(r"[^0-9A-Za-z]", "", haystack.needle)
    passed = bool(target) and target in normalised
    excerpt = completion.strip()[:excerpt_chars]
    payload: dict[str, object] = {
        "check_name": "needle_in_haystack",
        "passed": passed,
        "expected": haystack.needle,
        "observed_excerpt": excerpt,
    }
    return ContentCheckResult(
        check_name="needle_in_haystack",
        passed=passed,
        expected=haystack.needle,
        observed_excerpt=excerpt,
        proof=mint_proof(CONTENT_CHECK_DOMAIN, payload),
    )
