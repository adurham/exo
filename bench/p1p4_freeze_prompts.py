#!/usr/bin/env python3
"""Freeze every prompt for the P1-P4 campaign (2026-08-28) to disk ONCE.

Fixed-prompt discipline: build_prompt() embeds a uuid4 nonce, so every file
here is generated exactly once and reused byte-identical across all arms.
See docs/dspark-p1p4-campaign-preregister-2026-08-28.md.

Outputs under /tmp/ab/p1p4/prompts/:
  p100k_long.txt      100K corpus + long restatement tail (P1/P2 throughput, stream A)
  p100k_long_B.txt    distinct-nonce 100K + long tail (P2 stream B)
  p100k_codeA.txt     100K + embedded reference code ALPHA-7749 (P2 contamination A)
  p100k_codeB.txt     100K + embedded reference code BRAVO-3317 (P2 contamination B)
  bug3_<i>.txt        6x 100K + activation code ending in digit i (P2 Bug-3)
  depth_<t>.txt       P4a sweep prompts (4000, 7500, 9000, 14000, 32000, 64000)
"""
import pathlib
import sys

sys.path.insert(0, "/Users/adam.durham/repos/exo/bench")
from p3_depth_anchor_probe import build_prompt  # noqa: E402

OUT = pathlib.Path("/tmp/ab/p1p4/prompts")
OUT.mkdir(parents=True, exist_ok=True)

LONG_TAIL = (
    "\n\nWrite an exhaustive structured analysis of the corpus above: "
    "(1) enumerate every distinct topic pattern you can identify with three "
    "example section numbers each; (2) for EACH of the first 40 sections, "
    "restate its topic and its configuration number; (3) describe the "
    "numbering structure (configuration, stage pairs) in detail with worked "
    "examples; (4) end with a 500-word essay on what such a corpus could be "
    "used for. Be thorough and complete every part — do not stop early."
)
OLD_TAIL = "\n\nBriefly summarise the corpus above."


def freeze(name: str, text: str, predicted: int) -> None:
    p = OUT / name
    if p.exists():
        print(f"SKIP {name} (exists, {p.stat().st_size} chars)")
        return
    p.write_text(text)
    print(f"froze {name}: chars={len(text)} predicted_tokens={predicted}")


def with_tail(text: str, new_tail: str) -> str:
    assert text.endswith(OLD_TAIL), "tail mismatch — prompt builder changed?"
    return text[: -len(OLD_TAIL)] + new_tail


def inject(text: str, sentence: str) -> str:
    """Insert a sentence after the header line (early in the corpus)."""
    marker = "Corpus follows.\n\n"
    i = text.index(marker) + len(marker)
    return text[:i] + sentence + " " + text[i:]


def main() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        "/Users/adam.durham/.exo/models/deepseek-ai--DeepSeek-V4-Flash",
        trust_remote_code=True,
    )

    # P1 / P2 100K long-output prompts (distinct nonces by construction)
    t, n = build_prompt(100000, tok)
    freeze("p100k_long.txt", with_tail(t, LONG_TAIL), n)
    t, n = build_prompt(100000, tok)
    freeze("p100k_long_B.txt", with_tail(t, LONG_TAIL), n)

    # P2 contamination pair
    code_tail = "\n\nState the reference code for this corpus exactly, then briefly summarise the corpus."
    t, n = build_prompt(100000, tok)
    freeze("p100k_codeA.txt", with_tail(inject(t, "The reference code for this corpus is ALPHA-7749."), code_tail), n)
    t, n = build_prompt(100000, tok)
    freeze("p100k_codeB.txt", with_tail(inject(t, "The reference code for this corpus is BRAVO-3317."), code_tail), n)

    # P2 Bug-3 adversarial: activation code ending in each of 6 distinct digits.
    bug3_tail = "\n\nReply with ONLY the activation code, nothing else."
    for i, d in enumerate((0, 3, 4, 5, 7, 9)):
        code = f"8473921577{d}"
        t, n = build_prompt(100000, tok)
        freeze(f"bug3_{i}.txt", with_tail(inject(t, f"The activation code is {code}."), bug3_tail), n)

    # P4a depth sweep
    for target in (4000, 7500, 9000, 14000, 32000, 64000):
        t, n = build_prompt(target, tok)
        freeze(f"depth_{target}.txt", t, n)


if __name__ == "__main__":
    main()
