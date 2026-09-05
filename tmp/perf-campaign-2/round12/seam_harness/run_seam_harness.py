#!/usr/bin/env python3
"""Seam-rule test harness (PREP ONLY -- not wired into production).

Run: python3 run_seam_harness.py
Requires no network access. Degrades gracefully (prints a SKIP with the
exact missing prerequisite) if a local tokenizer or the vendored DSv4
encoder module cannot be found on this machine.

Implements the four properties from the round-12 pre-registration:
  1. Round-trip identity at safe seams (must PASS) and unsafe seams
     (must be correctly REJECTED, i.e. found NOT to match).
  2. Adversarial seam corpus (combining chars, ZWJ emoji, digit runs,
     whitespace runs) at both a safe and an unsafe seam each.
  3. Template position-invariance: render(msgs[:i]) is a byte-prefix of
     render(msgs[:j]) for i<j.
  4. Normalizer detection from tokenizer.json.

Exits 0 if the harness ran to completion (regardless of whether individual
seam checks passed/failed/rejected as expected -- see the printed summary
for the actual verdicts). Exits 1 only on a hard SKIP (no local tokenizer).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dsv4_encoder_loader import load_vendored_encoder, render, vendored_source_path
from normalizer_check import check_normalizer
from seam_core import (
    check_seam,
    count_bos_occurrences,
    find_added_token_seams,
    find_midtoken_seam,
    load_tokenizer,
)
from seam_corpus import ADVERSARIAL_HOSTILE_STRINGS, ROUND_TRIP_MESSAGE_CORPUS

SEP = "=" * 78


def section(title: str) -> None:
    print(f"\n{SEP}\n{title}\n{SEP}")


def main() -> int:
    section("0. Prerequisite check")
    loaded = load_tokenizer()
    if loaded is None:
        print(
            "SKIP: no local tokenizer found. Needed: a directory containing "
            "both tokenizer.json and tokenizer_config.json for "
            "DeepSeek-V4-Flash, under either ~/.exo/models/ or "
            "~/.cache/huggingface/hub/models--*DeepSeek-V4-Flash*/snapshots/*/. "
            "This harness does not download models -- point it at an "
            "existing local copy."
        )
        return 1
    tokenizer = loaded.tokenizer
    print(f"Loaded tokenizer from: {loaded.source_dir}")

    encoder_module = load_vendored_encoder()
    if encoder_module is None:
        print(
            "SKIP (property 3 only): vendored DSv4 encoder module not found "
            "at the expected repo path "
            "(src/exo/worker/engines/mlx/vendor/deepseek_v4_encoding.py). "
            "Properties 1, 2, 4 will still run against synthetic text."
        )
    else:
        print(f"Loaded vendored DSv4 encoder from: {vendored_source_path()}")

    # -----------------------------------------------------------------
    section("4. Normalizer detection")
    tok_json = loaded.source_dir / "tokenizer.json"
    normalizer_report = check_normalizer(tok_json)
    print(normalizer_report.summary())

    # -----------------------------------------------------------------
    section("1. Round-trip identity (safe seams PASS, unsafe seams REJECTED)")
    round_trip_pass = True
    for corpus_name, messages in ROUND_TRIP_MESSAGE_CORPUS.items():
        print(f"\n--- corpus: {corpus_name} ---")
        if encoder_module is None:
            full_text = " ".join(str(m.get("content", "")) for m in messages)
            print("  (no vendored encoder -- using plain joined text as a stand-in)")
        else:
            full_text = render(encoder_module, messages)
        print(f"  full_text (repr, truncated): {full_text[:120]!r}...")

        safe_seams = find_added_token_seams(tokenizer, full_text)
        print(f"  candidate SAFE seams found: {len(safe_seams)}")
        for seam in safe_seams:
            result = check_seam(tokenizer, full_text, seam)
            status = "PASS" if result.matches else "**FAIL (unexpected)**"
            if not result.matches:
                round_trip_pass = False
            print(f"    seam@{seam.offset} [{seam.description}]: {status}")
            bos_count = count_bos_occurrences(tokenizer, result.full_tokens)
            recon_bos_count = count_bos_occurrences(tokenizer, result.reconstructed)
            print(
                f"      BOS count -- full: {bos_count}, "
                f"prefix+suffix reconstruction: {recon_bos_count}"
            )

        unsafe_seam = find_midtoken_seam(tokenizer, full_text)
        if unsafe_seam is None:
            print("  no mid-token unsafe seam found in this corpus (all tokens len<=1 char)")
        else:
            result = check_seam(tokenizer, full_text, unsafe_seam)
            # Correct behavior here is REJECTION: we expect result.matches
            # to be False. If it unexpectedly matches, that's worth flagging
            # (it doesn't mean the seam is safe, just that this particular
            # split happened not to break tokenization for this string) --
            # but it is NOT a harness failure since the rule guards on
            # *provable* safety, not lucky non-breakage.
            verdict = "correctly REJECTED (mismatch, as expected)" if not result.matches else (
                "did NOT break tokenization for this string (no proof of "
                "general safety -- still not a candidate-safe seam per the rule)"
            )
            print(f"  unsafe seam@{unsafe_seam.offset} [{unsafe_seam.description}]: {verdict}")

    # -----------------------------------------------------------------
    section("2. Adversarial seam corpus")
    for case in ADVERSARIAL_HOSTILE_STRINGS:
        print(f"\n--- {case.name}: {case.description} ---")
        full_text = f"<｜User｜>prefix {case.hostile_text} suffix<｜Assistant｜>"
        print(f"  full_text: {full_text!r}")

        safe_seams = find_added_token_seams(tokenizer, full_text)
        for seam in safe_seams:
            result = check_seam(tokenizer, full_text, seam)
            status = "PASS" if result.matches else "**FAIL (unexpected)**"
            print(f"    SAFE seam@{seam.offset}: {status}")

        unsafe_seam = find_midtoken_seam(tokenizer, full_text)
        if unsafe_seam is not None:
            result = check_seam(tokenizer, full_text, unsafe_seam)
            verdict = "correctly REJECTED" if not result.matches else "did not break (no proof of safety)"
            print(f"    UNSAFE seam@{unsafe_seam.offset}: {verdict}")
        else:
            print("    UNSAFE seam: none found (no multi-char token spans in this string)")

    # -----------------------------------------------------------------
    section("3. Template position-invariance (HIGHEST-VALUE CHECK)")
    if encoder_module is None:
        print("SKIP: vendored encoder not available, cannot test position-invariance.")
        position_invariant = None
    else:
        position_invariant = True
        for corpus_name, messages in ROUND_TRIP_MESSAGE_CORPUS.items():
            print(f"\n--- corpus: {corpus_name} ({len(messages)} messages) ---")
            renders = []
            for i in range(1, len(messages) + 1):
                renders.append(render(encoder_module, messages[:i]))
            for i in range(len(renders) - 1):
                shorter, longer = renders[i], renders[i + 1]
                is_prefix = longer.startswith(shorter)
                if not is_prefix:
                    position_invariant = False
                    # find first divergence point for a useful report
                    divergence = 0
                    for a, b in zip(shorter, longer):
                        if a != b:
                            break
                        divergence += 1
                    print(
                        f"  render(msgs[:{i + 1}]) is a prefix of "
                        f"render(msgs[:{i + 2}])? NO -- diverges at char "
                        f"{divergence}."
                    )
                    print(f"    render(msgs[:{i + 1}]) tail: ...{shorter[max(0, divergence - 30):][:60]!r}")
                    print(f"    render(msgs[:{i + 2}]) tail: ...{longer[max(0, divergence - 30):][:60]!r}")
                else:
                    print(f"  render(msgs[:{i + 1}]) is a prefix of render(msgs[:{i + 2}])? YES")

        print()
        if position_invariant:
            print(
                "RESULT: render() IS prefix-stable across all corpora tested "
                "(no re-merge/re-sort divergence observed in these cases)."
            )
        else:
            print(
                "RESULT: render() is NOT prefix-stable -- position-invariance "
                "FAILS. This confirms the pre-registration's prediction that "
                "merge_tool_messages()/sort_tool_results_by_call_order() in "
                "the vendored DSv4 encoder re-merge/re-sort messages in a way "
                "that changes earlier bytes when later messages are appended. "
                "A prefix cache keyed on message-list position is UNSAFE for "
                "any conversation shape that exercises this path (multi tool "
                "result reordering), independent of tokenizer-level seam "
                "safety."
            )

    # -----------------------------------------------------------------
    section("SUMMARY")
    print(f"Normalizer: {normalizer_report.summary()}")
    print(f"Round-trip identity at safe seams: {'all PASSED' if round_trip_pass else 'SOME FAILED (see above)'}")
    if position_invariant is None:
        print("Position-invariance: NOT TESTED (vendored encoder unavailable)")
    else:
        print(f"Position-invariance: {'PREFIX-STABLE' if position_invariant else 'NOT PREFIX-STABLE (see above)'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
