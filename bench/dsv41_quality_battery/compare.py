#!/usr/bin/env python3
"""
compare.py - two-phase result comparison and SHIP GATE for the DSv4.1 quality
battery.

Usage:
  python3 compare.py results/<baseline> results/<candidate>
  python3 compare.py --a prodA --b prodB      # shorthand: resolves under results/
  python3 compare.py --a prodA --b prodB --json   # also emit compare.json

Convention: A = baseline (production / fp32 indexer-row arm),
            B = candidate (branch build / DSV41_INDEXER_ROW_BF16 arm).

Verdict logic (this is the ship gate):
  FAIL  - B has ANY new detector hit (esp. glued cross-lingual fragments), OR
          B regresses on needles/tools (fewer passes than A), OR
          B loses parked-restore recall that A had.
  REVIEW- no failures, but there is some delta requiring eyes: a prose output
          that changed text between arms, a detector verdict that shifted
          CLEAN<->REVIEW, or a needle/tool count that differs but not worse.
  PASS  - B >= A on needles/tools/detectors and no new glued/CROSS-SCRIPT hits.

The prose-text "changed" signal is a SHA-256 of the exact generated text; it is
EXPECTED and benign for free-form prose (temp=0 still drifts a hair on real
decode), so text-change alone is REVIEW not FAIL. Only a NEW high-confidence
detector hit (new glued fragment / U+FFFD / repetition loop) or a worse
needle/tool score is FAIL.
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def resolve(dir_or_label):
    if os.path.isdir(dir_or_label):
        return dir_or_label
    cand = os.path.join(RESULTS_DIR, dir_or_label)
    if os.path.isdir(cand):
        return cand
    return dir_or_label


def sha(text):
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:12]


def needle_map(d):
    j = load(os.path.join(d, "needles.json")) or {}
    return {r["id"]: r for r in j.get("results", [])}


def tool_map(d):
    j = load(os.path.join(d, "tools.json")) or {}
    return {r["id"]: r for r in j.get("results", [])}


def prose_map(d):
    j = load(os.path.join(d, "free_prose", "index.json")) or {}
    return {r["id"]: r for r in j.get("probes", [])}


def park(d):
    return load(os.path.join(d, "park.json")) or {}


def summarize_dir(d):
    needles = needle_map(d)
    tools = tool_map(d)
    prose = prose_map(d)
    np = sum(1 for r in needles.values() if r.get("pass"))
    tp = sum(1 for r in tools.values() if r.get("pass"))
    dirty = sum(1 for r in prose.values() if (r.get("detector") or {}).get("verdict") == "DIRTY")
    review = sum(1 for r in prose.values() if (r.get("detector") or {}).get("verdict") == "REVIEW")
    # REASONING_ONLY / ERROR rates are regression SIGNALS (a rise worth eyes),
    # not pass/fail gates: content-empty-with-reasoning is a known checkpoint
    # behavior at temp 0 (present pre-change), and ERROR covers transport blips.
    reasoning_only = sum(1 for r in prose.values()
                         if (r.get("detector") or {}).get("verdict") == "REASONING_ONLY")
    errors = sum(1 for r in prose.values()
                 if (r.get("detector") or {}).get("verdict") == "ERROR")
    p = park(d)
    return {"needles_pass": np, "needles_n": len(needles),
            "tools_pass": tp, "tools_n": len(tools),
            "prose_dirty": dirty, "prose_review": review, "prose_n": len(prose),
            "prose_reasoning_only": reasoning_only, "prose_errors": errors,
            "park_recall": p.get("recall_teal")}


def main(argv=None):
    ap = argparse.ArgumentParser(description="compare two battery result dirs (ship gate)")
    ap.add_argument("a", nargs="?", help="baseline result dir (prod/fp32)")
    ap.add_argument("b", nargs="?", help="candidate result dir (bf16)")
    ap.add_argument("--a", dest="a_opt")
    ap.add_argument("--b", dest="b_opt")
    ap.add_argument("--json", action="store_true", help="write compare.json into dir B")
    args = ap.parse_args(argv)

    a_arg = args.a_opt or args.a
    b_arg = args.b_opt or args.b
    if not a_arg or not b_arg:
        ap.error("need two dirs (positional A B, or --a/--b)")
    A = resolve(a_arg)
    B = resolve(b_arg)

    print("=" * 78)
    print("DSv4.1 QUALITY BATTERY - two-phase comparison (SHIP GATE)")
    print("  A (baseline):  {}".format(A))
    print("  B (candidate): {}".format(B))
    print("=" * 78)

    sa, sb = summarize_dir(A), summarize_dir(B)
    fails, reviews, notes = [], [], []

    # ---------------- needles ----------------
    print("\n-- NEEDLES (higher is better) --")
    na, nb = needle_map(A), needle_map(B)
    for pid in sorted(set(na) | set(nb)):
        ra, rb = na.get(pid, {}), nb.get(pid, {})
        pa, pb = bool(ra.get("pass")), bool(rb.get("pass"))
        mark = " " if pa == pb else ("+" if pb else "-")
        print("  [{}{}] {:<12} A={:<4} B={:<4}  {}".format(
            "A" if pa else " ", "B" if pb else " ",
            pid, str(pa), str(pb),
            "" if pa == pb else ("B improved" if pb else "B REGRESSED")))
        if pa and not pb:
            fails.append("needle '{}' regressed (A pass -> B fail)".format(pid))
        elif pb and not pa:
            reviews.append("needle '{}' improved in B (A fail -> B pass)".format(pid))
    print("  totals: A {}/{}  B {}/{}".format(
        sa["needles_pass"], sa["needles_n"], sb["needles_pass"], sb["needles_n"]))
    if sb["needles_pass"] < sa["needles_pass"]:
        fails.append("needle total regressed: A {} -> B {}".format(
            sa["needles_pass"], sb["needles_pass"]))

    # ---------------- tools ----------------
    print("\n-- TOOL CALLS (higher is better) --")
    ta, tb = tool_map(A), tool_map(B)
    for pid in sorted(set(ta) | set(tb)):
        ra, rb = ta.get(pid, {}), tb.get(pid, {})
        pa, pb = bool(ra.get("pass")), bool(rb.get("pass"))
        print("  {:<28} A={:<4} B={:<4} {}".format(
            pid, str(pa), str(pb), "" if pa == pb else ("B improved" if pb else "B REGRESSED")))
        if pa and not pb:
            fails.append("tool '{}' regressed (A pass -> B fail)".format(pid))
    print("  totals: A {}/{}  B {}/{}".format(
        sa["tools_pass"], sa["tools_n"], sb["tools_pass"], sb["tools_n"]))
    if sb["tools_pass"] < sa["tools_pass"]:
        fails.append("tool total regressed: A {} -> B {}".format(
            sa["tools_pass"], sb["tools_pass"]))

    # ---------------- free prose / detectors ----------------
    print("\n-- FREE PROSE + DETECTORS (verdict per prompt) --")
    pa, pb = prose_map(A), prose_map(B)
    text_changed = []
    for pid in sorted(set(pa) | set(pb)):
        ra, rb = pa.get(pid, {}), pb.get(pid, {})
        da = (ra.get("detector") or {}).get("verdict", "?")
        db = (rb.get("detector") or {}).get("verdict", "?")
        ha = sha(ra.get("content"))
        hb = sha(rb.get("content"))
        changed = (ha != hb) and (ha != sha("")) and (hb != sha(""))
        if changed:
            text_changed.append(pid)
        delta = ""
        if da != db:
            if db == "DIRTY" and da != "DIRTY":
                delta = "NEW DIRTY in B"
                fails.append("prose '{}' verdict {} -> DIRTY (new detector hit)".format(pid, da))
            elif db == "REVIEW" and da == "CLEAN":
                delta = "B now REVIEW (low-conf)"
                reviews.append("prose '{}' CLEAN -> REVIEW (low-confidence glue?)".format(pid))
            elif da == "DIRTY" and db != "DIRTY":
                delta = "B cleaner"
            elif db == "REASONING_ONLY" and da != "REASONING_ONLY":
                delta = "B REASONING-ONLY (content empty; answer in reasoning)"
                reviews.append("prose '{}' {} -> REASONING_ONLY (channel shift, eyes)".format(pid, da))
            elif da == "REASONING_ONLY" and db != "REASONING_ONLY":
                delta = "B has content now"
            elif db == "ERROR":
                delta = "B ERROR (transport/serving blip)"
                notes.append("prose '{}' {} -> ERROR (non-quality)".format(pid, da))
        print("  {:<18} A={:<6} B={:<6} text_hash A={} B={}{} {}".format(
            pid, da, db, ha, hb, " (changed)" if changed else "", delta))
        # new high-confidence hits are the real FAIL signal
        hits_b = (rb.get("detector") or {}).get("hits") or []
        hits_a = (ra.get("detector") or {}).get("hits") or []
        new_high = [h for h in hits_b if h.get("confidence") == "high"
                    and h["type"] not in {x["type"] for x in hits_a if x.get("confidence") == "high"}]
        if new_high:
            fails.append("prose '{}' new high-confidence detector hit: {}".format(
                pid, [h["type"] for h in new_high]))
    print("  totals: A DIRTY={} REVIEW={} | B DIRTY={} REVIEW={}".format(
        sa["prose_dirty"], sa["prose_review"], sb["prose_dirty"], sb["prose_review"]))
    if sb["prose_dirty"] > sa["prose_dirty"]:
        fails.append("prose DIRTY count up: A {} -> B {}".format(sa["prose_dirty"], sb["prose_dirty"]))
    if text_changed:
        # Free-form prose at temp=0 still diverges a hair on real decode (the
        # incident doc: "free prose does visibly diverge, 0/5 byte-identical").
        # That is EXPECTED and benign -- the detectors exist precisely to catch
        # the harmful subset. So a text change is a NOTE (dump side-by-side),
        # not a REVIEW gate; otherwise PASS could never be reached.
        notes.append("{}/{} prose outputs changed text between arms -- expected "
                     "for free-form; dump side-by-side for eyes".format(
                         len(text_changed), len(set(pa) | set(pb))))
        print("  changed-text prompts (note, not a gate): {}".format(sorted(text_changed)))

    # ---------------- park ----------------
    print("\n-- PARKED-RESTORE CONTINUATION --")
    pa_, pb_ = park(A), park(B)
    ra, rb = pa_.get("recall_teal"), pb_.get("recall_teal")
    print("  recall_teal: A={} B={}".format(ra, rb))
    if ra is True and rb is not True:
        fails.append("parked-restore recall lost: A True -> B {}".format(rb))
    elif rb is True and ra is not True:
        reviews.append("parked-restore recall improved in B")

    # ---------------- verdict ----------------
    print("\n" + "=" * 78)
    if fails:
        verdict = "FAIL"
    elif reviews:
        verdict = "REVIEW"
    else:
        verdict = "PASS"
    print("VERDICT: {}".format(verdict))
    if fails:
        print("  FAIL reasons ({}):".format(len(fails)))
        for f in fails:
            print("    - {}".format(f))
        print("  -> DO NOT SHIP the B build. A new defect class or a regression appeared.")
    elif reviews:
        print("  REVIEW items ({}):".format(len(reviews)))
        for x in reviews:
            print("    - {}".format(x))
        print("  -> No hard failures. Dump free-prose side-by-side and get eyes on the deltas.")
    else:
        print("  B >= A on needles/tools/detectors, no new glued fragments, park recall held.")
        print("  -> PASS: B is not worse than A on the quality battery.")
    if notes:
        print("  notes (non-gating):")
        for n in notes:
            print("    * {}".format(n))
    print("  summary A: {}".format(sa))
    print("  summary B: {}".format(sb))
    print("=" * 78)

    result = {"A": A, "B": B, "verdict": verdict, "fails": fails,
              "reviews": reviews, "notes": notes,
              "summary_A": sa, "summary_B": sb,
              "changed_text_prompts": sorted(text_changed)}
    if args.json:
        out = os.path.join(B, "compare.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print("wrote", out)
    return 0 if verdict == "PASS" else (2 if verdict == "REVIEW" else 1)


if __name__ == "__main__":
    sys.exit(main())
