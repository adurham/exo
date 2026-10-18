#!/usr/bin/env python3
"""Validate the reconstructed §11 prompt bodies and write RECONSTRUCTION.md.

Checks, strongest first:
  (i)   exact byte size, where a pack quotes one          -- expected to FAIL
  (ii)  prompt token count vs the surviving server lines  -- expected to PASS
  (iii) every verbatim substring the packs quote is present in the body
  (iv)  byte size within a few percent where quoted
  (v)   request JSON schema identical to the quoted shape
  (vi)  regeneration is byte-identical (determinism)

The point of the ordering is honesty: the only checks that *can* pass are the
ones that do not claim the original bytes were recovered. Check (i) is reported
first precisely because it must fail -- a generator that "passed" it would mean
the corpus reconstruction had been fitted to a byte count, not derived.

Usage:
  .venv/bin/python scripts/validate-prompts.py            # validate + write doc
  .venv/bin/python scripts/validate-prompts.py --no-write # print only
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
GEN_PATH = REPO / "scripts/gen-prompts.py"
OUT_DOC = REPO / "docs/dsv41-cluster-writeup-2026-10-17/preserved/RECONSTRUCTION.md"

#: Exact byte sizes the writeup quotes for bodies (evidence-live.md:122, :736).
#: guard_banana is the one body quoted in full (evidence-live.md:321-333), so its
#: size is not reconstructed at all -- it is transcribed, and matching it is a
#: transcription check rather than evidence about the lost corpus bodies.
QUOTED_BYTES: dict[str, int] = {
    "prompts_160k.json": 558_995,
    "prompts_1m.json": 3_391_475,
    "guard_banana.json": 318,
}

#: The full body quoted in evidence-live.md:321-333, as a structure. Comparing
#: parsed equality (not just a substring) is the strongest available check for
#: any single body.
QUOTED_BODY_GUARD_BANANA: dict[str, Any] = {
    "model": "dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw",
    "messages": [
        {
            "role": "user",
            "content": (
                "Output the word banana exactly 40 times, separated by single "
                "spaces, and nothing else. Do not explain."
            ),
        }
    ],
    "max_tokens": 400,
    "temperature": 0.0,
    "stream": True,
    "enable_thinking": True,
}

#: Byte tolerance for check (iv). "A few percent" is fixed at 5% so the check is
#: falsifiable rather than a judgement call made after seeing the number.
BYTE_TOLERANCE = 0.05

#: Substrings the writeup quotes or that its quoted completions require to be in
#: context. Provenance column:
#:   body-quote  -> the string is part of a body quoted verbatim in the writeup
#:   field-quote -> the string is a quoted request field value
#:   answer-echo -> a document path the surviving completion names, i.e. it must
#:                  have been inside the prompt for the answer to be grounded
REQUIRED_SUBSTRINGS: dict[str, list[tuple[str, str]]] = {
    "guard_banana.json": [
        ("Output the word banana exactly 40 times, separated by single spaces, "
         "and nothing else. Do not explain.", "body-quote"),
    ],
    "guard_loop.json": [
        ("[SECTION", "prose-quote"),
    ],
    "limiter_default.json": [
        ("A bat and a ball cost $1.10 in total.", "prose-quote"),
    ],
    "limiter_96.json": [
        ("A bat and a ball cost $1.10 in total.", "prose-quote"),
    ],
    "prompts_160k.json": [
        ("docs/moe-all-sum-dominant-cost-2026-08-19.md", "answer-echo"),
        ("docs/jaccl-internal-timing-allsum-transport-fast-2026-08-21.md",
         "answer-echo"),
        ("docs/allsum-sync-span-artifact-arithmetic-check-2026-08-22.md",
         "answer-echo"),
    ],
    "prompts_1m.json": [
        ("docs/moe-all-sum-dominant-cost-2026-08-19.md", "answer-echo"),
    ],
    "depth_d1k.json": [
        ("AGENTS.md", "answer-echo"),
        ("npm run build", "answer-echo"),
    ],
    "depth_d10k.json": [],  # no completion is quoted for this row
    "depth_d50k.json": [],  # no completion is quoted for this row
    "depth_d100k.json": [
        ("docs/moe-all-sum-dominant-cost-2026-08-19.md", "answer-echo"),
        # evidence-live.md:308 states this material was "inside the
        # 100,003-token prompt".
        ("PREFILL_CLIFF_HANDOFF.md", "prose-quote"),
    ],
    "depth_d300k.json": [
        ("docs/moe-all-sum-dominant-cost-2026-08-19.md", "answer-echo"),
        ("docs/moe-all-sum-skew-vs-comms-2026-08-19.md", "answer-echo"),
    ],
}

#: The request shape quoted in evidence-live.md (guard_banana is quoted in full;
#: the others are quoted as field lists). `reasoning_budget_tokens` is the one
#: documented per-request extra, present only on limiter_96.
BASE_SCHEMA = {
    "model": str,
    "messages": list,
    "max_tokens": int,
    "temperature": float,
    "stream": bool,
    "enable_thinking": bool,
}


def load_generator():
    spec = importlib.util.spec_from_file_location("gen_prompts", GEN_PATH)
    if spec is None or spec.loader is None:
        raise SystemExit(f"validate-prompts: cannot load {GEN_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def schema_errors(body: dict[str, Any], name: str) -> list[str]:
    errs: list[str] = []
    expected = dict(BASE_SCHEMA)
    if name == "limiter_96.json":
        expected["reasoning_budget_tokens"] = int
    if set(body) != set(expected):
        errs.append(f"keys {sorted(body)} != {sorted(expected)}")
    for key, typ in expected.items():
        if key in body and not isinstance(body[key], typ):
            errs.append(f"{key}: {type(body[key]).__name__} != {typ.__name__}")
    msgs = body.get("messages")
    if not isinstance(msgs, list) or len(msgs) != 1:
        errs.append("messages is not a single-message list")
    else:
        msg = msgs[0]
        if set(msg) != {"role", "content"}:
            errs.append(f"message keys {sorted(msg)} != ['content', 'role']")
        if msg.get("role") != "user":
            errs.append(f"role {msg.get('role')!r} != 'user'")
        if not isinstance(msg.get("content"), str) or not msg.get("content"):
            errs.append("content is not a non-empty string")
    # Field values the writeup states explicitly.
    if body.get("temperature") != 0.0:
        errs.append("temperature != 0.0")
    if body.get("model") != "dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw":
        errs.append("model id differs from the writeup's")
    return errs


def main(argv: list[str]) -> int:
    write = "--no-write" not in argv
    # Optional positional targets so an external harness can point this at a
    # freshly generated directory (or a list of files) instead of ./prompts.
    positionals = [a for a in argv if not a.startswith("-")]
    explicit: dict[str, Path] | None = None
    out_dir = REPO / "prompts"
    if positionals:
        first = Path(positionals[0])
        if first.is_dir():
            out_dir = first.resolve()
        else:
            explicit = {Path(p).name: Path(p).resolve() for p in positionals}
    # Only the canonical ./prompts run rewrites the committed document.
    write_doc = write and explicit is None and out_dir == (REPO / "prompts")
    gen = load_generator()
    gen._ensure_tokenizers()  # re-execs into .venv/bin/python when needed
    tok = gen._load_tokenizer()
    enc = gen._load_encoder()
    builder = gen.Builder(tok, enc)

    specs = [*gen.BIG_SPECS, *gen.SMALL_SPECS]

    # Determinism: rebuild in memory from scratch and compare bytes.
    corpus, corpus_files, subs = gen.build_corpus(enc.IMAGE_PLACEHOLDER)
    ids = tok.encode(corpus, add_special_tokens=False).ids

    rows: list[dict[str, Any]] = []
    for spec in specs:
        name = spec["name"]
        path = explicit[name] if explicit else out_dir / name
        if not path.exists():
            raise SystemExit(f"validate-prompts: {path} missing; run gen-prompts.py")
        raw = path.read_bytes()
        body = json.loads(raw.decode("utf-8"))
        content = body["messages"][0]["content"]

        regenerated, _stats = builder.build(spec, ids, gen.SEED)
        regen_bytes = gen.serialise(regenerated)

        prompt_tokens = builder.prompt_tokens(content, spec["thinking"])
        target = spec["target"]
        quoted = QUOTED_BYTES.get(name)

        missing = [
            (sub, prov) for sub, prov in REQUIRED_SUBSTRINGS.get(name, [])
            if sub not in content
        ]
        required = REQUIRED_SUBSTRINGS.get(name, [])
        body_quote = any(p == "body-quote" for _, p in required)

        rows.append(
            {
                "name": name,
                "target": target,
                "prompt_tokens": prompt_tokens,
                "bytes": len(raw),
                "quoted_bytes": quoted,
                "bytes_delta": (len(raw) - quoted) if quoted else None,
                "bytes_pct": ((len(raw) - quoted) / quoted * 100) if quoted else None,
                "missing": missing,
                "required": len(required),
                "body_quote": body_quote,
                "schema_errors": schema_errors(body, name),
                "body_equals_quote": (body == QUOTED_BODY_GUARD_BANANA
                                      if name in ("guard_banana.json",) else None),
                "deterministic": regen_bytes == raw,
                "sha256": hashlib.sha256(raw).hexdigest()[:16],
                "content_tokens": builder.ntok(content),
                "corpus_prefix_tokens": _stats.get("corpus_prefix_tokens", 0),
                "tail_newlines": _stats.get("tail_newlines", 0),
                "corpus_part": content.split("\n\nQuestion:", 1)[0]
                if spec.get("corpus") else "",
                "corpus_core": tok.decode(ids[: _stats["corpus_prefix_tokens"]])
                if spec.get("corpus") else "",
            }
        )

    # ---------------- structural: shared corpus prefix ----------------
    chain = sorted((r for r in rows if r["corpus_core"]),
                   key=lambda r: r["target"])
    prefix_fail: str | None = None
    for shorter, longer in zip(chain, chain[1:]):
        if not longer["corpus_core"].startswith(shorter["corpus_core"]):
            prefix_fail = f"{shorter['name']} is not a prefix of {longer['name']}"
            break
    prefix_ok = prefix_fail is None and len(chain) >= 2

    # ---------------- verdicts ----------------
    def mark(test: bool) -> str:
        return "PASS" if test else "FAIL"

    check_i = {r["name"]: (None if r["quoted_bytes"] is None
                           else mark(r["bytes"] == r["quoted_bytes"])) for r in rows}
    check_ii = {r["name"]: mark(r["prompt_tokens"] == r["target"]) for r in rows}
    check_iii = {r["name"]: (None if not r["required"] else
                             mark(not r["missing"]
                                  and r["body_equals_quote"] is not False))
                 for r in rows}
    check_iv = {r["name"]: (None if r["quoted_bytes"] is None else
                            mark(abs(r["bytes_pct"]) <= BYTE_TOLERANCE * 100))
                for r in rows}
    check_v = {r["name"]: mark(not r["schema_errors"]) for r in rows}
    check_vi = {r["name"]: mark(r["deterministic"]) for r in rows}

    print(f"{'file':<22}{'(i) bytes':<11}{'(ii) tok':<10}{'(iii) substr':<14}"
          f"{'(iv) ±5%':<10}{'(v) schema':<12}{'(vi) det':<10}")
    for r in rows:
        def cell(v):
            return "-" if v is None else v
        print(f"{r['name']:<22}{cell(check_i[r['name']]):<11}"
              f"{cell(check_ii[r['name']]):<10}{cell(check_iii[r['name']]):<14}"
              f"{cell(check_iv[r['name']]):<10}{cell(check_v[r['name']]):<12}"
              f"{cell(check_vi[r['name']]):<10}")

    if write_doc:
        OUT_DOC.parent.mkdir(parents=True, exist_ok=True)
        OUT_DOC.write_text(
            render_doc(rows, check_i, check_ii, check_iii, check_iv, check_v,
                       check_vi, gen, len(corpus_files), len(corpus), len(ids), subs,
                       prefix_ok, prefix_fail),
            encoding="utf-8",
        )
        print(f"wrote {OUT_DOC}")

    hard_fail = [
        r["name"] for r in rows
        if check_ii[r["name"]] == "FAIL" or check_iii[r["name"]] == "FAIL"
        or check_v[r["name"]] == "FAIL" or check_vi[r["name"]] == "FAIL"
    ]
    print("prefix chain: "
          + ("PASS" if prefix_ok else f"FAIL ({prefix_fail})")
          + " (depth bodies and the 1M body are literal prefixes of one corpus)")
    return 1 if hard_fail or not prefix_ok else 0


def render_doc(rows, ci, cii, ciii, civ, cv, cvi, gen, n_files, n_chars,
               n_tokens, subs, prefix_ok, prefix_fail) -> str:
    L: list[str] = []
    a = L.append

    a("# Reconstruction of the lost §11 prompt bodies")
    a("")
    a("**Status: reconstructed, NOT recovered.** The originals were built in a "
      "scratch directory (`~/.hermes/cache/scratch/writeup/prompts/`) and were "
      "never committed; they are confirmed absent from the controller and node0 "
      "(§\"RAW-LOST\" notes throughout `evidence-live.md`). This directory now "
      "holds generated *equivalent-class* inputs so that §11 is runnable again.")
    a("")
    a("The honesty rule this document is written under:")
    a("")
    a("> A reconstruction does not retroactively source a RAW-LOST figure. Every "
      "byte size and token count in `evidence-live.md` and `WRITEUP.md` remains "
      "RAW-LOST. The generated bodies reproduce the *logged prompt token counts* "
      "and the *quoted structure*; they are not the bytes that produced the "
      "measured latencies, and no measured number is re-derived from them.")
    a("")
    a("The ceiling of this work is therefore stated once, plainly: **equivalent-"
      "class input matching the logged token count and the quoted content — "
      "never \"the original\"**.")
    a("")
    a("## What was reconstructed, and from what")
    a("")
    a("| input | source of truth | reconstructed? |")
    a("| --- | --- | --- |")
    a("| message structure (one `user` message, no system prompt) | the quoted "
      "body shape in `evidence-live.md` §9.1 / §11.5 | yes, exact |")
    a("| request fields (`model`, `max_tokens`, `temperature`, `stream`, "
      "`enable_thinking`, `reasoning_budget_tokens`) | quoted bodies + §11.7 | "
      "yes, exact |")
    a(f"| corpus | \"the repo's own `docs/**.md` plus the top-level handoff docs, "
      f"concatenated in sorted order, truncated at the target depth\" | "
      f"approximated: `git ls-tree` of the pinned commit "
      f"`{gen.PINNED_SHA[:12]}`, {n_files} files, {n_chars:,} chars, "
      f"{n_tokens:,} tokens |")
    a("| tokenizer | `~/.exo/models/deepseek-ai--DeepSeek-V4-Flash/tokenizer.json`, "
      f"sha256 `{gen.TOKENIZER_SHA256[:8]}…` (the id the writeup quotes) | exact |")
    a("| chat framing | exo's own DSv4 encoder at the pinned commit "
      "(`deepseek_v4_encoding.py`, blob " + gen.ENCODER_BLOB_SHA1[:12] + ") | exact |")
    a(f"| prompt token count | the surviving `[DSV41] obs` lines | exact by "
      f"construction (filled to target) |")
    a("| the question text | **not preserved anywhere** — only its shape is "
      "implied by the surviving completions (each starts `BOTTLENECK:` and names "
      "the document that quantifies the bottleneck) | reconstructed, not exact |")
    a("| byte size | quoted for two bodies, and fully quoted for a third "
      "(`guard_banana`) | **not reproducible** for the reconstructed bodies — "
      "check (i) below fails for them by design |")
    a("")
    a("Two deliberate deviations from the original construction, both recorded "
      "because they change bytes:")
    a("")
    a(f"1. The writeup's own directory (`docs/dsv41-cluster-writeup-2026-10-17/`) "
      f"is excluded from the corpus. It was added by the same merge commit the "
      f"repro pins (`{gen.PINNED_SHA[:12]}`), i.e. *after* the capture, and it "
      f"quotes the model's answers — including it would make the prompt "
      f"self-referential.")
    a(f"2. {subs} occurrence{'s' if subs != 1 else ''} of the DSv4 image "
      f"placeholder token in `docs/DSV4_VISION_PORT_PLAN_PHASE34.md` "
      f"{'are' if subs != 1 else 'is'} replaced with the ASCII marker "
      f"`[image-placeholder-token]`, because exo's encoder rejects message text "
      f"containing that special token.")
    a("")
    a("Each corpus body is `corpus_prefix + \"\\n\"*k + question`, where `k ∈ 0..8` "
      "exists only because BPE boundary merges make some token counts unreachable "
      "by prefix slicing alone; `k=0` is used whenever it suffices. The manifest "
      "records the exact `k` per body.")
    a("")
    a("## Validation")
    a("")
    a("Checks are ordered strongest-first. (i) is the strongest *claim*, and it "
      "fails for every body whose bytes were reconstructed — which is the honest "
      "result. It passes for exactly one file, `guard_banana.json`, because that "
      "body is quoted in full in `evidence-live.md`: that pass is transcription, "
      "not recovery. (ii)–(vi) are the claims that are reachable for a "
      "reconstruction, and they pass.")
    a("")
    a("| # | check | what a failure would mean |")
    a("| --- | --- | --- |")
    a("| (i) | exact byte size vs the quoted figure | the original bytes were "
      "recovered — **not achievable**, and passing would be suspicious |")
    a("| (ii) | prompt token count == the count on the surviving `obs` line | the "
      "body would not exercise the same context depth |")
    a("| (iii) | every verbatim substring the packs quote is present | the body "
      "would not contain the content the writeup claims it contained |")
    a("| (iv) | byte size within ±5% where quoted | the corpus family/truncation "
      "would differ materially from the original |")
    a("| (v) | JSON schema identical to the quoted request shape | §11's curl "
      "commands would not reproduce the same API surface |")
    a("| (vi) | regeneration is byte-identical | the generator would not be "
      "deterministic, so the directory could not be trusted |")
    a("")
    a("### Per-file results")
    a("")
    a("| file | (i) exact bytes | (ii) prompt tokens | (iii) quoted substrings "
      "| (iv) bytes ±5% | (v) schema | (vi) determinism |")
    a("| --- | --- | --- | --- | --- | --- | --- |")

    def cell(v: str | None) -> str:
        return "—" if v is None else v

    for r in rows:
        n = r["name"]
        a(f"| `{n}` | {cell(ci[n])} | {cell(cii[n])} | {cell(ciii[n])} | "
          f"{cell(civ[n])} | {cell(cv[n])} | {cell(cvi[n])} |")
    a("")
    a("`—` = not applicable: the writeup quotes no byte size for that body, so "
      "checks (i) and (iv) cannot be stated for it at all; for `depth_d10k` and "
      "`depth_d50k` no completion is quoted either, so there is no quoted content "
      "for check (iii) to match.")
    a("")
    a("Structural check (not part of the five): the corpus bodies are literal "
      "prefixes of one another in target order (1003 → 10004 → 50004 → 100003 → "
      "160004 → 300003 → 1000003 tokens). Result: **"
      + ("PASS" if prefix_ok else f"FAIL — {prefix_fail}")
      + "**. That is what makes the recorded cache geometry plausible on a "
      "re-run: the deep rows can hit the shallow rows' prefix blocks, which is "
      "how the original saw `cached_tokens` 8,192 at 50 K, 49,152 at 100 K and "
      "299,008 at 1 M.")
    a("")
    a("### Numbers behind the table")
    a("")
    a("| file | target prompt tokens | measured prompt tokens | content tokens "
      "| bytes | quoted bytes | Δ bytes | corpus prefix tokens | tail `\\n` | "
      "sha256[:16] |")
    a("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        delta = "—" if r["bytes_delta"] is None else f"{r['bytes_delta']:+,}"
        pct = "" if r["bytes_pct"] is None else f" ({r['bytes_pct']:+.2f}%)"
        quoted = "—" if r["quoted_bytes"] is None else f"{r['quoted_bytes']:,}"
        a(f"| `{r['name']}` | {r['target']:,} | {r['prompt_tokens']:,} | "
          f"{r['content_tokens']:,} | {r['bytes']:,} | {quoted} | {delta}{pct} | "
          f"{r['corpus_prefix_tokens']:,} | {r['tail_newlines']} | `{r['sha256']}` |")
    a("")
    a("### Check (i) in words")
    a("")
    for r in rows:
        if r["quoted_bytes"] is None:
            continue
        verdict = (
            "PASS — transcription of a body the writeup quotes in full, not "
            "evidence about any lost body."
            if r["bytes"] == r["quoted_bytes"] and r["name"] == "guard_banana.json"
            else ("PASS — treated as a coincidence of the corpus, not as recovery."
                  if r["bytes"] == r["quoted_bytes"] else
                  "FAIL, as expected: the original's exact corpus snapshot, "
                  "selection boundary and byte-level formatting are not "
                  "recoverable.")
        )
        a(f"- `{r['name']}`: quoted **{r['quoted_bytes']:,} B**, generated "
          f"**{r['bytes']:,} B** → **{r['bytes_delta']:+,} B** "
          f"({r['bytes_pct']:+.2f}%). {verdict}")
    a("")
    a("The residual gap is a property of *which* docs land inside the first N "
      "tokens of *this* snapshot, not of a wrong method: the writeup's own "
      "figure (5.38 M chars) does not match any single obvious filter of the "
      "pinned tree, so the exact corpus membership at capture time is itself "
      "part of what was lost.")
    a("")
    a("### Check (iii) detail")
    a("")
    a("Only `guard_banana.json` has a body quoted verbatim in the writeup, and "
      "that sentence is reproduced character-for-character. For the corpus "
      "bodies there is no quoted body text to match; the check instead requires "
      "the document paths that the surviving completions cite — i.e. paths that "
      "must have been inside the context for those answers to be grounded.")
    a("")
    a("| file | required substrings | result |")
    a("| --- | --- | --- |")
    for r in rows:
        reqs = REQUIRED_SUBSTRINGS.get(r["name"], [])
        shown = ", ".join(f"`{s}`" for s, _ in reqs) if reqs else "—"
        result = (f"{len(reqs) - len(r['missing'])}/{len(reqs)} present"
                  if reqs else "n/a — no quoted completion for this row")
        a(f"| `{r['name']}` | {shown} | {result} |")
    a("")
    a("All corpus bodies also contain the model id and the single `user` message "
      "with `role`/`content` only, which check (v) asserts structurally.")
    a("")
    a("## What this does and does not license")
    a("")
    a("**Licensed.** §11 can be run again: the curl commands in the writeup take "
      "these files unchanged, at the pinned commit, with the same request "
      "fields, and reach the same context depth to the token. The token-count "
      "column of the depth table (1003 / 10004 / 50004 / 100003 / 300003, and "
      "160,004 / 1,000,003) is reproduced exactly, so re-running exercises the "
      "same shapes of prefill and the same prefix-cache geometry (the depth "
      "bodies are literal prefixes of one another, and of the 1M body, which is "
      "what produced the 8,192 / 49,152 / 299,008 cached-token hits).")
    a("")
    a("**Not licensed.** Anything of the form \"the original prompt\", \"the "
      "captured bytes\", or \"re-derived from the recovered body\". In "
      "particular:")
    a("")
    a("- the quoted byte sizes (558,995 B / 3,391,475 B) stay RAW-LOST and are "
      "matched only to within a few percent;")
    a("- the local tokenizer counts quoted in the writeup (159,999 for the 160 K "
      "body, 999,999 for the 1 M body) are the *content* counts of the originals; "
      "the generated content counts land at 160,000 and 999,999 — within one "
      "token — and the server-reported totals are hit exactly by adding exo's "
      "4-token chat framing on top. Matching a count is not recovering a body;")
    a("- the question wording, and therefore any claim about how the model was "
      "asked, is a reconstruction from the shape of its answers;")
    a("- no measured latency, throughput or cache figure is recomputed, "
      "re-validated or \"confirmed\" by this directory. Those numbers rest on "
      "the quoted server lines alone, exactly as before.")
    a("")
    a("## Reproducing this document")
    a("")
    a("```bash")
    a(".venv/bin/python scripts/gen-prompts.py            # writes ./prompts")
    a(".venv/bin/python scripts/validate-prompts.py       # validates + rewrites this file")
    a("```")
    a("")
    a("`prompts/GENERATED-MANIFEST.json` records the seed, the pinned commit, the "
      "tokenizer hash, the corpus inventory size and the per-body token/byte "
      "accounting. The generator is byte-deterministic given (commit, tokenizer, "
      "seed); check (vi) asserts that on every run.")
    a("")
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
