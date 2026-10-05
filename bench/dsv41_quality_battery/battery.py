#!/usr/bin/env python3
"""
DSv4.1 live quality battery
===========================

A parameterized, stdlib-only probe runner for the DSv4.1 cluster at
http://macstudio-m4-1.tail19c543.ts.net:52415 (model
dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw).

It exists to catch the defect CLASS that slipped past synthetic/unit gates
twice on this model (see
  ~/repos/exo/docs/incidents/lmhead-mxfp8-cross-lingual-glue-defect-2026-09-13.md
  ~/repos/exo/docs/lmhead-mxfp8-defect-and-fallback-investigations-2026-09-14.md):
   (1) hc_expand bf16-comb ~1.08% mean rel err - only real precision checks saw it
   (2) lm_head mxfp8 - garbled cross-lingual subword fragments glued onto real
       words with NO space ("angleсь", "camerauden", "pavement他身上"), invisible
       to 15 high-margin eval tasks that all scored byte-identical.

Design goals (from house style: ~/.hermes/cache/scratch/1m_soak2.sh,
park_live_proof.sh):
  * stdlib only (urllib) or curl-via-subprocess; no third-party deps.
  * resumable: every probe writes its own result file; re-runs skip done work
    unless --force.
  * no blind sleeps: waits are bounded polls on real state (server_running()).
  * the deep-context build reuses the proven filler+planted-needle pattern and
    the measured live ratio ~5.111 chars/token.

Modes:
  plan      - build all prompts OFFLINE, print sizes + token estimates, no net
  build     - construct the deep-context prompt (default depth 350000 tokens)
              with stratified planted needles, send it (warms the prefix cache),
              save the prompt + result
  eval      - run the full probe suite against the warm context:
              needles, low-margin free prose (+ multilingual), degeneration
              detectors, tool-call sanity, parked-restore continuation
  all       - build then eval
  selftest  - run the detector suite against synthetic strings (no cluster)

Usage:
  python3 battery.py --selftest
  python3 battery.py --label prodA --depth 350000 build
  python3 battery.py --label prodA eval
  python3 battery.py --label prodA --depth 120000 all      # quick mode
  python3 battery.py --label prodB --skip-park eval --force
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone

# --------------------------------------------------------------------------
# Constants / defaults
# --------------------------------------------------------------------------
DEFAULT_API = "http://macstudio-m4-1.tail19c543.ts.net:52415"
DEFAULT_MODEL = "dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
CHARS_PER_TOKEN = 5.111          # measured live ratio (543996 chars -> 106445 tok)
DEFAULT_DEPTH = 350000           # default eval depth (tokens)
QUICK_DEPTH = 120000             # lighter quick mode
SENTENCE = ("The quick brown fox jumps over the lazy dog while the cluster "
            "serves tokens at steady pace. ")

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

# needle depth stratifications (fractions of the filler)
NEEDLES = {
    # id -> (depth_fraction, planted_text)
    "exact":     (0.10, "\n\n[FACT] The vault code is 8492.\n\n"),
    "negation":  (0.25, "\n\n[CLARIFY] To be unambiguous: the vault code is NOT "
                        "8493, it is 8492.\n\n"),
    "multihop_a":(0.30, "\n\n[FACT] The project is named Aurora.\n\n"),
    "distractor":(0.50, "\n\n[DISTRACTOR] The number 8493 appears here but it is "
                        "NOT the vault code; do not use 8493.\n\n"),
    "control":   (0.55, "\n\n[FACT] The verification token is QK-7731.\n\n"),
    "multihop_b":(0.70, "\n\n[FACT] Aurora's launch month is November.\n\n"),
    "restate":   (0.90, "\n\n[FACT] Reminder: the vault code is 8492.\n\n"),
}

# Questions asked during eval (appended, as a reused-turn follow-up, to the
# exact same big prompt sent at build time so the warm prefix cache hits).
NEEDLE_PROBES = [
    # id, expects, question, keywords (any-of, case-insensitive)
    {"id": "exact",
     "q": "\n\nQuestion: What is the vault code? Reply with just the 4-digit number.",
     "keywords": ["8492"], "expect": "contains 8492"},
    {"id": "paraphrase",
     "q": "\n\nQuestion: What code opens the vault? Answer briefly.",
     "keywords": ["8492"], "expect": "contains 8492"},
    {"id": "negation",
     "q": "\n\nQuestion: Is the vault code 8493? Answer yes or no, then state the correct vault code.",
     "keywords": ["8492"], "expect": "contains 8492 (and should not affirm 8493)"},
    {"id": "distractor",
     "q": "\n\nQuestion: A number 8493 appears in the document. Is that the vault code? What is the actual vault code?",
     "keywords": ["8492"], "expect": "contains 8492"},
    {"id": "control",
     "q": "\n\nQuestion: What is the verification token? Reply with the exact token.",
     "keywords": ["qk-7731", "qk 7731", "7731"], "expect": "contains QK-7731"},
    {"id": "multihop",
     "q": "\n\nQuestion: What month does the project launch? Answer with the month name.",
     "keywords": ["november"], "expect": "contains November"},
]

# ---------------------------------------------------------------- free prose
# Low-margin free-form prose prompts, INCLUDING multilingual/mixed-language
# content. No auto pass/fail (detectors only). Saved per-prompt for blind
# side-by-side human/model comparison between the two phases.
FREE_PROSE = [
    {"id": "rain", "tag": "prose",
     "p": "Write a short paragraph about the sound of rain on a metal roof."},
    {"id": "rank_langs", "tag": "ranking",
     "p": ("Rank these five programming languages by how pleasant they are for a "
           "beginner, best to worst: Rust, Python, C, JavaScript, Haskell. Give a "
           "one-sentence reason for each, in your ranked order.")},
    {"id": "lighthouse", "tag": "continuation",
     "p": ("Continue this story in one vivid paragraph: 'The lighthouse keeper "
           "noticed the light was not his own.'")},
    {"id": "paperclip", "tag": "list",
     "p": "List ten genuinely different uses for a paperclip."},
    {"id": "zh_autumn", "tag": "multilingual",
     "p": ("先用一句中文描写秋雨的声音，然后用英文解释这句话的意思。"
           "(Write one Chinese sentence describing the sound of autumn rain, "
           "then explain it in English.)")},
    {"id": "zh_en_teapot", "tag": "multilingual",
     "p": ("Write a bilingual product description for a handmade ceramic teapot: "
           "the first half in English, the second half in Chinese (中文).")},
    {"id": "ja_sky", "tag": "multilingual",
     "p": "次の質問に日本語で答えてください：なぜ空は青いのですか？"},
    {"id": "ru_winter", "tag": "multilingual",
     "p": ("Напиши короткий абзац на русском языке о зимнем утре, "
           "затем переведи его на английский.")},
    {"id": "de_baum", "tag": "multilingual",
     "p": ("Schreibe zwei Sätze auf Deutsch über einen alten Baum, "
           "dann gib eine englische Übersetzung dazu.")},
    {"id": "fr_river", "tag": "multilingual",
     "p": ("Translate this to French, then add one short comment in English: "
           "'The river runs cold in November.'")},
    {"id": "market_dawn", "tag": "prose",
     "p": ("Describe a street market at dawn in vivid sensory detail, at least "
           "150 words.")},
    {"id": "tests_opinion", "tag": "opinion",
     "p": ("Are standardized tests a good measure of intelligence? Write two "
           "paragraphs arguing a position.")},
    {"id": "hash_table", "tag": "technical",
     "p": ("Explain how a hash table works, in plain language, as if to a curious "
           "teenager.")},
    {"id": "codeswitch_email", "tag": "multilingual",
     "p": ("Write a friendly email that opens with a Spanish greeting, includes a "
           "French compliment in the middle, and closes in English.")},
    {"id": "pencil_history", "tag": "prose",
     "p": "Write about the history of the pencil, at least 200 words."},
    {"id": "sequences", "tag": "continuation",
     "p": ("Complete these sequences and then explain your reasoning in prose: "
           "[2,4,8,16, ...], [a,c,e,g, ...], [1,1,2,3,5, ...].")},
    {"id": "ja_zh_intro", "tag": "multilingual",
     "p": "日本語で短い自己紹介を書いて、最後に中国語で一文だけ付け加えてください。"},
    {"id": "limerick", "tag": "prose",
     "p": "Write a limerick about the moon."},
    {"id": "es_continue", "tag": "multilingual",
     "p": ("Continúa en el mismo idioma y estilo: 'El viento soplaba fuerte sobre "
           "las colinas, y ...'")},
    {"id": "scene_image", "tag": "prose",
     "p": ("Describe this scene as if narrating an image you are looking at: a "
           "street market, a red umbrella, three pigeons on a wet pavement.")},
]

# --------------------------------------------------------------- tool calls
TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"],
                     "description": "Temperature unit"},
        }, "required": ["city", "unit"]}}},
    {"type": "function", "function": {
        "name": "get_forecast",
        "description": "Get a multi-day weather forecast for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name"},
            "days": {"type": "integer", "description": "Number of days to forecast"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        }, "required": ["city", "days"]}}},
    {"type": "function", "function": {
        "name": "set_thermostat",
        "description": "Set a target temperature for a thermostat in a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"},
            "target_temp": {"type": "number"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        }, "required": ["city", "target_temp"]}}},
]

TOOL_PROBES = [
    {"id": "t1_weather_celsius",
     "p": "What is the current weather in Paris, in celsius?",
     "expect_names": ["get_weather"], "expect_args": {"city": "paris", "unit": "celsius"}},
    {"id": "t2_forecast_tokyo",
     "p": "Give me a 5-day forecast for Tokyo.",
     "expect_names": ["get_forecast"], "expect_args": {"city": "tokyo", "days": 5}},
    {"id": "t3_thermostat_berlin",
     "p": "Set the thermostat to 21 degrees in Berlin.",
     "expect_names": ["set_thermostat"], "expect_args": {"city": "berlin", "target_temp": 21}},
    {"id": "t4_raining_london",
     "p": "Is it raining in London right now?",
     "expect_names": ["get_weather", "get_forecast"], "expect_args": {"city": "london"}},
    {"id": "t5_nyc_tomorrow",
     "p": "What is the temperature in New York City tomorrow?",
     "expect_names": ["get_forecast", "get_weather"], "expect_args": {"city": "new york"}},
    {"id": "t6_forecast_madrid_3",
     "p": "Forecast for Madrid for the next 3 days, in celsius.",
     "expect_names": ["get_forecast"], "expect_args": {"city": "madrid", "days": 3}},
    {"id": "t7_heat_rome",
     "p": "Turn the heat up to 23 degrees in Rome.",
     "expect_names": ["set_thermostat"], "expect_args": {"city": "rome", "target_temp": 23}},
    {"id": "t8_sydney_fahrenheit",
     "p": "Current weather in Sydney in fahrenheit.",
     "expect_names": ["get_weather"], "expect_args": {"city": "sydney", "unit": "fahrenheit"}},
    {"id": "t9_sydney_2day",
     "p": "A 2-day forecast for Sydney.",
     "expect_names": ["get_forecast"], "expect_args": {"city": "sydney", "days": 2}},
    {"id": "t10_unit_vs_days_confusion",
     "p": "For Madrid, I want the weather right now and separately a 4-day outlook in celsius.",
     "expect_names": ["get_weather", "get_forecast"], "expect_args": {"city": "madrid"}},
]

# ------------------------------------------------------------------ parking
PARK_MEMORY = "\n\n[PERSONAL] Remember: my favorite color is teal.\n\n"
PARK_Q = "\n\nQuestion: What is my favorite color? Answer with just the color."


# ==========================================================================
# Detectors
# ==========================================================================
_FOREIGN_SCRIPTS = {"cyrillic", "greek", "han", "kana", "hangul", "arabic",
                    "hebrew", "thai", "devanagari"}


def char_script(ch):
    """Coarse Unicode script class for a single character."""
    cp = ord(ch)
    if ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
        return "latin"
    if ch.isascii():
        if ch.isdigit():
            return "digit"
        return "ascii-other"
    if ch.isspace():
        return "space"
    if 0x0400 <= cp <= 0x04FF:
        return "cyrillic"
    if 0x0500 <= cp <= 0x052F:      # Cyrillic Supplement
        return "cyrillic"
    if 0x0370 <= cp <= 0x03FF:
        return "greek"
    if 0x3040 <= cp <= 0x30FF:
        return "kana"
    if (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or (0xF900 <= cp <= 0xFAFF):
        return "han"
    if 0xAC00 <= cp <= 0xD7AF:
        return "hangul"
    if 0x0600 <= cp <= 0x06FF:
        return "arabic"
    if 0x0590 <= cp <= 0x05FF:
        return "hebrew"
    if 0x0E00 <= cp <= 0x0E7F:
        return "thai"
    if 0x0900 <= cp <= 0x097F:
        return "devanagari"
    # Latin-1 supplement / extended (accents) are still "latin-ish" -> treat as
    # neutral so legit words like "café"/"über"/"naïve" do NOT trip the detector.
    if 0x00C0 <= cp <= 0x024F:
        return "latin"
    return "other-nonascii"


# small common-word list for the LOW-CONFIDENCE same-script glue heuristic.
# Best-effort supplementary signal only; see find_glued_fragments().
_COMMON_WORDS = set("""
the of and to a in is it you that he was for on are as with his they i at be this
have from or one had by word but not what all were we when your can said there use
an each which she do how their if will up other about out many then them these so
some her would make like him into time has look two more write go see number no way
could people my than first water been call who oil its now find long down day did get
come made may part over new sound take only little work know place year live me back
give most very after thing our just name good sentence man think say great where help
through much before line right too mean old any same tell boy follow came want show
also around form three small set put end does another well large must big even such
because turn here why ask went men read need land different home us move try kind hand
picture again change off play spell air away animal house point page letter mother
answer found study still learn should america world every near add food between own
below country plant last school father keep tree never start city earth eye light
thought head under story saw left few while along might close something seem next hard
open example begin life always those both paper together got group often run important
until children side feet car mile night walk white sea began grow took river four carry
state once book hear stop without second later miss idea enough eat face watch far real
almost let above girl sometimes mountain cut young talk soon list song being leave family
camera angle pattern pavement background grass thigh thigh s thigh
""".split())

# common English suffixes: a common-word+tail that ends in one of these is a
# legitimate word (reading, talked, quickly), NOT a glue fragment.
_COMMON_SUFFIXES = {"ing", "ed", "ly", "er", "es", "est", "tion", "sion",
                    "ness", "ment", "able", "ible", "ous", "ive", "ful",
                    "less", "ers", "ings", "ted", "ded", "led", "red", "ned"}


def find_glued_fragments(text):
    """Detect the lm_head glued-fragment signature.

    Primary (HIGH confidence): a whitespace-delimited token that mixes Latin
    letters with a non-Latin script (Cyrillic/Han/Kana/Hangul/Greek/...). This is
    the exact signature from the incident: "angleсь", "pattern火热",
    "backgroundфабрика", "pavement他身上".

    Secondary (LOW confidence): a same-script nonsense glue ("camerauden",
    "thighsuden") - a pure lowercase token that splits into a common word + a
    trailing fragment that is not itself a word. Best-effort; reported but only
    drives a REVIEW verdict on its own.
    """
    hits = []
    tokens = re.findall(r"\S+", text)
    for ti, raw in enumerate(tokens):
        tok = raw.strip(".,;:!?\"'()[]{}<>«»…—–-")
        if len(tok) < 4:
            continue
        scripts = [char_script(c) for c in tok]
        has_latin = "latin" in scripts
        foreign = {s for s in scripts if s in _FOREIGN_SCRIPTS}
        # The real signature is a script switch MID-WORD: a Latin letter
        # adjacent (no space/punctuation between) to a non-Latin-script letter.
        # "angleсь", "pattern火热", "pavement他身上" all have this adjacency.
        # Legitimate code-switching like "雨声淅沥。The rain ..." has CJK
        # sentence punctuation (。) between the runs, so it does NOT trip this.
        adjacent_switch = False
        for c1, c2, s1, s2 in zip(tok, tok[1:], scripts, scripts[1:]):
            if {s1, s2} == {"latin", "cyrillic"} or \
               (s1 == "latin" and s2 in _FOREIGN_SCRIPTS) or \
               (s2 == "latin" and s1 in _FOREIGN_SCRIPTS):
                adjacent_switch = True
                break
        if has_latin and foreign and adjacent_switch:
            # reconstruct the latin run + the adjoining foreign run for context
            latin_part = "".join(c for c, s in zip(tok, scripts) if s == "latin")
            foreign_part = "".join(c for c, s in zip(tok, scripts)
                                   if s in _FOREIGN_SCRIPTS)
            # context = ~40 chars either side of this token in the text
            pos = text.find(raw)
            ctx = text[max(0, pos - 40):pos + len(raw) + 40] if pos >= 0 else raw
            hits.append({
                "type": "cross_script_glue",
                "confidence": "high",
                "token": tok,
                "token_index": ti,
                "latin": latin_part if latin_part else "<interleaved>",
                "foreign": foreign_part,
                "foreign_scripts": sorted(foreign),
                "context": ctx,
            })
        elif len(tok) >= 6 and tok.isalpha() and tok.islower() and tok not in _COMMON_WORDS:
            # secondary same-script glue heuristic ("camerauden", "thighsuden"):
            # a common word + a short trailing fragment that is neither a word
            # nor a common English suffix. Deliberately conservative -- the
            # primary signal is the cross-script case above.
            for cut in range(4, min(len(tok) - 1, 9)):
                head, tail = tok[:cut], tok[cut:]
                # Stricter rule (2026-10-05): the original flagged innocent
                # English words ("against" = again+st). A genuine glued pair
                # needs a substantive head and a word-like tail.
                if (len(head) >= 4 and head in _COMMON_WORDS and 3 <= len(tail) <= 4
                        and any(ch in "aeiou" for ch in tail.lower())
                        and tail not in _COMMON_WORDS
                        and tail not in _COMMON_SUFFIXES):
                    hits.append({
                        "type": "same_script_glue",
                        # ADVISORY (2026-10-05): this heuristic cannot separate
                        # real English words ("beginner"=begin+ner,
                        # "against"=again+st) from genuine glued fragments
                        # without a dictionary. Kept as data (counts still
                        # recorded + printed) but it no longer drives REVIEW;
                        # the cross-script detector above is the incident
                        # signature and stays verdict-driving.
                        "confidence": "advisory",
                        "token": tok,
                        "token_index": ti,
                        "head": head,
                        "tail": tail,
                        "context": raw,
                    })
                    break
    return hits


def find_replacement_chars(text):
    return [{"type": "replacement_char", "confidence": "high",
             "count": text.count("\ufffd"),
             "positions": [m.start() for m in re.finditer("\ufffd", text)][:20]}
            ] if "\ufffd" in text else []


def find_repetitions(text, min_len=2, max_len=6, min_repeats=3):
    """3+ consecutive repeats of any word n-gram of length min_len..max_len."""
    words = re.findall(r"\S+", text.lower())
    n = len(words)
    hits = []
    for L in range(min_len, max_len + 1):
        i = 0
        while i + L <= n:
            ng = words[i:i + L]
            cnt = 1
            j = i + L
            while j + L <= n and words[j:j + L] == ng:
                cnt += 1
                j += L
            if cnt >= min_repeats:
                hits.append({"type": "repetition", "confidence": "high",
                             "ngram_len": L, "ngram": " ".join(ng),
                             "count": cnt, "start_word": i})
                i = j
            else:
                i += 1
    # keep the longest n-gram per overlapping start region (reduce noise)
    dedup = {}
    for h in hits:
        key = h["start_word"] // max(1, h["ngram_len"])
        if key not in dedup or h["count"] * h["ngram_len"] > dedup[key]["count"] * dedup[key]["ngram_len"]:
            dedup[key] = h
    return list(dedup.values())


_DOUBT_MARKERS = ["wait,", "let me reconsider", "actually,", "hmm,",
                  "i'm not sure", "let me re-evaluate", "on second thought",
                  "let me think again", "reconsidering"]


def find_self_doubt(text):
    low = text.lower()
    hits = []
    for m in _DOUBT_MARKERS:
        c = low.count(m)
        if c >= 4:
            hits.append({"type": "self_doubt_loop", "confidence": "high",
                         "marker": m, "count": c})
    qs = re.findall(r"[^.!?\n]{6,60}\?", low)
    qc = Counter(q.strip() for q in qs)
    for q, c in qc.items():
        if c >= 3 and len(q) > 10:
            hits.append({"type": "self_doubt_loop", "confidence": "high",
                         "repeated_question": q[:80], "count": c})
    return hits


def run_detectors(text):
    """Run every degeneration detector on one output string.

    Returns {verdict, hits, counts}. Verdict:
      DIRTY    - at least one HIGH-confidence hit (cross-script glue, U+FFFD,
                 repetition loop, self-doubt loop)
      REVIEW   - LOW-confidence hits (currently none active; reserved)
      CLEAN    - no verdict-driving hits (advisory hits still recorded)
    """
    hits = []
    hits += find_glued_fragments(text)
    hits += find_replacement_chars(text)
    hits += find_repetitions(text)
    hits += find_self_doubt(text)
    high = [h for h in hits if h.get("confidence") == "high"]
    low = [h for h in hits if h.get("confidence") == "low"]
    if high:
        verdict = "DIRTY"
    elif low:
        verdict = "REVIEW"
    else:
        verdict = "CLEAN"
    counts = Counter(h["type"] for h in hits)
    return {"verdict": verdict, "hits": hits, "counts": dict(counts),
            "n_high": len(high), "n_low": len(low)}


# ==========================================================================
# HTTP
# ==========================================================================
class Client:
    def __init__(self, api, model, timeout):
        self.api = api.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(self, messages, max_tokens=256, temperature=0.0,
             tools=None, reasoning_effort=None, extra=None):
        payload = {"model": self.model, "messages": messages,
                   "max_tokens": max_tokens, "temperature": temperature,
                   "stream": False}
        if tools is not None:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        if extra:
            payload.update(extra)
        return self._post(payload)

    def _post(self, payload):
        url = self.api + "/v1/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                body = r.read().decode("utf-8", "replace")
                return {"status": r.status, "body": body,
                        "elapsed": time.time() - t0, "error": None}
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", "replace")
            except Exception:
                body = ""
            return {"status": e.code, "body": body,
                    "elapsed": time.time() - t0, "error": "HTTPError"}
        except Exception as e:
            return {"status": None, "body": "",
                    "elapsed": time.time() - t0,
                    "error": "{}: {}".format(type(e).__name__, e)}

    @staticmethod
    def extract(resp):
        """Pull content/reasoning/tool_calls/usage out of a raw response."""
        out = {"content": "", "reasoning": "", "tool_calls": [],
               "finish_reason": None, "usage": {}, "error": None}
        if resp.get("error") and not resp.get("body"):
            out["error"] = resp["error"]
            return out
        try:
            d = json.loads(resp["body"])
        except Exception as e:
            out["error"] = "json parse: {}".format(e)
            return out
        if isinstance(d, dict) and d.get("error"):
            out["error"] = json.dumps(d["error"])[:400]
            return out
        try:
            ch = (d.get("choices") or [{}])[0]
            m = ch.get("message") or {}
            out["content"] = m.get("content") or ""
            out["reasoning"] = m.get("reasoning_content") or ""
            out["tool_calls"] = m.get("tool_calls") or []
            out["finish_reason"] = ch.get("finish_reason")
            out["usage"] = d.get("usage") or {}
        except Exception as e:
            out["error"] = "shape: {}".format(e)
        return out


# ==========================================================================
# Context builder
# ==========================================================================
def build_deep_prompt(target_tokens, include_needles=True):
    """Filler + stratified planted needles, exactly the 1m_soak2 pattern.

    Returns (prompt_text, meta). Needles planted at their fractions of the
    filler; the final instruction closes the document and warms the prefix.
    """
    total_chars = int(target_tokens * CHARS_PER_TOKEN)
    n = max(1, total_chars // len(SENTENCE))
    needle_at = {}
    if include_needles:
        for _id, (frac, txt) in NEEDLES.items():
            idx = min(n - 1, max(0, int(frac * n)))
            # if two needles land on the same slot, still append both
            needle_at.setdefault(idx, []).append(txt)
    parts = []
    for idx in range(n):
        if idx in needle_at:
            parts.extend(needle_at[idx])
        parts.append(SENTENCE)
    body = "".join(parts)
    prompt = body + ("\n\nNow reply with exactly: done")
    meta = {
        "target_tokens": target_tokens,
        "chars_per_token": CHARS_PER_TOKEN,
        "chars": len(prompt),
        "est_tokens": int(len(prompt) / CHARS_PER_TOKEN),
        "sentences": n,
        "needles": {k: {"fraction": v[0], "index": min(n - 1, max(0, int(v[0] * n)))}
                    for k, v in NEEDLES.items()} if include_needles else {},
    }
    return prompt, meta


def build_small_prompt(target_tokens, memory=None):
    """Small (~8K tok) context; optional planted memory line."""
    total_chars = int(target_tokens * CHARS_PER_TOKEN)
    n = max(1, total_chars // len(SENTENCE))
    body = (SENTENCE * n)
    if memory:
        body = memory + body
    return body


# ==========================================================================
# Scoring helpers
# ==========================================================================
def keyword_hit(text, keywords):
    low = text.lower()
    return any(k.lower() in low for k in keywords)


def check_tool_call(resp_extracted, expect_names, expect_args):
    """Validate tool_calls shape + intent + near-identical field correctness."""
    detail = {}
    if resp_extracted.get("error"):
        return False, {"error": resp_extracted["error"]}
    tcs = resp_extracted.get("tool_calls") or []
    if not tcs:
        # Classify the documented DSv4 DSML leak: the tool call was emitted as
        # DSML text in the content channel instead of a structured tool_call
        # (intermittent, serving-side; visible in both arms if it recurs).
        content = resp_extracted.get("content") or ""
        if "<invoke name=" in content or "<tool_calls>" in content or "< invoke" in content:
            return False, {"error": "dsml_leak_in_content", "content_head": content[:200]}
        return False, {"error": "no tool_calls in response"}
    ok = True
    parsed_all = []
    for tc in tcs:
        fn = (tc.get("function") or {})
        name = fn.get("name")
        raw_args = fn.get("arguments")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        except Exception as e:
            ok = False
            detail.setdefault("errors", []).append("unparseable args for {}: {}".format(name, e))
            args = {}
        parsed_all.append({"name": name, "args": args, "raw_arguments": raw_args})
    detail["tool_calls"] = parsed_all
    names = [p["name"] for p in parsed_all]
    if not any(nm in expect_names for nm in names):
        ok = False
        detail["errors"] = detail.get("errors", []) + [
            "chosen {} not in expected {}".format(names, expect_names)]
    # field correctness on the chosen call(s)
    merged = {}
    for p in parsed_all:
        merged.update(p["args"])
    low_keys = {k.lower(): k for k in merged}
    for k, v in expect_args.items():
        got = merged.get(k, merged.get(low_keys.get(k.lower(), ""), None))
        if got is None:
            ok = False
            detail.setdefault("errors", []).append("missing arg '{}'".format(k))
            continue
        if isinstance(v, str):
            got_s = str(got).strip().lower()
            want_s = v.lower()
            # City names commonly come back with suffixes ("New York City"
            # for "new york"); accept exact or containment.
            if got_s != want_s and want_s not in got_s:
                ok = False
                detail.setdefault("errors", []).append(
                    "arg '{}': got {!r} want {!r}".format(k, got, v))
        elif isinstance(v, (int, float)):
            try:
                if abs(float(got) - float(v)) > 1e-6:
                    ok = False
                    detail.setdefault("errors", []).append(
                        "arg '{}': got {!r} want {!r}".format(k, got, v))
            except Exception:
                ok = False
                detail.setdefault("errors", []).append(
                    "arg '{}': got non-numeric {!r} want {!r}".format(k, got, v))
    return ok, detail


# ==========================================================================
# Result I/O (resumable)
# ==========================================================================
def label_dir(label):
    d = os.path.join(RESULTS_DIR, label)
    os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(d, "free_prose"), exist_ok=True)
    os.makedirs(os.path.join(d, "raw"), exist_ok=True)
    return d


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat()


# ==========================================================================
# Probe runners
# ==========================================================================
def do_build(client, args, d):
    path = os.path.join(d, "build.json")
    if os.path.exists(path) and not args.force:
        print("[build] exists, skipping (use --force):", path)
        return load_json(path)
    prompt, meta = build_deep_prompt(args.depth)
    print("[build] depth={} chars={} est_tokens={} sentences={}".format(
        args.depth, meta["chars"], meta["est_tokens"], meta["sentences"]))
    # persist the exact prompt for eval reuse and for the record
    with open(os.path.join(d, "build_prompt.txt"), "w") as f:
        f.write(prompt)
    print("[build] sending (this is the long prefill; timeout={}s)...".format(client.timeout))
    resp = client.chat([{"role": "user", "content": prompt}],
                       max_tokens=96, reasoning_effort="low")
    ext = client.extract(resp)
    with open(os.path.join(d, "raw", "build.resp.json"), "w") as f:
        f.write(resp.get("body") or "")
    result = {"mode": "build", "started": now_iso(), "meta": meta,
              "http_status": resp["status"], "elapsed_s": resp["elapsed"],
              "error": resp.get("error") or ext.get("error"),
              "finish_reason": ext["finish_reason"],
              "usage": ext["usage"], "content": ext["content"][:300],
              # FULL reply kept for the probe replay (the multiturn shape
              # replays it as the assistant turn — see do_needles).
              "content_full": ext["content"]}
    result["finished"] = now_iso()
    save_json(path, result)
    print("[build] done status={} elapsed={:.1f}s usage={}".format(
        resp["status"], resp["elapsed"], ext["usage"]))
    return result


def load_build_prompt(d):
    p = os.path.join(d, "build_prompt.txt")
    if os.path.exists(p):
        with open(p) as f:
            return f.read()
    # reconstruct (eval without a prior build)
    prompt, meta = build_deep_prompt(DEFAULT_DEPTH)
    with open(p, "w") as f:
        f.write(prompt)
    return prompt


def _build_reply_for_replay(d):
    """The assistant text to replay in the probe's middle turn.

    The prefix-reuse contract needs everything up to the prompt-end checkpoint
    to match the resident cache; replaying the build's own reply (or 'done')
    as a separate assistant turn keeps the divergence ABOVE that checkpoint,
    so each probe is a ~10-row delta instead of a full re-prefill.
    """
    try:
        b = load_json(os.path.join(d, "build.json"))
        c = (b.get("content_full") or b.get("content") or "").strip()
        return c or "done"
    except Exception:
        return "done"


def do_needles(client, args, d, big_prompt):
    path = os.path.join(d, "needles.json")
    if os.path.exists(path) and not args.force:
        print("[needles] exists, skipping")
        return load_json(path)
    replay = _build_reply_for_replay(d)
    print("[needles] replaying build reply as middle turn: {!r}".format(replay[:60]))
    results = []
    for probe in NEEDLE_PROBES:
        q = probe["q"].strip()
        # reuse-turn (multiturn shape): [user: big prompt, assistant: build
        # reply, user: question].  The huge prefix matches the resident
        # conversation's cache through its prompt-end checkpoint, so only the
        # reply + question tail is re-fed (~10 rows), not the whole context.
        msgs = [{"role": "user", "content": big_prompt},
                {"role": "assistant", "content": replay},
                {"role": "user", "content": q}]
        resp = client.chat(msgs, max_tokens=512, reasoning_effort="low")
        ext = client.extract(resp)
        content = ext["content"]
        passed = keyword_hit(content, probe["keywords"]) if not ext.get("error") else False
        via_reasoning = False
        if not passed and not ext.get("error"):
            # The reasoning channel sometimes carries the answer while content
            # comes back empty (observed on this checkpoint both pre- and
            # post-change: reasoning ends 'So answer 8492. Brief.**8492**',
            # finish=stop, content='').  Count it, but record the channel so
            # compare.py can treat a channel SHIFT as REVIEW, not a regression.
            via_reasoning = keyword_hit(ext.get("reasoning") or "", probe["keywords"])
            passed = passed or via_reasoning
        rec = {"id": probe["id"], "question": q.strip(), "expect": probe["expect"],
               "keywords": probe["keywords"], "pass": passed,
               "answered_in_reasoning": via_reasoning,
               "http_status": resp["status"], "elapsed_s": resp["elapsed"],
               "error": resp.get("error") or ext.get("error"),
               "finish_reason": ext["finish_reason"],
               "usage": ext["usage"],
               "content": content, "reasoning": ext["reasoning"][:500]}
        with open(os.path.join(d, "raw", "needle_{}.json".format(probe["id"])), "w") as f:
            json.dump({"raw_body": resp.get("body")}, f, indent=2)
        results.append(rec)
        print("[needles] {:<12} {}  ({}s)  {!r}".format(
            probe["id"], "PASS" if passed else "FAIL",
            int(resp["elapsed"]), content[:60]))
    npass = sum(1 for r in results if r["pass"])
    out = {"n": len(results), "n_pass": npass, "results": results}
    save_json(path, out)
    print("[needles] {}/{} pass".format(npass, len(results)))
    return out


def do_free_prose(client, args, d):
    idx_path = os.path.join(d, "free_prose", "index.json")
    fpdir = os.path.join(d, "free_prose")
    index = []
    for probe in FREE_PROSE:
        fpath = os.path.join(fpdir, "{}.json".format(probe["id"]))
        if os.path.exists(fpath) and not args.force:
            rec = load_json(fpath)
            index.append(rec)
            print("[prose] {:<16} (cached) verdict={}".format(probe["id"], rec.get("detector", {}).get("verdict")))
            continue
        # reasoning_effort="low": on reasoning models the default effort can
        # consume the ENTIRE token budget before content is emitted (observed
        # 2026-10-05: prose probes returned content='' with finish=stop and the
        # full answer in the reasoning channel). Low effort + a content-fitted
        # budget keeps the visible-content signal alive.
        resp = client.chat([{"role": "user", "content": probe["p"]}],
                           max_tokens=max(args.prose_tokens, 4096), temperature=0.0,
                           reasoning_effort="low")
        ext = client.extract(resp)
        content = ext["content"]
        if content:
            det = run_detectors(content)
        elif (ext.get("reasoning") or "").strip():
            # Answer (or at least real output) landed in the reasoning channel
            # while content is empty. Observed at temp 0 on this checkpoint
            # both pre- and post-change (paraphrase needle, codeswitch_email).
            # Distinct verdict so compare.py can diff the rate across arms
            # (a RISE would be a regression signal) instead of false-FAILing.
            det = {"verdict": "REASONING_ONLY", "hits": [], "counts": {}}
        else:
            det = {"verdict": "ERROR", "hits": [], "counts": {}}
        rec = {"id": probe["id"], "tag": probe["tag"], "prompt": probe["p"],
               "http_status": resp["status"], "elapsed_s": resp["elapsed"],
               "error": resp.get("error") or ext.get("error"),
               "finish_reason": ext["finish_reason"], "usage": ext["usage"],
               "content": content, "reasoning": (ext.get("reasoning") or "")[:800],
               "detector": det}
        with open(os.path.join(d, "raw", "prose_{}.json".format(probe["id"])), "w") as f:
            json.dump({"raw_body": resp.get("body")}, f, indent=2)
        save_json(fpath, rec)
        index.append(rec)
        print("[prose] {:<16} {:>4}s  verdict={:<6} hits={}".format(
            probe["id"], int(resp["elapsed"]), det["verdict"], det["counts"]))
        # write a plain-text dump for blind side-by-side comparison
        with open(os.path.join(fpdir, "{}.txt".format(probe["id"])), "w") as f:
            f.write("# {} [{}]\nPROMPT: {}\n\n{}".format(
                probe["id"], probe["tag"], probe["p"], content))
    dirty = [r for r in index if (r.get("detector") or {}).get("verdict") == "DIRTY"]
    review = [r for r in index if (r.get("detector") or {}).get("verdict") == "REVIEW"]
    save_json(idx_path, {"n": len(index), "n_dirty": len(dirty),
                         "n_review": len(review), "probes": index})
    print("[prose] {} prompts, {} DIRTY, {} REVIEW".format(len(index), len(dirty), len(review)))
    return {"n": len(index), "n_dirty": len(dirty), "n_review": len(review), "probes": index}


def do_tools(client, args, d):
    path = os.path.join(d, "tools.json")
    if os.path.exists(path) and not args.force:
        print("[tools] exists, skipping")
        return load_json(path)
    results = []
    for probe in TOOL_PROBES:
        resp = client.chat([{"role": "user", "content": probe["p"]}],
                           max_tokens=512, temperature=0.0, tools=TOOLS,
                           reasoning_effort="low")
        ext = client.extract(resp)
        ok, detail = check_tool_call(ext, probe["expect_names"], probe["expect_args"])
        rec = {"id": probe["id"], "prompt": probe["p"],
               "expect_names": probe["expect_names"], "expect_args": probe["expect_args"],
               "pass": ok, "http_status": resp["status"], "elapsed_s": resp["elapsed"],
               "error": resp.get("error") or ext.get("error"),
               "finish_reason": ext["finish_reason"], "usage": ext["usage"],
               "tool_calls": ext["tool_calls"], "detail": detail,
               "content": ext["content"]}
        with open(os.path.join(d, "raw", "tools_{}.json".format(probe["id"])), "w") as f:
            json.dump({"raw_body": resp.get("body")}, f, indent=2)
        results.append(rec)
        names = [ (tc.get("function") or {}).get("name") for tc in ext.get("tool_calls") or [] ]
        print("[tools] {:<26} {}  calls={} {}".format(
            probe["id"], "PASS" if ok else "FAIL", names,
            "" if ok else detail.get("errors", "")))
    npass = sum(1 for r in results if r["pass"])
    out = {"n": len(results), "n_pass": npass, "results": results}
    save_json(path, out)
    print("[tools] {}/{} pass".format(npass, len(results)))
    return out


def do_park(client, args, d):
    """Parked-restore continuation.

    (i)   build a ~8K-token context carrying 'my favorite color is teal'
    (ii)  open 2 more distinct ~8K contexts to evict A (store keeps 2 resident,
          >4096 tok -> parks)
    (iii) re-request A as a true continuation turn (same prefix + 'what is my
          favorite color?') and assert recall == teal. Optionally grep the
          engine log (--ssh-park-check) for a restore line.
    """
    path = os.path.join(d, "park.json")
    if os.path.exists(path) and not args.force:
        print("[park] exists, skipping")
        return load_json(path)
    small_tok = args.park_tokens
    mem_ctx = build_small_prompt(small_tok, memory=PARK_MEMORY)
    # A turn-1: memory context, no question yet
    resp_a = client.chat([{"role": "user", "content": mem_ctx}],
                         max_tokens=192, reasoning_effort="low")
    ext_a = client.extract(resp_a)
    reply_a = ext_a["content"]
    # distinct B, C contexts to force LRU eviction of A
    b = build_small_prompt(small_tok, memory="\n\n[PERSONAL] My favorite animal is a pangolin.\n\n")
    c = build_small_prompt(small_tok, memory="\n\n[PERSONAL] My favorite number is seventeen.\n\n")
    resp_b = client.chat([{"role": "user", "content": b}], max_tokens=32, reasoning_effort="low")
    resp_c = client.chat([{"role": "user", "content": c}], max_tokens=32, reasoning_effort="low")
    # A turn-2: continuation (same prefix + assistant reply + new user question)
    msgs = [{"role": "user", "content": mem_ctx},
            {"role": "assistant", "content": reply_a},
            {"role": "user", "content": PARK_Q.strip()}]
    # 256 (was 32): on this checkpoint the reasoning channel consumed all 32
    # tokens and the answer never reached content (finish=length, empty).
    resp_a2 = client.chat(msgs, max_tokens=256, reasoning_effort="low")
    ext_a2 = client.extract(resp_a2)
    recalled = keyword_hit(ext_a2["content"], ["teal"]) or keyword_hit(
        ext_a2.get("reasoning") or "", ["teal"])
    ssh_log = None
    if args.ssh_park_check:
        ssh_log = _ssh_park_log()
    rec = {"mode": "park", "park_tokens": small_tok,
           "turn1": {"status": resp_a["status"], "elapsed_s": resp_a["elapsed"],
                     "content": reply_a, "error": resp_a.get("error") or ext_a.get("error")},
           "evict_B_elapsed_s": resp_b["elapsed"], "evict_C_elapsed_s": resp_c["elapsed"],
           "turn2": {"status": resp_a2["status"], "elapsed_s": resp_a2["elapsed"],
                     "content": ext_a2["content"], "error": resp_a2.get("error") or ext_a2.get("error"),
                     "finish_reason": ext_a2["finish_reason"], "usage": ext_a2["usage"]},
           "recall_teal": recalled,
           "restore_evidence_ssh": ssh_log,
           "note": ("TBD-VERIFY-ON-RUN: session keying/restore depends on engine "
                    "session-match semantics; confirm turn2 elapsed << cold prefill "
                    "and (if --ssh-park-check) a park/restore log line.")}
    save_json(path, rec)
    print("[park] turn2 recall_teal={}  elapsed={}s".format(
        recalled, int(resp_a2["elapsed"])))
    return rec


def _ssh_park_log():
    # Fixed, constant command (no user input interpolated) -> shell=True is safe
    # here; it simply reproduces the read-only grep pattern from
    # park_live_proof.sh. Kept as one string for readability.
    import subprocess
    cmd = ("ssh -o ConnectTimeout=8 -o BatchMode=yes macstudio-m4-1 "
           "'grep -a \"park\\|parked\\|turn reuse\\|session reuse\\|restored\" "
           "~/.exo/exo_log/exo.log 2>/dev/null | tail -14'")
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return {"rc": out.returncode, "stdout": out.stdout[-4000:], "stderr": out.stderr[-1000:]}
    except Exception as e:
        return {"rc": None, "error": str(e)}


# ==========================================================================
# Aggregation
# ==========================================================================
def aggregate(d):
    needles = load_json(os.path.join(d, "needles.json")) or {}
    tools = load_json(os.path.join(d, "tools.json")) or {}
    prose = load_json(os.path.join(d, "free_prose", "index.json")) or {}
    park = load_json(os.path.join(d, "park.json")) or {}
    build = load_json(os.path.join(d, "build.json")) or {}
    prose_probes = prose.get("probes") or []
    det_counts = Counter()
    for p in prose_probes:
        for k, v in ((p.get("detector") or {}).get("counts") or {}).items():
            det_counts[k] += v
    summary = {
        "label": os.path.basename(d),
        "generated": now_iso(),
        "build": {"status": build.get("http_status"), "est_tokens": (build.get("meta") or {}).get("est_tokens"),
                  "elapsed_s": build.get("elapsed_s")},
        "needles": {"n": needles.get("n"), "n_pass": needles.get("n_pass"),
                    "detail": [{"id": r["id"], "pass": r["pass"]} for r in needles.get("results", [])]},
        "tools": {"n": tools.get("n"), "n_pass": tools.get("n_pass")},
        "free_prose": {"n": prose.get("n"), "n_dirty": prose.get("n_dirty"),
                       "n_review": prose.get("n_review"),
                       "detector_hits": dict(det_counts),
                       "per_probe": [{"id": p["id"],
                                      "verdict": (p.get("detector") or {}).get("verdict"),
                                      "counts": (p.get("detector") or {}).get("counts")}
                                     for p in prose_probes]},
        "park": {"recall_teal": park.get("recall_teal"),
                 "turn2_elapsed_s": (park.get("turn2") or {}).get("elapsed_s")},
    }
    # headline verdict for this phase
    dirty = prose.get("n_dirty") or 0
    needle_miss = (needles.get("n", 0) - needles.get("n_pass", 0)) if needles else 0
    tool_miss = (tools.get("n", 0) - tools.get("n_pass", 0)) if tools else 0
    park_ok = park.get("recall_teal")
    if dirty or needle_miss or tool_miss or (park and not park_ok):
        summary["phase_verdict"] = "DIRTY"
    else:
        summary["phase_verdict"] = "CLEAN"
    save_json(os.path.join(d, "summary.json"), summary)
    return summary


# ==========================================================================
# Self-test
# ==========================================================================
SELFTEST_CASES = [
    ("clean_english",
     "The quick brown fox jumps over the lazy dog. It was a bright cold day in "
     "April, and the clocks were striking thirteen. The river ran quietly past "
     "the old mill.",
     "CLEAN", []),
    ("glued_fragments",
     "angleсь camerauden pavement他身上",
     "DIRTY", ["cross_script_glue"]),
    ("replacement_char",
     "The result is \ufffd\ufffd unknown and the reading was \ufffd.",
     "DIRTY", ["replacement_char"]),
    ("repetition_loop",
     ("go left go left go left go left go left and then go left go left "
      "go left go left again"),
     "DIRTY", ["repetition"]),
    ("codeswitch_punctuated_clean",
     "\u96e8\u58f0\u6dc5\u6ca5\u3002The rain whispers softly on the roof. "
     "\u3053\u308c\u3067\u7d42\u308f\u308a\u3002A calm end.",
     "CLEAN", []),
]


def run_selftest():
    print("=" * 72)
    print("battery.py --selftest  (detector suite vs synthetic strings)")
    print("=" * 72)
    all_ok = True
    for name, text, expect_verdict, expect_types in SELFTEST_CASES:
        det = run_detectors(text)
        types = set(det["counts"].keys())
        verdict_ok = det["verdict"] == expect_verdict
        types_ok = all(t in types for t in expect_types)
        ok = verdict_ok and types_ok
        all_ok = all_ok and ok
        print("\n[{}] {}  expected_verdict={} got={}  types_present={}".format(
            "PASS" if ok else "FAIL", name, expect_verdict, det["verdict"],
            sorted(types)))
        print("    sample: {!r}".format(text if len(text) < 70 else text[:67] + "..."))
        for h in det["hits"]:
            print("    hit: type={} conf={} token={!r}".format(
                h["type"], h.get("confidence"), h.get("token", h.get("ngram", ""))))
        if not verdict_ok:
            print("    !! verdict mismatch")
        if not types_ok:
            print("    !! missing expected hit types: {}".format(expect_types))
    # extra: the low-confidence same-script heuristic must catch camerauden
    glued = run_detectors("angleсь camerauden pavement他身上")
    ss = [h for h in glued["hits"] if h["type"] == "same_script_glue"]
    print("\n[{}] same_script heuristic caught camerauden: {}".format(
        "PASS" if ss else "REVIEW-ONLY", [h["token"] for h in ss]))
    print("\n" + "=" * 72)
    print("SELFTEST RESULT:", "ALL PASS" if all_ok else "FAILURES PRESENT")
    print("=" * 72)
    return 0 if all_ok else 1


# ==========================================================================
# plan (offline)
# ==========================================================================
def run_plan(args):
    print("OFFLINE PLAN (no network)")
    prompt, meta = build_deep_prompt(args.depth)
    print("  deep build: target={} chars={} est_tokens={}".format(
        args.depth, meta["chars"], meta["est_tokens"]))
    print("  needles planted:", json.dumps(meta["needles"], indent=2))
    print("  needle probes: {}".format([p["id"] for p in NEEDLE_PROBES]))
    print("  free prose prompts: {} ({})".format(
        len(FREE_PROSE), sorted({p["tag"] for p in FREE_PROSE})))
    for p in FREE_PROSE:
        print("    - {:<18} [{}] {}".format(p["id"], p["tag"], p["p"][:60]))
    print("  tool probes: {} ({} tools)".format(len(TOOL_PROBES), len(TOOLS)))
    small = build_small_prompt(args.park_tokens, memory=PARK_MEMORY)
    print("  park context: ~{} chars  est_tokens={}".format(
        len(small), int(len(small) / CHARS_PER_TOKEN)))
    return 0


# ==========================================================================
# main
# ==========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="DSv4.1 live quality battery")
    ap.add_argument("mode", nargs="?",
                    choices=["plan", "build", "eval", "all", "selftest", "aggregate",
                             "prose", "tools", "needles", "park"])
    ap.add_argument("--selftest", action="store_true",
                    help="run the detector suite against synthetic strings (no cluster)")
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--label", default="phase",
                    help="results/<label>/ subdir for this phase")
    ap.add_argument("--depth", type=int, default=DEFAULT_DEPTH,
                    help="deep-context target tokens (default {}; quick={})".format(
                        DEFAULT_DEPTH, QUICK_DEPTH))
    ap.add_argument("--prose-tokens", type=int, default=600)
    ap.add_argument("--park-tokens", type=int, default=8000)
    ap.add_argument("--timeout", type=float, default=5400.0,
                    help="per-request timeout seconds (deep prefill is slow)")
    ap.add_argument("--force", action="store_true", help="ignore cached results")
    ap.add_argument("--skip-park", action="store_true")
    ap.add_argument("--ssh-park-check", action="store_true",
                    help="grep the engine log for a park/restore line (read-only ssh)")
    args = ap.parse_args(argv)

    if args.selftest or args.mode == "selftest":
        return run_selftest()
    if args.mode is None:
        args.mode = "plan"
    if args.mode == "plan":
        return run_plan(args)

    if args.mode == "aggregate":
        d = label_dir(args.label)
        s = aggregate(d)
        print(json.dumps(s, indent=2, ensure_ascii=False))
        return 0

    d = label_dir(args.label)
    print("results dir:", d)
    meta = {"label": args.label, "api": args.api, "model": args.model,
            "depth": args.depth, "started": now_iso(),
            "env_hint": "DSV41_INDEXER_ROW_BF16 is read at import; phase control is "
                        "via the LAUNCHED process env (see runbook.md)"}
    save_json(os.path.join(d, "meta.json"), meta)

    client = Client(args.api, args.model, args.timeout)

    if args.mode in ("build", "all"):
        do_build(client, args, d)
    if args.mode == "needles":
        big_prompt = load_build_prompt(d)
        do_needles(client, args, d, big_prompt)
        aggregate(d)
        return 0
    if args.mode == "prose":
        do_free_prose(client, args, d)
        agg = aggregate(d)
        print("\nPHASE VERDICT: {}  (prose DIRTY {}, REVIEW {} of {})".format(
            agg["phase_verdict"], agg["free_prose"]["n_dirty"] or 0,
            agg["free_prose"]["n_review"] or 0, agg["free_prose"]["n"] or 0))
        return 0
    if args.mode == "tools":
        do_tools(client, args, d)
        aggregate(d)
        return 0
    if args.mode == "park":
        do_park(client, args, d)
        return 0
    if args.mode in ("eval", "all"):
        big_prompt = load_build_prompt(d)
        do_needles(client, args, d, big_prompt)
        do_free_prose(client, args, d)
        do_tools(client, args, d)
        if not args.skip_park:
            do_park(client, args, d)
        agg = aggregate(d)
        print("\nPHASE VERDICT: {}  (needles {}/{}, tools {}/{}, prose DIRTY {}, REVIEW {})".format(
            agg["phase_verdict"],
            agg["needles"]["n_pass"], agg["needles"]["n"],
            agg["tools"]["n_pass"], agg["tools"]["n"],
            agg["free_prose"]["n_dirty"] or 0, agg["free_prose"]["n_review"] or 0))
    meta["finished"] = now_iso()
    save_json(os.path.join(d, "meta.json"), meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
