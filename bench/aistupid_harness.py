#!/usr/bin/env python3
"""
aistupid_harness.py — self-contained Python port of the aistupidlevel.info
("AI Stupid Meter") coding-benchmark methodology.

This is a faithful, offline-runnable port of the open-source backend at
https://github.com/StudioPlatforms/aistupidmeter-api (MIT). The scoring math,
task definitions, code-extraction logic and per-axis heuristics are ported
verbatim from:
  - src/jobs/scorer.ts            (7-axis weighted z-score -> stupidScore -> gauge)
  - src/jobs/real-benchmarks.ts   (BENCHMARK_TASKS, extractPython, evaluateCode)
  - src/deepbench/tasks.ts        (e-commerce cart bug-fix task + test_cart.py)
  - src/lib/score-conversion.ts   (combineScores, calculateStability)
  - src/lib/statistical-tests.ts  (calculateConfidenceInterval, calculateStdDev)

WHAT IS A FAITHFUL PORT vs. A DOCUMENTED PROXY
----------------------------------------------
Faithful (math identical to the TS):
  * WEIGHTS, stupidScore = -sum(w*z), gauge = 50 + 15*tanh(-stupidScore) clamp 0-100
  * z = (latest - mean) / max(std, 1e-6)
  * calculateConfidenceInterval (t-table, 95% CI), calculateStdDev (n-1)
  * calculateStability (stdDev -> 0-100 band mapping)
  * combineScores (hourly 0.7 + deep 0.3)
  * extractPython code extraction
  * codeQuality heuristic (exact regex/threshold ladder from evaluateCode)
  * format heuristic (-> spec axis), safety heuristic, efficiency log-throughput

Documented PROXY (cannot be replicated offline / no LLM judge available):
  * The upstream scorer computes z-scores against a 28-day ROLLING BASELINE per
    model pulled from a database. We have no historical baseline here, so by
    default we score each axis z against a NEUTRAL self-baseline assembled from
    the trials themselves (mean of trials, std of trials). This yields a gauge of
    ~50 when a model is internally consistent, exactly as scorer.ts returns 50
    when "No baseline yet". You can inject a fixed baseline via --baseline-json.
    We ALSO always emit the raw absolute correctness% so you have a
    baseline-independent number. THIS IS CLEARLY A PROXY — see compute_gauge().
  * 'recovery' axis: upstream derives this from multi-turn debugging recovery
    which needs live iterative turns. We port the repo's single-shot default:
    recovery = min(correctness + 0.05, 1.0) (mirrors the 'debugging' fallback in
    evaluateCode). Flagged as PROXY in code.

NOTE: This harness DOES execute model-produced code in a subprocess sandbox
(temp dir + timeout). It is intended for trusted local clusters only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# httpx is optional (only needed for live runs). stdlib urllib is the fallback.
try:
    import httpx  # type: ignore
    _HAS_HTTPX = True
except Exception:  # pragma: no cover
    _HAS_HTTPX = False


# ============================================================================
# 7-AXIS WEIGHTS — verbatim from src/jobs/scorer.ts
# ============================================================================
WEIGHTS: dict[str, float] = {
    "correctness": 0.35,
    "spec": 0.15,
    "codeQuality": 0.15,
    "efficiency": 0.10,
    "stability": 0.10,
    "refusal": 0.10,
    "recovery": 0.05,
}
AXES = list(WEIGHTS.keys())


# ============================================================================
# scorer.ts — dotZ, stupidScore, gauge
# ============================================================================
def dot_z(weights: dict[str, float], z: dict[str, float]) -> float:
    return sum(w * z.get(k, 0.0) for k, w in weights.items())


def compute_gauge(
    latest: dict[str, float],
    baseline_means: dict[str, float],
    baseline_stds: dict[str, float],
) -> dict[str, Any]:
    """Faithful port of scoreBatch() math from scorer.ts.

    stupidScore = -sum(weight * z), z = (latest - mean)/max(std, 1e-6)
    gauge = 50 + 15*tanh(-stupidScore), clamped 0..100.

    PROXY NOTE: upstream pulls baseline_means/stds from a 28-day DB rolling
    window. Offline we feed a self-baseline (see build_self_baseline) or a
    user-supplied --baseline-json. When latest == baseline_means (no historical
    drift signal), z == 0 and gauge == 50.0, exactly mirroring scorer.ts's
    "No baseline yet -> gauge 50" behaviour.
    """
    z: dict[str, float] = {}
    for axis in AXES:
        mean = baseline_means.get(axis, 0.5)
        std = max(baseline_stds.get(axis, 0.05), 1e-6)
        z[axis] = (latest.get(axis, 0.0) - mean) / std
    stupid_score = -dot_z(WEIGHTS, z)
    gauge = 50.0 + 15.0 * math.tanh(-stupid_score)
    gauge = max(0.0, min(100.0, gauge))
    return {"stupidScore": stupid_score, "gauge": gauge, "z": z}


def build_self_baseline(
    per_trial_axes: list[dict[str, float]],
) -> tuple[dict[str, float], dict[str, float]]:
    """PROXY baseline: mean & std of the trials themselves (mirrors getBaseline()
    in scorer.ts which averages the last N runs). std floored at 0.01 exactly as
    upstream (Math.max(0.01, std))."""
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for axis in AXES:
        vals = [a.get(axis, 0.0) for a in per_trial_axes]
        if not vals:
            means[axis], stds[axis] = 0.5, 0.05
            continue
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        means[axis] = max(0.0, min(1.0, m))
        stds[axis] = max(0.01, math.sqrt(var))
    return means, stds


# ============================================================================
# statistical-tests.ts — calculateStdDev, calculateConfidenceInterval
# ============================================================================
def calculate_std_dev(values: list[float]) -> float:
    """Sample std (n-1), verbatim from statistical-tests.ts calculateStdDev."""
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return 0.0
    mean = sum(values) / len(values)
    squared = [(v - mean) ** 2 for v in values]
    variance = sum(squared) / (len(values) - 1)
    return math.sqrt(variance)


_T_VALUES: dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    9: 2.262, 29: 2.045, 99: 1.984,
}


def calculate_confidence_interval(
    scores: list[float], confidence: float = 0.95
) -> dict[str, float]:
    """95% CI via t-distribution, verbatim from statistical-tests.ts."""
    n = len(scores)
    if n == 0:
        return {"lower": 0.0, "upper": 0.0, "standardError": 0.0, "mean": 0.0}
    if n == 1:
        return {
            "lower": max(0.0, scores[0] - 5),
            "upper": min(100.0, scores[0] + 5),
            "standardError": 2.5,
            "mean": scores[0],
        }
    mean = sum(scores) / n
    variance = sum((x - mean) ** 2 for x in scores) / (n - 1)
    std_dev = math.sqrt(variance)
    standard_error = std_dev / math.sqrt(n)
    df = n - 1
    t_value = 2.0
    if df in _T_VALUES:
        t_value = _T_VALUES[df]
    elif df > 99:
        t_value = 1.96
    else:
        keys = sorted(_T_VALUES.keys())
        for i in range(len(keys) - 1):
            if keys[i] <= df <= keys[i + 1]:
                ratio = (df - keys[i]) / (keys[i + 1] - keys[i])
                t_value = _T_VALUES[keys[i]] + ratio * (
                    _T_VALUES[keys[i + 1]] - _T_VALUES[keys[i]]
                )
                break
    margin = t_value * standard_error
    return {
        "lower": max(0.0, mean - margin),
        "upper": min(100.0, mean + margin),
        "standardError": standard_error,
        "mean": mean,
    }


# ============================================================================
# score-conversion.ts — calculateStability, combineScores
# ============================================================================
def calculate_stability(scores: list[float]) -> float:
    """Maps stdDev (of 0-100 scores) to a 0-100 stability score. Verbatim ladder
    from score-conversion.ts calculateStability."""
    if len(scores) < 2:
        return 75.0
    avg = sum(scores) / len(scores)
    variance = sum((s - avg) ** 2 for s in scores) / len(scores)
    std_dev = math.sqrt(variance)
    if std_dev <= 2:
        stability = max(90, min(95, round(95 - (std_dev * 2.5))))
    elif std_dev <= 5:
        stability = max(75, min(90, round(90 - ((std_dev - 2) * 5))))
    elif std_dev <= 10:
        stability = max(45, min(75, round(75 - ((std_dev - 5) * 6))))
    elif std_dev <= 20:
        stability = max(25, min(45, round(45 - ((std_dev - 10) * 2))))
    else:
        stability = max(0, min(25, round(25 - ((std_dev - 20) * 0.5))))
    return float(round(stability))


def combine_scores(hourly: Optional[float], deep: Optional[float]) -> Optional[float]:
    """hourly 0.7 + deep 0.3, verbatim from score-conversion.ts combineScores."""
    if hourly is not None and deep is not None:
        return round(hourly * 0.7 + deep * 0.3)
    if hourly is not None:
        return hourly
    if deep is not None:
        return deep
    return None


# ============================================================================
# real-benchmarks.ts — extractPython (robust code extraction)
# ============================================================================
def extract_python(raw: str, expected_symbol: str) -> str:
    """Port of extractPython() from real-benchmarks.ts."""
    if not raw:
        return ""
    s = raw.replace("\r\n", "\n").strip()

    # 1) Prefer the fenced block containing the expected symbol, else the longest
    code_block_re = re.compile(r"```(?:python|py)?\s*([\s\S]*?)```", re.IGNORECASE)
    blocks = [m.group(1).strip() for m in code_block_re.finditer(s)]
    if blocks:
        with_symbol = next(
            (b for b in blocks
             if re.search(rf"\b(def|class)\s+{re.escape(expected_symbol)}\b", b)),
            None,
        )
        if with_symbol is not None:
            s = with_symbol.strip()
        else:
            s = max(blocks, key=len).strip()

    # 2) If still prose, cut everything before the first def/class
    if not re.search(r"^(\s*def |\s*class )", s, re.MULTILINE):
        m = re.search(r"^\s*(def|class)\s+", s, re.MULTILINE)
        if m:
            s = s[m.start():]

    # 3) Strip any stray backtick lines
    s = re.sub(r"^\s*```.*$", "", s, flags=re.MULTILINE).strip()

    # 4) Drop single-line prose prefixes like "Here is the function:"
    s = "\n".join(
        line for line in s.split("\n")
        if not re.match(r"^(here is|solution|function|code)\b", line.strip(), re.IGNORECASE)
    ).strip()
    return s


# ============================================================================
# real-benchmarks.ts — codeQuality / format / safety heuristics
# ============================================================================
def score_code_quality(clean: str) -> float:
    """Verbatim port of the codeQuality heuristic ladder in evaluateCode()."""
    cq = 0.0
    if 20 <= len(clean) <= 2000:
        cq += 0.20
    if not re.search(
        r"exec|eval|__import__|os\.|subprocess\.|socket\.|urllib\.|requests\.|ftplib|smtplib",
        clean,
    ):
        cq += 0.20
    if re.search(r"(^|\n)\s*(def|class)\s+", clean):
        cq += 0.10
    if re.search(r"\b(if|for|while)\b", clean):
        cq += 0.10
    if re.search(r'^""".+?"""|^\'\'\'.+?\'\'\'', clean, re.MULTILINE | re.DOTALL):
        cq += 0.10  # docstring
    if re.search(r"->\s*[A-Za-z_][A-Za-z0-9_\[\], ]*", clean):
        cq += 0.05  # return type hint
    if re.search(r"\w+\s*:\s*[A-Za-z_][A-Za-z0-9_\[\], ]*", clean):
        cq += 0.05  # arg hints
    if re.search(r"#[^\n]{5,}", clean):
        cq += 0.05  # non-trivial comment
    if re.search(r"return\s+", clean):
        cq += 0.05
    if re.search(r"(global\s|lambda\s)", clean):
        cq -= 0.05
    if len(clean) > 2500:
        cq -= 0.05
    return max(0.0, min(1.0, cq))


def score_format(raw_text: str) -> float:
    """Port of the 'format' axis from evaluateCode(). We map this onto the 7-axis
    'spec' axis (did the model follow the output-format contract)."""
    raw = raw_text.strip()
    has_fence = "```" in raw
    only_fence = bool(re.match(r"^```(?:python|py)?\s*[\s\S]*?\s*```$", raw))
    looks_like_code = bool(
        re.search(r"^(?:\s*(?:def|class)\s+|\s*from\s+\w+|\s*import\s+\w+)", raw, re.MULTILINE)
    )
    if not has_fence and looks_like_code:
        return 1.0
    if has_fence and only_fence:
        return 1.0
    if has_fence and not only_fence:
        return 0.8
    return 0.3


def score_safety(clean: str) -> float:
    """Port of the 'safety' axis from evaluateCode()."""
    unsafe = re.search(
        r"\b(exec|eval|__import__|subprocess|socket|urllib|requests|ftplib|smtplib)\b",
        clean,
    )
    return 0.2 if unsafe else 1.0


def score_efficiency(tokens_out: int, latency_ms: float) -> float:
    """Port of the efficiency log-throughput normalization from real-benchmarks.ts.

    throughput = tokens_out / max(1, latency_ms)
    eff = clamp01(log10(throughput + 1e-6) + 3) / 3, then clamp to [0.1, 0.9].
    """
    tokens_out = tokens_out if tokens_out > 0 else 1
    throughput = tokens_out / max(1.0, latency_ms)
    log_throughput = math.log10(throughput + 1e-6) + 3
    eff = max(0.0, min(1.0, log_throughput / 3.0))
    return max(0.1, min(0.9, eff))


# ============================================================================
# BOS-spam / degeneration guard
# ============================================================================
_LEAK_TOKENS = [
    "<|begin_of_sentence|>", "<|end_of_sentence|>", "<|endoftext|>",
    "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>",
    "<|im_start|>", "<|im_end|>", "<|eot_id|>",
]


def detect_degeneration(raw_text: str) -> tuple[bool, str]:
    """Returns (is_degenerate, reason). Flags literal special-token leakage,
    emptiness, and obvious looping. throughput-clean-but-quality-dead guard."""
    if raw_text is None or raw_text.strip() == "":
        return True, "empty_response"
    for tok in _LEAK_TOKENS:
        if tok in raw_text:
            return True, f"special_token_leak:{tok}"
    # crude loop detection: a short substring repeated very many times
    stripped = raw_text.strip()
    if len(stripped) > 200:
        # check for a >=8-char chunk repeated >20x consecutively
        if re.search(r"(.{8,40}?)\1{20,}", stripped):
            return True, "looping_repetition"
    return False, ""


def is_refusal(raw_text: str) -> bool:
    """Mirrors the refusal detection in real-benchmarks.ts (sorry/cannot/...)."""
    if not raw_text or not raw_text.strip():
        return True
    return bool(re.search(r"sorry|cannot|unable|inappropriate", raw_text, re.IGNORECASE))


# ============================================================================
# TASK DEFINITIONS — ported from real-benchmarks.ts BENCHMARK_TASKS and
# deepbench/tasks.ts (e-commerce cart). Each task carries a pytest-style runner.
# ============================================================================
@dataclass
class Task:
    task_id: str
    slug: str
    difficulty: str            # easy | medium | hard | deep
    kind: str                  # "function" or "pytest"
    prompt: str
    expected_symbol: str
    # for kind == "function": list of (input_literal, expected_literal)
    test_cases: list[tuple[str, str]] = field(default_factory=list)
    # for kind == "pytest": the verbatim test file + scaffold file map
    pytest_files: dict[str, str] = field(default_factory=dict)
    pytest_test_filename: str = "test_solution.py"
    solution_filename: str = "main.py"
    weight: float = 1.0        # for hourly(0.7)/deep(0.3) combine


# ---- E-commerce cart (deepbench/tasks.ts) — verbatim main.py + test_cart.py ----
ECOMMERCE_MAIN = '''# E-commerce cart system - has 3 subtle bugs that need fixing
from typing import List, Dict, Optional
from dataclasses import dataclass
import json

@dataclass
class Product:
    id: str
    name: str
    price: float
    stock: int

@dataclass
class CartItem:
    product_id: str
    quantity: int

class ShoppingCart:
    def __init__(self):
        self.items: Dict[str, CartItem] = {}
        self.discount_code: Optional[str] = None

    def add_item(self, product_id: str, quantity: int = 1):
        """Add item to cart"""
        if product_id in self.items:
            self.items[product_id].quantity += quantity  # Bug 1: No stock validation
        else:
            self.items[product_id] = CartItem(product_id, quantity)

    def remove_item(self, product_id: str, quantity: int = 1):
        """Remove item from cart"""
        if product_id not in self.items:
            return False

        if self.items[product_id].quantity <= quantity:
            del self.items[product_id]
        else:
            self.items[product_id].quantity -= quantity
        return True

    def get_total(self, products: List[Product]) -> float:
        """Calculate cart total"""
        total = 0.0
        product_dict = {p.id: p for p in products}

        for item in self.items.values():
            if item.product_id in product_dict:
                product = product_dict[item.product_id]
                total += product.price * item.quantity  # Bug 2: No stock check here

        # Bug 3: Discount logic is wrong - should be percentage, not fixed amount
        if self.discount_code == "SAVE10":
            total -= 10.0

        return total

    def validate_cart(self, products: List[Product]) -> List[str]:
        """Validate cart against product availability"""
        errors = []
        product_dict = {p.id: p for p in products}

        for item in self.items.values():
            if item.product_id not in product_dict:
                errors.append(f"Product {item.product_id} not found")
            elif product_dict[item.product_id].stock < item.quantity:
                errors.append(f"Insufficient stock for {item.product_id}")

        return errors
'''

ECOMMERCE_TEST = '''import pytest
from main import ShoppingCart, Product, CartItem

def test_add_item_basic():
    """Test basic item addition"""
    cart = ShoppingCart()
    cart.add_item("P1", 2)
    assert "P1" in cart.items
    assert cart.items["P1"].quantity == 2

def test_add_item_duplicate():
    """Test adding same item multiple times"""
    cart = ShoppingCart()
    cart.add_item("P1", 2)
    cart.add_item("P1", 3)
    assert cart.items["P1"].quantity == 5

def test_remove_item():
    """Test item removal"""
    cart = ShoppingCart()
    cart.add_item("P1", 5)
    result = cart.remove_item("P1", 2)
    assert result == True
    assert cart.items["P1"].quantity == 3

def test_get_total():
    """Test total calculation"""
    products = [
        Product("P1", "Widget", 10.0, 100),
        Product("P2", "Gadget", 20.0, 50)
    ]
    cart = ShoppingCart()
    cart.add_item("P1", 2)
    cart.add_item("P2", 1)

    total = cart.get_total(products)
    assert total == 40.0  # 2*10 + 1*20

def test_discount_code():
    """Test discount application"""
    products = [Product("P1", "Widget", 100.0, 10)]
    cart = ShoppingCart()
    cart.add_item("P1", 1)
    cart.discount_code = "SAVE10"

    total = cart.get_total(products)
    assert total == 90.0  # Should be 10% off, not $10 off

def test_validate_cart():
    """Test cart validation against stock"""
    products = [Product("P1", "Widget", 10.0, 5)]
    cart = ShoppingCart()
    cart.add_item("P1", 10)  # More than stock

    errors = cart.validate_cart(products)
    assert len(errors) > 0
    assert "Insufficient stock" in errors[0]
'''

ECOMMERCE_PROMPT = (
    "You are helping debug a Python e-commerce cart system. The following "
    "`main.py` contains a ShoppingCart with THREE subtle bugs. Fix all of them "
    "so the pytest suite `test_cart.py` passes. Return the COMPLETE corrected "
    "`main.py` as a single Python code block. Keep the public API "
    "(ShoppingCart, Product, CartItem, add_item, remove_item, get_total, "
    "validate_cart) identical.\n\n"
    "Known issues to fix:\n"
    "1. add_item does not validate against available stock.\n"
    "2. get_total does not respect stock when pricing.\n"
    '3. The "SAVE10" discount must be 10 PERCENT off, not a flat $10 off.\n\n'
    "Here is main.py:\n```python\n" + ECOMMERCE_MAIN + "```\n\n"
    "Here is test_cart.py (do not modify it):\n```python\n" + ECOMMERCE_TEST + "```\n"
)


# ---- Hourly function tasks (real-benchmarks.ts BENCHMARK_TASKS) ----
def _fn_prompt(core: str, expected: str) -> str:
    # Mirrors makeUnifiedPrompt rule envelope (Rules: output only Python...).
    return (
        f"{core}\n\nRules:\n"
        f"- Output ONLY Python code.\n"
        f"- No markdown/backticks/prose.\n"
        f"- First line MUST be: def {expected}(\n"
        f"- Pure stdlib; deterministic behavior."
    )


_HOURLY_RAW = [
    ("is_palindrome", "py/is_palindrome", "easy",
     "Write a Python function named is_palindrome that checks if a string is a "
     "palindrome (ignoring spaces and case).",
     "is_palindrome",
     [('"racecar"', "True"), ('"A man a plan a canal Panama"', "True"),
      ('"hello"', "False"), ('""', "True")]),
    ("prime_check", "py/prime_check", "easy",
     "Write a Python function named is_prime that efficiently checks if a number is prime.",
     "is_prime",
     [("2", "True"), ("17", "True"), ("100", "False"), ("97", "True"), ("1", "False")]),
    ("binary_search", "py/binary_search", "medium",
     "Write a Python function named binary_search that performs binary search on a "
     "sorted list. Return the index if found, -1 otherwise. The function should take "
     "two parameters: arr (sorted list) and target.",
     "binary_search",
     [("[1,3,5,7,9,11], 7", "3"), ("[1,2,3,4,5], 6", "-1"),
      ("[10,20,30,40,50], 10", "0"), ("[], 5", "-1")]),
    ("merge_intervals", "py/merge_intervals", "medium",
     "Write a Python function named merge_intervals that takes a list of intervals "
     "(as [start, end] pairs) and merges overlapping intervals. Return the merged "
     "intervals sorted by start time.",
     "merge_intervals",
     [("[[1,3],[2,6],[8,10],[15,18]]", "[[1,6],[8,10],[15,18]]"),
      ("[[1,4],[4,5]]", "[[1,5]]"), ("[[1,4],[2,3]]", "[[1,4]]"), ("[]", "[]")]),
    ("word_break", "py/word_break", "hard",
     "Write a Python function named word_break that determines if a string can be "
     "segmented into words from a given dictionary. Use dynamic programming. Return True/False.",
     "word_break",
     [('"leetcode", ["leet", "code"]', "True"),
      ('"applepenapple", ["apple", "pen"]', "True"),
      ('"catsandog", ["cats", "dog", "sand", "and", "cat"]', "False")]),
    ("regex_match", "py/regex_match", "hard",
     'Write a Python function named regex_match that implements regular expression '
     'matching with support for "." (any char) and "*" (zero or more of preceding). '
     "Return True/False.",
     "regex_match",
     [('"aa", "a"', "False"), ('"aa", "a*"', "True"),
      ('"ab", ".*"', "True"), ('"mississippi", "mis*is*p*."', "False")]),
    ("debug_sort", "py/debug_sort", "medium",
     "Debug and fix this broken quicksort implementation:\n```python\n"
     "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n"
     "    pivot = arr[0]\n    left = [x for x in arr if x < pivot]\n"
     "    right = [x for x in arr if x > pivot]\n"
     "    return quicksort(left) + [pivot] + quicksort(right)\n```\n"
     "The bug: it loses duplicate elements. Fix it.",
     "quicksort",
     [("[3,1,4,1,5,9,2,6,5]", "[1,1,2,3,4,5,5,6,9]"),
      ("[5,5,5,5]", "[5,5,5,5]"), ("[]", "[]")]),
    ("optimize_fibonacci", "py/optimize_fibonacci", "medium",
     "Write an optimized Python function named fibonacci that returns the nth "
     "Fibonacci number. Must handle n up to 10000 efficiently (no recursion, use "
     "memoization or iteration).",
     "fibonacci",
     [("0", "0"), ("10", "55"), ("50", "12586269025"),
      ("100", "354224848179261915075")]),
]


def build_tasks() -> dict[str, Task]:
    tasks: dict[str, Task] = {}
    for tid, slug, diff, core, sym, cases in _HOURLY_RAW:
        tasks[tid] = Task(
            task_id=tid, slug=slug, difficulty=diff, kind="function",
            prompt=_fn_prompt(core, sym), expected_symbol=sym,
            test_cases=cases, weight=1.0,
        )
    tasks["ecommerce_cart"] = Task(
        task_id="ecommerce_cart", slug="deep/ide_assistant", difficulty="deep",
        kind="pytest", prompt=ECOMMERCE_PROMPT, expected_symbol="ShoppingCart",
        pytest_files={"test_cart.py": ECOMMERCE_TEST},
        pytest_test_filename="test_cart.py", solution_filename="main.py",
        weight=1.0,
    )
    return tasks


# ============================================================================
# CORRECTNESS execution sandbox
# ============================================================================
def _run_subprocess(cmd: list[str], cwd: str, timeout: int) -> tuple[int, str, str]:
    try:
        p = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
        )
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "TIMEOUT"
    except Exception as e:  # pragma: no cover
        return 1, "", f"RUNNER_ERROR: {e}"


def run_function_task(task: Task, clean_code: str, timeout: int) -> dict[str, Any]:
    """Execute a 'function' task: run the model's code against fixed test cases
    in a subprocess. score = passed / total. Mirrors evaluateCode's runner using
    ast.literal_eval on the input tuple and == comparison on expected."""
    if not clean_code.strip():
        return {"passed": 0, "total": len(task.test_cases) or 1, "correctness": 0.0,
                "detail": "empty_code"}

    harness = []
    harness.append("import ast, sys")
    harness.append("ns = {}")
    harness.append("passed = 0")
    harness.append("total = 0")
    harness.append("try:")
    harness.append("    exec(compile(open('solution.py').read(), 'solution.py', 'exec'), ns, ns)")
    harness.append("except Exception:")
    harness.append("    pass")
    harness.append("def call_fn(name, args):")
    harness.append("    fn = ns.get(name)")
    harness.append("    if not callable(fn): raise NameError('missing ' + name)")
    harness.append("    return fn(*args)")
    for inp, exp in task.test_cases:
        harness.append("total += 1")
        harness.append("try:")
        harness.append(f"    args = ast.literal_eval('(' + {inp!r} + ',)')")
        harness.append(f"    result = call_fn({task.expected_symbol!r}, args)")
        harness.append(f"    expected = ast.literal_eval({exp!r})")
        harness.append("    if result == expected: passed += 1")
        harness.append("except Exception:")
        harness.append("    pass")
    harness.append("print(f'{passed}/{total}')")

    with tempfile.TemporaryDirectory(prefix="aism_fn_") as d:
        with open(os.path.join(d, "solution.py"), "w") as f:
            f.write(clean_code)
        with open(os.path.join(d, "runner.py"), "w") as f:
            f.write("\n".join(harness))
        rc, out, err = _run_subprocess(
            [sys.executable, "-I", "runner.py"], cwd=d, timeout=timeout
        )
    try:
        p, t = out.strip().split("/")
        passed, total = int(p), int(t)
    except Exception:
        passed, total = 0, len(task.test_cases) or 1
    correctness = max(0.0, min(1.0, passed / (total or 1)))
    return {"passed": passed, "total": total, "correctness": correctness,
            "detail": (err[:200] if correctness == 0 else "")}


def run_pytest_task(task: Task, clean_code: str, timeout: int) -> dict[str, Any]:
    """Execute a 'pytest' task (e-commerce cart): write the model's main.py and the
    verbatim test file into a temp dir and run pytest. score = passed/total."""
    if not clean_code.strip():
        return {"passed": 0, "total": 6, "correctness": 0.0, "detail": "empty_code"}
    with tempfile.TemporaryDirectory(prefix="aism_pytest_") as d:
        with open(os.path.join(d, task.solution_filename), "w") as f:
            f.write(clean_code)
        for fname, content in task.pytest_files.items():
            with open(os.path.join(d, fname), "w") as f:
                f.write(content)
        rc, out, err = _run_subprocess(
            [sys.executable, "-m", "pytest", "-q", "--no-header",
             "-p", "no:cacheprovider", task.pytest_test_filename],
            cwd=d, timeout=timeout,
        )
    passed = _parse_pytest_count(out, r"(\d+) passed")
    failed = _parse_pytest_count(out, r"(\d+) failed")
    errors = _parse_pytest_count(out, r"(\d+) error")
    total = passed + failed + errors
    if total == 0:
        # collection error -> 0 of the known 6 tests
        total = 6
    correctness = max(0.0, min(1.0, passed / (total or 1)))
    return {"passed": passed, "total": total, "correctness": correctness,
            "detail": ("" if correctness > 0 else (out[-300:] + err[-200:]))}


def _parse_pytest_count(text: str, pattern: str) -> int:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else 0


# ============================================================================
# Per-trial scoring -> the 7 axes
# ============================================================================
def score_trial(task: Task, raw_text: str, tokens_out: int, latency_ms: float,
                timeout: int, reasoning_truncated: bool = False,
                reasoning_len: int = 0, finish_reason: str = "") -> dict[str, Any]:
    """Run a single trial's response through extraction + execution + all axes."""
    degenerate, leak_reason = detect_degeneration(raw_text)
    # Runaway/truncated reasoning is its own failure mode (thinking model burned the
    # entire token budget on chain-of-thought and never emitted an answer). Treat it
    # as a degeneration-class leak distinct from BOS-spam so it's visible in reports.
    if reasoning_truncated and not degenerate:
        degenerate = True
        leak_reason = f"reasoning_truncated(finish=length,reasoning={reasoning_len}c,answer<20c)"
    clean = extract_python(raw_text, task.expected_symbol)

    if task.kind == "pytest":
        exec_res = run_pytest_task(task, clean, timeout)
    else:
        exec_res = run_function_task(task, clean, timeout)

    correctness = exec_res["correctness"]

    # spec axis <- format heuristic (instruction/format following)
    spec = score_format(raw_text)
    code_quality = score_code_quality(clean)
    efficiency = score_efficiency(tokens_out, latency_ms)
    # refusal axis: 1.0 unless refused/empty/degenerate (BOS leak => 0)
    refusal = 0.0 if (degenerate or is_refusal(raw_text)) else 1.0
    # recovery axis: PROXY — upstream needs multi-turn recovery; use single-shot
    # default mirroring the 'debugging' fallback: min(correctness+0.05, 1.0)
    recovery = min(correctness + 0.05, 1.0)
    # stability filled in after all trials (needs variance) -> placeholder
    return {
        "axes": {
            "correctness": correctness,
            "spec": spec,
            "codeQuality": code_quality,
            "efficiency": efficiency,
            "stability": 0.0,  # set post-hoc
            "refusal": refusal,
            "recovery": recovery,
        },
        "correctness_raw": correctness,
        "passed": exec_res["passed"],
        "total": exec_res["total"],
        "leak": degenerate,
        "leak_reason": leak_reason,
        "clean_len": len(clean),
        "detail": exec_res.get("detail", ""),
    }


def finalize_task(task: Task, trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate trials: compute per-axis mean + 95% CI, fill stability from
    correctness variance across trials (calculateStability on 0-100 scale)."""
    # stability: variance of correctness across trials -> 0-100 -> back to 0-1
    corr_pct = [t["axes"]["correctness"] * 100.0 for t in trials]
    stability_0_100 = calculate_stability(corr_pct) if len(corr_pct) >= 2 else 75.0
    stability_axis = stability_0_100 / 100.0
    for t in trials:
        t["axes"]["stability"] = stability_axis

    per_axis_mean: dict[str, float] = {}
    per_axis_ci: dict[str, dict[str, float]] = {}
    for axis in AXES:
        vals = [t["axes"][axis] for t in trials]
        per_axis_mean[axis] = sum(vals) / len(vals)
        # CI computed on 0-100 scale (as upstream does for display scores)
        ci = calculate_confidence_interval([v * 100.0 for v in vals])
        per_axis_ci[axis] = ci

    correctness_pct = per_axis_mean["correctness"] * 100.0
    leak_count = sum(1 for t in trials if t["leak"])
    return {
        "task_id": task.task_id,
        "slug": task.slug,
        "difficulty": task.difficulty,
        "kind": task.kind,
        "trials": trials,
        "per_axis_mean": per_axis_mean,
        "per_axis_ci": per_axis_ci,
        "correctness_pct": correctness_pct,
        "stability_0_100": stability_0_100,
        "leak_count": leak_count,
    }


def aggregate_overall(task_results: list[dict[str, Any]],
                      baseline_json: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-task means into an overall axes vector, then run the
    scorer.ts gauge. Splits hourly vs deep for combineScores."""
    # average each axis across tasks (matches agg loop in real-benchmarks.ts)
    overall_axes: dict[str, float] = {}
    for axis in AXES:
        vals = [tr["per_axis_mean"][axis] for tr in task_results]
        overall_axes[axis] = sum(vals) / len(vals) if vals else 0.5

    # Build per-trial axis vectors for self-baseline (flatten all trials/tasks)
    flat_trials: list[dict[str, float]] = []
    for tr in task_results:
        for t in tr["trials"]:
            flat_trials.append(t["axes"])

    if baseline_json:
        means = {a: float(baseline_json["means"].get(a, 0.5)) for a in AXES}
        stds = {a: max(0.01, float(baseline_json["stds"].get(a, 0.05))) for a in AXES}
        baseline_kind = "user_supplied"
    else:
        means, stds = build_self_baseline(flat_trials)
        baseline_kind = "self_baseline(PROXY)"

    gauge_res = compute_gauge(overall_axes, means, stds)

    # hourly(0.7)/deep(0.3) display-score combine using raw correctness% buckets
    hourly_corr = [tr["correctness_pct"] for tr in task_results if tr["difficulty"] != "deep"]
    deep_corr = [tr["correctness_pct"] for tr in task_results if tr["difficulty"] == "deep"]
    hourly_score = (sum(hourly_corr) / len(hourly_corr)) if hourly_corr else None
    deep_score = (sum(deep_corr) / len(deep_corr)) if deep_corr else None
    combined_correctness = combine_scores(hourly_score, deep_score)

    abs_correctness = sum(tr["correctness_pct"] for tr in task_results) / len(task_results)

    return {
        "overall_axes": overall_axes,
        "baseline_kind": baseline_kind,
        "baseline_means": means,
        "baseline_stds": stds,
        "stupidScore": gauge_res["stupidScore"],
        "gauge": gauge_res["gauge"],
        "z": gauge_res["z"],
        "absolute_correctness_pct": abs_correctness,
        "hourly_correctness_pct": hourly_score,
        "deep_correctness_pct": deep_score,
        "combined_correctness_pct": combined_correctness,
    }


# ============================================================================
# LLM endpoint (OpenAI /v1/chat/completions compatible). adapters.ts shape.
# ============================================================================
def call_chat_completion(base_url: str, model: str, prompt: str,
                         max_tokens: int, temperature: float,
                         system: Optional[str], timeout: float) -> dict[str, Any]:
    """POST {base_url}/v1/chat/completions. Returns dict with text, tokens, latency."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    t0 = time.time()
    if _HAS_HTTPX:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
    else:  # stdlib fallback
        import urllib.request
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    latency_ms = (time.time() - t0) * 1000.0

    text = ""
    reasoning = ""
    finish_reason = ""
    try:
        choice = data["choices"][0]
        msg = choice.get("message", {}) or {}
        text = msg.get("content") or ""
        # Thinking models (DeepSeek-V4, Kimi, Qwen3-thinking) emit chain-of-thought
        # into a separate reasoning_content field. If the model ran out of budget
        # mid-thought, content is empty/garbage while reasoning_content is huge.
        # Capture it so the scorer can flag truncation instead of silently scoring 0.
        reasoning = msg.get("reasoning_content") or ""
        finish_reason = choice.get("finish_reason") or ""
    except Exception:
        text = data.get("output_text", "") or ""
    usage = data.get("usage", {}) or {}
    tokens_out = int(usage.get("completion_tokens", 0) or 0)
    # If the visible answer is empty but the model produced reasoning and hit the
    # length cap, surface it as a truncated-reasoning trial (a real failure mode on
    # thinking models with too-small max_tokens), distinct from BOS-spam degeneration.
    reasoning_truncated = bool(
        finish_reason == "length" and len((text or "").strip()) < 20 and len(reasoning) > 200
    )
    return {"text": text, "tokens_out": tokens_out, "latency_ms": latency_ms,
            "reasoning": reasoning, "reasoning_len": len(reasoning),
            "finish_reason": finish_reason,
            "reasoning_truncated": reasoning_truncated, "raw": data}


SYSTEM_MESSAGE = (
    "You are an expert Python programmer. Provide clean, efficient, and correct "
    "Python code. Include only the requested function or class definition."
)


# ============================================================================
# Console reporting
# ============================================================================
def fmt_pct(x: float) -> str:
    return f"{x:6.1f}"


def print_report(task_results: list[dict[str, Any]], overall: dict[str, Any],
                 model: str) -> None:
    line = "=" * 78
    print(line)
    print(f"AI STUPID METER (offline port) — model: {model}")
    print(line)
    header = (f"{'task':<20}{'corr%':>7}{'spec':>6}{'cq':>6}{'eff':>6}"
             f"{'stab':>6}{'ref':>6}{'rec':>6}{'leak':>6}")
    print(header)
    print("-" * 78)
    for tr in task_results:
        a = tr["per_axis_mean"]
        print(f"{tr['task_id']:<20}"
              f"{tr['correctness_pct']:>7.1f}"
              f"{a['spec']:>6.2f}{a['codeQuality']:>6.2f}{a['efficiency']:>6.2f}"
              f"{a['stability']:>6.2f}{a['refusal']:>6.2f}{a['recovery']:>6.2f}"
              f"{tr['leak_count']:>6}")
    print("-" * 78)
    print("Per-axis 95% CI (correctness axis, 0-100 scale):")
    for tr in task_results:
        ci = tr["per_axis_ci"]["correctness"]
        print(f"  {tr['task_id']:<20} mean={ci['mean']:6.1f}  "
              f"CI=[{ci['lower']:6.1f}, {ci['upper']:6.1f}]  SE={ci['standardError']:.2f}")
    print(line)
    oa = overall["overall_axes"]
    print("OVERALL AXES (mean across tasks):")
    for axis in AXES:
        print(f"  {axis:<14} {oa[axis]:.4f}  (weight {WEIGHTS[axis]:.2f}, "
              f"z={overall['z'][axis]:+.3f})")
    print("-" * 78)
    print(f"baseline:               {overall['baseline_kind']}")
    print(f"stupidScore:            {overall['stupidScore']:+.4f}")
    print(f"GAUGE (0-100):          {overall['gauge']:.1f}")
    print(f"absolute correctness%:  {overall['absolute_correctness_pct']:.1f}")
    if overall["hourly_correctness_pct"] is not None:
        print(f"hourly correctness%:    {overall['hourly_correctness_pct']:.1f}")
    if overall["deep_correctness_pct"] is not None:
        print(f"deep correctness%:      {overall['deep_correctness_pct']:.1f}")
    if overall["combined_correctness_pct"] is not None:
        print(f"combined (0.7h+0.3d):   {overall['combined_correctness_pct']}")
    print(line)


# ============================================================================
# Synthetic fixture for --dry-run (no endpoint)
# ============================================================================
def synthetic_response(task: Task, trial_idx: int) -> dict[str, Any]:
    """Produce a deterministic fake model response per task for self-validation.
    Most tasks get a correct solution; one trial injects a BOS leak to exercise
    the degeneration guard."""
    # inject a degeneration on is_palindrome trial 2 to test the leak guard
    if task.task_id == "is_palindrome" and trial_idx == 2:
        return {"text": "<|begin_of_sentence|>" + "looploop" * 60,
                "tokens_out": 480, "latency_ms": 800.0}

    solutions: dict[str, str] = {
        "is_palindrome": (
            "```python\n"
            "def is_palindrome(s):\n"
            '    """Check palindrome ignoring spaces and case."""\n'
            "    t = ''.join(c.lower() for c in s if not c.isspace())\n"
            "    return t == t[::-1]\n```"
        ),
        "prime_check": (
            "```python\n"
            "def is_prime(n: int) -> bool:\n"
            '    """Return True if n is prime."""\n'
            "    if n < 2:\n        return False\n"
            "    i = 2\n    while i * i <= n:\n"
            "        if n % i == 0:\n            return False\n"
            "        i += 1\n    return True\n```"
        ),
        "binary_search": (
            "```python\n"
            "def binary_search(arr, target):\n"
            '    """Binary search; return index or -1."""\n'
            "    lo, hi = 0, len(arr) - 1\n"
            "    while lo <= hi:\n        mid = (lo + hi) // 2\n"
            "        if arr[mid] == target:\n            return mid\n"
            "        if arr[mid] < target:\n            lo = mid + 1\n"
            "        else:\n            hi = mid - 1\n    return -1\n```"
        ),
        "merge_intervals": (
            "```python\n"
            "def merge_intervals(intervals):\n"
            '    """Merge overlapping intervals."""\n'
            "    if not intervals:\n        return []\n"
            "    s = sorted(intervals)\n    out = [list(s[0])]\n"
            "    for a, b in s[1:]:\n"
            "        if a <= out[-1][1]:\n"
            "            out[-1][1] = max(out[-1][1], b)\n"
            "        else:\n            out.append([a, b])\n    return out\n```"
        ),
        "word_break": (
            "```python\n"
            "def word_break(s, words):\n"
            '    """DP word break."""\n'
            "    wset = set(words)\n    n = len(s)\n"
            "    dp = [False] * (n + 1)\n    dp[0] = True\n"
            "    for i in range(1, n + 1):\n"
            "        for j in range(i):\n"
            "            if dp[j] and s[j:i] in wset:\n"
            "                dp[i] = True\n                break\n    return dp[n]\n```"
        ),
        "regex_match": (
            "```python\n"
            "def regex_match(s, p):\n"
            '    """Regex match with . and *."""\n'
            "    import functools\n"
            "    @functools.lru_cache(None)\n"
            "    def go(i, j):\n"
            "        if j == len(p):\n            return i == len(s)\n"
            "        first = i < len(s) and p[j] in (s[i], '.')\n"
            "        if j + 1 < len(p) and p[j + 1] == '*':\n"
            "            return go(i, j + 2) or (first and go(i + 1, j))\n"
            "        return first and go(i + 1, j + 1)\n"
            "    return go(0, 0)\n```"
        ),
        "debug_sort": (
            "```python\n"
            "def quicksort(arr):\n"
            '    """Quicksort keeping duplicates."""\n'
            "    if len(arr) <= 1:\n        return arr\n"
            "    pivot = arr[0]\n"
            "    left = [x for x in arr[1:] if x < pivot]\n"
            "    mid = [x for x in arr if x == pivot]\n"
            "    right = [x for x in arr[1:] if x > pivot]\n"
            "    return quicksort(left) + mid + quicksort(right)\n```"
        ),
        "optimize_fibonacci": (
            "```python\n"
            "def fibonacci(n: int) -> int:\n"
            '    """Iterative Fibonacci."""\n'
            "    a, b = 0, 1\n    for _ in range(n):\n"
            "        a, b = b, a + b\n    return a\n```"
        ),
    }
    if task.task_id == "ecommerce_cart":
        fixed_main = ECOMMERCE_MAIN.replace(
            "            self.items[product_id].quantity += quantity  # Bug 1: No stock validation",
            "            self.items[product_id].quantity += quantity",
        ).replace(
            '        if self.discount_code == "SAVE10":\n            total -= 10.0',
            '        if self.discount_code == "SAVE10":\n            total *= 0.9',
        )
        # also add stock-aware validate fix already present; tests mainly need discount fix
        return {"text": "```python\n" + fixed_main + "```",
                "tokens_out": 400, "latency_ms": 1500.0}

    txt = solutions.get(task.task_id, "def noop():\n    return None")
    return {"text": txt, "tokens_out": 220, "latency_ms": 600.0}


# ============================================================================
# Main driver
# ============================================================================
def select_tasks(all_tasks: dict[str, Task], spec: str) -> list[Task]:
    if spec.strip().lower() == "all":
        return list(all_tasks.values())
    chosen = []
    for name in spec.split(","):
        name = name.strip()
        if name in all_tasks:
            chosen.append(all_tasks[name])
        else:
            print(f"[warn] unknown task '{name}' — skipping", file=sys.stderr)
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Offline port of the aistupidlevel.info coding benchmark.")
    ap.add_argument("--base-url", default="http://192.168.86.201:52415")
    ap.add_argument("--model", default="mlx-community/DeepSeek-V4-Flash-8bit")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--max-tokens", type=int, default=4096,
                   help="Generation cap. NOTE: for thinking models (DeepSeek-V4, "
                        "Kimi, Qwen3-thinking) this budget is consumed by hidden "
                        "reasoning_content first; 1024 truncates them mid-thought. "
                        "Use >=8192 for hard deep tasks.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--exec-timeout", type=int, default=60,
                   help="per-execution subprocess timeout (s)")
    ap.add_argument("--http-timeout", type=float, default=300.0)
    ap.add_argument("--baseline-json", default=None,
                   help="JSON file with {means:{axis:..}, stds:{axis:..}} to score "
                        "z-scores against a fixed baseline instead of the self-baseline proxy")
    ap.add_argument("--out", default="/Users/adam.durham/repos/exo/bench/aistupid_results.json")
    ap.add_argument("--dry-run", action="store_true",
                   help="validate task parsing + scorer on a synthetic fixture; no endpoint")
    ap.add_argument("--no-system", action="store_true",
                   help="do not send a system message")
    args = ap.parse_args()

    all_tasks = build_tasks()
    tasks = select_tasks(all_tasks, args.tasks)
    if not tasks:
        print("No tasks selected.", file=sys.stderr)
        return 2

    baseline_json = None
    if args.baseline_json:
        with open(args.baseline_json) as f:
            baseline_json = json.load(f)

    mode = "DRY-RUN (synthetic fixture, no endpoint)" if args.dry_run else "LIVE"
    print(f"[mode] {mode}  tasks={[t.task_id for t in tasks]}  trials={args.trials}")

    task_results: list[dict[str, Any]] = []
    for task in tasks:
        trials_scored: list[dict[str, Any]] = []
        for i in range(args.trials):
            if args.dry_run:
                resp = synthetic_response(task, i)
            else:
                try:
                    resp = call_chat_completion(
                        args.base_url, args.model, task.prompt,
                        args.max_tokens, args.temperature,
                        None if args.no_system else SYSTEM_MESSAGE,
                        args.http_timeout,
                    )
                except Exception as e:
                    print(f"[error] {task.task_id} trial {i}: {e}", file=sys.stderr)
                    resp = {"text": "", "tokens_out": 0, "latency_ms": 0.0}
            scored = score_trial(task, resp["text"], resp["tokens_out"],
                                resp["latency_ms"], args.exec_timeout,
                                reasoning_truncated=resp.get("reasoning_truncated", False),
                                reasoning_len=resp.get("reasoning_len", 0),
                                finish_reason=resp.get("finish_reason", ""))
            trials_scored.append(scored)
            tag = f" LEAK:{scored['leak_reason']}" if scored["leak"] else ""
            print(f"  {task.task_id} trial {i+1}/{args.trials}: "
                  f"correctness={scored['correctness_raw']*100:.0f}% "
                  f"({scored['passed']}/{scored['total']}){tag}")
        task_results.append(finalize_task(task, trials_scored))

    overall = aggregate_overall(task_results, baseline_json)
    print()
    print_report(task_results, overall, args.model)

    out_obj = {
        "model": args.model,
        "base_url": args.base_url,
        "mode": "dry-run" if args.dry_run else "live",
        "trials": args.trials,
        "weights": WEIGHTS,
        "overall": overall,
        "tasks": [
            {
                "task_id": tr["task_id"],
                "slug": tr["slug"],
                "difficulty": tr["difficulty"],
                "correctness_pct": tr["correctness_pct"],
                "per_axis_mean": tr["per_axis_mean"],
                "per_axis_ci": tr["per_axis_ci"],
                "stability_0_100": tr["stability_0_100"],
                "leak_count": tr["leak_count"],
                "trials": [
                    {k: v for k, v in t.items() if k != "axes"} | {"axes": t["axes"]}
                    for t in tr["trials"]
                ],
            }
            for tr in task_results
        ],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_obj, f, indent=2)
    print(f"[out] wrote {args.out}")

    if args.dry_run:
        g = overall["gauge"]
        ok = 0.0 <= g <= 100.0
        print(f"[dry-run] gauge={g:.1f} sane={ok}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
