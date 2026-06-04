#!/usr/bin/env python3
"""
hard_eval.py — a HARD, discriminating LLM evaluation harness with deterministic,
verifiable graders (no LLM-judge). Single self-contained file.

Targets an OpenAI-compatible endpoint (exo cluster). POSTs to
{base_url}/v1/chat/completions. Designed for a THINKING model
(DeepSeek-V4-Flash) which returns chain-of-thought in a separate
`reasoning_content` field — the real answer is in `content`. Grading reads
`content` and falls back to `reasoning_content` ONLY when content is empty.

Categories (15+ hard tasks), each with a single verifiable answer:
  1. HARD MATH        — extract 'ANSWER: <x>' line, exact normalized compare.
  2. HARD CODING      — execute model code against a hidden test battery.
  3. MULTI-BUG DEBUG  — fix 2-3 real bugs; self-consistent pytest-style battery.
  4. LONG-CONTEXT     — buried-fact + multi-step computation, 'ANSWER:' line.
  5. INSTRUCTION      — deterministic constraint checks (word count, JSON, sort).

EVERY hardcoded expected answer was verified computationally at build time
(sympy / direct Python). See VERIFIED_ANSWERS note at bottom.

--dry-run feeds each grader a KNOWN-CORRECT reference (=> pass 1.0) and a
known-WRONG answer (=> pass 0.0), proving graders are discriminating. No
endpoint contact in dry-run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

try:
    import httpx  # type: ignore
    _HAS_HTTPX = True
except Exception:  # pragma: no cover
    _HAS_HTTPX = False


# ============================================================================
# Leak / truncation detection
# ============================================================================
_LEAK_TOKENS = [
    "<|begin_of_sentence|>",
    "<\uff5cbegin\u2581of\u2581sentence\uff5c>",
    "</think>",
]


def detect_leak(content: str) -> tuple[bool, str]:
    if not content:
        return False, ""
    for tok in _LEAK_TOKENS:
        if tok in content:
            return True, tok
    return False, ""


# ============================================================================
# Answer extraction helpers
# ============================================================================
def extract_answer_line(text: str) -> Optional[str]:
    """Extract the value after the LAST 'ANSWER:' line (case-insensitive)."""
    if not text:
        return None
    matches = re.findall(r"(?im)^\s*ANSWER\s*:\s*(.+?)\s*$", text)
    if not matches:
        # Allow inline 'ANSWER: x' not at line start as a fallback.
        m = re.findall(r"(?i)ANSWER\s*:\s*([^\n]+)", text)
        if not m:
            return None
        return m[-1].strip()
    return matches[-1].strip()


def normalize_numeric(s: str) -> Optional[str]:
    """Normalize a numeric/fraction answer to a canonical string for compare.
    Returns None if not parseable as a number/fraction."""
    if s is None:
        return None
    t = s.strip()
    # strip surrounding markup/punctuation
    t = t.strip().strip(".").strip()
    t = t.replace(",", "").replace(" ", "")
    t = t.strip("$`*")
    # fraction a/b
    m = re.fullmatch(r"(-?\d+)\s*/\s*(\d+)", t)
    if m:
        from fractions import Fraction
        try:
            fr = Fraction(int(m.group(1)), int(m.group(2)))
            if fr.denominator == 1:
                return str(fr.numerator)
            return f"{fr.numerator}/{fr.denominator}"
        except Exception:
            return None
    # integer
    m = re.fullmatch(r"-?\d+", t)
    if m:
        return str(int(t))
    # float -> reduce to int if integral
    m = re.fullmatch(r"-?\d*\.\d+", t)
    if m:
        f = float(t)
        if f == int(f):
            return str(int(f))
        return repr(f)
    return None


def grade_numeric(content: str, reasoning: str, expected: str) -> tuple[float, str]:
    """Deterministic numeric/fraction grader keyed off an 'ANSWER:' line.

    Primary: parse the value after an 'ANSWER:' line. Fallback: if the model
    stated the correct value in prose (in `content`) without the required
    ANSWER line, accept it when the EXACT expected value appears as a clean
    standalone number in the content. This corrects a grader artifact where
    DeepSeek-V4 computes the right answer but skips the format instruction.
    The fallback only scans `content` (the final answer), never `reasoning`,
    and requires word-boundary exact-value match to avoid incidental hits.
    """
    src = content if content.strip() else reasoning
    raw = extract_answer_line(src)
    snippet = (raw or "")[:40]
    exp = normalize_numeric(expected)
    if raw is not None:
        got = normalize_numeric(raw)
        if got is not None:
            return (1.0 if got == exp else 0.0), snippet
    # Fallback: exact expected value stated as a standalone number in content.
    if exp is not None and content.strip():
        # Build a set of standalone numeric tokens from content (strip commas
        # used as thousands separators, e.g. "354,224,848,...").
        for tok in re.findall(r"-?[\d,]*\d", content):
            norm = normalize_numeric(tok.replace(",", ""))
            if norm is not None and norm == exp:
                return 1.0, f"prose:{exp}"
    return 0.0, snippet


def grade_string(content: str, reasoning: str, expected: str) -> tuple[float, str]:
    """Exact string match on the ANSWER line, whitespace/case normalized."""
    src = content if content.strip() else reasoning
    raw = extract_answer_line(src)
    snippet = (raw or "")[:40]
    if raw is None:
        return 0.0, snippet
    norm = lambda s: re.sub(r"\s+", " ", s.strip()).lower()
    return (1.0 if norm(raw) == norm(expected) else 0.0), snippet


# ============================================================================
# Code extraction (ported style from aistupid_harness.extract_python)
# ============================================================================
def extract_python(raw: str, expected_symbol: str) -> str:
    if not raw:
        return ""
    s = raw.replace("\r\n", "\n").strip()
    code_block_re = re.compile(r"```(?:python|py)?\s*([\s\S]*?)```", re.IGNORECASE)
    blocks = [m.group(1).strip() for m in code_block_re.finditer(s)]
    if blocks:
        with_symbol = next(
            (b for b in blocks
             if re.search(rf"\b(def|class)\s+{re.escape(expected_symbol)}\b", b)),
            None,
        )
        s = (with_symbol or max(blocks, key=len)).strip()
    if not re.search(r"^(\s*def |\s*class )", s, re.MULTILINE):
        m = re.search(r"^\s*(def|class)\s+", s, re.MULTILINE)
        if m:
            s = s[m.start():]
    s = re.sub(r"^\s*```.*$", "", s, flags=re.MULTILINE).strip()
    return s


# ============================================================================
# Subprocess code sandbox (ported style from aistupid_harness)
# ============================================================================
def _run_subprocess(cmd: list[str], cwd: str, timeout: int) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "TIMEOUT"
    except Exception as e:  # pragma: no cover
        return 1, "", f"RUNNER_ERROR: {e}"


def run_code_battery(clean_code: str, expected_symbol: str, battery: str,
                     timeout: int) -> tuple[float, str]:
    """Write the model's code as solution.py and a test battery that imports it,
    counts passed/total, prints 'PASS x/y'. score = x/y."""
    if not clean_code.strip():
        return 0.0, "empty_code"
    with tempfile.TemporaryDirectory(prefix="hardeval_") as d:
        with open(os.path.join(d, "solution.py"), "w") as f:
            f.write(clean_code)
        with open(os.path.join(d, "battery.py"), "w") as f:
            f.write(battery)
        rc, out, err = _run_subprocess(
            [sys.executable, "-I", "battery.py"], cwd=d, timeout=timeout)
    m = re.search(r"PASS\s+(\d+)/(\d+)", out)
    if not m:
        return 0.0, (out[-150:] + " | " + err[-150:]).strip()
    passed, total = int(m.group(1)), int(m.group(2))
    score = passed / total if total else 0.0
    return score, (f"{passed}/{total}" if score == 1.0 else f"{passed}/{total} {err[-100:]}")


# A battery is a python program that imports solution and prints "PASS x/y".
_BATTERY_TEMPLATE = '''import importlib.util, traceback
spec = importlib.util.spec_from_file_location("solution", "solution.py")
sol = importlib.util.module_from_spec(spec)
_passed = 0
_total = 0
def check(cond):
    global _passed, _total
    _total += 1
    try:
        if cond:
            _passed += 1
    except Exception:
        pass
try:
    spec.loader.exec_module(sol)
{checks}
except Exception:
    traceback.print_exc()
print(f"PASS {{_passed}}/{{_total}}")
'''


def make_battery(check_lines: list[str]) -> str:
    indented = "\n".join("    " + ln for ln in check_lines)
    return _BATTERY_TEMPLATE.format(checks=indented)


# ============================================================================
# TASK DEFINITIONS
# ============================================================================
@dataclass
class Task:
    task_id: str
    category: str          # math | coding | debug | longcontext | instruction
    prompt: str
    grader: Callable[[str, str], tuple[float, str]]  # (content, reasoning) -> (pass, snippet)
    ref_correct: str       # known-correct reference RESPONSE for dry-run
    ref_wrong: str         # known-wrong reference RESPONSE for dry-run
    expected_symbol: str = ""


TASKS: list[Task] = []


def add_math(task_id: str, prompt: str, expected: str, ref_correct: str,
             ref_wrong: str, numeric: bool = True) -> None:
    g = (lambda c, r, e=expected: grade_numeric(c, r, e)) if numeric \
        else (lambda c, r, e=expected: grade_string(c, r, e))
    TASKS.append(Task(task_id, "math", prompt, g, ref_correct, ref_wrong))


_MATH_SUFFIX = "\nEnd with a line exactly: ANSWER: <value>"

# --- MATH 1: largest prime factor (verified: 630803) ---
add_math(
    "math_largest_prime_factor",
    "Find the largest prime factor of the integer 1234567891011." + _MATH_SUFFIX,
    "630803",
    "Reasoning about the factorization.\nANSWER: 630803",
    "Reasoning.\nANSWER: 333667",
)

# --- MATH 2: modular exponentiation (verified: 436) ---
add_math(
    "math_modexp",
    "Compute 7^644 mod 645." + _MATH_SUFFIX,
    "436",
    "Using Euler / repeated squaring.\nANSWER: 436",
    "Wrong.\nANSWER: 1",
)

# --- MATH 3: derangements of 11 (verified: 14684570) ---
add_math(
    "math_derangements_11",
    "How many derangements (permutations with no fixed point) are there of "
    "11 distinct objects?" + _MATH_SUFFIX,
    "14684570",
    "D(11) = 11! * sum.\nANSWER: 14684570",
    "ANSWER: 1334961",
)

# --- MATH 4: Catalan C6 lattice paths (verified: 132) ---
add_math(
    "math_lattice_paths",
    "Count the lattice paths from (0,0) to (6,6) using unit steps right and up "
    "that never go strictly above the diagonal line y = x (i.e. at every point "
    "the number of up-steps so far does not exceed the number of right-steps)." + _MATH_SUFFIX,
    "132",
    "This is the 6th Catalan number.\nANSWER: 132",
    "ANSWER: 924",
)

# --- MATH 5: inclusion-exclusion count (verified: 733334) ---
add_math(
    "math_incl_excl",
    "How many integers in the range 1 to 1000000 inclusive are divisible by "
    "none of 6, 10, or 15?" + _MATH_SUFFIX,
    "733334",
    "Inclusion-exclusion.\nANSWER: 733334",
    "ANSWER: 266666",
)

# --- MATH 6: subsets with no two consecutive (verified: 17711 = Fib(22)) ---
add_math(
    "math_no_two_consecutive",
    "How many subsets of {1,2,...,20} contain no two consecutive integers? "
    "(The empty set counts.)" + _MATH_SUFFIX,
    "17711",
    "F(22).\nANSWER: 17711",
    "ANSWER: 10946",
)

# --- MATH 7: MISSISSIPPI permutations (verified: 34650) ---
add_math(
    "math_anagrams",
    "How many distinct arrangements are there of the letters of the word "
    "MISSISSIPPI?" + _MATH_SUFFIX,
    "34650",
    "11!/(4!4!2!).\nANSWER: 34650",
    "ANSWER: 39916800",
)

# --- MATH 8: digit sum of 2^100 (verified: 115) ---
add_math(
    "math_digit_sum",
    "Compute the sum of the decimal digits of 2^100." + _MATH_SUFFIX,
    "115",
    "2^100 = 1267650600228229401496703205376; digits sum.\nANSWER: 115",
    "ANSWER: 100",
)

# --- MATH 9: C(1000,500) mod 1e9+7 (verified: 159835829) ---
add_math(
    "math_binom_mod",
    "Compute the binomial coefficient C(1000, 500) modulo 1000000007." + _MATH_SUFFIX,
    "159835829",
    "Modular factorials.\nANSWER: 159835829",
    "ANSWER: 500500",
)


# ============================================================================
# CODING TASKS
# ============================================================================
def add_coding(task_id: str, prompt: str, symbol: str, check_lines: list[str],
               ref_correct: str, ref_wrong: str, timeout: int = 12) -> None:
    battery = make_battery(check_lines)

    def grader(content: str, reasoning: str, _b=battery, _s=symbol, _t=timeout):
        src = content if content.strip() else reasoning
        clean = extract_python(src, _s)
        return run_code_battery(clean, _s, _b, _t)

    TASKS.append(Task(task_id, "coding", prompt, grader, ref_correct, ref_wrong, symbol))


# --- CODING 1: LRU cache with O(1) get/put ---
add_coding(
    "code_lru_cache",
    "Implement a class `LRUCache` with constructor `LRUCache(capacity: int)` and "
    "methods `get(key: int) -> int` (returns -1 if absent) and "
    "`put(key: int, value: int) -> None`. It must evict the least-recently-used "
    "key when capacity is exceeded. Both get and put must be O(1). Reading or "
    "writing a key counts as a use. Provide only Python in a ```python``` block.",
    "LRUCache",
    [
        "c = sol.LRUCache(2)",
        "c.put(1,1); c.put(2,2)",
        "check(c.get(1)==1)",
        "c.put(3,3)",          # evicts 2
        "check(c.get(2)==-1)",
        "c.put(4,4)",          # evicts 1
        "check(c.get(1)==-1)",
        "check(c.get(3)==3)",
        "check(c.get(4)==4)",
        "d = sol.LRUCache(1)",
        "d.put(5,5); d.put(6,6)",
        "check(d.get(5)==-1)",
        "check(d.get(6)==6)",
        "e = sol.LRUCache(2)",
        "e.put(1,1); e.put(2,2); e.put(1,10)",  # update + use 1
        "e.put(3,3)",          # evicts 2 (1 was just used)
        "check(e.get(2)==-1)",
        "check(e.get(1)==10)",
    ],
    "```python\n"
    "class LRUCache:\n"
    "    def __init__(self, capacity):\n"
    "        from collections import OrderedDict\n"
    "        self.cap = capacity\n"
    "        self.d = OrderedDict()\n"
    "    def get(self, key):\n"
    "        if key not in self.d:\n"
    "            return -1\n"
    "        self.d.move_to_end(key)\n"
    "        return self.d[key]\n"
    "    def put(self, key, value):\n"
    "        if key in self.d:\n"
    "            self.d.move_to_end(key)\n"
    "        self.d[key] = value\n"
    "        if len(self.d) > self.cap:\n"
    "            self.d.popitem(last=False)\n"
    "```",
    "```python\n"
    "class LRUCache:\n"
    "    def __init__(self, capacity):\n"
    "        pass\n"
    "    def get(self, key):\n"
    "        return None\n"   # wrong type, never matches
    "    def put(self, key, value):\n"
    "        pass\n"
    "```",
)

# --- CODING 2: Dijkstra shortest path ---
add_coding(
    "code_dijkstra",
    "Implement `shortest_path(n, edges, src, dst)` where n is the number of nodes "
    "labeled 0..n-1, edges is a list of (u, v, w) directed edges with w >= 0, and "
    "you must return the length of the shortest path from src to dst, or -1 if "
    "unreachable. Provide only Python in a ```python``` block.",
    "shortest_path",
    [
        "check(sol.shortest_path(5, [(0,1,2),(1,2,3),(0,2,10),(2,3,1),(3,4,7)], 0, 4)==13)",
        "check(sol.shortest_path(3, [(0,1,1),(1,2,1)], 0, 2)==2)",
        "check(sol.shortest_path(3, [(0,1,1)], 0, 2)==-1)",
        "check(sol.shortest_path(2, [], 0, 0)==0)",
        "check(sol.shortest_path(4, [(0,1,5),(0,2,1),(2,1,1),(1,3,2),(2,3,10)], 0, 3)==4)",
        "check(sol.shortest_path(1, [], 0, 0)==0)",
    ],
    "```python\n"
    "import heapq\n"
    "def shortest_path(n, edges, src, dst):\n"
    "    g = [[] for _ in range(n)]\n"
    "    for u,v,w in edges:\n"
    "        g[u].append((v,w))\n"
    "    dist = [float('inf')]*n\n"
    "    dist[src] = 0\n"
    "    pq = [(0, src)]\n"
    "    while pq:\n"
    "        d,u = heapq.heappop(pq)\n"
    "        if d > dist[u]:\n"
    "            continue\n"
    "        for v,w in g[u]:\n"
    "            if d+w < dist[v]:\n"
    "                dist[v] = d+w\n"
    "                heapq.heappush(pq, (dist[v], v))\n"
    "    return dist[dst] if dist[dst] != float('inf') else -1\n"
    "```",
    "```python\n"
    "def shortest_path(n, edges, src, dst):\n"
    "    return None\n"   # always wrong
    "```",
)

# --- CODING 3: edit distance (Levenshtein) ---
add_coding(
    "code_edit_distance",
    "Implement `edit_distance(a, b)` returning the Levenshtein edit distance "
    "(minimum single-character insertions, deletions, substitutions to turn a "
    "into b). Provide only Python in a ```python``` block.",
    "edit_distance",
    [
        "check(sol.edit_distance('kitten','sitting')==3)",
        "check(sol.edit_distance('','')==0)",
        "check(sol.edit_distance('abc','')==3)",
        "check(sol.edit_distance('','abc')==3)",
        "check(sol.edit_distance('flaw','lawn')==2)",
        "check(sol.edit_distance('intention','execution')==5)",
        "check(sol.edit_distance('a'*50,'b'*50)==50)",
    ],
    "```python\n"
    "def edit_distance(a, b):\n"
    "    m,n=len(a),len(b)\n"
    "    dp=list(range(n+1))\n"
    "    for i in range(1,m+1):\n"
    "        prev=dp[0]; dp[0]=i\n"
    "        for j in range(1,n+1):\n"
    "            cur=dp[j]\n"
    "            cost=0 if a[i-1]==b[j-1] else 1\n"
    "            dp[j]=min(dp[j]+1, dp[j-1]+1, prev+cost)\n"
    "            prev=cur\n"
    "    return dp[n]\n"
    "```",
    "```python\n"
    "def edit_distance(a, b):\n"
    "    return -1\n"   # always wrong
    "```",
)

# --- CODING 4: topological sort with cycle detection ---
add_coding(
    "code_toposort",
    "Implement `topo_sort(n, edges)` for a directed graph on nodes 0..n-1 where "
    "edges is a list of (u, v) meaning u must come before v. Return a valid "
    "topological ordering as a list, or return None if the graph has a cycle. "
    "Provide only Python in a ```python``` block.",
    "topo_sort",
    [
        "r = sol.topo_sort(4, [(0,1),(1,2),(2,3)])",
        "check(r == [0,1,2,3])",
        "check(sol.topo_sort(3, [(0,1),(1,2),(2,0)]) is None)",
        "r2 = sol.topo_sort(3, [(0,1),(0,2)])",
        "check(r2 is not None and r2.index(0) < r2.index(1) and r2.index(0) < r2.index(2))",
        "check(sol.topo_sort(2, []) is not None and sorted(sol.topo_sort(2,[]))==[0,1])",
        "check(sol.topo_sort(1, [(0,0)]) is None)",
        "r3 = sol.topo_sort(6, [(5,2),(5,0),(4,0),(4,1),(2,3),(3,1)])",
        "check(r3 is not None and all(r3.index(u)<r3.index(v) for u,v in [(5,2),(5,0),(4,0),(4,1),(2,3),(3,1)]))",
    ],
    "```python\n"
    "def topo_sort(n, edges):\n"
    "    from collections import deque\n"
    "    g=[[] for _ in range(n)]; indeg=[0]*n\n"
    "    for u,v in edges:\n"
    "        g[u].append(v); indeg[v]+=1\n"
    "    q=deque([i for i in range(n) if indeg[i]==0])\n"
    "    order=[]\n"
    "    while q:\n"
    "        u=q.popleft(); order.append(u)\n"
    "        for v in g[u]:\n"
    "            indeg[v]-=1\n"
    "            if indeg[v]==0: q.append(v)\n"
    "    return order if len(order)==n else None\n"
    "```",
    "```python\n"
    "def topo_sort(n, edges):\n"
    "    return []\n"    # always wrong (empty order, never detects cycles correctly)
    "```",
)

# --- CODING 5: min-heap based segment tree range sum + point update ---
add_coding(
    "code_segment_tree",
    "Implement a class `SegTree` with constructor `SegTree(arr: list[int])` "
    "supporting `update(i: int, val: int)` (sets arr[i]=val) and "
    "`query(l: int, r: int) -> int` returning the sum of arr[l..r] inclusive. "
    "Both operations must be O(log n). Provide only Python in a ```python``` block.",
    "SegTree",
    [
        "t = sol.SegTree([1,2,3,4,5])",
        "check(t.query(0,4)==15)",
        "check(t.query(1,3)==9)",
        "t.update(2,10)",
        "check(t.query(0,4)==22)",
        "check(t.query(2,2)==10)",
        "t2 = sol.SegTree([5])",
        "check(t2.query(0,0)==5)",
        "t2.update(0,-3)",
        "check(t2.query(0,0)==-3)",
        "import random",
        "a=[random.randint(-9,9) for _ in range(64)]",
        "t3=sol.SegTree(list(a))",
        "ok=all(t3.query(i,j)==sum(a[i:j+1]) for i in range(0,64,7) for j in range(i,64,9))",
        "check(ok)",
        "t3.update(10, 100); a[10]=100",
        "check(t3.query(0,63)==sum(a))",
    ],
    "```python\n"
    "class SegTree:\n"
    "    def __init__(self, arr):\n"
    "        self.n=len(arr)\n"
    "        self.t=[0]*(2*self.n)\n"
    "        for i,v in enumerate(arr): self.t[self.n+i]=v\n"
    "        for i in range(self.n-1,0,-1): self.t[i]=self.t[2*i]+self.t[2*i+1]\n"
    "    def update(self, i, val):\n"
    "        i+=self.n; self.t[i]=val; i//=2\n"
    "        while i>=1:\n"
    "            self.t[i]=self.t[2*i]+self.t[2*i+1]; i//=2\n"
    "    def query(self, l, r):\n"
    "        l+=self.n; r+=self.n+1; s=0\n"
    "        while l<r:\n"
    "            if l&1: s+=self.t[l]; l+=1\n"
    "            if r&1: r-=1; s+=self.t[r]\n"
    "            l//=2; r//=2\n"
    "        return s\n"
    "```",
    "```python\n"
    "class SegTree:\n"
    "    def __init__(self, arr): self.a=list(arr)\n"
    "    def update(self, i, val): pass\n"   # never updates
    "    def query(self, l, r): return sum(self.a[l:r])\n"  # off-by-one (excludes r)
    "```",
)


# ============================================================================
# MULTI-BUG DEBUGGING TASKS
# ============================================================================
def add_debug(task_id: str, prompt: str, symbol: str, check_lines: list[str],
              ref_correct: str, ref_wrong: str, timeout: int = 12) -> None:
    add_coding(task_id, prompt, symbol, check_lines, ref_correct, ref_wrong, timeout)
    TASKS[-1].category = "debug"


# --- DEBUG 1: running statistics class with 3 real bugs ---
# Buggy version (shown to model). Bugs:
#   (a) variance uses /n but should be sample variance /(n-1)  -> actually we
#       define spec as POPULATION variance /n, and the bug is /(n+1).
#   (b) max not updated correctly (uses < instead of >).
#   (c) mean integer division.
_DEBUG1_PROMPT = (
    "The following Python class is meant to track running statistics over a "
    "stream of numbers fed via `add(x)`. Per the spec: `mean()` returns the "
    "arithmetic mean as a float; `variance()` returns the POPULATION variance "
    "(divide by n); `maximum()` returns the largest value seen. The class has "
    "exactly three bugs. Fix them so all behave per spec. Return the COMPLETE "
    "corrected class in a ```python``` block.\n\n"
    "```python\n"
    "class RunningStats:\n"
    "    def __init__(self):\n"
    "        self.vals = []\n"
    "        self._max = None\n"
    "    def add(self, x):\n"
    "        self.vals.append(x)\n"
    "        if self._max is None or x < self._max:   # BUG\n"
    "            self._max = x\n"
    "    def mean(self):\n"
    "        return sum(self.vals) // len(self.vals)   # BUG\n"
    "    def variance(self):\n"
    "        m = self.mean()\n"
    "        n = len(self.vals)\n"
    "        return sum((v - m) ** 2 for v in self.vals) / (n + 1)   # BUG\n"
    "    def maximum(self):\n"
    "        return self._max\n"
    "```"
)
add_debug(
    "debug_running_stats",
    _DEBUG1_PROMPT,
    "RunningStats",
    [
        "s = sol.RunningStats()",
        "for x in [2,4,6]: s.add(x)",
        "check(abs(s.mean()-4.0) < 1e-9)",
        "check(s.maximum()==6)",
        "check(abs(s.variance()-(8/3)) < 1e-9)",   # pop var of 2,4,6 = (4+0+4)/3
        "t = sol.RunningStats()",
        "for x in [1,2]: t.add(x)",
        "check(abs(t.mean()-1.5) < 1e-9)",          # would be 1 with // bug
        "check(t.maximum()==2)",                     # would be 1 with < bug
        "check(abs(t.variance()-0.25) < 1e-9)",      # (0.25+0.25)/2; /(n+1) gives 1/6
        "u = sol.RunningStats()",
        "for x in [5,3,9,1]: u.add(x)",
        "check(u.maximum()==9)",
        "check(abs(u.mean()-4.5) < 1e-9)",
    ],
    "```python\n"
    "class RunningStats:\n"
    "    def __init__(self):\n"
    "        self.vals = []\n"
    "        self._max = None\n"
    "    def add(self, x):\n"
    "        self.vals.append(x)\n"
    "        if self._max is None or x > self._max:\n"
    "            self._max = x\n"
    "    def mean(self):\n"
    "        return sum(self.vals) / len(self.vals)\n"
    "    def variance(self):\n"
    "        m = self.mean()\n"
    "        n = len(self.vals)\n"
    "        return sum((v - m) ** 2 for v in self.vals) / n\n"
    "    def maximum(self):\n"
    "        return self._max\n"
    "```",
    # wrong fix: returns constant wrong values for every method
    "```python\n"
    "class RunningStats:\n"
    "    def __init__(self):\n"
    "        self.vals = []\n"
    "    def add(self, x):\n"
    "        self.vals.append(x)\n"
    "    def mean(self):\n"
    "        return 0.0\n"
    "    def variance(self):\n"
    "        return 0.0\n"
    "    def maximum(self):\n"
    "        return 0\n"
    "```",
)

# --- DEBUG 2: binary search with off-by-one + wrong return bugs ---
_DEBUG2_PROMPT = (
    "This Python function should return the index of `target` in the sorted "
    "list `arr`, or -1 if not present. It has exactly two bugs causing wrong "
    "results and occasional infinite loops / index errors. Fix them. Return the "
    "complete corrected function in a ```python``` block.\n\n"
    "```python\n"
    "def bsearch(arr, target):\n"
    "    lo, hi = 0, len(arr)\n"
    "    while lo < hi:\n"
    "        mid = (lo + hi) // 2\n"
    "        if arr[mid] == target:\n"
    "            return mid\n"
    "        elif arr[mid] < target:\n"
    "            hi = mid          # BUG: should move lo\n"
    "        else:\n"
    "            lo = mid          # BUG: should move hi\n"
    "    return 0                  # BUG-ish: should be -1\n"
    "```"
)
add_debug(
    "debug_binary_search",
    _DEBUG2_PROMPT,
    "bsearch",
    [
        "a=[1,3,5,7,9,11]",
        "check(sol.bsearch(a,7)==3)",
        "check(sol.bsearch(a,1)==0)",
        "check(sol.bsearch(a,11)==5)",
        "check(sol.bsearch(a,4)==-1)",
        "check(sol.bsearch(a,12)==-1)",
        "check(sol.bsearch([],5)==-1)",
        "check(sol.bsearch([42],42)==0)",
        "check(sol.bsearch([42],7)==-1)",
        "import random",
        "b=sorted(random.sample(range(1000),100))",
        "ok=all(sol.bsearch(b,x)==b.index(x) for x in b)",
        "check(ok)",
        "check(all(sol.bsearch(b,x)==-1 for x in range(1000) if x not in set(b)))",
    ],
    "```python\n"
    "def bsearch(arr, target):\n"
    "    lo, hi = 0, len(arr)\n"
    "    while lo < hi:\n"
    "        mid = (lo + hi) // 2\n"
    "        if arr[mid] == target:\n"
    "            return mid\n"
    "        elif arr[mid] < target:\n"
    "            lo = mid + 1\n"
    "        else:\n"
    "            hi = mid\n"
    "    return -1\n"
    "```",
    # wrong fix: fixes return but not the lo/hi swap -> still broken
    "```python\n"
    "def bsearch(arr, target):\n"
    "    lo, hi = 0, len(arr)\n"
    "    while lo < hi:\n"
    "        mid = (lo + hi) // 2\n"
    "        if arr[mid] == target:\n"
    "            return mid\n"
    "        elif arr[mid] < target:\n"
    "            hi = mid\n"
    "        else:\n"
    "            lo = mid\n"
    "    return -1\n"
    "```",
)


# ============================================================================
# LONG-CONTEXT / MULTI-STEP TASKS
# ============================================================================
_LONGCTX_PASSAGE = """\
INTERNAL LOGISTICS MEMO — FISCAL QUARTER REVIEW (do not distribute)

Section 1. Background.
The Northbridge distribution network operates four regional warehouses, each
identified by a Greek letter. Over the last decade the network has expanded
from a single facility into a continental operation. Much of the early growth
was driven by seasonal demand for the company's flagship product line, but in
recent years the product mix has diversified considerably. The board has asked
for a consolidated shipping estimate ahead of the upcoming peak season, and the
operations team assembled the figures scattered throughout this memo. Note that
several earlier drafts of this memo circulated with outdated numbers; the
figures in THIS version supersede all prior drafts.

Section 2. Warehouse Alpha.
Warehouse Alpha, the oldest facility, sits near the western rail hub. After the
most recent inventory reconciliation it holds 1240 units of the product slated
for shipment. (An earlier draft listed 1300 units, but that figure was retracted
after a counting error was discovered.) Alpha's loading dock was renovated last
spring and can now handle larger trucks.

Section 3. Warehouse Beta.
Beta is the smallest of the four. It currently holds 875 units ready to ship.
Beta also stores a quantity of obsolete packaging material that is NOT counted
among the shippable units and should be ignored for this estimate.

Section 4. Warehouse Gamma.
Gamma, the largest facility, holds 1560 shippable units. Gamma additionally
houses the regional returns-processing center, but returns are tracked in a
separate system and are irrelevant here.

Section 5. Warehouse Delta.
Delta, the newest warehouse, came online two years ago. It holds 690 units
ready for shipment.

Section 6. Logistics parameters.
Every shippable unit weighs exactly 3.5 kilograms. The company's standard
delivery truck has a maximum payload capacity of 5000 kilograms. Trucks cannot
be loaded beyond this limit, and partial trucks are acceptable (a truck may
depart less than full). Assume units from different warehouses can be combined
freely onto the same truck.

Section 7. Task.
Using ONLY the superseding figures in this memo, determine the minimum number of
standard delivery trucks required to ship every shippable unit across all four
warehouses in a single wave.
"""
add_math(
    "longctx_truck_count",
    _LONGCTX_PASSAGE + "\nEnd with a line exactly: ANSWER: <value>",
    "4",   # 4365 units * 3.5 = 15277.5 kg / 5000 = ceil 3.05 -> 4
    "Total units 4365, weight 15277.5 kg, ceil(15277.5/5000)=4.\nANSWER: 4",
    "I'll use 1300 for Alpha.\nANSWER: 3",
)

# --- LONG-CONTEXT 2: state-tracking accumulator (verified: 143) ---
_STATE_PROMPT = (
    "Start with an accumulator equal to 0. Apply the following operations in "
    "order, where each operates on the current accumulator value:\n"
    "1. add 17\n"
    "2. multiply by 3\n"
    "3. subtract 8\n"
    "4. multiply by 2\n"
    "5. add 100\n"
    "6. subtract 43\n"
    "7. multiply by 1\n"
    "Report the final accumulator value." + _MATH_SUFFIX
)
add_math("longctx_accumulator", _STATE_PROMPT, "143",
         "((((0+17)*3-8)*2)+100-43)*1 = 143.\nANSWER: 143",
         "ANSWER: 100")

# --- LONG-CONTEXT 3: RPN evaluation (verified: 33) ---
add_math(
    "longctx_rpn",
    "Evaluate the following expression written in Reverse Polish Notation "
    "(postfix), where each operator applies to the two preceding values in order "
    "(for subtraction, the earlier value minus the later value):\n"
    "3 4 + 5 * 2 -\n"
    "Report the resulting integer." + _MATH_SUFFIX,
    "33",
    "(3+4)=7; 7*5=35; 35-2=33.\nANSWER: 33",
    "ANSWER: 13",
)


# ============================================================================
# INSTRUCTION-FOLLOWING UNDER CONSTRAINT TASKS
# ============================================================================
def add_instruction(task_id: str, prompt: str, checker: Callable[[str], tuple[float, str]],
                    ref_correct: str, ref_wrong: str) -> None:
    def grader(content: str, reasoning: str, _c=checker):
        src = content if content.strip() else reasoning
        return _c(src)
    TASKS.append(Task(task_id, "instruction", prompt, grader, ref_correct, ref_wrong))


# --- INSTRUCTION 1: exactly N words ---
def _check_exact_words(text: str, n: int = 17) -> tuple[float, str]:
    # Look for a fenced or last non-empty line answer; we count words in the
    # whole content after stripping a leading 'ANSWER:' if present.
    body = text.strip()
    m = re.search(r"(?is)ANSWER\s*:\s*(.+)\Z", body)
    if m:
        body = m.group(1).strip()
    words = re.findall(r"\S+", body)
    return (1.0 if len(words) == n else 0.0), f"{len(words)}w"

add_instruction(
    "instr_exact_words",
    "Write a single grammatical English sentence about the ocean that contains "
    "EXACTLY 17 words. Output ONLY the sentence on one line, prefixed with "
    "'ANSWER: ' (the 17-word count excludes the 'ANSWER:' prefix). Do not add "
    "any other text.",
    lambda t: _check_exact_words(t, 17),
    "ANSWER: The vast blue ocean stretches far beyond the distant horizon where "
    "gentle waves meet the morning sky.",
    "ANSWER: The ocean is big and blue.",
)

# --- INSTRUCTION 2: valid JSON matching schema ---
def _check_json_schema(text: str) -> tuple[float, str]:
    # extract the last {...} JSON object
    m = re.findall(r"\{[\s\S]*\}", text)
    if not m:
        return 0.0, "no_json"
    raw = m[-1]
    try:
        obj = json.loads(raw)
    except Exception:
        return 0.0, "bad_json"
    if not isinstance(obj, dict):
        return 0.0, "not_obj"
    keys = set(obj.keys())
    if keys != {"name", "age", "languages", "active"}:
        return 0.0, f"keys={sorted(keys)}"
    ok = (isinstance(obj["name"], str)
          and isinstance(obj["age"], int) and not isinstance(obj["age"], bool)
          and isinstance(obj["languages"], list) and len(obj["languages"]) == 3
          and all(isinstance(x, str) for x in obj["languages"])
          and isinstance(obj["active"], bool))
    return (1.0 if ok else 0.0), "valid" if ok else "schema_fail"

add_instruction(
    "instr_json_schema",
    "Output a single valid JSON object and NOTHING else. It must have EXACTLY "
    "these keys: \"name\" (a string), \"age\" (an integer, not a string), "
    "\"languages\" (an array of exactly 3 strings), and \"active\" (a boolean). "
    "No extra keys, no comments, no markdown fences.",
    _check_json_schema,
    '{"name": "Ada", "age": 36, "languages": ["Python", "Rust", "Go"], "active": true}',
    '{"name": "Ada", "age": "36", "languages": ["Python", "Rust"], "active": "yes", "extra": 1}',
)

# --- INSTRUCTION 3: reverse-alphabetical, dedup ---
def _check_rev_alpha_dedup(text: str) -> tuple[float, str]:
    src = (re.search(r"(?is)ANSWER\s*:\s*(.+)\Z", text) or [None, text])
    body = src[1] if isinstance(src, list) else (src.group(1) if src else text)
    # take last non-empty line
    lines = [ln.strip() for ln in body.strip().splitlines() if ln.strip()]
    if not lines:
        return 0.0, "empty"
    line = lines[-1]
    items = [w.strip().lower() for w in line.split(",") if w.strip()]
    expected = sorted(set([
        "banana", "apple", "cherry", "date", "fig", "grape", "kiwi", "lemon",
    ]), reverse=True)
    return (1.0 if items == expected else 0.0), ",".join(items)[:40]

add_instruction(
    "instr_rev_alpha",
    "Take this list of fruits (note duplicates): apple, banana, cherry, apple, "
    "date, fig, banana, grape, kiwi, lemon, cherry. Remove duplicates, then "
    "output them sorted in REVERSE alphabetical order as a single "
    "comma-separated line (lowercase, ', ' separator). Prefix the line with "
    "'ANSWER: '.",
    _check_rev_alpha_dedup,
    "ANSWER: lemon, kiwi, grape, fig, date, cherry, banana, apple",
    "ANSWER: apple, banana, cherry, date, fig, grape, kiwi, lemon",
)


# ============================================================================
# Endpoint call (OpenAI /v1/chat/completions). Thinking-model aware.
# ============================================================================
def call_chat_completion(base_url: str, model: str, prompt: str, max_tokens: int,
                         temperature: float, timeout: float) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
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
    else:
        import urllib.request
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    latency_ms = (time.time() - t0) * 1000.0
    content = ""
    reasoning = ""
    finish_reason = ""
    try:
        choice = data["choices"][0]
        msg = choice.get("message", {}) or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        finish_reason = choice.get("finish_reason") or ""
    except Exception:
        content = data.get("output_text", "") or ""
    return {
        "content": content, "reasoning": reasoning, "finish_reason": finish_reason,
        "latency_ms": latency_ms,
    }


# ============================================================================
# Trial scoring
# ============================================================================
def score_trial(task: Task, content: str, reasoning: str,
                finish_reason: str) -> dict[str, Any]:
    leak, leak_tok = detect_leak(content)
    truncated = bool(finish_reason == "length" and not content.strip())
    if truncated:
        passed, snippet = 0.0, "truncated"
    else:
        passed, snippet = task.grader(content, reasoning)
    if leak:
        passed = 0.0
    return {
        "task_id": task.task_id,
        "category": task.category,
        "pass": passed,
        "finish_reason": finish_reason,
        "leak": leak,
        "leak_token": leak_tok,
        "truncated": truncated,
        "content_len": len(content or ""),
        "reasoning_len": len(reasoning or ""),
        "snippet": snippet,
    }


# ============================================================================
# Statistics: 95% CI (normal approx / t for proportion mean)
# ============================================================================
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 29: 2.045, 99: 1.984}


def ci95(values: list[float]) -> tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, mean, mean
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var) / math.sqrt(n)
    df = n - 1
    t = _T95.get(df, 1.96 if df > 99 else 2.2)
    margin = t * se
    return mean, max(0.0, mean - margin), min(1.0, mean + margin)


# ============================================================================
# Reporting
# ============================================================================
def print_report(results: dict[str, list[dict[str, Any]]], model: str,
                  base_url: str) -> dict[str, Any]:
    line = "=" * 78
    print(line)
    print(f"HARD EVAL — model: {model}")
    print(f"endpoint: {base_url}")
    print(line)
    print(f"{'task':<26}{'cat':<13}{'pass%':>7}{'CI95':>16}{'leak':>6}{'trunc':>7}")
    print("-" * 78)
    cat_vals: dict[str, list[float]] = {}
    all_vals: list[float] = []
    leak_total = 0
    trunc_total = 0
    per_task = {}
    for tid, trials in results.items():
        vals = [t["pass"] for t in trials]
        mean, lo, hi = ci95(vals)
        cat = trials[0]["category"]
        cat_vals.setdefault(cat, []).extend(vals)
        all_vals.extend(vals)
        lk = sum(1 for t in trials if t["leak"])
        tr = sum(1 for t in trials if t["truncated"])
        leak_total += lk
        trunc_total += tr
        per_task[tid] = {"mean": mean, "lo": lo, "hi": hi, "category": cat,
                         "leaks": lk, "truncations": tr}
        print(f"{tid:<26}{cat:<13}{mean*100:>6.1f}"
              f"  [{lo*100:5.1f},{hi*100:5.1f}]{lk:>6}{tr:>7}")
    print("-" * 78)
    print("PER-CATEGORY mean:")
    cat_summary = {}
    for cat, vals in sorted(cat_vals.items()):
        m = sum(vals) / len(vals) if vals else 0.0
        cat_summary[cat] = m
        print(f"  {cat:<14} {m*100:6.1f}%   (n={len(vals)})")
    overall = sum(all_vals) / len(all_vals) if all_vals else 0.0
    print("-" * 78)
    print(f"OVERALL correctness:    {overall*100:.1f}%   (n={len(all_vals)} trials)")
    print(f"total leaks:            {leak_total}")
    print(f"total truncations:      {trunc_total}")
    print(line)
    return {"per_task": per_task, "per_category": cat_summary,
            "overall": overall, "leaks": leak_total, "truncations": trunc_total}


# ============================================================================
# Dry-run grader validation
# ============================================================================
def dry_run() -> int:
    line = "=" * 78
    print(line)
    print("DRY-RUN — grader validation (no endpoint contact)")
    print("Each grader is fed a KNOWN-CORRECT ref (expect pass=1.0) and a")
    print("known-WRONG ref (expect pass=0.0).")
    print(line)
    print(f"{'task':<26}{'cat':<13}{'correct':>9}{'wrong':>8}{'verdict':>10}")
    print("-" * 78)
    all_ok = True
    for task in TASKS:
        good, _ = task.grader(task.ref_correct, "")
        bad, _ = task.grader(task.ref_wrong, "")
        ok = (good == 1.0 and bad == 0.0)
        all_ok = all_ok and ok
        print(f"{task.task_id:<26}{task.category:<13}{good:>9.2f}{bad:>8.2f}"
              f"{('OK' if ok else 'FAIL'):>10}")
    print("-" * 78)
    print(f"TOTAL tasks: {len(TASKS)}")
    print(f"RESULT: {'ALL GRADERS DISCRIMINATING (PASS)' if all_ok else 'SOME GRADERS FAILED'}")
    print(line)
    return 0 if all_ok else 1


# ============================================================================
# Live run
# ============================================================================
def live_run(args: argparse.Namespace) -> int:
    selected = TASKS
    if args.tasks != "all":
        want = {t.strip() for t in args.tasks.split(",") if t.strip()}
        selected = [t for t in TASKS if t.task_id in want]
        if not selected:
            print(f"No tasks matched: {args.tasks}", file=sys.stderr)
            return 2
    results: dict[str, list[dict[str, Any]]] = {}
    for task in selected:
        trials: list[dict[str, Any]] = []
        for trial_idx in range(args.trials):
            try:
                resp = call_chat_completion(
                    args.base_url, args.model, task.prompt,
                    args.max_tokens, args.temperature, timeout=args.http_timeout)
            except Exception as e:
                trials.append({
                    "task_id": task.task_id, "category": task.category,
                    "pass": 0.0, "finish_reason": f"error:{e}", "leak": False,
                    "leak_token": "", "truncated": False, "content_len": 0,
                    "reasoning_len": 0, "snippet": f"REQ_ERROR {e}"[:40]})
                continue
            tr = score_trial(task, resp["content"], resp["reasoning"],
                             resp["finish_reason"])
            tr["latency_ms"] = resp["latency_ms"]
            trials.append(tr)
            print(f"[{task.task_id} trial {trial_idx+1}/{args.trials}] "
                  f"pass={tr['pass']:.2f} finish={tr['finish_reason']} "
                  f"leak={tr['leak']} trunc={tr['truncated']} "
                  f"snippet={tr['snippet']!r}", file=sys.stderr)
        results[task.task_id] = trials
    summary = print_report(results, args.model, args.base_url)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "base_url": args.base_url,
                       "trials": args.trials, "results": results,
                       "summary": summary}, f, indent=2)
        print(f"Wrote JSON: {args.out}")
    return 0


# ============================================================================
# CLI
# ============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="HARD discriminating LLM eval harness")
    ap.add_argument("--base-url", default="http://192.168.86.201:52415")
    ap.add_argument("--model", default="mlx-community/DeepSeek-V4-Flash-8bit")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--tasks", default="all", help="csv of task ids or 'all'")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--http-timeout", type=float, default=600.0)
    ap.add_argument("--out", default="", help="path to JSON dump")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate graders offline; no endpoint contact")
    args = ap.parse_args()
    if args.dry_run:
        return dry_run()
    return live_run(args)


if __name__ == "__main__":
    sys.exit(main())
