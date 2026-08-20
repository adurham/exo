#!/Users/adam.durham/repos/exo/.venv/bin/python3
"""PHASE 0c: cross-rank collective ISSUE-ORDER determinism probe.

Question
--------
If each rank builds TWO INDEPENDENT subgraphs (chunk A and chunk B of a
sequence-chunk pipeline), each containing 3 `mx.distributed.all_sum` calls,
do the 6 collectives get matched across ranks in a deterministic order?
Is that order (a) program/issue order, (b) eval order, or (c) unspecified?

Method: TAGGED PAYLOADS.  Every collective carries a payload whose value
encodes (chunk, index).  Both ranks use the SAME tag for the same logical
collective, so a correct pairing yields exactly `size * tag`.  If rank 0's
k-th collective is matched with rank 1's j-th (j != k), the sum is
`tag_k + tag_j` -- provably wrong and identifiable back to the mispaired
partner.  A tag is a distinct prime-ish integer so any wrong sum is unique.

Run:
  .venv/bin/mlx.launch -n 2 --backend ring \
      bench/phase0c_collective_order_determinism.py

Scenario selected via P0C_SCEN:
  same_order      : both ranks issue A0,A1,A2,B0,B1,B2, eval at end (baseline)
  interleaved     : both ranks issue A0,B0,A1,B1,A2,B2 (true pipelined shape)
  async_eval_skew : SAME issue order on both ranks, but the ranks call
                    mx.async_eval on the two chunk subgraphs in OPPOSITE
                    order.  This is the realistic hazard in a pipelined
                    implementation: does eval order re-order the wire ops?
  issue_skew      : rank0 issues chunk A first, rank1 issues chunk B first
                    (deliberately divergent PROGRAM order).  Expected to
                    mispair or hang -- this is the negative control that
                    tells us whether matching is positional or tagged.

Every scenario writes a per-rank JSON result to $P0C_OUT/<scen>.rank<N>.json
so the two ranks' views can be compared offline (a mispairing is only
visible by diffing what each rank *expected* vs *got*).
"""

import json
import os
import sys
import time

import mlx.core as mx

_e = os.environ.get
SCEN = _e("P0C_SCEN", "same_order")
N = int(_e("P0C_N", "4096"))          # payload elements
TRIALS = int(_e("P0C_TRIALS", "20"))  # repeat to catch nondeterminism
OUT = _e("P0C_OUT", "/tmp/phase0c")

world = mx.distributed.init()
rank, size = world.rank(), world.size()
os.makedirs(OUT, exist_ok=True)


def log(*a):
    print(f"[r{rank}]", *a, flush=True)


# Distinct tags: chunk A -> 101,103,107 ; chunk B -> 211,223,227.
# All pairwise sums of distinct tags differ from every 2*tag, so any
# mispairing produces a value that identifies both partners.
TAGS = {"A": [101, 103, 107], "B": [211, 223, 227]}


def collective(tag: int):
    """One tagged all_sum. Correct pairing => every element == size*tag."""
    x = mx.full((N,), float(tag), dtype=mx.float32)
    return mx.distributed.all_sum(x, group=world)


def build_chunk(name: str):
    """Build one INDEPENDENT subgraph: 3 chained tagged collectives.

    Chained so the 3 are ordered *within* the chunk, but the two chunks
    share no data dependency at all -- exactly the Lever-2 shape.
    """
    outs = []
    carry = None
    for tag in TAGS[name]:
        y = collective(tag)
        if carry is not None:
            # data-dependency within the chunk only (value-preserving:
            # add 0 * previous so the expected value is unchanged)
            y = y + carry * 0.0
        carry = y
        outs.append(y)
    return outs


def check(name, outs):
    """Return per-collective observed value + whether it matches size*tag."""
    res = []
    for tag, y in zip(TAGS[name], outs):
        v = float(y[0].item())
        allsame = bool(mx.all(y == y[0]).item())
        res.append(
            {
                "tag": tag,
                "expected": size * tag,
                "observed": v,
                "uniform": allsame,
                "ok": abs(v - size * tag) < 1e-3 and allsame,
            }
        )
    return res


def run_trial(scen: str):
    if scen == "same_order":
        a = build_chunk("A")
        b = build_chunk("B")
        mx.eval(*a, *b)

    elif scen == "interleaved":
        # interleave the ISSUE of the two independent chunks
        outs = {"A": [], "B": []}
        carry = {"A": None, "B": None}
        for i in range(3):
            for name in ("A", "B"):
                y = collective(TAGS[name][i])
                if carry[name] is not None:
                    y = y + carry[name] * 0.0
                carry[name] = y
                outs[name].append(y)
        a, b = outs["A"], outs["B"]
        mx.eval(*a, *b)

    elif scen == "async_eval_skew":
        # IDENTICAL issue order on both ranks ...
        a = build_chunk("A")
        b = build_chunk("B")
        # ... but opposite async_eval order per rank.
        if rank % 2 == 0:
            mx.async_eval(*a)
            mx.async_eval(*b)
        else:
            mx.async_eval(*b)
            mx.async_eval(*a)
        mx.eval(*a, *b)

    elif scen == "eval_arg_skew":
        # IDENTICAL issue order; single blocking mx.eval, but the ARGUMENT
        # ORDER of that eval differs per rank.  Isolates "async_eval
        # scheduling" from "any eval-order difference".
        a = build_chunk("A")
        b = build_chunk("B")
        if rank % 2 == 0:
            mx.eval(*a, *b)
        else:
            mx.eval(*b, *a)

    elif scen == "async_eval_same":
        # POSITIVE control: async_eval used identically on both ranks.
        a = build_chunk("A")
        b = build_chunk("B")
        mx.async_eval(*a)
        mx.async_eval(*b)
        mx.eval(*a, *b)

    elif scen == "issue_skew":
        # DIVERGENT program order (negative control).
        if rank % 2 == 0:
            a = build_chunk("A")
            b = build_chunk("B")
        else:
            b = build_chunk("B")
            a = build_chunk("A")
        mx.eval(*a, *b)

    else:
        raise SystemExit(f"unknown scenario {scen}")

    return check("A", a) + check("B", b)


def main():
    log(f"scenario={SCEN} size={size} N={N} trials={TRIALS}")
    trials = []
    t0 = time.perf_counter()
    for t in range(TRIALS):
        r = run_trial(SCEN)
        trials.append(r)
        bad = [x for x in r if not x["ok"]]
        if bad:
            log(f"trial {t}: MISPAIR/CORRUPT -> {bad}")
    dt = time.perf_counter() - t0

    all_ok = all(x["ok"] for tr in trials for x in tr)
    # determinism across trials: is every trial's observed vector identical?
    sigs = {tuple(x["observed"] for x in tr) for tr in trials}
    out = {
        "scenario": SCEN,
        "rank": rank,
        "size": size,
        "trials": TRIALS,
        "all_ok": all_ok,
        "distinct_observed_signatures": len(sigs),
        "signatures": [list(s) for s in sigs],
        "seconds": dt,
        "detail_first_trial": trials[0],
    }
    path = os.path.join(OUT, f"{SCEN}.rank{rank}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    log(
        f"all_ok={all_ok} distinct_signatures={len(sigs)} "
        f"({dt:.2f}s) -> {path}"
    )
    if not all_ok:
        sys.exit(3)


main()
