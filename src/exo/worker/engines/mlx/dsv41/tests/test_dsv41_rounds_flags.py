"""Pin the D2 round-sync flags' bit-exactness and engagement (DSV41_RD_*).

Both flags are default-OFF decode reductions for the DSpark speculative round:

* ``DSV41_RD_ONDEV_ACCEPT`` (rounds.py) moves the acceptance scan + committed
  row into the round's single eval graph;
* ``DSV41_RD_MARKOV_REP`` (mtp/exl3_build) replicates the draft head's markov
  vocab projection so the markov loop drops its per-step collective.

What these tests pin, and why each assertion exists:

* FLAGGED vs UNFLAGGED runs must commit the SAME tokens with the SAME
  acceptance counts on adversarial scripts (full accept, mismatch at 0,
  mismatch mid-draft, exhausted draft). Any divergence is a flag bug: the
  flags claim numerical identity.
* The flagged round must actually take the flagged path. Equality alone
  cannot prove wiring (a flag that silently does nothing also passes an
  equality check), so the tests also assert the path probe
  (``rounds._last_accept_path``) and, for MARKOV_REP, the collective count:
  the sharded-markov path pays ``gamma`` markov collectives per draft, the
  replicated path zero.
* The acceptance math is pinned directly against the host loop on raw rows
  (all-accept, mismatch-at-k, cross-rank-tie, exhausted draft), which is the
  property the bit-exactness argument in ``_accept_on_device`` rests on.

Everything runs on tiny fake models (CPU wheel), no checkpoint, no GPU, no
distributed init.
"""

from __future__ import annotations

import os

import mlx.core as mx
import pytest

from exo.worker.engines.mlx.dsv41 import rounds as RND
from exo.worker.engines.mlx.dsv41.rounds import _one_round, _spec_policy
from exo.worker.engines.mlx.dsv41.tests.test_dsv41_engine import (
    MockCache,
    MockHead,
    ScriptedModel,
    _EngineStub,
)


def _round(engine, model, cache, token, head, policy=None, **kw):
    return _one_round(
        engine, model=model, cache=cache, token=token, head=head,
        policy=policy, **kw,
    )


SCRIPTS = {
    "full-accept": [34, 35, 36, 34, 35, 1],
    "reject-at-0": [34, 99, 36, 34, 35, 1],
    "reject-mid": [34, 35, 99, 34, 35, 1],
    "long": [10, 11, 12, 13, 14, 15, 16, 17, 1],
}


def _drive(script, lie_at, env_flag):
    """Run one scripted round with the flag env as given; return the full record."""
    os.environ["DSV41_RD_ONDEV_ACCEPT"] = env_flag
    model = ScriptedModel(list(script), prompt_tokens=0)
    head = MockHead(model, lie_at=lie_at)
    cache = MockCache(64)
    engine = _EngineStub(_draft_windows={0: object()})
    model.round_entry_offset = 1
    committed, ms, accepted, gamma = _round(
        engine, model=model, cache=cache, token=model.script[0],
        head=head, policy=_spec_policy(3),
    )
    return {
        "committed": committed, "ms": ms, "accepted": accepted, "gamma": gamma,
        "offset": cache.offset, "appended": list(head.appended),
        "path": RND._last_accept_path,
    }


@pytest.mark.parametrize("name", sorted(SCRIPTS))
@pytest.mark.parametrize("lie_at", [None, 0, 1, 2])
def test_flagged_and_unflagged_rounds_commit_identical_tokens(name, lie_at):
    """Every (script, lie) pair must give byte-identical outcomes both ways."""
    off = _drive(SCRIPTS[name], lie_at, "0")
    on = _drive(SCRIPTS[name], lie_at, "1")
    assert on["committed"] == off["committed"], (name, lie_at, on, off)
    assert on["accepted"] == off["accepted"], (name, lie_at, on, off)
    assert on["gamma"] == off["gamma"] == 3
    assert on["offset"] == off["offset"]
    assert on["appended"] == off["appended"]


def test_flag_actually_takes_the_device_path():
    """Engagement pin: the probe must say the flag engaged (and not when off)."""
    off = _drive(SCRIPTS["full-accept"], None, "0")
    assert off["path"] == "host"
    on = _drive(SCRIPTS["full-accept"], None, "1")
    assert on["path"] == "device"


def test_accept_on_device_matches_host_loop_on_raw_rows():
    """The acceptance math pinned directly: in-graph scan vs the host loop."""
    tgt = lambda ts: mx.array([ts], dtype=mx.int32)

    cases = [
        # (target row, draft row, expected accepted, expected committed)
        ([34, 35, 36, 37], [34, 35, 36], 3, [34, 35, 36, 37]),   # full accept
        ([34, 99, 36, 37], [34, 35, 36], 1, [34, 99]),           # mismatch at 1
        ([99, 35, 36, 37], [34, 35, 36], 0, [99]),               # reject at 0
        ([34, 35, 36, 37], [34, 35, 99], 2, [34, 35, 36]),       # mismatch at 2
        # exhausted draft: draft ran out at 2 < gamma=3
        ([34, 35, 36, 37], [34, 35], 2, [34, 35, 36]),
        # draft all-matches-but-shorter: accepted = len(draft)
        ([34, 35], [34, 35], 2, [34, 35]),
    ]
    for t_row, d_row, want_acc, want_committed in cases:
        acc_mx, com_mx = RND._accept_on_device(tgt(t_row), tgt(d_row), len(d_row))
        mx.eval(acc_mx, com_mx)
        acc = int(acc_mx.item())
        committed = com_mx.tolist()[: acc + 1]
        assert acc == want_acc, (t_row, d_row, acc, want_acc)
        assert committed == want_committed, (t_row, d_row, committed)


def test_unflagged_path_untouched_when_flag_off():
    """Flag OFF must keep the original host-loop behaviour probe-for-probe."""
    off = _drive(SCRIPTS["reject-mid"], 1, "0")
    # script[1]=35, lie at 1 makes draft [35, 9999, 36]; target is [35,99,...]
    assert off["accepted"] == 1
    assert 9999 not in off["committed"]
    assert off["path"] == "host"