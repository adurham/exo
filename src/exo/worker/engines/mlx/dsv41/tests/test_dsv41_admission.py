"""The DSv4.1 cache-capacity admission contract, pinned.

WHAT HAPPENED IN PRODUCTION (the bug report this file pins): a client sized a
RAW prompt to the instance's advertised 16384-token context, the chat template
added 4 tokens, and the engine refused with "prompt 16388 + max_output_tokens 8
needs more than the 16384-token cache". Investigated 2026-10-02 (wp8):

* The refusal is CORRECT, not a bug. ``prompt_len`` is the POST-template count
  (``encode_prompt`` of the templated string, engine.py) -- the rows the
  prefill will actually write into a cache ``model.make_cache`` preallocated
  at exactly ``capacity`` rows. 16388 rows cannot prefill into 16384 rows at
  ANY max_output_tokens; the fork's own ``SessionCache._check_capacity``
  would raise the same boundary as ``CapacityError``.
* The +8 slack (``ADMISSION_SLACK``) covers what a decode round writes past
  the committed prefix before its rollback trims it back: the anchor plus up
  to ``gamma`` verified drafts in one forward (adaptive gamma tops out at 4,
  so <= 5 rows).
* The API layer advertises the CARD's context_length (the checkpoint's 1M
  here) and validates nothing against it; the per-INSTANCE cap
  (``max_kv_tokens``, 16384 in production) is what the engine enforces. There
  is no second check to be inconsistent with. A client that sizes its RAW
  prompt to any advertised context is still expected to leave room for the
  template + output, exactly as with vLLM/sglang/Ollama -- that convention is
  NOT changed here (no cap value was touched).

These tests pin that contract on ``rounds._admit``, the pure decision function
the engine's ``_generate`` delegates to (same boundary arithmetic, extracted
so it is testable without mlx or a checkpoint, and so the engine cannot Drift
from the tested rule without failing here).
"""

from __future__ import annotations

from exo.worker.engines.mlx.dsv41.rounds import (
    ADMISSION_SLACK,
    _admit,  # pyright: ignore[reportPrivateUsage]
)

#: The production shape: a DeepSeek-V4.1 instance capped at 16K KV tokens
#: (start_cluster.sh: DSV4_MAX_KV_TOKENS=16384), MAX_TOKENS fallback of 32168
#: (worker/engines/mlx/constants.py).
CAPACITY = 16384
MAX_TOKENS_DEFAULT = 32168


def admitted(capacity: int, prompt_len: int, max_output_tokens: int | None) -> int:
    """``_admit``'s accepted branch, asserted."""
    max_tokens, refusal = _admit(
        capacity=capacity,
        prompt_len=prompt_len,
        max_output_tokens=max_output_tokens,
        max_tokens_default=MAX_TOKENS_DEFAULT,
    )
    assert refusal is None and max_tokens is not None, refusal
    return max_tokens


def refused(capacity: int, prompt_len: int, max_output_tokens: int | None) -> str:
    """``_admit``'s refused branch, asserted; returns the message."""
    max_tokens, refusal = _admit(
        capacity=capacity,
        prompt_len=prompt_len,
        max_output_tokens=max_output_tokens,
        max_tokens_default=MAX_TOKENS_DEFAULT,
    )
    assert max_tokens is None and refusal is not None, (max_tokens, refusal)
    return refusal


# ------------------------------------------------- the production boundary


def test_post_template_prompt_one_over_the_cache_is_refused():
    """16388 templated rows into a 16384 cache: refused.

    The reported production case (with max_output_tokens=8). Every term of the
    comparison shown: prompt (post-template), the configured capacity. The
    template overhead must NOT be discounted -- the prefill writes 16388 rows,
    so the message is the prompt-length refusal, naming the capacity knob.
    """
    msg = refused(capacity=CAPACITY, prompt_len=16388, max_output_tokens=8)
    assert "prompt 16388" in msg
    assert f"longer than the {CAPACITY}-token cache" in msg
    assert "cannot be prefilled" in msg


def test_prompt_equal_to_the_cache_is_refused_even_with_zero_output():
    """16384 == capacity is still over: zero room for anchor + decode rows."""
    refused(capacity=CAPACITY, prompt_len=16384, max_output_tokens=8)


def test_largest_admitted_prompt_is_capacity_minus_slack_minus_output():
    """16384 - 8 - 8 = 16368 templated prompt rows with max_new 8: admitted."""
    max_tokens = admitted(capacity=CAPACITY, prompt_len=16368, max_output_tokens=8)
    assert max_tokens == 8


def test_one_row_over_the_boundary_is_refused():
    """The boundary is exact: 16369 templated rows with max_new 8 is refused."""
    msg = refused(capacity=CAPACITY, prompt_len=16369, max_output_tokens=8)
    assert "max_output_tokens 8" in msg


# ------------------------------------------------------- the clamp branch


def test_no_client_cap_clamps_output_to_what_the_cache_holds():
    """max_output_tokens=None: admitted with the output clamped, not refused.

    The dashboard sends no max_tokens; the 32168 default must collapse to the
    cache's room instead of refusing the request (the behaviour the no-max-tokens
    engine test pins end to end).
    """
    max_tokens = admitted(capacity=CAPACITY, prompt_len=16368, max_output_tokens=None)
    assert max_tokens == CAPACITY - 16368 - ADMISSION_SLACK
    assert max_tokens >= 1


def test_no_client_cap_with_an_overfull_prompt_is_refused():
    """The clamp cannot rescue a prompt that fills the cache past the slack."""
    prompt = CAPACITY - ADMISSION_SLACK
    msg = refused(capacity=CAPACITY, prompt_len=prompt, max_output_tokens=None)
    assert f"prompt {prompt}" in msg
    assert f"{CAPACITY}-token cache" in msg
    assert "-0" not in msg and "- -" not in msg  # no leaked negative arithmetic


def test_no_client_cap_on_a_prompt_longer_than_the_cache_is_refused():
    """prompt > capacity with no output cap: refused, no negative cap in the text."""
    msg = refused(capacity=CAPACITY, prompt_len=CAPACITY + 1, max_output_tokens=None)
    assert "prompt 16385" in msg
    assert "leaves no room to generate" in msg
    assert "-0" not in msg and "- -" not in msg  # no leaked negative arithmetic
    assert f"{CAPACITY}-token cache" in msg


# --------------------------------------------------- the oversized-output branch


def test_explicit_output_cap_over_the_cache_is_refused():
    """A small cache with a big explicit max_tokens: refused, message honest."""
    msg = refused(capacity=256, prompt_len=3, max_output_tokens=32168)
    assert "prompt 3" in msg
    assert "max_output_tokens 32168" in msg
    assert "more than the 256-token cache" in msg


def test_slack_is_part_of_the_contract_not_the_message():
    """One over the REAL boundary refused, AT the real boundary admitted.

    Guards the slack against accidental shrinking or growth: the acceptance
    surface is exactly ``prompt + max_output <= capacity - ADMISSION_SLACK``.
    """
    prompt = 100
    out = 50
    cap = prompt + out + ADMISSION_SLACK
    assert admitted(capacity=cap, prompt_len=prompt, max_output_tokens=out) == out
    refused(capacity=cap, prompt_len=prompt + 1, max_output_tokens=out)


def test_admit_matches_the_engine_predicate_everywhere():
    """``_admit``'s accept/refuse decision == the original inline predicate.

    The engine's rule (the one that produced the production refusal) was:

        max_tokens = max_output or MAX_TOKENS_DEFAULT
        if max_output is None: max_tokens = min(max_tokens, cap - prompt - 8)
        refuse if max_tokens < 1 or prompt + max_tokens + 8 > capacity

    This sweep proves the extraction changed the DECISION on no point of the
    space (cap x prompt x output) -- only the refusal TEXT became exact. The
    original engine code remains the oracle in the loop below.
    """
    for cap in (256, 16384, 1_048_576):
        for prompt in (0, 1, 3, 100, cap - 9, cap - 8, cap - 1, cap, cap + 1):
            for out in (None, 1, 8, 100):
                # The original predicate, inline (the oracle).
                max_tokens = out or MAX_TOKENS_DEFAULT
                if out is None:
                    max_tokens = min(max_tokens, cap - prompt - ADMISSION_SLACK)
                original_refuses = max_tokens < 1 or (
                    prompt + max_tokens + ADMISSION_SLACK > cap
                )
                try:
                    admitted(capacity=cap, prompt_len=prompt, max_output_tokens=out)
                    decided_refuses = False
                except AssertionError:
                    decided_refuses = True
                assert decided_refuses == original_refuses, (
                    f"_admit(cap={cap}, prompt={prompt}, out={out}) != original"
                )


# ------------------------------------------------- the engine uses this rule


def test_engine_wires_the_helper_into_its_generate_path():
    """``_generate`` must delegate to ``_admit`` (no second inline rule).

    A second, inline copy of the boundary arithmetic in engine.py is exactly
    the drift this file exists to prevent: if someone edits the engine's check
    without the helper, these tests keep passing while production changes.
    Source-level pin (the helpers are imported into the engine namespace and
    used at the one admission site).
    """
    import inspect

    import exo.worker.engines.mlx.dsv41.engine as engine_module

    source = inspect.getsource(engine_module.Dsv41Engine._generate)  # pyright: ignore[reportPrivateUsage]
    assert "_admit(" in source, (
        "Dsv41Engine._generate must route admission through rounds._admit"
    )
    # No second inline comparison left behind in the engine.
    assert "prompt_len + max_tokens" not in source, (
        "an inline capacity comparison in _generate is the drift this pin prevents"
    )
    assert engine_module._admit is _admit  # pyright: ignore[reportPrivateUsage]
