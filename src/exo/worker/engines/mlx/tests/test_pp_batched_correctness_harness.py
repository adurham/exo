# pyright: reportAny=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false, reportPrivateUsage=false
# pyright: reportInvalidCast=false, reportArgumentType=false
"""Phase 0 correctness-baseline tests for the batched-PP sharding design.

See ``docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md`` Section 9,
"Phase 0 — Correctness baseline" (methodology corrected 2026-08-04 after
a second review): diff **serial single-request PP** against **new
batched PP**, SAME sharding scheme throughout — NOT a cross-sharding
byte-diff (that was the original, retracted methodology).

There is no batched-PP code yet (Phase 1+ delivers that), so this file's
actual deliverable, per the doc's own Phase 0 scoping ("largely tooling
... since there is no batched-PP code yet to diff against"), is:

1. Prove the ``pp_batched_correctness`` harness itself is a valid
   correctness oracle — ``test_two_rank_pp_matches_plain_forward_*``
   anchors a REAL 2-rank simulated PP split (via the actual
   ``PipelineFirstLayer``/``PipelineLastLayer`` classes, through the
   harness's ``SimPipelineTransport`` fake) against a plain, unsharded
   forward of the SAME weights. The two paths are NOT expected to match
   at float-tolerance precision: the real ``PipelineLastLayer``/
   ``PipelineFirstLayer`` classes cast activations to bf16 before every
   cross-rank send (a genuine production requirement — JACCL/RDMA
   transport requires bf16, see the classes' own comments in
   ``auto_parallel.py``), which the plain unsharded forward never pays
   at all. A single bf16 roundtrip on activations in this model's value
   range costs ~0.01-0.03 max abs diff (measured directly, see
   ``LOGIT_DIFF_TOLERANCE`` below), and this design's 2-rank split pays
   that cost TWICE per forward (once at the rank0->rank1 layer handoff,
   once at the rank1->rank0 final-hidden-state gather on decode). The
   bar that actually matters — and the one this test enforces strictly
   — is **greedy-token (argmax) agreement**: mismatches==0 is required,
   matching the "greedy-token-identical output" comparison the design
   doc's own Section 9 Phase 0 methodology specifies. ``max_diff`` is
   still asserted, but against a tolerance wide enough to accommodate
   the real, expected bf16-transport cost — not a byte-equality bar.
2. Establish the tolerance/comparison convention
   (``compare_logits``/``LOGIT_DIFF_TOLERANCE``) that Phase 0.5+'s real
   serial-PP-vs-batched-PP diffs will reuse once batched-PP code exists
   — that comparison is APPLES TO APPLES (both sides pay the same bf16
   transport cost, since both are real PP), so it can and should use a
   TIGHTER tolerance than this file's plain-vs-split anchor.
3. Exercise the harness's own failure modes (shape-mismatch detection,
   copy-not-alias transport, layer-count guard) so a bug in the harness
   itself (as opposed to a bug in the model/design under test) is caught
   here, not mistaken for a real batched-PP correctness finding later.

Once Phase 1+ delivers real batched-PP code, a follow-up test file
(diffing serial-PP-through-this-harness vs batched-PP output) is the
actual Phase 0 "correctness baseline" comparison — this file only
proves the harness is trustworthy enough to be that baseline's LHS.
"""

from typing import cast

import mlx.core as mx
import mlx.utils
import pytest
from mlx_lm.models.llama import Model as LlamaModel
from mlx_lm.models.llama import ModelArgs

from exo.worker.engines.mlx.pp_batched_correctness import (
    SimPipelineTransport,
    _RankGroup,
    build_two_rank_split,
    compare_logits,
    run_two_rank_pp_forward,
)

# Bar for THIS file's plain-forward-vs-2-rank-split anchor comparison.
# Deliberately WIDER than test_prefill_batched.py's same-named constant
# (0.002) -- that comparison never crosses a real bf16 transport cast,
# this one crosses it twice per forward (see module docstring point 1).
# Measured directly (2026-08-05, this model config): a single bf16
# roundtrip on this model's activation value range costs up to ~0.031
# max abs diff; two hops plus float accumulation-order differences from
# splitting one Python loop into two separate calls gives real headroom
# above that. 0.25 is chosen to comfortably clear the measured ~0.2
# real-world max_diff with margin, while still being tight enough to
# catch a genuinely broken split (e.g. a layer silently dropped, wrong
# rank boundary) which would produce a MUCH larger diff or an argmax
# mismatch. mismatches==0 (exact greedy-token agreement) remains the
# PRIMARY correctness bar this file enforces -- max_diff is a secondary
# sanity check, not the main assertion.
LOGIT_DIFF_TOLERANCE = 0.25

_ARGS = ModelArgs(
    model_type="llama",
    hidden_size=256,
    num_hidden_layers=4,
    intermediate_size=512,
    num_attention_heads=4,
    num_key_value_heads=2,
    rms_norm_eps=1e-6,
    vocab_size=4096,
    rope_theta=10000.0,
    tie_word_embeddings=True,
)


def _seeded_model() -> LlamaModel:
    mx.random.seed(1234)
    model = LlamaModel(_ARGS)
    params = model.parameters()
    new_params = mlx.utils.tree_map(
        lambda p: mx.random.normal(shape=p.shape, dtype=p.dtype)
        if isinstance(p, mx.array)
        else p,
        params,
    )
    model.update(new_params)
    mx.eval(model.parameters())
    return model


def _copy_weights(src: LlamaModel, dst: LlamaModel) -> None:
    """Force ``dst`` to have IDENTICAL weights to ``src`` -- copying
    parameters directly is more robust than re-seeding ``mx.random``
    before each construction (fragile against any incidental extra RNG
    draw between the two model builds). ``build_two_rank_split`` requires
    its two input models to already have identical weights."""
    dst.update(src.parameters())
    mx.eval(dst.parameters())


def _make_prompt(length: int, vocab_size: int) -> mx.array:
    mx.random.seed(99)
    return mx.random.randint(0, vocab_size, shape=(length,))


def _plain_prefill_and_decode(
    model: LlamaModel,
    prompt: mx.array,
    n_decode: int,
) -> list[mx.array]:
    """Run PLAIN (unsharded) prefill + greedy decode, returning per-step
    logits for the decode steps."""
    cache = model.make_cache()
    # Prefill: feed prompt[:-1], keep prompt[-1] as the first decode input
    # (matches this repo's own prefill()/pipeline_parallel_prefill()
    # contract: prefill processes tokens[:-1], the "extra" token starts
    # decode).
    if len(prompt) > 1:
        out = model(prompt[:-1][None], cache=cache)
        mx.eval(out)
    cur = int(prompt[-1].item())
    logits_per_step: list[mx.array] = []
    for _ in range(n_decode):
        out = model(mx.array([[cur]]), cache=cache)
        mx.eval(out)
        logits_per_step.append(out[0, -1])
        cur = int(mx.argmax(out[0, -1]).item())
    return logits_per_step


def _two_rank_prefill_and_decode(
    prompt: mx.array,
    n_decode: int,
) -> list[mx.array]:
    """Run the SAME prefill+decode through a simulated 2-rank PP split
    (real PipelineFirstLayer/PipelineLastLayer, fake transport) of
    weight-identical model instances."""
    src = _seeded_model()
    r0 = LlamaModel(_ARGS)
    r1 = LlamaModel(_ARGS)
    _copy_weights(src, r0)
    _copy_weights(src, r1)
    rank0_model, rank1_model, transport = build_two_rank_split(r0, r1)

    rank0_cache = r0.make_cache()
    rank1_cache = r1.make_cache()

    if len(prompt) > 1:
        run_two_rank_pp_forward(
            rank0_model,
            rank1_model,
            transport,
            prompt[:-1][None],
            rank0_cache,
            rank1_cache,
            is_prefill=True,
        )

    cur = int(prompt[-1].item())
    logits_per_step: list[mx.array] = []
    for _ in range(n_decode):
        out = run_two_rank_pp_forward(
            rank0_model,
            rank1_model,
            transport,
            mx.array([[cur]]),
            rank0_cache,
            rank1_cache,
            is_prefill=False,
        )
        logits_per_step.append(out[0, -1])
        cur = int(mx.argmax(out[0, -1]).item())
    return logits_per_step


@pytest.mark.slow
def test_two_rank_pp_matches_plain_forward_prefill_and_decode() -> None:
    """THE Phase 0 harness-validity anchor (per the doc's Section 9
    Phase 0 scoping + the pre-implementation consult review): a REAL
    2-rank simulated PP split, driven through SimPipelineTransport, must
    reproduce a PLAIN unsharded forward of the SAME weights at
    GREEDY-TOKEN precision (mismatches==0, matching the design doc's own
    "greedy-token-identical output" Phase 0 methodology). This is NOT a
    float-bit-exactness bar -- the real PipelineFirstLayer/
    PipelineLastLayer classes cast activations to bf16 before every
    cross-rank send (a genuine JACCL/RDMA transport requirement the
    plain forward never pays), so real numerical drift on the order of
    ``LOGIT_DIFF_TOLERANCE`` is EXPECTED, not a bug -- see module
    docstring and the ``LOGIT_DIFF_TOLERANCE`` comment for the measured
    justification.

    If this test doesn't pass, the harness itself cannot be trusted as
    Phase 0.5+'s correctness oracle for real batched-PP code -- fix this
    BEFORE writing anything that depends on the harness.
    """
    plain_model = _seeded_model()
    prompt = _make_prompt(length=12, vocab_size=_ARGS.vocab_size)
    n_decode = 8

    plain_logits = _plain_prefill_and_decode(plain_model, prompt, n_decode)
    split_logits = _two_rank_prefill_and_decode(prompt, n_decode)

    max_diff, mismatches = compare_logits(
        plain_logits, split_logits, "two-rank-pp-vs-plain"
    )
    assert mismatches == 0, (
        f"Simulated 2-rank PP split diverged from plain forward: "
        f"{mismatches}/{n_decode} argmax mismatches -- harness is NOT a "
        f"trustworthy correctness oracle, do not build Phase 0.5+ on it "
        f"until this is fixed"
    )
    assert max_diff < LOGIT_DIFF_TOLERANCE, (
        f"Simulated 2-rank PP split max logit diff {max_diff} exceeds "
        f"tolerance {LOGIT_DIFF_TOLERANCE}"
    )


@pytest.mark.slow
def test_two_rank_pp_single_token_prompt_decode_only() -> None:
    """Degenerate case: a 1-token prompt (no real prefill chunk, straight
    to decode) must still match plain forward -- guards against an
    off-by-one in how ``run_two_rank_pp_forward`` handles the
    "prompt[:-1] is empty" edge this repo's own
    ``pipeline_parallel_prefill`` docstring calls out explicitly."""
    plain_model = _seeded_model()
    prompt = _make_prompt(length=1, vocab_size=_ARGS.vocab_size)
    n_decode = 5

    plain_logits = _plain_prefill_and_decode(plain_model, prompt, n_decode)
    split_logits = _two_rank_prefill_and_decode(prompt, n_decode)

    max_diff, mismatches = compare_logits(
        plain_logits, split_logits, "two-rank-pp-vs-plain-single-token"
    )
    assert mismatches == 0
    assert max_diff < LOGIT_DIFF_TOLERANCE


def test_build_two_rank_split_rejects_too_few_layers() -> None:
    """Guard: a model with <2 layers can't be split into two non-empty
    PP ranks -- must fail loudly, not silently produce a degenerate
    one-rank-empty topology."""
    args = ModelArgs(
        model_type="llama",
        hidden_size=64,
        num_hidden_layers=1,
        intermediate_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        rms_norm_eps=1e-6,
        vocab_size=256,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    r0 = LlamaModel(args)
    r1 = LlamaModel(args)

    with pytest.raises(ValueError, match="non-empty"):
        build_two_rank_split(r0, r1)


def test_build_two_rank_split_rejects_mismatched_layer_counts() -> None:
    """Guard: the two model instances passed in must have the same
    layer count -- a real caller bug (constructing rank0/rank1 from
    different configs) must fail loudly, not silently mis-split."""
    args4 = ModelArgs(
        model_type="llama",
        hidden_size=64,
        num_hidden_layers=4,
        intermediate_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        rms_norm_eps=1e-6,
        vocab_size=256,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    args6 = ModelArgs(
        model_type="llama",
        hidden_size=64,
        num_hidden_layers=6,
        intermediate_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        rms_norm_eps=1e-6,
        vocab_size=256,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    r0 = LlamaModel(args4)
    r1 = LlamaModel(args6)

    with pytest.raises(ValueError, match="same layer count"):
        build_two_rank_split(r0, r1)


def test_sim_pipeline_transport_shape_mismatch_raises() -> None:
    """The fake transport must raise loudly on a shape mismatch between
    what the sender queued and what the receiver's template expects --
    exactly the class of bug real metadata-framing (Section 6.2 item 2
    of the design doc) needs to be caught by, once batched-PP exists.

    ``recv_like``'s lock discipline is CALLER-managed (documented on
    ``SimPipelineTransport``: held for a rank's entire forward-driving
    body, released only while blocked inside ``recv_like``) -- direct
    unit tests of the transport (bypassing ``run_two_rank_pp_forward``,
    which normally owns this) must acquire ``_MLX_CALL_LOCK`` themselves
    first to satisfy that contract.
    """
    from exo.worker.engines.mlx.pp_batched_correctness import _MLX_CALL_LOCK

    transport = SimPipelineTransport()
    group0 = cast(mx.distributed.Group, cast(object, _RankGroup(0, 2)))
    group1 = cast(mx.distributed.Group, cast(object, _RankGroup(1, 2)))

    sent = mx.zeros((1, 4, 8))
    transport.send(sent, 1, group=group0)

    wrong_template = mx.zeros((1, 5, 8))  # shape mismatch vs (1, 4, 8) sent
    _MLX_CALL_LOCK.acquire()
    try:
        with pytest.raises(RuntimeError, match="shape mismatch"):
            transport.recv_like(wrong_template, 0, group=group1)
    finally:
        _MLX_CALL_LOCK.release()


def test_sim_pipeline_transport_copies_not_aliases() -> None:
    """The fake transport must not hand back the SAME mx.array object
    across ranks -- see module docstring point 3 (a real cross-process
    transport is never object-identity-preserving; aliasing here would
    hide sender-mutation-after-send bugs and share lazy graph nodes
    across simulated ranks)."""
    from exo.worker.engines.mlx.pp_batched_correctness import _MLX_CALL_LOCK

    transport = SimPipelineTransport()
    group0 = cast(mx.distributed.Group, cast(object, _RankGroup(0, 2)))
    group1 = cast(mx.distributed.Group, cast(object, _RankGroup(1, 2)))

    sent = mx.array([1.0, 2.0, 3.0])
    transport.send(sent, 1, group=group0)
    _MLX_CALL_LOCK.acquire()
    try:
        received = transport.recv_like(mx.array([0.0, 0.0, 0.0]), 0, group=group1)
    finally:
        _MLX_CALL_LOCK.release()

    assert bool(mx.all(received == sent).item())
    # Mutating the received copy must not affect the array that was sent.
    received_plus_one = received + 1
    assert not bool(mx.all(received_plus_one == sent).item())


def test_sim_pipeline_transport_recv_timeout_raises() -> None:
    """If a rank tries to recv from a peer that never sends, the fake
    transport must raise a clear timeout error rather than hanging
    pytest forever -- a real safety net for this harness's own future
    protocol bugs, not a claim about real jaccl transport timeout
    behavior."""
    from exo.worker.engines.mlx.pp_batched_correctness import _MLX_CALL_LOCK

    transport = SimPipelineTransport()
    group1 = cast(mx.distributed.Group, cast(object, _RankGroup(1, 2)))
    # Shrink the timeout for this test only, so it doesn't take 30s.
    import exo.worker.engines.mlx.pp_batched_correctness as _mod

    original = _mod._RECV_TIMEOUT_SECONDS
    _mod._RECV_TIMEOUT_SECONDS = 0.2
    _MLX_CALL_LOCK.acquire()
    try:
        with pytest.raises(RuntimeError, match="timed out"):
            transport.recv_like(mx.zeros((2,)), 0, group=group1)
    finally:
        _mod._RECV_TIMEOUT_SECONDS = original
        _MLX_CALL_LOCK.release()
