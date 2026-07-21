# pyright: reportPrivateUsage=false
import mlx.core as mx


def _make_layer_pair():
    """Build a minimal SpecPipelineFirstLayer/SpecPipelineLastLayer pair
    with no real model/group -- enough to exercise _configure_layers'
    flag-reset behavior without touching distributed machinery or a
    real forward pass (this test never calls __call__)."""
    from exo.worker.engines.mlx.auto_parallel import (
        PipelineFirstLayer,
        PipelineLastLayer,
    )
    from exo.worker.engines.mlx.pp_speculation import (
        SpecPipelineFirstLayer,
        SpecPipelineLastLayer,
    )

    def _identity_layer(x: mx.array, *a: object, **kw: object) -> mx.array:
        return x

    base_first = PipelineFirstLayer(_identity_layer, r=1, group=None)  # pyright: ignore[reportArgumentType]
    base_last = PipelineLastLayer(_identity_layer, r=0, s=2, group=None)  # pyright: ignore[reportArgumentType]
    spec_first = SpecPipelineFirstLayer(base_first)
    spec_last = SpecPipelineLastLayer(base_last)
    return spec_first, spec_last


def test_configure_layers_resets_dirty_flags_to_defaults() -> None:
    """Regression test for the 2026-07-20 jaccl transport-fault fix.

    ROOT CAUSE: SpecPipelineFirstLayer/SpecPipelineLastLayer are the
    SAME persistent layer objects across every request once installed
    (_install_spec_layers' isinstance guard never re-wraps them) --
    their _pp_recv/_pp_send/_speculative mode flags are mutable
    instance state that can outlive the request whose decode loop set
    them, if that decode loop's generator `finally: _configure_layers(
    ...)` cleanup never ran (e.g. an exception/disconnect path that
    bypasses _close_pp_spec_gen()). Live-traced: a subsequent request's
    plain (non-speculative) first-token stream_generate() call ran
    with rank1's SpecPipelineFirstLayer still armed _pp_recv=True from
    a PRIOR request's decode loop -- an asymmetric mode mismatch
    against rank0 (correctly _pp_send=False) that caused a real jaccl
    transport fault (`[jaccl] recv() deadline in drain`).

    This test simulates that "dirty" state directly (no cluster/
    distributed group needed) and confirms `_configure_layers` called
    with no kwargs (as batch_generate.py's _submit_pp_spec now does,
    defensively, before any new request's first-token call) resets
    every mode flag to its default (all False / None).
    """
    from exo.worker.engines.mlx.pp_speculation import _configure_layers

    spec_first, spec_last = _make_layer_pair()

    # Simulate the dirty state left behind by a decode loop whose
    # cleanup never ran: rank1's recv armed, rank0's send/decode/
    # speculative modes armed, with a stashed state_list/hidden_idx.
    spec_first._pp_recv = True
    spec_last._pp_send = True
    spec_last._pp_decode = True
    spec_last._speculative = True
    spec_last._state_list = [mx.zeros(1)]
    spec_last._hidden_idx = 0

    _configure_layers(spec_first, spec_last)

    assert spec_first._pp_recv is False, (
        "a fresh request's first-token call must not inherit a stale "
        "_pp_recv=True from a previous request's unclosed decode loop"
    )
    assert spec_last._pp_send is False
    assert spec_last._pp_decode is False
    assert spec_last._speculative is False
    assert spec_last._state_list is None
    assert spec_last._hidden_idx == -1


def test_configure_layers_tolerates_none_layers() -> None:
    """_configure_layers must no-op cleanly when either layer is None
    (e.g. a non-PP model, or a model where _install_spec_layers found
    no matching layer) -- this is the call shape batch_generate.py's
    defensive reset uses unconditionally, so it must never raise even
    when spec_first/spec_last come back None."""
    from exo.worker.engines.mlx.pp_speculation import _configure_layers

    _configure_layers(None, None)  # must not raise
