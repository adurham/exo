"""Error types for the DSv4.1 (EXL3) engine.

Kept in their own module so both the loader and the engine can raise/import
them without a cycle, and so callers (the runner supervisor's error path, the
integration harness, tests) can distinguish "this placement/card is not
something DSv4.1 supports" from a genuine bug.
"""

from __future__ import annotations


class Dsv41Error(Exception):
    """Base class for DSv4.1 engine errors."""


class Dsv41UnsupportedPlacement(Dsv41Error):  # noqa: N818 - name is public API
    """The instance's shard metadata cannot be served by this engine.

    Today that means: a Pipeline shard spanning more than one rank. DSv4.1
    layers are wired through per-forward shared state AND cross-layer caches
    (``kv_source_layers`` publishing a compressed-KV buffer that later layers
    read, ``index_source_layers`` publishing top-k selections, the engram
    token-id history), which cannot be split across a layer range boundary
    without reimplementing that hand-off over the wire. Tensor (rank/world)
    sharding is supported -- it is built INTO the loader (see
    ``exo.worker.engines.mlx.dsv41.load``), not delegated to exo's
    ``tensor_auto_parallel``.
    """


class Dsv41UnsupportedFeature(Dsv41Error):  # noqa: N818 - name is public API
    """A requested feature is not wired for DSv4.1 yet.

    Raised loudly rather than silently degrading: prefix-cache reuse and
    disaggregated (remote) prefill are both real exo features whose DSv4.1
    implementations are owned by other workstreams. Claiming to honour a
    request we cannot honour would corrupt results invisibly, which is the
    failure mode this engine deliberately refuses.
    """
