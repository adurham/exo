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


class Dsv41InvalidRequest(Dsv41Error):  # noqa: N818 - name is public API
    """The request's own input is invalid -- and the request's fault only.

    Raised for request-content validation failures: e.g. the literal image
    placeholder token typed inside message text, which the vendored DSv4
    encoder refuses because an image must arrive as its own content block.
    The engine fails ONLY that request (an error chunk carrying the reason)
    and moves on to the next queued task -- every rank raises the same
    refusal for the same params, so they stay in step, and no internal state
    is implicated. A failure that IS an engine bug must keep raising its own
    exception loudly instead: this class is for input, not for crashes.
    """


class Dsv41ConfigError(Dsv41Error):  # noqa: N818 - name is public API
    """A launch-time configuration value is invalid -- fail the boot.

    Raised at engine construction (e.g. an out-of-set ``DSV41_SPEC_GAMMA``).
    This is NOT request input and NOT a per-request refusal: it must stop the
    runner from starting rather than degrade any behaviour, so it is a distinct
    type from :class:`Dsv41InvalidRequest` (which the request path catches and
    recovers from by failing only that one request). A typo'd experiment arm
    has to fail loudly instead of silently running the default value.
    """


#: Message fragments that identify a ValueError raised while RENDERING or
#: EXPANDING a request as request-input validation rather than an engine bug.
#: Kept explicit (not a blanket ``except ValueError``) so a genuine internal
#: failure -- the resize solver's token-budget overflow, the embedding-table
#: guard, an MLX shape error -- still propagates and crashes loudly.
#:
#: The first three are the vendored DSv4 encoder's own request checks
#: (``vendor/deepseek_v4_encoding.py``); the rest come from expanding the
#: request's images in mlx-lm's DSv4 image processor. Both layers are vendored
#: upstream code this engine does not edit, so the messages themselves are the
#: stable interface; ``test_dsv41_engine``'s invalid-request tests pin them so
#: a vendored refresh that rewords one fails a test instead of silently
#: restoring the runner crash.
_INPUT_ERROR_MARKERS: tuple[str, ...] = (
    # encoder: placeholder token in `content` / `reasoning_content`
    "image special token",
    # encoder: placeholder token inside a text content block
    "Text block contains image placeholder",
    # encoder: an image block with no usable source
    "Image block does not contain a valid source",
    # image processor: placeholder count does not match the image list
    "image tokens but got",
    # image processor: the request's image payload cannot be read
    "Unsupported data URL encoding",
    "Cannot load image from record",
    "Invalid base64-encoded string",
    "Incorrect padding",
)


def reclassify_input_error(e: ValueError) -> None:
    """Raise :class:`Dsv41InvalidRequest` when ``e`` is request-input validation.

    Use at a render/expansion boundary::

        except ValueError as e:
            reclassify_input_error(e)  # input -> Dsv41InvalidRequest (from e)
            raise                      # anything else -> unchanged, crashes loudly

    The caller's fall-through ``raise`` is what keeps engine bugs loud; this
    function itself never returns silently for a recognized input error.
    """
    message = str(e)
    if any(marker in message for marker in _INPUT_ERROR_MARKERS):
        raise Dsv41InvalidRequest(message) from e
