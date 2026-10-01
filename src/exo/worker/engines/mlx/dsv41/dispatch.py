"""Which model card gets the DSv4.1 engine.

The decision is a CARD property, not a guess about the checkpoint: a card that
needs this engine declares it (``engine = "dsv41"`` in the TOML, see
``ModelCard.engine``), and everything else takes the normal MLX path. Guessing
from, say, ``quantization == "exl3"`` would silently route the NEXT EXL3 card
(any DeepSeek variant) into a builder whose loader speaks one checkpoint
layout; guessing from the model id would break the moment someone re-quantizes
or renames it. The card field is also what a reviewer reads to answer "which
engine serves this model?" without following three imports.

The loader still asserts the checkpoint's own ``model_type``
(``deepseek_v41``) at build time -- the card says which engine, the checkpoint
says whether it is actually the model that engine speaks.
"""

from __future__ import annotations

from exo.shared.models.model_cards import ModelCard
from exo.shared.types.worker.instances import BoundInstance
from exo.worker.engines.base import Builder

#: The ``engine`` value in a model card that selects this engine.
DSV41_ENGINE = "dsv41"


def is_dsv41_card(model_card: ModelCard) -> bool:
    """True when ``model_card`` asks for the DSv4.1 (EXL3) engine."""
    return model_card.engine == DSV41_ENGINE


def is_dsv41_instance(bound_instance: BoundInstance) -> bool:
    """True when the bound instance's card asks for the DSv4.1 engine."""
    return is_dsv41_card(bound_instance.bound_shard.model_card)


def make_mlx_family_builder(
    bound_instance: BoundInstance,
    *,
    event_sender,
    cancel_receiver,
) -> Builder:
    """The MLX-family builder for an instance: DSv4.1 if the card says so.

    Called from ``exo.worker.runner.bootstrap`` for every non-image instance,
    so this function IS the dispatch: one branch, on an explicit card field.
    ``Dsv41Builder`` subclasses ``MlxBuilder``, so the two branches differ only
    in load/build while sharing connect's MLX bring-up and RDMA probe.
    """
    shard = bound_instance.bound_shard
    if is_dsv41_card(shard.model_card):
        from exo.worker.engines.mlx.dsv41.builder import Dsv41Builder

        return Dsv41Builder(
            model_id=shard.model_card.model_id,
            event_sender=event_sender,
            cancel_receiver=cancel_receiver,
        )

    from exo.worker.engines.mlx.builder import MlxBuilder

    return MlxBuilder(
        model_id=shard.model_card.model_id,
        event_sender=event_sender,
        cancel_receiver=cancel_receiver,
    )
