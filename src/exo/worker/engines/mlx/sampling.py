"""Five-tier sampling default resolution: request → instance → card → cluster → hardcoded.

Each tier supplies optional values; the first non-None wins. Hardcoded defaults
preserve historical Exo behavior when no other tier is set.

Covers both `make_sampler` fields (temp, top_p, top_k, min_p) and
`make_logits_processors` fields (presence_penalty, repetition_penalty,
frequency_penalty).
"""
import os
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from exo.shared.models.model_cards import SamplingValues
    from exo.shared.types.common import ModelId

_T = TypeVar("_T")

# --- Per-cluster defaults (env vars). None = fall through to hardcoded. ---
CLUSTER_DEFAULT_TEMPERATURE: float | None = (
    float(os.environ["EXO_DEFAULT_TEMPERATURE"])
    if os.environ.get("EXO_DEFAULT_TEMPERATURE") else None
)
CLUSTER_DEFAULT_TOP_P: float | None = (
    float(os.environ["EXO_DEFAULT_TOP_P"])
    if os.environ.get("EXO_DEFAULT_TOP_P") else None
)
CLUSTER_DEFAULT_TOP_K: int | None = (
    int(os.environ["EXO_DEFAULT_TOP_K"])
    if os.environ.get("EXO_DEFAULT_TOP_K") else None
)
CLUSTER_DEFAULT_MIN_P: float | None = (
    float(os.environ["EXO_DEFAULT_MIN_P"])
    if os.environ.get("EXO_DEFAULT_MIN_P") else None
)
CLUSTER_DEFAULT_PRESENCE_PENALTY: float | None = (
    float(os.environ["EXO_DEFAULT_PRESENCE_PENALTY"])
    if os.environ.get("EXO_DEFAULT_PRESENCE_PENALTY") else None
)
CLUSTER_DEFAULT_REPETITION_PENALTY: float | None = (
    float(os.environ["EXO_DEFAULT_REPETITION_PENALTY"])
    if os.environ.get("EXO_DEFAULT_REPETITION_PENALTY") else None
)
CLUSTER_DEFAULT_FREQUENCY_PENALTY: float | None = (
    float(os.environ["EXO_DEFAULT_FREQUENCY_PENALTY"])
    if os.environ.get("EXO_DEFAULT_FREQUENCY_PENALTY") else None
)

# --- Hardcoded last-resort fallbacks (existing Exo behavior). ---
HARDCODED_TEMPERATURE: float = 0.7
HARDCODED_TOP_P: float = 1.0
HARDCODED_TOP_K: int = 0
HARDCODED_MIN_P: float = 0.05
# Penalties: None preserves existing behavior — no processor added in mlx-lm.
HARDCODED_PRESENCE_PENALTY: float | None = None
HARDCODED_REPETITION_PENALTY: float | None = None
HARDCODED_FREQUENCY_PENALTY: float | None = None


def _first_non_none(*values: _T | None) -> _T | None:
    for v in values:
        if v is not None:
            return v
    return None


def resolve_sampling(
    *,
    request_temperature: float | None = None,
    request_top_p: float | None = None,
    request_top_k: int | None = None,
    request_min_p: float | None = None,
    request_presence_penalty: float | None = None,
    request_repetition_penalty: float | None = None,
    request_frequency_penalty: float | None = None,
    instance_temperature: float | None = None,
    instance_top_p: float | None = None,
    instance_top_k: int | None = None,
    instance_min_p: float | None = None,
    instance_presence_penalty: float | None = None,
    instance_repetition_penalty: float | None = None,
    instance_frequency_penalty: float | None = None,
    card_temperature: float | None = None,
    card_top_p: float | None = None,
    card_top_k: int | None = None,
    card_min_p: float | None = None,
    card_presence_penalty: float | None = None,
    card_repetition_penalty: float | None = None,
    card_frequency_penalty: float | None = None,
) -> dict[str, Any]:
    """Return all seven resolved sampling params as a single dict.

    Resolution order per field: request → instance → card → cluster env →
    hardcoded. Caller plucks `temp/top_p/top_k/min_p` for `make_sampler`
    and the penalty fields for `make_logits_processors`.
    """
    return {
        "temp": _first_non_none(
            request_temperature, instance_temperature, card_temperature,
            CLUSTER_DEFAULT_TEMPERATURE, HARDCODED_TEMPERATURE,
        ),
        "top_p": _first_non_none(
            request_top_p, instance_top_p, card_top_p,
            CLUSTER_DEFAULT_TOP_P, HARDCODED_TOP_P,
        ),
        "top_k": _first_non_none(
            request_top_k, instance_top_k, card_top_k,
            CLUSTER_DEFAULT_TOP_K, HARDCODED_TOP_K,
        ),
        "min_p": _first_non_none(
            request_min_p, instance_min_p, card_min_p,
            CLUSTER_DEFAULT_MIN_P, HARDCODED_MIN_P,
        ),
        "presence_penalty": _first_non_none(
            request_presence_penalty, instance_presence_penalty, card_presence_penalty,
            CLUSTER_DEFAULT_PRESENCE_PENALTY, HARDCODED_PRESENCE_PENALTY,
        ),
        "repetition_penalty": _first_non_none(
            request_repetition_penalty, instance_repetition_penalty, card_repetition_penalty,
            CLUSTER_DEFAULT_REPETITION_PENALTY, HARDCODED_REPETITION_PENALTY,
        ),
        "frequency_penalty": _first_non_none(
            request_frequency_penalty, instance_frequency_penalty, card_frequency_penalty,
            CLUSTER_DEFAULT_FREQUENCY_PENALTY, HARDCODED_FREQUENCY_PENALTY,
        ),
    }


def card_sampling_values(
    model_id: "ModelId", enable_thinking: bool | None
) -> "SamplingValues | None":
    """Pick the right SamplingValues branch from a model card.

    Returns a `SamplingValues` instance (or ``None`` if the model has no card),
    selecting the `thinking` / `non_thinking` sub-profile when the request
    explicitly enables or disables thinking and the card has that branch;
    otherwise returns the flat card-level defaults.
    """
    from exo.shared.models.model_cards import get_card

    card = get_card(model_id)
    if card is None:
        return None
    flat = card.sampling_defaults
    if enable_thinking is True and flat.thinking is not None:
        return flat.thinking
    if enable_thinking is False and flat.non_thinking is not None:
        return flat.non_thinking
    return flat
