"""Depth-matrix validity labelling (design point 6).

A constant is validated only over the cells it was actually exercised at. A
retransmit timer tuned at shallow context and shipped as a global default broke
generation entirely at depth - four separate times. So a record carries the
explicit set of matrix cells it covers, and a claim of general/default-safe
validity must be checked against the full matrix rather than asserted.
"""

from __future__ import annotations

from typing import Final, Literal, final, get_args

from pydantic import BaseModel, ConfigDict

__all__ = [
    "ALL_DEPTH_CELLS",
    "CONTEXT_DEPTHS",
    "THERMAL_STATES",
    "ContextDepth",
    "DepthMatrixCell",
    "ThermalState",
    "coverage_gaps",
    "is_default_safe",
]

ContextDepth = Literal["shallow", "mid", "near_max_context"]
ThermalState = Literal["cold_start", "warm"]

CONTEXT_DEPTHS: Final[tuple[ContextDepth, ...]] = get_args(ContextDepth)
THERMAL_STATES: Final[tuple[ThermalState, ...]] = get_args(ThermalState)


@final
class DepthMatrixCell(BaseModel):
    """One {context depth} x {thermal state} cell of the validity matrix."""

    model_config = ConfigDict(frozen=True, strict=True)

    context_depth: ContextDepth
    thermal_state: ThermalState
    prompt_tokens: int

    def label(self) -> str:
        return f"{self.context_depth}/{self.thermal_state}"


ALL_DEPTH_CELLS: Final[tuple[tuple[ContextDepth, ThermalState], ...]] = tuple(
    (depth, thermal) for depth in CONTEXT_DEPTHS for thermal in THERMAL_STATES
)


def coverage_gaps(
    cells: tuple[DepthMatrixCell, ...],
) -> tuple[tuple[ContextDepth, ThermalState], ...]:
    """Matrix cells NOT covered by ``cells``."""
    covered = {(cell.context_depth, cell.thermal_state) for cell in cells}
    return tuple(cell for cell in ALL_DEPTH_CELLS if cell not in covered)


def is_default_safe(cells: tuple[DepthMatrixCell, ...]) -> bool:
    """Whether a result may be presented as a general / default-safe finding.

    Only true when every matrix cell was actually exercised. Anything less is a
    cell-scoped result and must be reported with its scope attached.
    """
    return not coverage_gaps(cells)
