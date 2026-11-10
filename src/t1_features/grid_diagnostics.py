"""Pure summaries for word-level correctness trajectories across T1 C values."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


T1_C_GRID = (0.1, 0.3, 1.0, 3.0, 7.0, 10.0, 13.0, 30.0, 42.0, 100.0)


class T1GridDiagnosticError(ValueError):
    """A correctness trajectory does not match the approved C grid."""


@dataclass(frozen=True)
class TrajectorySummary:
    trajectory: str
    n_wrong: int
    n_runs: int
    n_flips: int
    c_threshold: float | None
    flip_direction: str
    category: str


def summarize_trajectory(
    correctness: Sequence[bool],
    c_values: Sequence[float] = T1_C_GRID,
) -> TrajectorySummary:
    """Summarize Top-1 correctness across the ordered, approved C values."""
    if len(correctness) != len(c_values) or not correctness:
        raise T1GridDiagnosticError(
            "correctness and C values must have the same nonzero length"
        )
    if any(type(value) is not bool for value in correctness):
        raise T1GridDiagnosticError("correctness values must be booleans")
    numeric_c: list[float] = []
    for value in c_values:
        if type(value) is bool or not isinstance(value, (int, float)):
            raise T1GridDiagnosticError(f"invalid C value {value!r}")
        if not math.isfinite(value) or value <= 0:
            raise T1GridDiagnosticError(f"invalid C value {value!r}")
        numeric_c.append(float(value))
    if any(left >= right for left, right in zip(numeric_c, numeric_c[1:])):
        raise T1GridDiagnosticError("C values must be strictly increasing")

    states = tuple(correctness)
    n_flips = sum(left != right for left, right in zip(states, states[1:]))
    n_wrong = sum(not state for state in states)
    threshold: float | None = None
    if all(states):
        category = "always_correct"
        direction = "none"
        threshold = numeric_c[0]
    elif not any(states):
        category = "always_wrong"
        direction = "none"
    elif n_flips == 1 and not states[0] and states[-1]:
        category = "threshold_x_to_plus"
        direction = "x_to_plus"
        threshold = numeric_c[next(index for index, state in enumerate(states) if state)]
    elif n_flips == 1 and states[0] and not states[-1]:
        category = "regression_plus_to_x"
        direction = "plus_to_x"
    else:
        category = "unstable"
        direction = "multiple"

    return TrajectorySummary(
        trajectory="".join("+" if state else "x" for state in states),
        n_wrong=n_wrong,
        n_runs=len(states),
        n_flips=n_flips,
        c_threshold=threshold,
        flip_direction=direction,
        category=category,
    )
