"""Candidate containers for adaptive DOE acquisition."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostAwareCandidate:
    """One possible next observation in the cost-aware acquisition loop."""

    kind: str
    score: float
    rho: float
    cost: float
    x: np.ndarray = None
    x_idx: int = None
    direction: np.ndarray = None
    order: int = None
