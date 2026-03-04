"""QUBO Feature Selector — Stage 3a (quantum-ready).

Encodes Minimum Redundancy Maximum Relevance (mRMR) as a QUBO
and solves with any registered solver (SA, QAOA, D-Wave).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from qstrainer.solvers.base import QUBOResult, QUBOSolverBase
from qstrainer.solvers.sa import SimulatedAnnealingSolver

logger = logging.getLogger(__name__)


class QUBOFeatureSelector:
    """Select optimal feature subset via QUBO.

    Encodes mRMR:
      min  -α Σ_i relevance(i)·x_i
           + (1-α) Σ_{i<j} redundancy(i,j)·x_i x_j
           + P · (Σ_i x_i - k)²
    """

    def __init__(self, n_select: int = 8, alpha: float = 0.5) -> None:
        self.n_select = n_select
        self.alpha = alpha

    @classmethod
    def from_config(cls, cfg: Dict) -> "QUBOFeatureSelector":
        qc = cfg.get("qubo_selector", {})
        return cls(
            n_select=qc.get("n_select", 8),
            alpha=qc.get("alpha", 0.5),
        )

    def build_qubo(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Build QUBO matrix for feature selection."""
        n_feat = X.shape[1]
        Q = np.zeros((n_feat, n_feat), dtype=np.float64)

        # Relevance: |cor(feature_i, label)|
        relevance = np.zeros(n_feat)
        for i in range(n_feat):
            if np.std(X[:, i]) > 1e-10 and np.std(y) > 1e-10:
                relevance[i] = abs(np.corrcoef(X[:, i], y)[0, 1])

        # Redundancy: |cor(feature_i, feature_j)|
        corr_matrix = np.corrcoef(X.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        penalty = 2.0 * max(np.max(relevance), 1.0)
        k = self.n_select

        # Relevance (diagonal, negative = maximise)
        for i in range(n_feat):
            Q[i, i] -= self.alpha * relevance[i]

        # Redundancy (off-diagonal, positive = penalise)
        for i in range(n_feat):
            for j in range(i + 1, n_feat):
                Q[i, j] += (1 - self.alpha) * abs(corr_matrix[i, j])

        # Cardinality constraint: (Σ x_i - k)²
        for i in range(n_feat):
            Q[i, i] += penalty * (1 - 2 * k)
        for i in range(n_feat):
            for j in range(i + 1, n_feat):
                Q[i, j] += penalty * 2

        return Q

    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        solver: Optional[QUBOSolverBase] = None,
    ) -> Tuple[List[int], QUBOResult]:
        """Select optimal feature subset.

        Returns (selected_indices, qubo_result).
        """
        Q = self.build_qubo(X, y)

        if solver is None:
            solver = SimulatedAnnealingSolver(num_reads=200, num_sweeps=1000)

        result = solver.solve(Q)
        selected = [i for i, v in enumerate(result.solution) if v == 1]

        # Fix cardinality if needed
        if len(selected) != self.n_select:
            scores = {i: abs(Q[i, i]) for i in range(Q.shape[0])}
            if len(selected) > self.n_select:
                selected = sorted(selected, key=lambda i: scores[i])[
                    : self.n_select
                ]
            else:
                missing = sorted(
                    [i for i in range(Q.shape[0]) if i not in selected],
                    key=lambda i: scores[i],
                )
                while len(selected) < self.n_select and missing:
                    selected.append(missing.pop(0))

        return sorted(selected), result
