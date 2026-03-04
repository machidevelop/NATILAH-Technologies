"""Mock quantum solver — wraps classical SA for testing quantum code paths."""

from __future__ import annotations

from typing import Any

import numpy as np

from qstrainer.solvers.base import QUBOResult, QUBOSolverBase
from qstrainer.solvers.sa import SimulatedAnnealingSolver


class MockQuantumSolver(QUBOSolverBase):
    """Classical SA that tags results as 'quantum_mock'.

    Use for integration testing of quantum code paths without hardware.
    """

    def __init__(self, **sa_kwargs: Any) -> None:
        self._inner = SimulatedAnnealingSolver(**sa_kwargs)

    @property
    def solver_type(self) -> str:
        return "quantum_mock"

    def solve(self, Q: np.ndarray) -> QUBOResult:
        result = self._inner.solve(Q)
        result.solver_name = "mock_quantum"
        result.metadata["note"] = "Classical SA posing as quantum for testing"
        return result
