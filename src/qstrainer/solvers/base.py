"""Abstract base class and result dataclass for all QUBO solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class QUBOResult:
    """Result from any QUBO solver (classical or quantum)."""

    solution: np.ndarray  # binary vector
    energy: float
    solver_name: str
    solve_time_s: float
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "solution": self.solution.tolist(),
            "energy": self.energy,
            "solver_name": self.solver_name,
            "solve_time_s": self.solve_time_s,
            "metadata": self.metadata,
        }


class QUBOSolverBase(ABC):
    """Every QUBO solver — classical or quantum — implements this.

    Swap solvers with zero pipeline changes.
    """

    @abstractmethod
    def solve(self, Q: np.ndarray) -> QUBOResult:
        """Solve a QUBO defined by upper-triangular matrix Q."""
        ...

    @property
    @abstractmethod
    def solver_type(self) -> str:
        """One of: 'classical', 'quantum_sim', 'quantum_hw'."""
        ...

    def is_available(self) -> bool:
        """Override to check runtime availability (e.g. D-Wave token)."""
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.solver_type})"
