"""QUBO solvers: SA, QAOA, D-Wave, Mock."""

from qstrainer.solvers.base import QUBOResult, QUBOSolverBase
from qstrainer.solvers.dwave import DWaveSolver
from qstrainer.solvers.mock import MockQuantumSolver
from qstrainer.solvers.qaoa import QAOASolver
from qstrainer.solvers.sa import SimulatedAnnealingSolver

__all__ = [
    "QUBOSolverBase",
    "QUBOResult",
    "SimulatedAnnealingSolver",
    "QAOASolver",
    "DWaveSolver",
    "MockQuantumSolver",
]
