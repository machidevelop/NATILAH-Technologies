"""Tests for QUBO solvers."""

from __future__ import annotations

import numpy as np
import pytest

from qstrainer.solvers.base import QUBOResult
from qstrainer.solvers.sa import SimulatedAnnealingSolver
from qstrainer.solvers.qaoa import QAOASolver
from qstrainer.solvers.mock import MockQuantumSolver


def _random_qubo(n: int = 10, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((n, n))
    return (Q + Q.T) / 2


class TestSimulatedAnnealing:
    def test_solve_returns_result(self):
        Q = _random_qubo(8)
        solver = SimulatedAnnealingSolver(num_reads=10, num_sweeps=100, seed=1)
        result = solver.solve(Q)
        assert isinstance(result, QUBOResult)
        assert result.solution.shape == (8,)
        assert set(result.solution).issubset({0, 1})

    def test_solver_type(self):
        solver = SimulatedAnnealingSolver()
        assert solver.solver_type == "classical"

    def test_deterministic_with_seed(self):
        Q = _random_qubo(6)
        r1 = SimulatedAnnealingSolver(seed=42).solve(Q)
        r2 = SimulatedAnnealingSolver(seed=42).solve(Q)
        assert np.array_equal(r1.solution, r2.solution)


class TestQAOA:
    def test_solve_small(self):
        Q = _random_qubo(5)
        solver = QAOASolver(p=1, n_restarts=2, maxfev=30, seed=42)
        result = solver.solve(Q)
        assert result.solution.shape == (5,)
        assert result.metadata["backend"] == "numpy_statevector"

    def test_rejects_large(self):
        Q = _random_qubo(25)
        solver = QAOASolver()
        with pytest.raises(ValueError, match="20 qubits"):
            solver.solve(Q)

    def test_solver_type(self):
        assert QAOASolver().solver_type == "quantum_sim"


class TestMock:
    def test_wraps_sa(self):
        Q = _random_qubo(6)
        solver = MockQuantumSolver(num_reads=10, num_sweeps=100)
        result = solver.solve(Q)
        assert result.solver_name == "mock_quantum"
        assert solver.solver_type == "quantum_mock"
