"""Simulated Annealing QUBO solver — production-grade classical baseline."""

from __future__ import annotations

import time

import numpy as np

from qstrainer.solvers.base import QUBOResult, QUBOSolverBase


class SimulatedAnnealingSolver(QUBOSolverBase):
    """Production-grade classical SA for QUBO.

    Default solver.  Fast, reliable, identical interface to quantum solvers.
    """

    def __init__(
        self,
        num_reads: int = 500,
        num_sweeps: int = 1500,
        beta_range: tuple[float, float] = (0.1, 5.0),
        seed: int = 42,
    ) -> None:
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.beta_range = beta_range
        self.rng = np.random.default_rng(seed)

    @property
    def solver_type(self) -> str:
        return "classical"

    def solve(self, Q: np.ndarray) -> QUBOResult:
        n = Q.shape[0]
        best_x: np.ndarray | None = None
        best_energy = float("inf")
        betas = np.linspace(self.beta_range[0], self.beta_range[1], self.num_sweeps)
        t0 = time.perf_counter()

        for _ in range(self.num_reads):
            x = self.rng.integers(0, 2, size=n).astype(np.float64)
            energy = float(x @ Q @ x)

            for sweep in range(self.num_sweeps):
                flip = self.rng.integers(0, n)
                x_new = x.copy()
                x_new[flip] = 1.0 - x_new[flip]
                new_energy = float(x_new @ Q @ x_new)
                delta = new_energy - energy
                if delta < 0 or self.rng.random() < np.exp(-betas[sweep] * delta):
                    x, energy = x_new, new_energy

            if energy < best_energy:
                best_energy, best_x = energy, x.copy()

        assert best_x is not None
        return QUBOResult(
            solution=best_x.astype(int),
            energy=best_energy,
            solver_name="simulated_annealing",
            solve_time_s=time.perf_counter() - t0,
            metadata={
                "num_reads": self.num_reads,
                "num_sweeps": self.num_sweeps,
                "backend": "numpy",
            },
        )
