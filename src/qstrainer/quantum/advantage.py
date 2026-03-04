"""Quantum advantage benchmarking suite.

Compares quantum solvers (QAOA sim, Qiskit Runtime, D-Wave) against
classical baselines (SA) across a range of problem sizes to quantify
any quantum advantage in feature-selection QUBO problems.

Reports:
  - Solution quality (energy gap to best-known)
  - Time-to-solution (wall clock)
  - Scaling behaviour (time vs problem size)
  - Statistical significance (multiple trials)

Usage::

    from qstrainer.quantum.advantage import QuantumAdvantageBenchmark
    bench = QuantumAdvantageBenchmark()
    report = bench.run(problem_sizes=[8, 12, 16, 20])
    print(report.summary())
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from qstrainer.solvers.base import QUBOResult, QUBOSolverBase

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TrialResult:
    """Results from one (solver, problem_size) trial."""
    solver_name: str
    problem_size: int
    energy: float
    solve_time_s: float
    solution: np.ndarray
    trial_index: int
    metadata: Dict = field(default_factory=dict)


@dataclass(slots=True)
class SizeReport:
    """Aggregated results for one problem size across all solvers."""
    problem_size: int
    best_known_energy: float
    solver_results: Dict[str, List[TrialResult]]

    def gap(self, solver_name: str) -> float:
        """Mean energy gap to best-known (lower = better)."""
        trials = self.solver_results.get(solver_name, [])
        if not trials:
            return float("inf")
        mean_energy = sum(t.energy for t in trials) / len(trials)
        return mean_energy - self.best_known_energy

    def mean_time(self, solver_name: str) -> float:
        """Mean solve time for a solver."""
        trials = self.solver_results.get(solver_name, [])
        if not trials:
            return float("inf")
        return sum(t.solve_time_s for t in trials) / len(trials)


@dataclass
class BenchmarkReport:
    """Full benchmark report across all problem sizes."""
    size_reports: List[SizeReport] = field(default_factory=list)
    solvers: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary table."""
        lines = ["═══ Quantum Advantage Benchmark ═══\n"]
        header = f"{'Size':>6}"
        for name in self.solvers:
            header += f"  {name:>20} (E/t)"
        lines.append(header)
        lines.append("─" * len(header))

        for sr in self.size_reports:
            row = f"{sr.problem_size:>6}"
            for name in self.solvers:
                gap = sr.gap(name)
                mt = sr.mean_time(name)
                row += f"  {gap:>8.3f} / {mt:>7.3f}s"
            lines.append(row)

        # Winner analysis
        lines.append("\n── Winner by problem size ──")
        for sr in self.size_reports:
            best_name = min(
                self.solvers,
                key=lambda s: sr.gap(s) if sr.gap(s) < float("inf") else 1e9,
            )
            lines.append(
                f"  n={sr.problem_size:>3}: {best_name} "
                f"(gap={sr.gap(best_name):.4f}, time={sr.mean_time(best_name):.3f}s)"
            )

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Serialisable representation."""
        return {
            "solvers": self.solvers,
            "sizes": [
                {
                    "n": sr.problem_size,
                    "best_energy": sr.best_known_energy,
                    "results": {
                        name: [
                            {"energy": t.energy, "time": t.solve_time_s}
                            for t in trials
                        ]
                        for name, trials in sr.solver_results.items()
                    },
                }
                for sr in self.size_reports
            ],
        }


class QuantumAdvantageBenchmark:
    """Run structured benchmarks across quantum and classical solvers.

    Parameters
    ----------
    n_trials : int
        Number of random QUBO instances per problem size.
    seed : int
        RNG seed for reproducible QUBO generation.
    """

    def __init__(self, n_trials: int = 5, seed: int = 42) -> None:
        self._n_trials = n_trials
        self._rng = np.random.default_rng(seed)
        self._solvers: Dict[str, QUBOSolverBase] = {}

    def register_solver(self, name: str, solver: QUBOSolverBase) -> None:
        """Add a solver to the benchmark."""
        self._solvers[name] = solver

    def generate_qubo(self, n: int) -> np.ndarray:
        """Generate a random QUBO matrix of size n×n.

        Designed to resemble real feature-selection QUBO problems:
        diagonal terms (feature relevance) + off-diagonal (redundancy).
        """
        # Diagonal: feature relevance scores (negative = good)
        Q = np.diag(self._rng.uniform(-1.0, 0.0, size=n))
        # Off-diagonal: pairwise redundancy (positive = penalty)
        for i in range(n):
            for j in range(i + 1, n):
                # Sparse redundancy structure (~30% non-zero)
                if self._rng.random() < 0.3:
                    w = self._rng.uniform(0.05, 0.5)
                    Q[i, j] = w
        return Q

    def brute_force_optimal(self, Q: np.ndarray) -> float:
        """Brute-force the optimal QUBO energy (only feasible for n ≤ 20)."""
        n = Q.shape[0]
        if n > 20:
            return float("-inf")  # unknown
        N = 1 << n
        k = np.arange(N, dtype=np.int64)
        bits = ((k[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)
        energies = np.einsum("ki,ij,kj->k", bits, Q, bits)
        return float(np.min(energies))

    def run(
        self,
        problem_sizes: Optional[List[int]] = None,
    ) -> BenchmarkReport:
        """Run the full benchmark suite.

        Parameters
        ----------
        problem_sizes : list of int
            QUBO sizes to test.  Default: [6, 10, 14, 18].
        """
        if problem_sizes is None:
            problem_sizes = [6, 10, 14, 18]

        report = BenchmarkReport(solvers=list(self._solvers.keys()))

        for n in problem_sizes:
            logger.info("Benchmarking n=%d (%d trials)", n, self._n_trials)
            solver_results: Dict[str, List[TrialResult]] = {
                name: [] for name in self._solvers
            }
            best_known = float("inf")

            for trial in range(self._n_trials):
                Q = self.generate_qubo(n)

                # Try brute force for ground truth
                if n <= 20:
                    optimal = self.brute_force_optimal(Q)
                    best_known = min(best_known, optimal)

                for name, solver in self._solvers.items():
                    try:
                        result = solver.solve(Q)
                        tr = TrialResult(
                            solver_name=name,
                            problem_size=n,
                            energy=result.energy,
                            solve_time_s=result.solve_time_s,
                            solution=result.solution,
                            trial_index=trial,
                            metadata=result.metadata,
                        )
                        solver_results[name].append(tr)
                        best_known = min(best_known, result.energy)
                    except Exception as e:
                        logger.warning(
                            "Solver %s failed on n=%d trial=%d: %s",
                            name, n, trial, e,
                        )

            report.size_reports.append(SizeReport(
                problem_size=n,
                best_known_energy=best_known,
                solver_results=solver_results,
            ))

        return report

    @classmethod
    def from_scheduler(cls, scheduler, n_trials: int = 5, seed: int = 42) -> "QuantumAdvantageBenchmark":
        """Build from a QOSScheduler, inheriting all registered solvers."""
        bench = cls(n_trials=n_trials, seed=seed)
        for name in scheduler.available_solvers():
            solver = scheduler.get_solver(name)
            if solver is not None:
                bench.register_solver(name, solver)
        return bench
