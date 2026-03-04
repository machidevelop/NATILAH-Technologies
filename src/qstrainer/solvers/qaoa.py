"""QAOA Solver — gate-model quantum simulation via NumPy statevector.

TODAY:  Pure-NumPy statevector simulation (2^n amplitudes, ≤20 qubits).
FUTURE: Swap inner loop for Qiskit QuantumCircuit + IBM Runtime.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from qstrainer.solvers.base import QUBOResult, QUBOSolverBase


class QAOASolver(QUBOSolverBase):
    """Quantum Approximate Optimization Algorithm for QUBO.

    Uses reshape-based mixer (no fancy indexing) + scipy COBYLA
    optimizer for fast convergence.
    """

    MAX_QUBITS = 20

    def __init__(
        self,
        p: int = 1,
        n_restarts: int = 3,
        maxfev: int = 80,
        seed: int = 42,
    ) -> None:
        self.p = p
        self.n_restarts = n_restarts
        self.maxfev = maxfev
        self.rng = np.random.default_rng(seed)

    @property
    def solver_type(self) -> str:
        return "quantum_sim"

    def _precompute_costs(self, Q: np.ndarray) -> np.ndarray:
        n = Q.shape[0]
        N = 1 << n
        k = np.arange(N, dtype=np.int64)
        bits = ((k[:, None] >> np.arange(n, dtype=np.int64)[None, :]) & 1).astype(
            np.float64
        )
        return np.einsum("ki,ij,kj->k", bits, Q, bits)

    def _apply_mixer(self, state: np.ndarray, beta: float, n: int) -> None:
        c = np.cos(beta)
        ms = -1j * np.sin(beta)
        for q in range(n):
            step = 1 << q
            view = state.reshape(-1, 2, step)
            s0 = view[:, 0, :].copy()
            s1 = view[:, 1, :].copy()
            view[:, 0, :] = c * s0 + ms * s1
            view[:, 1, :] = ms * s0 + c * s1

    def _qaoa_eval(
        self, params: np.ndarray, costs: np.ndarray, n: int
    ) -> float:
        N = 1 << n
        state = np.full(N, 1.0 / np.sqrt(N), dtype=np.complex128)
        for layer in range(self.p):
            state *= np.exp(-1j * params[layer] * costs)
            self._apply_mixer(state, params[self.p + layer], n)
        probs = np.abs(state) ** 2
        return float(probs @ costs)

    def solve(self, Q: np.ndarray) -> QUBOResult:
        n = Q.shape[0]
        if n > self.MAX_QUBITS:
            raise ValueError(
                f"QAOA statevector sim limited to {self.MAX_QUBITS} qubits "
                f"(got {n}).  Use DWaveSolver or Qiskit Runtime."
            )

        t0 = time.perf_counter()
        costs = self._precompute_costs(Q)
        N = 1 << n

        best_energy = float("inf")
        best_params: np.ndarray | None = None

        for _ in range(self.n_restarts):
            x0 = self.rng.uniform(0, 2 * np.pi, size=2 * self.p)
            res = scipy_minimize(
                lambda p: self._qaoa_eval(p, costs, n),
                x0,
                method="COBYLA",
                options={"maxiter": self.maxfev, "rhobeg": 0.5},
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_params = res.x.copy()

        assert best_params is not None
        state = np.full(N, 1.0 / np.sqrt(N), dtype=np.complex128)
        for layer in range(self.p):
            state *= np.exp(-1j * best_params[layer] * costs)
            self._apply_mixer(state, best_params[self.p + layer], n)

        probs = np.abs(state) ** 2
        top_k = np.argsort(-probs)[: max(N // 10, 10)]
        best_idx = top_k[np.argmin(costs[top_k])]
        solution = np.array([(best_idx >> q) & 1 for q in range(n)], dtype=int)

        return QUBOResult(
            solution=solution,
            energy=float(costs[best_idx]),
            solver_name="qaoa_statevector",
            solve_time_s=time.perf_counter() - t0,
            metadata={
                "p": self.p,
                "n_qubits": n,
                "n_restarts": self.n_restarts,
                "optimal_gammas": best_params[: self.p].tolist(),
                "optimal_betas": best_params[self.p :].tolist(),
                "backend": "numpy_statevector",
                "upgrade_path": (
                    "qiskit.primitives.StatevectorEstimator -> IBM Runtime"
                ),
            },
        )
