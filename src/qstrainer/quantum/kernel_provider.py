"""Quantum kernel provider — ZZ feature map statevector simulation.

TODAY:  NumPy statevector (validates math, ≤12 qubits feasible).
FUTURE: qiskit_machine_learning.kernels.FidelityQuantumKernel
        with IBM Runtime backend for real QPU execution.
"""

from __future__ import annotations

import numpy as np


class QuantumKernelProvider:
    """Computes quantum kernel matrix via statevector simulation.

    Feature map: ZZ feature map (standard in quantum ML literature).
    Kernel: K(x_i, x_j) = |⟨φ(x_i)|φ(x_j)⟩|²
    """

    def __init__(self, n_qubits: int = 8, reps: int = 2, seed: int = 42) -> None:
        self.n_qubits = n_qubits
        self.reps = reps
        self.rng = np.random.default_rng(seed)
        self._N = 1 << n_qubits

    def _apply_feature_map(self, x: np.ndarray) -> np.ndarray:
        """Apply ZZ feature map to |0⟩^n, return statevector."""
        n = self.n_qubits
        N = self._N

        state = np.zeros(N, dtype=np.complex128)
        state[0] = 1.0

        for _rep in range(self.reps):
            # Hadamards
            for q in range(n):
                step = 1 << q
                for start in range(0, N, 2 * step):
                    idx0 = np.arange(start, start + step)
                    idx1 = idx0 + step
                    s0 = state[idx0].copy()
                    s1 = state[idx1].copy()
                    state[idx0] = (s0 + s1) / np.sqrt(2)
                    state[idx1] = (s0 - s1) / np.sqrt(2)

            # Single-qubit Z rotations
            for q in range(n):
                angle = x[q % len(x)]
                for k in range(N):
                    bit = (k >> q) & 1
                    phase = angle if bit == 0 else -angle
                    state[k] *= np.exp(1j * phase)

            # ZZ entangling
            for q1 in range(n):
                for q2 in range(q1 + 1, min(q1 + 3, n)):
                    angle = (np.pi - x[q1 % len(x)]) * (np.pi - x[q2 % len(x)])
                    for k in range(N):
                        b1 = (k >> q1) & 1
                        b2 = (k >> q2) & 1
                        parity = 1 - 2 * (b1 ^ b2)
                        state[k] *= np.exp(1j * angle * parity)

        return state

    def kernel_value(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute K(x1, x2) = |⟨φ(x1)|φ(x2)⟩|²."""
        sv1 = self._apply_feature_map(x1)
        sv2 = self._apply_feature_map(x2)
        return float(np.abs(np.vdot(sv1, sv2)) ** 2)

    def kernel_matrix(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        """Compute full kernel matrix.  Symmetric if Y is None."""
        if Y is None:
            Y = X
            symmetric = True
        else:
            symmetric = False

        n = X.shape[0]
        m = Y.shape[0]
        K = np.zeros((n, m))

        sv_X = [self._apply_feature_map(X[i]) for i in range(n)]
        sv_Y = sv_X if symmetric else [self._apply_feature_map(Y[j]) for j in range(m)]

        for i in range(n):
            j_start = i if symmetric else 0
            for j in range(j_start, m):
                val = float(np.abs(np.vdot(sv_X[i], sv_Y[j])) ** 2)
                K[i, j] = val
                if symmetric and i != j:
                    K[j, i] = val

        return K
