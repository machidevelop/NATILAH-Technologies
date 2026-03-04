"""QUBO ↔ Ising Hamiltonian conversion.

The Ising model uses spin variables σ_i ∈ {-1, +1}.
The QUBO uses binary variables x_i ∈ {0, 1}.

Relationship: x_i = (1 − σ_i) / 2

Ising Hamiltonian:
    H(σ) = Σ_i h_i σ_i  +  Σ_{i<j} J_ij σ_i σ_j  +  offset

This module converts between the two representations and provides
energy evaluation + spin/binary helpers:

    qubo_to_ising(Q) → (h, J, offset)
    ising_to_qubo(h, J) → (Q, offset)
    qubo_energy(x, Q) → float
    ising_energy(σ, h, J) → float
    binary_to_spin(x) → σ
    spin_to_binary(σ) → x
"""

from __future__ import annotations

import numpy as np

# ── Conversion ───────────────────────────────────────────────


def qubo_to_ising(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert QUBO matrix Q to Ising parameters (h, J, offset).

    Substituting x_i = (1 − σ_i)/2 into E(x) = x^T Q x yields:

        h_i    = −(1/2) · Σ_j Qs_{ij}          (row-sum of symmetrised Q)
        J_{ij} =  (1/2) · Qs_{ij}               for i < j
        offset =  (1/2) · Σ_i Qs_{ii}  +  (1/2) · Σ_{i<j} Qs_{ij}

    where Qs = (Q + Q^T) / 2.

    Parameters
    ----------
    Q : ndarray, shape (n, n)
        QUBO matrix (upper-triangular *or* symmetric).

    Returns
    -------
    h : ndarray, shape (n,)
        Local-field (linear) coefficients.
    J : ndarray, shape (n, n)
        Coupling (quadratic) coefficients — upper-triangular.
    offset : float
        Constant energy shift so that
        ising_energy(binary_to_spin(x), h, J) + offset == qubo_energy(x, Q)
        for all x ∈ {0,1}^n.
    """
    Q.shape[0]
    Qs = (Q + Q.T) / 2.0  # symmetrise

    # h_i = −(row_sum_i) / 2
    h = -np.sum(Qs, axis=1) / 2.0

    # J_{ij} = Qs_{ij} / 2  (upper-triangular)
    J = np.triu(Qs, k=1) / 2.0

    # offset = diag_sum/2  +  upper_off_diag_sum/2
    offset = float(np.trace(Qs) / 2.0 + np.sum(np.triu(Qs, k=1)) / 2.0)

    return h, J, offset


def ising_to_qubo(
    h: np.ndarray,
    J: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Convert Ising parameters (h, J) to a QUBO matrix Q.

    Uses σ_i = 1 − 2 x_i in H(σ) = Σ h_i σ_i + Σ_{i<j} J_{ij} σ_i σ_j.

    Returns
    -------
    Q : ndarray, shape (n, n)
        QUBO matrix (upper-triangular).
    offset : float
        Constant energy shift.
    """
    n = len(h)
    Q = np.zeros((n, n), dtype=np.float64)
    offset = float(np.sum(h))

    # Linear: h_i σ_i = h_i (1 − 2x_i) → −2h_i x_i + h_i
    for i in range(n):
        Q[i, i] -= 2.0 * h[i]

    # Quadratic: J_{ij} σ_i σ_j = J_{ij}(1−2x_i)(1−2x_j)
    #   = J_{ij} − 2J_{ij}x_i − 2J_{ij}x_j + 4J_{ij}x_ix_j
    for i in range(n):
        for j in range(i + 1, n):
            jij = J[i, j]
            if jij == 0.0:
                continue
            Q[i, j] += 4.0 * jij
            Q[i, i] -= 2.0 * jij
            Q[j, j] -= 2.0 * jij
            offset += jij

    return Q, offset


# ── Energy evaluation ────────────────────────────────────────


def qubo_energy(x: np.ndarray, Q: np.ndarray) -> float:
    """E(x) = x^T Q x  for binary x ∈ {0,1}^n."""
    return float(x @ Q @ x)


def ising_energy(sigma: np.ndarray, h: np.ndarray, J: np.ndarray) -> float:
    """H(σ) = h·σ + σ^T J σ  for spin σ ∈ {-1,+1}^n  (J upper-tri)."""
    return float(h @ sigma + sigma @ J @ sigma)


# ── Spin / binary helpers ────────────────────────────────────


def binary_to_spin(x: np.ndarray) -> np.ndarray:
    """Binary {0,1} → spin {+1,−1}.  x=0 → σ=+1, x=1 → σ=−1."""
    return 1.0 - 2.0 * np.asarray(x, dtype=np.float64)


def spin_to_binary(sigma: np.ndarray) -> np.ndarray:
    """Spin {+1,−1} → binary {0,1}.  σ=+1 → x=0, σ=−1 → x=1."""
    return ((1.0 - np.asarray(sigma, dtype=np.float64)) / 2.0).astype(int)
