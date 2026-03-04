"""QAOASampler — QAOA circuit with multi-bitstring sampling.

Unlike :class:`~qstrainer.solvers.qaoa.QAOASolver` which returns the single
best bitstring, this sampler produces a **population** of bitstrings sampled
from the QAOA output distribution.  The population is the input to graph
purification: each sample is a proposed partition, and edges that are
consistently separated across samples are "easy" conflicts to drop.

Pipeline position:

    QUBO → **Ising (h, J)** →  QAOASampler.build_and_optimise(h, J)
                             →  QAOASampler.sample(n_shots)
                                 → List[(bitstring, probability)]
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize as scipy_minimize

# ── Result container ─────────────────────────────────────────


@dataclass(slots=True)
class SampleResult:
    """One sampled bitstring together with its probability."""

    bitstring: np.ndarray  # binary {0,1}^n
    probability: float
    ising_energy: float


@dataclass
class SamplerOutput:
    """Complete output from :meth:`QAOASampler.sample`."""

    samples: list[SampleResult]
    optimal_energy: float
    optimal_params: np.ndarray
    optimize_time_s: float
    sample_time_s: float
    n_qubits: int
    p_layers: int

    @property
    def best(self) -> SampleResult:
        return min(self.samples, key=lambda s: s.ising_energy)


# ── QAOASampler ──────────────────────────────────────────────


class QAOASampler:
    """QAOA circuit (statevector) with multi-bitstring sampling.

    Operates on the **Ising Hamiltonian**:

        H_C  =  Σ_i h_i Z_i  +  Σ_{i<j} J_{ij} Z_i Z_j

    The circuit is  |ψ(γ,β)⟩ = U_M(β_p) U_C(γ_p) … U_M(β_1) U_C(γ_1) |+⟩

    where

        U_C(γ)  = exp(−i γ H_C)         (cost unitary)
        U_M(β)  = exp(−i β Σ X_i)       (mixer unitary)

    The classical outer loop optimises (γ*, β*) via COBYLA, then samples
    bitstrings from the output distribution.

    Parameters
    ----------
    p_layers : int
        Number of QAOA layers (higher = better quality, slower).
    n_restarts : int
        Random restarts for the parameter optimiser.
    maxfev : int
        Max function evaluations per restart.
    seed : int
        Random seed for reproducibility.
    """

    MAX_QUBITS: int = 22  # 2^22 = 4M amplitudes — practical ceiling

    def __init__(
        self,
        p_layers: int = 2,
        n_restarts: int = 5,
        maxfev: int = 120,
        seed: int = 42,
    ) -> None:
        self.p = p_layers
        self.n_restarts = n_restarts
        self.maxfev = maxfev
        self.rng = np.random.default_rng(seed)

        # Set by build_and_optimise
        self._n: int = 0
        self._h: np.ndarray | None = None
        self._J: np.ndarray | None = None
        self._costs: np.ndarray | None = None  # cost per basis state
        self._optimal_params: np.ndarray | None = None
        self._optimal_state: np.ndarray | None = None
        self._optimal_energy: float = float("inf")
        self._optimise_time: float = 0.0

    # ── Build + optimise ─────────────────────────────────────

    def build_and_optimise(
        self,
        h: np.ndarray,
        J: np.ndarray,
    ) -> float:
        """Construct the QAOA circuit from Ising parameters and optimise.

        Returns the optimal expectation value.
        """
        n = len(h)
        if n > self.MAX_QUBITS:
            raise ValueError(
                f"QAOASampler statevector sim limited to {self.MAX_QUBITS} "
                f"qubits (got {n}).  Reduce batch size or use SA fallback."
            )

        self._n = n
        self._h = h.copy()
        self._J = J.copy()
        N = 1 << n

        # Pre-compute Ising energy for every computational-basis state
        #   σ_k[q] = +1 if bit q of k is 0, −1 if bit q is 1
        k = np.arange(N, dtype=np.int64)
        bits = ((k[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)
        spins = 1.0 - 2.0 * bits  # σ = 1 − 2x

        # H(σ_k) = h·σ_k + σ_k^T J σ_k
        self._costs = np.einsum("ki,i->k", spins, h) + np.einsum("ki,ij,kj->k", spins, J, spins)

        t0 = time.perf_counter()
        best_energy = float("inf")
        best_params: np.ndarray | None = None

        for _ in range(self.n_restarts):
            x0 = self.rng.uniform(0, 2 * np.pi, size=2 * self.p)
            res = scipy_minimize(
                lambda p: self._eval_expectation(p),
                x0,
                method="COBYLA",
                options={"maxiter": self.maxfev, "rhobeg": 0.5},
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_params = res.x.copy()

        self._optimise_time = time.perf_counter() - t0
        assert best_params is not None
        self._optimal_params = best_params
        self._optimal_energy = best_energy

        # Compute final optimised state
        self._optimal_state = self._build_state(best_params)

        return best_energy

    # ── Sampling ─────────────────────────────────────────────

    def sample(
        self,
        n_shots: int = 1024,
        top_k: int = 64,
    ) -> SamplerOutput:
        """Sample bitstrings from the optimised QAOA output distribution.

        Parameters
        ----------
        n_shots : int
            Number of measurement shots to simulate.
        top_k : int
            Return at most this many unique bitstrings (by frequency).

        Returns
        -------
        SamplerOutput
            Contains sampled bitstrings, probabilities, energies, and timing.
        """
        if self._optimal_state is None:
            raise RuntimeError("Call build_and_optimise() first.")

        t0 = time.perf_counter()
        n = self._n
        N = 1 << n
        probs = np.abs(self._optimal_state) ** 2

        # Simulate measurement: sample indices from the probability distribution
        shot_indices = self.rng.choice(N, size=n_shots, p=probs)

        # Count unique bitstrings
        unique, counts = np.unique(shot_indices, return_counts=True)
        # Sort by frequency (descending)
        order = np.argsort(-counts)
        unique = unique[order]
        counts = counts[order]

        # Limit to top_k
        unique = unique[:top_k]
        counts = counts[:top_k]

        assert self._costs is not None
        results: list[SampleResult] = []
        for idx, _cnt in zip(unique, counts, strict=False):
            bits = np.array([(int(idx) >> q) & 1 for q in range(n)], dtype=np.int64)
            results.append(
                SampleResult(
                    bitstring=bits,
                    probability=float(probs[idx]),
                    ising_energy=float(self._costs[idx]),
                )
            )

        sample_time = time.perf_counter() - t0

        return SamplerOutput(
            samples=results,
            optimal_energy=self._optimal_energy,
            optimal_params=self._optimal_params.copy() if self._optimal_params is not None else np.array([]),
            optimize_time_s=self._optimise_time,
            sample_time_s=sample_time,
            n_qubits=self._n,
            p_layers=self.p,
        )

    # ── Circuit internals ────────────────────────────────────

    def _eval_expectation(self, params: np.ndarray) -> float:
        """⟨ψ(γ,β)| H_C |ψ(γ,β)⟩  — the objective for classical optimiser."""
        state = self._build_state(params)
        probs = np.abs(state) ** 2
        return float(probs @ self._costs)

    def _build_state(self, params: np.ndarray) -> np.ndarray:
        """Build the QAOA state |ψ(γ,β)⟩ for given parameters."""
        n = self._n
        N = 1 << n
        state = np.full(N, 1.0 / np.sqrt(N), dtype=np.complex128)  # |+⟩^n

        for layer in range(self.p):
            gamma = params[layer]
            beta = params[self.p + layer]

            # Cost unitary: |k⟩ → exp(−i γ E_k) |k⟩
            state *= np.exp(-1j * gamma * self._costs)

            # Mixer unitary: exp(−i β Σ X_q) = Π_q exp(−i β X_q)
            self._apply_mixer(state, beta, n)

        return state

    @staticmethod
    def _apply_mixer(state: np.ndarray, beta: float, n: int) -> None:
        """Apply mixer unitary via qubit-by-qubit X rotations (reshape trick)."""
        c = np.cos(beta)
        ms = -1j * np.sin(beta)
        for q in range(n):
            step = 1 << q
            view = state.reshape(-1, 2, step)
            s0 = view[:, 0, :].copy()
            s1 = view[:, 1, :].copy()
            view[:, 0, :] = c * s0 + ms * s1
            view[:, 1, :] = ms * s0 + c * s1
