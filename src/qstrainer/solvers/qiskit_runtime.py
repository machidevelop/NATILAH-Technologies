"""IBM Qiskit Runtime solver — gate-model quantum computing via IBM cloud.

Supports:
  - **Local Aer simulation** (for development, no IBM account needed)
  - **IBM Quantum Runtime** (real hardware + cloud simulators)

Requires: ``pip install qiskit qiskit-aer`` (sim)
          ``pip install qiskit-ibm-runtime`` (hardware)

The solver builds a QAOA circuit from the QUBO matrix, then dispatches
it to the selected backend.

Usage::

    from qstrainer.solvers.qiskit_runtime import QiskitRuntimeSolver
    solver = QiskitRuntimeSolver(backend="ibm_brisbane")
    result = solver.solve(Q)
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional

import numpy as np

from qstrainer.solvers.base import QUBOResult, QUBOSolverBase

logger = logging.getLogger(__name__)

# Feature-gated imports
_HAS_QISKIT = False
_HAS_AER = False
_HAS_RUNTIME = False

try:
    from qiskit.circuit import QuantumCircuit, Parameter
    from qiskit.quantum_info import SparsePauliOp

    _HAS_QISKIT = True
except ImportError:
    pass

try:
    from qiskit_aer import AerSimulator

    _HAS_AER = True
except ImportError:
    pass

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2

    _HAS_RUNTIME = True
except ImportError:
    pass


class QiskitRuntimeSolver(QUBOSolverBase):
    """QAOA solver targeting IBM backends via Qiskit.

    Parameters
    ----------
    backend : str
        Backend name: ``"aer"`` for local sim, or an IBM device name
        like ``"ibm_brisbane"`` / ``"ibm_sherbrooke"``.
    p : int
        Number of QAOA layers.
    shots : int
        Measurement shots per circuit execution.
    optimization_level : int
        Qiskit transpiler optimization level (0–3).
    instance : str or None
        IBM Quantum instance (hub/group/project) if using hardware.
    token : str or None
        IBM Quantum API token.  If None, uses saved credentials.
    max_qubits : int
        Safety cap.  QAOA on >127 qubits is generally impractical.
    """

    def __init__(
        self,
        backend: str = "aer",
        *,
        p: int = 1,
        shots: int = 4096,
        optimization_level: int = 1,
        instance: Optional[str] = None,
        token: Optional[str] = None,
        max_qubits: int = 127,
    ) -> None:
        if not _HAS_QISKIT:
            raise ImportError(
                "qiskit is required. Install with: pip install qiskit"
            )
        self._backend_name = backend
        self._p = p
        self._shots = shots
        self._opt_level = optimization_level
        self._instance = instance
        self._token = token
        self._max_qubits = max_qubits
        self._backend = None
        self._service = None

    @property
    def solver_type(self) -> str:
        if self._backend_name == "aer":
            return "quantum_sim"
        return "quantum_hw"

    def is_available(self) -> bool:
        if self._backend_name == "aer":
            return _HAS_AER
        return _HAS_RUNTIME

    # ── Build QAOA circuit ───────────────────────────────────

    def _build_qaoa_circuit(
        self, Q: np.ndarray, gammas: np.ndarray, betas: np.ndarray,
    ) -> "QuantumCircuit":
        """Construct a fixed-parameter QAOA circuit for the QUBO."""
        n = Q.shape[0]
        qc = QuantumCircuit(n)

        # Initial superposition
        qc.h(range(n))

        for layer in range(self._p):
            gamma = gammas[layer]
            beta = betas[layer]

            # Cost unitary: Z_i Z_j interactions + Z_i linear terms
            for i in range(n):
                for j in range(i + 1, n):
                    w = Q[i, j] + Q[j, i]
                    if abs(w) > 1e-10:
                        qc.cx(i, j)
                        qc.rz(gamma * w, j)
                        qc.cx(i, j)
                # Diagonal (linear) terms
                if abs(Q[i, i]) > 1e-10:
                    qc.rz(gamma * Q[i, i], i)

            # Mixer unitary
            for i in range(n):
                qc.rx(2 * beta, i)

        qc.measure_all()
        return qc

    # ── Solve ────────────────────────────────────────────────

    def solve(self, Q: np.ndarray) -> QUBOResult:
        n = Q.shape[0]
        if n > self._max_qubits:
            raise ValueError(
                f"Problem size {n} exceeds max_qubits={self._max_qubits}. "
                "Consider DWaveSolver for large problems."
            )

        t0 = time.perf_counter()

        # Simple parameter sweep (production would use a classical optimizer)
        rng = np.random.default_rng(42)
        best_solution = np.zeros(n, dtype=int)
        best_energy = float("inf")

        n_restarts = 3
        for _ in range(n_restarts):
            gammas = rng.uniform(0, 2 * np.pi, size=self._p)
            betas = rng.uniform(0, np.pi, size=self._p)

            qc = self._build_qaoa_circuit(Q, gammas, betas)
            counts = self._execute(qc)

            # Evaluate all measured bitstrings
            for bitstring, count in counts.items():
                bits = np.array([int(b) for b in reversed(bitstring)], dtype=int)
                if len(bits) != n:
                    continue
                energy = float(bits @ Q @ bits)
                if energy < best_energy:
                    best_energy = energy
                    best_solution = bits.copy()

        return QUBOResult(
            solution=best_solution,
            energy=best_energy,
            solver_name=f"qiskit_{self._backend_name}",
            solve_time_s=time.perf_counter() - t0,
            metadata={
                "backend": self._backend_name,
                "p": self._p,
                "shots": self._shots,
                "n_qubits": n,
            },
        )

    # ── Backend dispatch ─────────────────────────────────────

    def _execute(self, qc: "QuantumCircuit") -> Dict[str, int]:
        """Execute circuit and return measurement counts."""
        if self._backend_name == "aer":
            return self._execute_aer(qc)
        return self._execute_runtime(qc)

    def _execute_aer(self, qc: "QuantumCircuit") -> Dict[str, int]:
        """Local Aer simulator."""
        if not _HAS_AER:
            raise ImportError("qiskit-aer required. pip install qiskit-aer")
        sim = AerSimulator()
        from qiskit import transpile

        tqc = transpile(qc, sim, optimization_level=self._opt_level)
        result = sim.run(tqc, shots=self._shots).result()
        return dict(result.get_counts())

    def _execute_runtime(self, qc: "QuantumCircuit") -> Dict[str, int]:
        """IBM Quantum Runtime execution."""
        if not _HAS_RUNTIME:
            raise ImportError(
                "qiskit-ibm-runtime required. pip install qiskit-ibm-runtime"
            )

        if self._service is None:
            kwargs = {}
            if self._token:
                kwargs["token"] = self._token
            if self._instance:
                kwargs["instance"] = self._instance
            self._service = QiskitRuntimeService(**kwargs)

        backend = self._service.backend(self._backend_name)
        from qiskit import transpile

        tqc = transpile(qc, backend, optimization_level=self._opt_level)

        with Session(service=self._service, backend=backend) as session:
            sampler = SamplerV2(session=session)
            job = sampler.run([tqc], shots=self._shots)
            result = job.result()

        # Extract counts from SamplerV2 result
        pub_result = result[0]
        counts: Dict[str, int] = {}
        if hasattr(pub_result, "data"):
            # SamplerV2 returns BitArray in data.meas
            meas = pub_result.data.meas
            if hasattr(meas, "get_counts"):
                counts = dict(meas.get_counts())
        return counts

    @classmethod
    def from_config(cls, cfg: dict) -> "QiskitRuntimeSolver":
        """Build from config dict."""
        qc = cfg.get("solvers", {}).get("qiskit", {})
        return cls(
            backend=qc.get("backend", "aer"),
            p=qc.get("p", 1),
            shots=qc.get("shots", 4096),
            optimization_level=qc.get("optimization_level", 1),
            instance=qc.get("instance"),
            token=qc.get("token"),
            max_qubits=qc.get("max_qubits", 127),
        )
