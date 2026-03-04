"""QOSScheduler — routes QUBO jobs to the best available solver.

Decision tree (auto mode):
    n <= 18  → QAOA statevector sim (exact quantum simulation)
    n <= 50  → SA (fast, reliable baseline)
    n <= 127 → Qiskit Runtime if available, else SA heavy
    n  > 127 → D-Wave if available, else SA heavy
    any      → User can force a specific solver

Hybrid scheduling: for problems in the 18–127 qubit range, the
scheduler can split the problem or route to the fastest available
quantum backend (Qiskit Runtime > QAOA sim > D-Wave > SA).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from qstrainer.solvers.base import QUBOSolverBase

logger = logging.getLogger(__name__)


class QOSScheduler:
    """Routes QUBO jobs to the best available solver."""

    def __init__(self) -> None:
        self._solvers: Dict[str, QUBOSolverBase] = {}
        self._preference_order: List[Tuple[int, str]] = []

    def register_solver(
        self, name: str, solver: QUBOSolverBase, priority: int = 50
    ) -> None:
        """Register a solver backend.  Lower priority = preferred."""
        self._solvers[name] = solver
        self._preference_order.append((priority, name))
        self._preference_order.sort()

    def available_solvers(self) -> List[str]:
        return [name for _, name in self._preference_order]

    def get_solver(self, name: str) -> Optional[QUBOSolverBase]:
        return self._solvers.get(name)

    def select_solver(
        self,
        n_variables: int,
        prefer: Optional[str] = None,
        max_time_s: Optional[float] = None,
    ) -> Tuple[str, QUBOSolverBase]:
        """Select the best solver for a QUBO of given size.

        Returns (solver_name, solver_instance).

        Hybrid routing logic:
          - n <= 18:  QAOA statevector sim (exact)
          - 18 < n <= 127: Qiskit Runtime (if available) > SA
          - n > 127: D-Wave (if available) > SA heavy
        """
        # User override
        if prefer and prefer in self._solvers:
            return prefer, self._solvers[prefer]

        # Auto-select based on problem size
        for _, name in self._preference_order:
            solver = self._solvers[name]
            stype = solver.solver_type

            # QAOA sim: only for small problems (≤18 qubits)
            if stype == "quantum_sim" and n_variables <= 18:
                if hasattr(solver, "is_available") and not solver.is_available():
                    continue
                return name, solver

            # Qiskit Runtime: medium problems (18 < n ≤ 127)
            if stype == "quantum_hw" and name.startswith("qiskit") and 18 < n_variables <= 127:
                if hasattr(solver, "is_available") and not solver.is_available():
                    continue
                return name, solver

            # D-Wave quantum hardware: large problems (n > 127)
            if stype == "quantum_hw" and n_variables > 127:
                if hasattr(solver, "is_available") and not solver.is_available():
                    continue
                return name, solver

            # Classical: always works as fallback
            if stype == "classical":
                return name, solver

        # Fallback: first registered solver
        name = self._preference_order[0][1]
        return name, self._solvers[name]

    @classmethod
    def from_config(cls, cfg: dict) -> "QOSScheduler":
        """Build a scheduler from a config dict, registering default solvers."""
        from qstrainer.solvers.sa import SimulatedAnnealingSolver
        from qstrainer.solvers.qaoa import QAOASolver
        from qstrainer.solvers.dwave import DWaveSolver
        from qstrainer.solvers.mock import MockQuantumSolver

        sched = cls()

        solvers_cfg = cfg.get("solvers", {})

        # QAOA
        qaoa_cfg = solvers_cfg.get("qaoa", {})
        sched.register_solver(
            "qaoa_sim",
            QAOASolver(
                p=qaoa_cfg.get("p", 2),
                n_restarts=qaoa_cfg.get("n_restarts", 6),
                seed=qaoa_cfg.get("seed", 42),
            ),
            priority=10,
        )

        # SA default
        sa_cfg = solvers_cfg.get("sa", {})
        sched.register_solver(
            "sa_default",
            SimulatedAnnealingSolver(
                num_reads=sa_cfg.get("num_reads", 300),
                num_sweeps=sa_cfg.get("num_sweeps", 1500),
            ),
            priority=20,
        )

        # SA heavy
        sched.register_solver(
            "sa_heavy",
            SimulatedAnnealingSolver(
                num_reads=sa_cfg.get("num_reads_heavy", 1000),
                num_sweeps=sa_cfg.get("num_sweeps_heavy", 3000),
            ),
            priority=30,
        )

        # Mock quantum (for testing)
        sched.register_solver(
            "mock_quantum",
            MockQuantumSolver(num_reads=300, num_sweeps=1500),
            priority=40,
        )

        # D-Wave (highest priority but checks availability)
        dwave_cfg = solvers_cfg.get("dwave", {})
        sched.register_solver(
            "dwave",
            DWaveSolver(num_reads=dwave_cfg.get("num_reads", 500)),
            priority=5,
        )

        # Qiskit Runtime (for medium-to-large gate-model problems)
        try:
            from qstrainer.solvers.qiskit_runtime import QiskitRuntimeSolver

            qiskit_cfg = solvers_cfg.get("qiskit", {})
            sched.register_solver(
                "qiskit_runtime",
                QiskitRuntimeSolver(
                    backend=qiskit_cfg.get("backend", "aer"),
                    p=qiskit_cfg.get("p", 1),
                    shots=qiskit_cfg.get("shots", 4096),
                    optimization_level=qiskit_cfg.get("optimization_level", 1),
                    instance=qiskit_cfg.get("instance"),
                    token=qiskit_cfg.get("token"),
                ),
                priority=8,  # between D-Wave (5) and QAOA sim (10)
            )
        except ImportError:
            logger.debug("Qiskit not installed — skipping qiskit_runtime solver")

        return sched
