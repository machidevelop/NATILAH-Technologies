"""
QOS Scheduler — Routes QUBO jobs to the best available solver.
"""
from typing import Dict, List, Optional, Tuple


class QOSScheduler:
    """
    Routes QUBO jobs to the best available solver based on:
    - Problem size (n_variables)
    - Available backends
    - Time/quality constraints

    Decision tree:
      n <= 18  -> QAOA (statevector sim) — exact quantum simulation
      n <= 50  -> SA (fast, reliable baseline)
      n > 50   -> D-Wave if available, else SA with more reads
      any      -> User can force a specific solver
    """

    def __init__(self):
        self._solvers: Dict[str, object] = {}
        self._preference_order: List[Tuple[int, str]] = []

    def register_solver(self, name: str, solver: object, priority: int = 50):
        """Register a solver backend.  Lower priority number = preferred."""
        self._solvers[name] = solver
        self._preference_order.append((priority, name))
        self._preference_order.sort()

    def available_solvers(self) -> List[str]:
        return [name for _, name in self._preference_order]

    def select_solver(
        self,
        n_variables: int,
        prefer: Optional[str] = None,
        max_time_s: Optional[float] = None,
    ) -> Tuple[str, object]:
        """
        Select the best solver for a QUBO of given size.
        Returns (solver_name, solver_instance).
        """
        # User override
        if prefer and prefer in self._solvers:
            return prefer, self._solvers[prefer]

        for _, name in self._preference_order:
            solver = self._solvers[name]
            stype = getattr(solver, "solver_type", "classical")

            # QAOA sim: only for small problems
            if stype == "quantum_sim" and n_variables <= 18:
                return name, solver

            # Quantum hardware: preferred for large problems
            if stype == "quantum_hw" and n_variables > 18:
                if hasattr(solver, "is_available") and not solver.is_available():
                    continue
                return name, solver

            # Classical: always works
            if stype == "classical":
                return name, solver

        # Fallback
        name = self._preference_order[0][1]
        return name, self._solvers[name]
