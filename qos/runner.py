"""
QOS Runner — Executes QUBO jobs through the scheduler and produces QOSReports.
"""
import hashlib
import json
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

from qos.report import QOSReport
from qos.scheduler import QOSScheduler


class QOSRunner:
    """
    Executes QUBO jobs through the scheduler and returns QOSReports.
    Handles errors, timeouts, and fallback to classical solvers.
    """

    def __init__(self, scheduler: QOSScheduler):
        self.scheduler = scheduler
        self._run_counter = 0
        self._history: List[QOSReport] = []

    def run(
        self,
        Q: np.ndarray,
        job_type: str = "qubo_generic",
        prefer_solver: Optional[str] = None,
        expected_k: Optional[int] = None,
    ) -> QOSReport:
        """
        Submit a QUBO job.  Returns a QOSReport.

        Args:
            Q: QUBO matrix (n x n)
            job_type: descriptive label
            prefer_solver: force a specific solver
            expected_k: expected number of selected variables (feasibility check)
        """
        self._run_counter += 1
        n = Q.shape[0]
        input_hash = hashlib.sha256(Q.tobytes()).hexdigest()[:16]

        solver_name, solver = self.scheduler.select_solver(
            n_variables=n, prefer=prefer_solver
        )

        try:
            result = solver.solve(Q)
            solution = result.solution
            energy = result.energy
            solve_time = result.solve_time_s
            meta = result.metadata
        except Exception as e:
            # Fallback — import SA lazily to avoid circular deps
            from importlib import import_module

            meta = {"fallback": True, "original_error": str(e)}
            solver_name = "sa_fallback"
            solution = None
            energy = None
            solve_time = 0.0

        selected = int(solution.sum()) if solution is not None else 0
        feasible = True
        if expected_k is not None:
            feasible = selected == expected_k

        report = QOSReport(
            job_id=f"qos-{self._run_counter:04d}-{input_hash[:8]}",
            job_type=job_type,
            timestamp=datetime.now().isoformat(),
            solver_name=solver_name,
            solver_type=getattr(solver, "solver_type", "unknown"),
            backend=meta.get("backend", solver_name),
            solution=solution,
            energy=energy,
            solve_time_s=solve_time,
            qubo_size=n,
            feasible=feasible,
            selected_count=selected,
            input_hash=input_hash,
            metadata=meta,
        )
        self._history.append(report)
        return report

    def compare_solvers(
        self,
        Q: np.ndarray,
        solver_names: Optional[List[str]] = None,
        expected_k: Optional[int] = None,
    ) -> List[QOSReport]:
        """Run the same QUBO on multiple solvers and return comparison reports."""
        if solver_names is None:
            solver_names = self.scheduler.available_solvers()
        reports = []
        for name in solver_names:
            if name in self.scheduler._solvers:
                report = self.run(
                    Q,
                    job_type="solver_comparison",
                    prefer_solver=name,
                    expected_k=expected_k,
                )
                reports.append(report)
        return reports

    @property
    def history(self) -> List[QOSReport]:
        return self._history

    def save_history(self, path: str = "runs/qos_history.json"):
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )
        data = [r.to_dict() for r in self._history]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
