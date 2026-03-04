"""QOSRunner — executes QUBO jobs through the scheduler and returns QOSReports.

Handles errors, timeouts, and automatic fallback to classical SA.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from qstrainer.qos.report import QOSReport
from qstrainer.qos.scheduler import QOSScheduler
from qstrainer.solvers.sa import SimulatedAnnealingSolver

logger = logging.getLogger(__name__)


class QOSRunner:
    """Execute QUBO jobs through the scheduler and produce QOSReports."""

    def __init__(self, scheduler: QOSScheduler) -> None:
        self.scheduler = scheduler
        self._run_counter: int = 0
        self._history: List[QOSReport] = []

    def run(
        self,
        Q: np.ndarray,
        job_type: str = "qubo_generic",
        prefer_solver: Optional[str] = None,
        expected_k: Optional[int] = None,
    ) -> QOSReport:
        """Submit a QUBO job.  Returns a :class:`QOSReport`.

        Parameters
        ----------
        Q : np.ndarray
            QUBO matrix (n x n).
        job_type : str
            Descriptive label (e.g. ``"feature_selection"``).
        prefer_solver : str, optional
            Force a specific solver name.
        expected_k : int, optional
            Expected number of selected variables (for feasibility check).
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
        except Exception as exc:
            logger.warning(
                "Solver %s failed: %s. Falling back to SA.", solver_name, exc
            )
            fallback = SimulatedAnnealingSolver(num_reads=300, num_sweeps=2000)
            result = fallback.solve(Q)
            solution = result.solution
            energy = result.energy
            solve_time = result.solve_time_s
            meta = {
                **result.metadata,
                "fallback": True,
                "original_error": str(exc),
            }
            solver_name = "sa_fallback"

        selected = int(solution.sum()) if solution is not None else 0
        feasible = True
        if expected_k is not None:
            feasible = selected == expected_k

        report = QOSReport(
            job_id=f"qos-{self._run_counter:04d}-{input_hash[:8]}",
            job_type=job_type,
            timestamp=datetime.now().isoformat(),
            solver_name=solver_name,
            solver_type=(
                solver.solver_type if hasattr(solver, "solver_type") else "unknown"
            ),
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

        reports: List[QOSReport] = []
        for name in solver_names:
            if self.scheduler.get_solver(name) is not None:
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
        return list(self._history)

    def save_history(self, path: str = "runs/qos_history.json") -> None:
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        data = [r.to_dict() for r in self._history]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load_history(self, path: str = "runs/qos_history.json") -> None:
        """Load previous run history from JSON."""
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        for d in data:
            self._history.append(QOSReport.from_dict(d))
        self._run_counter = len(self._history)
