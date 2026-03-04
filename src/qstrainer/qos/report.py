"""QOSReport — standardised output from any quantum/classical job execution.

Every run — SA, QAOA, D-Wave, IBM — produces the same report format.
Enables reproducibility, comparison, and auditing.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class QOSReport:
    """Standardised report for any quantum/classical job execution."""

    # Identity
    job_id: str
    job_type: str  # "qubo_feature_selection", "kernel_eval", etc.
    timestamp: str

    # Solver info
    solver_name: str
    solver_type: str  # "classical", "quantum_sim", "quantum_hw"
    backend: str  # "numpy", "dwave_advantage", "ibm_eagle", etc.

    # Results
    solution: np.ndarray | None
    energy: float | None
    solve_time_s: float

    # Quality metrics
    qubo_size: int
    feasible: bool  # does solution satisfy constraints?
    selected_count: int | None  # for feature selection

    # Reproducibility
    input_hash: str  # SHA-256 prefix of QUBO matrix
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── Serialisation ───────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["solution"] is not None:
            d["solution"] = list(map(int, d["solution"]))
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def save(self, path: str) -> None:
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())

    def summary(self) -> str:
        feas = "FEASIBLE" if self.feasible else "INFEASIBLE"
        energy_str = f"{self.energy:.4f}" if self.energy is not None else "N/A"
        return (
            f"[{self.solver_type}] {self.solver_name} | "
            f"energy={energy_str} | time={self.solve_time_s:.3f}s | "
            f"n={self.qubo_size} | {feas}"
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QOSReport:
        """Reconstruct from a dict (e.g. loaded from JSON)."""
        sol = d.get("solution")
        if sol is not None:
            sol = np.array(sol, dtype=int)
        return cls(
            job_id=d["job_id"],
            job_type=d["job_type"],
            timestamp=d["timestamp"],
            solver_name=d["solver_name"],
            solver_type=d["solver_type"],
            backend=d["backend"],
            solution=sol,
            energy=d.get("energy"),
            solve_time_s=d["solve_time_s"],
            qubo_size=d["qubo_size"],
            feasible=d["feasible"],
            selected_count=d.get("selected_count"),
            input_hash=d["input_hash"],
            metadata=d.get("metadata", {}),
        )
