"""
QOS Report — Standardized output from any quantum/classical job execution.
"""
import json
import os
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional


@dataclass
class QOSReport:
    """
    Standardized report for any quantum/classical job execution.
    Every run — SA, QAOA, D-Wave, IBM — produces the same report format.
    Enables reproducibility, comparison, and auditing.
    """
    # Identity
    job_id: str
    # "qubo_feature_selection", "kernel_eval", etc.
    job_type: str
    timestamp: str

    # Solver info
    solver_name: str
    solver_type: str                     # "classical", "quantum_sim", "quantum_hw"
    # "numpy", "dwave_advantage", "ibm_eagle", etc.
    backend: str

    # Results
    solution: Optional[np.ndarray]
    energy: Optional[float]
    solve_time_s: float

    # Quality metrics
    qubo_size: int
    feasible: bool                       # does solution satisfy constraints?
    selected_count: Optional[int]        # for feature selection

    # Reproducibility
    input_hash: str                      # SHA-256 of QUBO matrix
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        if d["solution"] is not None:
            d["solution"] = d["solution"].tolist()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(
            path) else ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_json(cls, path: str) -> "QOSReport":
        with open(path, "r") as f:
            d = json.load(f)
        if d.get("solution") is not None:
            d["solution"] = np.array(d["solution"])
        return cls(**d)

    def summary(self) -> str:
        feas = "PASS" if self.feasible else "FAIL"
        return (
            f"[{self.solver_type}] {self.solver_name} | "
            f"energy={self.energy:.4f} | time={self.solve_time_s:.3f}s | "
            f"n={self.qubo_size} | {feas}"
        )
