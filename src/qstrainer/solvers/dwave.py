"""D-Wave Advantage solver — real quantum annealing hardware.

Requires: pip install dwave-ocean-sdk
          dwave config create --auto-token YOUR_TOKEN
"""

from __future__ import annotations

import logging
import time

import numpy as np

from qstrainer.solvers.base import QUBOResult, QUBOSolverBase

logger = logging.getLogger(__name__)


class DWaveSolver(QUBOSolverBase):
    """D-Wave Advantage system via Ocean SDK.

    Same interface as SimulatedAnnealingSolver — swap freely.
    """

    def __init__(self, num_reads: int = 1000, annealing_time_us: int = 20) -> None:
        self.num_reads = num_reads
        self.annealing_time_us = annealing_time_us
        self._sampler = None

    @property
    def solver_type(self) -> str:
        return "quantum_hw"

    def _connect(self) -> None:
        if self._sampler is not None:
            return
        from dwave.system import DWaveSampler, EmbeddingComposite

        base = DWaveSampler()
        self._sampler = EmbeddingComposite(base)
        chip = base.properties.get("chip_id", "unknown")
        qubits = base.properties.get("num_qubits", "?")
        logger.info("Connected to D-Wave: %s, %s qubits", chip, qubits)

    def solve(self, Q: np.ndarray) -> QUBOResult:
        self._connect()
        n = Q.shape[0]

        qubo_dict = {}
        for i in range(n):
            for j in range(i, n):
                if abs(Q[i, j]) > 1e-10:
                    qubo_dict[(i, j)] = float(Q[i, j])

        max_w = max(abs(v) for v in qubo_dict.values()) if qubo_dict else 1.0

        t0 = time.perf_counter()
        assert self._sampler is not None
        response = self._sampler.sample_qubo(
            qubo_dict,
            num_reads=self.num_reads,
            annealing_time=self.annealing_time_us,
            chain_strength=max_w * 1.5,
            label=f"q_strainer_feat_select_{n}",
        )
        solve_time = time.perf_counter() - t0

        best = response.first
        solution = np.array([best.sample[i] for i in range(n)])
        timing = response.info.get("timing", {})

        return QUBOResult(
            solution=solution.astype(int),
            energy=float(best.energy),
            solver_name="dwave_advantage",
            solve_time_s=solve_time,
            metadata={
                "num_reads": self.num_reads,
                "qpu_access_time_us": timing.get("qpu_access_time"),
                "num_logical_qubits": n,
                "backend": "dwave_advantage",
            },
        )

    def is_available(self) -> bool:
        try:
            from dwave.cloud import Client

            client = Client.from_config()
            return len(client.get_solvers()) > 0
        except Exception:
            return False
