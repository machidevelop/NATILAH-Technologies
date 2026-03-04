"""Tests for QOS — Scheduler, Runner, Report."""

from __future__ import annotations

import numpy as np
import pytest

from qstrainer.qos.report import QOSReport
from qstrainer.qos.scheduler import QOSScheduler
from qstrainer.qos.runner import QOSRunner
from qstrainer.solvers.sa import SimulatedAnnealingSolver
from qstrainer.solvers.mock import MockQuantumSolver


def _random_qubo(n: int = 10) -> np.ndarray:
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((n, n))
    return (Q + Q.T) / 2


class TestQOSScheduler:
    def test_register_and_select(self):
        sched = QOSScheduler()
        sched.register_solver("sa", SimulatedAnnealingSolver(), priority=10)
        name, solver = sched.select_solver(n_variables=10)
        assert name == "sa"

    def test_prefer_override(self):
        sched = QOSScheduler()
        sched.register_solver("sa", SimulatedAnnealingSolver(), priority=10)
        sched.register_solver("mock", MockQuantumSolver(), priority=20)
        name, _ = sched.select_solver(n_variables=10, prefer="mock")
        assert name == "mock"

    def test_available_solvers(self):
        sched = QOSScheduler()
        sched.register_solver("sa", SimulatedAnnealingSolver(), priority=10)
        sched.register_solver("mock", MockQuantumSolver(), priority=20)
        assert sched.available_solvers() == ["sa", "mock"]


class TestQOSRunner:
    def test_run_produces_report(self):
        sched = QOSScheduler()
        sched.register_solver("sa", SimulatedAnnealingSolver(num_reads=5, num_sweeps=50))
        runner = QOSRunner(sched)
        Q = _random_qubo(6)
        report = runner.run(Q, job_type="test")
        assert isinstance(report, QOSReport)
        assert report.qubo_size == 6

    def test_history(self):
        sched = QOSScheduler()
        sched.register_solver("sa", SimulatedAnnealingSolver(num_reads=5, num_sweeps=50))
        runner = QOSRunner(sched)
        Q = _random_qubo(6)
        runner.run(Q)
        runner.run(Q)
        assert len(runner.history) == 2

    def test_compare_solvers(self):
        sched = QOSScheduler()
        sched.register_solver("sa", SimulatedAnnealingSolver(num_reads=5, num_sweeps=50))
        sched.register_solver("mock", MockQuantumSolver(num_reads=5, num_sweeps=50))
        runner = QOSRunner(sched)
        Q = _random_qubo(6)
        reports = runner.compare_solvers(Q, solver_names=["sa", "mock"])
        assert len(reports) == 2


class TestQOSReport:
    def test_to_dict_roundtrip(self):
        report = QOSReport(
            job_id="test-001",
            job_type="test",
            timestamp="2024-01-01T00:00:00",
            solver_name="sa",
            solver_type="classical",
            backend="numpy",
            solution=np.array([1, 0, 1, 0]),
            energy=-1.5,
            solve_time_s=0.01,
            qubo_size=4,
            feasible=True,
            selected_count=2,
            input_hash="abcdef1234567890",
        )
        d = report.to_dict()
        assert d["solution"] == [1, 0, 1, 0]
        restored = QOSReport.from_dict(d)
        assert np.array_equal(restored.solution, report.solution)

    def test_summary(self):
        report = QOSReport(
            job_id="t", job_type="t", timestamp="t",
            solver_name="sa", solver_type="classical", backend="numpy",
            solution=np.array([1]), energy=-1.0, solve_time_s=0.1,
            qubo_size=1, feasible=True, selected_count=1, input_hash="x",
        )
        s = report.summary()
        assert "classical" in s
        assert "FEASIBLE" in s
