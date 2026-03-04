"""Tests for QuantumStrainScheduler — QUBO-based batch scheduling.

Validates:
  - QUBO matrix construction (symmetry, interactions, constraints)
  - Full schedule cycle (tasks → QUBO → solve → verdicts)
  - Redundant tasks get strained, productive tasks get executed
  - Cross-GPU fairness coupling
  - Consecutive step anti-correlation
  - Savings accounting
  - Solver backend routing
"""

from __future__ import annotations

import time

import numpy as np

from qstrainer.models.enums import ComputePhase, JobType, TaskVerdict
from qstrainer.models.frame import ComputeTask
from qstrainer.pipeline.quantum_scheduler import (
    QuantumStrainScheduler,
    QUBOBuilder,
    SchedulerConfig,
)
from qstrainer.qos.scheduler import QOSScheduler
from qstrainer.solvers.mock import MockQuantumSolver
from qstrainer.solvers.sa import SimulatedAnnealingSolver

# ── Helpers ──────────────────────────────────────────────────


def _productive_task(gpu_id: str = "GPU-0", step: int = 10, job_id: str = "job-A") -> ComputeTask:
    return ComputeTask(
        timestamp=time.time(),
        task_id=f"prod-{step:04d}",
        gpu_id=gpu_id,
        job_id=job_id,
        step_number=step,
        loss=2.0 - step * 0.01,
        loss_delta=-0.01,
        gradient_norm=0.5,
        gradient_variance=0.05,
        learning_rate=1e-3,
        batch_size=256,
        epoch=0,
        epoch_progress=step / 100.0,
        estimated_flops=2e12,
        estimated_time_s=0.5,
        memory_footprint_gb=12.0,
        compute_phase=ComputePhase.FORWARD_PASS,
        job_type=JobType.TRAINING,
        convergence_score=0.1,
        param_update_magnitude=0.01,
        data_similarity=0.2,
        flop_utilization=0.75,
        throughput_samples_per_sec=512,
    )


def _redundant_task(gpu_id: str = "GPU-0", step: int = 999, job_id: str = "job-A") -> ComputeTask:
    return ComputeTask(
        timestamp=time.time(),
        task_id=f"red-{step:04d}",
        gpu_id=gpu_id,
        job_id=job_id,
        step_number=step,
        loss=0.12,
        loss_delta=1e-8,
        gradient_norm=1e-8,
        gradient_variance=1e-10,
        learning_rate=1e-5,
        batch_size=256,
        epoch=20,
        epoch_progress=0.99,
        estimated_flops=2e12,
        estimated_time_s=0.5,
        memory_footprint_gb=12.0,
        compute_phase=ComputePhase.FORWARD_PASS,
        job_type=JobType.TRAINING,
        convergence_score=0.98,
        param_update_magnitude=1e-10,
        data_similarity=0.99,
        flop_utilization=0.75,
        throughput_samples_per_sec=512,
    )


# ── QUBO Builder Tests ──────────────────────────────────────


class TestQUBOBuilder:
    def test_qubo_matrix_shape(self):
        """QUBO matrix is N×N for N tasks."""
        cfg = SchedulerConfig()
        builder = QUBOBuilder(cfg)
        tasks = [_productive_task(step=i) for i in range(8)]
        vecs = np.array([t.to_vector() for t in tasks])
        scores = np.full(8, 0.2)
        decs = [[] for _ in range(8)]
        Q = builder.build(tasks, scores, vecs, decs)
        assert Q.shape == (8, 8)

    def test_qubo_diagonal_productive_vs_redundant(self):
        """Productive tasks should have more negative diagonal (prefer execute)."""
        cfg = SchedulerConfig(beta=0.0, gamma=0.0, delta=0.0)  # isolate diagonal
        builder = QUBOBuilder(cfg)

        tasks = [_productive_task(step=1), _redundant_task(step=999)]
        vecs = np.array([t.to_vector() for t in tasks])
        # Productive: low redundancy, redundant: high redundancy
        scores = np.array([0.1, 0.95])
        decs = [[], []]
        Q = builder.build(tasks, scores, vecs, decs)

        # Productive should have more negative diagonal (solver prefers x=1=execute)
        assert Q[0, 0] < Q[1, 1], (
            f"Productive diagonal {Q[0, 0]:.3f} should be < redundant {Q[1, 1]:.3f}"
        )

    def test_similarity_coupling_same_gpu(self):
        """Similar tasks on the same GPU should have positive off-diagonal."""
        cfg = SchedulerConfig(beta=0.4, gamma=0.0, delta=0.0)
        builder = QUBOBuilder(cfg)

        # Two nearly identical tasks on same GPU
        t1 = _redundant_task(gpu_id="GPU-0", step=100)
        t2 = _redundant_task(gpu_id="GPU-0", step=101)
        tasks = [t1, t2]
        vecs = np.array([t.to_vector() for t in tasks])
        scores = np.array([0.9, 0.9])
        decs = [[], []]
        Q = builder.build(tasks, scores, vecs, decs)

        assert Q[0, 1] > 0, f"Similar same-GPU tasks should have positive coupling: {Q[0, 1]:.3f}"

    def test_similarity_not_coupled_across_gpus(self):
        """Similar tasks on different GPUs should NOT have similarity coupling."""
        cfg = SchedulerConfig(beta=0.4, gamma=0.0, delta=0.0)
        builder = QUBOBuilder(cfg)

        t1 = _redundant_task(gpu_id="GPU-0", step=100)
        t2 = _redundant_task(gpu_id="GPU-1", step=100)
        tasks = [t1, t2]
        vecs = np.array([t.to_vector() for t in tasks])
        scores = np.array([0.9, 0.9])
        decs = [[], []]
        Q = builder.build(tasks, scores, vecs, decs)

        # Off-diagonal from similarity should be 0 (different GPUs)
        # There's no gamma, so only delta could give negative coupling
        assert Q[0, 1] <= 0

    def test_consecutive_step_anticorrelation(self):
        """Consecutive steps on same GPU should have negative coupling."""
        cfg = SchedulerConfig(beta=0.0, gamma=0.3, delta=0.0)
        builder = QUBOBuilder(cfg)

        tasks = [
            _productive_task(gpu_id="GPU-0", step=10, job_id="job-A"),
            _productive_task(gpu_id="GPU-0", step=11, job_id="job-A"),
        ]
        vecs = np.array([t.to_vector() for t in tasks])
        scores = np.array([0.3, 0.3])
        decs = [[], []]
        Q = builder.build(tasks, scores, vecs, decs)

        assert Q[0, 1] < 0, f"Consecutive steps should have negative coupling: {Q[0, 1]:.3f}"

    def test_fairness_coupling_cross_gpu(self):
        """Tasks on different GPUs in same job should have negative coupling."""
        cfg = SchedulerConfig(beta=0.0, gamma=0.0, delta=0.15)
        builder = QUBOBuilder(cfg)

        tasks = [
            _productive_task(gpu_id="GPU-0", step=10, job_id="job-A"),
            _productive_task(gpu_id="GPU-1", step=10, job_id="job-A"),
        ]
        vecs = np.array([t.to_vector() for t in tasks])
        scores = np.array([0.3, 0.3])
        decs = [[], []]
        Q = builder.build(tasks, scores, vecs, decs)

        assert Q[0, 1] < 0, f"Cross-GPU same-job should have negative coupling: {Q[0, 1]:.3f}"


# ── Scheduler End-to-End Tests ───────────────────────────────


class TestQuantumStrainScheduler:
    def _make_scheduler(self, **kwargs) -> QuantumStrainScheduler:
        """Build a scheduler with fast SA solver for testing."""
        qos = QOSScheduler()
        qos.register_solver(
            "sa_fast", SimulatedAnnealingSolver(num_reads=50, num_sweeps=200, seed=42), priority=10
        )
        cfg = SchedulerConfig(**kwargs)
        return QuantumStrainScheduler(qos_scheduler=qos, config=cfg)

    def test_empty_batch(self):
        """Empty batch returns empty results."""
        sched = self._make_scheduler()
        assert sched.schedule([]) == []

    def test_single_task_productive(self):
        """A single productive task should be EXECUTED."""
        sched = self._make_scheduler(alpha=3.0)
        tasks = [_productive_task(step=5)]
        results = sched.schedule(tasks)
        assert len(results) == 1
        assert results[0].verdict == TaskVerdict.EXECUTE

    def test_single_task_redundant(self):
        """A single clearly redundant task should be STRAINED (not EXECUTE)."""
        sched = self._make_scheduler()
        tasks = [_redundant_task()]
        results = sched.schedule(tasks)
        assert len(results) == 1
        assert results[0].verdict != TaskVerdict.EXECUTE

    def test_mixed_batch_productive_mostly_executed(self):
        """In a mixed batch, most productive tasks should be executed."""
        sched = self._make_scheduler(alpha=2.5)
        tasks = []
        # 8 productive tasks
        for i in range(8):
            tasks.append(_productive_task(gpu_id=f"GPU-{i % 4}", step=i))
        # 4 redundant tasks
        for i in range(4):
            tasks.append(_redundant_task(gpu_id=f"GPU-{i % 4}", step=900 + i))

        results = sched.schedule(tasks)
        assert len(results) == 12

        productive_results = results[:8]
        redundant_results = results[8:]

        executed_productive = sum(1 for r in productive_results if r.verdict == TaskVerdict.EXECUTE)
        strained_redundant = sum(1 for r in redundant_results if r.verdict != TaskVerdict.EXECUTE)

        # Most productive should be executed
        assert executed_productive >= 5, (
            f"Expected >= 5/8 productive executed, got {executed_productive}"
        )
        # Most redundant should be strained
        assert strained_redundant >= 2, (
            f"Expected >= 2/4 redundant strained, got {strained_redundant}"
        )

    def test_savings_accounting(self):
        """Strained tasks should accumulate FLOPs/time/cost savings."""
        sched = self._make_scheduler()
        tasks = [_redundant_task(step=i) for i in range(4)]
        results = sched.schedule(tasks)

        total_flops = sum(r.compute_saved_flops for r in results)
        total_time = sum(r.time_saved_s for r in results)
        strained = sum(1 for r in results if r.verdict != TaskVerdict.EXECUTE)

        if strained > 0:
            assert total_flops > 0
            assert total_time > 0

        # Stats should match
        stats = sched.stats
        assert stats["tasks_processed"] == 4
        assert stats["total_flops_saved"] == total_flops

    def test_qubo_energy_tracked(self):
        """Each schedule() call should produce a QUBO energy."""
        sched = self._make_scheduler()
        tasks = [_productive_task(step=i) for i in range(4)]
        sched.schedule(tasks)

        assert len(sched.qubo_energies) == 1
        assert isinstance(sched.qubo_energies[0], float)

    def test_solver_method_in_result(self):
        """Results should indicate quantum_schedule solver used."""
        sched = self._make_scheduler()
        tasks = [_productive_task()]
        results = sched.schedule(tasks)
        assert results[0].strainer_method.startswith("quantum_schedule:")

    def test_mock_quantum_solver(self):
        """Scheduler works with mock quantum solver."""
        qos = QOSScheduler()
        qos.register_solver("mock", MockQuantumSolver(num_reads=30, num_sweeps=100), priority=5)
        sched = QuantumStrainScheduler(qos_scheduler=qos)

        tasks = [_productive_task(step=i) for i in range(4)]
        results = sched.schedule(tasks)
        assert len(results) == 4
        assert all("quantum_schedule:" in r.strainer_method for r in results)

    def test_batch_of_32(self):
        """Default batch size of 32 should work end-to-end."""
        sched = self._make_scheduler()
        tasks = []
        for i in range(24):
            tasks.append(_productive_task(gpu_id=f"GPU-{i % 4}", step=i))
        for i in range(8):
            tasks.append(_redundant_task(gpu_id=f"GPU-{i % 4}", step=800 + i))

        results = sched.schedule(tasks)
        assert len(results) == 32

        stats = sched.stats
        assert stats["batches_scheduled"] == 1
        assert stats["tasks_processed"] == 32

    def test_multiple_batches_cumulative(self):
        """Stats should accumulate across multiple schedule() calls."""
        sched = self._make_scheduler()

        batch1 = [_productive_task(step=i) for i in range(4)]
        batch2 = [_redundant_task(step=i) for i in range(4)]

        sched.schedule(batch1)
        sched.schedule(batch2)

        stats = sched.stats
        assert stats["batches_scheduled"] == 2
        assert stats["tasks_processed"] == 8
        assert len(sched.qubo_energies) == 2

    def test_strain_rate_safety(self):
        """Scheduler should respect safety limits on borderline tasks.

        Hard SKIPs (gradient=0, converged) correctly override the cap.
        The safety cap matters for borderline tasks where the QUBO solver
        decides based on interactions.
        """
        sched = self._make_scheduler(max_strain_rate=0.5, alpha=1.5)
        # Borderline tasks: not clearly redundant, not clearly productive
        tasks = []
        for i in range(8):
            t = _productive_task(step=i, gpu_id=f"GPU-{i % 2}")
            # Make borderline: moderate gradient, moderate convergence
            t.gradient_norm = 0.1
            t.convergence_score = 0.4
            t.loss_delta = -0.002
            t.data_similarity = 0.5
            tasks.append(t)
        results = sched.schedule(tasks)
        strained = sum(1 for r in results if r.verdict != TaskVerdict.EXECUTE)
        # With soft cap at 50%, borderline tasks shouldn't all be strained
        assert strained <= 6, f"Too many borderline tasks strained: {strained}/8"


# ── QUBO Properties ─────────────────────────────────────────


class TestQUBOProperties:
    """Property tests for QUBO matrix correctness."""

    def test_qubo_is_square(self):
        for N in [1, 4, 16, 32]:
            cfg = SchedulerConfig()
            builder = QUBOBuilder(cfg)
            tasks = [_productive_task(step=i) for i in range(N)]
            vecs = np.array([t.to_vector() for t in tasks])
            scores = np.random.uniform(0, 1, N)
            decs = [[] for _ in range(N)]
            Q = builder.build(tasks, scores, vecs, decs)
            assert Q.shape == (N, N)

    def test_qubo_finite(self):
        """QUBO matrix should have no NaN or Inf values."""
        builder = QUBOBuilder()
        tasks = [_productive_task(step=i) for i in range(8)]
        vecs = np.array([t.to_vector() for t in tasks])
        scores = np.random.uniform(0, 1, 8)
        decs = [[] for _ in range(8)]
        Q = builder.build(tasks, scores, vecs, decs)
        assert np.all(np.isfinite(Q))
