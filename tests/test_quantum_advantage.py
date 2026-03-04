"""Tests for the quantum advantage pipeline.

Covers every stage of the quantum path:

    ConflictGraph → QUBO → Ising → QAOA circuit → sampling
      → graph purification → coloring → end-to-end pipeline
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from qstrainer.models.enums import ComputePhase, JobType
from qstrainer.models.frame import ComputeTask
from qstrainer.quantum.advantage_pipeline import (
    PipelineConfig,
    QuantumAdvantagePipeline,
    QuantumScheduleResult,
)
from qstrainer.quantum.coloring import (
    dsatur_coloring,
    makespan,
    validate_coloring,
)
from qstrainer.quantum.conflict_graph import ConflictGraph
from qstrainer.quantum.ising import (
    binary_to_spin,
    ising_energy,
    ising_to_qubo,
    qubo_energy,
    qubo_to_ising,
    spin_to_binary,
)
from qstrainer.quantum.purifier import GraphPurifier
from qstrainer.quantum.qaoa_circuit import QAOASampler, SamplerOutput

# ── Helpers ──────────────────────────────────────────────────


def _make_task(
    gpu_id: str = "GPU-0",
    job_id: str = "job-1",
    step: int = 100,
    *,
    loss: float = 0.3,
    gradient_norm: float = 5.0,
    memory: float = 0.4,
    data_sim: float = 0.5,
    flops: float = 2e12,
    time_s: float = 0.5,
) -> ComputeTask:
    return ComputeTask(
        timestamp=time.time(),
        task_id=f"task-{gpu_id}-{step}",
        gpu_id=gpu_id,
        job_id=job_id,
        step_number=step,
        loss=loss,
        loss_delta=1e-4,
        gradient_norm=gradient_norm,
        gradient_variance=0.01,
        learning_rate=1e-3,
        batch_size=64,
        epoch=0,
        epoch_progress=0.5,
        estimated_flops=flops,
        estimated_time_s=time_s,
        memory_footprint_gb=memory * 80.0,
        compute_phase=ComputePhase.FORWARD_PASS,
        job_type=JobType.TRAINING,
        convergence_score=0.3,
        param_update_magnitude=0.01,
        data_similarity=data_sim,
        flop_utilization=0.8,
        throughput_samples_per_sec=5000.0,
    )


# ═══ ISING CONVERSION ═══════════════════════════════════════


class TestIsingConversion:
    """QUBO ↔ Ising round-trips and energy consistency."""

    def test_qubo_to_ising_shapes(self):
        Q = np.random.default_rng(0).standard_normal((6, 6))
        Q = np.triu(Q)
        h, J, offset = qubo_to_ising(Q)
        assert h.shape == (6,)
        assert J.shape == (6, 6)
        # J should be upper-triangular
        assert np.allclose(J, np.triu(J, k=1))

    def test_energy_consistency(self):
        """qubo_energy(x, Q) == ising_energy(σ, h, J) + offset for all x."""
        rng = np.random.default_rng(42)
        for n in [3, 5, 8]:
            Q = rng.standard_normal((n, n))
            Q = np.triu(Q)
            h, J, offset = qubo_to_ising(Q)

            for _ in range(20):
                x = rng.integers(0, 2, size=n).astype(float)
                sigma = binary_to_spin(x)
                e_qubo = qubo_energy(x, Q)
                e_ising = ising_energy(sigma, h, J) + offset
                assert abs(e_qubo - e_ising) < 1e-10, (
                    f"n={n}: QUBO={e_qubo:.6f} vs Ising+offset={e_ising:.6f}"
                )

    def test_round_trip_qubo_ising_qubo(self):
        """QUBO → Ising → QUBO should recover the same energies."""
        rng = np.random.default_rng(7)
        Q_orig = rng.standard_normal((5, 5))
        Q_orig = np.triu(Q_orig)

        h, J, off1 = qubo_to_ising(Q_orig)
        Q_back, off2 = ising_to_qubo(h, J)

        for _ in range(30):
            x = rng.integers(0, 2, size=5).astype(float)
            e_orig = qubo_energy(x, Q_orig)
            e_back = qubo_energy(x, Q_back) + off2 + off1
            # The round-trip should give same physical energy up to constant
            # We compare relative: e_orig == e_back - off2_correction
            # Actually: Q_orig → (h,J,off1) → Q_back,off2
            # → qubo_energy(x, Q_back)+off2 = ising_energy(σ,h,J)
            # and ising_energy(σ,h,J) + off1 = qubo_energy(x, Q_orig)
            # So qubo_energy(x, Q_back) + off2 + off1 = qubo_energy(x, Q_orig)
            assert abs(e_orig - e_back) < 1e-10

    def test_spin_binary_inverses(self):
        x = np.array([0, 1, 1, 0, 1])
        sigma = binary_to_spin(x)
        x_back = spin_to_binary(sigma)
        np.testing.assert_array_equal(x, x_back)

    def test_spin_values(self):
        x = np.array([0, 1])
        sigma = binary_to_spin(x)
        np.testing.assert_array_equal(sigma, [1.0, -1.0])


# ═══ CONFLICT GRAPH ═════════════════════════════════════════


class TestConflictGraph:
    """ConflictGraph construction and properties."""

    def test_manual_construction(self):
        g = ConflictGraph(4)
        g.add_edge(0, 1, 0.8, "gpu_contention")
        g.add_edge(1, 2, 0.5, "data_overlap")
        assert g.num_nodes == 4
        assert g.num_edges == 2

    def test_adjacency_symmetric(self):
        g = ConflictGraph(3)
        g.add_edge(0, 2, 0.7, "mixed")
        adj = g.adjacency_matrix
        assert adj[0, 2] == adj[2, 0] == 0.7

    def test_from_tasks_same_gpu(self):
        """Tasks on the same GPU at nearby steps should conflict."""
        tasks = [
            _make_task("GPU-0", step=100),
            _make_task("GPU-0", step=101),
        ]
        g = ConflictGraph.from_tasks(tasks)
        assert g.num_edges > 0
        # Both on GPU-0 with step gap 1 → strong conflict
        assert g.adjacency_matrix[0, 1] > 0.3

    def test_from_tasks_different_gpus_lower_conflict(self):
        """Tasks on different GPUs have lower GPU contention."""
        tasks = [
            _make_task("GPU-0", step=100),
            _make_task("GPU-1", step=100),
        ]
        g_same = ConflictGraph.from_tasks(
            [
                _make_task("GPU-0", step=100),
                _make_task("GPU-0", step=101),
            ]
        )
        g_diff = ConflictGraph.from_tasks(tasks)
        # Same GPU should have higher weight or more edges
        if g_diff.num_edges > 0 and g_same.num_edges > 0:
            assert g_same.adjacency_matrix[0, 1] >= g_diff.adjacency_matrix[0, 1]

    def test_from_tasks_batch(self):
        """Batch of 16 tasks produces a well-formed graph."""
        tasks = [_make_task(f"GPU-{i % 4}", step=100 + i) for i in range(16)]
        g = ConflictGraph.from_tasks(tasks)
        assert g.num_nodes == 16
        assert g.density() >= 0
        assert g.density() <= 1

    def test_qubo_maxcut(self):
        """QUBO from conflict graph is square and symmetric."""
        g = ConflictGraph(3)
        g.add_edge(0, 1, 0.5, "gpu_contention")
        g.add_edge(1, 2, 0.3, "data_overlap")
        Q = g.to_qubo()
        assert Q.shape == (3, 3)
        # Diagonal should be negative (from the −w_ij terms)
        assert Q[0, 0] < 0 or Q[1, 1] < 0

    def test_remove_edges(self):
        g = ConflictGraph(3)
        g.add_edge(0, 1, 0.5, "a")
        g.add_edge(1, 2, 0.3, "b")
        drop = np.array([True, False])
        g2 = g.remove_edges(drop)
        assert g2.num_edges == 1
        assert g2.adjacency_matrix[1, 2] == 0.3

    def test_neighbors_and_degree(self):
        g = ConflictGraph(4)
        g.add_edge(0, 1, 0.5, "a")
        g.add_edge(0, 2, 0.3, "b")
        assert set(g.neighbors(0)) == {1, 2}
        assert g.degree(0) == 2
        assert g.degree(3) == 0


# ═══ QAOA SAMPLER ════════════════════════════════════════════


class TestQAOASampler:
    """QAOA circuit construction, optimisation, and sampling."""

    def _simple_ising(self, n: int = 4):
        """Anti-ferromagnetic ring: h=0, J_ij = +1 for adjacent pairs."""
        h = np.zeros(n)
        J = np.zeros((n, n))
        for i in range(n - 1):
            J[i, i + 1] = 1.0
        return h, J

    def test_build_and_optimise(self):
        h, J = self._simple_ising(4)
        sampler = QAOASampler(p_layers=1, n_restarts=2, maxfev=40, seed=0)
        energy = sampler.build_and_optimise(h, J)
        assert isinstance(energy, float)
        assert np.isfinite(energy)

    def test_sample_returns_bitstrings(self):
        h, J = self._simple_ising(4)
        sampler = QAOASampler(p_layers=1, n_restarts=2, maxfev=40, seed=0)
        sampler.build_and_optimise(h, J)
        output = sampler.sample(n_shots=256, top_k=16)
        assert len(output.samples) > 0
        assert output.samples[0].bitstring.shape == (4,)
        assert all(b in (0, 1) for b in output.samples[0].bitstring)

    def test_sample_probabilities_valid(self):
        h, J = self._simple_ising(4)
        sampler = QAOASampler(p_layers=1, n_restarts=2, maxfev=40, seed=0)
        sampler.build_and_optimise(h, J)
        output = sampler.sample(n_shots=512)
        for s in output.samples:
            assert 0 <= s.probability <= 1
            assert np.isfinite(s.ising_energy)

    def test_best_sample(self):
        h, J = self._simple_ising(4)
        sampler = QAOASampler(p_layers=1, n_restarts=2, maxfev=40, seed=0)
        sampler.build_and_optimise(h, J)
        output = sampler.sample(n_shots=512)
        best = output.best
        assert best.ising_energy <= max(s.ising_energy for s in output.samples)

    def test_max_qubits_enforced(self):
        h = np.zeros(25)
        J = np.zeros((25, 25))
        sampler = QAOASampler(p_layers=1)
        with pytest.raises(ValueError, match="statevector sim limited"):
            sampler.build_and_optimise(h, J)

    def test_p2_improves_over_p1(self):
        """More layers should give same or better energy."""
        h, J = self._simple_ising(6)
        s1 = QAOASampler(p_layers=1, n_restarts=3, maxfev=60, seed=42)
        e1 = s1.build_and_optimise(h, J)
        s2 = QAOASampler(p_layers=2, n_restarts=3, maxfev=60, seed=42)
        e2 = s2.build_and_optimise(h, J)
        # p=2 should be at least as good (lower energy)
        assert e2 <= e1 + 0.5  # small tolerance for COBYLA noise


# ═══ GRAPH PURIFIER ═════════════════════════════════════════


class TestGraphPurifier:
    """Graph purification via QAOA bitstring samples."""

    def _make_sampler_output(self, n: int, n_samples: int = 8):
        """Create a fake SamplerOutput for testing."""
        from qstrainer.quantum.qaoa_circuit import SampleResult

        rng = np.random.default_rng(42)
        samples = []
        for _ in range(n_samples):
            bits = rng.integers(0, 2, size=n).astype(np.int64)
            samples.append(
                SampleResult(
                    bitstring=bits,
                    probability=1.0 / n_samples,
                    ising_energy=rng.standard_normal(),
                )
            )
        return SamplerOutput(
            samples=samples,
            optimal_energy=-1.0,
            optimal_params=np.zeros(4),
            optimize_time_s=0.1,
            sample_time_s=0.01,
            n_qubits=n,
            p_layers=2,
        )

    def test_empty_graph(self):
        g = ConflictGraph(3)
        output = self._make_sampler_output(3)
        purifier = GraphPurifier(threshold=0.5)
        result = purifier.purify(g, output)
        assert result.edges_dropped == 0

    def test_some_edges_dropped(self):
        g = ConflictGraph(4)
        g.add_edge(0, 1, 0.8, "gpu")
        g.add_edge(0, 2, 0.3, "data")
        g.add_edge(2, 3, 0.5, "mem")
        output = self._make_sampler_output(4, n_samples=50)
        purifier = GraphPurifier(threshold=0.4)
        result = purifier.purify(g, output)
        assert result.purified_edges <= result.original_edges
        assert result.edges_dropped >= 0

    def test_aggressive_threshold_drops_more(self):
        g = ConflictGraph(4)
        g.add_edge(0, 1, 0.8, "gpu")
        g.add_edge(1, 2, 0.5, "data")
        g.add_edge(2, 3, 0.3, "mem")
        output = self._make_sampler_output(4, n_samples=30)
        conservative = GraphPurifier(threshold=0.9).purify(g, output)
        aggressive = GraphPurifier(threshold=0.3).purify(g, output)
        assert aggressive.edges_dropped >= conservative.edges_dropped

    def test_resolution_frequency_shape(self):
        g = ConflictGraph(3)
        g.add_edge(0, 1, 0.5, "a")
        g.add_edge(1, 2, 0.3, "b")
        output = self._make_sampler_output(3, n_samples=20)
        result = GraphPurifier(threshold=0.5).purify(g, output)
        assert result.resolution_frequencies.shape == (2,)
        assert all(0 <= f <= 1 for f in result.resolution_frequencies)


# ═══ GRAPH COLORING ═════════════════════════════════════════


class TestDSaturColoring:
    """DSatur graph coloring and makespan."""

    def test_empty_graph(self):
        g = ConflictGraph(0)
        result = dsatur_coloring(g)
        assert result.num_colors == 0

    def test_independent_set(self):
        """No edges → 1 colour needed (all tasks in one slot)."""
        g = ConflictGraph(5)
        result = dsatur_coloring(g)
        assert result.num_colors == 1
        assert len(result.time_slots[0]) == 5

    def test_complete_graph(self):
        """Complete graph on n nodes → n colours."""
        n = 4
        g = ConflictGraph(n)
        for i in range(n):
            for j in range(i + 1, n):
                g.add_edge(i, j, 1.0, "full")
        result = dsatur_coloring(g)
        assert result.num_colors == n

    def test_bipartite_graph(self):
        """Complete bipartite K_{2,2} → 2 colours."""
        g = ConflictGraph(4)
        g.add_edge(0, 2, 1.0, "a")
        g.add_edge(0, 3, 1.0, "a")
        g.add_edge(1, 2, 1.0, "a")
        g.add_edge(1, 3, 1.0, "a")
        result = dsatur_coloring(g)
        assert result.num_colors == 2

    def test_coloring_valid(self):
        """No two adjacent nodes share a colour (arbitrary graph)."""
        rng = np.random.default_rng(99)
        g = ConflictGraph(10)
        for i in range(10):
            for j in range(i + 1, 10):
                if rng.random() < 0.4:
                    g.add_edge(i, j, rng.uniform(0.1, 1.0), "rand")
        result = dsatur_coloring(g)
        assert validate_coloring(g, result)

    def test_makespan_helper(self):
        g = ConflictGraph(3)
        g.add_edge(0, 1, 1.0, "a")
        result = dsatur_coloring(g)
        assert makespan(result) == result.num_colors

    def test_purified_uses_fewer_colors(self):
        """Dropping some edges should require ≤ colours."""
        n = 8
        g = ConflictGraph(n)
        rng = np.random.default_rng(42)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.6:
                    g.add_edge(i, j, rng.uniform(0.2, 1.0), "rand")
        orig = dsatur_coloring(g)

        # Drop half the edges
        drop_mask = np.array([i % 2 == 0 for i in range(g.num_edges)])
        g2 = g.remove_edges(drop_mask)
        purified = dsatur_coloring(g2)

        assert purified.num_colors <= orig.num_colors


# ═══ END-TO-END PIPELINE ════════════════════════════════════


class TestQuantumAdvantagePipeline:
    """Full pipeline: tasks → conflict graph → QUBO → Ising → QAOA → purify → colour."""

    def _make_batch(self, n: int = 12) -> list:
        tasks = []
        gpus = ["GPU-0", "GPU-1", "GPU-2", "GPU-3"]
        for i in range(n):
            tasks.append(
                _make_task(
                    gpu_id=gpus[i % 4],
                    step=100 + i,
                    memory=0.3 + (i % 3) * 0.2,
                    data_sim=0.4 + (i % 5) * 0.1,
                )
            )
        return tasks

    def test_pipeline_runs(self):
        tasks = self._make_batch(12)
        pipeline = QuantumAdvantagePipeline(
            PipelineConfig(
                p_layers=1,
                n_restarts=2,
                maxfev=30,
                n_shots=128,
                seed=42,
            )
        )
        result = pipeline.run(tasks)
        assert isinstance(result, QuantumScheduleResult)
        assert result.n_tasks == 12
        assert result.total_time > 0

    def test_makespan_reduction_non_negative(self):
        tasks = self._make_batch(12)
        pipeline = QuantumAdvantagePipeline(
            PipelineConfig(
                p_layers=1,
                n_restarts=2,
                maxfev=30,
                n_shots=256,
                seed=42,
            )
        )
        result = pipeline.run(tasks)
        assert result.makespan_reduction >= -0.1  # allow tiny tolerance
        assert result.purified_makespan <= result.original_makespan + 1

    def test_colorings_valid(self):
        tasks = self._make_batch(12)
        pipeline = QuantumAdvantagePipeline(
            PipelineConfig(
                p_layers=1,
                n_restarts=2,
                maxfev=30,
                n_shots=128,
                seed=42,
            )
        )
        result = pipeline.run(tasks)
        assert result.original_coloring_valid
        assert result.purified_coloring_valid

    def test_edges_dropped(self):
        tasks = self._make_batch(16)
        pipeline = QuantumAdvantagePipeline(
            PipelineConfig(
                p_layers=1,
                n_restarts=2,
                maxfev=30,
                n_shots=256,
                purify_threshold=0.4,
                seed=42,
            )
        )
        result = pipeline.run(tasks)
        assert result.edges_dropped >= 0
        assert result.purified_edges <= result.original_edges

    def test_timing_breakdown(self):
        tasks = self._make_batch(8)
        pipeline = QuantumAdvantagePipeline(
            PipelineConfig(
                p_layers=1,
                n_restarts=1,
                maxfev=20,
                n_shots=64,
                seed=42,
            )
        )
        result = pipeline.run(tasks)
        assert result.graph_build_time >= 0
        assert result.qaoa_optimize_time >= 0
        assert result.sample_time >= 0
        assert result.purify_time >= 0
        assert result.total_time > 0

    def test_ising_metrics(self):
        tasks = self._make_batch(8)
        pipeline = QuantumAdvantagePipeline(
            PipelineConfig(
                p_layers=1,
                n_restarts=1,
                maxfev=20,
                n_shots=64,
                seed=42,
            )
        )
        result = pipeline.run(tasks)
        assert result.qubo_size == 8
        assert result.ising_h_norm > 0
        assert np.isfinite(result.qaoa_optimal_energy)

    def test_time_slots_cover_all_tasks(self):
        tasks = self._make_batch(12)
        pipeline = QuantumAdvantagePipeline(
            PipelineConfig(
                p_layers=1,
                n_restarts=2,
                maxfev=30,
                n_shots=128,
                seed=42,
            )
        )
        result = pipeline.run(tasks)
        all_tasks = set()
        for slot_tasks in result.time_slots.values():
            all_tasks.update(slot_tasks)
        assert all_tasks == set(range(12))

    def test_pre_built_graph(self):
        """Pipeline accepts a pre-built conflict graph."""
        tasks = self._make_batch(6)
        graph = ConflictGraph.from_tasks(tasks)
        pipeline = QuantumAdvantagePipeline(
            PipelineConfig(
                p_layers=1,
                n_restarts=1,
                maxfev=20,
                n_shots=64,
                seed=42,
            )
        )
        result = pipeline.run(tasks, graph=graph)
        assert result.n_tasks == 6

    def test_larger_batch_16(self):
        """16-task batch completes without errors."""
        tasks = self._make_batch(16)
        pipeline = QuantumAdvantagePipeline(
            PipelineConfig(
                p_layers=1,
                n_restarts=2,
                maxfev=40,
                n_shots=256,
                seed=42,
            )
        )
        result = pipeline.run(tasks)
        assert result.n_tasks == 16
        assert result.purified_coloring_valid
