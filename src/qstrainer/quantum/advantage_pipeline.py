"""QuantumAdvantagePipeline — end-to-end quantum-advantage GPU scheduling.

The full quantum path that turns a batch of GPU tasks into a
minimum-makespan parallel schedule:

    tasks
      → **ConflictGraph**            build task-conflict edges
      → **QUBO** (Max-Cut)           encode conflicts as binary optimisation
      → **Ising** (h, J)             standard transform for quantum hardware
      → **QAOA circuit** (p layers)  parameterised quantum circuit
      → **sampling** (n_shots)       measure bitstring population
      → **graph purification**       drop "easy" edges (high resolution freq)
      → **DSatur coloring**          colour the sparser graph
      → **schedule**                 colours → time slots, fewer slots = faster

Why quantum?
    Classical greedy coloring operates on the full conflict graph.
    QAOA samples reveal the graph's *separability structure* — which
    conflicts are easy to satisfy vs. hard.  Dropping easy edges yields a
    sparser graph that needs fewer colours, reducing makespan without
    sacrificing correctness of the hard constraints.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from qstrainer.models.frame import ComputeTask
from qstrainer.quantum.coloring import (
    ColoringResult,
    dsatur_coloring,
    makespan,
    validate_coloring,
)
from qstrainer.quantum.conflict_graph import ConflictGraph
from qstrainer.quantum.ising import qubo_to_ising
from qstrainer.quantum.purifier import GraphPurifier, PurificationResult
from qstrainer.quantum.qaoa_circuit import QAOASampler, SamplerOutput

# ── Pipeline config ──────────────────────────────────────────


@dataclass
class PipelineConfig:
    """Tuning knobs for the quantum advantage pipeline."""

    # Conflict graph
    gpu_weight: float = 0.6
    data_weight: float = 0.25
    memory_weight: float = 0.15
    conflict_threshold: float = 0.10

    # QAOA
    p_layers: int = 2
    n_restarts: int = 5
    maxfev: int = 120
    n_shots: int = 1024
    top_k_samples: int = 64
    seed: int = 42

    # Purification
    purify_threshold: float = 0.55
    use_percentile: bool = False
    weight_by_probability: bool = True


# ── Pipeline result ──────────────────────────────────────────


@dataclass
class QuantumScheduleResult:
    """Complete output from the quantum advantage pipeline."""

    # Schedule
    original_makespan: int  # colours needed BEFORE purification
    purified_makespan: int  # colours needed AFTER purification
    makespan_reduction: float  # (orig − purified) / orig
    time_slots: dict[int, list[int]]  # slot → task indices (purified)

    # Graph stats
    n_tasks: int
    original_edges: int
    purified_edges: int
    edges_dropped: int
    edge_drop_ratio: float
    graph_density_before: float
    graph_density_after: float

    # Parallelism
    max_parallelism: int
    avg_parallelism: float

    # Quantum metrics
    qubo_size: int
    ising_h_norm: float
    ising_j_nnz: int  # number of non-zero couplings
    qaoa_optimal_energy: float
    n_samples_used: int
    purify_threshold: float
    p_layers: int

    # Coloring validity
    original_coloring_valid: bool
    purified_coloring_valid: bool

    # Timing breakdown (seconds)
    graph_build_time: float
    qubo_build_time: float
    ising_convert_time: float
    qaoa_optimize_time: float
    sample_time: float
    purify_time: float
    color_original_time: float
    color_purified_time: float
    total_time: float

    # Raw intermediates (for dashboard / debugging)
    sampler_output: SamplerOutput | None = field(default=None, repr=False)
    purification_result: PurificationResult | None = field(default=None, repr=False)
    original_coloring: ColoringResult | None = field(default=None, repr=False)
    purified_coloring: ColoringResult | None = field(default=None, repr=False)


# ── QuantumAdvantagePipeline ─────────────────────────────────


class QuantumAdvantagePipeline:
    """End-to-end quantum advantage pipeline for GPU task scheduling.

    Usage::

        pipeline = QuantumAdvantagePipeline()
        result = pipeline.run(tasks)
        print(f"Makespan: {result.original_makespan} → {result.purified_makespan}")
        print(f"Edges dropped: {result.edges_dropped} ({result.edge_drop_ratio:.0%})")
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.cfg = config or PipelineConfig()

    def run(
        self,
        tasks: list[ComputeTask],
        graph: ConflictGraph | None = None,
    ) -> QuantumScheduleResult:
        """Execute the full quantum advantage pipeline.

        Parameters
        ----------
        tasks : list[ComputeTask]
            Batch of GPU tasks to schedule.
        graph : ConflictGraph, optional
            Pre-built conflict graph.  If *None*, built from *tasks*.

        Returns
        -------
        QuantumScheduleResult
            Schedule, metrics, and timing for every pipeline stage.
        """
        total_t0 = time.perf_counter()

        # ── 1. Build conflict graph ─────────────────────────
        t0 = time.perf_counter()
        if graph is None:
            graph = ConflictGraph.from_tasks(
                tasks,
                gpu_weight=self.cfg.gpu_weight,
                data_weight=self.cfg.data_weight,
                memory_weight=self.cfg.memory_weight,
                conflict_threshold=self.cfg.conflict_threshold,
            )
        graph_time = time.perf_counter() - t0

        n = graph.num_nodes

        # ── 2. Conflict graph → QUBO (Max-Cut) ─────────────
        t0 = time.perf_counter()
        Q = graph.to_qubo()
        qubo_time = time.perf_counter() - t0

        # ── 3. QUBO → Ising (h, J, offset) ─────────────────
        t0 = time.perf_counter()
        h, J, offset = qubo_to_ising(Q)
        ising_time = time.perf_counter() - t0

        # ── 4. Build QAOA circuit + optimise ────────────────
        sampler = QAOASampler(
            p_layers=self.cfg.p_layers,
            n_restarts=self.cfg.n_restarts,
            maxfev=self.cfg.maxfev,
            seed=self.cfg.seed,
        )

        t0 = time.perf_counter()
        sampler.build_and_optimise(h, J)
        qaoa_time = time.perf_counter() - t0

        # ── 5. Sample bitstrings ────────────────────────────
        t0 = time.perf_counter()
        sampler_output = sampler.sample(
            n_shots=self.cfg.n_shots,
            top_k=self.cfg.top_k_samples,
        )
        sample_time = time.perf_counter() - t0

        # ── 6. Purify graph ─────────────────────────────────
        purifier = GraphPurifier(
            threshold=self.cfg.purify_threshold,
            use_percentile=self.cfg.use_percentile,
            weight_by_probability=self.cfg.weight_by_probability,
        )

        t0 = time.perf_counter()
        purification = purifier.purify(graph, sampler_output)
        purify_time = time.perf_counter() - t0

        purified = purification.purified_graph

        # ── 7a. Colour ORIGINAL graph (baseline) ────────────
        t0 = time.perf_counter()
        orig_coloring = dsatur_coloring(graph)
        color_orig_time = time.perf_counter() - t0

        # ── 7b. Colour PURIFIED graph (quantum advantage) ──
        t0 = time.perf_counter()
        puri_coloring = dsatur_coloring(purified)
        color_puri_time = time.perf_counter() - t0

        total_time = time.perf_counter() - total_t0

        # ── Metrics ─────────────────────────────────────────
        orig_mk = makespan(orig_coloring)
        puri_mk = makespan(puri_coloring)
        reduction = (orig_mk - puri_mk) / max(orig_mk, 1)

        return QuantumScheduleResult(
            # Schedule
            original_makespan=orig_mk,
            purified_makespan=puri_mk,
            makespan_reduction=reduction,
            time_slots=puri_coloring.time_slots,
            # Graph
            n_tasks=n,
            original_edges=graph.num_edges,
            purified_edges=purified.num_edges,
            edges_dropped=purification.edges_dropped,
            edge_drop_ratio=purification.drop_ratio,
            graph_density_before=graph.density(),
            graph_density_after=purified.density(),
            # Parallelism
            max_parallelism=puri_coloring.max_parallelism,
            avg_parallelism=puri_coloring.avg_parallelism,
            # Quantum
            qubo_size=n,
            ising_h_norm=float(np.linalg.norm(h)),
            ising_j_nnz=int(np.count_nonzero(J)),
            qaoa_optimal_energy=sampler_output.optimal_energy,
            n_samples_used=len(sampler_output.samples),
            purify_threshold=purification.threshold_used,
            p_layers=self.cfg.p_layers,
            # Validity
            original_coloring_valid=validate_coloring(graph, orig_coloring),
            purified_coloring_valid=validate_coloring(purified, puri_coloring),
            # Timing
            graph_build_time=graph_time,
            qubo_build_time=qubo_time,
            ising_convert_time=ising_time,
            qaoa_optimize_time=qaoa_time,
            sample_time=sample_time,
            purify_time=purify_time,
            color_original_time=color_orig_time,
            color_purified_time=color_puri_time,
            total_time=total_time,
            # Raw intermediates
            sampler_output=sampler_output,
            purification_result=purification,
            original_coloring=orig_coloring,
            purified_coloring=puri_coloring,
        )
