"""Quantum components — QUBO feature selector, quantum kernel, advantage pipeline.

The quantum advantage pipeline:

    tasks → ConflictGraph → QUBO → Ising (h,J) → QAOA circuit → sampling
      → graph purification → fewer edges → fewer colours → smaller makespan
"""

from qstrainer.quantum.advantage_pipeline import (
    PipelineConfig,
    QuantumAdvantagePipeline,
    QuantumScheduleResult,
)
from qstrainer.quantum.coloring import (
    ColoringResult,
    dsatur_coloring,
    makespan,
    validate_coloring,
)
from qstrainer.quantum.conflict_graph import ConflictGraph, Edge
from qstrainer.quantum.feature_selector import QUBOFeatureSelector
from qstrainer.quantum.ising import (
    binary_to_spin,
    ising_energy,
    ising_to_qubo,
    qubo_energy,
    qubo_to_ising,
    spin_to_binary,
)
from qstrainer.quantum.kernel_detector import QuantumKernelDetector
from qstrainer.quantum.kernel_provider import QuantumKernelProvider
from qstrainer.quantum.purifier import GraphPurifier, PurificationResult
from qstrainer.quantum.qaoa_circuit import QAOASampler, SampleResult, SamplerOutput

__all__ = [
    # Existing
    "QUBOFeatureSelector",
    "QuantumKernelProvider",
    "QuantumKernelDetector",
    # Conflict graph
    "ConflictGraph",
    "Edge",
    # Ising ↔ QUBO
    "qubo_to_ising",
    "ising_to_qubo",
    "qubo_energy",
    "ising_energy",
    "binary_to_spin",
    "spin_to_binary",
    # QAOA sampler
    "QAOASampler",
    "SamplerOutput",
    "SampleResult",
    # Purifier
    "GraphPurifier",
    "PurificationResult",
    # Coloring
    "dsatur_coloring",
    "makespan",
    "validate_coloring",
    "ColoringResult",
    # End-to-end pipeline
    "QuantumAdvantagePipeline",
    "PipelineConfig",
    "QuantumScheduleResult",
]
