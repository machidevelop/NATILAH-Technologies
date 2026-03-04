"""ConflictGraph — task conflict graph for quantum-advantage scheduling.

Nodes are GPU compute tasks.  Edges represent resource conflicts:
    • GPU contention    — same GPU at nearby steps
    • Data overlap      — similar feature vectors (near-duplicate batches)
    • Memory pressure   — combined footprints exceed budget

The graph is the input to the quantum advantage pipeline:

    ConflictGraph → QUBO (Max-Cut) → Ising → QAOA → samples
      → graph purification → fewer edges → fewer colors → smaller makespan
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from qstrainer.models.frame import ComputeTask, N_BASE_FEATURES


# ── Edge ─────────────────────────────────────────────────────

@dataclass(slots=True)
class Edge:
    """Weighted edge between two conflicting tasks."""

    i: int
    j: int
    weight: float
    conflict_type: str  # "gpu_contention" | "data_overlap" | "memory_pressure"


# ── ConflictGraph ────────────────────────────────────────────

class ConflictGraph:
    """Undirected weighted conflict graph over GPU compute tasks.

    Nodes : task indices ``0 .. N-1``
    Edges : ``Edge(i, j, weight, conflict_type)``
    Weight: w ∈ [0, 1] — severity of the conflict
        1.0 = absolute conflict (same GPU, same instant, same memory)
        0.0 = no conflict
    """

    def __init__(self, n_nodes: int) -> None:
        self.n = n_nodes
        self._adj = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        self._edges: List[Edge] = []
        self._node_labels: List[str] = [""] * n_nodes

    # ── Factory ──────────────────────────────────────────────

    @classmethod
    def from_tasks(
        cls,
        tasks: List[ComputeTask],
        *,
        gpu_weight: float = 0.6,
        data_weight: float = 0.25,
        memory_weight: float = 0.15,
        conflict_threshold: float = 0.10,
    ) -> "ConflictGraph":
        """Build a conflict graph from a batch of compute tasks.

        Parameters
        ----------
        tasks : list[ComputeTask]
        gpu_weight, data_weight, memory_weight
            Relative importance of each conflict source (sum to 1).
        conflict_threshold
            Minimum combined weight to create an edge (ignore weak noise).
        """
        N = len(tasks)
        graph = cls(N)

        # Feature vectors for data-similarity
        vectors = np.empty((N, N_BASE_FEATURES), dtype=np.float64)
        for i, t in enumerate(tasks):
            vectors[i] = t.to_vector()
            graph._node_labels[i] = t.task_id

        # Pairwise cosine similarity (vectorised)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        data_sim = (vectors / norms) @ (vectors / norms).T  # (N, N)

        for i in range(N):
            for j in range(i + 1, N):
                # 1. GPU contention — same GPU + close steps
                gpu_c = 0.0
                if tasks[i].gpu_id == tasks[j].gpu_id:
                    gap = abs(tasks[i].step_number - tasks[j].step_number)
                    gpu_c = max(0.0, 1.0 - gap / 50.0)

                # 2. Data overlap — cosine similarity of feature vectors
                data_c = max(0.0, float(data_sim[i, j]))

                # 3. Memory pressure — combined footprints > budget
                mem_i = tasks[i].memory_footprint_gb / 80.0
                mem_j = tasks[j].memory_footprint_gb / 80.0
                mem_c = max(0.0, min(1.0, mem_i + mem_j - 1.0))

                w = gpu_weight * gpu_c + data_weight * data_c + memory_weight * mem_c

                if w >= conflict_threshold:
                    ctype = _dominant(gpu_c, data_c, mem_c)
                    graph.add_edge(i, j, w, ctype)

        return graph

    # ── Mutation ─────────────────────────────────────────────

    def add_edge(
        self, i: int, j: int, weight: float, conflict_type: str = "mixed"
    ) -> None:
        """Add or update an edge."""
        self._adj[i, j] = weight
        self._adj[j, i] = weight
        self._edges.append(Edge(i=i, j=j, weight=weight, conflict_type=conflict_type))

    # ── Properties ───────────────────────────────────────────

    @property
    def num_nodes(self) -> int:
        return self.n

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    @property
    def edges(self) -> List[Edge]:
        return list(self._edges)

    @property
    def adjacency_matrix(self) -> np.ndarray:
        return self._adj.copy()

    @property
    def node_labels(self) -> List[str]:
        return list(self._node_labels)

    def edge_list(self) -> List[Tuple[int, int, float]]:
        """List of ``(i, j, weight)`` tuples."""
        return [(e.i, e.j, e.weight) for e in self._edges]

    def neighbors(self, node: int) -> List[int]:
        """Indices of neighbors (non‐zero adjacency)."""
        return [j for j in range(self.n) if self._adj[node, j] > 0]

    def degree(self, node: int) -> int:
        return sum(1 for j in range(self.n) if self._adj[node, j] > 0)

    def density(self) -> float:
        """Edge density: |E| / (n choose 2)."""
        max_e = self.n * (self.n - 1) / 2
        return self.num_edges / max_e if max_e > 0 else 0.0

    # ── QUBO (Max‑Cut) ──────────────────────────────────────

    def to_qubo(self) -> np.ndarray:
        """Convert conflict graph to a **Max-Cut QUBO**.

        Binary variable ``x_i ∈ {0,1}`` per node (task):
            x_i = 1 → partition A
            x_i = 0 → partition B

        Minimise intra-partition conflict weight:

            E(x) = Σ_{(i,j)∈E} w_{ij} · (2 x_i x_j − x_i − x_j)

        A good solution separates conflicting tasks into different
        partitions — which the QAOA sampler exploits for graph
        purification.
        """
        N = self.n
        Q = np.zeros((N, N), dtype=np.float64)

        for e in self._edges:
            w = e.weight
            Q[e.i, e.j] += 2.0 * w   # quadratic
            Q[e.i, e.i] -= w          # linear
            Q[e.j, e.j] -= w          # linear

        return Q

    # ── Graph surgery ────────────────────────────────────────

    def remove_edges(self, drop_mask: np.ndarray) -> "ConflictGraph":
        """Return a **new** graph with edges removed where *drop_mask* is True.

        Parameters
        ----------
        drop_mask : ndarray of bool, length ``num_edges``
            ``True`` → drop the edge, ``False`` → keep.
        """
        purified = ConflictGraph(self.n)
        purified._node_labels = list(self._node_labels)

        for idx, e in enumerate(self._edges):
            if not drop_mask[idx]:
                purified.add_edge(e.i, e.j, e.weight, e.conflict_type)

        return purified

    def subgraph(self, node_mask: np.ndarray) -> "ConflictGraph":
        """Induced subgraph on nodes where *node_mask* is True."""
        indices = np.where(node_mask)[0]
        sub = ConflictGraph(len(indices))
        idx_map = {int(old): new for new, old in enumerate(indices)}
        sub._node_labels = [self._node_labels[i] for i in indices]

        for e in self._edges:
            if e.i in idx_map and e.j in idx_map:
                sub.add_edge(idx_map[e.i], idx_map[e.j], e.weight, e.conflict_type)

        return sub


# ── Helpers ──────────────────────────────────────────────────

def _dominant(gpu: float, data: float, memory: float) -> str:
    vals = {"gpu_contention": gpu, "data_overlap": data, "memory_pressure": memory}
    return max(vals, key=lambda k: vals[k])
