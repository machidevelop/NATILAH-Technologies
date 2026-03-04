"""GraphPurifier — prune conflict-graph edges using QAOA bitstring samples.

The quantum path:
    Conflict graph → QUBO → Ising → QAOA → **sampling** → **purification**
      → fewer edges → fewer colors → smaller makespan

How it works
------------
1. Each QAOA sample is a bitstring ``x ∈ {0,1}^n`` — a proposed partition
   of tasks into two groups.
2. For every edge ``(i, j)``, we compute the **resolution frequency**:
   the fraction of samples in which ``x_i ≠ x_j`` (the edge is "cut").
3. Edges with resolution frequency **above** the threshold are considered
   "easy" conflicts — the quantum solver consistently separates them.
   These edges are **dropped**.
4. Edges with low resolution frequency are "hard" conflicts that must be
   respected.  They are **kept**.

Result: a **purified** (sparser) conflict graph that needs fewer colors
to schedule — reducing makespan and GPU idle time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from qstrainer.quantum.conflict_graph import ConflictGraph
from qstrainer.quantum.qaoa_circuit import SamplerOutput


# ── Result container ─────────────────────────────────────────

@dataclass
class PurificationResult:
    """Statistics from one graph purification run."""

    original_edges: int
    purified_edges: int
    edges_dropped: int
    drop_ratio: float
    resolution_frequencies: np.ndarray   # per-edge resolution freq
    threshold_used: float
    purified_graph: ConflictGraph


# ── GraphPurifier ────────────────────────────────────────────

class GraphPurifier:
    """Purify a conflict graph using QAOA bitstring samples.

    Parameters
    ----------
    threshold : float, default 0.55
        Edges with resolution frequency ≥ threshold are dropped.
        Higher threshold = more conservative (fewer edges dropped).
        Lower threshold = more aggressive (more edges dropped).
    use_percentile : bool, default False
        If True, interpret *threshold* as a percentile of the resolution
        frequency distribution (e.g. 75 → keep the bottom 75 % of edges).
    weight_by_probability : bool, default True
        If True, weight each sample by its QAOA probability when computing
        resolution frequency.  This gives higher-quality solutions more
        influence on the purification decision.
    """

    def __init__(
        self,
        threshold: float = 0.55,
        use_percentile: bool = False,
        weight_by_probability: bool = True,
    ) -> None:
        self.threshold = threshold
        self.use_percentile = use_percentile
        self.weight_by_probability = weight_by_probability

    def purify(
        self,
        graph: ConflictGraph,
        sampler_output: SamplerOutput,
    ) -> PurificationResult:
        """Purify a conflict graph using QAOA samples.

        Parameters
        ----------
        graph : ConflictGraph
            The original (dense) conflict graph.
        sampler_output : SamplerOutput
            QAOA bitstring samples from :meth:`QAOASampler.sample`.

        Returns
        -------
        PurificationResult
            The purified graph and statistics.
        """
        edges = graph.edges
        n_edges = len(edges)
        samples = sampler_output.samples

        if n_edges == 0 or not samples:
            return PurificationResult(
                original_edges=n_edges,
                purified_edges=n_edges,
                edges_dropped=0,
                drop_ratio=0.0,
                resolution_frequencies=np.array([]),
                threshold_used=self.threshold,
                purified_graph=graph,
            )

        # ── Compute resolution frequency for each edge ──────
        resolution_freq = np.zeros(n_edges, dtype=np.float64)
        total_weight = 0.0

        for sample in samples:
            x = sample.bitstring
            w = sample.probability if self.weight_by_probability else 1.0
            total_weight += w

            for idx, edge in enumerate(edges):
                # Edge is "resolved" if endpoints are in different partitions
                if x[edge.i] != x[edge.j]:
                    resolution_freq[idx] += w

        if total_weight > 0:
            resolution_freq /= total_weight

        # ── Determine which edges to drop ────────────────────
        if self.use_percentile:
            # threshold is a percentile (0–100)
            cutoff = float(np.percentile(resolution_freq, self.threshold))
        else:
            cutoff = self.threshold

        drop_mask = resolution_freq >= cutoff

        # ── Build purified graph ─────────────────────────────
        purified = graph.remove_edges(drop_mask)
        n_dropped = int(np.sum(drop_mask))

        return PurificationResult(
            original_edges=n_edges,
            purified_edges=n_edges - n_dropped,
            edges_dropped=n_dropped,
            drop_ratio=n_dropped / max(n_edges, 1),
            resolution_frequencies=resolution_freq,
            threshold_used=cutoff,
            purified_graph=purified,
        )
