"""GraphColoring — colour a conflict graph to produce a time-slot schedule.

Each colour = one time slot.  Tasks sharing a colour run **in parallel**
(no conflict between them).  Fewer colours → fewer time slots → smaller
makespan → less GPU idle time → lower cost.

Algorithm: **DSatur** (Degree-of-Saturation greedy).

    1. Pick the uncolored vertex with the highest *saturation degree*
       (number of distinct colours among its colored neighbours),
       breaking ties by highest vertex degree.
    2. Assign it the smallest colour not used by its neighbours.
    3. Repeat until all vertices are coloured.

DSatur is a well-known near-optimal heuristic (Brélaz, 1979) that
usually achieves chromatic number or close to it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qstrainer.quantum.conflict_graph import ConflictGraph

# ── Result container ─────────────────────────────────────────


@dataclass
class ColoringResult:
    """Result from graph coloring → schedule generation."""

    # Coloring
    num_colors: int  # = makespan (time slots)
    color_of: dict[int, int]  # node → colour index
    time_slots: dict[int, list[int]]  # colour → list of node indices

    # Derived
    max_parallelism: int  # largest time slot
    avg_parallelism: float  # mean tasks per slot


# ── DSatur coloring ──────────────────────────────────────────


def dsatur_coloring(graph: ConflictGraph) -> ColoringResult:
    """Colour a conflict graph using the DSatur heuristic.

    Returns
    -------
    ColoringResult
        Colour assignment, time-slot schedule, and parallelism stats.
    """
    n = graph.num_nodes
    if n == 0:
        return ColoringResult(
            num_colors=0,
            color_of={},
            time_slots={},
            max_parallelism=0,
            avg_parallelism=0.0,
        )

    adj = graph.adjacency_matrix  # (n, n), >0 means edge

    color_of: dict[int, int] = {}
    # saturation[v] = set of distinct colours among coloured neighbours
    saturation: list[set[int]] = [set() for _ in range(n)]
    uncolored: set[int] = set(range(n))

    # Pre-compute degrees for tie-breaking
    degrees = np.array([graph.degree(v) for v in range(n)])

    for _ in range(n):
        # Pick the uncolored vertex with highest saturation, then highest degree
        best_v = -1
        best_sat = -1
        best_deg = -1
        for v in uncolored:
            s = len(saturation[v])
            d = int(degrees[v])
            if s > best_sat or (s == best_sat and d > best_deg):
                best_v = v
                best_sat = s
                best_deg = d

        # Assign smallest available colour
        neighbour_colors = saturation[best_v]
        colour = 0
        while colour in neighbour_colors:
            colour += 1

        color_of[best_v] = colour
        uncolored.discard(best_v)

        # Update saturation of uncolored neighbours
        for u in range(n):
            if adj[best_v, u] > 0 and u in uncolored:
                saturation[u].add(colour)

    # ── Build time-slot schedule ─────────────────────────────
    num_colors = max(color_of.values()) + 1 if color_of else 0
    time_slots: dict[int, list[int]] = {c: [] for c in range(num_colors)}
    for v, c in color_of.items():
        time_slots[c].append(v)

    sizes = [len(ts) for ts in time_slots.values()] if time_slots else [0]
    max_par = max(sizes) if sizes else 0
    avg_par = float(np.mean(sizes)) if sizes else 0.0

    return ColoringResult(
        num_colors=num_colors,
        color_of=color_of,
        time_slots=time_slots,
        max_parallelism=max_par,
        avg_parallelism=avg_par,
    )


# ── Schedule helpers ─────────────────────────────────────────


def makespan(coloring: ColoringResult) -> int:
    """The makespan = number of time slots = chromatic number."""
    return coloring.num_colors


def validate_coloring(graph: ConflictGraph, coloring: ColoringResult) -> bool:
    """Check that no two adjacent nodes share a colour."""
    for edge in graph.edges:
        if coloring.color_of.get(edge.i) == coloring.color_of.get(edge.j):
            return False
    return True
