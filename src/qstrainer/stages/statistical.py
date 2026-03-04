"""Stage 2 — Convergence Strainer (Welford's Online Algorithm).

O(1) memory and compute per update.  Tracks the training trajectory
and scores how much each compute task contributes to convergence.

Catches gradual convergence that deterministic thresholds miss:
  - Loss plateau detection via rolling statistics
  - Gradient distribution shift (vanishing/exploding)
  - Training oscillation detection
  - Diminishing returns scoring

NEVER quantum — always fast, always online.
"""

from __future__ import annotations

import logging

import numpy as np

from qstrainer.models.frame import FEATURE_NAMES

logger = logging.getLogger(__name__)


class ConvergenceStrainer:
    """Per-GPU/job Welford's online convergence tracking.

    Tracks the statistical behaviour of compute signals over time
    and scores how redundant each new task is relative to the
    established training trajectory.
    """

    def __init__(self, z_threshold: float = 3.0, min_samples: int = 20) -> None:
        self.z_threshold = z_threshold
        self.min_samples = min_samples
        self._baselines: dict[str, dict] = {}

    @classmethod
    def from_config(cls, cfg: dict) -> ConvergenceStrainer:
        sc = cfg.get("convergence", {})
        return cls(
            z_threshold=sc.get("z_threshold", 3.0),
            min_samples=sc.get("min_samples", 20),
        )

    def update_and_score(
        self, gpu_id: str, feature_vector: np.ndarray
    ) -> tuple[float, list[tuple[str, float]]]:
        """Update baseline and return (redundancy_score, dominant_signals).

        redundancy_score: 0.0 = this task is unique/valuable
                          1.0 = this task is fully redundant (strain it)
        """
        n_features = len(feature_vector)
        names = FEATURE_NAMES

        if gpu_id not in self._baselines:
            self._baselines[gpu_id] = {
                "mean": np.zeros(n_features),
                "m2": np.zeros(n_features),
                "count": 0,
            }

        bl = self._baselines[gpu_id]
        bl["count"] += 1
        n = bl["count"]

        # Welford's online update
        delta = feature_vector - bl["mean"]
        bl["mean"] += delta / n
        delta2 = feature_vector - bl["mean"]
        bl["m2"] += delta * delta2

        if n < self.min_samples:
            return 0.0, []  # Not enough data to judge

        variance = bl["m2"] / (n - 1)
        std = np.sqrt(np.maximum(variance, 1e-10))
        z_scores = np.abs((feature_vector - bl["mean"]) / std)

        # LOW z-scores = task is similar to what we've seen = REDUNDANT
        # HIGH z-scores = task is different from baseline = VALUABLE
        # (This is the OPPOSITE of anomaly detection — we WANT to strain
        #  the boring stuff, not flag the unusual stuff)
        redundant_mask = z_scores < (self.z_threshold * 0.3)  # very close to mean
        if redundant_mask.all():
            # Everything is close to the mean → highly redundant
            score = float(1.0 - np.mean(z_scores) / self.z_threshold)
            score = max(min(score, 1.0), 0.5)
        elif redundant_mask.sum() > n_features * 0.7:
            # Mostly redundant
            score = float(0.3 + 0.4 * (redundant_mask.sum() / n_features))
            score = min(score, 0.8)
        else:
            # Task has significant deviations → valuable compute
            novel_z = z_scores[~redundant_mask]
            score = float(1.0 - min(np.mean(novel_z) / (self.z_threshold * 3), 1.0))
            score = max(score, 0.0)

        # Report which signals are most redundant or most novel
        top_idx = np.argsort(z_scores)[:5]  # lowest z = most redundant
        dominant = [(names[i] if i < len(names) else f"f{i}", float(z_scores[i])) for i in top_idx]

        return score, dominant

    def reset(self, gpu_id: str | None = None) -> None:
        if gpu_id:
            self._baselines.pop(gpu_id, None)
        else:
            self._baselines.clear()

    def get_baseline_state(self) -> dict:
        """Return serialisable baseline state for checkpointing."""
        state = {}
        for gid, bl in self._baselines.items():
            state[gid] = {
                "mean": bl["mean"].tolist(),
                "m2": bl["m2"].tolist(),
                "count": bl["count"],
            }
        return state

    def load_baseline_state(self, state: dict) -> None:
        """Restore baselines from a checkpoint."""
        self._baselines.clear()
        for gid, data in state.items():
            self._baselines[gid] = {
                "mean": np.array(data["mean"]),
                "m2": np.array(data["m2"]),
                "count": data["count"],
            }
