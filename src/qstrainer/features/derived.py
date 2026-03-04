"""Derived feature extractor — expand 15 base compute features to 55+.

These derived features capture the signals that matter for straining:
convergence trajectory, gradient dynamics, compute efficiency ratios,
and cross-correlations that reveal redundant computation.

Pushes the QUBO feature selection from 15 → 55 variables,
where the combinatorial explosion (2^55 ≈ 3.6×10^16 subsets) makes
quantum solvers genuinely relevant.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from qstrainer.models.frame import FEATURE_NAMES

# Meaningful cross-feature pairs for compute straining (domain knowledge)
CROSS_PAIRS: list[tuple[int, int]] = [
    (0, 1),  # loss × loss_delta (convergence direction)
    (0, 2),  # loss × gradient_norm (loss-gradient coupling)
    (1, 2),  # loss_delta × gradient_norm (effective learning signal)
    (2, 4),  # gradient_norm × learning_rate (effective step size)
    (2, 3),  # gradient_norm × gradient_variance (gradient noise ratio)
    (5, 6),  # batch_size × compute_cost (batch efficiency)
    (6, 8),  # compute_cost × estimated_time (cost-time correlation)
    (9, 11),  # convergence_score × param_update_magnitude (convergence quality)
    (0, 9),  # loss × convergence_score (loss-convergence alignment)
    (12, 6),  # data_similarity × compute_cost (redundant compute signal)
    (13, 14),  # flop_utilization × throughput (hardware efficiency)
    (1, 9),  # loss_delta × convergence (stagnation detector)
]


def extended_feature_count() -> int:
    """Total: 15 base + 15 deltas + 15 rolling_std + 12 cross + 2 CV + 1 trend = 60."""
    return 15 + 15 + 15 + len(CROSS_PAIRS) + 2 + 1


def extended_feature_names() -> list[str]:
    base = list(FEATURE_NAMES)
    names = list(base)
    names += [f"d_{n}" for n in base]
    names += [f"std_{n}" for n in base]
    cross_names = [
        "loss×delta",
        "loss×grad",
        "delta×grad",
        "grad×lr",
        "grad×var",
        "batch×cost",
        "cost×time",
        "conv×update",
        "loss×conv",
        "sim×cost",
        "flops×thru",
        "delta×conv",
    ]
    names += cross_names
    names += ["cv_mean", "cv_max"]
    names += ["loss_trend"]
    return names


class DerivedFeatureExtractor:
    """Expand 15 base compute features into ~60 derived features.

    Categories:
      - Rates of change (how fast loss/gradients are changing)
      - Rolling statistics (stability/volatility signals)
      - Cross-correlations (compute efficiency & redundancy signals)
      - Coefficient of variation (noise/signal ratio)
      - Loss trend (linear regression slope over window)
    """

    def __init__(self, window_size: int = 10) -> None:
        self.window_size = window_size
        self._history: dict[str, list[np.ndarray]] = defaultdict(list)

    def extract(self, gpu_id: str, base_vector: np.ndarray) -> np.ndarray:
        """Expand base 15-feature vector into ~60 features."""
        self._history[gpu_id].append(base_vector.copy())
        if len(self._history[gpu_id]) > self.window_size:
            self._history[gpu_id] = self._history[gpu_id][-self.window_size :]

        history = self._history[gpu_id]
        features = [base_vector]  # 15 base

        # Rates of change (15) — how fast each signal is moving
        delta = history[-1] - history[-2] if len(history) >= 2 else np.zeros_like(base_vector)
        features.append(delta)

        # Rolling std (15) — volatility of each signal
        if len(history) >= 3:
            mat = np.vstack(history)
            rolling_std = np.std(mat, axis=0)
        else:
            rolling_std = np.zeros_like(base_vector)
        features.append(rolling_std)

        # Cross-correlations (12) — compute efficiency signals
        cross = [base_vector[i] * base_vector[j] for i, j in CROSS_PAIRS]
        features.append(np.array(cross))

        # Coefficient of variation (2) — noise ratio
        if len(history) >= 5:
            mat = np.vstack(history[-5:])
            mean_abs = np.abs(np.mean(mat, axis=0))
            std_val = np.std(mat, axis=0)
            cv = np.divide(
                std_val,
                mean_abs,
                out=np.zeros_like(mean_abs),
                where=mean_abs > 1e-10,
            )
            features.append(np.array([np.mean(cv), np.max(cv)]))
        else:
            features.append(np.zeros(2))

        # Loss trend (1) — linear regression slope of loss over window
        if len(history) >= 3:
            losses = np.array([h[0] for h in history])  # feature 0 = loss
            x = np.arange(len(losses), dtype=np.float64)
            slope = np.polyfit(x, losses, 1)[0]
            features.append(np.array([slope]))
        else:
            features.append(np.zeros(1))

        return np.concatenate(features)

    def reset(self, gpu_id: str | None = None) -> None:
        if gpu_id:
            self._history.pop(gpu_id, None)
        else:
            self._history.clear()
