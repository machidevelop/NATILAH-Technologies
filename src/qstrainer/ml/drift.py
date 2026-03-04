"""Online model retraining with concept-drift detection.

Monitors the distribution of incoming telemetry features and triggers
retraining of the ML anomaly detector when the data has drifted too
far from the training baseline.

Drift detection methods:
  - **Page–Hinkley test** — detects sustained shifts in feature means
  - **Population Stability Index (PSI)** — compares recent vs baseline
    feature distributions

Usage::

    from qstrainer.ml.drift import DriftDetector, OnlineRetrainer
    detector = DriftDetector(window=500, psi_threshold=0.2)
    retrainer = OnlineRetrainer(detector, pipeline, buffer)
    # called each frame in the daemon loop:
    retrainer.observe(frame)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DriftReport:
    """Summary of a drift-detection check."""
    timestamp: float
    is_drifted: bool
    psi_scores: Dict[str, float]   # feature_name → PSI
    max_psi: float
    page_hinkley_triggered: bool
    n_samples_since_baseline: int


class DriftDetector:
    """Detect concept drift via PSI + Page–Hinkley test.

    Parameters
    ----------
    window : int
        Number of recent frames to compare against baseline.
    psi_threshold : float
        PSI > this triggers a drift flag (0.1 = slight, 0.2 = significant).
    ph_delta : float
        Page–Hinkley detection threshold (lower = more sensitive).
    ph_lambda : float
        Page–Hinkley allowance for deviation.
    n_bins : int
        Number of bins for PSI histogram computation.
    """

    def __init__(
        self,
        window: int = 500,
        psi_threshold: float = 0.20,
        ph_delta: float = 0.005,
        ph_lambda: float = 50.0,
        n_bins: int = 10,
    ) -> None:
        self._window = window
        self._psi_thresh = psi_threshold
        self._ph_delta = ph_delta
        self._ph_lambda = ph_lambda
        self._n_bins = n_bins

        self._baseline: Optional[np.ndarray] = None   # (n_baseline, n_features)
        self._recent: List[np.ndarray] = []            # ring of recent vectors
        self._n_since_baseline: int = 0

        # Page–Hinkley accumulators (per feature)
        self._ph_sum: Optional[np.ndarray] = None
        self._ph_mean: Optional[np.ndarray] = None
        self._ph_min: Optional[np.ndarray] = None
        self._ph_count: int = 0

    # ── Set baseline ─────────────────────────────────────────

    def set_baseline(self, X_baseline: np.ndarray) -> None:
        """Record the baseline feature distribution (from training data)."""
        self._baseline = X_baseline.copy()
        n_feat = X_baseline.shape[1]
        self._ph_sum = np.zeros(n_feat)
        self._ph_mean = np.zeros(n_feat)
        self._ph_min = np.zeros(n_feat)
        self._ph_count = 0
        self._n_since_baseline = 0
        logger.info(
            "Drift baseline set: %d samples × %d features",
            X_baseline.shape[0], n_feat,
        )

    # ── Observe ──────────────────────────────────────────────

    def observe(self, feature_vector: np.ndarray) -> None:
        """Feed a single observation (from a new frame's to_vector)."""
        self._recent.append(feature_vector)
        if len(self._recent) > self._window * 2:
            self._recent = self._recent[-self._window:]
        self._n_since_baseline += 1
        self._update_page_hinkley(feature_vector)

    # ── Check drift ──────────────────────────────────────────

    def check(self, feature_names: Optional[List[str]] = None) -> DriftReport:
        """Run drift detection and return a report.

        Call periodically (e.g. every ``window`` frames).
        """
        now = time.time()

        if self._baseline is None or len(self._recent) < self._window:
            return DriftReport(
                timestamp=now,
                is_drifted=False,
                psi_scores={},
                max_psi=0.0,
                page_hinkley_triggered=False,
                n_samples_since_baseline=self._n_since_baseline,
            )

        recent_matrix = np.array(self._recent[-self._window:])
        n_features = recent_matrix.shape[1]
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(n_features)]

        # PSI per feature
        psi_scores: Dict[str, float] = {}
        for i in range(n_features):
            psi = self._compute_psi(
                self._baseline[:, i], recent_matrix[:, i]
            )
            name = feature_names[i] if i < len(feature_names) else f"f{i}"
            psi_scores[name] = psi

        max_psi = max(psi_scores.values()) if psi_scores else 0.0
        ph_triggered = self._page_hinkley_triggered()

        is_drifted = max_psi > self._psi_thresh or ph_triggered

        report = DriftReport(
            timestamp=now,
            is_drifted=is_drifted,
            psi_scores=psi_scores,
            max_psi=max_psi,
            page_hinkley_triggered=ph_triggered,
            n_samples_since_baseline=self._n_since_baseline,
        )

        if is_drifted:
            logger.warning(
                "Drift detected: max_PSI=%.3f, PH=%s, samples=%d",
                max_psi, ph_triggered, self._n_since_baseline,
            )

        return report

    # ── PSI computation ──────────────────────────────────────

    def _compute_psi(self, baseline: np.ndarray, recent: np.ndarray) -> float:
        """Population Stability Index between baseline and recent distributions."""
        eps = 1e-6

        # Determine bins from the combined range
        combined = np.concatenate([baseline, recent])
        bins = np.linspace(combined.min() - eps, combined.max() + eps, self._n_bins + 1)

        baseline_hist, _ = np.histogram(baseline, bins=bins)
        recent_hist, _ = np.histogram(recent, bins=bins)

        # Convert to proportions
        baseline_pct = (baseline_hist + eps) / (baseline_hist.sum() + eps * self._n_bins)
        recent_pct = (recent_hist + eps) / (recent_hist.sum() + eps * self._n_bins)

        psi = np.sum((recent_pct - baseline_pct) * np.log(recent_pct / baseline_pct))
        return float(psi)

    # ── Page–Hinkley ─────────────────────────────────────────

    def _update_page_hinkley(self, vec: np.ndarray) -> None:
        """Update Page–Hinkley accumulators for all features."""
        if self._ph_sum is None:
            return

        self._ph_count += 1
        # Running mean
        self._ph_mean += (vec - self._ph_mean) / self._ph_count
        # Cumulative sum of deviations
        self._ph_sum += vec - self._ph_mean - self._ph_delta
        self._ph_min = np.minimum(self._ph_min, self._ph_sum)

    def _page_hinkley_triggered(self) -> bool:
        """True if PH test fires on ANY feature."""
        if self._ph_sum is None or self._ph_count < 30:
            return False
        ph_values = self._ph_sum - self._ph_min
        return bool(np.any(ph_values > self._ph_lambda))


class OnlineRetrainer:
    """Orchestrates drift detection and model retraining.

    Parameters
    ----------
    drift_detector : DriftDetector
        The drift detector instance to query.
    check_interval : int
        Check for drift every N frames.
    min_retrain_samples : int
        Minimum healthy frames needed to trigger a retrain.
    max_retrain_interval : int
        Force a retrain after this many frames even without drift.
    """

    def __init__(
        self,
        drift_detector: DriftDetector,
        *,
        check_interval: int = 500,
        min_retrain_samples: int = 200,
        max_retrain_interval: int = 50_000,
    ) -> None:
        self._detector = drift_detector
        self._check_interval = check_interval
        self._min_samples = min_retrain_samples
        self._max_interval = max_retrain_interval

        self._frame_count: int = 0
        self._frames_since_retrain: int = 0
        self._retrain_count: int = 0
        self._healthy_vectors: List[np.ndarray] = []
        self._last_report: Optional[DriftReport] = None

    def observe(self, feature_vector: np.ndarray, is_healthy: bool = True) -> Optional[DriftReport]:
        """Feed one frame.  Returns a DriftReport when a check is performed.

        Parameters
        ----------
        feature_vector : np.ndarray
            The frame's feature vector (from ``to_vector()``).
        is_healthy : bool
            Whether this frame was classified as healthy (used for
            retraining data — we only retrain on healthy samples).
        """
        self._frame_count += 1
        self._frames_since_retrain += 1
        self._detector.observe(feature_vector)

        if is_healthy:
            self._healthy_vectors.append(feature_vector)
            # Cap stored healthy vectors to avoid memory bloat
            if len(self._healthy_vectors) > self._min_samples * 5:
                self._healthy_vectors = self._healthy_vectors[-self._min_samples * 3:]

        # Periodic drift check
        if self._frame_count % self._check_interval == 0:
            report = self._detector.check()
            self._last_report = report
            return report

        return None

    def should_retrain(self) -> bool:
        """Whether the model should be retrained now."""
        # Forced retrain after max interval
        if self._frames_since_retrain >= self._max_interval:
            return True

        # Drift-triggered retrain (only if we have enough healthy data)
        if (
            self._last_report is not None
            and self._last_report.is_drifted
            and len(self._healthy_vectors) >= self._min_samples
        ):
            return True

        return False

    def get_retrain_data(self) -> Optional[np.ndarray]:
        """Return the healthy training data matrix, or None if insufficient."""
        if len(self._healthy_vectors) < self._min_samples:
            return None
        return np.array(self._healthy_vectors[-self._min_samples * 3:])

    def mark_retrained(self, new_baseline: np.ndarray) -> None:
        """Called after retraining to update the drift baseline."""
        self._detector.set_baseline(new_baseline)
        self._frames_since_retrain = 0
        self._retrain_count += 1
        self._healthy_vectors.clear()
        logger.info("Retrain #%d complete, baseline refreshed", self._retrain_count)

    @property
    def retrain_count(self) -> int:
        return self._retrain_count

    @property
    def last_report(self) -> Optional[DriftReport]:
        return self._last_report

    @classmethod
    def from_config(cls, cfg: dict) -> "OnlineRetrainer":
        """Build from config dict."""
        dc = cfg.get("drift", {})
        detector = DriftDetector(
            window=dc.get("window", 500),
            psi_threshold=dc.get("psi_threshold", 0.20),
            ph_delta=dc.get("ph_delta", 0.005),
            ph_lambda=dc.get("ph_lambda", 50.0),
        )
        return cls(
            drift_detector=detector,
            check_interval=dc.get("check_interval", 500),
            min_retrain_samples=dc.get("min_retrain_samples", 200),
            max_retrain_interval=dc.get("max_retrain_interval", 50_000),
        )
