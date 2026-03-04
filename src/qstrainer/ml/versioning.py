"""Model versioning and A/B testing for the ML anomaly detector.

Manages multiple trained model versions and enables shadow-mode
A/B comparisons to validate a new model before promoting it.

Terminology:
  - **Champion**: the active model serving real decisions.
  - **Challenger**: a newly trained model running in shadow mode.
  - **Promotion**: replacing the champion with the challenger
    after it demonstrates better performance.

Usage::

    from qstrainer.ml.versioning import ModelRegistry
    registry = ModelRegistry(storage_dir="runs/models/")
    vid = registry.register(detector, metadata={"trigger": "drift"})
    registry.promote(vid)   # make it the active champion
"""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ModelVersion:
    """Metadata for one trained model snapshot."""

    version_id: str
    created_at: float
    metrics: dict[str, float]  # training/eval metrics
    metadata: dict[str, Any]  # user-supplied context
    artifact_path: str | None = None
    is_champion: bool = False
    is_challenger: bool = False


@dataclass(slots=True)
class ABResult:
    """Per-frame comparison between champion and challenger."""

    frame_index: int
    champion_score: float
    challenger_score: float
    champion_health: str
    challenger_health: str


class ModelRegistry:
    """Versioned model storage with champion/challenger lifecycle.

    Parameters
    ----------
    storage_dir : str
        Directory for persisting model artifacts.
    max_versions : int
        Maximum number of versions to retain (FIFO).
    """

    def __init__(
        self,
        storage_dir: str = "runs/models/",
        max_versions: int = 10,
    ) -> None:
        self._dir = Path(storage_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max = max_versions
        self._versions: dict[str, ModelVersion] = {}
        self._champion_id: str | None = None
        self._challenger_id: str | None = None
        self._counter: int = 0

    # ── Register ─────────────────────────────────────────────

    def register(
        self,
        model_state: dict,
        *,
        metrics: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Register a new model version and persist its state.

        Parameters
        ----------
        model_state : dict
            State dict from ``PredictiveStrainer.get_state()``.
        metrics : dict
            Evaluation metrics (e.g., ``{"precision": 0.95}``).
        metadata : dict
            Free-form metadata (e.g., trigger reason, drift report).

        Returns
        -------
        str
            The version ID.
        """
        self._counter += 1
        vid = f"v{self._counter:04d}_{int(time.time())}"

        # Persist artifact
        artifact_path = self._dir / f"{vid}.pkl"
        with open(artifact_path, "wb") as f:
            pickle.dump(model_state, f, protocol=pickle.HIGHEST_PROTOCOL)

        version = ModelVersion(
            version_id=vid,
            created_at=time.time(),
            metrics=metrics or {},
            metadata=metadata or {},
            artifact_path=str(artifact_path),
        )
        self._versions[vid] = version

        # Prune oldest versions (keep champion/challenger)
        self._prune()

        logger.info("Registered model %s (metrics=%s)", vid, metrics or {})
        return vid

    # ── Champion / Challenger ────────────────────────────────

    def promote(self, version_id: str) -> None:
        """Promote a version to champion (active model)."""
        if version_id not in self._versions:
            raise KeyError(f"Version {version_id!r} not found")

        # Demote previous champion
        if self._champion_id and self._champion_id in self._versions:
            self._versions[self._champion_id].is_champion = False

        self._versions[version_id].is_champion = True
        self._versions[version_id].is_challenger = False
        self._champion_id = version_id

        # Clear challenger if it was promoted
        if self._challenger_id == version_id:
            self._challenger_id = None

        logger.info("Promoted %s to champion", version_id)

    def set_challenger(self, version_id: str) -> None:
        """Set a version as challenger for A/B testing."""
        if version_id not in self._versions:
            raise KeyError(f"Version {version_id!r} not found")

        if self._challenger_id and self._challenger_id in self._versions:
            self._versions[self._challenger_id].is_challenger = False

        self._versions[version_id].is_challenger = True
        self._challenger_id = version_id
        logger.info("Set %s as challenger", version_id)

    def dismiss_challenger(self) -> None:
        """Remove the current challenger without promoting."""
        if self._challenger_id and self._challenger_id in self._versions:
            self._versions[self._challenger_id].is_challenger = False
        self._challenger_id = None

    # ── Load ─────────────────────────────────────────────────

    def load_state(self, version_id: str) -> dict:
        """Load a model's state dict from disk."""
        v = self._versions.get(version_id)
        if v is None or v.artifact_path is None:
            raise KeyError(f"Version {version_id!r} not found or has no artifact")
        with open(v.artifact_path, "rb") as f:
            return dict(pickle.load(f))

    def load_champion_state(self) -> dict | None:
        """Load the champion model state (or None if no champion)."""
        if self._champion_id is None:
            return None
        return self.load_state(self._champion_id)

    def load_challenger_state(self) -> dict | None:
        """Load the challenger model state (or None if no challenger)."""
        if self._challenger_id is None:
            return None
        return self.load_state(self._challenger_id)

    # ── Queries ──────────────────────────────────────────────

    @property
    def champion_id(self) -> str | None:
        return self._champion_id

    @property
    def challenger_id(self) -> str | None:
        return self._challenger_id

    def list_versions(self) -> list[ModelVersion]:
        """All registered versions, newest first."""
        return sorted(
            self._versions.values(),
            key=lambda v: v.created_at,
            reverse=True,
        )

    def get_version(self, version_id: str) -> ModelVersion | None:
        return self._versions.get(version_id)

    # ── Pruning ──────────────────────────────────────────────

    def _prune(self) -> None:
        """Remove oldest versions beyond max, never pruning champion/challenger."""
        if len(self._versions) <= self._max:
            return
        by_age = sorted(self._versions.values(), key=lambda v: v.created_at)
        for v in by_age:
            if len(self._versions) <= self._max:
                break
            if v.is_champion or v.is_challenger:
                continue
            # Remove artifact
            if v.artifact_path:
                p = Path(v.artifact_path)
                if p.exists():
                    p.unlink()
            del self._versions[v.version_id]
            logger.debug("Pruned model version %s", v.version_id)


class ABTestRunner:
    """Run A/B comparisons between champion and challenger models.

    Scores each frame with both models and accumulates comparison
    metrics to decide whether to promote the challenger.

    Parameters
    ----------
    promote_after : int
        Number of frames to evaluate before making a promotion decision.
    promote_threshold : float
        Challenger must score >= this fraction better on anomaly
        detection (precision proxy) to be promoted.
    """

    def __init__(
        self,
        promote_after: int = 1000,
        promote_threshold: float = 0.05,
    ) -> None:
        self._promote_after = promote_after
        self._promote_thresh = promote_threshold
        self._results: list[ABResult] = []
        self._decided: bool = False
        self._decision: str | None = None  # "promote" or "dismiss"

    def record(
        self,
        frame_index: int,
        champion_score: float,
        challenger_score: float,
        champion_health: str = "",
        challenger_health: str = "",
    ) -> None:
        """Record one A/B observation."""
        self._results.append(
            ABResult(
                frame_index=frame_index,
                champion_score=champion_score,
                challenger_score=challenger_score,
                champion_health=champion_health,
                challenger_health=challenger_health,
            )
        )

    def evaluate(self) -> str | None:
        """Evaluate accumulated results.

        Returns ``"promote"``, ``"dismiss"``, or ``None`` if not enough data.
        """
        if len(self._results) < self._promote_after:
            return None

        if self._decided:
            return self._decision

        # Compare mean anomaly scores — challenger should have tighter
        # (lower) scores on healthy data and higher on anomalous data.
        # Simplified: compare mean absolute difference from 0.5 (calibration).
        champ_scores = np.array([r.champion_score for r in self._results])
        chall_scores = np.array([r.challenger_score for r in self._results])

        # Metric: lower false-alarm rate (scores near 0 on healthy frames)
        # Higher true positive rate (scores near 1 on anomalous frames)
        # Use variance as a proxy for calibration quality
        champ_var = float(np.var(champ_scores))
        chall_var = float(np.var(chall_scores))

        # Challenger should have higher variance (better separation)
        improvement = (chall_var - champ_var) / max(champ_var, 1e-9)

        self._decided = True
        if improvement > self._promote_thresh:
            self._decision = "promote"
            logger.info(
                "A/B result: PROMOTE challenger (improvement=%.1f%%)",
                improvement * 100,
            )
        else:
            self._decision = "dismiss"
            logger.info(
                "A/B result: DISMISS challenger (improvement=%.1f%%)",
                improvement * 100,
            )
        return self._decision

    @property
    def results(self) -> list[ABResult]:
        return list(self._results)

    @property
    def sample_count(self) -> int:
        return len(self._results)

    def reset(self) -> None:
        """Reset for a new A/B test."""
        self._results.clear()
        self._decided = False
        self._decision = None
