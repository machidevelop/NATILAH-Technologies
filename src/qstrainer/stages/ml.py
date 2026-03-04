"""Stage 3 — Predictive Strainer (quantum-ready ML).

Learns to predict which compute tasks actually change the outcome,
and which can be safely strained (skipped/approximated).

TODAY:    Classical RBF kernel OneClassSVM (~1 ms per predict).
TOMORROW: Quantum kernel SVM (same interface, swap kernel).

The model is trained on VALUABLE compute tasks (tasks that produced
meaningful parameter updates).  New tasks that look like the valuable
ones → EXECUTE.  Tasks that look unlike them → candidates for straining.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

logger = logging.getLogger(__name__)


class PredictiveStrainer:
    """ML-based prediction of compute task value.

    Trained on tasks that produced meaningful results.
    Scores new tasks: high score = likely redundant (strain it),
    low score = likely valuable (execute it).

    Uses QUBO-selected features for quantum advantage on feature selection.
    """

    def __init__(self, nu: float = 0.05, kernel: str = "rbf") -> None:
        self.nu = nu
        self.kernel = kernel
        self._model: Optional[OneClassSVM] = None
        self._scaler: Optional[StandardScaler] = None
        self._selected_features: Optional[List[int]] = None

    @classmethod
    def from_config(cls, cfg: Dict) -> "PredictiveStrainer":
        mc = cfg.get("ml_predictor", {})
        return cls(
            nu=mc.get("nu", 0.05),
            kernel=mc.get("kernel", "rbf"),
        )

    def train(
        self,
        X_valuable: np.ndarray,
        selected_features: Optional[List[int]] = None,
    ) -> None:
        """Train on VALUABLE compute tasks — tasks that produced
        meaningful parameter updates / loss improvements.

        Tasks far from this distribution are likely redundant.
        """
        self._selected_features = selected_features
        if selected_features is not None:
            X_valuable = X_valuable[:, selected_features]

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_valuable)

        self._model = OneClassSVM(kernel=self.kernel, gamma="scale", nu=self.nu)
        self._model.fit(X_scaled)
        logger.info(
            "PredictiveStrainer trained on %d valuable tasks, %d features",
            X_scaled.shape[0],
            X_scaled.shape[1],
        )

    def score(self, feature_vector: np.ndarray) -> float:
        """Score one task.  Returns 0.0 (valuable, execute) → 1.0 (redundant, strain).

        Higher score = more redundant = better candidate for straining.
        """
        if self._model is None:
            return 0.0  # No model yet → assume task is valuable

        x = feature_vector.copy()
        if self._selected_features is not None:
            x = x[self._selected_features]
        x = self._scaler.transform(x.reshape(1, -1))  # type: ignore[union-attr]

        # decision_function: high = inside (valuable), low = outside (redundant)
        raw = self._model.decision_function(x)[0]
        # Flip: we want high score = redundant
        return float(1.0 / (1.0 + np.exp(raw * 2)))

    def batch_score(self, X: np.ndarray) -> np.ndarray:
        """Score a batch of tasks."""
        if self._model is None:
            return np.zeros(X.shape[0])

        if self._selected_features is not None:
            X = X[:, self._selected_features]
        X_scaled = self._scaler.transform(X)  # type: ignore[union-attr]

        raw = self._model.decision_function(X_scaled)
        return 1.0 / (1.0 + np.exp(raw * 2))

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def get_state(self) -> Dict:
        """Return state dict for checkpointing (JSON-safe)."""
        import pickle, base64

        state: Dict = {
            "nu": self.nu,
            "kernel": self.kernel,
            "selected_features": self._selected_features,
        }
        if self._model is not None:
            state["model_b64"] = base64.b64encode(pickle.dumps(self._model)).decode()
        if self._scaler is not None:
            state["scaler_b64"] = base64.b64encode(pickle.dumps(self._scaler)).decode()
        return state

    def load_state(self, state: Dict) -> None:
        """Restore from checkpoint state."""
        import pickle, base64

        self.nu = state.get("nu", self.nu)
        self.kernel = state.get("kernel", self.kernel)
        self._selected_features = state.get("selected_features")
        if "model_b64" in state:
            self._model = pickle.loads(base64.b64decode(state["model_b64"]))
        if "scaler_b64" in state:
            self._scaler = pickle.loads(base64.b64decode(state["scaler_b64"]))
