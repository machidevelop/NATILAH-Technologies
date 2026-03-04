"""Quantum Kernel Anomaly Detector — drop-in replacement for classical RBF.

Swap kernel_provider to switch between:
  - QuantumKernelProvider  (statevector sim)
  - Qiskit FidelityQuantumKernel (real QPU)
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from qstrainer.quantum.kernel_provider import QuantumKernelProvider

logger = logging.getLogger(__name__)


class QuantumKernelDetector:
    """Anomaly detector using quantum kernel + OneClassSVM."""

    def __init__(
        self, n_qubits: int = 8, nu: float = 0.05, reps: int = 2
    ) -> None:
        self.n_qubits = n_qubits
        self.nu = nu
        self.kernel_provider = QuantumKernelProvider(
            n_qubits=n_qubits, reps=reps
        )
        self._model: Optional[OneClassSVM] = None
        self._scaler: Optional[StandardScaler] = None
        self._selected_features: Optional[List[int]] = None
        self._X_train: Optional[np.ndarray] = None

    def train(
        self,
        X_normal: np.ndarray,
        selected_features: Optional[List[int]] = None,
        max_train_samples: int = 80,
    ) -> None:
        self._selected_features = selected_features
        if selected_features is not None:
            X_normal = X_normal[:, selected_features]

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_normal)

        if X_scaled.shape[0] > max_train_samples:
            idx = np.random.default_rng(42).choice(
                X_scaled.shape[0], max_train_samples, replace=False
            )
            X_scaled = X_scaled[idx]

        X_proj = (
            X_scaled[:, : self.n_qubits]
            if X_scaled.shape[1] > self.n_qubits
            else X_scaled
        )
        X_proj = np.pi * (X_proj - X_proj.min(axis=0)) / (
            X_proj.max(axis=0) - X_proj.min(axis=0) + 1e-10
        )
        self._X_train = X_proj

        logger.info(
            "Computing quantum kernel matrix (%d x %d)...",
            X_proj.shape[0],
            X_proj.shape[0],
        )
        t0 = time.perf_counter()
        K_train = self.kernel_provider.kernel_matrix(X_proj)
        logger.info("Quantum kernel computed in %.2fs", time.perf_counter() - t0)

        self._model = OneClassSVM(kernel="precomputed", nu=self.nu)
        self._model.fit(K_train)

    def score(self, feature_vector: np.ndarray) -> float:
        if self._model is None or self._X_train is None:
            return 0.0

        x = feature_vector.copy()
        if self._selected_features is not None:
            x = x[self._selected_features]
        x = self._scaler.transform(x.reshape(1, -1))  # type: ignore[union-attr]

        x_proj = (
            x[0, : self.n_qubits] if x.shape[1] > self.n_qubits else x[0]
        )
        x_proj = np.clip(
            np.pi * (x_proj - 0) / (np.pi + 1e-10), 0, np.pi
        )

        k_vec = np.array(
            [
                self.kernel_provider.kernel_value(x_proj, self._X_train[j])
                for j in range(self._X_train.shape[0])
            ]
        ).reshape(1, -1)

        raw = self._model.decision_function(k_vec)[0]
        return float(1.0 / (1.0 + np.exp(raw * 2)))
