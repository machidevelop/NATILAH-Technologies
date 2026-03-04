"""Shared test fixtures for Q-Strainer."""

from __future__ import annotations

import numpy as np
import pytest

from qstrainer.ingestion.synthetic import SyntheticTelemetryGenerator
from qstrainer.models.frame import ComputeTask


@pytest.fixture
def gen() -> SyntheticTelemetryGenerator:
    return SyntheticTelemetryGenerator(seed=42)


@pytest.fixture
def healthy_frame(gen: SyntheticTelemetryGenerator) -> ComputeTask:
    """A productive compute task (meaningful gradients, loss improving)."""
    return gen.generate_healthy("GPU-TEST-0000", "node-test-000")


@pytest.fixture
def degrading_frame(gen: SyntheticTelemetryGenerator) -> ComputeTask:
    """A converging compute task (diminishing returns)."""
    return gen.generate_degrading("GPU-TEST-0001", "node-test-000", severity=0.7)


@pytest.fixture
def failing_frame(gen: SyntheticTelemetryGenerator) -> ComputeTask:
    """A fully redundant compute task (should be strained)."""
    return gen.generate_failing("GPU-TEST-0002", "node-test-000")


@pytest.fixture
def healthy_dataset(gen: SyntheticTelemetryGenerator):
    """Return (X, y) with 200 productive + 30 converging + 10 redundant tasks.

    y=0 for productive (valuable), y=1 for converging/redundant (strain candidates).
    """
    X_list, y_list = [], []
    for _ in range(200):
        f = gen.generate_healthy("GPU-DS", "node-ds")
        X_list.append(f.to_vector())
        y_list.append(0)
    for i in range(30):
        f = gen.generate_degrading("GPU-DS2", "node-ds", severity=i / 29)
        X_list.append(f.to_vector())
        y_list.append(1)
    for _ in range(10):
        f = gen.generate_failing("GPU-DS3", "node-ds")
        X_list.append(f.to_vector())
        y_list.append(1)
    return np.vstack(X_list), np.array(y_list, dtype=np.float64)
