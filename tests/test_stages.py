"""Tests for the three straining stages."""

from __future__ import annotations

import numpy as np

from qstrainer.models.enums import TaskVerdict
from qstrainer.stages.ml import PredictiveStrainer
from qstrainer.stages.statistical import ConvergenceStrainer
from qstrainer.stages.threshold import RedundancyStrainer


class TestRedundancyStrainer:
    def test_productive_task_no_decisions(self, healthy_frame):
        rs = RedundancyStrainer()
        decisions = rs.check(healthy_frame)
        # Productive tasks should produce 0 decisions (or very few soft ones)
        hard_skips = [d for d in decisions if d.verdict == TaskVerdict.SKIP]
        assert len(hard_skips) == 0

    def test_redundant_produces_decisions(self, failing_frame):
        rs = RedundancyStrainer()
        decisions = rs.check(failing_frame)
        assert len(decisions) > 0

    def test_from_config(self):
        cfg = {"redundancy": {"gradient": {"floor": 1e-6, "low": 1e-4}}}
        rs = RedundancyStrainer.from_config(cfg)
        assert rs.gradient_norm_floor == 1e-6


class TestConvergenceStrainer:
    def test_warmup_no_redundancy(self, gen):
        cs = ConvergenceStrainer(z_threshold=3.0, min_samples=10)
        for _ in range(5):
            f = gen.generate_healthy("GPU-SS", "node-ss")
            score, signals = cs.update_and_score(f.gpu_id, f.to_vector())
        # During warmup, score should be 0
        assert score == 0.0

    def test_redundant_after_warmup(self, gen):
        cs = ConvergenceStrainer(z_threshold=3.0, min_samples=10)
        # Warm up with productive tasks
        for _ in range(30):
            f = gen.generate_healthy("GPU-SS2", "node-ss")
            cs.update_and_score(f.gpu_id, f.to_vector())
        # Send a redundant task (very different from productive baseline)
        f = gen.generate_failing("GPU-SS2", "node-ss")
        score, signals = cs.update_and_score(f.gpu_id, f.to_vector())
        # Should detect something (score > 0)
        assert score > 0.0

    def test_baseline_state_roundtrip(self, gen):
        cs = ConvergenceStrainer()
        for _ in range(50):
            f = gen.generate_healthy("GPU-BL", "node-bl")
            cs.update_and_score(f.gpu_id, f.to_vector())
        state = cs.get_baseline_state()
        cs2 = ConvergenceStrainer()
        cs2.load_baseline_state(state)
        assert "GPU-BL" in cs2._baselines


class TestPredictiveStrainer:
    def test_train_and_score(self, healthy_dataset):
        X, y = healthy_dataset
        X_valuable = X[y == 0]  # productive tasks
        X_redundant = X[y == 1]  # converging/redundant tasks

        ps = PredictiveStrainer(kernel="rbf", nu=0.05)
        ps.train(X_valuable)

        score_v = np.mean([ps.score(X_valuable[i]) for i in range(min(20, len(X_valuable)))])
        score_r = np.mean([ps.score(X_redundant[i]) for i in range(len(X_redundant))])

        # Redundant tasks should score higher (more redundant) on average
        assert score_r > score_v

    def test_state_roundtrip(self, healthy_dataset):
        X, y = healthy_dataset
        ps = PredictiveStrainer()
        ps.train(X[y == 0])
        state = ps.get_state()
        ps2 = PredictiveStrainer()
        ps2.load_state(state)
        # Both should give same score
        s1 = ps.score(X[0])
        s2 = ps2.score(X[0])
        assert abs(s1 - s2) < 1e-6
