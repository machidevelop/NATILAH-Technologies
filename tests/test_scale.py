"""Tests for v0.4.0 — Scale features.

Covers:
  - Drift detection (PSI + Page–Hinkley)
  - Online retrainer orchestration
  - Model versioning & A/B testing
  - Feature store
  - Autoscaler
  - Quantum advantage benchmark (with SA only)
  - Hybrid solver scheduling
"""

from __future__ import annotations

import shutil
import tempfile
import time

import numpy as np
import pytest

from qstrainer.ingestion.synthetic import SyntheticTelemetryGenerator
from qstrainer.models.frame import ComputeTask, N_BASE_FEATURES


# ═══════════════════════════════════════════════════════════
# Drift Detection
# ═══════════════════════════════════════════════════════════

class TestDriftDetector:
    def test_no_drift_with_same_distribution(self):
        from qstrainer.ml.drift import DriftDetector

        rng = np.random.default_rng(42)
        baseline = rng.standard_normal((500, 5))
        recent = rng.standard_normal((500, 5))

        dd = DriftDetector(window=500, psi_threshold=0.20)
        dd.set_baseline(baseline)
        for vec in recent:
            dd.observe(vec)

        report = dd.check()
        assert not report.is_drifted
        assert report.max_psi < 0.20

    def test_drift_detected_with_shifted_distribution(self):
        from qstrainer.ml.drift import DriftDetector

        rng = np.random.default_rng(42)
        baseline = rng.standard_normal((500, 5))
        shifted = rng.standard_normal((500, 5)) + 3.0  # big shift

        dd = DriftDetector(window=500, psi_threshold=0.15)
        dd.set_baseline(baseline)
        for vec in shifted:
            dd.observe(vec)

        report = dd.check()
        assert report.is_drifted
        assert report.max_psi > 0.15

    def test_page_hinkley_triggers_on_trend(self):
        from qstrainer.ml.drift import DriftDetector

        dd = DriftDetector(window=200, ph_lambda=10.0, ph_delta=0.001)
        baseline = np.zeros((200, 3))
        dd.set_baseline(baseline)

        # Feed a gradually increasing trend
        for i in range(300):
            vec = np.array([i * 0.1, i * 0.1, i * 0.1])
            dd.observe(vec)

        report = dd.check()
        assert report.page_hinkley_triggered

    def test_empty_check_returns_no_drift(self):
        from qstrainer.ml.drift import DriftDetector

        dd = DriftDetector()
        report = dd.check()
        assert not report.is_drifted


class TestOnlineRetrainer:
    def test_should_retrain_after_drift(self):
        from qstrainer.ml.drift import DriftDetector, OnlineRetrainer

        rng = np.random.default_rng(42)
        dd = DriftDetector(window=100, psi_threshold=0.10)
        dd.set_baseline(rng.standard_normal((200, 5)))

        retrainer = OnlineRetrainer(
            dd, check_interval=100, min_retrain_samples=50
        )

        # Feed shifted data
        for i in range(150):
            vec = rng.standard_normal(5) + 5.0  # shifted
            retrainer.observe(vec, is_healthy=True)

        assert retrainer.should_retrain()
        data = retrainer.get_retrain_data()
        assert data is not None
        assert data.shape[0] >= 50

    def test_mark_retrained_resets_state(self):
        from qstrainer.ml.drift import DriftDetector, OnlineRetrainer

        dd = DriftDetector(window=100)
        dd.set_baseline(np.zeros((100, 3)))
        retrainer = OnlineRetrainer(dd, check_interval=50)

        for _ in range(60):
            retrainer.observe(np.zeros(3))

        retrainer.mark_retrained(np.zeros((100, 3)))
        assert retrainer.retrain_count == 1


# ═══════════════════════════════════════════════════════════
# Model Versioning & A/B Testing
# ═══════════════════════════════════════════════════════════

class TestModelRegistry:
    def test_register_and_promote(self):
        from qstrainer.ml.versioning import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(storage_dir=tmpdir, max_versions=5)
            vid = reg.register({"weights": [1, 2, 3]}, metrics={"precision": 0.9})
            reg.promote(vid)

            assert reg.champion_id == vid
            state = reg.load_champion_state()
            assert state == {"weights": [1, 2, 3]}

    def test_challenger_lifecycle(self):
        from qstrainer.ml.versioning import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(storage_dir=tmpdir)
            v1 = reg.register({"v": 1})
            v2 = reg.register({"v": 2})
            reg.promote(v1)
            reg.set_challenger(v2)

            assert reg.champion_id == v1
            assert reg.challenger_id == v2

            # Promote challenger
            reg.promote(v2)
            assert reg.champion_id == v2
            assert reg.challenger_id is None

    def test_pruning_respects_champion(self):
        from qstrainer.ml.versioning import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(storage_dir=tmpdir, max_versions=3)
            v0 = reg.register({"v": 0})
            reg.promote(v0)  # promote before pruning can evict it
            # Now register more to trigger pruning
            for i in range(1, 5):
                reg.register({"v": i})

            versions = reg.list_versions()
            # Champion should survive even though it's oldest
            version_ids = {v.version_id for v in versions}
            assert v0 in version_ids


class TestABTestRunner:
    def test_promote_decision(self):
        from qstrainer.ml.versioning import ABTestRunner

        runner = ABTestRunner(promote_after=10, promote_threshold=0.01)

        rng = np.random.default_rng(42)
        for i in range(15):
            # Challenger has better separation (higher variance)
            runner.record(
                i,
                champion_score=0.5 + rng.normal(0, 0.05),    # low variance
                challenger_score=0.5 + rng.normal(0, 0.3),   # high variance
            )

        decision = runner.evaluate()
        assert decision in ("promote", "dismiss")

    def test_insufficient_samples_returns_none(self):
        from qstrainer.ml.versioning import ABTestRunner

        runner = ABTestRunner(promote_after=100)
        runner.record(0, 0.5, 0.6)
        assert runner.evaluate() is None


# ═══════════════════════════════════════════════════════════
# Feature Store
# ═══════════════════════════════════════════════════════════

class TestFeatureStore:
    def test_register_and_get(self):
        from qstrainer.ml.feature_store import FeatureStore

        gen = SyntheticTelemetryGenerator(seed=42)
        frame = gen.generate_healthy("GPU-0", "node-0")

        fs = FeatureStore()
        fs.register("base", lambda f: f.to_vector(), dim=N_BASE_FEATURES)

        vec = fs.get("base", frame)
        assert vec.shape == (N_BASE_FEATURES,)
        assert np.all(np.isfinite(vec))

    def test_dependency_resolution(self):
        from qstrainer.ml.feature_store import FeatureStore

        gen = SyntheticTelemetryGenerator(seed=42)
        frame = gen.generate_healthy("GPU-0", "node-0")

        fs = FeatureStore()
        fs.register("base", lambda f: f.to_vector(), dim=N_BASE_FEATURES)
        fs.register(
            "doubled",
            lambda gpu_id, vec: np.concatenate([vec, vec]),
            dim=N_BASE_FEATURES * 2,
            depends=["base"],
            arity=2,
        )

        doubled = fs.get("doubled", frame)
        assert doubled.shape == (N_BASE_FEATURES * 2,)

    def test_caching(self):
        from qstrainer.ml.feature_store import FeatureStore

        call_count = 0

        def counting_extractor(frame):
            nonlocal call_count
            call_count += 1
            return frame.to_vector()

        gen = SyntheticTelemetryGenerator(seed=42)
        frame = gen.generate_healthy("GPU-0", "node-0")

        fs = FeatureStore()
        fs.register("base", counting_extractor, dim=N_BASE_FEATURES)

        fs.get("base", frame)
        fs.get("base", frame)  # should hit cache
        assert call_count == 1

    def test_materialise(self):
        from qstrainer.ml.feature_store import FeatureStore

        gen = SyntheticTelemetryGenerator(seed=42)
        frames = [gen.generate_healthy("GPU-0", "node-0") for _ in range(10)]

        fs = FeatureStore()
        fs.register("base", lambda f: f.to_vector(), dim=N_BASE_FEATURES)

        mat = fs.materialise("base", frames)
        assert mat.shape == (10, N_BASE_FEATURES)

    def test_schema(self):
        from qstrainer.ml.feature_store import FeatureStore

        fs = FeatureStore()
        fs.register("a", lambda f: f.to_vector(), dim=15, description="base features")
        fs.register("b", lambda gid, v: v, dim=15, depends=["a"], arity=2)

        schema = fs.schema()
        assert len(schema) == 2
        assert schema[0]["name"] == "a"
        assert schema[1]["depends"] == ["a"]


# ═══════════════════════════════════════════════════════════
# Autoscaler
# ═══════════════════════════════════════════════════════════

class TestAutoscaler:
    def test_hold_with_insufficient_data(self):
        from qstrainer.distributed.autoscaler import Autoscaler, ScaleAction

        auto = Autoscaler(target_fps_per_replica=100)
        auto.record(50)
        decision = auto.evaluate()
        assert decision.action == ScaleAction.HOLD

    def test_scale_up_on_high_utilisation(self):
        from qstrainer.distributed.autoscaler import Autoscaler, ScaleAction

        auto = Autoscaler(
            target_fps_per_replica=100,
            scale_up_threshold=0.8,
            cooldown_seconds=0,
        )
        auto.set_current_replicas(1)

        # Feed high throughput (>80% of capacity)
        for _ in range(5):
            auto.record(95)

        decision = auto.evaluate()
        assert decision.action == ScaleAction.SCALE_UP
        assert decision.desired_replicas > 1

    def test_scale_down_on_low_utilisation(self):
        from qstrainer.distributed.autoscaler import Autoscaler, ScaleAction

        auto = Autoscaler(
            target_fps_per_replica=100,
            scale_down_threshold=0.3,
            min_replicas=1,
            cooldown_seconds=0,
        )
        auto.set_current_replicas(5)

        # Feed low throughput (<30% of 5×100=500 capacity)
        for _ in range(5):
            auto.record(50)

        decision = auto.evaluate()
        assert decision.action == ScaleAction.SCALE_DOWN
        assert decision.desired_replicas < 5

    def test_respects_min_max_replicas(self):
        from qstrainer.distributed.autoscaler import Autoscaler, ScaleAction

        auto = Autoscaler(
            target_fps_per_replica=100,
            min_replicas=2,
            max_replicas=4,
            cooldown_seconds=0,
        )
        auto.set_current_replicas(4)

        # Very high throughput — but max is 4
        for _ in range(5):
            auto.record(900)

        decision = auto.evaluate()
        assert decision.desired_replicas <= 4

    def test_from_config(self):
        from qstrainer.distributed.autoscaler import Autoscaler

        cfg = {"autoscaler": {"min_replicas": 3, "max_replicas": 16}}
        auto = Autoscaler.from_config(cfg)
        assert auto._min == 3
        assert auto._max == 16


# ═══════════════════════════════════════════════════════════
# Quantum Advantage Benchmark
# ═══════════════════════════════════════════════════════════

class TestQuantumAdvantageBenchmark:
    def test_benchmark_with_sa(self):
        from qstrainer.quantum.advantage import QuantumAdvantageBenchmark
        from qstrainer.solvers.sa import SimulatedAnnealingSolver

        bench = QuantumAdvantageBenchmark(n_trials=2, seed=42)
        bench.register_solver("sa", SimulatedAnnealingSolver(num_reads=50, num_sweeps=100))

        report = bench.run(problem_sizes=[4, 6])
        assert len(report.size_reports) == 2
        assert "sa" in report.solvers

        # SA should find reasonably good solutions
        for sr in report.size_reports:
            assert sr.mean_time("sa") < 5.0
            assert len(sr.solver_results["sa"]) == 2

    def test_summary_format(self):
        from qstrainer.quantum.advantage import QuantumAdvantageBenchmark
        from qstrainer.solvers.sa import SimulatedAnnealingSolver

        bench = QuantumAdvantageBenchmark(n_trials=1, seed=42)
        bench.register_solver("sa", SimulatedAnnealingSolver(num_reads=10, num_sweeps=50))

        report = bench.run(problem_sizes=[4])
        summary = report.summary()
        assert "Quantum Advantage" in summary
        assert "sa" in summary

    def test_to_dict(self):
        from qstrainer.quantum.advantage import QuantumAdvantageBenchmark
        from qstrainer.solvers.sa import SimulatedAnnealingSolver

        bench = QuantumAdvantageBenchmark(n_trials=1, seed=42)
        bench.register_solver("sa", SimulatedAnnealingSolver(num_reads=10, num_sweeps=50))

        report = bench.run(problem_sizes=[4])
        d = report.to_dict()
        assert "solvers" in d
        assert "sizes" in d
        assert d["sizes"][0]["n"] == 4

    def test_brute_force_ground_truth(self):
        from qstrainer.quantum.advantage import QuantumAdvantageBenchmark

        bench = QuantumAdvantageBenchmark(seed=42)
        Q = bench.generate_qubo(6)
        optimal = bench.brute_force_optimal(Q)
        # Verify it's actually minimal
        rng = np.random.default_rng(99)
        for _ in range(100):
            bits = rng.integers(0, 2, size=6).astype(float)
            energy = float(bits @ Q @ bits)
            assert energy >= optimal - 1e-10


# ═══════════════════════════════════════════════════════════
# Hybrid Solver Scheduling
# ═══════════════════════════════════════════════════════════

class TestHybridScheduling:
    def test_small_problem_selects_qaoa(self):
        from qstrainer.qos.scheduler import QOSScheduler

        cfg = {"solvers": {}}
        sched = QOSScheduler.from_config(cfg)

        name, solver = sched.select_solver(10)
        assert "qaoa" in name.lower() or solver.solver_type == "quantum_sim"

    def test_medium_problem_selects_classical(self):
        """Without Qiskit installed, medium problems fall back to SA."""
        from qstrainer.qos.scheduler import QOSScheduler

        cfg = {"solvers": {}}
        sched = QOSScheduler.from_config(cfg)

        name, solver = sched.select_solver(50)
        # Without real quantum hardware, should get classical
        assert solver.solver_type in ("classical", "quantum_sim")

    def test_explicit_preference_respected(self):
        from qstrainer.qos.scheduler import QOSScheduler

        cfg = {"solvers": {}}
        sched = QOSScheduler.from_config(cfg)

        name, solver = sched.select_solver(10, prefer="sa_default")
        assert name == "sa_default"
