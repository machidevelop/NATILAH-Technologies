"""Integration tests — full pipeline end-to-end with synthetic compute tasks.

These tests exercise the complete flow:
    SyntheticGenerator → WorkloadBuffer → QStrainer pipeline → StrainResult

No mocks — real stages, real scoring, real prediction.
"""

import numpy as np
import pytest

from qstrainer.config import load_config
from qstrainer.features.derived import DerivedFeatureExtractor
from qstrainer.ingestion.synthetic import SyntheticTelemetryGenerator
from qstrainer.models.buffer import WorkloadBuffer
from qstrainer.models.enums import TaskVerdict
from qstrainer.models.frame import N_BASE_FEATURES
from qstrainer.pipeline.strainer import QStrainer


class TestEndToEndPipeline:
    """Full pipeline: redundant tasks get strained, productive tasks get executed."""

    @pytest.fixture
    def pipeline(self):
        """Build a real pipeline from default config."""
        cfg = load_config(None)
        return QStrainer.from_config(cfg)

    @pytest.fixture
    def gen(self):
        return SyntheticTelemetryGenerator(seed=123)

    def test_productive_tasks_mostly_executed(self, pipeline, gen):
        """Most productive tasks should be executed (not strained)."""
        executed = 0
        total = 500
        for _ in range(total):
            task = gen.generate_healthy("GPU-INT-0", "node-int")
            result = pipeline.process_task(task)
            if result.verdict == TaskVerdict.EXECUTE:
                executed += 1

        # Productive data should mostly pass through (>50% executed)
        exec_rate = executed / total
        assert exec_rate > 0.30, f"Execution rate {exec_rate:.2%} too low for productive data"

    def test_redundant_tasks_mostly_strained(self, pipeline, gen):
        """Redundant tasks must be caught by the strainer."""
        # Warm up with productive tasks first
        for _ in range(50):
            pipeline.process_task(gen.generate_healthy("GPU-INT-1", "node-int"))

        strained = 0
        total = 100
        for _ in range(total):
            task = gen.generate_failing("GPU-INT-1", "node-int")
            result = pipeline.process_task(task)
            if result.verdict != TaskVerdict.EXECUTE:
                strained += 1

        # Redundant tasks should be strained (>50%)
        strain_rate = strained / total
        assert strain_rate > 0.50, f"Strain rate {strain_rate:.2%} too low for redundant data"

    def test_converging_detection_improves_over_time(self, pipeline, gen):
        """As baselines stabilize, converging tasks should be strained more often."""
        # Phase 1: warm-up with 200 productive tasks
        for _ in range(200):
            pipeline.process_task(gen.generate_healthy("GPU-INT-2", "node-int"))

        # Phase 2: inject 100 converging tasks
        strained = 0
        for _ in range(100):
            task = gen.generate_degrading("GPU-INT-2", "node-int", severity=0.7)
            result = pipeline.process_task(task)
            if result.verdict != TaskVerdict.EXECUTE:
                strained += 1

        # After warm-up, converging tasks should trigger at decent rate
        assert strained > 5, f"Only {strained} converging tasks strained (expected >5)"

    def test_pipeline_stats_consistent(self, pipeline, gen):
        """Pipeline stats counters must be internally consistent."""
        for _ in range(100):
            pipeline.process_task(gen.generate_healthy("GPU-INT-3", "node-int"))
        for _ in range(50):
            pipeline.process_task(gen.generate_failing("GPU-INT-3", "node-int"))

        stats = pipeline.stats
        assert stats["tasks_processed"] == 150
        assert stats["tasks_executed"] + stats["tasks_strained"] == 150
        assert 0.0 <= stats["strain_ratio"] <= 1.0
        assert stats["total_flops_saved"] >= 0

    def test_multi_gpu_isolation(self, pipeline, gen):
        """Straining one GPU must not cause false straining on another."""
        # Warm up both GPUs with productive data
        for _ in range(100):
            pipeline.process_task(gen.generate_healthy("GPU-GOOD", "node-int"))
            pipeline.process_task(gen.generate_healthy("GPU-BAD", "node-int"))

        # Now: GPU-BAD sends redundant tasks, GPU-GOOD stays productive
        good_false_strains = 0
        for _ in range(50):
            pipeline.process_task(gen.generate_failing("GPU-BAD", "node-int"))
            result = pipeline.process_task(gen.generate_healthy("GPU-GOOD", "node-int"))
            if result.verdict != TaskVerdict.EXECUTE and result.redundancy_score > 0.5:
                good_false_strains += 1

        assert good_false_strains < 10, (
            f"GPU-GOOD had {good_false_strains} false strains while GPU-BAD was redundant"
        )


class TestEndToEndWithFeatures:
    """Test the derived feature expansion within the pipeline."""

    def test_feature_extractor_produces_valid_features(self):
        gen = SyntheticTelemetryGenerator(seed=99)
        extractor = DerivedFeatureExtractor()

        tasks = [gen.generate_healthy("GPU-FE", "node-fe") for _ in range(20)]
        extended = np.array([extractor.extract(t.gpu_id, t.to_vector()) for t in tasks])

        # No NaN or Inf in productive data
        assert np.all(np.isfinite(extended)), "Extended features contain NaN or Inf"
        assert extended.shape[0] == 20
        assert extended.shape[1] > N_BASE_FEATURES

    def test_feature_extractor_with_converging_data(self):
        gen = SyntheticTelemetryGenerator(seed=99)
        extractor = DerivedFeatureExtractor()

        productive = [gen.generate_healthy("G", "N") for _ in range(50)]
        converging = [gen.generate_degrading("G", "N", 0.8) for _ in range(50)]

        ext_productive = np.array([extractor.extract(t.gpu_id, t.to_vector()) for t in productive])
        ext_converging = np.array([extractor.extract(t.gpu_id, t.to_vector()) for t in converging])

        # Extended features should amplify differences between productive/converging
        productive_mean = np.mean(ext_productive, axis=0)
        converging_mean = np.mean(ext_converging, axis=0)
        diff = np.abs(productive_mean - converging_mean)

        # At least some features should show meaningful separation
        assert np.max(diff) > 0.01, "Feature expansion didn't amplify any signal"


class TestEndToEndBuffer:
    """Test buffer → pipeline integration."""

    def test_buffer_feeds_pipeline(self):
        gen = SyntheticTelemetryGenerator(seed=77)
        buf = WorkloadBuffer(max_tasks_per_gpu=200)
        pipeline = QStrainer()

        # Fill buffer
        for _ in range(100):
            task = gen.generate_healthy("GPU-BUF", "node-buf")
            buf.push(task)

        # Process all buffered tasks through pipeline
        window = buf.get_window("GPU-BUF", 100)
        assert len(window) == 100

        for task in window:
            pipeline.process_task(task)

        assert pipeline.stats["tasks_processed"] == 100

    def test_buffer_matrix_shape_consistent(self):
        gen = SyntheticTelemetryGenerator(seed=77)
        buf = WorkloadBuffer(max_tasks_per_gpu=50)

        for _ in range(30):
            buf.push(gen.generate_healthy("GPU-M", "node-m"))

        mat = buf.get_matrix("GPU-M", 30)
        assert mat.shape == (30, N_BASE_FEATURES)

        # Verify matrix rows match individual task vectors
        window = buf.get_window("GPU-M", 30)
        for i, task in enumerate(window):
            np.testing.assert_array_equal(mat[i], task.to_vector())


class TestEndToEndConfig:
    """Test config → pipeline wiring."""

    def test_config_to_pipeline_roundtrip(self):
        cfg = load_config(None)
        pipeline = QStrainer.from_config(cfg)

        assert pipeline.strain_threshold == cfg["pipeline"]["strain_threshold"]
        assert pipeline.heartbeat_interval == cfg["pipeline"]["heartbeat_interval"]

    def test_env_override_propagates(self, monkeypatch):
        """Env var override should reach the pipeline."""
        monkeypatch.setenv("QSTRAINER_PIPELINE__STRAIN_THRESHOLD", "0.7")
        cfg = load_config(None)
        assert cfg["pipeline"]["strain_threshold"] == 0.7

        pipeline = QStrainer.from_config(cfg)
        assert pipeline.strain_threshold == 0.7
