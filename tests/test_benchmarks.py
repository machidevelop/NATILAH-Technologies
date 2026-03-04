"""Load and performance benchmark tests.

Validates that the pipeline meets latency and throughput targets
under realistic fleet-scale conditions.

Markers:
    @pytest.mark.slow — excluded from default test runs
"""

import time
import statistics

import pytest
import numpy as np

from qstrainer.ingestion.synthetic import SyntheticTelemetryGenerator
from qstrainer.pipeline.strainer import QStrainer
from qstrainer.models.buffer import WorkloadBuffer
from qstrainer.features.derived import DerivedFeatureExtractor


@pytest.mark.slow
class TestPipelineLatency:
    """Per-task latency must be <1 ms for the core pipeline."""

    def test_single_task_latency_under_1ms(self):
        """Single-task processing should be well under 1 ms."""
        gen = SyntheticTelemetryGenerator(seed=42)
        pipeline = QStrainer()

        # Warm up
        for _ in range(100):
            pipeline.process_task(gen.generate_healthy("GPU-0", "N"))

        # Measure
        latencies = []
        for _ in range(1000):
            task = gen.generate_healthy("GPU-0", "N")
            t0 = time.perf_counter_ns()
            pipeline.process_task(task)
            latencies.append((time.perf_counter_ns() - t0) / 1e6)  # ms

        p50 = statistics.median(latencies)
        p95 = sorted(latencies)[int(0.95 * len(latencies))]
        p99 = sorted(latencies)[int(0.99 * len(latencies))]

        print(f"\nPipeline latency: p50={p50:.3f}ms  p95={p95:.3f}ms  p99={p99:.3f}ms")

        assert p50 < 1.0, f"p50 latency {p50:.3f}ms exceeds 1ms target"
        assert p99 < 5.0, f"p99 latency {p99:.3f}ms exceeds 5ms target"

    def test_to_vector_latency(self):
        """ComputeTask.to_vector() must be <0.1 ms."""
        gen = SyntheticTelemetryGenerator(seed=42)
        task = gen.generate_healthy("GPU-0", "N")

        latencies = []
        for _ in range(5000):
            t0 = time.perf_counter_ns()
            task.to_vector()
            latencies.append((time.perf_counter_ns() - t0) / 1e6)

        p99 = sorted(latencies)[int(0.99 * len(latencies))]
        print(f"\nto_vector latency: p99={p99:.4f}ms")
        assert p99 < 0.1, f"to_vector p99 {p99:.4f}ms exceeds 0.1ms target"


@pytest.mark.slow
class TestFleetThroughput:
    """Fleet-scale throughput benchmarks."""

    def test_1000_gpu_throughput(self):
        """Process 1000 GPUs x 1 task in under 1 second."""
        gen = SyntheticTelemetryGenerator(seed=42)
        pipeline = QStrainer()

        tasks = []
        for gpu_idx in range(1000):
            tasks.append(gen.generate_healthy(f"GPU-{gpu_idx:04d}", "node-bench"))

        t0 = time.perf_counter()
        for task in tasks:
            pipeline.process_task(task)
        elapsed = time.perf_counter() - t0

        tps = len(tasks) / elapsed
        print(f"\n1000-GPU batch: {elapsed:.3f}s ({tps:.0f} tasks/sec)")
        assert elapsed < 1.0, f"1000-GPU batch took {elapsed:.3f}s (target <1s)"

    def test_sustained_throughput_100hz(self):
        """Sustain 100 Hz x 8 GPUs (800 tasks/sec) for 5 simulated seconds."""
        gen = SyntheticTelemetryGenerator(seed=42)
        pipeline = QStrainer()

        n_gpus = 8
        hz = 100
        duration_s = 5
        total_tasks = n_gpus * hz * duration_s  # 4000 tasks

        tasks = []
        for _ in range(total_tasks):
            gpu_id = f"GPU-{np.random.randint(0, n_gpus):04d}"
            tasks.append(gen.generate_healthy(gpu_id, "node-bench"))

        t0 = time.perf_counter()
        for task in tasks:
            pipeline.process_task(task)
        elapsed = time.perf_counter() - t0

        achieved_tps = total_tasks / elapsed
        target_tps = n_gpus * hz
        print(f"\nSustained: {total_tasks} tasks in {elapsed:.3f}s "
              f"({achieved_tps:.0f} tps, target={target_tps} tps)")
        assert achieved_tps > target_tps, (
            f"Only achieved {achieved_tps:.0f} tps (need {target_tps})"
        )


@pytest.mark.slow
class TestBufferPerformance:
    """Buffer operation benchmarks."""

    def test_push_throughput(self):
        """Buffer push must handle >100k tasks/sec."""
        gen = SyntheticTelemetryGenerator(seed=42)
        buf = WorkloadBuffer(max_tasks_per_gpu=10000)

        tasks = [gen.generate_healthy("GPU-0", "N") for _ in range(10000)]

        t0 = time.perf_counter()
        for t in tasks:
            buf.push(t)
        elapsed = time.perf_counter() - t0

        tps = len(tasks) / elapsed
        print(f"\nBuffer push: {tps:.0f} tasks/sec")
        assert tps > 100_000, f"Buffer push only {tps:.0f} tps (target >100k)"

    def test_get_matrix_performance(self):
        """get_matrix for 1000 tasks must be <10 ms."""
        gen = SyntheticTelemetryGenerator(seed=42)
        buf = WorkloadBuffer(max_tasks_per_gpu=2000)

        for _ in range(1000):
            buf.push(gen.generate_healthy("GPU-0", "N"))

        latencies = []
        for _ in range(100):
            t0 = time.perf_counter_ns()
            buf.get_matrix("GPU-0", 1000)
            latencies.append((time.perf_counter_ns() - t0) / 1e6)

        p50 = statistics.median(latencies)
        print(f"\nget_matrix(1000): p50={p50:.2f}ms")
        assert p50 < 50.0, f"get_matrix p50 {p50:.2f}ms exceeds 50ms"


@pytest.mark.slow
class TestFeatureExtractorPerformance:
    """DerivedFeatureExtractor benchmarks."""

    def test_transform_throughput(self):
        """Feature expansion: 1000 tasks from 15->60 features in <100 ms."""
        gen = SyntheticTelemetryGenerator(seed=42)
        tasks = [gen.generate_healthy("G", "N") for _ in range(1000)]
        extractor = DerivedFeatureExtractor()

        t0 = time.perf_counter()
        for t in tasks:
            extractor.extract(t.gpu_id, t.to_vector())
        elapsed_ms = (time.perf_counter() - t0) * 1000

        print(f"\nFeature transform (1000x15->60): {elapsed_ms:.1f}ms")
        assert elapsed_ms < 500, f"Feature transform took {elapsed_ms:.1f}ms (target <500ms)"

