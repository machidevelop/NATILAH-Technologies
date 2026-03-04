"""Property-based tests using Hypothesis.

Tests invariants that must hold for ANY valid compute task data,
not just hand-picked examples.
"""

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from qstrainer.models.buffer import WorkloadBuffer
from qstrainer.models.enums import TaskVerdict
from qstrainer.models.frame import FEATURE_NAMES, N_BASE_FEATURES, ComputeTask
from qstrainer.pipeline.strainer import QStrainer
from qstrainer.stages.threshold import RedundancyStrainer

# ── Strategies (data generators for Hypothesis) ─────────────

_step_counter = 0


@st.composite
def compute_tasks(draw):
    """Generate arbitrary valid ComputeTask instances."""
    global _step_counter
    _step_counter += 1
    return ComputeTask(
        timestamp=draw(st.floats(min_value=1e9, max_value=2e9, allow_nan=False)),
        task_id=f"task-hyp-{_step_counter}",
        gpu_id=draw(st.text(min_size=1, max_size=16, alphabet="GPU-0123456789")),
        job_id=draw(st.text(min_size=1, max_size=16, alphabet="job-0123456789")),
        step_number=draw(st.integers(min_value=0, max_value=1_000_000)),
        loss=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False)),
        loss_delta=draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False)),
        gradient_norm=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False)),
        gradient_variance=draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False)),
        learning_rate=draw(st.floats(min_value=1e-8, max_value=1.0, allow_nan=False)),
        batch_size=draw(st.integers(min_value=1, max_value=4096)),
        epoch=draw(st.integers(min_value=0, max_value=1000)),
        epoch_progress=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        estimated_flops=draw(st.floats(min_value=0.0, max_value=1e15, allow_nan=False)),
        estimated_time_s=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False)),
        memory_footprint_gb=draw(st.floats(min_value=0.0, max_value=200.0, allow_nan=False)),
        convergence_score=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        param_update_magnitude=draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False)),
        data_similarity=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        flop_utilization=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        throughput_samples_per_sec=draw(
            st.floats(min_value=0.0, max_value=100000.0, allow_nan=False)
        ),
    )


# ── Task Invariants ────────────────────────────────────────


class TestTaskProperties:
    """Properties that must hold for any ComputeTask."""

    @given(task=compute_tasks())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_to_vector_always_returns_correct_shape(self, task):
        """to_vector() always produces a (15,) float array."""
        vec = task.to_vector()
        assert vec.shape == (N_BASE_FEATURES,)
        assert vec.dtype == np.float64 or vec.dtype == np.float32

    @given(task=compute_tasks())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_to_vector_no_nan_for_valid_input(self, task):
        """Finite input fields should produce finite feature vectors."""
        vec = task.to_vector()
        assert np.all(np.isfinite(vec)), f"NaN/Inf in vector: {vec}"

    @given(task=compute_tasks())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_feature_names_count_matches_vector(self, task):
        """FEATURE_NAMES length must equal to_vector() length."""
        vec = task.to_vector()
        assert len(FEATURE_NAMES) == len(vec)

    def test_feature_names_classmethod_matches_constant(self):
        """feature_names() class method and FEATURE_NAMES constant must agree."""
        assert ComputeTask.feature_names() == FEATURE_NAMES


# ── Buffer Invariants ───────────────────────────────────────


class TestBufferProperties:
    """Properties that must hold for WorkloadBuffer under any input."""

    @given(
        capacity=st.integers(min_value=1, max_value=500),
        n_tasks=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_buffer_never_exceeds_capacity(self, capacity, n_tasks):
        """Buffer must never store more than max_tasks_per_gpu tasks."""
        buf = WorkloadBuffer(max_tasks_per_gpu=capacity)
        gen_rng = np.random.default_rng(42)

        for step, _ in enumerate(range(n_tasks), start=1):
            task = ComputeTask(
                timestamp=1e9,
                task_id=f"task-{step}",
                gpu_id="GPU-0",
                job_id="job-0",
                step_number=step,
                loss=gen_rng.random(),
                loss_delta=-gen_rng.random() * 0.01,
                gradient_norm=gen_rng.random(),
            )
            buf.push(task)

        window = buf.get_window("GPU-0", capacity + 100)
        assert len(window) <= capacity

    @given(
        capacity=st.integers(min_value=5, max_value=100),
        n_tasks=st.integers(min_value=1, max_value=200),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_get_matrix_rows_match_window(self, capacity, n_tasks):
        """get_matrix() rows must equal get_window() count."""
        buf = WorkloadBuffer(max_tasks_per_gpu=capacity)

        for i in range(n_tasks):
            task = ComputeTask(
                timestamp=float(i),
                task_id=f"task-{i}",
                gpu_id="GPU-0",
                job_id="job-0",
                step_number=i,
                loss=0.5,
                loss_delta=-0.01,
                gradient_norm=0.1,
            )
            buf.push(task)

        req = min(n_tasks, capacity)
        mat = buf.get_matrix("GPU-0", req)
        window = buf.get_window("GPU-0", req)
        assert mat.shape[0] == len(window)
        assert mat.shape[1] == N_BASE_FEATURES

    @given(capacity=st.integers(min_value=1, max_value=100))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_empty_buffer_returns_empty_matrix(self, capacity):
        """Empty buffer should return (0, 15) matrix, never crash."""
        buf = WorkloadBuffer(max_tasks_per_gpu=capacity)
        mat = buf.get_matrix("GPU-NONEXIST", 10)
        assert mat.shape == (0, N_BASE_FEATURES)


# ── Redundancy Strainer Invariants ──────────────────────────


class TestRedundancyProperties:
    """Properties of the redundancy stage."""

    @given(task=compute_tasks())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_check_always_returns_list(self, task):
        """RedundancyStrainer.check() must always return a list (possibly empty)."""
        rs = RedundancyStrainer()
        result = rs.check(task)
        assert isinstance(result, list)

    @given(task=compute_tasks())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_decisions_have_valid_verdict(self, task):
        """Every decision must have a valid TaskVerdict."""
        rs = RedundancyStrainer()
        for decision in rs.check(task):
            assert isinstance(decision.verdict, TaskVerdict)


# ── Pipeline Invariants ─────────────────────────────────────


class TestPipelineProperties:
    """Properties of the full QStrainer pipeline."""

    @given(task=compute_tasks())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_process_always_returns_strain_result(self, task):
        """process_task() must always return a valid StrainResult."""
        from qstrainer.models.alert import StrainResult

        pipeline = QStrainer()
        result = pipeline.process_task(task)
        assert isinstance(result, StrainResult)

    @given(task=compute_tasks())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_redundancy_score_in_range(self, task):
        """Redundancy score must be in [0, 1]."""
        pipeline = QStrainer()
        result = pipeline.process_task(task)
        assert 0.0 <= result.redundancy_score <= 1.0, (
            f"Redundancy score {result.redundancy_score} out of [0,1]"
        )

    @given(task=compute_tasks())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_stats_always_consistent_after_one_task(self, task):
        """After processing one task, stats must be internally consistent."""
        pipeline = QStrainer()
        pipeline.process_task(task)
        stats = pipeline.stats
        assert stats["tasks_processed"] == 1
        assert stats["tasks_executed"] + stats["tasks_strained"] == 1
