"""Tests for qstrainer.models — ComputeTask, WorkloadBuffer, Enums, StrainDecision."""

from __future__ import annotations

import numpy as np
import pytest

from qstrainer.models.enums import TaskVerdict, StrainAction, GPUType, ComputePhase, JobType
from qstrainer.models.frame import ComputeTask, FEATURE_NAMES, N_BASE_FEATURES
from qstrainer.models.buffer import WorkloadBuffer
from qstrainer.models.alert import StrainDecision, StrainResult


class TestEnums:
    def test_task_verdict_members(self):
        assert TaskVerdict.EXECUTE.name == "EXECUTE"
        assert len(TaskVerdict) == 4

    def test_strain_action_ordering(self):
        assert StrainAction.PASS_THROUGH.value < StrainAction.OPTIMISE.value
        assert StrainAction.OPTIMISE.value < StrainAction.REDUCE.value
        assert StrainAction.REDUCE.value < StrainAction.ELIMINATE.value

    def test_gpu_type_specs(self):
        assert GPUType.A100_40GB.vram_gb == 40
        assert GPUType.A100_40GB.flops_fp16 > 0
        assert GPUType.A100_40GB.cost_per_hour > 0

    def test_compute_phase_members(self):
        assert len(ComputePhase) == 8
        assert ComputePhase.FORWARD_PASS.name == "FORWARD_PASS"

    def test_job_type_members(self):
        assert len(JobType) == 5
        assert JobType.TRAINING.name == "TRAINING"


class TestComputeTask:
    def test_to_vector_shape(self, healthy_frame: ComputeTask):
        vec = healthy_frame.to_vector()
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (N_BASE_FEATURES,)

    def test_feature_names_length(self):
        assert len(FEATURE_NAMES) == N_BASE_FEATURES

    def test_feature_names_class_method(self):
        names = ComputeTask.feature_names()
        assert len(names) == N_BASE_FEATURES

    def test_vector_values_reasonable(self, healthy_frame: ComputeTask):
        vec = healthy_frame.to_vector()
        # loss should be positive
        assert vec[0] > 0
        # loss_delta should be negative (improving)
        assert vec[1] < 0
        # gradient_norm should be positive
        assert vec[2] > 0

    def test_redundant_task_different(self, healthy_frame, failing_frame):
        v_h = healthy_frame.to_vector()
        v_f = failing_frame.to_vector()
        assert not np.allclose(v_h, v_f)


class TestWorkloadBuffer:
    def test_add_and_retrieve(self, gen):
        buf = WorkloadBuffer(max_tasks_per_gpu=10)
        tasks = [gen.generate_healthy("GPU-BUF", "node-buf") for _ in range(5)]
        for t in tasks:
            buf.push(t)
        assert buf.total_tasks == 5
        assert len(buf.get_window("GPU-BUF", 100)) == 5

    def test_max_tasks_limit(self, gen):
        buf = WorkloadBuffer(max_tasks_per_gpu=3)
        for _ in range(10):
            buf.push(gen.generate_healthy("GPU-LIM", "node-lim"))
        assert len(buf.get_window("GPU-LIM", 100)) == 3

    def test_get_matrix(self, gen):
        buf = WorkloadBuffer(max_tasks_per_gpu=20)
        for _ in range(5):
            buf.push(gen.generate_healthy("GPU-MAT", "node-mat"))
        mat = buf.get_matrix("GPU-MAT", 20)
        assert mat.shape == (5, N_BASE_FEATURES)


class TestStrainDecision:
    def test_decision_creation(self):
        decision = StrainDecision(
            verdict=TaskVerdict.SKIP,
            action=StrainAction.ELIMINATE,
            reason="Gradient vanished",
            metric="gradient_norm",
            value=1e-9,
            threshold=1e-7,
            gpu_id="GPU-TEST",
            job_id="job-test",
            task_id="task-001",
            timestamp=1.0,
        )
        assert decision.verdict == TaskVerdict.SKIP
        assert decision.metric == "gradient_norm"

    def test_strain_result_summary(self, healthy_frame):
        result = StrainResult(
            timestamp=healthy_frame.timestamp,
            task_id=healthy_frame.task_id,
            gpu_id=healthy_frame.gpu_id,
            job_id=healthy_frame.job_id,
            step_number=healthy_frame.step_number,
            verdict=TaskVerdict.EXECUTE,
            redundancy_score=0.1,
            convergence_score=0.2,
            confidence=0.5,
            dominant_signals=[],
            decisions=[],
        )
        s = result.summary()
        assert "EXECUTE" in s
        assert "0.100" in s

