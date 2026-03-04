"""Synthetic compute task generator for development and testing.

Three profiles:
  - productive: meaningful gradients, loss improving — MUST execute
  - converging: loss plateauing, diminishing returns — candidates for straining
  - redundant:  near-zero gradients, duplicate data — should be strained

Calibrated to realistic ML training dynamics.
"""

from __future__ import annotations

import time

import numpy as np

from qstrainer.models.enums import ComputePhase, JobType
from qstrainer.models.frame import ComputeTask


class SyntheticTelemetryGenerator:
    """Generate realistic GPU compute tasks for dev/testing.

    Kept as SyntheticTelemetryGenerator for backward compatibility.
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self._t = time.time()
        self._step = 0

    def _tick(self) -> float:
        self._t += 0.1  # 10 Hz
        return self._t

    def _next_step(self) -> int:
        self._step += 1
        return self._step

    def generate_healthy(
        self,
        gpu_id: str = "GPU-0001",
        node_id: str = "node-01",
    ) -> ComputeTask:
        """Productive training step — meaningful gradients, loss improving."""
        step = self._next_step()
        loss = self.rng.uniform(0.5, 2.0)
        return ComputeTask(
            timestamp=self._tick(),
            task_id=f"task-{step:06d}",
            gpu_id=gpu_id,
            job_id=f"job-{gpu_id}",
            step_number=step,
            loss=loss,
            loss_delta=self.rng.uniform(-0.05, -0.001),  # improving
            gradient_norm=self.rng.uniform(0.01, 1.0),  # meaningful gradient
            gradient_variance=self.rng.uniform(0.001, 0.1),
            learning_rate=self.rng.uniform(1e-4, 1e-3),
            batch_size=int(self.rng.choice([32, 64, 128, 256])),
            epoch=int(self.rng.integers(0, 10)),
            epoch_progress=self.rng.uniform(0.0, 1.0),
            estimated_flops=self.rng.uniform(1e10, 1e12),
            estimated_time_s=self.rng.uniform(0.1, 2.0),
            memory_footprint_gb=self.rng.uniform(5.0, 40.0),
            compute_phase=ComputePhase.BACKWARD_PASS,
            job_type=JobType.TRAINING,
            convergence_score=self.rng.uniform(0.0, 0.4),  # not converged
            param_update_magnitude=self.rng.uniform(0.001, 0.1),  # meaningful updates
            data_similarity=self.rng.uniform(0.0, 0.3),  # diverse data
            flop_utilization=self.rng.uniform(0.5, 0.9),
            throughput_samples_per_sec=self.rng.uniform(500, 5000),
            model_name="test-model",
            node_id=node_id,
        )

    def generate_degrading(
        self,
        gpu_id: str = "GPU-0002",
        node_id: str = "node-01",
        severity: float = 0.5,
    ) -> ComputeTask:
        """Converging training — diminishing returns, approaching plateau."""
        s = float(np.clip(severity, 0.0, 1.0))
        step = self._next_step()
        return ComputeTask(
            timestamp=self._tick(),
            task_id=f"task-{step:06d}",
            gpu_id=gpu_id,
            job_id=f"job-{gpu_id}",
            step_number=step,
            loss=self.rng.uniform(0.05, 0.3),
            loss_delta=self.rng.uniform(-1e-4, 1e-4) * (1 - s),  # flattening
            gradient_norm=self.rng.uniform(1e-4, 1e-2) * (1 - s * 0.8),  # shrinking
            gradient_variance=self.rng.uniform(1e-4, 1e-2),
            learning_rate=self.rng.uniform(1e-5, 1e-4),
            batch_size=int(self.rng.choice([64, 128, 256])),
            epoch=int(self.rng.integers(10, 50)),
            epoch_progress=self.rng.uniform(0.0, 1.0),
            estimated_flops=self.rng.uniform(1e10, 1e12),
            estimated_time_s=self.rng.uniform(0.1, 2.0),
            memory_footprint_gb=self.rng.uniform(5.0, 40.0),
            compute_phase=ComputePhase.BACKWARD_PASS,
            job_type=JobType.TRAINING,
            convergence_score=self.rng.uniform(0.5 + s * 0.3, 0.7 + s * 0.25),  # converging
            param_update_magnitude=self.rng.uniform(1e-5, 1e-3) * (1 - s * 0.7),
            data_similarity=self.rng.uniform(0.3 + s * 0.3, 0.6 + s * 0.3),  # increasing overlap
            flop_utilization=self.rng.uniform(0.3, 0.6),
            throughput_samples_per_sec=self.rng.uniform(300, 3000),
            model_name="test-model",
            node_id=node_id,
        )

    def generate_failing(
        self,
        gpu_id: str = "GPU-0003",
        node_id: str = "node-01",
    ) -> ComputeTask:
        """Fully redundant task — converged, near-zero gradients, duplicate data."""
        step = self._next_step()
        return ComputeTask(
            timestamp=self._tick(),
            task_id=f"task-{step:06d}",
            gpu_id=gpu_id,
            job_id=f"job-{gpu_id}",
            step_number=step,
            loss=self.rng.uniform(0.01, 0.05),  # converged loss
            loss_delta=self.rng.uniform(-1e-9, 1e-9),  # flat
            gradient_norm=self.rng.uniform(1e-9, 1e-7),  # vanished
            gradient_variance=self.rng.uniform(1e-10, 1e-8),
            learning_rate=self.rng.uniform(1e-6, 1e-5),
            batch_size=int(self.rng.choice([64, 128])),
            epoch=int(self.rng.integers(50, 200)),
            epoch_progress=self.rng.uniform(0.0, 1.0),
            estimated_flops=self.rng.uniform(1e10, 1e12),
            estimated_time_s=self.rng.uniform(0.1, 2.0),
            memory_footprint_gb=self.rng.uniform(5.0, 40.0),
            compute_phase=ComputePhase.BACKWARD_PASS,
            job_type=JobType.TRAINING,
            convergence_score=self.rng.uniform(0.96, 1.0),  # fully converged
            param_update_magnitude=self.rng.uniform(1e-11, 1e-9),
            data_similarity=self.rng.uniform(0.98, 1.0),  # duplicate data
            flop_utilization=self.rng.uniform(0.1, 0.3),
            throughput_samples_per_sec=self.rng.uniform(100, 500),
            model_name="test-model",
            node_id=node_id,
        )

    def generate_fleet(
        self,
        n_gpus: int,
        n_tasks: int,
        redundant_rate: float = 0.02,
        converging_rate: float = 0.05,
    ) -> list[ComputeTask]:
        """Generate tasks for a fleet of GPUs."""
        tasks: list[ComputeTask] = []
        for gpu_idx in range(n_gpus):
            gpu_id = f"GPU-{gpu_idx:04d}"
            node_id = f"node-{gpu_idx // 8:03d}"

            r = self.rng.random()
            if r < redundant_rate:
                profile = "redundant"
            elif r < redundant_rate + converging_rate:
                profile = "converging"
            else:
                profile = "productive"

            for fi in range(n_tasks):
                if profile == "productive":
                    tasks.append(self.generate_healthy(gpu_id, node_id))
                elif profile == "converging":
                    sev = fi / max(n_tasks - 1, 1)
                    tasks.append(self.generate_degrading(gpu_id, node_id, sev))
                else:
                    tasks.append(self.generate_failing(gpu_id, node_id))
        return tasks
