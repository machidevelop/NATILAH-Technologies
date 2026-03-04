"""ComputeTask — a single unit of GPU work to be strained.

This is the fundamental input to Q-Strainer.  Each task represents one
step of GPU compute (training step, inference batch, gradient sync, etc.)
that Q-Strainer evaluates to decide: EXECUTE, APPROXIMATE, DEFER, or SKIP.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qstrainer.models.enums import ComputePhase, JobType

# 15 base features (stable contract — never reorder)
FEATURE_NAMES: list[str] = [
    "loss",
    "loss_delta",
    "gradient_norm",
    "gradient_variance",
    "learning_rate",
    "batch_size_norm",
    "compute_cost_norm",
    "memory_footprint_norm",
    "estimated_time_norm",
    "convergence_score",
    "epoch_progress",
    "param_update_magnitude",
    "data_similarity",
    "flop_utilization",
    "throughput_ratio",
]

N_BASE_FEATURES: int = len(FEATURE_NAMES)  # 15


@dataclass(slots=True)
class ComputeTask:
    """A single GPU compute task to be evaluated by Q-Strainer.

    Represents one training step, one inference batch, or one compute
    block that the strainer decides to execute, skip, or approximate.
    """

    timestamp: float
    task_id: str  # unique per task
    gpu_id: str  # which GPU would run this
    job_id: str  # parent job / training run
    step_number: int  # global step count within the job

    # ── Loss & gradients (the core signals for straining) ──
    loss: float  # current loss value
    loss_delta: float  # loss change from previous step
    gradient_norm: float  # L2 norm of gradient vector
    gradient_variance: float = 0.0  # variance across gradient entries

    # ── Training state ──────────────────────────────────────
    learning_rate: float = 1e-3
    batch_size: int = 32
    epoch: int = 0
    epoch_progress: float = 0.0  # 0.0–1.0 within current epoch

    # ── Compute cost signals ────────────────────────────────
    estimated_flops: float = 0.0  # estimated FLOPs for this task
    estimated_time_s: float = 0.0  # wall-clock seconds this would take
    memory_footprint_gb: float = 0.0  # VRAM required
    compute_phase: ComputePhase = ComputePhase.FORWARD_PASS
    job_type: JobType = JobType.TRAINING

    # ── Convergence signals ─────────────────────────────────
    convergence_score: float = 0.0  # 0.0 (not converging) → 1.0 (fully converged)
    param_update_magnitude: float = 0.0  # how much parameters would change
    data_similarity: float = 0.0  # similarity to previously seen batches (0–1)

    # ── Resource utilization ────────────────────────────────
    flop_utilization: float = 0.0  # actual / peak FLOPS (0–1)
    throughput_samples_per_sec: float = 0.0

    # ── Optional metadata ───────────────────────────────────
    model_name: str = ""
    node_id: str = ""
    tags: dict[str, str] = field(default_factory=dict)

    def to_vector(self) -> np.ndarray:
        """Convert to normalised 15-element feature vector (stable order).

        These are the features the three strainer stages use to decide
        whether this compute task is worth running.
        """
        return np.array(
            [
                self.loss,
                self.loss_delta,
                self.gradient_norm,
                self.gradient_variance,
                self.learning_rate * 1000,  # scale to ~1.0 range
                self.batch_size / 256.0,  # normalise around common sizes
                self.estimated_flops / 1e12,  # TFLOPs
                self.memory_footprint_gb / 80.0,  # fraction of common GPU VRAM
                self.estimated_time_s / 10.0,  # fraction of 10s
                self.convergence_score,
                self.epoch_progress,
                self.param_update_magnitude,
                self.data_similarity,
                self.flop_utilization,
                self.throughput_samples_per_sec / 10000.0,  # normalise
            ],
            dtype=np.float64,
        )

    @staticmethod
    def feature_names() -> list[str]:
        return list(FEATURE_NAMES)
