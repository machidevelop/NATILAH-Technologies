"""Core enumerations for Q-Strainer — GPU compute workload strainer."""

from __future__ import annotations

from enum import Enum, auto


class TaskVerdict(Enum):
    """What the strainer decides for a compute task."""

    EXECUTE = auto()      # Must run — contributes meaningfully
    APPROXIMATE = auto()  # Run a cheaper approximation (lower precision, subset)
    DEFER = auto()        # Postpone — batch with similar tasks later
    SKIP = auto()         # Don't run — redundant or converged


class StrainAction(Enum):
    """Action severity / type of the straining decision."""

    PASS_THROUGH = 0   # No straining, task executes normally
    OPTIMISE = 1       # Task runs but with resource optimisation
    REDUCE = 2         # Significant compute reduction applied
    ELIMINATE = 3       # Task entirely eliminated (biggest savings)


class ComputePhase(Enum):
    """Phases of GPU compute that can be strained."""

    FORWARD_PASS = auto()
    BACKWARD_PASS = auto()
    GRADIENT_SYNC = auto()
    OPTIMIZER_STEP = auto()
    DATA_LOADING = auto()
    CHECKPOINT = auto()
    INFERENCE = auto()
    UNKNOWN = auto()


class JobType(Enum):
    """Types of GPU workloads Q-Strainer can optimise."""

    TRAINING = auto()
    FINE_TUNING = auto()
    INFERENCE = auto()
    DATA_PREPROCESSING = auto()
    HYPERPARAMETER_SEARCH = auto()


class GPUType(Enum):
    """Supported NVIDIA GPU models."""

    A100_40GB = auto()
    A100_80GB = auto()
    H100_80GB = auto()
    H200_141GB = auto()
    B200 = auto()

    @property
    def vram_gb(self) -> int:
        return _VRAM_CAPS[self]

    @property
    def flops_fp16(self) -> float:
        """Peak FP16 TFLOPS."""
        return _PEAK_FLOPS[self]

    @property
    def cost_per_hour(self) -> float:
        """Approximate cloud $/hr (on-demand)."""
        return _COST_PER_HOUR[self]


_VRAM_CAPS = {
    GPUType.A100_40GB: 40,
    GPUType.A100_80GB: 80,
    GPUType.H100_80GB: 80,
    GPUType.H200_141GB: 141,
    GPUType.B200: 192,
}

_PEAK_FLOPS = {
    GPUType.A100_40GB: 312.0,
    GPUType.A100_80GB: 312.0,
    GPUType.H100_80GB: 989.0,
    GPUType.H200_141GB: 989.0,
    GPUType.B200: 2250.0,
}

_COST_PER_HOUR = {
    GPUType.A100_40GB: 1.10,
    GPUType.A100_80GB: 1.60,
    GPUType.H100_80GB: 2.50,
    GPUType.H200_141GB: 3.50,
    GPUType.B200: 5.00,
}
