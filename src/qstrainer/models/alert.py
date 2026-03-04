"""StrainDecision and StrainResult — what comes out of the strainer.

Q-Strainer doesn't produce alerts about hardware health.
It produces DECISIONS about compute tasks: execute, skip, approximate, or defer.
Each decision comes with a cost/savings estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qstrainer.models.enums import StrainAction, TaskVerdict


@dataclass(slots=True)
class StrainDecision:
    """A single straining decision for one compute task."""

    verdict: TaskVerdict
    action: StrainAction
    reason: str
    metric: str  # which signal triggered this decision
    value: float  # the signal value
    threshold: float  # the threshold it crossed
    gpu_id: str
    job_id: str
    task_id: str
    timestamp: float

    # Cost estimates
    compute_saved_flops: float = 0.0
    time_saved_s: float = 0.0
    cost_saved_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.name,
            "action": self.action.name,
            "reason": self.reason,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "gpu_id": self.gpu_id,
            "job_id": self.job_id,
            "task_id": self.task_id,
            "timestamp": self.timestamp,
            "compute_saved_flops": self.compute_saved_flops,
            "time_saved_s": self.time_saved_s,
            "cost_saved_usd": self.cost_saved_usd,
        }


@dataclass
class StrainResult:
    """Complete result from the strainer pipeline for one compute task.

    Contains the verdict (execute/skip/approximate/defer), the reasons,
    and the savings metrics that matter: time, FLOPs, money.
    """

    timestamp: float
    task_id: str
    gpu_id: str
    job_id: str
    step_number: int

    # The decision
    verdict: TaskVerdict
    redundancy_score: float  # 0.0 (unique/valuable) → 1.0 (fully redundant)
    convergence_score: float  # 0.0 (not converged) → 1.0 (fully converged)
    confidence: float  # how confident the strainer is in this decision

    # What the strainer found
    dominant_signals: list[tuple[str, float]]
    decisions: list[StrainDecision] = field(default_factory=list)

    # Savings
    compute_saved_flops: float = 0.0
    time_saved_s: float = 0.0
    cost_saved_usd: float = 0.0
    quality_impact: float = 0.0  # estimated accuracy/loss impact of straining

    # Pipeline metadata
    tasks_analyzed: int = 0
    tasks_strained: int = 0  # tasks that were skipped/approximated/deferred
    strain_ratio: float = 0.0  # fraction of compute eliminated
    strainer_method: str = ""

    def summary(self) -> str:
        savings = f"saved {self.time_saved_s:.1f}s" if self.time_saved_s > 0 else ""
        return (
            f"[{self.verdict.name}] task={self.task_id} | "
            f"redundancy={self.redundancy_score:.3f} "
            f"convergence={self.convergence_score:.3f} "
            f"conf={self.confidence:.2f}"
            f"{f' | {savings}' if savings else ''}"
            f"{f' ({len(self.decisions)} decisions)' if self.decisions else ''}"
        )

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "task_id": self.task_id,
            "gpu_id": self.gpu_id,
            "job_id": self.job_id,
            "step_number": self.step_number,
            "verdict": self.verdict.name,
            "redundancy_score": self.redundancy_score,
            "convergence_score": self.convergence_score,
            "confidence": self.confidence,
            "dominant_signals": self.dominant_signals,
            "decisions": [d.to_dict() for d in self.decisions],
            "compute_saved_flops": self.compute_saved_flops,
            "time_saved_s": self.time_saved_s,
            "cost_saved_usd": self.cost_saved_usd,
            "quality_impact": self.quality_impact,
            "tasks_analyzed": self.tasks_analyzed,
            "tasks_strained": self.tasks_strained,
            "strain_ratio": self.strain_ratio,
            "strainer_method": self.strainer_method,
        }
