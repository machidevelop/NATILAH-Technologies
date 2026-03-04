"""Stage 1 — Redundancy Strainer (deterministic, <0.1 ms, NEVER quantum).

Fast deterministic checks that catch obviously redundant compute:
  - Gradient too small → training step won't change anything
  - Loss fully converged → no point continuing
  - Duplicate/near-duplicate batches → same work twice
  - Parameter update below noise floor → wasted FLOPs

This is the cheapest filter — catches the easy wins first.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from qstrainer.models.enums import TaskVerdict, StrainAction
from qstrainer.models.frame import ComputeTask
from qstrainer.models.alert import StrainDecision

logger = logging.getLogger(__name__)


class RedundancyStrainer:
    """Deterministic rule-based compute redundancy checks.  <0.1 ms/task."""

    def __init__(
        self,
        gradient_norm_floor: float = 1e-7,
        gradient_norm_low: float = 1e-5,
        loss_delta_floor: float = 1e-8,
        loss_delta_low: float = 1e-5,
        convergence_threshold: float = 0.95,
        convergence_warn: float = 0.85,
        data_similarity_threshold: float = 0.98,
        data_similarity_warn: float = 0.90,
        param_update_floor: float = 1e-9,
    ) -> None:
        self.gradient_norm_floor = gradient_norm_floor
        self.gradient_norm_low = gradient_norm_low
        self.loss_delta_floor = loss_delta_floor
        self.loss_delta_low = loss_delta_low
        self.convergence_threshold = convergence_threshold
        self.convergence_warn = convergence_warn
        self.data_similarity_threshold = data_similarity_threshold
        self.data_similarity_warn = data_similarity_warn
        self.param_update_floor = param_update_floor

    @classmethod
    def from_config(cls, cfg: Dict) -> "RedundancyStrainer":
        rc = cfg.get("redundancy", {})
        grad = rc.get("gradient", {})
        loss = rc.get("loss", {})
        conv = rc.get("convergence", {})
        data = rc.get("data", {})
        return cls(
            gradient_norm_floor=grad.get("floor", 1e-7),
            gradient_norm_low=grad.get("low", 1e-5),
            loss_delta_floor=loss.get("floor", 1e-8),
            loss_delta_low=loss.get("low", 1e-5),
            convergence_threshold=conv.get("threshold", 0.95),
            convergence_warn=conv.get("warn", 0.85),
            data_similarity_threshold=data.get("threshold", 0.98),
            data_similarity_warn=data.get("warn", 0.90),
            param_update_floor=rc.get("param_update_floor", 1e-9),
        )

    def check(self, task: ComputeTask) -> List[StrainDecision]:
        """Check a compute task for obvious redundancy.

        Returns a list of StrainDecisions. Empty list = no redundancy found.
        Multiple decisions possible (e.g., both gradient AND loss converged).
        """
        decisions: List[StrainDecision] = []
        ts = task.timestamp
        gid = task.gpu_id
        jid = task.job_id
        tid = task.task_id

        # ── Gradient norm check: is the gradient too small to matter? ──
        if task.gradient_norm <= self.gradient_norm_floor:
            decisions.append(StrainDecision(
                verdict=TaskVerdict.SKIP,
                action=StrainAction.ELIMINATE,
                gpu_id=gid, job_id=jid, task_id=tid,
                reason=f"Gradient norm {task.gradient_norm:.2e} below floor — step does nothing",
                metric="gradient_norm", value=task.gradient_norm,
                threshold=self.gradient_norm_floor, timestamp=ts,
                compute_saved_flops=task.estimated_flops,
                time_saved_s=task.estimated_time_s,
            ))
        elif task.gradient_norm <= self.gradient_norm_low:
            decisions.append(StrainDecision(
                verdict=TaskVerdict.APPROXIMATE,
                action=StrainAction.REDUCE,
                gpu_id=gid, job_id=jid, task_id=tid,
                reason=f"Gradient norm {task.gradient_norm:.2e} very low — use cheaper approximation",
                metric="gradient_norm", value=task.gradient_norm,
                threshold=self.gradient_norm_low, timestamp=ts,
                compute_saved_flops=task.estimated_flops * 0.5,
                time_saved_s=task.estimated_time_s * 0.5,
            ))

        # ── Loss delta check: has loss stopped changing? ──
        if abs(task.loss_delta) <= self.loss_delta_floor:
            decisions.append(StrainDecision(
                verdict=TaskVerdict.SKIP,
                action=StrainAction.ELIMINATE,
                gpu_id=gid, job_id=jid, task_id=tid,
                reason=f"Loss delta {task.loss_delta:.2e} — loss plateau, compute is wasted",
                metric="loss_delta", value=abs(task.loss_delta),
                threshold=self.loss_delta_floor, timestamp=ts,
                compute_saved_flops=task.estimated_flops,
                time_saved_s=task.estimated_time_s,
            ))
        elif abs(task.loss_delta) <= self.loss_delta_low:
            decisions.append(StrainDecision(
                verdict=TaskVerdict.APPROXIMATE,
                action=StrainAction.REDUCE,
                gpu_id=gid, job_id=jid, task_id=tid,
                reason=f"Loss delta {task.loss_delta:.2e} — diminishing returns",
                metric="loss_delta", value=abs(task.loss_delta),
                threshold=self.loss_delta_low, timestamp=ts,
                compute_saved_flops=task.estimated_flops * 0.3,
                time_saved_s=task.estimated_time_s * 0.3,
            ))

        # ── Convergence score check: has the model converged? ──
        if task.convergence_score >= self.convergence_threshold:
            decisions.append(StrainDecision(
                verdict=TaskVerdict.SKIP,
                action=StrainAction.ELIMINATE,
                gpu_id=gid, job_id=jid, task_id=tid,
                reason=f"Convergence score {task.convergence_score:.3f} — model is converged",
                metric="convergence_score", value=task.convergence_score,
                threshold=self.convergence_threshold, timestamp=ts,
                compute_saved_flops=task.estimated_flops,
                time_saved_s=task.estimated_time_s,
            ))
        elif task.convergence_score >= self.convergence_warn:
            decisions.append(StrainDecision(
                verdict=TaskVerdict.APPROXIMATE,
                action=StrainAction.OPTIMISE,
                gpu_id=gid, job_id=jid, task_id=tid,
                reason=f"Convergence score {task.convergence_score:.3f} — nearly converged",
                metric="convergence_score", value=task.convergence_score,
                threshold=self.convergence_warn, timestamp=ts,
                compute_saved_flops=task.estimated_flops * 0.3,
                time_saved_s=task.estimated_time_s * 0.3,
            ))

        # ── Data similarity check: is this batch near-duplicate? ──
        if task.data_similarity >= self.data_similarity_threshold:
            decisions.append(StrainDecision(
                verdict=TaskVerdict.SKIP,
                action=StrainAction.ELIMINATE,
                gpu_id=gid, job_id=jid, task_id=tid,
                reason=f"Data similarity {task.data_similarity:.3f} — near-duplicate batch",
                metric="data_similarity", value=task.data_similarity,
                threshold=self.data_similarity_threshold, timestamp=ts,
                compute_saved_flops=task.estimated_flops,
                time_saved_s=task.estimated_time_s,
            ))
        elif task.data_similarity >= self.data_similarity_warn:
            decisions.append(StrainDecision(
                verdict=TaskVerdict.DEFER,
                action=StrainAction.OPTIMISE,
                gpu_id=gid, job_id=jid, task_id=tid,
                reason=f"Data similarity {task.data_similarity:.3f} — consider deferring",
                metric="data_similarity", value=task.data_similarity,
                threshold=self.data_similarity_warn, timestamp=ts,
                compute_saved_flops=task.estimated_flops * 0.2,
                time_saved_s=task.estimated_time_s * 0.2,
            ))

        # ── Parameter update magnitude: is the update below noise? ──
        if task.param_update_magnitude <= self.param_update_floor:
            decisions.append(StrainDecision(
                verdict=TaskVerdict.SKIP,
                action=StrainAction.ELIMINATE,
                gpu_id=gid, job_id=jid, task_id=tid,
                reason=f"Param update {task.param_update_magnitude:.2e} below noise floor",
                metric="param_update_magnitude", value=task.param_update_magnitude,
                threshold=self.param_update_floor, timestamp=ts,
                compute_saved_flops=task.estimated_flops,
                time_saved_s=task.estimated_time_s,
            ))

        return decisions
