"""QStrainer — the main three-stage compute workload strainer.

Q-Strainer intercepts GPU compute tasks and decides which ones to
EXECUTE, SKIP, APPROXIMATE, or DEFER — saving GPU hours and cost
while maintaining training quality.

Stage 1: Redundancy   (deterministic, <0.1 ms) — catches obvious waste
Stage 2: Convergence  (Welford's,     <1 ms)   — tracks trajectory
Stage 3: Predictive   (kernel SVM,    <10 ms)  — ML predicts task value

Tasks that pass all three stages → EXECUTE.
Tasks caught by any stage → decision (SKIP/APPROXIMATE/DEFER) with
savings estimate in FLOPs, wall-clock seconds, and dollars.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from qstrainer.models.alert import StrainDecision, StrainResult
from qstrainer.models.enums import TaskVerdict, StrainAction
from qstrainer.models.frame import ComputeTask
from qstrainer.stages.ml import PredictiveStrainer
from qstrainer.stages.statistical import ConvergenceStrainer
from qstrainer.stages.threshold import RedundancyStrainer
from qstrainer.tracing import trace_stage, add_span_event

logger = logging.getLogger(__name__)


class QStrainer:
    """Main strainer engine — three-stage GPU compute workload filter.

    Processes compute tasks and decides: execute, skip, approximate, or defer.
    Tracks cumulative savings (FLOPs, time, cost).
    """

    def __init__(
        self,
        redundancy: Optional[RedundancyStrainer] = None,
        convergence: Optional[ConvergenceStrainer] = None,
        predictor: Optional[PredictiveStrainer] = None,
        strain_threshold: float = 0.5,
        heartbeat_interval: int = 100,
    ) -> None:
        self.redundancy = redundancy or RedundancyStrainer()
        self.convergence = convergence or ConvergenceStrainer()
        self.predictor = predictor
        self.strain_threshold = strain_threshold
        self.heartbeat_interval = heartbeat_interval

        # Cumulative counters
        self._task_count: int = 0
        self._strained_count: int = 0   # tasks that were skipped/approximated/deferred
        self._executed_count: int = 0    # tasks that passed through
        self._total_flops_saved: float = 0.0
        self._total_time_saved: float = 0.0
        self._total_cost_saved: float = 0.0
        self._verdict_counts: Dict[str, int] = defaultdict(int)

    @classmethod
    def from_config(cls, cfg: Dict) -> "QStrainer":
        pc = cfg.get("pipeline", {})
        return cls(
            redundancy=RedundancyStrainer.from_config(cfg),
            convergence=ConvergenceStrainer.from_config(cfg),
            predictor=PredictiveStrainer.from_config(cfg),
            strain_threshold=pc.get("strain_threshold", 0.5),
            heartbeat_interval=pc.get("heartbeat_interval", 100),
        )

    def process_task(
        self, task: ComputeTask
    ) -> StrainResult:
        """Process one compute task through the three-stage strainer.

        Always returns a StrainResult — either EXECUTE (do the work)
        or SKIP/APPROXIMATE/DEFER (save the compute).
        """
        self._task_count += 1
        vec = task.to_vector()
        decisions: List[StrainDecision] = []

        # ── Stage 1: Redundancy check (deterministic, <0.1 ms) ──
        with trace_stage("redundancy_check", gpu_id=task.gpu_id):
            redundancy_decisions = self.redundancy.check(task)
            decisions.extend(redundancy_decisions)

        # If Stage 1 found hard SKIP signals, short-circuit
        has_skip = any(d.verdict == TaskVerdict.SKIP for d in redundancy_decisions)
        if has_skip:
            return self._make_result(
                task, vec, TaskVerdict.SKIP,
                redundancy_score=1.0,
                convergence_score=task.convergence_score,
                decisions=decisions,
                method="redundancy_skip",
            )

        # ── Stage 2: Convergence scoring (Welford's, <1 ms) ──
        with trace_stage("convergence_score", gpu_id=task.gpu_id):
            conv_score, conv_signals = self.convergence.update_and_score(
                task.gpu_id, vec
            )

        # ── Stage 3: Predictive scoring (<10 ms) ──
        with trace_stage("predictive_score", gpu_id=task.gpu_id):
            ml_score = self.predictor.score(vec) if self.predictor else 0.0

        # ── Combine scores ──────────────────────────────────
        # Higher = more redundant = better candidate for straining
        redundancy_score = max(conv_score, ml_score)
        dominant = conv_signals

        # If Stage 1 found NO absolute red flags, this task looks productive
        # by hard criteria (gradient is flowing, loss is changing, etc.).
        # Discount the convergence score to avoid flagging healthy repeated work.
        if not redundancy_decisions:
            redundancy_score *= 0.3

        # Apply Stage 1 soft decisions (APPROXIMATE/DEFER)
        if redundancy_decisions:
            # Stage 1 found soft signals — boost redundancy score
            stage1_boost = sum(
                0.2 for d in redundancy_decisions
                if d.verdict in (TaskVerdict.APPROXIMATE, TaskVerdict.DEFER)
            )
            redundancy_score = min(redundancy_score + stage1_boost, 1.0)

        # ── Make verdict ────────────────────────────────────
        if redundancy_score >= self.strain_threshold:
            if redundancy_score >= 0.8:
                verdict = TaskVerdict.SKIP
            elif redundancy_score >= 0.6:
                verdict = TaskVerdict.APPROXIMATE
            else:
                verdict = TaskVerdict.DEFER
        else:
            verdict = TaskVerdict.EXECUTE

        return self._make_result(
            task, vec, verdict,
            redundancy_score=redundancy_score,
            convergence_score=conv_score,
            decisions=decisions,
            dominant=dominant,
            method="full_pipeline",
        )

    def _make_result(
        self, task: ComputeTask, vec: np.ndarray,
        verdict: TaskVerdict,
        redundancy_score: float,
        convergence_score: float,
        decisions: List[StrainDecision],
        dominant: list | None = None,
        method: str = "",
    ) -> StrainResult:
        """Build result and update cumulative savings."""
        # Calculate savings for this task
        if verdict == TaskVerdict.SKIP:
            flops_saved = task.estimated_flops
            time_saved = task.estimated_time_s
        elif verdict == TaskVerdict.APPROXIMATE:
            flops_saved = task.estimated_flops * 0.5
            time_saved = task.estimated_time_s * 0.5
        elif verdict == TaskVerdict.DEFER:
            flops_saved = task.estimated_flops * 0.2
            time_saved = task.estimated_time_s * 0.2
        else:
            flops_saved = 0.0
            time_saved = 0.0

        # Estimate cost saved (rough: time * $/hr / 3600)
        cost_saved = time_saved * 2.50 / 3600.0  # assume H100 rate

        # Update counters
        self._verdict_counts[verdict.name] += 1
        if verdict != TaskVerdict.EXECUTE:
            self._strained_count += 1
            self._total_flops_saved += flops_saved
            self._total_time_saved += time_saved
            self._total_cost_saved += cost_saved
        else:
            self._executed_count += 1

        # Confidence based on how much data we have
        confidence = min(self._task_count / 100.0, 1.0)

        return StrainResult(
            timestamp=task.timestamp,
            task_id=task.task_id,
            gpu_id=task.gpu_id,
            job_id=task.job_id,
            step_number=task.step_number,
            verdict=verdict,
            redundancy_score=redundancy_score,
            convergence_score=convergence_score,
            confidence=confidence,
            dominant_signals=dominant or [],
            decisions=decisions,
            compute_saved_flops=flops_saved,
            time_saved_s=time_saved,
            cost_saved_usd=cost_saved,
            quality_impact=redundancy_score * 0.01,  # estimated small impact
            tasks_analyzed=self._task_count,
            tasks_strained=self._strained_count,
            strain_ratio=(
                self._strained_count / max(self._task_count, 1)
            ),
            strainer_method=method,
        )

    def process_batch(
        self, tasks: List[ComputeTask],
    ) -> List[StrainResult]:
        """Process multiple compute tasks.

        Uses vectorised scoring where possible, falls back to
        per-task path for the deterministic redundancy checks.
        """
        if not tasks:
            return []

        n = len(tasks)

        # 1) Build feature matrix
        vecs = np.empty((n, tasks[0].to_vector().shape[0]), dtype=np.float64)
        for i, t in enumerate(tasks):
            vecs[i] = t.to_vector()

        # 2) Batch redundancy check (per-task — lightweight)
        batch_decisions: List[List[StrainDecision]] = [
            self.redundancy.check(t) for t in tasks
        ]
        skip_mask = np.array(
            [
                any(d.verdict == TaskVerdict.SKIP for d in ds)
                for ds in batch_decisions
            ],
            dtype=bool,
        )

        # 3) Convergence scoring
        conv_scores = np.empty(n, dtype=np.float64)
        conv_signals_list: list = [None] * n
        for i, t in enumerate(tasks):
            s, sig = self.convergence.update_and_score(t.gpu_id, vecs[i])
            conv_scores[i] = s
            conv_signals_list[i] = sig

        # 4) ML predictive scoring (vectorised)
        if self.predictor is not None:
            if hasattr(self.predictor, "batch_score"):
                ml_scores = np.asarray(
                    self.predictor.batch_score(vecs), dtype=np.float64
                )
            else:
                ml_scores = np.array(
                    [self.predictor.score(vecs[i]) for i in range(n)],
                    dtype=np.float64,
                )
        else:
            ml_scores = np.zeros(n, dtype=np.float64)

        # 5) Combine
        redundancy_scores = np.maximum(conv_scores, ml_scores)

        # Build results
        results: List[StrainResult] = []
        for i in range(n):
            self._task_count += 1
            if skip_mask[i]:
                result = self._make_result(
                    tasks[i], vecs[i], TaskVerdict.SKIP,
                    redundancy_score=1.0,
                    convergence_score=conv_scores[i],
                    decisions=batch_decisions[i],
                    method="batch_redundancy_skip",
                )
            else:
                score = redundancy_scores[i]
                # Discount if Stage 1 found nothing suspicious
                if not batch_decisions[i]:
                    score *= 0.3
                # Apply soft decision boost
                soft_boost = sum(
                    0.2 for d in batch_decisions[i]
                    if d.verdict in (TaskVerdict.APPROXIMATE, TaskVerdict.DEFER)
                )
                score = min(score + soft_boost, 1.0)

                if score >= 0.8:
                    verdict = TaskVerdict.SKIP
                elif score >= 0.6:
                    verdict = TaskVerdict.APPROXIMATE
                elif score >= self.strain_threshold:
                    verdict = TaskVerdict.DEFER
                else:
                    verdict = TaskVerdict.EXECUTE

                result = self._make_result(
                    tasks[i], vecs[i], verdict,
                    redundancy_score=float(score),
                    convergence_score=float(conv_scores[i]),
                    decisions=batch_decisions[i],
                    dominant=conv_signals_list[i] or [],
                    method="batch_pipeline",
                )
            results.append(result)

        return results

    @property
    def stats(self) -> Dict:
        return {
            "tasks_processed": self._task_count,
            "tasks_executed": self._executed_count,
            "tasks_strained": self._strained_count,
            "strain_ratio": self._strained_count / max(self._task_count, 1),
            "total_flops_saved": self._total_flops_saved,
            "total_time_saved_s": self._total_time_saved,
            "total_cost_saved_usd": self._total_cost_saved,
            "verdict_distribution": dict(self._verdict_counts),
        }
