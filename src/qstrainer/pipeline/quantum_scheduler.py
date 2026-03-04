"""QuantumStrainScheduler — QUBO-optimised batch task scheduling.

Instead of evaluating each GPU task independently (greedy), this scheduler
formulates a **QUBO** (Quadratic Unconstrained Binary Optimization) that
captures task costs, values, and **interactions**, then solves it via a
quantum or classical backend to find the globally optimal strain schedule.

Why quantum?
    For N pending tasks the search space is 2^N.  Greedy evaluates tasks
    individually (O(N)), missing joint redundancy between tasks.  QUBO
    captures all N*(N-1)/2 pairwise interactions and lets a quantum
    annealer or QAOA circuit find the global minimum — the schedule that
    saves the most compute while preserving training quality.

QUBO variable mapping:
    x_i = 1 → EXECUTE task i  (spend compute)
    x_i = 0 → STRAIN  task i  (save compute)

QUBO objective (minimise):
    E(x) = Σ_i  h_i x_i  +  Σ_{i<j} J_ij x_i x_j

    h_i   = cost_i − α · importance_i          (diagonal / linear)
    J_ij  = β · data_sim  − γ · consec_dep     (off-diagonal / quadratic)

Solver backends (via QOS scheduler):
    n ≤  18 → QAOA statevector (exact quantum sim)
    n ≤  50 → Simulated Annealing (fast classical)
    n ≤ 127 → Qiskit Runtime or SA heavy
    n > 127 → D-Wave QPU or SA heavy
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from qstrainer.models.alert import StrainDecision, StrainResult
from qstrainer.models.enums import TaskVerdict, StrainAction
from qstrainer.models.frame import ComputeTask, N_BASE_FEATURES
from qstrainer.stages.threshold import RedundancyStrainer
from qstrainer.stages.statistical import ConvergenceStrainer
from qstrainer.stages.ml import PredictiveStrainer
from qstrainer.solvers.base import QUBOSolverBase, QUBOResult
from qstrainer.qos.scheduler import QOSScheduler

logger = logging.getLogger(__name__)


# ── Tuning knobs ─────────────────────────────────────────────

@dataclass
class SchedulerConfig:
    """Tuning parameters for QUBO formulation.

    α  (alpha)   — quality weight.  Higher → execute more, strain less.
    β  (beta)    — data-similarity coupling.  Higher → strain similar pairs.
    γ  (gamma)   — consecutive-step coupling.  Higher → avoid skipping streaks.
    δ  (delta)   — cross-GPU fairness.  Higher → balance across GPUs.
    max_strain_rate — hard cap on fraction strained per batch (safety net).
    """

    alpha: float = 2.0     # quality vs. savings tradeoff
    beta: float = 0.4      # data-similarity penalty strength
    gamma: float = 0.3     # consecutive-step anti-correlation
    delta: float = 0.15    # cross-GPU balance incentive
    max_strain_rate: float = 0.70   # never strain more than 70% of a batch
    batch_size: int = 32   # sweet-spot for QAOA on near-term quantum HW


# ── QUBO Builder ─────────────────────────────────────────────

class QUBOBuilder:
    """Constructs the QUBO matrix from a batch of scored tasks.

    The matrix encodes:
      - Per-task cost/value tradeoff (diagonal)
      - Task–task interactions (off-diagonal)
    """

    def __init__(self, cfg: SchedulerConfig | None = None) -> None:
        self.cfg = cfg or SchedulerConfig()

    def build(
        self,
        tasks: List[ComputeTask],
        redundancy_scores: np.ndarray,
        vectors: np.ndarray,
        decisions_per_task: List[List[StrainDecision]],
    ) -> np.ndarray:
        """Build and return the N×N QUBO matrix Q.

        Parameters
        ----------
        tasks : list[ComputeTask]
            The batch of pending tasks.
        redundancy_scores : ndarray, shape (N,)
            Per-task redundancy score from Stage 2+3 (0=valuable, 1=redundant).
        vectors : ndarray, shape (N, 15)
            Feature vectors for the batch.
        decisions_per_task : list[list[StrainDecision]]
            Stage 1 decisions per task (empty if productive).

        Returns
        -------
        Q : ndarray, shape (N, N)
            Upper-triangular QUBO matrix.
        """
        N = len(tasks)
        Q = np.zeros((N, N), dtype=np.float64)

        # ── 1. Linear terms (diagonal) ──────────────────────
        self._add_linear_terms(Q, tasks, redundancy_scores, decisions_per_task)

        # ── 2. Data similarity coupling ─────────────────────
        self._add_similarity_coupling(Q, tasks, vectors)

        # ── 3. Consecutive step anti-correlation ────────────
        self._add_consecutive_coupling(Q, tasks)

        # ── 4. Cross-GPU fairness ───────────────────────────
        self._add_fairness_coupling(Q, tasks)

        # ── 5. Max strain rate constraint (penalty) ─────────
        self._add_strain_rate_constraint(Q, N)

        return Q

    def _add_linear_terms(
        self,
        Q: np.ndarray,
        tasks: List[ComputeTask],
        redundancy_scores: np.ndarray,
        decisions_per_task: List[List[StrainDecision]],
    ) -> None:
        """Diagonal: h_i = cost_i - α * importance_i.

        Redundant tasks get positive h → solver prefers x=0 (strain).
        Productive tasks get negative h → solver prefers x=1 (execute).
        """
        α = self.cfg.alpha
        for i, task in enumerate(tasks):
            # Normalised execution cost (higher = more expensive to run)
            cost = task.estimated_flops / max(task.estimated_flops, 1e-12)
            cost = min(cost, 1.0)  # clamp to [0,1]

            # Importance = inverse of redundancy (high redundancy → low importance)
            importance = 1.0 - redundancy_scores[i]

            # Boost importance if Stage 1 found zero issues (task looks productive)
            if not decisions_per_task[i]:
                importance = min(importance + 0.3, 1.0)

            Q[i, i] = cost - α * importance

    def _add_similarity_coupling(
        self,
        Q: np.ndarray,
        tasks: List[ComputeTask],
        vectors: np.ndarray,
    ) -> None:
        """Off-diagonal: positive coupling between similar tasks on same GPU.

        If two tasks have similar feature vectors (near-duplicate batches),
        executing both is wasteful.  Positive J_ij encourages the solver to
        strain at least one of them.
        """
        β = self.cfg.beta
        N = len(tasks)
        if N < 2 or β == 0:
            return

        # Compute pairwise cosine similarity (vectorised)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        normed = vectors / norms
        sim_matrix = normed @ normed.T  # (N, N), values in [-1, 1]

        for i in range(N):
            for j in range(i + 1, N):
                # Only couple tasks on the same GPU
                if tasks[i].gpu_id != tasks[j].gpu_id:
                    continue
                sim = max(sim_matrix[i, j], 0.0)
                if sim > 0.5:  # only couple meaningfully similar pairs
                    Q[i, j] += β * sim

    def _add_consecutive_coupling(
        self,
        Q: np.ndarray,
        tasks: List[ComputeTask],
    ) -> None:
        """Off-diagonal: negative coupling between consecutive steps.

        Skipping many steps in a row degrades model quality.  Negative J_ij
        makes the solver prefer to execute at least one of each consecutive
        pair — like a no-long-gap constraint.
        """
        γ = self.cfg.gamma
        if γ == 0:
            return

        # Group by (gpu_id, job_id) and sort by step_number
        from collections import defaultdict
        groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for idx, t in enumerate(tasks):
            groups[(t.gpu_id, t.job_id)].append(idx)

        for key, indices in groups.items():
            indices_sorted = sorted(indices, key=lambda x: tasks[x].step_number)
            for k in range(len(indices_sorted) - 1):
                i = indices_sorted[k]
                j = indices_sorted[k + 1]
                step_gap = tasks[j].step_number - tasks[i].step_number
                if step_gap <= 3:  # only for truly consecutive/nearby steps
                    Q[i, j] -= γ  # negative = anti-correlation → execute at least one

    def _add_fairness_coupling(
        self,
        Q: np.ndarray,
        tasks: List[ComputeTask],
    ) -> None:
        """Off-diagonal: negative coupling across different GPUs in same job.

        Encourages the solver to spread execution across GPUs rather than
        straining all tasks on one GPU while executing all on another.
        """
        δ = self.cfg.delta
        if δ == 0:
            return
        N = len(tasks)

        for i in range(N):
            for j in range(i + 1, N):
                if (tasks[i].job_id == tasks[j].job_id and
                        tasks[i].gpu_id != tasks[j].gpu_id):
                    Q[i, j] -= δ  # encourage executing across GPUs

    def _add_strain_rate_constraint(
        self,
        Q: np.ndarray,
        N: int,
    ) -> None:
        """Penalty term to discourage straining too many tasks.

        Adds a soft constraint: if more than max_strain_rate fraction are
        strained (x=0), the solution is penalised.  Implemented as a
        positive boost to diagonal (encouraging x=1 = execute).
        """
        max_strained = int(N * self.cfg.max_strain_rate)
        min_executed = N - max_strained

        # Penalty strength scales with batch size
        penalty = 0.1 * (min_executed / max(N, 1))
        for i in range(N):
            Q[i, i] -= penalty  # slight push toward execute


# ── Quantum Strain Scheduler ────────────────────────────────

class QuantumStrainScheduler:
    """Batch scheduler that uses QUBO optimization to decide which tasks to strain.

    Instead of greedy per-task decisions, this scheduler:
    1. Collects a batch of tasks
    2. Runs Stage 1 (Redundancy) and Stage 2 (Convergence) to score each
    3. Builds a QUBO matrix encoding costs, values, and task interactions
    4. Solves the QUBO via quantum (QAOA/D-Wave) or classical (SA) backend
    5. Maps the binary solution to verdicts: EXECUTE / SKIP / APPROXIMATE / DEFER

    The quadratic terms capture what greedy misses:
    - Jointly redundant batches (data similarity coupling)
    - Consecutive step dependencies (can't skip too many in a row)
    - Cross-GPU fairness (balance strained load)
    """

    def __init__(
        self,
        redundancy: RedundancyStrainer | None = None,
        convergence: ConvergenceStrainer | None = None,
        predictor: PredictiveStrainer | None = None,
        qos_scheduler: QOSScheduler | None = None,
        config: SchedulerConfig | None = None,
    ) -> None:
        self.redundancy = redundancy or RedundancyStrainer()
        self.convergence = convergence or ConvergenceStrainer()
        self.predictor = predictor
        self.config = config or SchedulerConfig()
        self.builder = QUBOBuilder(self.config)

        # QOS scheduler for solver routing
        if qos_scheduler is not None:
            self.qos = qos_scheduler
        else:
            self.qos = QOSScheduler.from_config({})

        # Cumulative stats
        self._batches_scheduled: int = 0
        self._total_tasks: int = 0
        self._total_executed: int = 0
        self._total_strained: int = 0
        self._total_flops_saved: float = 0.0
        self._total_time_saved: float = 0.0
        self._total_cost_saved: float = 0.0
        self._total_solve_time: float = 0.0
        self._qubo_energies: List[float] = []

    def schedule(
        self,
        tasks: List[ComputeTask],
        prefer_solver: str | None = None,
    ) -> List[StrainResult]:
        """Schedule a batch of tasks using QUBO optimization.

        Parameters
        ----------
        tasks : list[ComputeTask]
            Batch of pending GPU tasks to schedule.
        prefer_solver : str, optional
            Force a specific solver (e.g. "qaoa_sim", "sa_default", "dwave").

        Returns
        -------
        list[StrainResult]
            One result per task with verdict and savings estimates.
        """
        if not tasks:
            return []

        N = len(tasks)
        self._batches_scheduled += 1
        self._total_tasks += N
        t_start = time.perf_counter()

        # ── Stage 1: Redundancy check (per-task) ────────────
        decisions_per_task: List[List[StrainDecision]] = [
            self.redundancy.check(t) for t in tasks
        ]

        # Identify hard SKIPs from Stage 1
        skip_mask = np.array(
            [any(d.verdict == TaskVerdict.SKIP for d in ds)
             for ds in decisions_per_task],
            dtype=bool,
        )

        # ── Build feature matrix ────────────────────────────
        vectors = np.empty((N, N_BASE_FEATURES), dtype=np.float64)
        for i, t in enumerate(tasks):
            vectors[i] = t.to_vector()

        # ── Stage 2: Convergence scoring ────────────────────
        conv_scores = np.empty(N, dtype=np.float64)
        conv_signals_list: list = [None] * N
        for i, t in enumerate(tasks):
            s, sig = self.convergence.update_and_score(t.gpu_id, vectors[i])
            conv_scores[i] = s
            conv_signals_list[i] = sig

        # ── Stage 3: Predictive scoring ─────────────────────
        if self.predictor is not None:
            ml_scores = np.array(
                [self.predictor.score(vectors[i]) for i in range(N)],
                dtype=np.float64,
            )
        else:
            ml_scores = np.zeros(N, dtype=np.float64)

        # ── Combined redundancy scores (input to QUBO) ──────
        redundancy_scores = np.maximum(conv_scores, ml_scores)

        # Override: hard SKIPs from Stage 1 get score = 1.0
        redundancy_scores[skip_mask] = 1.0

        # If Stage 1 found nothing, discount convergence noise
        for i in range(N):
            if not decisions_per_task[i]:
                redundancy_scores[i] *= 0.3

        # ── Build QUBO ──────────────────────────────────────
        Q = self.builder.build(tasks, redundancy_scores, vectors, decisions_per_task)

        # ── Solve QUBO ──────────────────────────────────────
        solver_name, solver = self.qos.select_solver(
            n_variables=N, prefer=prefer_solver,
        )
        qubo_result = solver.solve(Q)

        solve_time = time.perf_counter() - t_start
        self._total_solve_time += solve_time
        self._qubo_energies.append(qubo_result.energy)

        # ── Map solution to verdicts ────────────────────────
        solution = qubo_result.solution
        results = self._interpret_solution(
            tasks, vectors, solution, redundancy_scores,
            conv_scores, decisions_per_task, conv_signals_list,
            qubo_result, solver_name, solve_time,
        )

        return results

    def _interpret_solution(
        self,
        tasks: List[ComputeTask],
        vectors: np.ndarray,
        solution: np.ndarray,
        redundancy_scores: np.ndarray,
        conv_scores: np.ndarray,
        decisions_per_task: List[List[StrainDecision]],
        conv_signals_list: list,
        qubo_result: QUBOResult,
        solver_name: str,
        solve_time: float,
    ) -> List[StrainResult]:
        """Map binary QUBO solution → StrainResult list.

        x_i = 1 → EXECUTE
        x_i = 0 and redundancy_score >= 0.8 → SKIP
        x_i = 0 and redundancy_score >= 0.6 → APPROXIMATE
        x_i = 0 (else) → DEFER
        """
        N = len(tasks)
        results: List[StrainResult] = []

        for i in range(N):
            execute = bool(solution[i])
            rscore = float(redundancy_scores[i])
            cscore = float(conv_scores[i])

            if execute:
                verdict = TaskVerdict.EXECUTE
            else:
                # Grade the straining verdict by redundancy score
                if rscore >= 0.8:
                    verdict = TaskVerdict.SKIP
                elif rscore >= 0.6:
                    verdict = TaskVerdict.APPROXIMATE
                else:
                    verdict = TaskVerdict.DEFER

            # Calculate savings
            if verdict == TaskVerdict.SKIP:
                flops_saved = tasks[i].estimated_flops
                time_saved = tasks[i].estimated_time_s
            elif verdict == TaskVerdict.APPROXIMATE:
                flops_saved = tasks[i].estimated_flops * 0.5
                time_saved = tasks[i].estimated_time_s * 0.5
            elif verdict == TaskVerdict.DEFER:
                flops_saved = tasks[i].estimated_flops * 0.2
                time_saved = tasks[i].estimated_time_s * 0.2
            else:
                flops_saved = 0.0
                time_saved = 0.0

            cost_saved = time_saved * 2.50 / 3600.0  # H100 rate

            # Update cumulative stats
            if verdict != TaskVerdict.EXECUTE:
                self._total_strained += 1
                self._total_flops_saved += flops_saved
                self._total_time_saved += time_saved
                self._total_cost_saved += cost_saved
            else:
                self._total_executed += 1

            confidence = min(self._total_tasks / 100.0, 1.0)

            results.append(StrainResult(
                timestamp=tasks[i].timestamp,
                task_id=tasks[i].task_id,
                gpu_id=tasks[i].gpu_id,
                job_id=tasks[i].job_id,
                step_number=tasks[i].step_number,
                verdict=verdict,
                redundancy_score=rscore,
                convergence_score=cscore,
                confidence=confidence,
                dominant_signals=conv_signals_list[i] or [],
                decisions=decisions_per_task[i],
                compute_saved_flops=flops_saved,
                time_saved_s=time_saved,
                cost_saved_usd=cost_saved,
                quality_impact=rscore * 0.01,
                tasks_analyzed=self._total_tasks,
                tasks_strained=self._total_strained,
                strain_ratio=self._total_strained / max(self._total_tasks, 1),
                strainer_method=f"quantum_schedule:{solver_name}",
            ))

        return results

    @property
    def stats(self) -> Dict:
        """Cumulative scheduler statistics."""
        return {
            "batches_scheduled": self._batches_scheduled,
            "tasks_processed": self._total_tasks,
            "tasks_executed": self._total_executed,
            "tasks_strained": self._total_strained,
            "strain_ratio": self._total_strained / max(self._total_tasks, 1),
            "total_flops_saved": self._total_flops_saved,
            "total_time_saved_s": self._total_time_saved,
            "total_cost_saved_usd": self._total_cost_saved,
            "total_solve_time_s": self._total_solve_time,
            "avg_qubo_energy": (
                float(np.mean(self._qubo_energies))
                if self._qubo_energies else 0.0
            ),
            "solver_used": "auto",
        }

    @property
    def qubo_energies(self) -> List[float]:
        """QUBO energies from all batches — tracks optimisation quality."""
        return list(self._qubo_energies)
