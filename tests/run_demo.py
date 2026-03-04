"""Q-Strainer Datacenter-Scale Demo — no GPU required.

Simulates a **64-GPU heterogeneous cluster** (H100 / A100 / V100) running
mixed workloads (training, fine-tuning, inference, hyper-parameter search)
across 8 nodes.  Q-Strainer evaluates thousands of compute tasks and shows
how the three-stage pipeline + QUBO quantum scheduler + QAOA conflict-graph
purification saves GPU-hours, FLOPs, and cloud cost at datacenter scale.

Run with:  py tests/run_demo.py       (Windows)
           python tests/run_demo.py   (Linux/macOS)

Results are saved to runs/demo_<timestamp>.json after each run,
then a localhost dashboard opens at http://localhost:8050.
"""

import json
import math
import os
import sys
import time
import random
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict

from qstrainer.models import ComputeTask, WorkloadBuffer
from qstrainer.models.enums import TaskVerdict, ComputePhase, JobType, GPUType
from qstrainer.features import DerivedFeatureExtractor
from qstrainer.stages.threshold import RedundancyStrainer
from qstrainer.stages.statistical import ConvergenceStrainer
from qstrainer.pipeline import QStrainer, QuantumStrainScheduler, SchedulerConfig
from qstrainer.quantum import QuantumAdvantagePipeline, PipelineConfig

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"

# ═══════════════════════════════════════════════════════════════
# DATACENTER CLUSTER DEFINITION — 64 GPUs across 8 nodes
# ═══════════════════════════════════════════════════════════════

# Realistic GPU specs for cost calculation
GPU_SPECS = {
    "H100": {"vram_gb": 80,  "peak_tflops_fp16": 989.0,  "cost_per_hr": 2.50, "tdp_w": 700},
    "A100": {"vram_gb": 80,  "peak_tflops_fp16": 312.0,  "cost_per_hr": 1.60, "tdp_w": 400},
    "V100": {"vram_gb": 32,  "peak_tflops_fp16": 125.0,  "cost_per_hr": 1.10, "tdp_w": 300},
}

CLUSTER_LAYOUT = [
    # (node_id, gpu_model, gpu_count)
    ("node-0", "H100", 8),   # 8x H100 — flagship training
    ("node-1", "A100", 8),   # 8x A100 — mixed training / fine-tuning
    ("node-2", "A100", 8),   # 8x A100 — data-parallel training
    ("node-3", "A100", 8),   # 8x A100 — HPC / genomics
    ("node-4", "V100", 8),   # 8x V100 — inference cluster
    ("node-5", "V100", 8),   # 8x V100 — inference cluster
    ("node-6", "V100", 8),   # 8x V100 — inference + hyper-param search
    ("node-7", "V100", 8),   # 8x V100 — inference + preprocessing
]

def build_gpu_roster() -> List[Dict]:
    """Build the full 64-GPU roster with node/rack topology."""
    gpus = []
    for node_id, model, count in CLUSTER_LAYOUT:
        for idx in range(count):
            gpus.append({
                "gpu_id": f"{node_id}-GPU-{idx}",
                "node_id": node_id,
                "model": model,
                "vram_gb": GPU_SPECS[model]["vram_gb"],
                "peak_tflops": GPU_SPECS[model]["peak_tflops_fp16"],
                "cost_per_hr": GPU_SPECS[model]["cost_per_hr"],
            })
    return gpus

GPU_ROSTER = build_gpu_roster()
ALL_GPU_IDS = [g["gpu_id"] for g in GPU_ROSTER]


# ═══════════════════════════════════════════════════════════════
# WORKLOAD PROFILES — inspired by Alibaba PAI / Helios / Philly
# ═══════════════════════════════════════════════════════════════

WORKLOAD_PROFILES = {
    "llm_pretrain": {
        "name": "LLM Pre-training (GPT-3 scale)",
        "job_type": JobType.TRAINING,
        "phase": ComputePhase.BACKWARD_PASS,
        "gpus": ["H100"],          # only runs on H100
        "batch_size": 2048,
        "flops_per_step": 180e12,  # 180 TFLOP per step (large model)
        "time_per_step_s": 12.0,   # ~12s per training step
        "memory_gb": 72.0,         # near-full H100 VRAM
        "initial_loss": 8.5,
        "convergence_rate": 0.002, # slow convergence
        "model_name": "gpt3-175b",
        "redundancy_onset": 0.65,  # becomes redundant after 65% training
    },
    "vision_train": {
        "name": "Vision Model Training (ResNet/ViT)",
        "job_type": JobType.TRAINING,
        "phase": ComputePhase.FORWARD_PASS,
        "gpus": ["H100", "A100"],
        "batch_size": 512,
        "flops_per_step": 45e12,   # 45 TFLOP per step
        "time_per_step_s": 4.5,
        "memory_gb": 38.0,
        "initial_loss": 4.2,
        "convergence_rate": 0.005,
        "model_name": "vit-large",
        "redundancy_onset": 0.55,
    },
    "finetune_bert": {
        "name": "BERT Fine-tuning",
        "job_type": JobType.FINE_TUNING,
        "phase": ComputePhase.BACKWARD_PASS,
        "gpus": ["A100", "V100"],
        "batch_size": 128,
        "flops_per_step": 8e12,   # 8 TFLOP per step
        "time_per_step_s": 1.8,
        "memory_gb": 22.0,
        "initial_loss": 2.8,
        "convergence_rate": 0.012,  # fast convergence -> lots of redundancy
        "model_name": "bert-large",
        "redundancy_onset": 0.40,
    },
    "inference_llm": {
        "name": "LLM Inference Serving",
        "job_type": JobType.INFERENCE,
        "phase": ComputePhase.INFERENCE,
        "gpus": ["A100", "V100"],
        "batch_size": 64,
        "flops_per_step": 2.5e12,
        "time_per_step_s": 0.35,
        "memory_gb": 28.0,
        "initial_loss": 0.5,       # stable
        "convergence_rate": 0.0,   # inference doesn't converge
        "model_name": "llama2-70b-infer",
        "redundancy_onset": 0.0,   # inference requests are never redundant by convergence
    },
    "hyperparam_search": {
        "name": "Hyperparameter Search (Optuna)",
        "job_type": JobType.HYPERPARAMETER_SEARCH,
        "phase": ComputePhase.FORWARD_PASS,
        "gpus": ["V100"],
        "batch_size": 64,
        "flops_per_step": 3e12,
        "time_per_step_s": 1.2,
        "memory_gb": 18.0,
        "initial_loss": 5.0,
        "convergence_rate": 0.008,
        "model_name": "resnet50-hpsearch",
        "redundancy_onset": 0.50,
    },
    "data_preprocess": {
        "name": "Data Pipeline / Preprocessing",
        "job_type": JobType.DATA_PREPROCESSING,
        "phase": ComputePhase.DATA_LOADING,
        "gpus": ["V100"],
        "batch_size": 256,
        "flops_per_step": 1e12,
        "time_per_step_s": 0.6,
        "memory_gb": 10.0,
        "initial_loss": 0.0,
        "convergence_rate": 0.0,
        "model_name": "data-pipeline",
        "redundancy_onset": 0.0,
    },
    "diffusion_train": {
        "name": "Diffusion Model Training (SDXL)",
        "job_type": JobType.TRAINING,
        "phase": ComputePhase.BACKWARD_PASS,
        "gpus": ["H100", "A100"],
        "batch_size": 256,
        "flops_per_step": 95e12,   # large diffusion model
        "time_per_step_s": 8.0,
        "memory_gb": 65.0,
        "initial_loss": 6.0,
        "convergence_rate": 0.003,
        "model_name": "sdxl-v2",
        "redundancy_onset": 0.60,
    },
    "rl_training": {
        "name": "Reinforcement Learning (PPO)",
        "job_type": JobType.TRAINING,
        "phase": ComputePhase.FORWARD_PASS,
        "gpus": ["A100", "V100"],
        "batch_size": 128,
        "flops_per_step": 12e12,
        "time_per_step_s": 2.0,
        "memory_gb": 20.0,
        "initial_loss": 15.0,       # RL reward is noisy, starts high
        "convergence_rate": 0.004,
        "model_name": "ppo-robotics",
        "redundancy_onset": 0.55,
    },
}


# ═══════════════════════════════════════════════════════════════
# JOB GENERATOR — multi-domain, calibrated to real traces
# ═══════════════════════════════════════════════════════════════

class DatacenterJobGenerator:
    """Generates realistic GPU compute tasks across the 64-GPU cluster.

    Job mix calibrated to Alibaba PAI / Microsoft Philly workload traces:
    - 25% training (LLM + vision + diffusion + RL)
    - 15% fine-tuning
    - 35% inference serving
    - 15% hyperparameter search
    - 10% data preprocessing
    """

    PROFILE_WEIGHTS = {
        "llm_pretrain": 0.10,
        "vision_train": 0.08,
        "diffusion_train": 0.04,
        "rl_training": 0.03,
        "finetune_bert": 0.15,
        "inference_llm": 0.35,
        "hyperparam_search": 0.15,
        "data_preprocess": 0.10,
    }

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._job_counter = 0
        self._step_tracker: Dict[str, int] = defaultdict(int)

    def _pick_profile(self) -> Tuple[str, dict]:
        """Weighted random profile selection."""
        names = list(self.PROFILE_WEIGHTS.keys())
        weights = list(self.PROFILE_WEIGHTS.values())
        choice = self.rng.choices(names, weights=weights, k=1)[0]
        return choice, WORKLOAD_PROFILES[choice]

    def _pick_gpu(self, allowed_models: List[str]) -> Dict:
        """Pick a GPU from the roster that matches the workload profile."""
        candidates = [g for g in GPU_ROSTER if g["model"] in allowed_models]
        return self.rng.choice(candidates)

    def generate_job_batch(self, batch_size: int, base_step: int = 0) -> List[ComputeTask]:
        """Generate a batch of realistic compute tasks."""
        tasks = []
        for i in range(batch_size):
            profile_name, profile = self._pick_profile()
            gpu = self._pick_gpu(profile["gpus"])

            job_id = f"{profile_name}-{self._job_counter:04d}"
            self._job_counter += 1
            step = base_step + i

            # Progress through training (0.0 -> 1.0)
            total_steps = 1000
            progress = min(step / total_steps, 1.0)

            # Loss trajectory
            onset = profile["redundancy_onset"]
            conv_rate = profile["convergence_rate"]
            if conv_rate > 0:
                loss = max(
                    profile["initial_loss"] * math.exp(-conv_rate * step)
                    + self.rng.gauss(0, 0.05),
                    0.01,
                )
                loss_delta = -conv_rate * loss + self.rng.gauss(0, 0.003)

                # After convergence onset, gradients become tiny
                if progress > onset:
                    frac_past = (progress - onset) / max(1.0 - onset, 0.01)
                    grad_norm = max(1e-7 * (1 - frac_past) + 1e-10, 1e-10)
                    grad_var = 1e-10
                    convergence_score = min(0.5 + 0.5 * frac_past, 0.99)
                    param_update_mag = 1e-9 * (1 - frac_past)
                    data_sim = 0.7 + 0.3 * frac_past
                else:
                    grad_norm = 0.5 + self.rng.gauss(0, 0.15)
                    grad_var = 0.05 + self.rng.gauss(0, 0.01)
                    convergence_score = progress * 0.4
                    param_update_mag = 0.01 + self.rng.gauss(0, 0.002)
                    data_sim = 0.3 + self.rng.gauss(0, 0.1)
            else:
                # Inference / preprocessing — no convergence trajectory
                loss = profile["initial_loss"] + self.rng.gauss(0, 0.01)
                loss_delta = self.rng.gauss(0, 1e-5)
                grad_norm = 0.0
                grad_var = 0.0
                convergence_score = 0.0
                param_update_mag = 0.0
                data_sim = self.rng.uniform(0.1, 0.4)

            # Add noise to FLOP and time estimates
            flops = profile["flops_per_step"] * (1.0 + self.rng.gauss(0, 0.05))
            time_s = profile["time_per_step_s"] * (1.0 + self.rng.gauss(0, 0.05))

            task = ComputeTask(
                timestamp=time.time() + i * 0.001,
                task_id=f"task-{profile_name[:4]}-{step:06d}",
                gpu_id=gpu["gpu_id"],
                job_id=job_id,
                step_number=step,
                loss=loss,
                loss_delta=loss_delta,
                gradient_norm=max(grad_norm, 0.0),
                gradient_variance=max(grad_var, 0.0),
                learning_rate=1e-4 if profile["job_type"] == JobType.TRAINING else 5e-5,
                batch_size=profile["batch_size"],
                epoch=step // 100,
                epoch_progress=(step % 100) / 100.0,
                estimated_flops=flops,
                estimated_time_s=time_s,
                memory_footprint_gb=profile["memory_gb"],
                compute_phase=profile["phase"],
                job_type=profile["job_type"],
                convergence_score=convergence_score,
                param_update_magnitude=max(param_update_mag, 0.0),
                data_similarity=max(min(data_sim, 1.0), 0.0),
                flop_utilization=0.65 + self.rng.gauss(0, 0.1),
                throughput_samples_per_sec=profile["batch_size"] / max(time_s, 0.01),
                model_name=profile["model_name"],
                node_id=gpu["node_id"],
            )
            tasks.append(task)

        return tasks

    def generate_redundant_burst(self, count: int, profile_name: str = "finetune_bert") -> List[ComputeTask]:
        """Generate a burst of clearly redundant tasks (converged workloads)."""
        profile = WORKLOAD_PROFILES[profile_name]
        tasks = []
        for i in range(count):
            gpu = self._pick_gpu(profile["gpus"])
            job_id = f"{profile_name}-converged-{i:04d}"
            step = 950 + i  # near end of training

            task = ComputeTask(
                timestamp=time.time() + i * 0.001,
                task_id=f"task-conv-{step:06d}",
                gpu_id=gpu["gpu_id"],
                job_id=job_id,
                step_number=step,
                loss=0.08 + self.rng.gauss(0, 0.001),
                loss_delta=self.rng.gauss(0, 1e-7),
                gradient_norm=1e-8 + abs(self.rng.gauss(0, 1e-9)),
                gradient_variance=1e-10,
                learning_rate=1e-6,
                batch_size=profile["batch_size"],
                epoch=step // 100,
                epoch_progress=(step % 100) / 100.0,
                estimated_flops=profile["flops_per_step"],
                estimated_time_s=profile["time_per_step_s"],
                memory_footprint_gb=profile["memory_gb"],
                compute_phase=ComputePhase.BACKWARD_PASS,
                job_type=profile["job_type"],
                convergence_score=0.98,
                param_update_magnitude=1e-10,
                data_similarity=0.99,
                flop_utilization=0.70,
                throughput_samples_per_sec=profile["batch_size"] / profile["time_per_step_s"],
                model_name=profile["model_name"],
                node_id=gpu["node_id"],
            )
            tasks.append(task)
        return tasks


# ═══════════════════════════════════════════════════════════════
# COST MODEL — realistic cloud pricing
# ═══════════════════════════════════════════════════════════════

def estimate_cost_saved(task: ComputeTask) -> float:
    """Estimate dollar savings from straining a task, based on GPU model."""
    gpu_info = next((g for g in GPU_ROSTER if g["gpu_id"] == task.gpu_id), None)
    if gpu_info is None:
        rate = 2.50  # default H100 rate
    else:
        rate = gpu_info["cost_per_hr"]
    return task.estimated_time_s * rate / 3600.0


def format_flops(flops: float) -> str:
    """Human-readable FLOPs."""
    if flops >= 1e18:      return f"{flops / 1e18:.2f} EFLOP"
    elif flops >= 1e15:    return f"{flops / 1e15:.2f} PFLOP"
    elif flops >= 1e12:    return f"{flops / 1e12:.1f} TFLOP"
    elif flops >= 1e9:     return f"{flops / 1e9:.1f} GFLOP"
    else:                  return f"{flops:.2e} FLOP"


def format_time(seconds: float) -> str:
    """Human-readable time."""
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    elif seconds >= 60:
        return f"{seconds / 60:.1f}min"
    else:
        return f"{seconds:.1f}s"


def format_cost(usd: float) -> str:
    """Human-readable cost."""
    if usd >= 1000:
        return f"${usd:,.0f}"
    elif usd >= 1:
        return f"${usd:.2f}"
    else:
        return f"${usd:.4f}"


# ═══════════════════════════════════════════════════════════════
# MAIN DEMO
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  Q-Strainer — Datacenter-Scale GPU Compute Workload Strainer")
    print("  64 GPUs | 8 Nodes | H100/A100/V100 Heterogeneous Cluster")
    print("=" * 70)

    # --- Cluster summary ---
    model_counts = defaultdict(int)
    total_tflops = 0.0
    total_vram = 0
    hourly_cost = 0.0
    for g in GPU_ROSTER:
        model_counts[g["model"]] += 1
        total_tflops += g["peak_tflops"]
        total_vram += g["vram_gb"]
        hourly_cost += g["cost_per_hr"]

    print(f"\n  Cluster topology:")
    for node_id, model, count in CLUSTER_LAYOUT:
        spec = GPU_SPECS[model]
        print(f"    {node_id}: {count}x {model}  "
              f"({spec['vram_gb']}GB VRAM, {spec['peak_tflops_fp16']} TFLOPS, "
              f"${spec['cost_per_hr']:.2f}/hr)")
    print(f"    {'─' * 55}")
    print(f"    Total: {len(GPU_ROSTER)} GPUs | "
          f"{total_tflops:,.0f} TFLOPS aggregate | "
          f"{total_vram:,} GB VRAM | "
          f"${hourly_cost:.0f}/hr cluster cost")

    gen = DatacenterJobGenerator(seed=42)
    pipeline = QStrainer()
    buffer = WorkloadBuffer(max_tasks_per_gpu=1000)
    extractor = DerivedFeatureExtractor()

    # ══════════════════════════════════════════════════════════
    # Phase 1: Warm-up — 1000 productive tasks across all 64 GPUs
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("[Phase 1] Cluster warm-up — 1,000 productive tasks across 64 GPUs")
    print(f"{'─' * 70}")

    warmup_tasks = gen.generate_job_batch(1000, base_step=0)
    executed_warmup = 0
    strained_warmup = 0
    warmup_flops_saved = 0.0
    warmup_time_saved = 0.0
    warmup_cost_saved = 0.0
    t_start = time.perf_counter()

    for task in warmup_tasks:
        buffer.push(task)
        result = pipeline.process_task(task)
        if result.verdict == TaskVerdict.EXECUTE:
            executed_warmup += 1
        else:
            strained_warmup += 1
            warmup_flops_saved += result.compute_saved_flops
            warmup_time_saved += result.time_saved_s
            warmup_cost_saved += estimate_cost_saved(task)

    warmup_wall = time.perf_counter() - t_start

    # Profile distribution
    warmup_profiles = defaultdict(int)
    warmup_gpus_used = set()
    warmup_nodes_used = set()
    for t in warmup_tasks:
        warmup_profiles[t.model_name] += 1
        warmup_gpus_used.add(t.gpu_id)
        warmup_nodes_used.add(t.node_id)

    print(f"  Tasks processed: 1,000 across {len(warmup_gpus_used)} GPUs / {len(warmup_nodes_used)} nodes")
    print(f"  Workload mix:")
    for model, count in sorted(warmup_profiles.items(), key=lambda x: -x[1]):
        profile_info = next((v for v in WORKLOAD_PROFILES.values() if v["model_name"] == model), None)
        name = profile_info["name"] if profile_info else model
        print(f"    {name:40s}  {count:>4} tasks")
    print(f"  Executed:  {executed_warmup:,}")
    print(f"  Strained:  {strained_warmup:,} ({strained_warmup/10:.1f}%)")
    print(f"  FLOPs saved: {format_flops(warmup_flops_saved)}")
    print(f"  Time saved:  {format_time(warmup_time_saved)}")
    print(f"  Cost saved:  {format_cost(warmup_cost_saved)}")
    print(f"  Throughput:  {1000 / warmup_wall:,.0f} tasks/sec  ({warmup_wall:.3f}s wall)")

    # ══════════════════════════════════════════════════════════
    # Phase 2: Feature extraction at scale
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("[Phase 2] Feature extraction — multi-GPU feature expansion")
    print(f"{'─' * 70}")

    sample = warmup_tasks[0]
    raw = sample.to_vector()
    derived = extractor.extract(sample.gpu_id, raw)

    print(f"  Raw features:     {len(raw)} per task (loss, gradient, compute cost ...)")
    print(f"  Derived features: {len(derived)} per task (cross-correlations, z-scores ...)")
    print(f"  Expansion ratio:  {len(raw)} -> {len(derived)} ({len(derived)/len(raw):.1f}x)")
    print(f"  Total feature matrix: {len(warmup_tasks):,} tasks x {len(derived)} features "
          f"= {len(warmup_tasks) * len(derived):,} values")

    # ══════════════════════════════════════════════════════════
    # Phase 3: Inject 200 redundant tasks (converged fine-tuning jobs)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("[Phase 3] Redundancy injection — 200 converged fine-tuning tasks")
    print(f"{'─' * 70}")

    redundant_tasks = gen.generate_redundant_burst(200, profile_name="finetune_bert")
    redundant_decisions = []
    redundant_flops = 0.0
    redundant_time = 0.0
    redundant_cost = 0.0

    for task in redundant_tasks:
        buffer.push(task)
        result = pipeline.process_task(task)
        if result.verdict != TaskVerdict.EXECUTE:
            redundant_decisions.append(result)
            redundant_flops += result.compute_saved_flops
            redundant_time += result.time_saved_s
            redundant_cost += estimate_cost_saved(task)

    print(f"  Injected:  200 converged BERT fine-tuning tasks")
    print(f"  Detected:  {len(redundant_decisions)} as redundant "
          f"({len(redundant_decisions)/2:.0f}% detection rate)")
    print(f"  FLOPs saved: {format_flops(redundant_flops)}")
    print(f"  Time saved:  {format_time(redundant_time)}")
    print(f"  Cost saved:  {format_cost(redundant_cost)}")

    if redundant_decisions:
        print(f"\n  Sample decisions (first 10):")
        for i, r in enumerate(redundant_decisions[:10]):
            print(f"    [{i+1:2d}] {r.verdict.name:12s} | "
                  f"redundancy={r.redundancy_score:.3f} | "
                  f"saved {r.time_saved_s:.2f}s, {format_flops(r.compute_saved_flops)}")
            if r.decisions:
                for d in r.decisions[:2]:
                    print(f"         -> {d.reason[:65]}")

    # ══════════════════════════════════════════════════════════
    # Phase 4: Heavy mixed traffic — 2,000 tasks (datacenter burst)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("[Phase 4] Datacenter burst — 2,000 mixed-workload tasks")
    print(f"{'─' * 70}")

    mixed_tasks = gen.generate_job_batch(2000, base_step=500)
    # Inject 15% explicitly redundant tasks into the mix
    n_redundant_inject = 300
    redundant_burst = gen.generate_redundant_burst(n_redundant_inject, profile_name="finetune_bert")
    # Also add some converged LLM training tasks
    redundant_llm = gen.generate_redundant_burst(100, profile_name="llm_pretrain")
    mixed_tasks = mixed_tasks + redundant_burst + redundant_llm

    # Shuffle to simulate real interleaved arrival
    random.Random(123).shuffle(mixed_tasks)

    executed_mixed = 0
    strained_mixed = 0
    mixed_flops_saved = 0.0
    mixed_time_saved = 0.0
    mixed_cost_saved = 0.0
    mixed_verdicts = defaultdict(int)
    t_start = time.perf_counter()

    for task in mixed_tasks:
        buffer.push(task)
        result = pipeline.process_task(task)
        mixed_verdicts[result.verdict.name] += 1
        if result.verdict == TaskVerdict.EXECUTE:
            executed_mixed += 1
        else:
            strained_mixed += 1
            mixed_flops_saved += result.compute_saved_flops
            mixed_time_saved += result.time_saved_s
            mixed_cost_saved += estimate_cost_saved(task)

    mixed_wall = time.perf_counter() - t_start

    print(f"  Total tasks:  {len(mixed_tasks):,} (2,000 mixed + 300 converged FT + 100 converged LLM)")
    print(f"  Executed:     {executed_mixed:,}")
    print(f"  Strained:     {strained_mixed:,} ({strained_mixed/len(mixed_tasks)*100:.1f}%)")
    print(f"  FLOPs saved:  {format_flops(mixed_flops_saved)}")
    print(f"  Time saved:   {format_time(mixed_time_saved)}")
    print(f"  Cost saved:   {format_cost(mixed_cost_saved)}")
    print(f"  Throughput:   {len(mixed_tasks) / mixed_wall:,.0f} tasks/sec  ({mixed_wall:.3f}s wall)")
    print(f"  Verdicts:     { {k: v for k, v in sorted(mixed_verdicts.items())} }")

    # ══════════════════════════════════════════════════════════
    # Phase 5: Buffer & matrix stats (cluster-wide)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("[Phase 5] Buffer & telemetry matrix — cluster-wide state")
    print(f"{'─' * 70}")

    # Pick a representative GPU
    rep_gpu = "node-0-GPU-0"
    matrix = buffer.get_matrix(rep_gpu, n_tasks=100)

    print(f"  Buffer active GPUs: {len(buffer.gpu_ids)}")
    print(f"  Total buffered:     {buffer.total_tasks:,} tasks")
    print(f"  Matrix ({rep_gpu}): {matrix.shape[0]} tasks x {matrix.shape[1]} features")

    # Aggregate across nodes
    node_task_counts = defaultdict(int)
    for gid in buffer.gpu_ids:
        node = gid.rsplit("-GPU-", 1)[0] if "-GPU-" in gid else "unknown"
        node_task_counts[node] += 1
    print(f"  Tasks by node:")
    for node in sorted(node_task_counts.keys()):
        print(f"    {node}: {node_task_counts[node]} GPU streams")

    # ══════════════════════════════════════════════════════════
    # Phase 6: Redundancy Strainer standalone (multi-workload)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("[Phase 6] Redundancy strainer — standalone multi-workload check")
    print(f"{'─' * 70}")

    redundancy = RedundancyStrainer()

    # Test against different workload profiles
    profiles_to_test = ["llm_pretrain", "finetune_bert", "inference_llm"]
    for prof_name in profiles_to_test:
        profile = WORKLOAD_PROFILES[prof_name]
        productive = gen.generate_job_batch(1, base_step=50)[0]
        converged = gen.generate_redundant_burst(1, profile_name=prof_name)[0]

        prod_d = redundancy.check(productive)
        conv_d = redundancy.check(converged)

        print(f"\n  {profile['name']}:")
        print(f"    Productive task -> {len(prod_d)} decisions")
        print(f"    Converged task  -> {len(conv_d)} decisions")
        for d in conv_d:
            print(f"      {d.verdict.name:12s} | {d.metric:>25s} | {d.reason[:55]}")

    # ══════════════════════════════════════════════════════════
    # Phase 7: Convergence strainer — trajectory analysis
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("[Phase 7] Convergence strainer — trajectory analysis (4 GPUs)")
    print(f"{'─' * 70}")

    convergence = ConvergenceStrainer()

    # Warm up on 4 GPUs simultaneously (simulating cluster monitoring)
    test_gpus = ["node-0-GPU-0", "node-1-GPU-0", "node-4-GPU-0", "node-7-GPU-0"]
    for gpu_id in test_gpus:
        for i in range(200):
            t = gen.generate_job_batch(1, base_step=i)[0]
            vec = t.to_vector()
            convergence.update_and_score(gpu_id, vec)

    print(f"  Warmed up {len(test_gpus)} GPUs with 200 steps each (800 total)")

    # Test with converged task on each
    for gpu_id in test_gpus:
        red = gen.generate_redundant_burst(1, profile_name="finetune_bert")[0]
        red_vec = red.to_vector()
        conv_score, conv_signals = convergence.update_and_score(gpu_id, red_vec)
        gpu_model = next(g["model"] for g in GPU_ROSTER if g["gpu_id"] == gpu_id)
        print(f"\n  {gpu_id} ({gpu_model}):")
        print(f"    Redundancy score: {conv_score:.4f}")
        print(f"    Top signals:")
        for name, z_val in conv_signals[:4]:
            print(f"      {name:>28s}  z={z_val:.2f}")

    # ══════════════════════════════════════════════════════════
    # Phase 8: QUBO Quantum Scheduler — 64-task batch
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("[Phase 8] QUBO Quantum Scheduler — 64-task batch optimisation")
    print(f"{'─' * 70}")

    q_scheduler = QuantumStrainScheduler(
        config=SchedulerConfig(
            alpha=2.0,
            beta=0.4,
            gamma=0.3,
            delta=0.15,
            batch_size=64,
        ),
    )

    # Build a 64-task batch: 48 productive + 16 redundant
    qubo_productive = gen.generate_job_batch(48, base_step=2000)
    qubo_redundant = gen.generate_redundant_burst(16, profile_name="finetune_bert")
    qubo_batch = qubo_productive + qubo_redundant
    random.Random(99).shuffle(qubo_batch)

    t_qubo = time.perf_counter()
    qubo_results = q_scheduler.schedule(qubo_batch)
    qubo_wall = time.perf_counter() - t_qubo

    qubo_executed = sum(1 for r in qubo_results if r.verdict == TaskVerdict.EXECUTE)
    qubo_strained = len(qubo_results) - qubo_executed
    qubo_flops = sum(r.compute_saved_flops for r in qubo_results)
    qubo_time_saved = sum(r.time_saved_s for r in qubo_results)
    qubo_cost_saved = sum(estimate_cost_saved(qubo_batch[i])
                          for i, r in enumerate(qubo_results)
                          if r.verdict != TaskVerdict.EXECUTE)

    qubo_verdicts = defaultdict(int)
    for r in qubo_results:
        qubo_verdicts[r.verdict.name] += 1

    print(f"  Batch size:    64 (48 productive + 16 redundant)")
    print(f"  QUBO matrix:   64 x 64 ({64*63//2:,} pairwise interactions)")
    solver_name = qubo_results[0].strainer_method.split(':')[1] if qubo_results else 'n/a'
    print(f"  Solver:        {solver_name}")
    print(f"  QUBO energy:   {q_scheduler.qubo_energies[0]:.4f}")
    print(f"  Solve time:    {qubo_wall:.4f}s")
    print(f"  Executed:      {qubo_executed}")
    print(f"  Strained:      {qubo_strained}")
    print(f"  FLOPs saved:   {format_flops(qubo_flops)}")
    print(f"  Time saved:    {format_time(qubo_time_saved)}")
    print(f"  Cost saved:    {format_cost(qubo_cost_saved)}")
    print(f"  Verdicts:      { {k: v for k, v in sorted(qubo_verdicts.items())} }")
    print()
    print("  Why quantum?  The QUBO captures task-task interactions:")
    print("    * Data similarity coupling (jointly redundant batches)")
    print("    * Consecutive step anti-correlation (no long skip gaps)")
    print("    * Cross-GPU fairness (balanced strain across GPUs)")
    print("    * 64x64 = 2,016 pairwise interactions — beyond greedy reach")

    # ══════════════════════════════════════════════════════════
    # Phase 9: Second QUBO batch — LLM training heavy
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("[Phase 9] QUBO Quantum Scheduler — 48-task LLM training batch")
    print(f"{'─' * 70}")

    qubo_llm_productive = gen.generate_job_batch(36, base_step=3000)
    qubo_llm_redundant = gen.generate_redundant_burst(12, profile_name="llm_pretrain")
    qubo_llm_batch = qubo_llm_productive + qubo_llm_redundant
    random.Random(77).shuffle(qubo_llm_batch)

    t_qubo2 = time.perf_counter()
    qubo_llm_results = q_scheduler.schedule(qubo_llm_batch)
    qubo2_wall = time.perf_counter() - t_qubo2

    qubo2_executed = sum(1 for r in qubo_llm_results if r.verdict == TaskVerdict.EXECUTE)
    qubo2_strained = len(qubo_llm_results) - qubo2_executed
    qubo2_flops = sum(r.compute_saved_flops for r in qubo_llm_results)
    qubo2_time_saved = sum(r.time_saved_s for r in qubo_llm_results)
    qubo2_cost_saved = sum(estimate_cost_saved(qubo_llm_batch[i])
                           for i, r in enumerate(qubo_llm_results)
                           if r.verdict != TaskVerdict.EXECUTE)

    print(f"  Batch size:    48 (36 mixed + 12 converged LLM)")
    print(f"  QUBO matrix:   48 x 48 ({48*47//2:,} pairwise interactions)")
    print(f"  QUBO energy:   {q_scheduler.qubo_energies[-1]:.4f}")
    print(f"  Solve time:    {qubo2_wall:.4f}s")
    print(f"  Executed:      {qubo2_executed}")
    print(f"  Strained:      {qubo2_strained}")
    print(f"  FLOPs saved:   {format_flops(qubo2_flops)}  <- LLM training steps are expensive!")
    print(f"  Time saved:    {format_time(qubo2_time_saved)}")
    print(f"  Cost saved:    {format_cost(qubo2_cost_saved)}")

    # ══════════════════════════════════════════════════════════
    # Phase 10: Quantum Advantage — 16-task conflict graph
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("[Phase 10] Quantum Advantage — 16-task conflict graph purification")
    print(f"{'─' * 70}")

    qa_batch = gen.generate_job_batch(12, base_step=4000)
    qa_redundant = gen.generate_redundant_burst(4, profile_name="vision_train")
    qa_full = qa_batch + qa_redundant

    qa_pipeline = QuantumAdvantagePipeline(PipelineConfig(
        p_layers=2,
        n_restarts=3,
        maxfev=80,
        n_shots=1024,
        top_k_samples=64,
        purify_threshold=0.55,
        seed=42,
    ))

    qa_result = qa_pipeline.run(qa_full)

    print(f"  Tasks:           {qa_result.n_tasks}")
    print(f"  Conflict graph:  {qa_result.original_edges} edges "
          f"(density {qa_result.graph_density_before:.2%})")
    print(f"  QUBO size:       {qa_result.qubo_size} x {qa_result.qubo_size}")
    print(f"  Ising ||h||:     {qa_result.ising_h_norm:.4f}")
    print(f"  Ising J nnz:     {qa_result.ising_j_nnz}")
    print(f"  QAOA p layers:   {qa_result.p_layers}")
    print(f"  QAOA energy:     {qa_result.qaoa_optimal_energy:.4f}")
    print(f"  Samples used:    {qa_result.n_samples_used}")
    print(f"  Edges dropped:   {qa_result.edges_dropped} "
          f"({qa_result.edge_drop_ratio:.0%})")
    print(f"  Purified graph:  {qa_result.purified_edges} edges "
          f"(density {qa_result.graph_density_after:.2%})")
    print(f"  Makespan BEFORE: {qa_result.original_makespan} time slots")
    print(f"  Makespan AFTER:  {qa_result.purified_makespan} time slots")
    print(f"  Makespan saving: {qa_result.makespan_reduction:.0%}")
    print(f"  Max parallelism: {qa_result.max_parallelism} tasks/slot")
    print(f"  Coloring valid:  original={qa_result.original_coloring_valid}, "
          f"purified={qa_result.purified_coloring_valid}")
    print(f"  Total time:      {qa_result.total_time:.4f}s")

    # ══════════════════════════════════════════════════════════
    # Phase 11: Quantum Advantage — 18-task larger conflict graph
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("[Phase 11] Quantum Advantage — 18-task conflict graph (larger)")
    print(f"{'─' * 70}")

    qa2_batch = gen.generate_job_batch(12, base_step=5000)
    qa2_redundant = gen.generate_redundant_burst(6, profile_name="diffusion_train")
    qa2_full = qa2_batch + qa2_redundant

    qa2_pipeline = QuantumAdvantagePipeline(PipelineConfig(
        p_layers=2,
        n_restarts=3,
        maxfev=80,
        n_shots=1024,
        top_k_samples=64,
        purify_threshold=0.55,
        seed=99,
    ))

    qa2_result = qa2_pipeline.run(qa2_full)

    print(f"  Tasks:           {qa2_result.n_tasks}")
    print(f"  Conflict graph:  {qa2_result.original_edges} edges "
          f"(density {qa2_result.graph_density_before:.2%})")
    print(f"  QUBO size:       {qa2_result.qubo_size} x {qa2_result.qubo_size}")
    print(f"  QAOA energy:     {qa2_result.qaoa_optimal_energy:.4f}")
    print(f"  Edges dropped:   {qa2_result.edges_dropped} "
          f"({qa2_result.edge_drop_ratio:.0%})")
    print(f"  Purified graph:  {qa2_result.purified_edges} edges "
          f"(density {qa2_result.graph_density_after:.2%})")
    print(f"  Makespan BEFORE: {qa2_result.original_makespan} time slots")
    print(f"  Makespan AFTER:  {qa2_result.purified_makespan} time slots")
    print(f"  Makespan saving: {qa2_result.makespan_reduction:.0%}")
    print(f"  Max parallelism: {qa2_result.max_parallelism} tasks/slot")
    print(f"  Total time:      {qa2_result.total_time:.4f}s")

    print()
    print("  Pipeline: conflict graph -> QUBO -> Ising -> QAOA circuit")
    print("    -> sampling -> graph purification -> DSatur coloring -> schedule")

    # ══════════════════════════════════════════════════════════
    #  SUMMARY — datacenter-scale results
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  DEMO COMPLETE — Q-Strainer Datacenter-Scale Results")
    print(f"{'=' * 70}")

    total_tasks = (len(warmup_tasks) + len(redundant_tasks) + len(mixed_tasks)
                   + len(qubo_batch) + len(qubo_llm_batch)
                   + len(qa_full) + len(qa2_full))

    stats = pipeline.stats
    all_qubo_flops = qubo_flops + qubo2_flops
    all_qubo_time = qubo_time_saved + qubo2_time_saved
    all_qubo_cost = qubo_cost_saved + qubo2_cost_saved
    all_qubo_strained = qubo_strained + qubo2_strained
    all_qubo_executed = qubo_executed + qubo2_executed

    combined_flops = stats['total_flops_saved'] + all_qubo_flops
    combined_time = stats['total_time_saved_s'] + all_qubo_time
    combined_cost = (warmup_cost_saved + redundant_cost + mixed_cost_saved
                     + all_qubo_cost)
    combined_strained = stats['tasks_strained'] + all_qubo_strained
    combined_executed = stats['tasks_executed'] + all_qubo_executed

    print(f"\n  Cluster:       {len(GPU_ROSTER)} GPUs | {len(CLUSTER_LAYOUT)} nodes | "
          f"${hourly_cost:.0f}/hr")
    print(f"  Total tasks:   {total_tasks:,}")
    print(f"  Executed:      {combined_executed:,}")
    print(f"  Strained:      {combined_strained:,}")
    print(f"  Strain ratio:  {combined_strained / max(total_tasks, 1):.1%}")
    print(f"  {'─' * 46}")
    print(f"  FLOPs saved:   {format_flops(combined_flops)}")
    print(f"  Compute saved: {format_time(combined_time)}")
    print(f"  Cost saved:    {format_cost(combined_cost)}")
    print(f"  {'─' * 46}")
    print(f"  Pipeline stages:  Redundancy -> Convergence -> Predictive")
    print(f"  QUBO scheduler:   64x64 + 48x48 via {solver_name}")
    print(f"  Quantum advantage (16 tasks): makespan {qa_result.original_makespan} -> "
          f"{qa_result.purified_makespan} ({qa_result.makespan_reduction:.0%} reduction)")
    print(f"  Quantum advantage (18 tasks): makespan {qa2_result.original_makespan} -> "
          f"{qa2_result.purified_makespan} ({qa2_result.makespan_reduction:.0%} reduction)")
    print(f"\n  Projected daily savings (at 100% utilisation):")
    daily_tasks = total_tasks * (86400 / max(warmup_wall + mixed_wall, 1))
    daily_cost = combined_cost * (86400 / max(warmup_wall + mixed_wall, 1))
    print(f"    Tasks/day:     ~{daily_tasks:,.0f}")
    print(f"    Daily savings: ~{format_cost(daily_cost)}")
    print(f"    Monthly:       ~{format_cost(daily_cost * 30)}")
    print(f"{'=' * 70}")

    # ── Save results ──
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%d_%H%M%S")

    results = {
        "run_id": run_id,
        "timestamp": now.isoformat(),
        "cluster": {
            "total_gpus": len(GPU_ROSTER),
            "nodes": len(CLUSTER_LAYOUT),
            "layout": [
                {"node": n, "model": m, "count": c}
                for n, m, c in CLUSTER_LAYOUT
            ],
            "total_tflops": total_tflops,
            "total_vram_gb": total_vram,
            "hourly_cost_usd": hourly_cost,
        },
        "summary": {
            "total_tasks": total_tasks,
            "tasks_executed": combined_executed,
            "tasks_strained": combined_strained,
            "strain_ratio": round(combined_strained / max(total_tasks, 1), 4),
            "total_flops_saved": combined_flops,
            "total_time_saved_s": round(combined_time, 2),
            "total_cost_saved_usd": round(combined_cost, 4),
            "feature_expansion": f"{len(raw)} -> {len(derived)}",
            "raw_features": len(raw),
            "derived_features": len(derived),
            "verdict_distribution": {
                **stats.get("verdict_distribution", {}),
                **{f"quantum_{k}": v for k, v in qubo_verdicts.items()},
            },
            "quantum_scheduler": {
                "batch_1": {
                    "batch_size": 64,
                    "qubo_dimensions": "64x64",
                    "pairwise_interactions": 64 * 63 // 2,
                    "solver": solver_name,
                    "qubo_energy": round(q_scheduler.qubo_energies[0], 4),
                    "executed": qubo_executed,
                    "strained": qubo_strained,
                    "flops_saved": qubo_flops,
                    "time_saved_s": round(qubo_time_saved, 4),
                    "cost_saved_usd": round(qubo_cost_saved, 4),
                },
                "batch_2_llm": {
                    "batch_size": 48,
                    "qubo_dimensions": "48x48",
                    "pairwise_interactions": 48 * 47 // 2,
                    "qubo_energy": round(q_scheduler.qubo_energies[-1], 4),
                    "executed": qubo2_executed,
                    "strained": qubo2_strained,
                    "flops_saved": qubo2_flops,
                    "time_saved_s": round(qubo2_time_saved, 4),
                    "cost_saved_usd": round(qubo2_cost_saved, 4),
                },
            },
            "quantum_advantage": {
                "batch_16": {
                    "tasks": qa_result.n_tasks,
                    "original_edges": qa_result.original_edges,
                    "purified_edges": qa_result.purified_edges,
                    "edges_dropped": qa_result.edges_dropped,
                    "edge_drop_ratio": round(qa_result.edge_drop_ratio, 4),
                    "original_makespan": qa_result.original_makespan,
                    "purified_makespan": qa_result.purified_makespan,
                    "makespan_reduction": round(qa_result.makespan_reduction, 4),
                    "qaoa_energy": round(qa_result.qaoa_optimal_energy, 4),
                    "max_parallelism": qa_result.max_parallelism,
                    "total_time_s": round(qa_result.total_time, 4),
                },
                "batch_18": {
                    "tasks": qa2_result.n_tasks,
                    "original_edges": qa2_result.original_edges,
                    "purified_edges": qa2_result.purified_edges,
                    "edges_dropped": qa2_result.edges_dropped,
                    "edge_drop_ratio": round(qa2_result.edge_drop_ratio, 4),
                    "original_makespan": qa2_result.original_makespan,
                    "purified_makespan": qa2_result.purified_makespan,
                    "makespan_reduction": round(qa2_result.makespan_reduction, 4),
                    "qaoa_energy": round(qa2_result.qaoa_optimal_energy, 4),
                    "max_parallelism": qa2_result.max_parallelism,
                    "total_time_s": round(qa2_result.total_time, 4),
                },
            },
            "projected_daily": {
                "tasks_per_day": round(daily_tasks),
                "daily_savings_usd": round(daily_cost, 2),
                "monthly_savings_usd": round(daily_cost * 30, 2),
            },
        },
        "phases": [
            {
                "name": "Cluster Warm-up (1,000 tasks across 64 GPUs)",
                "phase": 1,
                "tasks": len(warmup_tasks),
                "executed": executed_warmup,
                "strained": strained_warmup,
                "flops_saved": warmup_flops_saved,
                "time_saved_s": round(warmup_time_saved, 4),
                "cost_saved_usd": round(warmup_cost_saved, 4),
                "gpus_active": len(warmup_gpus_used),
                "nodes_active": len(warmup_nodes_used),
                "wall_time_s": round(warmup_wall, 6),
                "throughput_tps": round(1000 / warmup_wall),
            },
            {
                "name": "Feature Extraction (multi-GPU)",
                "phase": 2,
                "raw_features": len(raw),
                "derived_features": len(derived),
            },
            {
                "name": "Redundancy Injection (200 converged tasks)",
                "phase": 3,
                "tasks": 200,
                "strained": len(redundant_decisions),
                "flops_saved": redundant_flops,
                "time_saved_s": round(redundant_time, 4),
                "cost_saved_usd": round(redundant_cost, 4),
            },
            {
                "name": "Datacenter Burst (2,400 mixed tasks)",
                "phase": 4,
                "tasks": len(mixed_tasks),
                "executed": executed_mixed,
                "strained": strained_mixed,
                "flops_saved": mixed_flops_saved,
                "time_saved_s": round(mixed_time_saved, 4),
                "cost_saved_usd": round(mixed_cost_saved, 4),
                "wall_time_s": round(mixed_wall, 6),
                "throughput_tps": round(len(mixed_tasks) / mixed_wall),
            },
            {
                "name": "Buffer & Telemetry Matrix",
                "phase": 5,
                "gpu_count": len(buffer.gpu_ids),
                "total_tasks": buffer.total_tasks,
                "matrix_shape": list(matrix.shape),
            },
            {
                "name": "Redundancy Strainer Standalone (multi-workload)",
                "phase": 6,
            },
            {
                "name": "Convergence Strainer (4-GPU trajectory)",
                "phase": 7,
            },
            {
                "name": "QUBO Quantum Scheduler (64-task batch)",
                "phase": 8,
                "batch_size": 64,
                "qubo_dimensions": "64 x 64",
                "pairwise_interactions": 64 * 63 // 2,
                "solver": solver_name,
                "qubo_energy": round(q_scheduler.qubo_energies[0], 4),
                "solve_time_s": round(qubo_wall, 6),
                "executed": qubo_executed,
                "strained": qubo_strained,
                "flops_saved": qubo_flops,
                "time_saved_s": round(qubo_time_saved, 4),
                "cost_saved_usd": round(qubo_cost_saved, 4),
            },
            {
                "name": "QUBO Quantum Scheduler (48-task LLM batch)",
                "phase": 9,
                "batch_size": 48,
                "solver": solver_name,
                "qubo_energy": round(q_scheduler.qubo_energies[-1], 4),
                "solve_time_s": round(qubo2_wall, 6),
                "executed": qubo2_executed,
                "strained": qubo2_strained,
                "flops_saved": qubo2_flops,
                "time_saved_s": round(qubo2_time_saved, 4),
                "cost_saved_usd": round(qubo2_cost_saved, 4),
            },
            {
                "name": "Quantum Advantage (16-task conflict graph)",
                "phase": 10,
                "tasks": qa_result.n_tasks,
                "original_makespan": qa_result.original_makespan,
                "purified_makespan": qa_result.purified_makespan,
                "makespan_reduction": round(qa_result.makespan_reduction, 4),
                "conflict_graph_edges": qa_result.original_edges,
                "edges_dropped": qa_result.edges_dropped,
                "qaoa_energy": round(qa_result.qaoa_optimal_energy, 4),
                "max_parallelism": qa_result.max_parallelism,
                "total_time_s": round(qa_result.total_time, 4),
            },
            {
                "name": "Quantum Advantage (18-task conflict graph)",
                "phase": 11,
                "tasks": qa2_result.n_tasks,
                "original_makespan": qa2_result.original_makespan,
                "purified_makespan": qa2_result.purified_makespan,
                "makespan_reduction": round(qa2_result.makespan_reduction, 4),
                "conflict_graph_edges": qa2_result.original_edges,
                "edges_dropped": qa2_result.edges_dropped,
                "qaoa_energy": round(qa2_result.qaoa_optimal_energy, 4),
                "max_parallelism": qa2_result.max_parallelism,
                "total_time_s": round(qa2_result.total_time, 4),
            },
        ],
    }

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUNS_DIR / f"demo_{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    # ── Launch dashboard in browser (optional) ──
    if "--dashboard" in sys.argv:
        print("\n  Launching dashboard on http://localhost:8050 ...")
        project_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(project_root))
        from dashboard import main as dashboard_main
        dashboard_main()
    else:
        print("\n  Run with --dashboard to launch the live dashboard.")


if __name__ == "__main__":
    main()
