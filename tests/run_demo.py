"""Q-Strainer local demo — no GPU required.

Demonstrates the compute workload strainer: generates synthetic training
tasks and shows how Q-Strainer decides which to execute vs skip/approximate.

Run with:  py tests/run_demo.py       (Windows)
           python tests/run_demo.py   (Linux/macOS)

Results are saved to runs/demo_<timestamp>.json after each run.
"""

import json
import os
import time
import random
from datetime import datetime, timezone
from pathlib import Path

from qstrainer.models import ComputeTask, WorkloadBuffer
from qstrainer.models.enums import TaskVerdict, ComputePhase, JobType
from qstrainer.features import DerivedFeatureExtractor
from qstrainer.stages.threshold import RedundancyStrainer
from qstrainer.stages.statistical import ConvergenceStrainer
from qstrainer.pipeline import QStrainer, QuantumStrainScheduler, SchedulerConfig
from qstrainer.quantum import QuantumAdvantagePipeline, PipelineConfig

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def make_productive_task(gpu_id: str = "GPU-0", step: int = 0) -> ComputeTask:
    """Generate a productive training task — meaningful gradients, loss improving."""
    loss = max(2.5 - step * 0.01 + random.gauss(0, 0.05), 0.1)
    return ComputeTask(
        timestamp=time.time(),
        task_id=f"task-{step:06d}",
        gpu_id=gpu_id,
        job_id="train-resnet-50",
        step_number=step,
        loss=loss,
        loss_delta=-0.01 + random.gauss(0, 0.003),
        gradient_norm=0.5 + random.gauss(0, 0.1),
        gradient_variance=0.05 + random.gauss(0, 0.01),
        learning_rate=1e-3,
        batch_size=256,
        epoch=step // 50,
        epoch_progress=(step % 50) / 50.0,
        estimated_flops=2e12,         # 2 TFLOP per step
        estimated_time_s=0.5,         # 0.5s per step
        memory_footprint_gb=12.0,
        compute_phase=ComputePhase.FORWARD_PASS,
        job_type=JobType.TRAINING,
        convergence_score=min(step * 0.003, 0.7),
        param_update_magnitude=0.01 + random.gauss(0, 0.002),
        data_similarity=0.3 + random.gauss(0, 0.1),
        flop_utilization=0.75 + random.gauss(0, 0.05),
        throughput_samples_per_sec=512 + random.gauss(0, 30),
        node_id="node-0",
    )


def make_redundant_task(gpu_id: str = "GPU-0", step: int = 999) -> ComputeTask:
    """Generate a clearly redundant task — converged, tiny gradients."""
    return ComputeTask(
        timestamp=time.time(),
        task_id=f"task-{step:06d}",
        gpu_id=gpu_id,
        job_id="train-resnet-50",
        step_number=step,
        loss=0.12 + random.gauss(0, 0.001),          # Loss plateaued
        loss_delta=random.gauss(0, 1e-7),             # No improvement
        gradient_norm=1e-8 + abs(random.gauss(0, 1e-9)),  # Tiny gradient
        gradient_variance=1e-10,
        learning_rate=1e-5,
        batch_size=256,
        epoch=step // 50,
        epoch_progress=(step % 50) / 50.0,
        estimated_flops=2e12,
        estimated_time_s=0.5,
        memory_footprint_gb=12.0,
        compute_phase=ComputePhase.FORWARD_PASS,
        job_type=JobType.TRAINING,
        convergence_score=0.98,                       # Fully converged
        param_update_magnitude=1e-10,                 # Nothing changes
        data_similarity=0.99,                         # Duplicate data
        flop_utilization=0.75,
        throughput_samples_per_sec=512,
        node_id="node-0",
    )


def main():
    print("=" * 60)
    print("  Q-Strainer Local Demo")
    print("  GPU Compute Workload Strainer")
    print("=" * 60)

    # --- Initialize pipeline ---
    gpu_ids = [f"GPU-{i}" for i in range(4)]
    pipeline = QStrainer()
    buffer = WorkloadBuffer(max_tasks_per_gpu=500)
    extractor = DerivedFeatureExtractor()

    # ── Phase 1: Productive training warmup ──
    print("\n[Phase 1] Training warmup — 200 productive steps...")
    print("-" * 60)

    executed_warmup = 0
    strained_warmup = 0
    t_start = time.perf_counter()

    for i in range(200):
        gid = gpu_ids[i % 4]
        task = make_productive_task(gpu_id=gid, step=i)
        buffer.push(task)
        result = pipeline.process_task(task)
        if result.verdict == TaskVerdict.EXECUTE:
            executed_warmup += 1
        else:
            strained_warmup += 1

    warmup_time = time.perf_counter() - t_start
    print(f"  Tasks processed  : 200")
    print(f"  Executed         : {executed_warmup}")
    print(f"  Strained (saved) : {strained_warmup}")
    print(f"  Wall time        : {warmup_time:.4f}s")
    print(f"  Throughput       : {200 / warmup_time:.0f} tasks/sec")

    # ── Phase 2: Feature extraction ──
    print("\n[Phase 2] Feature extraction demo...")
    print("-" * 60)

    sample = make_productive_task(step=50)
    raw = sample.to_vector()
    derived = extractor.extract(sample.gpu_id, raw)

    print(f"  Raw features     : {len(raw)} fields")
    print(f"  Derived features : {len(derived)} features")
    print(f"  Expansion ratio  : {len(raw)} -> {len(derived)}")

    # ── Phase 3: Inject redundant tasks ──
    print("\n[Phase 3] Injecting 10 redundant tasks...")
    print("-" * 60)

    redundant_decisions = []
    for i in range(10):
        gid = gpu_ids[i % 4]
        task = make_redundant_task(gpu_id=gid, step=1000 + i)
        buffer.push(task)
        result = pipeline.process_task(task)
        if result.verdict != TaskVerdict.EXECUTE:
            redundant_decisions.append(result)

    print(f"  Redundant tasks  : 10")
    print(f"  Strained (saved) : {len(redundant_decisions)}")
    print()

    if redundant_decisions:
        print("  Straining decisions:")
        for i, r in enumerate(redundant_decisions[:10]):
            savings = f"saved {r.time_saved_s:.2f}s" if r.time_saved_s > 0 else ""
            print(f"    [{i+1:2d}] {r.verdict.name:12s} | "
                  f"redundancy={r.redundancy_score:.3f} | {savings}")
            if r.decisions:
                for d in r.decisions[:2]:
                    print(f"         -> {d.reason[:60]}")

    # ── Phase 4: Mixed traffic ──
    print(f"\n[Phase 4] Mixed traffic (90 productive + 10 redundant)...")
    print("-" * 60)

    executed_mixed = 0
    strained_mixed = 0
    total_mixed = 100
    flops_saved_mixed = 0.0
    time_saved_mixed = 0.0
    t_start = time.perf_counter()

    for i in range(total_mixed):
        gid = gpu_ids[i % 4]
        step = 200 + i
        if i % 10 == 7:  # Every 10th task is redundant
            task = make_redundant_task(gpu_id=gid, step=step)
        else:
            task = make_productive_task(gpu_id=gid, step=step)
        buffer.push(task)
        result = pipeline.process_task(task)
        if result.verdict == TaskVerdict.EXECUTE:
            executed_mixed += 1
        else:
            strained_mixed += 1
            flops_saved_mixed += result.compute_saved_flops
            time_saved_mixed += result.time_saved_s

    mixed_time = time.perf_counter() - t_start

    print(f"  Total tasks      : {total_mixed}")
    print(f"  Executed         : {executed_mixed}")
    print(f"  Strained (saved) : {strained_mixed}")
    print(f"  FLOPs saved      : {flops_saved_mixed:.2e}")
    print(f"  Time saved       : {time_saved_mixed:.1f}s")
    print(f"  Wall time        : {mixed_time:.4f}s")
    print(f"  Throughput       : {total_mixed / mixed_time:.0f} tasks/sec")

    # ── Phase 5: Buffer stats ──
    print(f"\n[Phase 5] Buffer & matrix export...")
    print("-" * 60)

    matrix = buffer.get_matrix("GPU-0", n_tasks=50)
    print(f"  Buffer GPUs      : {len(buffer.gpu_ids)}")
    print(f"  Buffer filled    : {buffer.total_tasks} total tasks")
    print(f"  Exported matrix  : {matrix.shape[0]} rows x {matrix.shape[1]} cols")

    # ── Phase 6: Redundancy strainer standalone ──
    print(f"\n[Phase 6] Redundancy strainer standalone check...")
    print("-" * 60)

    redundancy = RedundancyStrainer()
    productive = make_productive_task(step=50)
    redundant = make_redundant_task()

    productive_decisions = redundancy.check(productive)
    redundant_decisions_standalone = redundancy.check(redundant)

    print(f"  Productive task  -> {len(productive_decisions)} decisions")
    print(f"  Redundant task   -> {len(redundant_decisions_standalone)} decisions")
    for d in redundant_decisions_standalone:
        print(f"    {d.verdict.name:12s} | {d.metric:>25s} | {d.reason[:50]}")

    # ── Phase 7: Convergence strainer ──
    print(f"\n[Phase 7] Convergence strainer trajectory demo...")
    print("-" * 60)

    convergence = ConvergenceStrainer()
    # Warm up on productive tasks
    for i in range(100):
        t = make_productive_task(gpu_id="GPU-CONV", step=i)
        vec = t.to_vector()
        convergence.update_and_score("GPU-CONV", vec)

    # Now feed a redundant task
    red = make_redundant_task(gpu_id="GPU-CONV")
    red_vec = red.to_vector()
    conv_score, conv_signals = convergence.update_and_score("GPU-CONV", red_vec)
    print(f"  Warmup tasks     : 100 productive")
    print(f"  Test task        : 1 redundant")
    print(f"  Redundancy score : {conv_score:.4f}")
    print(f"  Dominant signals :")
    for name, z_val in conv_signals[:5]:
        print(f"    {name:>25s}  z={z_val:.2f}")

    # ── Phase 8: Quantum Scheduler (QUBO batch optimisation) ──
    print(f"\n[Phase 8] QUBO Quantum Scheduler — batch optimisation...")
    print("-" * 60)

    q_scheduler = QuantumStrainScheduler(
        config=SchedulerConfig(
            alpha=2.0,
            beta=0.4,
            gamma=0.3,
            delta=0.15,
            batch_size=32,
        ),
    )

    # Build a mixed batch: 24 productive + 8 redundant across 4 GPUs
    qubo_batch = []
    for i in range(24):
        qubo_batch.append(make_productive_task(gpu_id=gpu_ids[i % 4], step=500 + i))
    for i in range(8):
        qubo_batch.append(make_redundant_task(gpu_id=gpu_ids[i % 4], step=2000 + i))

    t_qubo = time.perf_counter()
    qubo_results = q_scheduler.schedule(qubo_batch)
    qubo_time = time.perf_counter() - t_qubo

    q_stats = q_scheduler.stats
    qubo_executed = sum(1 for r in qubo_results if r.verdict == TaskVerdict.EXECUTE)
    qubo_strained = len(qubo_results) - qubo_executed
    qubo_flops = sum(r.compute_saved_flops for r in qubo_results)
    qubo_time_saved = sum(r.time_saved_s for r in qubo_results)
    qubo_cost_saved = sum(r.cost_saved_usd for r in qubo_results)

    # Verdict breakdown
    qubo_verdicts = {}
    for r in qubo_results:
        qubo_verdicts[r.verdict.name] = qubo_verdicts.get(r.verdict.name, 0) + 1

    print(f"  Batch size       : 32 (24 productive + 8 redundant)")
    print(f"  QUBO matrix      : 32 × 32 ({32*31//2} pairwise interactions)")
    print(f"  Solver           : {qubo_results[0].strainer_method.split(':')[1] if qubo_results else 'n/a'}")
    print(f"  QUBO energy      : {q_scheduler.qubo_energies[0]:.4f}")
    print(f"  Solve time       : {qubo_time:.4f}s")
    print(f"  Executed         : {qubo_executed}")
    print(f"  Strained (saved) : {qubo_strained}")
    print(f"  FLOPs saved      : {qubo_flops:.2e}")
    print(f"  Time saved       : {qubo_time_saved:.1f}s")
    print(f"  Cost saved       : ${qubo_cost_saved:.4f}")
    print(f"  Verdicts         : {qubo_verdicts}")
    print()
    print("  Why quantum? The QUBO captures task-task interactions that")
    print("  greedy per-task evaluation misses:")
    print("    • Data similarity coupling (jointly redundant batches)")
    print("    • Consecutive step anti-correlation (no long skip gaps)")
    print("    • Cross-GPU fairness (balanced strain across GPUs)")

    # ── Phase 9: Quantum Advantage Pipeline ──
    print(f"\n[Phase 9] Quantum Advantage — conflict graph purification...")
    print("-" * 60)

    # Build a 16-task batch for the quantum advantage pipeline
    qa_batch = []
    for i in range(12):
        qa_batch.append(make_productive_task(gpu_id=gpu_ids[i % 4], step=700 + i))
    for i in range(4):
        qa_batch.append(make_redundant_task(gpu_id=gpu_ids[i % 4], step=3000 + i))

    qa_pipeline = QuantumAdvantagePipeline(PipelineConfig(
        p_layers=2,
        n_restarts=3,
        maxfev=60,
        n_shots=512,
        top_k_samples=32,
        purify_threshold=0.55,
        seed=42,
    ))

    qa_result = qa_pipeline.run(qa_batch)

    print(f"  Tasks            : {qa_result.n_tasks}")
    print(f"  Conflict graph   : {qa_result.original_edges} edges "
          f"(density {qa_result.graph_density_before:.2%})")
    print(f"  QUBO size        : {qa_result.qubo_size} × {qa_result.qubo_size}")
    print(f"  Ising ||h||      : {qa_result.ising_h_norm:.4f}")
    print(f"  Ising J nnz      : {qa_result.ising_j_nnz}")
    print(f"  QAOA p layers    : {qa_result.p_layers}")
    print(f"  QAOA energy      : {qa_result.qaoa_optimal_energy:.4f}")
    print(f"  Samples used     : {qa_result.n_samples_used}")
    print(f"  Edges dropped    : {qa_result.edges_dropped} "
          f"({qa_result.edge_drop_ratio:.0%})")
    print(f"  Purified graph   : {qa_result.purified_edges} edges "
          f"(density {qa_result.graph_density_after:.2%})")
    print(f"  Makespan BEFORE  : {qa_result.original_makespan} time slots")
    print(f"  Makespan AFTER   : {qa_result.purified_makespan} time slots")
    print(f"  Makespan savings : {qa_result.makespan_reduction:.0%}")
    print(f"  Max parallelism  : {qa_result.max_parallelism} tasks/slot")
    print(f"  Coloring valid   : "
          f"original={qa_result.original_coloring_valid}, "
          f"purified={qa_result.purified_coloring_valid}")
    print(f"  Total time       : {qa_result.total_time:.4f}s")
    print()
    print("  Pipeline: conflict graph → QUBO → Ising → QAOA circuit")
    print("    → sampling → graph purification → DSatur coloring → schedule")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  DEMO COMPLETE — Q-Strainer Compute Workload Results")
    print("=" * 60)
    total_tasks = 200 + 10 + 100 + 32 + 16  # include quantum batch + qa batch
    stats = pipeline.stats
    combined_flops = stats['total_flops_saved'] + qubo_flops
    combined_time = stats['total_time_saved_s'] + qubo_time_saved
    combined_cost = stats['total_cost_saved_usd'] + qubo_cost_saved
    combined_strained = stats['tasks_strained'] + qubo_strained
    combined_executed = stats['tasks_executed'] + qubo_executed
    print(f"  Total tasks processed  : {total_tasks}")
    print(f"  Tasks executed         : {combined_executed}")
    print(f"  Tasks strained (saved) : {combined_strained}")
    print(f"  Strain ratio           : {combined_strained / total_tasks:.1%}")
    print(f"  Total FLOPs saved      : {combined_flops:.2e}")
    print(f"  Total time saved       : {combined_time:.1f}s")
    print(f"  Total cost saved       : ${combined_cost:.4f}")
    print(f"  Feature expansion      : 15 -> {len(derived)}")
    print(f"  Pipeline stages        : Redundancy -> Convergence -> Predictive")
    print(f"  Quantum scheduler      : QUBO {32}×{32} via {qubo_results[0].strainer_method.split(':')[1]}")
    print(f"  Quantum advantage      : makespan {qa_result.original_makespan} → {qa_result.purified_makespan} "
          f"({qa_result.makespan_reduction:.0%} reduction)")
    print("=" * 60)

    # ── Save results ──
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%d_%H%M%S")

    results = {
        "run_id": run_id,
        "timestamp": now.isoformat(),
        "summary": {
            "total_tasks": total_tasks,
            "tasks_executed": combined_executed,
            "tasks_strained": combined_strained,
            "strain_ratio": round(combined_strained / total_tasks, 4),
            "total_flops_saved": combined_flops,
            "total_time_saved_s": round(combined_time, 2),
            "total_cost_saved_usd": round(combined_cost, 6),
            "feature_expansion": f"15 -> {len(derived)}",
            "raw_features": 15,
            "derived_features": len(derived),
            "verdict_distribution": {
                **stats["verdict_distribution"],
                **{f"quantum_{k}": v for k, v in qubo_verdicts.items()},
            },
            "quantum_scheduler": {
                "batch_size": 32,
                "qubo_dimensions": 32,
                "pairwise_interactions": 32 * 31 // 2,
                "solver": qubo_results[0].strainer_method.split(":")[1] if qubo_results else "n/a",
                "qubo_energy": round(q_scheduler.qubo_energies[0], 4),
                "solve_time_s": round(qubo_time, 6),
                "executed": qubo_executed,
                "strained": qubo_strained,
                "flops_saved": qubo_flops,
                "time_saved_s": round(qubo_time_saved, 4),
                "cost_saved_usd": round(qubo_cost_saved, 6),
                "verdict_distribution": qubo_verdicts,
            },
            "quantum_advantage": {
                "tasks": 16,
                "conflict_graph": {
                    "nodes": qa_result.n_tasks,
                    "original_edges": qa_result.original_edges,
                    "purified_edges": qa_result.purified_edges,
                    "edges_dropped": qa_result.edges_dropped,
                    "edge_drop_ratio": round(qa_result.edge_drop_ratio, 4),
                    "original_density": round(qa_result.graph_density_before, 4),
                    "purified_density": round(qa_result.graph_density_after, 4),
                },
                "ising": {
                    "qubo_size": qa_result.qubo_size,
                    "h_norm": round(qa_result.ising_h_norm, 4),
                    "j_nnz": qa_result.ising_j_nnz,
                },
                "qaoa": {
                    "p_layers": qa_result.p_layers,
                    "n_samples": qa_result.n_samples_used,
                    "optimal_energy": round(qa_result.qaoa_optimal_energy, 4),
                },
                "makespan": {
                    "original": qa_result.original_makespan,
                    "purified": qa_result.purified_makespan,
                    "reduction": round(qa_result.makespan_reduction, 4),
                },
                "parallelism": {
                    "max": qa_result.max_parallelism,
                    "avg": round(qa_result.avg_parallelism, 2),
                },
                "coloring_valid": {
                    "original": qa_result.original_coloring_valid,
                    "purified": qa_result.purified_coloring_valid,
                },
                "timing": {
                    "total_s": round(qa_result.total_time, 4),
                    "graph_build_s": round(qa_result.graph_build_time, 6),
                    "qubo_build_s": round(qa_result.qubo_build_time, 6),
                    "ising_convert_s": round(qa_result.ising_convert_time, 6),
                    "qaoa_optimize_s": round(qa_result.qaoa_optimize_time, 4),
                    "sample_s": round(qa_result.sample_time, 6),
                    "purify_s": round(qa_result.purify_time, 6),
                    "color_original_s": round(qa_result.color_original_time, 6),
                    "color_purified_s": round(qa_result.color_purified_time, 6),
                },
            },
        },
        "phases": [
            {
                "name": "Training Warmup (200 productive steps)",
                "phase": 1,
                "tasks": 200,
                "executed": executed_warmup,
                "strained": strained_warmup,
                "wall_time_s": round(warmup_time, 6),
                "throughput_tps": round(200 / warmup_time),
            },
            {
                "name": "Feature Extraction",
                "phase": 2,
                "raw_features": len(raw),
                "derived_features": len(derived),
            },
            {
                "name": "Redundant Task Injection (10 tasks)",
                "phase": 3,
                "tasks": 10,
                "strained": len(redundant_decisions),
                "decision_details": [
                    {
                        "verdict": r.verdict.name,
                        "redundancy_score": round(r.redundancy_score, 4),
                        "time_saved_s": round(r.time_saved_s, 4),
                        "decisions": [
                            {"reason": d.reason, "metric": d.metric}
                            for d in r.decisions[:3]
                        ],
                    }
                    for r in redundant_decisions[:10]
                ],
            },
            {
                "name": "Mixed Traffic (90 productive + 10 redundant)",
                "phase": 4,
                "tasks": total_mixed,
                "executed": executed_mixed,
                "strained": strained_mixed,
                "flops_saved": flops_saved_mixed,
                "time_saved_s": round(time_saved_mixed, 4),
                "wall_time_s": round(mixed_time, 6),
                "throughput_tps": round(total_mixed / mixed_time),
            },
            {
                "name": "Buffer & Matrix Export",
                "phase": 5,
                "gpu_count": len(buffer.gpu_ids),
                "total_tasks": buffer.total_tasks,
                "matrix_shape": list(matrix.shape),
            },
            {
                "name": "Redundancy Strainer Standalone",
                "phase": 6,
                "productive_decisions": len(productive_decisions),
                "redundant_decisions": len(redundant_decisions_standalone),
                "decision_details": [
                    {
                        "verdict": d.verdict.name,
                        "metric": d.metric,
                        "reason": d.reason,
                    }
                    for d in redundant_decisions_standalone
                ],
            },
            {
                "name": "Convergence Strainer Trajectory",
                "phase": 7,
                "warmup_tasks": 100,
                "redundancy_score": round(conv_score, 4),
                "dominant_signals": [
                    {"feature": name, "z_score": round(z, 2)}
                    for name, z in conv_signals[:5]
                ],
            },
            {
                "name": "QUBO Quantum Scheduler",
                "phase": 8,
                "batch_size": 32,
                "qubo_dimensions": "32 × 32",
                "pairwise_interactions": 32 * 31 // 2,
                "solver": (
                    qubo_results[0].strainer_method.split(":")[1]
                    if qubo_results else "n/a"
                ),
                "qubo_energy": round(q_scheduler.qubo_energies[0], 4),
                "solve_time_s": round(qubo_time, 6),
                "executed": qubo_executed,
                "strained": qubo_strained,
                "flops_saved": qubo_flops,
                "time_saved_s": round(qubo_time_saved, 4),
                "cost_saved_usd": round(qubo_cost_saved, 6),
                "verdict_distribution": qubo_verdicts,
            },
            {
                "name": "Quantum Advantage Pipeline",
                "phase": 9,
                "tasks": 16,
                "original_makespan": qa_result.original_makespan,
                "purified_makespan": qa_result.purified_makespan,
                "makespan_reduction": round(qa_result.makespan_reduction, 4),
                "conflict_graph_edges": qa_result.original_edges,
                "edges_dropped": qa_result.edges_dropped,
                "edge_drop_ratio": round(qa_result.edge_drop_ratio, 4),
                "qaoa_energy": round(qa_result.qaoa_optimal_energy, 4),
                "max_parallelism": qa_result.max_parallelism,
                "avg_parallelism": round(qa_result.avg_parallelism, 2),
                "total_time_s": round(qa_result.total_time, 4),
            },
        ],
    }

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUNS_DIR / f"demo_{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()