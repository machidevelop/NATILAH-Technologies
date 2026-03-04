"""Benchmarks — fleet-scale simulation and solver comparison.

Called from the CLI via::

    qstrainer benchmark -n 100 -t 100
    qstrainer compare-solvers -n 15
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from qstrainer.ingestion.synthetic import SyntheticTelemetryGenerator
from qstrainer.models.enums import TaskVerdict
from qstrainer.pipeline.strainer import QStrainer


def run_fleet_benchmark(
    n_gpus: int = 100,
    tasks_per_gpu: int = 100,
    cfg: dict[str, Any] | None = None,
    redundant_rate: float = 0.02,
    converging_rate: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Simulate a GPU fleet and measure straining efficiency.

    Prints results to stdout and returns a summary dict.
    """
    rng = np.random.default_rng(seed)
    gen = SyntheticTelemetryGenerator(seed=seed)

    pipeline_cfg = (cfg or {}).get("pipeline", {})
    strainer = QStrainer.from_config(pipeline_cfg)

    total_tasks = 0
    total_strained = 0
    total_skipped = 0
    total_flops_saved = 0.0

    t0 = time.perf_counter()

    for gpu_idx in range(n_gpus):
        gpu_id = f"GPU-{gpu_idx:04d}"
        node_id = f"node-{gpu_idx // 8:03d}"

        r = rng.random()
        if r < redundant_rate:
            profile = "redundant"
        elif r < redundant_rate + converging_rate:
            profile = "converging"
        else:
            profile = "productive"

        for task_idx in range(tasks_per_gpu):
            if profile == "productive":
                task = gen.generate_healthy(gpu_id, node_id)
            elif profile == "converging":
                severity = task_idx / max(tasks_per_gpu - 1, 1)
                task = gen.generate_degrading(gpu_id, node_id, severity)
            else:
                task = gen.generate_failing(gpu_id, node_id)

            sr = strainer.process_task(task)
            total_tasks += 1

            if sr.verdict != TaskVerdict.EXECUTE:
                total_strained += 1
            if sr.verdict == TaskVerdict.SKIP:
                total_skipped += 1
            total_flops_saved += sr.compute_saved_flops

    elapsed = time.perf_counter() - t0

    strain_ratio = total_strained / max(total_tasks, 1)
    summary: dict[str, Any] = {
        "n_gpus": n_gpus,
        "tasks_per_gpu": tasks_per_gpu,
        "total_tasks": total_tasks,
        "strained": total_strained,
        "skipped": total_skipped,
        "strain_ratio": strain_ratio,
        "total_flops_saved": total_flops_saved,
        "elapsed_s": elapsed,
        "tasks_per_sec": total_tasks / max(elapsed, 1e-9),
    }

    # Print report
    print("=" * 70)
    print("FLEET-SCALE BENCHMARK")
    print("=" * 70)
    print(f"  Fleet size:        {n_gpus} GPUs")
    print(f"  Tasks per GPU:     {tasks_per_gpu}")
    print(f"  Total tasks:       {total_tasks:,}")
    print(f"  Strained:          {total_strained:,}")
    print(f"  Skipped:           {total_skipped:,}")
    print(f"  Strain ratio:      {strain_ratio:.1%}")
    print(f"  FLOPs saved:       {total_flops_saved:.2e}")
    print(f"  Elapsed:           {elapsed:.2f}s")
    print(f"  Throughput:        {summary['tasks_per_sec']:,.0f} tasks/s")
    print("=" * 70)

    return summary


def run_solver_comparison(
    n_features: int = 15,
    cfg: dict[str, Any] | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Compare all QUBO solvers on a synthetic feature selection problem.

    Prints results and returns a list of report dicts.
    """
    from qstrainer.qos.runner import QOSRunner
    from qstrainer.qos.scheduler import QOSScheduler
    from qstrainer.quantum.feature_selector import QUBOFeatureSelector

    # Build a synthetic dataset
    np.random.default_rng(seed)
    gen = SyntheticTelemetryGenerator(seed=seed)

    if n_features <= 15:
        # Use base features
        X_list, y_list = [], []
        for _ in range(300):
            f = gen.generate_healthy("GPU-CMP", "node-cmp")
            X_list.append(f.to_vector())
            y_list.append(0)
        for i in range(50):
            f = gen.generate_degrading("GPU-CMP", "node-cmp", severity=i / 49)
            X_list.append(f.to_vector())
            y_list.append(1)
        for _ in range(10):
            f = gen.generate_failing("GPU-CMP", "node-cmp")
            X_list.append(f.to_vector())
            y_list.append(1)
        X = np.vstack(X_list)
        y = np.array(y_list, dtype=np.float64)
    else:
        # Use extended features
        from qstrainer.features.derived import DerivedFeatureExtractor

        extractor = DerivedFeatureExtractor(window_size=10)
        X_list, y_list = [], []
        for _ in range(300):
            f = gen.generate_healthy("GPU-CMP", "node-cmp")
            X_list.append(extractor.extract("GPU-CMP", f.to_vector()))
            y_list.append(0)
        for i in range(50):
            f = gen.generate_degrading("GPU-CMP2", "node-cmp", severity=i / 49)
            X_list.append(extractor.extract("GPU-CMP2", f.to_vector()))
            y_list.append(1)
        for _ in range(10):
            f = gen.generate_failing("GPU-CMP3", "node-cmp")
            X_list.append(extractor.extract("GPU-CMP3", f.to_vector()))
            y_list.append(1)
        X = np.vstack(X_list)
        y = np.array(y_list, dtype=np.float64)

    # Build QUBO
    n_select = min(8, n_features // 2)
    selector = QUBOFeatureSelector(n_select=n_select, alpha=0.5)
    Q = selector.build_qubo(X, y)

    print("=" * 70)
    print(f"SOLVER COMPARISON — {Q.shape[0]}-variable QUBO")
    print("=" * 70)

    # Build scheduler with all solvers
    solvers_cfg = (cfg or {}).get("solvers", {})
    scheduler = QOSScheduler.from_config({"solvers": solvers_cfg})
    runner = QOSRunner(scheduler)

    # Compare
    solver_names = ["qaoa_sim", "sa_default", "sa_heavy", "mock_quantum"]
    if Q.shape[0] > 20:
        solver_names = [n for n in solver_names if n != "qaoa_sim"]

    reports = runner.compare_solvers(Q, solver_names=solver_names, expected_k=n_select)

    print(
        f"\n{'Solver':<20s} {'Type':<15s} {'Energy':>10s} {'Time':>10s} "
        f"{'Selected':>10s} {'Feasible':>10s}"
    )
    print("-" * 75)
    for r in reports:
        feas = "YES" if r.feasible else "NO"
        print(
            f"{r.solver_name:<20s} {r.solver_type:<15s} {r.energy:>10.4f} "
            f"{r.solve_time_s:>9.3f}s {r.selected_count:>10d} {feas:>10s}"
        )

    # Feature overlap
    solutions = {}
    for r in reports:
        sel = (
            frozenset(i for i, v in enumerate(r.solution) if v == 1)
            if r.solution is not None
            else frozenset()
        )
        solutions[r.solver_name] = sel

    names_list = list(solutions.keys())
    print("\nFeature Overlap (Jaccard similarity):")
    for i, n1 in enumerate(names_list):
        for j, n2 in enumerate(names_list):
            if j > i:
                s1, s2 = solutions[n1], solutions[n2]
                jaccard = len(s1 & s2) / max(len(s1 | s2), 1)
                print(f"  {n1} ^ {n2}: {len(s1 & s2)}/{len(s1 | s2)} (J={jaccard:.2f})")

    print("=" * 70)
    return [r.to_dict() for r in reports]
