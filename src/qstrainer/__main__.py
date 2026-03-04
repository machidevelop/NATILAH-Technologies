"""Q-Strainer CLI entry point."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

from qstrainer import __version__


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="qstrainer",
        description="Q-Strainer: Real-Time GPU Telemetry Filtering Engine",
    )
    parser.add_argument("--version", action="version", version=f"qstrainer {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    # --- agent ---
    p_agent = sub.add_parser("agent", help="Run the Q-Strainer telemetry agent")
    p_agent.add_argument(
        "-c", "--config", type=Path, default=Path("config/default.yaml"),
        help="Path to YAML config file",
    )
    p_agent.add_argument(
        "--dry-run", action="store_true",
        help="Use synthetic telemetry instead of NVML",
    )
    p_agent.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    p_agent.add_argument(
        "--json-logs", action="store_true",
        help="Output structured JSON log lines (for log aggregation)",
    )

    # --- benchmark ---
    p_bench = sub.add_parser("benchmark", help="Run benchmark suite on synthetic data")
    p_bench.add_argument("-n", "--num-gpus", type=int, default=100)
    p_bench.add_argument("-f", "--frames-per-gpu", type=int, default=100)
    p_bench.add_argument(
        "-c", "--config", type=Path, default=Path("config/default.yaml"),
    )

    # --- compare-solvers ---
    p_compare = sub.add_parser("compare-solvers", help="Compare QUBO solvers")
    p_compare.add_argument("-n", "--n-features", type=int, default=17)
    p_compare.add_argument(
        "-c", "--config", type=Path, default=Path("config/default.yaml"),
    )

    # --- checkpoint ---
    p_ckpt = sub.add_parser("checkpoint", help="Manage checkpoints")
    p_ckpt.add_argument("action", choices=["show", "verify", "clean"])
    p_ckpt.add_argument("--path", type=Path, default=Path("runs/"))

    args = parser.parse_args()

    from qstrainer.logging import setup_logging

    log_level = getattr(args, "log_level", "INFO")
    json_logs = getattr(args, "json_logs", False)
    setup_logging(level=log_level, json_output=json_logs)
    logger = logging.getLogger("qstrainer")

    if args.command == "agent":
        _run_agent(args, logger)
    elif args.command == "benchmark":
        _run_benchmark(args, logger)
    elif args.command == "compare-solvers":
        _run_compare_solvers(args, logger)
    elif args.command == "checkpoint":
        _run_checkpoint(args, logger)


def _run_agent(args: argparse.Namespace, logger: logging.Logger) -> None:
    from qstrainer.config import load_config
    from qstrainer.agent.daemon import QStrainerDaemon

    cfg = load_config(args.config)
    daemon = QStrainerDaemon(cfg, dry_run=args.dry_run)

    # Graceful shutdown
    loop = asyncio.new_event_loop()

    def _shutdown(sig: signal.Signals) -> None:
        logger.info(f"Received {sig.name}, shutting down...")
        daemon.request_shutdown()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown, sig)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            signal.signal(sig, lambda s, f: _shutdown(signal.Signals(s)))

    try:
        loop.run_until_complete(daemon.run())
    finally:
        loop.close()
        logger.info("Q-Strainer agent stopped.")


def _run_benchmark(args: argparse.Namespace, logger: logging.Logger) -> None:
    from qstrainer.config import load_config
    from qstrainer.benchmarks import run_fleet_benchmark

    cfg = load_config(args.config)
    run_fleet_benchmark(
        n_gpus=args.num_gpus,
        frames_per_gpu=args.frames_per_gpu,
        cfg=cfg,
    )


def _run_compare_solvers(args: argparse.Namespace, logger: logging.Logger) -> None:
    from qstrainer.config import load_config
    from qstrainer.benchmarks import run_solver_comparison

    cfg = load_config(args.config)
    run_solver_comparison(n_features=args.n_features, cfg=cfg)


def _run_checkpoint(args: argparse.Namespace, logger: logging.Logger) -> None:
    from qstrainer.checkpoint.persistence import CheckpointManager

    mgr = CheckpointManager(base_dir=args.path)
    if args.action == "show":
        mgr.show_checkpoints()
    elif args.action == "verify":
        mgr.verify_checkpoints()
    elif args.action == "clean":
        mgr.clean_old_checkpoints()


if __name__ == "__main__":
    main()
