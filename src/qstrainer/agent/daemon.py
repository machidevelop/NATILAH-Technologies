"""Q-Strainer Agent Daemon — continuous ingest → strain → emit loop.

Runs as a long-lived asyncio service.  In production, started via::

    qstrainer agent -c config/default.yaml
    qstrainer agent --dry-run   # synthetic telemetry

Architecture:
    ┌────────────┐     ┌──────────┐     ┌──────────┐
    │ Ingestor   │ ──→ │ QStrainer│ ──→ │ Emitter  │
    │ (NVML/Syn) │     │ Pipeline │     │ (Prom/gRPC)
    └────────────┘     └──────────┘     └──────────┘
         ↑                                    │
         │            ┌──────────┐            │
         └────────────│ Checkpoint│←───────────┘
                      └──────────┘
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class QStrainerDaemon:
    """Asyncio-based daemon that continuously strains GPU telemetry.

    Parameters
    ----------
    cfg : dict
        Merged configuration (from ``load_config``).
    dry_run : bool
        If True, use synthetic telemetry instead of NVML.
    """

    def __init__(self, cfg: Dict[str, Any], *, dry_run: bool = False) -> None:
        self._cfg = cfg
        self._dry_run = dry_run
        self._shutdown_event = asyncio.Event()
        self._poll_hz: float = cfg.get("agent", {}).get("poll_hz", 10.0)
        self._checkpoint_interval: int = cfg.get("agent", {}).get(
            "checkpoint_interval_tasks", 10_000
        )
        self._total_tasks = 0

    # ── Public API ──────────────────────────────────────────
    def request_shutdown(self) -> None:
        """Signal the daemon to stop gracefully."""
        self._shutdown_event.set()

    async def run(self) -> None:
        """Main entry point — runs until shutdown is requested."""
        from qstrainer.logging import set_correlation_id, set_context
        from qstrainer.profiling import MemoryProfiler

        set_correlation_id()  # unique ID for this agent run
        logger.info("Q-Strainer agent starting (dry_run=%s)", self._dry_run)

        # ---- Build components from config ----
        ingestor = self._build_ingestor()
        pipeline = self._build_pipeline()
        emitters = self._build_emitters()
        checkpoint_mgr = self._build_checkpoint_mgr()

        # ---- Memory profiler ----
        profiler = MemoryProfiler(
            snapshot_interval=int(self._poll_hz * 60),  # ~once per minute
            tracemalloc_enabled=self._cfg.get("agent", {}).get(
                "tracemalloc", False
            ),
        )
        profiler.start()

        # ---- Restore from checkpoint if available ----
        if checkpoint_mgr is not None:
            checkpoint_mgr.try_restore(pipeline)

        interval = 1.0 / self._poll_hz
        logger.info(
            "Agent loop: %.1f Hz (%d ms per tick), %d emitter(s)",
            self._poll_hz, interval * 1000, len(emitters),
        )

        try:
            while not self._shutdown_event.is_set():
                t0 = time.perf_counter()

                # 1. Ingest
                tasks = ingestor.poll()

                # 2. Strain
                for task in tasks:
                    result = pipeline.process_task(task)
                    self._total_tasks += 1

                    # Always emit (process_task always returns a StrainResult)
                    for emitter in emitters:
                        try:
                            emitter.emit(result)
                        except Exception:
                            logger.exception("Emitter %s failed", type(emitter).__name__)

                    # Memory profiling tick
                    profiler.tick()

                # 4. Periodic checkpoint
                if (
                    checkpoint_mgr is not None
                    and self._total_tasks % self._checkpoint_interval == 0
                    and self._total_tasks > 0
                ):
                    checkpoint_mgr.save(pipeline)

                # 5. Sleep to maintain target Hz
                elapsed = time.perf_counter() - t0
                sleep_for = max(0.0, interval - elapsed)
                if sleep_for > 0:
                    try:
                        await asyncio.wait_for(
                            self._shutdown_event.wait(), timeout=sleep_for
                        )
                    except asyncio.TimeoutError:
                        pass
        finally:
            # Final checkpoint on shutdown
            if checkpoint_mgr is not None:
                checkpoint_mgr.save(pipeline)
            profiler.stop()
            logger.info("\n%s", profiler.report())
            self._teardown(ingestor, emitters)
            logger.info(
                "Agent stopped. Total tasks processed: %d", self._total_tasks
            )

    # ── Component Builders ──────────────────────────────────
    def _build_ingestor(self):
        """Build the telemetry ingestor (NVML or synthetic)."""
        if self._dry_run:
            from qstrainer.ingestion.synthetic import SyntheticTelemetryGenerator

            seed = self._cfg.get("synthetic", {}).get("seed", 42)
            n_gpus = self._cfg.get("synthetic", {}).get("n_gpus", 8)
            gen = SyntheticTelemetryGenerator(seed=seed)
            return _SyntheticIngestorAdapter(gen, n_gpus=n_gpus)
        else:
            from qstrainer.ingestion.nvml import NVMLIngestor

            node_id = self._cfg.get("agent", {}).get("node_id", "node-00")
            ingestor = NVMLIngestor(node_id=node_id)
            ingestor.init()
            return ingestor

    def _build_pipeline(self):
        """Build the QStrainer pipeline from config."""
        from qstrainer.pipeline.strainer import QStrainer

        pipeline_cfg = self._cfg.get("pipeline", {})
        return QStrainer.from_config(pipeline_cfg)

    def _build_emitters(self):
        """Build configured emitters."""
        emitters = []
        emit_cfg = self._cfg.get("emitters", {})

        if emit_cfg.get("prometheus", {}).get("enabled", False):
            try:
                from qstrainer.emission.prometheus import PrometheusEmitter

                prom_cfg = emit_cfg["prometheus"]
                emitters.append(PrometheusEmitter(port=prom_cfg.get("port", 9090)))
            except ImportError:
                logger.warning("prometheus-client not installed, skipping Prometheus emitter")

        if emit_cfg.get("grpc", {}).get("enabled", False):
            try:
                from qstrainer.emission.grpc_emitter import GRPCEmitter

                grpc_cfg = emit_cfg["grpc"]
                emitters.append(GRPCEmitter(target=grpc_cfg.get("target", "localhost:50051")))
            except ImportError:
                logger.warning("grpcio not installed, skipping gRPC emitter")

        if emit_cfg.get("kafka", {}).get("enabled", False):
            try:
                from qstrainer.emission.kafka_emitter import KafkaEmitter

                kafka_cfg = emit_cfg["kafka"]
                emitters.append(KafkaEmitter(
                    bootstrap_servers=kafka_cfg.get("bootstrap_servers", "localhost:9092"),
                    topic=kafka_cfg.get("topic", "qstrainer-alerts"),
                ))
            except ImportError:
                logger.warning("confluent-kafka not installed, skipping Kafka emitter")

        # Always include a log emitter
        emitters.append(_LogEmitter())
        return emitters

    def _build_checkpoint_mgr(self):
        """Build checkpoint manager if configured."""
        ckpt_cfg = self._cfg.get("checkpoint", {})
        if not ckpt_cfg.get("enabled", True):
            return None
        from qstrainer.checkpoint.persistence import CheckpointManager

        return CheckpointManager(
            base_dir=ckpt_cfg.get("dir", "runs/"),
            max_checkpoints=ckpt_cfg.get("max_checkpoints", 5),
        )

    def _teardown(self, ingestor, emitters):
        """Clean shutdown of components."""
        if hasattr(ingestor, "shutdown"):
            try:
                ingestor.shutdown()
            except Exception:
                logger.exception("Ingestor shutdown failed")
        for em in emitters:
            if hasattr(em, "close"):
                try:
                    em.close()
                except Exception:
                    logger.exception("Emitter close failed")


# ── Adapters ────────────────────────────────────────────────

class _SyntheticIngestorAdapter:
    """Wraps SyntheticTelemetryGenerator to expose the same poll() interface."""

    def __init__(self, gen, *, n_gpus: int = 8):
        from qstrainer.ingestion.synthetic import SyntheticTelemetryGenerator

        self._gen: SyntheticTelemetryGenerator = gen
        self._n_gpus = n_gpus
        self._frame_idx = 0
        self._rng = gen.rng

    def poll(self):
        frames = []
        for gpu_idx in range(self._n_gpus):
            gpu_id = f"GPU-SYN-{gpu_idx:04d}"
            node_id = "node-synthetic"
            r = self._rng.random()
            if r < 0.02:
                frame = self._gen.generate_failing(gpu_id, node_id)
            elif r < 0.07:
                severity = self._rng.random()
                frame = self._gen.generate_degrading(gpu_id, node_id, severity)
            else:
                frame = self._gen.generate_healthy(gpu_id, node_id)
            frames.append(frame)
        self._frame_idx += 1
        return frames

    def shutdown(self):
        pass


class _LogEmitter:
    """Default emitter: logs strain results at INFO level."""

    def emit(self, result) -> None:
        if result.decisions:
            for d in result.decisions:
                logger.warning(
                    "STRAIN [%s] %s task=%s: %s",
                    d.verdict.name if hasattr(d.verdict, "name") else d.verdict,
                    d.action.name if hasattr(d.action, "name") else d.action,
                    d.task_id,
                    d.reason,
                )
        elif result.redundancy_score > 0.5:
            logger.info(
                "REDUNDANT verdict=%s redundancy=%.3f confidence=%.3f",
                result.verdict.name if hasattr(result.verdict, "name") else result.verdict,
                result.redundancy_score,
                result.confidence,
            )

    def close(self):
        pass
