"""Prometheus metrics emitter for Q-Strainer.

Exposes an HTTP endpoint (default :9090/metrics) with:
  - qstrainer_tasks_total             (counter)
  - qstrainer_strained_total          (counter)
  - qstrainer_decisions_total         (counter, by verdict)
  - qstrainer_redundancy_score        (gauge, by gpu_id)
  - qstrainer_process_seconds         (histogram)

Requires: ``pip install prometheus-client``
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        start_http_server,
    )

    _HAS_PROM = True
except ImportError:
    _HAS_PROM = False


class PrometheusEmitter:
    """Expose Q-Strainer metrics for Prometheus scraping."""

    def __init__(self, port: int = 9090, *, start_server: bool = True) -> None:
        if not _HAS_PROM:
            raise ImportError(
                "prometheus-client is required. "
                "Install with: pip install prometheus-client"
            )

        self.port = port
        self._started = False

        # Metrics
        self._tasks_total = Counter(
            "qstrainer_tasks_total",
            "Total compute tasks processed",
        )
        self._strained_total = Counter(
            "qstrainer_strained_total",
            "Total tasks strained (non-EXECUTE verdict)",
        )
        self._decisions_total = Counter(
            "qstrainer_decisions_total",
            "Total strain decisions by verdict",
            ["verdict"],
        )
        self._redundancy_score = Gauge(
            "qstrainer_redundancy_score",
            "Latest redundancy score per GPU",
            ["gpu_id"],
        )
        self._verdict_state = Gauge(
            "qstrainer_verdict_state",
            "Verdict state (0=EXECUTE, 1=APPROXIMATE, 2=DEFER, 3=SKIP)",
            ["gpu_id"],
        )
        self._process_latency = Histogram(
            "qstrainer_process_seconds",
            "Pipeline processing latency per frame",
            buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1),
        )

        if start_server:
            self._start_server()

    def _start_server(self) -> None:
        if not self._started:
            start_http_server(self.port)
            self._started = True
            logger.info("Prometheus metrics server on :%d/metrics", self.port)

    def emit(self, result) -> None:
        """Record a StrainResult."""
        self._strained_total.inc()

        gpu_id = getattr(result, "gpu_id", "unknown")

        self._redundancy_score.labels(gpu_id=gpu_id).set(
            result.redundancy_score
        )

        verdict_val = result.verdict.value if hasattr(result.verdict, "value") else 0
        self._verdict_state.labels(gpu_id=gpu_id).set(verdict_val)

        verdict_name = result.verdict.name if hasattr(result.verdict, "name") else str(result.verdict)
        self._decisions_total.labels(verdict=verdict_name).inc()

    def record_task(self, latency_s: float) -> None:
        """Record a single task processing."""
        self._tasks_total.inc()
        self._process_latency.observe(latency_s)

    def close(self) -> None:
        pass  # prometheus_client doesn't require cleanup
