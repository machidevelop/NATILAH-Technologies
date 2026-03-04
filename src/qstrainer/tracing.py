"""OpenTelemetry tracing for Q-Strainer pipeline stages.

Usage::

    from qstrainer.tracing import init_tracing, trace_stage
    init_tracing(service_name="qstrainer", endpoint="http://localhost:4317")

    with trace_stage("threshold_check", gpu_id="GPU-0001"):
        alerts = threshold.check(frame)

If OpenTelemetry SDK is not installed, all operations are no-ops.

Requires: ``pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp``
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

_tracer = None
_NOOP = True

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False


def init_tracing(
    service_name: str = "qstrainer",
    endpoint: str | None = None,
    console: bool = False,
) -> None:
    """Initialize OpenTelemetry tracing.

    Parameters
    ----------
    service_name : str
        Service name reported in traces.
    endpoint : str, optional
        OTLP gRPC endpoint (e.g. ``http://localhost:4317``).
        If None, traces are exported to console (if ``console=True``) or dropped.
    console : bool
        If True, export spans to stderr (for development).
    """
    global _tracer, _NOOP

    if not _HAS_OTEL:
        logger.info("OpenTelemetry SDK not installed — tracing disabled")
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(endpoint=endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info("OTLP trace exporter → %s", endpoint)
        except ImportError:
            logger.warning(
                "opentelemetry-exporter-otlp not installed — falling back to console exporter"
            )
            if console:
                provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    elif console:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("qstrainer", "0.3.0")
    _NOOP = False
    logger.info("OpenTelemetry tracing initialized (service=%s)", service_name)


@contextmanager
def trace_stage(
    name: str,
    attributes: dict[str, Any] | None = None,
    **extra_attrs: Any,
) -> Generator[Any, None, None]:
    """Context manager to trace a pipeline stage.

    Usage::

        with trace_stage("threshold_check", gpu_id="GPU-0001"):
            alerts = threshold.check(frame)

    If tracing is not initialized, this is a zero-cost no-op.
    """
    if _NOOP or _tracer is None:
        yield None
        return

    merged = {}
    if attributes:
        merged.update(attributes)
    merged.update(extra_attrs)

    with _tracer.start_as_current_span(name, attributes=merged) as span:
        yield span


def add_span_event(name: str, attributes: dict[str, Any] | None = None) -> None:
    """Add an event to the current span (if tracing is active)."""
    if _NOOP:
        return

    span = trace.get_current_span()
    if span and span.is_recording():
        span.add_event(name, attributes=attributes or {})


def record_exception(exc: Exception) -> None:
    """Record an exception on the current span."""
    if _NOOP:
        return

    span = trace.get_current_span()
    if span and span.is_recording():
        span.set_status(trace.StatusCode.ERROR, str(exc))
        span.record_exception(exc)
