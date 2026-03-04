"""Kafka emitter — publish StrainResult to a Kafka topic.

Requires: ``pip install confluent-kafka``
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from confluent_kafka import Producer as _KafkaProducer

    _HAS_KAFKA = True
except ImportError:
    _HAS_KAFKA = False


class KafkaEmitter:
    """Publish strained outputs to a Kafka topic.

    Supports:
      - Plaintext (default, for development)
      - SASL_SSL (SCRAM-SHA-256 / SCRAM-SHA-512 / PLAIN)
      - SSL (certificate-based auth)
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "qstrainer-alerts",
        *,
        security_protocol: str = "PLAINTEXT",
        sasl_mechanism: str | None = None,
        sasl_username: str | None = None,
        sasl_password: str | None = None,
        ssl_ca_location: str | None = None,
        ssl_certificate_location: str | None = None,
        ssl_key_location: str | None = None,
        ssl_key_password: str | None = None,
        **producer_kwargs: Any,
    ) -> None:
        if not _HAS_KAFKA:
            raise ImportError(
                "confluent-kafka is required. Install with: pip install confluent-kafka"
            )

        self._topic = topic
        self._producer: _KafkaProducer | None = None

        # Build config
        config = {"bootstrap.servers": bootstrap_servers}
        config["security.protocol"] = security_protocol

        # SASL auth
        if sasl_mechanism:
            config["sasl.mechanism"] = sasl_mechanism
        if sasl_username:
            config["sasl.username"] = sasl_username
        if sasl_password:
            config["sasl.password"] = sasl_password

        # SSL/TLS
        if ssl_ca_location:
            config["ssl.ca.location"] = ssl_ca_location
        if ssl_certificate_location:
            config["ssl.certificate.location"] = ssl_certificate_location
        if ssl_key_location:
            config["ssl.key.location"] = ssl_key_location
        if ssl_key_password:
            config["ssl.key.password"] = ssl_key_password

        config.update(producer_kwargs)
        self._config = config

    def _ensure_producer(self) -> None:
        if self._producer is None:
            self._producer = _KafkaProducer(self._config)
            logger.info("Kafka producer connected to %s", self._config["bootstrap.servers"])

    def _delivery_callback(self, err: Any, msg: Any) -> None:
        if err is not None:
            logger.error("Kafka delivery failed: %s", err)

    def emit(self, result: Any) -> None:
        """Publish a strain result as JSON to the configured Kafka topic."""
        self._ensure_producer()
        assert self._producer is not None

        payload = {
            "gpu_id": getattr(result, "gpu_id", "unknown"),
            "timestamp": getattr(result, "timestamp", 0),
            "verdict": (
                result.verdict.name if hasattr(result.verdict, "name") else str(result.verdict)
            ),
            "redundancy_score": result.redundancy_score,
            "convergence_score": result.convergence_score,
            "confidence": result.confidence,
            "decisions": [
                {
                    "verdict": d.verdict.name if hasattr(d.verdict, "name") else str(d.verdict),
                    "metric": d.metric,
                    "reason": d.reason,
                }
                for d in result.decisions
            ],
        }

        gpu_id = getattr(result, "gpu_id", "unknown")
        self._producer.produce(
            self._topic,
            key=gpu_id.encode(),
            value=json.dumps(payload, default=str).encode(),
            callback=self._delivery_callback,
        )
        self._producer.poll(0)  # trigger delivery callbacks

    def flush(self, timeout: float = 5.0) -> None:
        """Flush pending messages."""
        if self._producer is not None:
            self._producer.flush(timeout)

    def close(self) -> None:
        if self._producer is not None:
            self._producer.flush(10.0)
            logger.info("Kafka producer flushed and closed")
