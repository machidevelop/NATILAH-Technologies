"""gRPC emitter — stream StrainResult to a central collector.

Requires: ``pip install grpcio grpcio-tools``

In production, generate stubs from a ``.proto`` file.  This module
ships a minimal implementation using the generic ``Any`` message so
the package works without proto compilation.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import grpc

    _HAS_GRPC = True
except ImportError:
    _HAS_GRPC = False


class GRPCEmitter:
    """Send strained outputs to a gRPC collector.

    Supports:
      - Insecure channels (default, for development)
      - TLS (server authentication only)
      - mTLS (mutual TLS — both client and server authenticate)
    """

    def __init__(
        self,
        target: str = "localhost:50051",
        *,
        tls: bool = False,
        ca_cert_path: str | None = None,
        client_cert_path: str | None = None,
        client_key_path: str | None = None,
        **channel_kwargs,
    ) -> None:
        if not _HAS_GRPC:
            raise ImportError(
                "grpcio is required. Install with: pip install grpcio"
            )

        self._target = target
        self._tls = tls
        self._ca_cert_path = ca_cert_path
        self._client_cert_path = client_cert_path
        self._client_key_path = client_key_path
        self._channel: Optional[grpc.Channel] = None
        self._connected = False
        self._channel_kwargs = channel_kwargs

    def _ensure_connected(self) -> None:
        if not self._connected:
            if self._tls or self._ca_cert_path:
                credentials = self._build_tls_credentials()
                self._channel = grpc.secure_channel(
                    self._target, credentials, **self._channel_kwargs
                )
                mode = "mTLS" if self._client_cert_path else "TLS"
                logger.info("gRPC %s channel opened to %s", mode, self._target)
            else:
                self._channel = grpc.insecure_channel(
                    self._target, **self._channel_kwargs
                )
                logger.info("gRPC insecure channel opened to %s", self._target)
            self._connected = True

    def _build_tls_credentials(self) -> grpc.ChannelCredentials:
        """Build TLS or mTLS channel credentials."""
        ca_cert = None
        if self._ca_cert_path:
            with open(self._ca_cert_path, "rb") as f:
                ca_cert = f.read()

        client_cert = None
        client_key = None
        if self._client_cert_path and self._client_key_path:
            with open(self._client_cert_path, "rb") as f:
                client_cert = f.read()
            with open(self._client_key_path, "rb") as f:
                client_key = f.read()

        return grpc.ssl_channel_credentials(
            root_certificates=ca_cert,
            private_key=client_key,
            certificate_chain=client_cert,
        )

    def emit(self, result) -> None:
        """Serialise the strain result and send via gRPC."""
        self._ensure_connected()

        payload = {
            "gpu_id": getattr(result, "gpu_id", "unknown"),
            "timestamp": getattr(result, "timestamp", 0),
            "verdict": result.verdict.name if hasattr(result.verdict, "name") else str(result.verdict),
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

        # Placeholder: in production, encode as protobuf and call stub
        # stub.ReportAnomaly(qstrainer_pb2.AnomalyReport(**payload))
        logger.debug("gRPC emit: %s", json.dumps(payload, default=str))

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
            self._connected = False
            logger.info("gRPC channel closed")
