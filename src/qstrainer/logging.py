"""Structured JSON logging for Q-Strainer.

Usage::

    from qstrainer.logging import setup_logging
    setup_logging(level="INFO", json_output=True)

Produces machine-parseable JSON lines for log aggregation (ELK, Loki, Splunk).
Falls back to human-readable format when ``json_output=False``.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

# ── Context Vars (correlation across async tasks) ───────────
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")
_gpu_id: ContextVar[str] = ContextVar("gpu_id", default="")
_node_id: ContextVar[str] = ContextVar("node_id", default="")


def set_correlation_id(cid: Optional[str] = None) -> str:
    """Set (or generate) a correlation ID for the current context."""
    cid = cid or uuid.uuid4().hex[:12]
    _correlation_id.set(cid)
    return cid


def set_context(*, gpu_id: str = "", node_id: str = "") -> None:
    """Set GPU/node context for log enrichment."""
    if gpu_id:
        _gpu_id.set(gpu_id)
    if node_id:
        _node_id.set(node_id)


def get_correlation_id() -> str:
    return _correlation_id.get()


# ── JSON Formatter ──────────────────────────────────────────

class JSONFormatter(logging.Formatter):
    """Emit log records as JSON lines.

    Output format::

        {"ts":"2026-03-04T12:00:00.123Z","level":"INFO","logger":"qstrainer.pipeline",
         "msg":"Frame emitted","correlation_id":"a1b2c3d4e5f6","gpu_id":"GPU-0001",
         "node_id":"node-01","extra":{}}
    """

    def format(self, record: logging.LogRecord) -> str:
        entry: Dict[str, Any] = {
            "ts": self._iso_time(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Inject context vars
        cid = _correlation_id.get()
        if cid:
            entry["correlation_id"] = cid
        gpu = _gpu_id.get()
        if gpu:
            entry["gpu_id"] = gpu
        node = _node_id.get()
        if node:
            entry["node_id"] = node

        # Extra fields passed via `logger.info("msg", extra={...})`
        extra = {}
        for k, v in record.__dict__.items():
            if k.startswith("_qs_"):
                extra[k[4:]] = v
        if extra:
            entry["extra"] = extra

        # Exception info
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(entry, default=str, separators=(",", ":"))

    @staticmethod
    def _iso_time(record: logging.LogRecord) -> str:
        t = time.gmtime(record.created)
        return time.strftime("%Y-%m-%dT%H:%M:%S", t) + f".{int(record.msecs):03d}Z"


# ── Human-Readable Formatter ───────────────────────────────

class HumanFormatter(logging.Formatter):
    """Coloured, human-readable log output for development."""

    COLORS = {
        "DEBUG": "\033[36m",     # cyan
        "INFO": "\033[32m",      # green
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[41m",  # red bg
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        cid = _correlation_id.get()
        cid_str = f" [{cid}]" if cid else ""
        gpu = _gpu_id.get()
        gpu_str = f" gpu={gpu}" if gpu else ""

        msg = record.getMessage()
        base = (
            f"{color}{record.levelname:8s}{self.RESET} "
            f"{record.name}{cid_str}{gpu_str} | {msg}"
        )

        if record.exc_info and record.exc_info[1]:
            base += "\n" + "".join(traceback.format_exception(*record.exc_info))

        return base


# ── Setup ───────────────────────────────────────────────────

def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
    stream=None,
) -> None:
    """Configure logging for the entire ``qstrainer`` namespace.

    Parameters
    ----------
    level : str
        Log level (DEBUG, INFO, WARNING, ERROR).
    json_output : bool
        If True, emit JSON lines. If False, coloured human-readable.
    stream :
        Output stream. Defaults to ``sys.stderr``.
    """
    root = logging.getLogger("qstrainer")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    root.handlers.clear()

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JSONFormatter() if json_output else HumanFormatter())
    root.addHandler(handler)

    # Don't propagate to root logger
    root.propagate = False
