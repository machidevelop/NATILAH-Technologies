"""Configuration loader — YAML + environment variable overrides.

Usage::

    from qstrainer.config import load_config
    cfg = load_config("config/default.yaml")

Environment variable overrides follow the pattern::

    QSTRAINER_AGENT__POLL_HZ=20       →  cfg["agent"]["poll_hz"] = 20.0
    QSTRAINER_PIPELINE__STRAIN_THRESHOLD=0.5
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

_ENV_PREFIX = "QSTRAINER_"
_SEPARATOR = "__"


def load_config(path: Path | str | None = None) -> Dict[str, Any]:
    """Load configuration from YAML, then overlay environment variables.

    Parameters
    ----------
    path : Path or str, optional
        Path to a YAML config file.  If ``None`` or the file does not exist,
        returns a default config with env overrides applied.
    """
    cfg = _defaults()

    if path is not None:
        p = Path(path)
        if p.exists():
            cfg = _deep_merge(cfg, _load_yaml(p))
            logger.info("Loaded config from %s", p)
        else:
            logger.warning("Config file %s not found — using defaults", p)

    _apply_env_overrides(cfg)

    # Resolve secret references (env://, file://, sops://, vault://)
    from qstrainer.secrets import resolve_secrets  # noqa: E402
    resolve_secrets(cfg)

    return cfg


def _defaults() -> Dict[str, Any]:
    """Sensible defaults for every config section."""
    return {
        "agent": {
            "node_id": "node-00",
            "poll_hz": 10.0,
            "checkpoint_interval_tasks": 10_000,
        },
        "synthetic": {
            "seed": 42,
            "n_gpus": 8,
        },
        "pipeline": {
            "strain_threshold": 0.5,
            "heartbeat_interval": 200,
            "redundancy": {
                "gradient": {
                    "floor": 1e-7,
                    "low": 1e-5,
                },
                "loss": {
                    "floor": 1e-8,
                    "low": 1e-5,
                },
                "convergence": {
                    "threshold": 0.95,
                    "warn": 0.85,
                },
                "data": {
                    "threshold": 0.98,
                    "warn": 0.90,
                },
                "param_update_floor": 1e-9,
            },
            "convergence": {
                "z_threshold": 3.0,
                "min_samples": 20,
            },
            "ml_predictor": {
                "kernel": "rbf",
                "nu": 0.05,
            },
        },
        "features": {
            "window_size": 10,
            "n_select": 8,
            "alpha": 0.5,
        },
        "solvers": {
            "sa": {
                "num_reads": 300,
                "num_sweeps": 1500,
                "num_reads_heavy": 1000,
                "num_sweeps_heavy": 3000,
            },
            "qaoa": {
                "p": 2,
                "n_restarts": 6,
                "seed": 42,
            },
            "dwave": {
                "num_reads": 500,
            },
        },
        "emitters": {
            "prometheus": {
                "enabled": False,
                "port": 9090,
            },
            "grpc": {
                "enabled": False,
                "target": "localhost:50051",
            },
            "kafka": {
                "enabled": False,
                "bootstrap_servers": "localhost:9092",
                "topic": "qstrainer-alerts",
            },
        },
        "checkpoint": {
            "enabled": True,
            "dir": "runs/",
            "max_checkpoints": 5,
        },
    }


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Parse a YAML file.  Falls back to empty dict on error."""
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML not installed — cannot load %s", path)
        return {}

    with open(path) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge *override* into *base* (override wins)."""
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _apply_env_overrides(cfg: Dict[str, Any]) -> None:
    """Override config values with ``QSTRAINER_SECTION__KEY`` env vars."""
    for key, val in os.environ.items():
        if not key.startswith(_ENV_PREFIX):
            continue
        parts = key[len(_ENV_PREFIX):].lower().split(_SEPARATOR)
        _set_nested(cfg, parts, _coerce(val))


def _set_nested(d: Dict, keys: list, value: Any) -> None:
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _coerce(val: str) -> Any:
    """Attempt to cast env var string to int / float / bool."""
    if val.lower() in ("true", "yes", "1"):
        return True
    if val.lower() in ("false", "no", "0"):
        return False
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val
