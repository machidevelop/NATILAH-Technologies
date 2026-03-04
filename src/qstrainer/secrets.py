"""Secret management for Q-Strainer configuration.

Supports multiple secret backends:
  - Environment variables (default, always available)
  - File references (``file:///path/to/secret``)
  - SOPS-encrypted YAML (``sops://path/to/encrypted.yaml#key``)
  - HashiCorp Vault (``vault://secret/data/qstrainer#key``)

Config values referencing secrets use the pattern::

    emitters:
      kafka:
        sasl_password: "env://KAFKA_PASSWORD"
        ssl_key: "file:///etc/qstrainer/tls/client.key"

Usage::

    from qstrainer.secrets import resolve_secrets
    cfg = load_config("config/default.yaml")
    resolve_secrets(cfg)  # replaces secret references in-place
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# ── Secret Resolvers ────────────────────────────────────────


def _resolve_env(ref: str) -> str:
    """Resolve ``env://VAR_NAME`` → os.environ[VAR_NAME]."""
    var_name = ref[len("env://"):]
    value = os.environ.get(var_name)
    if value is None:
        raise KeyError(f"Secret env var {var_name!r} not set")
    return value


def _resolve_file(ref: str) -> str:
    """Resolve ``file:///path/to/file`` → file contents (stripped)."""
    path_str = ref[len("file://"):]
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Secret file not found: {p}")
    return p.read_text().strip()


def _resolve_sops(ref: str) -> str:
    """Resolve ``sops://path/to/encrypted.yaml#key`` → decrypted value.

    Requires ``sops`` CLI to be installed.
    """
    # Parse: sops://path#key.subkey
    rest = ref[len("sops://"):]
    if "#" not in rest:
        raise ValueError(f"SOPS ref must include #key: {ref}")
    path_str, key_path = rest.split("#", 1)

    try:
        result = subprocess.run(
            ["sops", "--decrypt", "--extract", f'["{key_path}"]', path_str],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return result.stdout.strip()
    except FileNotFoundError:
        raise RuntimeError("sops CLI not found — install from https://github.com/getsops/sops")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"sops decryption failed: {e.stderr}")


def _resolve_vault(ref: str) -> str:
    """Resolve ``vault://secret/data/path#key`` → Vault KV value.

    Uses the ``vault`` CLI or VAULT_TOKEN + VAULT_ADDR env vars.
    Requires ``vault`` CLI to be installed.
    """
    rest = ref[len("vault://"):]
    if "#" not in rest:
        raise ValueError(f"Vault ref must include #key: {ref}")
    secret_path, key = rest.split("#", 1)

    try:
        result = subprocess.run(
            ["vault", "kv", "get", f"-field={key}", secret_path],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return result.stdout.strip()
    except FileNotFoundError:
        raise RuntimeError(
            "vault CLI not found — install from https://www.vaultproject.io/downloads"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Vault read failed: {e.stderr}")


_RESOLVERS = {
    "env://": _resolve_env,
    "file://": _resolve_file,
    "sops://": _resolve_sops,
    "vault://": _resolve_vault,
}


def resolve_value(value: str) -> str:
    """Resolve a single secret reference string.

    If the value doesn't start with a known scheme, returns it unchanged.
    """
    for scheme, resolver in _RESOLVERS.items():
        if value.startswith(scheme):
            resolved = resolver(value)
            logger.debug("Resolved secret %s → [%d chars]", scheme + "***", len(resolved))
            return resolved
    return value


def resolve_secrets(cfg: Dict[str, Any]) -> None:
    """Walk the config dict and resolve all secret references in-place.

    Modifies ``cfg`` by replacing string values that start with a
    secret scheme (``env://``, ``file://``, ``sops://``, ``vault://``)
    with their resolved values.
    """
    _walk_and_resolve(cfg)


def _walk_and_resolve(d: Dict[str, Any]) -> None:
    for key, value in d.items():
        if isinstance(value, dict):
            _walk_and_resolve(value)
        elif isinstance(value, str):
            for scheme in _RESOLVERS:
                if value.startswith(scheme):
                    try:
                        d[key] = resolve_value(value)
                    except Exception as e:
                        logger.error("Failed to resolve secret %s.%s: %s", key, scheme, e)
                    break
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, str):
                    for scheme in _RESOLVERS:
                        if item.startswith(scheme):
                            try:
                                value[i] = resolve_value(item)
                            except Exception as e:
                                logger.error("Failed to resolve secret in list: %s", e)
                            break
                elif isinstance(item, dict):
                    _walk_and_resolve(item)
