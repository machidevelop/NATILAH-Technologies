"""Feature store — centralised feature computation, caching and serving.

Provides a single interface that all pipeline stages and ML models
use to request features.  The store handles:

  - **Registration**: declare named features with their extraction logic.
  - **Caching**: avoid recomputing features within a single task or
    across sequential tasks for the same GPU.
  - **Materialisation**: bulk-compute features into NumPy matrices for
    training.
  - **Schema**: validate that feature shapes match expectations.

Usage::

    from qstrainer.ml.feature_store import FeatureStore
    fs = FeatureStore()
    fs.register("base", lambda task: task.to_vector(), dim=15)
    fs.register("derived", extractor.extract, dim=60, depends=["base"])
    vec = fs.get("derived", task)

For a Redis-backed distributed feature store, wrap ``FeatureStore``
with ``RedisFeatureCache``.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from qstrainer.models.frame import ComputeTask

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FeatureSpec:
    """Definition of one named feature set."""
    name: str
    extractor: Callable  # (frame) → np.ndarray  OR  (gpu_id, base_vec) → np.ndarray
    dim: int             # expected output dimension
    depends: List[str] = field(default_factory=list)
    description: str = ""
    # arity: 1 = (frame,)  2 = (gpu_id, vec)
    arity: int = 1


class FeatureStore:
    """Centralised feature computation with per-task caching.

    Parameters
    ----------
    cache_size : int
        Number of (gpu_id, task_ts) results to cache per feature set.
    """

    def __init__(self, cache_size: int = 256) -> None:
        self._specs: Dict[str, FeatureSpec] = {}
        self._cache_size = cache_size
        # Cache: (feature_name, gpu_id, timestamp) → np.ndarray
        self._cache: OrderedDict[Tuple[str, str, float], np.ndarray] = OrderedDict()

    # ── Registration ─────────────────────────────────────────

    def register(
        self,
        name: str,
        extractor: Callable,
        dim: int,
        *,
        depends: Optional[List[str]] = None,
        description: str = "",
        arity: int = 1,
    ) -> None:
        """Register a named feature set.

        Parameters
        ----------
        name : str
            Unique feature set name.
        extractor : callable
            For *arity=1*: ``extractor(frame) → np.ndarray``.
            For *arity=2*: ``extractor(gpu_id, base_vector) → np.ndarray``.
        dim : int
            Expected output dimension.
        depends : list of str, optional
            Names of feature sets that must be computed first.
        arity : int
            1 or 2 — number of arguments the extractor expects.
        """
        if name in self._specs:
            logger.warning("Feature %r re-registered (overwriting)", name)

        self._specs[name] = FeatureSpec(
            name=name,
            extractor=extractor,
            dim=dim,
            depends=depends or [],
            description=description,
            arity=arity,
        )

    # ── Compute features ─────────────────────────────────────

    def get(self, name: str, task: ComputeTask) -> np.ndarray:
        """Compute (or retrieve cached) features for a task.

        Resolves dependencies automatically: if feature ``"derived"``
        depends on ``"base"``, computing ``"derived"`` will first
        compute ``"base"`` and pass its output.
        """
        spec = self._specs.get(name)
        if spec is None:
            raise KeyError(f"Feature {name!r} not registered")

        cache_key = (name, task.gpu_id, task.timestamp)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Resolve dependencies
        for dep_name in spec.depends:
            self.get(dep_name, task)  # ensures dep is in cache

        # Compute
        if spec.arity == 2:
            # Needs (gpu_id, base_vector)
            if spec.depends:
                base_key = (spec.depends[0], task.gpu_id, task.timestamp)
                base_vec = self._cache.get(base_key)
                if base_vec is None:
                    base_vec = self.get(spec.depends[0], task)
            else:
                base_vec = task.to_vector()
            result = spec.extractor(task.gpu_id, base_vec)
        else:
            result = spec.extractor(task)

        # Validate dimension
        if result.shape[0] != spec.dim:
            logger.warning(
                "Feature %r dimension mismatch: expected %d, got %d",
                name, spec.dim, result.shape[0],
            )

        # Cache
        self._cache[cache_key] = result
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

        return result

    def get_multi(
        self, names: List[str], task: ComputeTask,
    ) -> np.ndarray:
        """Compute multiple feature sets and concatenate them."""
        parts = [self.get(n, task) for n in names]
        return np.concatenate(parts)

    # ── Materialisation ──────────────────────────────────────

    def materialise(
        self,
        name: str,
        tasks: Sequence[ComputeTask],
    ) -> np.ndarray:
        """Compute a feature matrix for a batch of tasks.

        Returns shape ``(len(tasks), dim)``.
        """
        spec = self._specs.get(name)
        if spec is None:
            raise KeyError(f"Feature {name!r} not registered")
        rows = [self.get(name, t) for t in tasks]
        return np.vstack(rows)

    # ── Schema / introspection ───────────────────────────────

    @property
    def feature_names(self) -> List[str]:
        """All registered feature set names."""
        return list(self._specs.keys())

    def total_dim(self, names: Optional[List[str]] = None) -> int:
        """Total dimensionality across the requested feature sets."""
        if names is None:
            names = list(self._specs.keys())
        return sum(self._specs[n].dim for n in names if n in self._specs)

    def schema(self) -> List[Dict[str, Any]]:
        """Description of all registered feature sets."""
        return [
            {
                "name": s.name,
                "dim": s.dim,
                "depends": s.depends,
                "description": s.description,
            }
            for s in self._specs.values()
        ]

    def clear_cache(self) -> None:
        """Flush the feature cache."""
        self._cache.clear()


class RedisFeatureCache:
    """Redis-backed distributed feature cache layer.

    Wraps a :class:`FeatureStore` and caches computed features in
    Redis so that multiple agents sharing the same fleet can skip
    redundant computation.

    Requires: ``pip install redis``
    """

    def __init__(
        self,
        store: FeatureStore,
        redis_url: str = "redis://localhost:6379/0",
        ttl_seconds: int = 300,
        key_prefix: str = "qstrainer:features:",
    ) -> None:
        try:
            import redis as _redis
        except ImportError:
            raise ImportError("redis is required for RedisFeatureCache")

        self._store = store
        self._client = _redis.Redis.from_url(redis_url, decode_responses=False)
        self._ttl = ttl_seconds
        self._prefix = key_prefix

    def get(self, name: str, task: ComputeTask) -> np.ndarray:
        """Get features, checking Redis first, then computing locally."""
        cache_key = f"{self._prefix}{name}:{task.gpu_id}:{task.timestamp}"
        raw = self._client.get(cache_key)
        if raw is not None:
            return np.frombuffer(raw, dtype=np.float64)

        # Compute locally
        result = self._store.get(name, task)

        # Cache in Redis
        self._client.setex(cache_key, self._ttl, result.tobytes())
        return result
