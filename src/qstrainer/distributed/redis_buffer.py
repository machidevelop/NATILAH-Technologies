"""Redis-backed shared workload buffer for distributed fleet state.

Extends the local :class:`WorkloadBuffer` semantics across multiple
Q-Strainer agent processes.  Each agent pushes tasks into Redis
(sorted set per GPU, scored by timestamp) so that every node has a
consistent view of fleet compute workloads.

Requires: ``pip install redis``

Usage::

    from qstrainer.distributed.redis_buffer import RedisBuffer
    buf = RedisBuffer(redis_url="redis://sentinel:26379/0")
    buf.push(task)
    matrix = buf.get_matrix("GPU-0", n_tasks=100)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Dict, List, Optional

import numpy as np

from qstrainer.models.frame import ComputeTask, N_BASE_FEATURES
from qstrainer.models.enums import ComputePhase, JobType

logger = logging.getLogger(__name__)

try:
    import redis

    _HAS_REDIS = True
except ImportError:
    _HAS_REDIS = False

# Key schema
_KEY_PREFIX = "qstrainer:"
_TASKS_KEY = _KEY_PREFIX + "tasks:{gpu_id}"     # Sorted Set (score=timestamp)
_META_KEY = _KEY_PREFIX + "meta:{gpu_id}"        # Hash  — latest verdict, score
_FLEET_KEY = _KEY_PREFIX + "fleet:gpu_ids"       # Set   — all known GPU IDs
_COUNTER_KEY = _KEY_PREFIX + "stats:task_count"  # String — global task counter


def _task_to_json(task: ComputeTask) -> str:
    """Serialise a ComputeTask to compact JSON."""
    return json.dumps({
        "ts": task.timestamp,
        "tid": task.task_id,
        "gpu": task.gpu_id,
        "jid": task.job_id,
        "sn": task.step_number,
        "loss": task.loss,
        "ld": task.loss_delta,
        "gn": task.gradient_norm,
        "gv": task.gradient_variance,
        "lr": task.learning_rate,
        "bs": task.batch_size,
        "ep": task.epoch,
        "epg": task.epoch_progress,
        "efl": task.estimated_flops,
        "ets": task.estimated_time_s,
        "mem": task.memory_footprint_gb,
        "cp": task.compute_phase.name,
        "jt": task.job_type.name,
        "cs": task.convergence_score,
        "pum": task.param_update_magnitude,
        "ds": task.data_similarity,
        "fu": task.flop_utilization,
        "thr": task.throughput_samples_per_sec,
        "node": task.node_id,
    })


def _json_to_task(data: str) -> ComputeTask:
    """Deserialise JSON back to a ComputeTask."""
    d = json.loads(data)
    return ComputeTask(
        timestamp=d["ts"],
        task_id=d["tid"],
        gpu_id=d["gpu"],
        job_id=d["jid"],
        step_number=d["sn"],
        loss=d["loss"],
        loss_delta=d["ld"],
        gradient_norm=d["gn"],
        gradient_variance=d["gv"],
        learning_rate=d["lr"],
        batch_size=d["bs"],
        epoch=d["ep"],
        epoch_progress=d["epg"],
        estimated_flops=d["efl"],
        estimated_time_s=d["ets"],
        memory_footprint_gb=d["mem"],
        compute_phase=ComputePhase[d["cp"]],
        job_type=JobType[d["jt"]],
        convergence_score=d["cs"],
        param_update_magnitude=d["pum"],
        data_similarity=d["ds"],
        flop_utilization=d["fu"],
        throughput_samples_per_sec=d["thr"],
        node_id=d.get("node", "unknown"),
    )


class RedisBuffer:
    """Distributed workload buffer backed by Redis.

    Parameters
    ----------
    redis_url : str
        Redis connection URL (``redis://host:port/db``).
    max_tasks_per_gpu : int
        Maximum tasks retained per GPU (oldest trimmed).
    key_prefix : str
        Namespace prefix for all Redis keys.
    ttl_seconds : int
        TTL applied to task sorted sets (auto-expire stale GPUs).
    pipeline_batch : int
        Number of commands to batch in a Redis pipeline for push operations.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        *,
        max_tasks_per_gpu: int = 1000,
        key_prefix: str = _KEY_PREFIX,
        ttl_seconds: int = 3600,
        pipeline_batch: int = 50,
    ) -> None:
        if not _HAS_REDIS:
            raise ImportError(
                "redis is required. Install with: pip install redis"
            )
        self._url = redis_url
        self._max_tasks = max_tasks_per_gpu
        self._prefix = key_prefix
        self._ttl = ttl_seconds
        self._batch_size = pipeline_batch
        self._client: Optional[redis.Redis] = None

    # ── Connection ───────────────────────────────────────────

    def _ensure_connected(self) -> "redis.Redis":
        if self._client is None:
            self._client = redis.Redis.from_url(
                self._url, decode_responses=True
            )
            logger.info("RedisBuffer connected to %s", self._url)
        return self._client

    # ── Write path ───────────────────────────────────────────

    def push(self, task: ComputeTask) -> None:
        """Push a single task into the distributed buffer."""
        r = self._ensure_connected()
        key = self._tasks_key(task.gpu_id)
        payload = _task_to_json(task)

        pipe = r.pipeline(transaction=False)
        pipe.zadd(key, {payload: task.timestamp})
        pipe.zremrangebyrank(key, 0, -(self._max_tasks + 1))
        pipe.expire(key, self._ttl)
        pipe.sadd(self._fleet_key(), task.gpu_id)
        pipe.incr(self._counter_key())
        pipe.execute()

    def push_batch(self, tasks: List[ComputeTask]) -> None:
        """Push multiple tasks efficiently using Redis pipelines."""
        if not tasks:
            return
        r = self._ensure_connected()
        pipe = r.pipeline(transaction=False)

        for i, task in enumerate(tasks):
            key = self._tasks_key(task.gpu_id)
            payload = _task_to_json(task)
            pipe.zadd(key, {payload: task.timestamp})
            pipe.zremrangebyrank(key, 0, -(self._max_tasks + 1))
            pipe.expire(key, self._ttl)
            pipe.sadd(self._fleet_key(), task.gpu_id)
            pipe.incr(self._counter_key())

            if (i + 1) % self._batch_size == 0:
                pipe.execute()
                pipe = r.pipeline(transaction=False)

        pipe.execute()

    # ── Read path ────────────────────────────────────────────

    def get_window(self, gpu_id: str, n_tasks: int) -> List[ComputeTask]:
        """Return the last *n_tasks* for a GPU (most recent last)."""
        r = self._ensure_connected()
        key = self._tasks_key(gpu_id)
        raw = r.zrange(key, -n_tasks, -1)
        return [_json_to_task(item) for item in raw]

    def get_matrix(self, gpu_id: str, n_tasks: int) -> np.ndarray:
        """Return the last *n_tasks* as an (n × 15) NumPy matrix."""
        tasks = self.get_window(gpu_id, n_tasks)
        if not tasks:
            return np.empty((0, N_BASE_FEATURES))
        return np.vstack([t.to_vector() for t in tasks])

    def get_latest(self, gpu_id: str) -> Optional[ComputeTask]:
        """Return the most recent task for a GPU."""
        r = self._ensure_connected()
        key = self._tasks_key(gpu_id)
        raw = r.zrange(key, -1, -1)
        if not raw:
            return None
        return _json_to_task(raw[0])

    # ── Fleet queries ────────────────────────────────────────

    @property
    def gpu_ids(self) -> List[str]:
        """All known GPU IDs across the fleet."""
        r = self._ensure_connected()
        return list(r.smembers(self._fleet_key()))

    @property
    def total_tasks(self) -> int:
        """Global task counter across all agents."""
        r = self._ensure_connected()
        val = r.get(self._counter_key())
        return int(val) if val else 0

    def task_count(self, gpu_id: str) -> int:
        """Number of tasks currently stored for a specific GPU."""
        r = self._ensure_connected()
        return r.zcard(self._tasks_key(gpu_id))

    # ── Metadata ─────────────────────────────────────────────

    def set_gpu_meta(self, gpu_id: str, verdict: str, redundancy_score: float) -> None:
        """Update the latest strain verdict for a GPU."""
        r = self._ensure_connected()
        key = f"{self._prefix}meta:{gpu_id}"
        r.hset(key, mapping={
            "verdict": verdict,
            "redundancy_score": str(redundancy_score),
            "updated_at": str(time.time()),
        })
        r.expire(key, self._ttl)

    def get_gpu_meta(self, gpu_id: str) -> Optional[Dict]:
        """Get the latest metadata for a GPU."""
        r = self._ensure_connected()
        key = f"{self._prefix}meta:{gpu_id}"
        data = r.hgetall(key)
        if not data:
            return None
        return {
            "verdict": data.get("verdict", "UNKNOWN"),
            "redundancy_score": float(data.get("redundancy_score", "0.0")),
            "updated_at": float(data.get("updated_at", "0.0")),
        }

    def fleet_summary(self) -> Dict[str, Dict]:
        """Return verdict metadata for every known GPU."""
        result = {}
        for gpu_id in self.gpu_ids:
            meta = self.get_gpu_meta(gpu_id)
            result[gpu_id] = meta or {"verdict": "UNKNOWN", "redundancy_score": 0.0}
        return result

    # ── Cleanup ──────────────────────────────────────────────

    def clear(self, gpu_id: Optional[str] = None) -> None:
        """Remove tasks (and metadata) for one or all GPUs."""
        r = self._ensure_connected()
        if gpu_id:
            r.delete(self._tasks_key(gpu_id))
            r.delete(f"{self._prefix}meta:{gpu_id}")
            r.srem(self._fleet_key(), gpu_id)
        else:
            for gid in self.gpu_ids:
                r.delete(self._tasks_key(gid))
                r.delete(f"{self._prefix}meta:{gid}")
            r.delete(self._fleet_key())
            r.delete(self._counter_key())

    def close(self) -> None:
        """Close the Redis connection."""
        if self._client is not None:
            self._client.close()
            self._client = None

    # ── Key helpers ──────────────────────────────────────────

    def _tasks_key(self, gpu_id: str) -> str:
        return f"{self._prefix}tasks:{gpu_id}"

    def _fleet_key(self) -> str:
        return f"{self._prefix}fleet:gpu_ids"

    def _counter_key(self) -> str:
        return f"{self._prefix}stats:task_count"
