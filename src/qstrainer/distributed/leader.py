"""Leader election for checkpoint coordination.

Uses Redis-based distributed locking so that exactly one agent in a
multi-node fleet is the *leader* at any time.  The leader is responsible
for:

  - Periodic fleet-wide checkpoints (write)
  - Baseline model redistribution (write)
  - Solver scheduling decisions (read)

Non-leaders continue ingesting and straining but skip checkpoint writes
and defer to the leader's model state.

Requires: ``pip install redis``

Example::

    from qstrainer.distributed.leader import LeaderElector
    elector = LeaderElector(redis_url="redis://localhost:6379/0",
                            node_id="agent-us-east-1a")
    if elector.try_acquire():
        # I am the leader — safe to write checkpoints
        ...
    elector.release()
"""

from __future__ import annotations

import logging
import os
import time
import threading
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import redis

    _HAS_REDIS = True
except ImportError:
    _HAS_REDIS = False

_LOCK_KEY = "qstrainer:leader:lock"
_LEADER_KEY = "qstrainer:leader:info"


class LeaderElector:
    """Redis-based leader election with TTL heartbeat.

    Parameters
    ----------
    redis_url : str
        Redis connection URL.
    node_id : str
        Unique identifier for this agent instance.
    lease_ttl : float
        Leadership lease duration in seconds.  The leader must renew
        before this expires or another node may claim leadership.
    renew_interval : float
        How often (seconds) to auto-renew the lease in the background.
    key_prefix : str
        Redis key namespace prefix.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        *,
        node_id: str | None = None,
        lease_ttl: float = 30.0,
        renew_interval: float = 10.0,
        key_prefix: str = "qstrainer:",
    ) -> None:
        if not _HAS_REDIS:
            raise ImportError("redis is required. Install with: pip install redis")

        self._url = redis_url
        self._node_id = node_id or f"agent-{os.getpid()}"
        self._lease_ttl = lease_ttl
        self._renew_interval = renew_interval
        self._prefix = key_prefix
        self._lock_key = f"{key_prefix}leader:lock"
        self._info_key = f"{key_prefix}leader:info"

        self._client: Optional[redis.Redis] = None
        self._lock: Optional[redis.lock.Lock] = None
        self._is_leader: bool = False
        self._renewal_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ── Connection ───────────────────────────────────────────

    def _ensure_connected(self) -> "redis.Redis":
        if self._client is None:
            self._client = redis.Redis.from_url(self._url, decode_responses=True)
        return self._client

    # ── Acquire / release ────────────────────────────────────

    def try_acquire(self) -> bool:
        """Attempt to become leader.  Returns True if successful."""
        r = self._ensure_connected()

        self._lock = r.lock(
            self._lock_key,
            timeout=self._lease_ttl,
            blocking=False,
        )

        acquired = self._lock.acquire(blocking=False)
        if acquired:
            self._is_leader = True
            r.hset(self._info_key, mapping={
                "node_id": self._node_id,
                "acquired_at": str(time.time()),
                "pid": str(os.getpid()),
            })
            r.expire(self._info_key, int(self._lease_ttl * 2))
            logger.info("Leadership acquired by %s", self._node_id)
            self._start_renewal()
            return True

        logger.debug("Leadership NOT acquired by %s (another leader active)", self._node_id)
        return False

    def release(self) -> None:
        """Voluntarily release leadership."""
        self._stop_renewal()
        if self._lock is not None and self._is_leader:
            try:
                self._lock.release()
                logger.info("Leadership released by %s", self._node_id)
            except redis.exceptions.LockNotOwnedError:
                logger.warning("Lock already expired before release")
        self._is_leader = False

    # ── Queries ──────────────────────────────────────────────

    @property
    def is_leader(self) -> bool:
        """Whether this instance currently holds leadership."""
        return self._is_leader

    def current_leader(self) -> Optional[str]:
        """Return the node_id of the current leader, or None."""
        r = self._ensure_connected()
        info = r.hgetall(self._info_key)
        return info.get("node_id") if info else None

    def leader_info(self) -> Optional[dict]:
        """Return full leader metadata dict."""
        r = self._ensure_connected()
        info = r.hgetall(self._info_key)
        return dict(info) if info else None

    # ── Renewal ──────────────────────────────────────────────

    def _start_renewal(self) -> None:
        """Start background thread that renews the lease."""
        self._stop_event.clear()
        self._renewal_thread = threading.Thread(
            target=self._renewal_loop, daemon=True, name="leader-renewal"
        )
        self._renewal_thread.start()

    def _stop_renewal(self) -> None:
        self._stop_event.set()
        if self._renewal_thread is not None:
            self._renewal_thread.join(timeout=5.0)
            self._renewal_thread = None

    def _renewal_loop(self) -> None:
        """Periodically extend the lock TTL."""
        while not self._stop_event.is_set():
            try:
                if self._lock is not None and self._is_leader:
                    self._lock.reacquire()
                    r = self._ensure_connected()
                    r.expire(self._info_key, int(self._lease_ttl * 2))
                    logger.debug("Leadership lease renewed by %s", self._node_id)
            except redis.exceptions.LockNotOwnedError:
                logger.warning("Lost leadership (lock expired) — %s", self._node_id)
                self._is_leader = False
                return
            except Exception:
                logger.exception("Error renewing leadership lease")

            self._stop_event.wait(timeout=self._renew_interval)

    # ── Context manager ──────────────────────────────────────

    def __enter__(self) -> "LeaderElector":
        self.try_acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    # ── Cleanup ──────────────────────────────────────────────

    def close(self) -> None:
        """Release leadership and close the Redis connection."""
        self.release()
        if self._client is not None:
            self._client.close()
            self._client = None
