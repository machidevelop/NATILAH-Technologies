"""Memory profiling utilities for Q-Strainer long-running daemons.

Provides helpers to:
- Track peak / current RSS at regular intervals.
- Expose memory metrics via Prometheus gauges.
- Identify top allocation sites using :mod:`tracemalloc`.

Usage from the agent daemon::

    from qstrainer.profiling import MemoryProfiler
    mp = MemoryProfiler(snapshot_interval=500)
    mp.start()
    ...
    mp.snapshot()   # periodic call
    ...
    mp.report()     # on shutdown
"""

from __future__ import annotations

import logging
import os
import tracemalloc
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MemorySnapshot:
    """A single observation of process memory."""

    frame_count: int
    rss_mb: float
    tracemalloc_peak_mb: float
    tracemalloc_current_mb: float


class MemoryProfiler:
    """Lightweight memory profiler for long-running daemon processes.

    Parameters
    ----------
    snapshot_interval : int
        Take a snapshot every *snapshot_interval* calls to :meth:`tick`.
    tracemalloc_enabled : bool
        If True, enable ``tracemalloc`` for allocation-site tracking.
    tracemalloc_nframes : int
        Call-stack depth for tracemalloc (higher = more detail, more overhead).
    warn_rss_mb : float
        Emit a warning when RSS exceeds this value (MB).
    """

    def __init__(
        self,
        snapshot_interval: int = 1000,
        tracemalloc_enabled: bool = False,
        tracemalloc_nframes: int = 5,
        warn_rss_mb: float = 2048.0,
    ) -> None:
        self._interval = snapshot_interval
        self._tracemalloc = tracemalloc_enabled
        self._nframes = tracemalloc_nframes
        self._warn_rss_mb = warn_rss_mb

        self._tick_count: int = 0
        self._snapshots: list[MemorySnapshot] = []
        self._started: bool = False

    # ── Lifecycle ────────────────────────────────────────────

    def start(self) -> None:
        """Begin profiling.  Call once before the processing loop."""
        if self._tracemalloc:
            tracemalloc.start(self._nframes)
            logger.info("tracemalloc started (nframes=%d)", self._nframes)
        self._started = True

    def stop(self) -> None:
        """Stop profiling and release tracemalloc resources."""
        if self._tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()
        self._started = False

    # ── Per-frame tick ───────────────────────────────────────

    def tick(self) -> MemorySnapshot | None:
        """Called once per frame.  Returns a snapshot at each interval."""
        self._tick_count += 1
        if self._tick_count % self._interval != 0:
            return None
        return self.snapshot()

    def snapshot(self) -> MemorySnapshot:
        """Force a memory snapshot right now."""
        rss = self._get_rss_mb()
        tm_peak, tm_cur = 0.0, 0.0
        if self._tracemalloc and tracemalloc.is_tracing():
            cur, peak = tracemalloc.get_traced_memory()
            tm_cur = cur / (1024 * 1024)
            tm_peak = peak / (1024 * 1024)

        snap = MemorySnapshot(
            frame_count=self._tick_count,
            rss_mb=rss,
            tracemalloc_peak_mb=tm_peak,
            tracemalloc_current_mb=tm_cur,
        )
        self._snapshots.append(snap)

        if rss > self._warn_rss_mb:
            logger.warning(
                "RSS %.1f MB exceeds threshold %.1f MB at frame %d",
                rss,
                self._warn_rss_mb,
                self._tick_count,
            )

        return snap

    # ── Reporting ────────────────────────────────────────────

    def report(self, top_n: int = 10) -> str:
        """Return a human-readable memory report."""
        lines = ["═══ Q-Strainer Memory Report ═══"]
        if self._snapshots:
            latest = self._snapshots[-1]
            peak_rss = max(s.rss_mb for s in self._snapshots)
            lines.append(f"  Frames processed : {self._tick_count:,}")
            lines.append(f"  Snapshots taken  : {len(self._snapshots)}")
            lines.append(f"  Current RSS      : {latest.rss_mb:.1f} MB")
            lines.append(f"  Peak RSS         : {peak_rss:.1f} MB")
            if self._tracemalloc:
                lines.append(f"  tracemalloc cur  : {latest.tracemalloc_current_mb:.1f} MB")
                lines.append(f"  tracemalloc peak : {latest.tracemalloc_peak_mb:.1f} MB")
        else:
            lines.append("  No snapshots collected")

        if self._tracemalloc and tracemalloc.is_tracing():
            snap = tracemalloc.take_snapshot()
            top_stats = snap.statistics("lineno")[:top_n]
            lines.append(f"\n  Top {top_n} allocation sites:")
            for i, stat in enumerate(top_stats, 1):
                lines.append(f"    {i}. {stat}")

        return "\n".join(lines)

    @property
    def snapshots(self) -> list[MemorySnapshot]:
        return list(self._snapshots)

    # ── Internals ────────────────────────────────────────────

    @staticmethod
    def _get_rss_mb() -> float:
        """Return current RSS in MB (cross-platform)."""
        try:
            import psutil

            proc = psutil.Process(os.getpid())
            return float(proc.memory_info().rss / (1024 * 1024))
        except ImportError:
            pass
        # Fallback for Linux
        try:
            with open(f"/proc/{os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024  # kB → MB
        except (OSError, ValueError):
            pass
        return 0.0
