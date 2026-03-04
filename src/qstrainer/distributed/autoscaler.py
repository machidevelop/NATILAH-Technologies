"""Horizontal autoscaler — adjusts agent fleet size based on throughput.

Monitors frame throughput across the fleet and emits scaling
recommendations.  In a Kubernetes deployment these feed into the
Horizontal Pod Autoscaler (HPA); in bare-metal deployments an
operator reads the recommendations and provisions nodes.

Works with either:
  - **Prometheus pull**: exposes ``qstrainer_autoscale_desired_replicas``
  - **Redis push**: writes scaling events for a coordinator to consume

Architecture::

    ┌───────────────────────────────────────────────────────┐
    │  Autoscaler                                           │
    │  ┌──────────┐   ┌───────────┐   ┌────────────────┐   │
    │  │ Collector │──→│ Algorithm │──→│ ScaleDecision  │   │
    │  │ (metrics) │   │ (PID/step)│   │ (up/down/hold) │   │
    │  └──────────┘   └───────────┘   └────────────────┘   │
    └───────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ScaleAction(Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    HOLD = "hold"


@dataclass(slots=True)
class ScaleDecision:
    """A single scaling recommendation."""

    action: ScaleAction
    current_replicas: int
    desired_replicas: int
    reason: str
    timestamp: float = field(default_factory=time.time)
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ThroughputSample:
    """One observation of fleet throughput."""

    timestamp: float
    frames_per_second: float
    active_gpus: int
    avg_latency_ms: float


class Autoscaler:
    """Horizontal autoscaler that recommends replica counts.

    Parameters
    ----------
    target_fps_per_replica : float
        Target frame-throughput each replica should sustain.
    min_replicas : int
        Never scale below this.
    max_replicas : int
        Never scale above this.
    scale_up_threshold : float
        Scale up when utilisation exceeds this fraction (0–1).
    scale_down_threshold : float
        Scale down when utilisation drops below this fraction.
    cooldown_seconds : float
        Minimum time between consecutive scale actions.
    window_size : int
        Number of throughput samples to average for decisions.
    """

    def __init__(
        self,
        target_fps_per_replica: float = 500.0,
        min_replicas: int = 1,
        max_replicas: int = 32,
        scale_up_threshold: float = 0.80,
        scale_down_threshold: float = 0.30,
        cooldown_seconds: float = 120.0,
        window_size: int = 10,
    ) -> None:
        self._target_fps = target_fps_per_replica
        self._min = min_replicas
        self._max = max_replicas
        self._up_thresh = scale_up_threshold
        self._down_thresh = scale_down_threshold
        self._cooldown = cooldown_seconds
        self._window = window_size

        self._current_replicas: int = 1
        self._samples: list[ThroughputSample] = []
        self._last_action_time: float = 0.0
        self._history: list[ScaleDecision] = []

    # ── Feed metrics ─────────────────────────────────────────

    def record(
        self,
        frames_per_second: float,
        active_gpus: int = 0,
        avg_latency_ms: float = 0.0,
    ) -> None:
        """Record a throughput observation."""
        self._samples.append(
            ThroughputSample(
                timestamp=time.time(),
                frames_per_second=frames_per_second,
                active_gpus=active_gpus,
                avg_latency_ms=avg_latency_ms,
            )
        )
        # Keep only the window
        if len(self._samples) > self._window * 2:
            self._samples = self._samples[-self._window :]

    def set_current_replicas(self, n: int) -> None:
        """Update the autoscaler's view of current replica count."""
        self._current_replicas = max(1, n)

    # ── Decision engine ──────────────────────────────────────

    def evaluate(self) -> ScaleDecision:
        """Evaluate current metrics and return a scaling decision.

        Call this periodically (e.g. every 30 s).
        """
        now = time.time()

        # Not enough data yet
        if len(self._samples) < 3:
            return self._decision(ScaleAction.HOLD, "insufficient samples")

        # Cooldown check
        if now - self._last_action_time < self._cooldown:
            return self._decision(ScaleAction.HOLD, "cooldown active")

        # Average over window
        recent = self._samples[-self._window :]
        avg_fps = sum(s.frames_per_second for s in recent) / len(recent)
        avg_lat = sum(s.avg_latency_ms for s in recent) / len(recent)

        capacity = self._current_replicas * self._target_fps
        utilisation = avg_fps / capacity if capacity > 0 else 1.0

        metrics = {
            "avg_fps": avg_fps,
            "avg_latency_ms": avg_lat,
            "capacity_fps": capacity,
            "utilisation": utilisation,
        }

        # Scale up?
        if utilisation > self._up_thresh:
            desired = math.ceil(avg_fps / (self._target_fps * self._up_thresh))
            desired = min(desired, self._max)
            if desired > self._current_replicas:
                return self._scale(
                    ScaleAction.SCALE_UP,
                    desired,
                    f"utilisation {utilisation:.0%} > {self._up_thresh:.0%}",
                    metrics,
                )

        # Scale down?
        if utilisation < self._down_thresh:
            desired = max(math.ceil(avg_fps / self._target_fps), self._min)
            if desired < self._current_replicas:
                return self._scale(
                    ScaleAction.SCALE_DOWN,
                    desired,
                    f"utilisation {utilisation:.0%} < {self._down_thresh:.0%}",
                    metrics,
                )

        return self._decision(
            ScaleAction.HOLD,
            f"utilisation {utilisation:.0%} within range",
            metrics,
        )

    # ── Internals ────────────────────────────────────────────

    def _decision(
        self,
        action: ScaleAction,
        reason: str,
        metrics: dict[str, float] | None = None,
    ) -> ScaleDecision:
        d = ScaleDecision(
            action=action,
            current_replicas=self._current_replicas,
            desired_replicas=self._current_replicas,
            reason=reason,
            metrics=metrics or {},
        )
        self._history.append(d)
        return d

    def _scale(
        self,
        action: ScaleAction,
        desired: int,
        reason: str,
        metrics: dict[str, float],
    ) -> ScaleDecision:
        d = ScaleDecision(
            action=action,
            current_replicas=self._current_replicas,
            desired_replicas=desired,
            reason=reason,
            metrics=metrics,
        )
        self._current_replicas = desired
        self._last_action_time = time.time()
        self._history.append(d)
        logger.info(
            "Autoscaler: %s → %d replicas (%s)",
            action.value,
            desired,
            reason,
        )
        return d

    # ── Queries ──────────────────────────────────────────────

    @property
    def current_replicas(self) -> int:
        return self._current_replicas

    @property
    def history(self) -> list[ScaleDecision]:
        return list(self._history)

    @classmethod
    def from_config(cls, cfg: dict) -> Autoscaler:
        """Build from config dict."""
        ac = cfg.get("autoscaler", {})
        return cls(
            target_fps_per_replica=ac.get("target_fps_per_replica", 500.0),
            min_replicas=ac.get("min_replicas", 1),
            max_replicas=ac.get("max_replicas", 32),
            scale_up_threshold=ac.get("scale_up_threshold", 0.80),
            scale_down_threshold=ac.get("scale_down_threshold", 0.30),
            cooldown_seconds=ac.get("cooldown_seconds", 120.0),
            window_size=ac.get("window_size", 10),
        )
