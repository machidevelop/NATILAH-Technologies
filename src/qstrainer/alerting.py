"""Strain decision routing — dispatch strain decisions to external systems.

Supports:
  - Webhook (generic HTTP POST with JSON payload)
  - Slack (Incoming Webhook with block kit formatting)
  - PagerDuty (Events API v2)

Usage::

    from qstrainer.alerting import DecisionRouter, WebhookRoute, SlackRoute
    router = DecisionRouter()
    router.add_route(SlackRoute(webhook_url="https://hooks.slack.com/..."))
    router.dispatch(decision, result)
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from qstrainer.models.alert import StrainDecision, StrainResult
from qstrainer.models.enums import StrainAction, TaskVerdict

logger = logging.getLogger(__name__)


# ── Base Route ──────────────────────────────────────────────


class DecisionRoute(ABC):
    """Abstract base for strain decision destinations."""

    @abstractmethod
    def send(self, decision: StrainDecision, context: StrainResult | None = None) -> bool:
        """Send a strain decision. Returns True on success."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable route name."""
        ...


# ── Backward compatibility aliases ──────────────────────────
AlertRoute = DecisionRoute  # alias for existing code


# ── Webhook Route ───────────────────────────────────────────


class WebhookRoute(DecisionRoute):
    """Send strain decisions to a generic HTTP webhook endpoint."""

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout_s: float = 5.0,
    ) -> None:
        self._url = url
        self._headers = {"Content-Type": "application/json", **(headers or {})}
        self._timeout = timeout_s

    @property
    def name(self) -> str:
        return f"webhook({self._url[:40]}...)"

    def send(self, decision: StrainDecision, context: StrainResult | None = None) -> bool:
        payload = {
            "source": "qstrainer",
            "timestamp": decision.timestamp,
            "verdict": decision.verdict.name,
            "action": decision.action.name,
            "gpu_id": decision.gpu_id,
            "job_id": decision.job_id,
            "task_id": decision.task_id,
            "metric": decision.metric,
            "value": decision.value,
            "threshold": decision.threshold,
            "reason": decision.reason,
        }
        if context:
            payload["redundancy_score"] = context.redundancy_score
            payload["strain_verdict"] = context.verdict.name
            payload["compute_saved_flops"] = context.compute_saved_flops

        return self._post(payload)

    def _post(self, payload: dict) -> bool:
        try:
            data = json.dumps(payload, default=str).encode()
            req = Request(self._url, data=data, headers=self._headers, method="POST")
            with urlopen(req, timeout=self._timeout) as resp:
                if resp.status < 300:
                    return True
                logger.warning("Webhook returned %d", resp.status)
                return False
        except (URLError, OSError) as e:
            logger.error("Webhook delivery failed: %s", e)
            return False


# ── Slack Route ─────────────────────────────────────────────


class SlackRoute(DecisionRoute):
    """Send strain decisions to Slack via Incoming Webhook."""

    VERDICT_EMOJI = {
        TaskVerdict.EXECUTE: ":large_green_circle:",
        TaskVerdict.APPROXIMATE: ":large_yellow_circle:",
        TaskVerdict.DEFER: ":large_orange_circle:",
        TaskVerdict.SKIP: ":red_circle:",
    }

    def __init__(
        self,
        webhook_url: str,
        channel: str | None = None,
        timeout_s: float = 5.0,
    ) -> None:
        self._url = webhook_url
        self._channel = channel
        self._timeout = timeout_s

    @property
    def name(self) -> str:
        return "slack"

    def send(self, decision: StrainDecision, context: StrainResult | None = None) -> bool:
        emoji = self.VERDICT_EMOJI.get(decision.verdict, ":question:")
        score_str = ""
        if context:
            score_str = f"  |  Redundancy: *{context.redundancy_score:.2f}*"

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{decision.verdict.name} — {decision.task_id}",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"{emoji} *{decision.reason}*\n"
                        f"GPU: `{decision.gpu_id}` | Job: `{decision.job_id}`\n"
                        f"Metric: `{decision.metric}` = {decision.value:.4f} "
                        f"(threshold: {decision.threshold:.4f}){score_str}"
                    ),
                },
            },
        ]

        if context and context.compute_saved_flops > 0:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f":zap: *Compute saved:* {context.compute_saved_flops:.2e} FLOPs | "
                            f"${context.cost_saved_usd:.4f}"
                        ),
                    },
                }
            )

        payload: dict[str, Any] = {"blocks": blocks}
        if self._channel:
            payload["channel"] = self._channel

        try:
            data = json.dumps(payload).encode()
            req = Request(
                self._url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=self._timeout) as resp:
                return bool(resp.status < 300)
        except (URLError, OSError) as e:
            logger.error("Slack delivery failed: %s", e)
            return False


# ── PagerDuty Route ─────────────────────────────────────────


class PagerDutyRoute(DecisionRoute):
    """Send strain decisions to PagerDuty Events API v2."""

    VERDICT_SEVERITY = {
        TaskVerdict.EXECUTE: "info",
        TaskVerdict.APPROXIMATE: "warning",
        TaskVerdict.DEFER: "warning",
        TaskVerdict.SKIP: "critical",
    }
    EVENTS_URL = "https://events.pagerduty.com/v2/enqueue"

    def __init__(self, routing_key: str, timeout_s: float = 5.0) -> None:
        self._routing_key = routing_key
        self._timeout = timeout_s

    @property
    def name(self) -> str:
        return "pagerduty"

    def send(self, decision: StrainDecision, context: StrainResult | None = None) -> bool:
        severity = self.VERDICT_SEVERITY.get(decision.verdict, "warning")
        payload = {
            "routing_key": self._routing_key,
            "event_action": "trigger",
            "payload": {
                "summary": f"[{decision.verdict.name}] {decision.reason} — {decision.task_id}",
                "source": f"{decision.gpu_id}/{decision.job_id}",
                "severity": severity,
                "component": "qstrainer",
                "group": decision.gpu_id,
                "class": decision.metric,
                "custom_details": {
                    "gpu_id": decision.gpu_id,
                    "job_id": decision.job_id,
                    "task_id": decision.task_id,
                    "metric": decision.metric,
                    "value": decision.value,
                    "threshold": decision.threshold,
                    "redundancy_score": context.redundancy_score if context else None,
                    "verdict": context.verdict.name if context else None,
                    "compute_saved_flops": context.compute_saved_flops if context else None,
                },
            },
        }

        try:
            data = json.dumps(payload, default=str).encode()
            req = Request(
                self.EVENTS_URL,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=self._timeout) as resp:
                return bool(resp.status < 300)
        except (URLError, OSError) as e:
            logger.error("PagerDuty delivery failed: %s", e)
            return False


# ── Decision Filter ─────────────────────────────────────────


@dataclass
class DecisionFilter:
    """Filter strain decisions before routing."""

    min_action: StrainAction = StrainAction.REDUCE
    gpu_ids: list[str] | None = None  # None = all GPUs
    cooldown_s: float = 60.0  # suppress duplicates within window

    def should_route(self, decision: StrainDecision) -> bool:
        if decision.action.value < self.min_action.value:
            return False
        return not (self.gpu_ids is not None and decision.gpu_id not in self.gpu_ids)


# Backward compatibility alias
AlertFilter = DecisionFilter


class DecisionRouter:
    """Dispatch strain decisions to configured routes with filtering.

    Usage::

        router = DecisionRouter()
        router.add_route(SlackRoute(webhook_url="..."))
        router.add_route(PagerDutyRoute(routing_key="..."),
                         filter=DecisionFilter(min_action=StrainAction.ELIMINATE))
        router.dispatch(decision, context=strain_result)
    """

    def __init__(self) -> None:
        self._routes: list[tuple[DecisionRoute, DecisionFilter]] = []
        self._last_sent: dict[str, float] = {}

    def add_route(self, route: DecisionRoute, filter: DecisionFilter | None = None) -> None:
        self._routes.append((route, filter or DecisionFilter()))

    def dispatch(
        self, decision: StrainDecision, context: StrainResult | None = None
    ) -> dict[str, bool]:
        """Send decision to all matching routes. Returns {route_name: success}."""
        results = {}
        for route, filt in self._routes:
            if not filt.should_route(decision):
                continue

            cooldown_key = (
                f"{route.name}:{decision.gpu_id}:{decision.metric}:{decision.verdict.name}"
            )
            now = time.time()
            last = self._last_sent.get(cooldown_key, 0)
            if now - last < filt.cooldown_s:
                logger.debug("Cooldown suppressed: %s", cooldown_key)
                continue

            success = route.send(decision, context)
            results[route.name] = success
            if success:
                self._last_sent[cooldown_key] = now

        return results

    @classmethod
    def from_config(cls, cfg: dict) -> DecisionRouter:
        """Build router from config dict.

        Example config::

            alerting:
              routes:
                - type: slack
                  webhook_url: https://hooks.slack.com/...
                  min_action: REDUCE
                - type: pagerduty
                  routing_key: abc123
                  min_action: ELIMINATE
                - type: webhook
                  url: https://my-server.com/decisions
        """
        router = cls()
        alerting_cfg = cfg.get("alerting", {})
        routes = alerting_cfg.get("routes", [])

        for route_cfg in routes:
            route_type = route_cfg.get("type", "").lower()
            min_act = getattr(
                StrainAction,
                route_cfg.get("min_action", "REDUCE"),
                StrainAction.REDUCE,
            )
            filt = DecisionFilter(
                min_action=min_act,
                cooldown_s=route_cfg.get("cooldown_s", 60.0),
            )

            if route_type == "webhook":
                route: DecisionRoute = WebhookRoute(
                    url=route_cfg["url"],
                    headers=route_cfg.get("headers"),
                )
            elif route_type == "slack":
                route = SlackRoute(
                    webhook_url=route_cfg["webhook_url"],
                    channel=route_cfg.get("channel"),
                )
            elif route_type == "pagerduty":
                route = PagerDutyRoute(
                    routing_key=route_cfg["routing_key"],
                )
            else:
                logger.warning("Unknown route type: %s", route_type)
                continue

            router.add_route(route, filt)
            logger.info("Route added: %s (min_action=%s)", route.name, min_act.name)

        return router


# Backward compatibility alias
AlertRouter = DecisionRouter
