"""Tests for the full QStrainer pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from qstrainer.models.enums import TaskVerdict
from qstrainer.pipeline.strainer import QStrainer


class TestQStrainerPipeline:
    def test_productive_tasks_executed(self, gen):
        qs = QStrainer(strain_threshold=0.5, heartbeat_interval=9999)
        executed = 0
        for _ in range(50):
            f = gen.generate_healthy("GPU-P", "node-p")
            result = qs.process_task(f)
            if result.verdict == TaskVerdict.EXECUTE:
                executed += 1
        # Most productive tasks should be executed
        assert executed > 10

    def test_redundant_tasks_strained(self, gen):
        qs = QStrainer(strain_threshold=0.5, heartbeat_interval=9999)
        # Warm up
        for _ in range(30):
            f = gen.generate_healthy("GPU-F", "node-f")
            qs.process_task(f)
        # Send redundant
        strained = 0
        for _ in range(10):
            f = gen.generate_failing("GPU-F", "node-f")
            result = qs.process_task(f)
            if result.verdict != TaskVerdict.EXECUTE:
                strained += 1
        assert strained > 0

    def test_stats_tracking(self, gen):
        qs = QStrainer()
        for _ in range(10):
            qs.process_task(gen.generate_healthy("GPU-S", "node-s"))
        stats = qs.stats
        assert stats["tasks_processed"] == 10

    def test_from_config(self):
        cfg = {"pipeline": {"strain_threshold": 0.6, "heartbeat_interval": 100}}
        qs = QStrainer.from_config(cfg)
        assert qs.strain_threshold == 0.6

    def test_process_always_returns_result(self, gen):
        qs = QStrainer()
        f = gen.generate_healthy("GPU-R", "node-r")
        result = qs.process_task(f)
        assert result is not None
        assert result.verdict in TaskVerdict

