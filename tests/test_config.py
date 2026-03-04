"""Tests for the config loader."""

from __future__ import annotations

import os
import pytest
from pathlib import Path

from qstrainer.config import load_config, _defaults, _deep_merge, _coerce


class TestConfigDefaults:
    def test_defaults_structure(self):
        cfg = _defaults()
        assert "agent" in cfg
        assert "pipeline" in cfg
        assert "solvers" in cfg
        assert "emitters" in cfg
        assert cfg["agent"]["poll_hz"] == 10.0

    def test_load_missing_file(self):
        cfg = load_config(Path("nonexistent.yaml"))
        # Should return defaults
        assert cfg["agent"]["poll_hz"] == 10.0

    def test_load_none(self):
        cfg = load_config(None)
        assert isinstance(cfg, dict)


class TestDeepMerge:
    def test_simple_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3}
        merged = _deep_merge(base, override)
        assert merged == {"a": 1, "b": 3}

    def test_nested_override(self):
        base = {"outer": {"a": 1, "b": 2}}
        override = {"outer": {"b": 3}}
        merged = _deep_merge(base, override)
        assert merged["outer"] == {"a": 1, "b": 3}

    def test_new_key(self):
        base = {"a": 1}
        override = {"b": 2}
        merged = _deep_merge(base, override)
        assert merged == {"a": 1, "b": 2}


class TestCoerce:
    def test_int(self):
        assert _coerce("42") == 42

    def test_float(self):
        assert _coerce("3.14") == 3.14

    def test_bool_true(self):
        assert _coerce("true") is True
        assert _coerce("yes") is True

    def test_bool_false(self):
        assert _coerce("false") is False

    def test_string(self):
        assert _coerce("hello") == "hello"


class TestEnvOverrides:
    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("QSTRAINER_AGENT__POLL_HZ", "20")
        cfg = load_config(None)
        assert cfg["agent"]["poll_hz"] == 20.0  # coerced from string

    def test_env_nested(self, monkeypatch):
        monkeypatch.setenv("QSTRAINER_PIPELINE__STRAIN_THRESHOLD", "0.5")
        cfg = load_config(None)
        assert cfg["pipeline"]["strain_threshold"] == 0.5
