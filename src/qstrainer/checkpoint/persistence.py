"""CheckpointManager — JSON-based state persistence for Q-Strainer.

Production agents call ``save()`` periodically so that on restart the
Welford baselines, trained SVM, and pipeline counters are preserved.
"""

from __future__ import annotations

import glob
import logging
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manage checkpoint files in a directory.

    Parameters
    ----------
    base_dir : str
        Directory where checkpoint files are stored.
    max_checkpoints : int
        Maximum number of checkpoint files to retain (FIFO).
    """

    CKPT_PREFIX = "qstrainer_ckpt_"
    CKPT_SUFFIX = ".pkl"

    def __init__(self, base_dir: str = "runs/", max_checkpoints: int = 5) -> None:
        self.base_dir = Path(base_dir)
        self.max_checkpoints = max_checkpoints
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ── Save / Restore ──────────────────────────────────────
    def save(self, pipeline: Any) -> Path:
        """Save the pipeline state to a timestamped checkpoint file."""
        from qstrainer import __version__

        state: dict[str, Any] = {
            "version": __version__,
            "timestamp": datetime.now().isoformat(),
            "task_count": getattr(pipeline, "_task_count", 0),
            "strained_count": getattr(pipeline, "_strained_count", 0),
            "executed_count": getattr(pipeline, "_executed_count", 0),
        }

        # Welford baselines
        if hasattr(pipeline, "convergence"):
            state["convergence_state"] = pipeline.convergence.get_baseline_state()

        # ML predictor
        if hasattr(pipeline, "predictor") and pipeline.predictor is not None:
            state["predictor_state"] = pipeline.predictor.get_state()

        # Verdict counts
        if hasattr(pipeline, "_verdict_counts"):
            state["verdict_counts"] = {
                (k.name if hasattr(k, "name") else str(k)): v
                for k, v in pipeline._verdict_counts.items()
            }

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.base_dir / f"{self.CKPT_PREFIX}{ts}{self.CKPT_SUFFIX}"
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("Checkpoint saved: %s (%d bytes)", path, path.stat().st_size)
        self._prune_old()
        return path

    def try_restore(self, pipeline: Any) -> bool:
        """Restore the most recent checkpoint into *pipeline*.  Returns True on success."""
        latest = self._latest_checkpoint()
        if latest is None:
            logger.info("No checkpoint found — starting fresh")
            return False

        try:
            with open(latest, "rb") as f:
                state = pickle.load(f)
        except Exception:
            logger.exception("Failed to load checkpoint %s", latest)
            return False

        # Restore counters
        pipeline._task_count = state.get("task_count", 0)
        pipeline._strained_count = state.get("strained_count", 0)
        pipeline._executed_count = state.get("executed_count", 0)

        # Restore convergence baselines
        if "convergence_state" in state and hasattr(pipeline, "convergence"):
            pipeline.convergence.load_baseline_state(state["convergence_state"])

        # Restore ML predictor
        if (
            "predictor_state" in state
            and hasattr(pipeline, "predictor")
            and pipeline.predictor is not None
        ):
            pipeline.predictor.load_state(state["predictor_state"])

        # Restore verdict counts
        if "verdict_counts" in state:
            from qstrainer.models.enums import TaskVerdict

            pipeline._verdict_counts = defaultdict(int)
            for k, v in state["verdict_counts"].items():
                try:
                    pipeline._verdict_counts[TaskVerdict[k]] = v
                except KeyError:
                    pipeline._verdict_counts[k] = v

        logger.info(
            "Restored from checkpoint %s (v%s, %d tasks)",
            latest.name,
            state.get("version", "?"),
            state.get("task_count", 0),
        )
        return True

    # ── CLI helpers ─────────────────────────────────────────
    def show_checkpoints(self) -> None:
        """Print all checkpoints in base_dir."""
        files = self._all_checkpoints()
        if not files:
            print("No checkpoints found.")
            return
        print(f"{'File':<45s} {'Size':>10s} {'Modified':<25s}")
        print("-" * 80)
        for f in files:
            stat = f.stat()
            mod = datetime.fromtimestamp(stat.st_mtime).isoformat()
            print(f"{f.name:<45s} {stat.st_size:>10,d} {mod:<25s}")

    def verify_checkpoints(self) -> None:
        """Try to load every checkpoint and report validity."""
        files = self._all_checkpoints()
        for f in files:
            try:
                with open(f, "rb") as fh:
                    state = pickle.load(fh)
                ver = state.get("version", "?")
                frames = state.get("frame_count", "?")
                print(f"  OK  {f.name}  v{ver}  frames={frames}")
            except Exception as exc:
                print(f"  ERR {f.name}  {exc}")

    def clean_old_checkpoints(self, keep: int = 1) -> None:
        """Remove all but the *keep* most recent checkpoints."""
        files = self._all_checkpoints()
        to_remove = files[:-keep] if keep > 0 else files
        for f in to_remove:
            f.unlink()
            print(f"  Removed {f.name}")
        print(f"  Kept {min(keep, len(files))} checkpoint(s)")

    # ── Internal ────────────────────────────────────────────
    def _all_checkpoints(self) -> list:
        pattern = str(self.base_dir / f"{self.CKPT_PREFIX}*{self.CKPT_SUFFIX}")
        files = sorted(Path(p) for p in glob.glob(pattern))
        return files

    def _latest_checkpoint(self) -> Path | None:
        files = self._all_checkpoints()
        return files[-1] if files else None

    def _prune_old(self) -> None:
        files = self._all_checkpoints()
        while len(files) > self.max_checkpoints:
            old = files.pop(0)
            old.unlink()
            logger.debug("Pruned old checkpoint: %s", old.name)
