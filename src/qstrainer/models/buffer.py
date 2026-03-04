"""WorkloadBuffer — per-GPU/job ring buffer for compute tasks."""

from __future__ import annotations

from collections import deque
from typing import Dict, List

import numpy as np

from qstrainer.models.frame import ComputeTask, N_BASE_FEATURES


class WorkloadBuffer:
    """Ring buffer holding the last N compute tasks per GPU.

    Used by the strainer stages to look at recent compute history
    and decide whether new tasks are redundant or converged.
    """

    def __init__(self, max_tasks_per_gpu: int = 1000) -> None:
        self.max_tasks = max_tasks_per_gpu
        self._buffers: Dict[str, deque] = {}
        self._task_count: int = 0

    def push(self, task: ComputeTask) -> None:
        if task.gpu_id not in self._buffers:
            self._buffers[task.gpu_id] = deque(maxlen=self.max_tasks)
        self._buffers[task.gpu_id].append(task)
        self._task_count += 1

    def get_window(self, gpu_id: str, n_tasks: int) -> List[ComputeTask]:
        """Last *n_tasks* for a GPU."""
        buf = self._buffers.get(gpu_id, deque())
        return list(buf)[-n_tasks:]

    def get_matrix(self, gpu_id: str, n_tasks: int) -> np.ndarray:
        """Last *n_tasks* as an (n × 15) feature matrix."""
        tasks = self.get_window(gpu_id, n_tasks)
        if not tasks:
            return np.empty((0, N_BASE_FEATURES))
        return np.vstack([t.to_vector() for t in tasks])

    @property
    def gpu_ids(self) -> List[str]:
        return list(self._buffers.keys())

    @property
    def total_tasks(self) -> int:
        return self._task_count

    def clear(self, gpu_id: str | None = None) -> None:
        if gpu_id:
            self._buffers.pop(gpu_id, None)
        else:
            self._buffers.clear()
            self._task_count = 0
