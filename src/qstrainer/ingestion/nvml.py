"""NVML Ingestor — capture GPU hardware state as ComputeTask probes.

This module polls NVIDIA GPUs via the NVML (NVIDIA Management Library)
C API through the pynvml Python bindings.  Each call to poll() captures
one ComputeTask per GPU representing the current workload snapshot.

The hardware telemetry (utilisation, memory, thermals) is mapped into
ComputeTask fields so Q-Strainer can decide whether ongoing GPU work
is productive or redundant.

Install: pip install pynvml (or: pip install qstrainer[gpu])
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

from qstrainer.models.frame import ComputeTask
from qstrainer.models.enums import ComputePhase, JobType

logger = logging.getLogger(__name__)


class NVMLIngestor:
    """Poll GPUs via pynvml and yield ComputeTask probes.

    Lifecycle:
        ingestor = NVMLIngestor(node_id="node-01")
        ingestor.init()
        tasks = ingestor.poll()   # one task per GPU
        ...
        ingestor.shutdown()
    """

    def __init__(self, node_id: str = "node-00") -> None:
        self.node_id = node_id
        self._handles: List = []
        self._gpu_ids: List[str] = []
        self._device_count: int = 0
        self._initialised: bool = False
        self._step_counter: Dict[str, int] = {}

    # ── Lifecycle ───────────────────────────────────────────
    def init(self) -> int:
        """Initialise NVML and enumerate GPUs.  Returns GPU count."""
        try:
            import pynvml
        except ImportError as e:
            raise ImportError(
                "pynvml is required for NVML ingestion. "
                "Install with: pip install pynvml  (or pip install qstrainer[gpu])"
            ) from e

        pynvml.nvmlInit()
        self._device_count = pynvml.nvmlDeviceGetCount()
        self._handles = []
        self._gpu_ids = []

        for i in range(self._device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            self._handles.append(handle)
            try:
                uuid = pynvml.nvmlDeviceGetUUID(handle)
            except Exception:
                uuid = f"GPU-{i:04d}"
            self._gpu_ids.append(uuid)
            self._step_counter[uuid] = 0

        self._initialised = True
        logger.info(
            "NVML initialised: %d GPUs on %s", self._device_count, self.node_id
        )
        return self._device_count

    def shutdown(self) -> None:
        """Clean shutdown of NVML."""
        if self._initialised:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._initialised = False
            logger.info("NVML shutdown complete.")

    # ── Polling ─────────────────────────────────────────────
    def poll(self) -> List[ComputeTask]:
        """Poll all GPUs once.  Returns one ComputeTask per GPU."""
        import pynvml

        if not self._initialised:
            raise RuntimeError("NVMLIngestor not initialised. Call init() first.")

        ts = time.time()
        tasks: List[ComputeTask] = []

        for idx, handle in enumerate(self._handles):
            try:
                task = self._read_gpu(idx, handle, ts)
                tasks.append(task)
            except Exception as exc:
                logger.warning(
                    "Failed to read GPU %d (%s): %s",
                    idx, self._gpu_ids[idx], exc,
                )

        return tasks

    def _read_gpu(
        self, idx: int, handle: object, ts: float
    ) -> ComputeTask:
        """Read hardware state from a single GPU and wrap as ComputeTask."""
        import pynvml

        gpu_id = self._gpu_ids[idx]
        self._step_counter[gpu_id] = self._step_counter.get(gpu_id, 0) + 1

        # Utilisation
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        sm_util = util.gpu / 100.0
        mem_util = util.memory / 100.0

        # Memory
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_used_gb = mem_info.used / (1024**3)
        mem_total_gb = mem_info.total / (1024**3)

        # Power
        try:
            power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except pynvml.NVMLError:
            power_draw = 0.0

        # Clock
        try:
            clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        except pynvml.NVMLError:
            clock = 0

        # Process count → proxy for active workloads
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            proc_count = len(procs)
        except pynvml.NVMLError:
            proc_count = 0

        # Map hardware state to ComputeTask fields:
        #   - flop_utilization = SM utilisation (direct proxy)
        #   - throughput = clock speed normalised
        #   - memory_footprint = actual VRAM usage
        #   - convergence_score = 0 (unknown from hardware)
        #   - data_similarity = 0 (unknown from hardware)
        #   - loss/gradient fields = 0 (hardware probe, not training step)
        return ComputeTask(
            timestamp=ts,
            task_id=f"{gpu_id}-step-{self._step_counter[gpu_id]}",
            gpu_id=gpu_id,
            job_id=f"hw-probe-{gpu_id}",
            step_number=self._step_counter[gpu_id],
            loss=0.0,
            loss_delta=0.0,
            gradient_norm=0.0,
            gradient_variance=0.0,
            learning_rate=0.0,
            batch_size=0,
            epoch=0,
            epoch_progress=0.0,
            estimated_flops=sm_util * 1e12,  # rough TFLOP proxy
            estimated_time_s=1.0 / max(clock, 1),
            memory_footprint_gb=mem_used_gb,
            compute_phase=ComputePhase.FORWARD,
            job_type=JobType.INFERENCE,
            convergence_score=0.0,
            param_update_magnitude=0.0,
            data_similarity=0.0,
            flop_utilization=sm_util,
            throughput_samples_per_sec=clock * proc_count,
            node_id=self.node_id,
            tags={"source": "nvml", "mem_util": f"{mem_util:.2f}",
                  "power_w": f"{power_draw:.0f}"},
        )

    @property
    def gpu_count(self) -> int:
        return self._device_count

    @property
    def gpu_ids(self) -> List[str]:
        return list(self._gpu_ids)
