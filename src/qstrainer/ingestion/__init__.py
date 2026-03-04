"""Workload ingestion: NVML (real GPUs) and synthetic (testing)."""

from qstrainer.ingestion.synthetic import SyntheticTelemetryGenerator

# NVMLIngestor requires pynvml — lazy import to avoid blocking tests
def __getattr__(name: str):
    if name == "NVMLIngestor":
        from qstrainer.ingestion.nvml import NVMLIngestor
        return NVMLIngestor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["SyntheticTelemetryGenerator", "NVMLIngestor"]
