# Q-Strainer — GPU Telemetry Filtering Engine

Real-time GPU telemetry straining engine that filters 30 GB/s of raw datacenter telemetry down to actionable signals using a three-stage pipeline with QUBO-based quantum feature selection.

## Architecture

```
GPU Fleet (NVML)
    │
    ▼
┌──────────────────────────────────────────────┐
│  Q-Strainer Pipeline                         │
│                                              │
│  Stage 1: Threshold   (<0.1ms)  Hard rules   │
│  Stage 2: Statistical (<1ms)    Z-score drift │
│  Stage 3: ML Kernel   (<10ms)   OneClassSVM   │
│                                              │
│  QUBO Feature Selection (SA / D-Wave / QAOA) │
└──────────────┬───────────────────────────────┘
               │ emit only anomalies
               ▼
     Prometheus / Kafka / gRPC
               │
               ▼
          Grafana Dashboard
```

## Project Structure

```
├── src/qstrainer/           ← Production Python package
│   ├── models/              ← Frame, Buffer, Alert, Enums
│   ├── stages/              ← Threshold, Statistical, ML strainers
│   ├── solvers/             ← SA, QAOA, D-Wave, Mock QUBO solvers
│   ├── quantum/             ← Feature selector, kernel provider/detector
│   ├── features/            ← 17→63 derived feature expansion
│   ├── pipeline/            ← QStrainer orchestrator
│   ├── ingestion/           ← Synthetic generator + NVML ingestor
│   ├── emission/            ← Prometheus, gRPC, Kafka emitters
│   ├── agent/               ← Async daemon with signal handling
│   ├── checkpoint/          ← State persistence (save/restore)
│   ├── qos/                 ← QOS report, scheduler, runner
│   ├── config.py            ← YAML + env-var config loader
│   ├── benchmarks.py        ← Fleet & solver benchmarks
│   └── __main__.py          ← CLI entry point
├── config/default.yaml      ← Default configuration
├── tests/                   ← 58 tests (100% pass)
├── deploy/
│   ├── helm/qstrainer/      ← Kubernetes Helm chart
│   ├── systemd/             ← systemd service unit
│   ├── prometheus/          ← Prometheus scrape config
│   └── grafana/             ← Dashboard + provisioning
├── Dockerfile               ← Multi-stage production image
├── docker-compose.yml       ← Full stack (Q-Strainer + Prometheus + Grafana)
├── QStrainer.ipynb          ← Original research notebook
├── pyproject.toml           ← Build config, deps, tool config
├── THESIS.md                ← Research thesis
└── ROADMAP.md               ← Development roadmap
```

## Quick Start

### Install from source

```bash
git clone https://github.com/YOUR_USERNAME/qstrainer.git
cd qstrainer
pip install -e ".[dev]"
```

### Run with synthetic telemetry

```bash
# Start the agent (no GPU required)
qstrainer agent --config config/default.yaml

# Run fleet benchmark
qstrainer benchmark --fleet-size 100 --duration 60

# Compare QUBO solvers
qstrainer compare-solvers --matrix-size 20
```

### Run with Docker Compose (full stack)

```bash
# Synthetic mode (no GPU)
docker compose up

# With GPU passthrough
docker compose --profile gpu up
```

Then open Grafana at `http://localhost:3000` (admin / qstrainer).

### Run with Kubernetes

```bash
helm install qstrainer deploy/helm/qstrainer/ \
  --set config.mode=nvml \
  --set gpu.enabled=true
```

### Enable D-Wave quantum backend

```bash
pip install dwave-ocean-sdk
dwave config create --auto-token YOUR_DWAVE_LEAP_TOKEN
```

Free tier: [cloud.dwavesys.com/leap](https://cloud.dwavesys.com/leap/)

## Three-Stage Pipeline

| Stage | Method | Latency | Purpose |
|-------|--------|---------|---------|
| **1. Threshold** | Hard rules (temp ≥90°C, fatal XIDs, ECC DBE) | <0.1 ms | Catch obvious failures instantly |
| **2. Statistical** | Welford's online z-score (z > 3.0) | <1 ms | Detect drift from per-GPU baselines |
| **3. ML Kernel** | OneClassSVM on QUBO-selected features | <10 ms | Catch subtle multi-dimensional anomalies |

## Quantum Feature Selection

QUBO feature selection (Minimum Redundancy Maximum Relevance) selects the best subset of telemetry features for anomaly detection. With 63+ derived features, the search space is 2^63 ≈ 9.2×10¹⁸ — this is where D-Wave quantum annealing provides real value.

| Solver | Type | Max Size | Use Case |
|--------|------|----------|----------|
| `SimulatedAnnealingSolver` | Classical | Unlimited | Default fallback, good baseline |
| `QAOASolver` | Quantum sim | ≤20 qubits | Development & validation |
| `DWaveSolver` | Quantum HW | 5000+ qubits | Production feature selection |
| `MockQuantumSolver` | Test double | Any | Unit testing |

## Configuration

Config is loaded from YAML with environment variable overrides:

```yaml
# config/default.yaml (excerpt)
agent:
  mode: synthetic          # synthetic | nvml
  fleet_size: 8
pipeline:
  emit_threshold: 0.3
  heartbeat_interval: 100
checkpoint:
  enabled: true
  interval_sec: 300
```

Override any value via env vars: `QSTRAINER_PIPELINE__EMIT_THRESHOLD=0.5`

## Validated Results

| Metric | Value |
|--------|-------|
| Compression | ~60% data reduction |
| Recall | 100% (all failures caught) |
| Latency | 106 µs mean per frame (99% headroom at 100 Hz) |
| Fleet scale | Validated to 1000 GPUs, linear scaling |

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

58 tests covering models, stages, solvers, pipeline, features, config, and QOS.

## Observability

Prometheus metrics exposed on port 9100:

| Metric | Type | Description |
|--------|------|-------------|
| `qstrainer_frames_total` | Counter | Total frames processed |
| `qstrainer_emitted_total` | Counter | Anomalous frames emitted |
| `qstrainer_alerts_total` | Counter | Alerts by severity |
| `qstrainer_anomaly_score` | Gauge | Per-GPU anomaly score |
| `qstrainer_gpu_health` | Gauge | Per-GPU health state |
| `qstrainer_process_seconds` | Histogram | Per-frame processing latency |

Pre-built Grafana dashboard included in `deploy/grafana/dashboards/`.

## License

Proprietary. All rights reserved.



## AFTER RUNNING ANY DEMO: 

py tests/run_demo.py    # generates runs/demo_<timestamp>.json
py dashboard.py         # opens dashboard at localhost