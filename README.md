# QOS — Quantum Operating System for GPU Datacenters

Real-time GPU telemetry straining engine that filters 30 GB/s of raw telemetry down to actionable signals using a three-stage pipeline with QUBO-based quantum feature selection.

## Project Structure

```
qos/
├── QStrainer.ipynb          ← Q-Strainer: GPU telemetry filtering engine (main)
├── SuperQuantumEngine.ipynb ← Original quantum scheduler (reference)
├── QOS.ipynb                ← QOS notebook (legacy)
├── opus_qengine_core.ipynb  ← QEngine core exploration
├── v1.0QuantumEngine.ipynb  ← v1.0 quantum engine (legacy)
├── qos/                     ← Python package (scheduler modules)
│   ├── __init__.py
│   ├── scheduler.py
│   ├── runner.py
│   └── report.py
├── examples/
│   └── bell.py
├── runs/                    ← Training datasets
│   ├── qengine_dataset.json
│   └── qaoa_training_dataset.json
├── requirements.txt
├── THESIS.md
├── ROADMAP.md
└── README.md
```

## Q-Strainer — What It Does

**Problem:** A 1000-GPU datacenter at 100 Hz produces ~500 KB/s of telemetry. Most of it is noise — healthy GPUs reporting normal values. Monitoring backends drown in data, and real failures get buried.

**Solution:** Q-Strainer sits between GPU telemetry sources (NVML) and monitoring backends (Prometheus/Grafana), straining out noise and emitting only actionable signals.

### Three-Stage Pipeline

| Stage | Method | Latency | Purpose |
|-------|--------|---------|---------|
| **1. Threshold** | Hard rules (temp ≥90°C, fatal XIDs, ECC DBE) | <0.1 ms | Catch obvious failures instantly |
| **2. Statistical** | Welford's online z-score (z > 3.0) | <1 ms | Detect drift from per-GPU baselines |
| **3. ML Kernel** | OneClassSVM on QUBO-selected features | <10 ms | Catch subtle multi-dimensional anomalies |

### Quantum Component

QUBO feature selection (Minimum Redundancy Maximum Relevance) selects the best subset of telemetry features for anomaly detection. With 63+ derived features, the search space is 2^63 ≈ 9.2×10¹⁸ — this is where D-Wave quantum annealing provides real value.

- **Classical fallback:** Simulated Annealing solver (works everywhere, same interface)
- **Quantum backend:** D-Wave Advantage via Ocean SDK (ready, needs Leap token)

### Validated Results

| Metric | Value |
|--------|-------|
| Compression | ~60% data reduction |
| Recall | 100% (all failures caught) |
| Latency | 106 µs mean per frame (99% headroom at 100 Hz) |
| Fleet scale | Validated to 1000 GPUs, linear scaling |

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/qos.git
cd qos

# Install dependencies
pip install -r requirements.txt

# Open the notebook
jupyter notebook QStrainer.ipynb
```

Run all cells top-to-bottom. No GPU or quantum hardware needed — synthetic telemetry and SA solver are included.

### Optional: Enable D-Wave Quantum Backend

```bash
pip install dwave-ocean-sdk
dwave config create --auto-token YOUR_DWAVE_LEAP_TOKEN
```

Free tier available at [cloud.dwavesys.com/leap](https://cloud.dwavesys.com/leap/)

## Roadmap

| Week | Milestone |
|------|-----------|
| 1-2 | Deploy NVML agent on test GPU node, validate live telemetry |
| 3 | Collect labeled anomaly data from real GPU incidents |
| 4 | Train production anomaly model on real data |
| 5 | Get D-Wave Leap access, run 63-variable QUBO on QPU |
| 6 | A/B test: D-Wave vs SA feature selection accuracy |
| 7-8 | Deploy to first datacenter cluster |

## License

Proprietary. All rights reserved.
