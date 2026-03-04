# Q-Strainer Roadmap

## v0.2.0 — Production Foundation (CURRENT)

### ✅ P0 — Core Extraction & Packaging
- [x] Extract notebook into `src/qstrainer/` Python package
- [x] `pyproject.toml` with setuptools build, CLI entry point, dependency groups
- [x] YAML config system with env-var overrides (`QSTRAINER_SECTION__KEY`)
- [x] Default config (`config/default.yaml`)
- [x] CLI: `qstrainer agent | benchmark | compare-solvers | checkpoint`

### ✅ P0 — Three-Stage Pipeline
- [x] `ThresholdStrainer` — static limit checks (temp, power, ECC, VRAM, utilization, fan)
- [x] `StatisticalStrainer` — z-score anomaly detection with rolling baselines
- [x] `KernelAnomalyDetector` — Isolation Forest ML stage
- [x] `QStrainer` pipeline orchestrator with emit/heartbeat logic

### ✅ P0 — Models & Data
- [x] `TelemetryFrame` (17-feature dataclass) with `to_vector()`, `feature_names()`
- [x] `TelemetryBuffer` — per-GPU deque with windowed matrix extraction
- [x] `Alert` / `StrainedOutput` structured outputs
- [x] `GPUHealth` / `AlertSeverity` / `GPUType` enums with per-type thermal specs

### ✅ P0 — Solvers
- [x] `QUBOSolverBase` ABC + `QUBOResult` dataclass
- [x] `SimulatedAnnealingSolver` — classical baseline
- [x] `QAOASolver` — numpy-based QAOA simulation (≤20 qubits)
- [x] `DWaveSolver` — D-Wave Ocean SDK integration
- [x] `MockQuantumSolver` — testing double

### ✅ P1 — Quantum Feature Selection
- [x] `QUBOFeatureSelector` — mRMR-based QUBO encoding for feature selection
- [x] `QuantumKernelProvider` — ZZ feature map statevector kernel
- [x] `QuantumKernelDetector` — quantum kernel + OneClassSVM pipeline
- [x] `DerivedFeatureExtractor` — 17→63 feature expansion (cross-products, ratios, rolling)

### ✅ P1 — Ingestion
- [x] `SyntheticTelemetryGenerator` — healthy / degrading / failing profiles + fleet generation
- [x] `NVMLIngestor` — real GPU polling via pynvml (init/poll/shutdown lifecycle)

### ✅ P1 — Emission & Observability
- [x] `PrometheusEmitter` — counters, gauges, histograms for all pipeline metrics
- [x] `GRPCEmitter` — gRPC stub (proto compilation required for production)
- [x] `KafkaEmitter` — confluent-kafka producer
- [x] Grafana dashboard (`deploy/grafana/dashboards/qstrainer-overview.json`)
  - Fleet overview stats, throughput/latency charts, per-GPU anomaly scores, health status, alert rate by severity

### ✅ P1 — Reliability
- [x] `CheckpointManager` — pickle-based save/restore with FIFO pruning & verification
- [x] `QStrainerDaemon` — async agent with graceful shutdown signal handling

### ✅ P1 — Quality of Service
- [x] `QOSReport` — structured benchmark results with serialization
- [x] `QOSScheduler` — solver registry with preference-based selection
- [x] `QOSRunner` — benchmark execution, solver comparison, history tracking

### ✅ P2 — Deployment
- [x] `Dockerfile` — multi-stage build (builder + slim runtime), non-root, HEALTHCHECK
- [x] `docker-compose.yml` — Q-Strainer + Prometheus + Grafana stack (GPU profile)
- [x] `deploy/systemd/qstrainer.service` — hardened systemd unit
- [x] Helm chart (`deploy/helm/qstrainer/`)
  - Deployment, Service, ConfigMap, PVC, ServiceAccount, ServiceMonitor
  - GPU support via NVIDIA device plugin, Prometheus ServiceMonitor

### ✅ P2 — Testing
- [x] 58 tests across 7 test modules (100% pass rate)
  - Models, stages, solvers, pipeline, features, config, QOS
- [x] `conftest.py` with reusable fixtures

---

## v0.3.0 — Hardening ✅

### ✅ P1 — Extended Testing
- [x] Integration tests — full pipeline E2E with synthetic data (`tests/test_integration.py`)
  - Healthy compression >50%, failing recall >90%, degrading detection, multi-GPU isolation
- [x] Property-based testing (Hypothesis) for frame/buffer/pipeline invariants (`tests/test_properties.py`)
- [x] Load & perf benchmarks — latency, throughput, buffer, feature extraction (`tests/test_benchmarks.py`)
- [x] CI/CD pipeline — GitHub Actions: lint → test → benchmark → typecheck → build → docker (`.github/workflows/ci.yml`)
  - Matrix: ubuntu/windows × Python 3.10–3.13

### ✅ P1 — Observability
- [x] Structured JSON logging with correlation IDs (`src/qstrainer/logging.py`)
  - JSONFormatter, HumanFormatter, ContextVar-based `correlation_id` / `gpu_id` / `node_id`
  - CLI `--json-logs` flag on agent subcommand
- [x] OpenTelemetry tracing — spans per pipeline stage (`src/qstrainer/tracing.py`)
  - `init_tracing()`, `trace_stage()` context manager, no-op safe when SDK absent
  - Integrated into `QStrainer.process_frame()` (threshold, statistical, ml stages)
- [x] Alert routing — webhook / Slack / PagerDuty (`src/qstrainer/alerting.py`)
  - `AlertRouter.from_config()`, cooldown dedup, severity filtering

### ✅ P2 — Security
- [x] mTLS for gRPC emitter — TLS / mTLS channel credentials with ca/client cert/key
- [x] Kafka SASL/SSL authentication — SCRAM, PLAIN, SSL with cert auth
- [x] Config secret management — `env://`, `file://`, `sops://`, `vault://` secret refs (`src/qstrainer/secrets.py`)
  - Auto-resolved during `load_config()`, walks full config tree

### ✅ P2 — Performance
- [x] NumPy vectorized batch processing — `QStrainer.process_batch()` with matrix-based scoring
- [x] Memory profiling & optimization — `MemoryProfiler` with tracemalloc, RSS tracking (`src/qstrainer/profiling.py`)
  - Integrated into daemon loop with periodic snapshots and shutdown report

### Bug Fixes
- [x] Fixed `result.frame.gpu_id` → `result.gpu_id` in all three emitters (Prometheus, gRPC, Kafka)

---

## v0.4.0 — Scale ✅

### ✅ P2 — Multi-Node
- [x] Redis-backed shared buffer (`src/qstrainer/distributed/redis_buffer.py`)
  - Sorted sets per GPU scored by timestamp, push/push_batch (pipelined), get_window/get_matrix
  - Fleet queries (gpu_ids, total_frames, frame_count), GPU metadata, TTL cleanup
- [x] Leader election for checkpoint coordination (`src/qstrainer/distributed/leader.py`)
  - Redis distributed lock with TTL heartbeat, background renewal thread, context manager
- [x] Horizontal autoscaling (`src/qstrainer/distributed/autoscaler.py`)
  - Throughput-based scaling (SCALE_UP / SCALE_DOWN / HOLD), cooldown, min/max replicas
  - Configurable target FPS per replica, window averaging, from_config()

### ✅ P2 — ML Pipeline
- [x] Online model retraining with drift detection (`src/qstrainer/ml/drift.py`)
  - DriftDetector: PSI per-feature distribution comparison + Page-Hinkley sustained-shift test
  - OnlineRetrainer: periodic drift checks, healthy-vector accumulation, forced retraining interval
- [x] Model versioning and A/B testing (`src/qstrainer/ml/versioning.py`)
  - ModelRegistry: register/promote champion, set_challenger, FIFO pruning preserving champion
  - ABTestRunner: shadow-mode per-frame comparisons, variance-based promote/dismiss decision
- [x] Feature store integration (`src/qstrainer/ml/feature_store.py`)
  - FeatureStore: registration, dependency resolution, per-frame caching, bulk materialisation
  - RedisFeatureCache: distributed cache layer for multi-agent deployment

### ✅ P3 — Quantum
- [x] IBM Quantum runtime integration (`src/qstrainer/solvers/qiskit_runtime.py`)
  - QiskitRuntimeSolver: QAOA circuit builder (RZ/CX cost + RX mixer), Aer sim + IBM Runtime dispatch
  - Supports up to 127 qubits, configurable shots, optimization level, from_config()
- [x] Hybrid classical/quantum solver scheduling (updated `src/qstrainer/qos/scheduler.py`)
  - n≤18 → QAOA sim, 18<n≤127 → Qiskit Runtime (if available), n>127 → D-Wave, fallback SA
- [x] Quantum advantage benchmarking suite (`src/qstrainer/quantum/advantage.py`)
  - QuantumAdvantageBenchmark: multi-solver comparison, brute-force ground truth (n≤20)
  - BenchmarkReport: summary table with energy gaps & times, winner analysis, to_dict()

### Testing
- [x] 28 new tests in `tests/test_scale.py` — drift detection, model versioning, A/B testing,
  feature store, autoscaler, quantum advantage benchmarks, hybrid scheduling
- [x] **116 total tests, all passing** (88 v0.3.0 + 28 v0.4.0)

---

## v1.0.0 — Production GA

- [ ] API stability guarantee (semantic versioning)
- [ ] Full documentation (Sphinx/MkDocs)
- [ ] Backward-compatible config migration
- [ ] Certified container images (NVIDIA NGC catalog)
- [ ] Datacenter deployment guide
