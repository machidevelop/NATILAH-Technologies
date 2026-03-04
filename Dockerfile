# ── Q-Strainer production image ──────────────────────────────────────────────
# Multi-stage build: builder + slim runtime
# Usage:
#   docker build -t qstrainer:latest .
#   docker run --gpus all -v /etc/qstrainer:/etc/qstrainer qstrainer:latest agent
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ───────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/

# Build wheel
RUN pip install --no-cache-dir build && \
    python -m build --wheel --outdir /build/dist

# ── Stage 2: runtime ──────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL maintainer="qstrainer-team"
LABEL description="Q-Strainer GPU Telemetry Filtering Engine"
LABEL version="0.2.0"

# Create non-root user
RUN groupadd -r qstrainer && useradd -r -g qstrainer -d /home/qstrainer -s /bin/bash qstrainer

WORKDIR /app

# Copy wheel from builder and install
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -f /tmp/*.whl

# Install optional NVML support (pynvml)
RUN pip install --no-cache-dir pynvml || true

# Copy default config
COPY config/default.yaml /etc/qstrainer/config.yaml

# Create directories for data and checkpoints
RUN mkdir -p /var/lib/qstrainer/checkpoints /var/log/qstrainer && \
    chown -R qstrainer:qstrainer /var/lib/qstrainer /var/log/qstrainer /etc/qstrainer

# Prometheus metrics port
EXPOSE 9100

# Health check endpoint (Prometheus /metrics)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9100/metrics')" || exit 1

USER qstrainer

# Environment defaults
ENV QSTRAINER_CONFIG=/etc/qstrainer/config.yaml
ENV QSTRAINER_CHECKPOINT__DIR=/var/lib/qstrainer/checkpoints
ENV QSTRAINER_AGENT__MODE=nvml

ENTRYPOINT ["qstrainer"]
CMD ["agent", "--config", "/etc/qstrainer/config.yaml"]
