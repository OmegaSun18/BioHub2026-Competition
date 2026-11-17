# Stage 1: Install the official MinIO Python SDK into an isolated virtual environment
FROM cgr.dev/chainguard/python:latest-dev AS builder

USER root
WORKDIR /build
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir "minio>=7.2.0" && \
    chown -R 65532:65532 /opt/venv

# Stage 2: Combine Chainguard's zero-CVE distroless Python runtime + MinIO server binary
FROM cgr.dev/chainguard/python:latest

# Copy the MinIO server binary directly from cgr.dev/chainguard/minio:latest
COPY --from=cgr.dev/chainguard/minio:latest /usr/bin/minio /usr/bin/minio

# Copy the Python virtual environment with the `minio` SDK
COPY --from=builder --chown=65532:65532 /opt/venv /opt/venv

# Default MinIO credentials & bucket configuration (override at `docker run` via -e)
ENV PATH="/opt/venv/bin:/usr/bin:/bin" \
    PYTHONUNBUFFERED=1 \
    MINIO_ROOT_USER="biohubadmin" \
    MINIO_ROOT_PASSWORD="biohub-minio-secret-key-2026" \
    MINIO_BUCKET="model-registry" \
    MINIO_DATA_DIR="/data" \
    MINIO_API_PORT="9000" \
    MINIO_CONSOLE_PORT="9001"

WORKDIR /app
COPY --chown=65532:65532 load_models.py /app/load_models.py

# Expose S3 API port (9000) for model downloads and Web Console port (9001)
EXPOSE 9000 9001

# Launch MinIO server, wait until healthy, upload all 20 models, and keep MinIO running
ENTRYPOINT ["/opt/venv/bin/python", "/app/load_models.py"]