# BioHub Model Registry (MinIO Storage Service)

This directory contains the Docker setup and startup ingestion script for the **MinIO** object-storage model registry (`cgr.dev/chainguard/minio:latest`). On container startup, `load_models.py` automatically registers all 20 cell-detection model variants (`last.pt` + `config.yaml`) along with their metadata (`name`, `version`, `source`, `size_on_disk`, and `offline_accuracy: null`).

---

## 1. Exposed Ports

| Port | Protocol | Purpose | URL / Endpoint |
| :--- | :--- | :--- | :--- |
| **`9000`** | TCP (HTTP/S3) | **S3-Compatible Object Storage API** (used by inference/gRPC containers to download `last.pt`, `config.yaml`, and `metadata.json`) | `http://<host-or-service>:9000` |
| **`9001`** | TCP (HTTP) | **MinIO Web Management Console** (browser UI for inspecting buckets, objects, and creating service keys) | `http://<host>:9001` |

---

## 2. Access Keys & Credentials

MinIO uses S3-compatible credentials where the **Root User** acts as the **Public Access Key (`AWS_ACCESS_KEY_ID`)** and the **Root Password** acts as the **Secret Access Key (`AWS_SECRET_ACCESS_KEY`)**.

| Parameter | Environment Variable | Default Value | Notes |
| :--- | :--- | :--- | :--- |
| **Public Access Key** | `MINIO_ROOT_USER` | `biohubadmin` | Override with `-e MINIO_ROOT_USER=...` |
| **Secret Access Key** | `MINIO_ROOT_PASSWORD` | `biohub-minio-secret-key-2026` | Must be at least 8 characters; override with `-e MINIO_ROOT_PASSWORD=...` |
| **Default Bucket** | `MINIO_BUCKET` | `model-registry` | Created automatically on startup |
| **TLS / HTTPS** | `MINIO_SECURE` | `false` | Set `secure=False` in Python `Minio(...)` client |

> **Security Tip:** In production/Kubernetes, override `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD` at runtime via Docker `-e` flags or a Kubernetes `Secret`, or generate a read-only service account key pair inside the Web Console (`http://localhost:9001` -> **Access Keys**).

---

## 3. Build & Run Instructions

### Step 1: Prepare Persistent Storage on `/mnt/data`
Chainguard containers run as non-root user `65532:65532`. Create the persistent storage directory on `/mnt/data` (to keep your root OS disk free) and assign ownership to UID `65532`:

```bash
sudo mkdir -p /mnt/data/minio_data
sudo chown -R 65532:65532 /mnt/data/minio_data