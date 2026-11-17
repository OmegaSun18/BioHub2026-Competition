# BioHub Model Registry (MinIO Object Storage Service)

This directory contains the Docker configuration, startup ingestion script, test client, and `Makefile` automation for the **MinIO** model registry (`cgr.dev/chainguard/minio:latest`).

When the MinIO container starts, `load_models.py` launches the MinIO server, waits until the S3 API is healthy, and automatically registers all **20 trained cell-detection model variants** (`ckpt/last.pt` + `config.yaml`) into the `model-registry` bucket alongside structured metadata (`name`, `version`, `source`, `size_on_disk`, and `offline_accuracy: null`).

---

## Directory Structure

```text
minio/
├── Dockerfile                  # Multi-stage Chainguard MinIO + Python runtime image
├── load_models.py              # Startup entrypoint that starts MinIO and ingests all 20 models
├── Makefile                    # Automation for build, run, verify, update, and cleanup
├── README.md                   # Documentation for operators and downstream inference containers
└── minio_test/
    ├── Dockerfile              # Lightweight Python client image for testing and updates
    ├── test_minio.py           # Verification script that reads metadata and downloads weights
    └── update_model.py         # Script for updating offline_accuracy, weights, or config.yaml
```

---

## 1. Exposed Ports & Health Endpoints

| Port | Protocol | Service | Endpoint Format | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`9000`** | TCP (HTTP/S3) | **S3-Compatible Object Storage API** | `http://<host-or-ip>:9000` | Used by inference, gRPC, and evaluation containers to fetch `last.pt`, `config.yaml`, and `metadata.json`. |
| **`9001`** | TCP (HTTP) | **MinIO Web Management Console** | `http://<host-or-ip>:9001` | Browser UI for browsing buckets, inspecting object metadata, and generating scoped access keys. |

* **Liveness Health Check:** `GET http://<host-or-ip>:9000/minio/health/live` (returns HTTP `200 OK` when the S3 API is ready).

---

## 2. Access Keys, Credentials & Environment Variables

MinIO uses S3-compatible authentication where `MINIO_ROOT_USER` serves as the **Public Access Key (`AWS_ACCESS_KEY_ID`)** and `MINIO_ROOT_PASSWORD` serves as the **Secret Access Key (`AWS_SECRET_ACCESS_KEY`)**.

| Parameter | Environment Variable | Default Value | Notes |
| :--- | :--- | :--- | :--- |
| **Public Access Key** | `MINIO_ROOT_USER` | `biohubadmin` | Override at runtime with `-e MINIO_ROOT_USER=...` |
| **Secret Access Key** | `MINIO_ROOT_PASSWORD` | `biohub-minio-secret-key-2026` | Minimum 8 characters; override with `-e MINIO_ROOT_PASSWORD=...` |
| **Registry Bucket** | `MINIO_BUCKET` | `model-registry` | Created automatically on startup if it does not exist |
| **TLS / HTTPS** | `MINIO_SECURE` | `false` | Pass `secure=False` to the Python `Minio(...)` client |
| **Container Data Path** | `MINIO_DATA_DIR` | `/data` | Bind-mounted to `/mnt/data/minio_data` on the host |
| **S3 API Port** | `MINIO_API_PORT` | `9000` | Internal listener port for S3 requests |
| **Console Port** | `MINIO_CONSOLE_PORT` | `9001` | Internal listener port for the Web UI |
| **Force Re-upload** | `FORCE_REUPLOAD` | `false` | Set to `true` to overwrite cached objects even when byte sizes match |

> **Security Recommendation:** When deploying to shared servers or Kubernetes, override `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD` via runtime environment variables or a Kubernetes `Secret`, or create read-only service accounts in the Web Console (`http://localhost:9001` -> **Access Keys**).

---

## 3. Quick Start with `Makefile`

All `make` commands must be executed from inside the `minio/` directory:

```bash
cd minio
```

| Command | Description |
| :--- | :--- |
| `make all` | Runs the complete end-to-end verification pipeline: builds `biohub-minio`, starts the container, waits for model ingestion, builds and runs `minio-client-test`, and removes the MinIO container afterward. |
| `make build-minio` | Builds the `biohub-minio:latest` image from `minio/Dockerfile`. |
| `make run-minio` | Creates `/mnt/data/minio_data` (owned by UID `65532`), removes any existing `biohub-minio` container, and starts a fresh background container on ports `9000` and `9001`. |
| `make ensure-minio` | Starts `biohub-minio` only if it is not already running. |
| `make wait-minio` | Blocks until `http://127.0.0.1:9000/minio/health/live` responds and startup ingestion logs `"Ingestion complete"`. |
| `make build-test` | Builds the `minio-client-test:latest` image from `minio/minio_test/Dockerfile`. |
| `make run-test` | Runs `test_minio.py` inside a temporary container (`--rm`) connecting via `172.17.0.1:9000` to verify metadata and artifact downloads. |
| `make update-model` | Ensures MinIO is running, mounts `./minio_test` and `/mnt/data` (read-only), and executes `update_model.py` to apply your edits. |
| `make clean-minio` | Stops and removes the `biohub-minio` container while preserving all persisted data on `/mnt/data/minio_data`. |
| `make clean-all` | Stops/removes the container and deletes both `biohub-minio:latest` and `minio-client-test:latest` Docker images. |

---

## 4. Manual Build & Run Instructions (Without `make`)

### Step 1: Prepare Persistent Storage on `/mnt/data`
Chainguard containers run as non-root user `65532:65532`. Create the persistent storage directory on `/mnt/data` (keeping your root `/` OS disk free) and assign ownership to UID `65532`:

```bash
sudo mkdir -p /mnt/data/minio_data
sudo chown -R 65532:65532 /mnt/data/minio_data
```

### Step 2: Build the MinIO Docker Image
From inside `minio/`:

```bash
docker build -t biohub-minio:latest .
```

### Step 3: Start the MinIO Container
Mount `/mnt/data/minio_data` to `/data` for persistent S3 storage, and mount `/mnt/data` read-only (`:ro`) so `load_models.py` can ingest all 20 model checkpoints on startup without copying them into the image layers:

```bash
docker run -d --name biohub-minio \
  --restart unless-stopped \
  -p 9000:9000 \
  -p 9001:9001 \
  -e MINIO_ROOT_USER=biohubadmin \
  -e MINIO_ROOT_PASSWORD=biohub-minio-secret-key-2026 \
  -e MINIO_BUCKET=model-registry \
  -v /mnt/data/minio_data:/data \
  -v /mnt/data:/mnt/data:ro \
  biohub-minio:latest
```

### Step 4: Verify Startup Model Ingestion
Inspect the container logs to confirm all 20 variants were uploaded (or verified as cached on subsequent restarts):

```bash
docker logs -f biohub-minio
```

Expected log output:
```text
[minio-init] MinIO server is live on port 9000.
[minio-init] Created bucket 'model-registry'.
[minio-init] [UPLOADED] temporal_diff+dog@small | size_on_disk=... B | offline_accuracy=None
...
[minio-init] Ingestion complete: 20 uploaded, 0 already cached, 0 missing.
```

---

## 5. Bucket Layout, Registered Models & Metadata Schema

### Registered Model Variants (20 Total)
Each of the **4 model families** is stored across **5 training variants**:

* **Model Families (`<name>`):**
  * `temporal_diff+dog`
  * `temporal_stack+erode`
  * `temporal_diff+erode`
  * `raw_only+dog+edge+erode`
* **Versions (`<version>`):**
  * `small` (from `biohub_runs/grid/<name>@small`)
  * `medium` (from `biohub_runs/grid/<name>@medium`)
  * `medium-4000` (from `biohub_runs/long/<name>@medium-4000`)
  * `large` (from `biohub_runs/grid/<name>@large`)
  * `large-6000` (from `biohub_runs/long/<name>@large-6000`)

### S3 Object Hierarchy
Every variant lives under `<name>/<version>/` inside the `model-registry` bucket:

```text
model-registry/
├── temporal_diff+dog/
│   ├── small/
│   │   ├── last.pt          # PyTorch model checkpoint weights
│   │   ├── config.yaml      # Training/architecture configuration YAML
│   │   └── metadata.json    # Structured JSON metadata sidecar
│   ├── medium/
│   ├── medium-4000/
│   ├── large/
│   └── large-6000/
├── temporal_stack+erode/
│   └── ...
├── temporal_diff+erode/
│   └── ...
└── raw_only+dog+edge+erode/
    └── ...
```

### `metadata.json` Schema
Each variant's `metadata.json` preserves native JSON types (including `null` for `offline_accuracy` until populated):

```json
{
  "name": "temporal_diff+dog",
  "version": "large-6000",
  "source": "/mnt/data/biohub_working/split_dataset/biohub_runs/long/temporal_diff+dog@large-6000",
  "size_on_disk": 12345678,
  "config_size_on_disk": 2048,
  "offline_accuracy": null,
  "weights_object": "temporal_diff+dog/large-6000/last.pt",
  "config_object": "temporal_diff+dog/large-6000/config.yaml",
  "updated_at_utc": "2026-11-17T06:00:00+00:00"
}
```

In addition, S3 object metadata headers (`x-amz-meta-name`, `x-amz-meta-version`, `x-amz-meta-source`, `x-amz-meta-size-on-disk`, `x-amz-meta-offline-accuracy`) are attached directly to `last.pt` and `config.yaml`.

---

## 6. Connecting From Downstream & Future Containers

### Choosing the Right `MINIO_ENDPOINT`

| Deployment Scenario | `MINIO_ENDPOINT` Value | Notes |
| :--- | :--- | :--- |
| **Separate container on the same Linux host (default `bridge` network)** | `172.17.0.1:9000` | Routes directly through Docker's host gateway IP without needing a custom Docker network. |
| **Containers sharing a custom Docker network (`--network biohub-net`)** | `biohub-minio:9000` | Resolves automatically via Docker's embedded DNS server. |
| **Remote host / external evaluation node** | `<host-ip-or-hostname>:9000` | Requires TCP port `9000` reachable between nodes. |
| **Kubernetes cluster** | `minio.<namespace>.svc.cluster.local:9000` | Uses standard Kubernetes internal Service DNS. |

### Python Client Example (`minio>=7.2.0`)
Install `minio>=7.2.0` in your inference/gRPC container and fetch artifacts as follows:

```python
import json
import os
from minio import Minio

client = Minio(
    endpoint=os.getenv("MINIO_ENDPOINT", "172.17.0.1:9000"),
    access_key=os.getenv("MINIO_ACCESS_KEY", "biohubadmin"),
    secret_key=os.getenv("MINIO_SECRET_KEY", "biohub-minio-secret-key-2026"),
    secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
)

bucket = os.getenv("MINIO_BUCKET", "model-registry")
model_name = "temporal_diff+dog"
version = "large-6000"

# 1. Read metadata.json
resp = client.get_object(bucket, f"{model_name}/{version}/metadata.json")
metadata = json.loads(resp.read().decode("utf-8"))
resp.close()
resp.release_conn()
print("Loaded metadata:", metadata)

# 2. Download weights and configuration YAML to local container storage
os.makedirs("/models", exist_ok=True)
client.fget_object(bucket, f"{model_name}/{version}/last.pt", "/models/last.pt")
client.fget_object(bucket, f"{model_name}/{version}/config.yaml", "/models/config.yaml")
```

---

## 7. Updating Existing Models & Metadata

Use `minio_test/update_model.py` together with `make update-model` whenever you need to populate `offline_accuracy`, replace `last.pt` weights or `config.yaml`, or register a brand-new model variant on a case-by-case basis.

### Step 1: Edit `minio_test/update_model.py`
Open `minio/minio_test/update_model.py` and add entries to the `UPDATES` list. Only `name` and `version` are required—any omitted field (or field set to `KEEP`) remains untouched in MinIO:

```python
UPDATES = [
    # Case A: Metadata-only update (populates offline_accuracy without touching weights)
    {
        "name": "temporal_diff+dog",
        "version": "large-6000",
        "offline_accuracy": 0.894,
    },
    # Case B: Weights and/or config update (recomputes size_on_disk automatically)
    {
        "name": "temporal_stack+erode",
        "version": "medium-4000",
        "weights_path": "/mnt/data/biohub_working/split_dataset/biohub_runs/long/temporal_stack+erode@medium-4000/ckpt/last.pt",
        "config_path": "/mnt/data/biohub_working/split_dataset/biohub_runs/long/temporal_stack+erode@medium-4000/config.yaml",
        "offline_accuracy": None,  # Resets accuracy to null for newly trained weights
    },
]
```

### Step 2: Apply the Update via `Makefile`
From inside `minio/`:

```bash
make update-model
```

* `make update-model` starts `biohub-minio` automatically if it is stopped, waits until it is healthy, mounts `./minio_test` and `/mnt/data` (read-only) into `minio-client-test`, executes `update_model.py`, and removes the temporary client container.
* Because `./minio_test` is bind-mounted into the container at runtime, you **do not** need to rebuild `minio-client-test` after editing `UPDATES`.
* Updates are persisted immediately to `/mnt/data/minio_data` and are preserved across future container restarts.

---

## 8. Cleanup & Troubleshooting

* **Stop and remove the MinIO container (keeping stored models):**
  ```bash
  make clean-minio
  ```
* **Remove containers and built Docker images:**
  ```bash
  make clean-all
  ```
* **Permission denied writing to `/data` on startup:**
  Ensure `/mnt/data/minio_data` is owned by Chainguard's non-root UID/GID `65532:65532`:
  ```bash
  sudo chown -R 65532:65532 /mnt/data/minio_data
  ```
* **Client container cannot reach `172.17.0.1:9000`:**
  Verify `biohub-minio` is running with ports published (`docker ps | grep biohub-minio`) and confirm your Docker bridge gateway IP with `docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}'`.