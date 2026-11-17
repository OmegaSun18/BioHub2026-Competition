# BioHub 2026 Cell Detection & Tracking Pipeline — Kubernetes Deployment & Troubleshooting Guide

This repository runs a containerized, two-stage gRPC inference pipeline (**Cell Detection** $\rightarrow$ **Cell Tracking**) on Kubernetes (K3s), monitored with **Prometheus** and **Grafana**, and benchmarked with an **open-loop load generator** (`load_gen.py`).

---

## 1. Architecture & Storage Layout

To avoid filling the VM's root (`/`) partition, source code lives in the home directory while all heavy container state, cluster storage, and Zarr datasets reside on `/mnt/data`:

* **Code & Manifests (`~/shankar/BioHub2026-Competition`):**
* `Dockerfile` — Builds `biohub-pipeline:latest` from your local repo (`COPY . /workspace`) and compiles `proto/pipeline.proto` into `pipeline_pb2.py` and `pipeline_pb2_grpc.py`.
* `proto/pipeline.proto` — gRPC service definitions (`CellDetection.Detect` and `CellTracking.Track`).
* `server.py` — Unified gRPC model server (`STAGE_TYPE=detection|tracking`) with configurable dynamic batching (`MAX_BATCH_SIZE`, `BATCH_TIMEOUT`), dynamic module loading (`INFERENCE_CODE` with automatic fallback to `None` / null passthrough mode), and Prometheus `/metrics` on port `8000`.
* `load_gen.py` — Asynchronous open-loop load generator that deterministically samples 20% of `.zarr` datasets (`seed=8001`) and dispatches requests at a fixed `TARGET_QPS`.
* `k8s/pipeline.yaml` — Kubernetes Deployments, `ClusterIP` Services, Prometheus scrape config, and Grafana deployment.
* `scale_stage.sh` — Scripted horizontal scaling wrapper around `kubectl scale`.


* **Persistent Storage (`/mnt/data`):**
* `/mnt/data/docker` — Docker image layers and container filesystem (`data-root`).
* `/mnt/data/k3s` — K3s cluster state and pod storage (`data-dir`).
* `/mnt/data/biohub_working/split_dataset/test` — Labeled `.zarr` test dataset mounted into pods via `hostPath`.



---

## 2. Quickstart Runbook

### Step 1: Build the Docker Image

Whenever you modify `server.py`, `load_gen.py`, `proto/pipeline.proto`, or add modules under `inference_codes/`, `fiji_detectors/`, or `fiji_trackers/`, rebuild the local image:

```bash
cd ~/shankar/BioHub2026-Competition
docker build -t biohub-pipeline:latest .

```

### Step 2: Configure & Start K3s (One-Time Host Setup)

Install `socat` (required for `kubectl port-forward` with Docker) and configure K3s to store data on `/mnt/data/k3s`, use the local Docker daemon, and disable K3s's restrictive `kube-router` network policy controller:

```bash
sudo apt-get update && sudo apt-get install -y socat

sudo mkdir -p /mnt/data/k3s /etc/rancher/k3s
cat << 'EOF' | sudo tee /etc/rancher/k3s/config.yaml
data-dir: "/mnt/data/k3s"
docker: true
write-kubeconfig-mode: "0644"
disable-network-policy: true
disable:
  - traefik
EOF

curl -sfL https://get.k3s.io | sh -
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

```

### Step 3: Deploy the Pipeline, Prometheus & Grafana

Ensure `detection-service` and `tracking-service` in `k8s/pipeline.yaml` use `type: ClusterIP` (not `clusterIP: None`), then apply the manifest:

```bash
cd ~/shankar/BioHub2026-Competition
kubectl apply -f k8s/pipeline.yaml
kubectl get pods -w

```

Wait until `cell-detection`, `cell-tracking`, `prometheus`, and `grafana` all show `1/1 Running`.

### Step 4: Scale Stages via Script

Use `scale_stage.sh` to adjust replica counts without editing YAML files:

```bash
./scale_stage.sh detection 3
./scale_stage.sh tracking 2

```

### Step 5: Run the Open-Loop Load Generator

Launch `load-gen` inside the cluster with `GRPC_DNS_RESOLVER=native` and the `/mnt/data` volume mounted:

```bash
kubectl delete pod load-gen --ignore-not-found

kubectl run load-gen --image=biohub-pipeline:latest --image-pull-policy=IfNotPresent \
  --restart=Never \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "load-gen",
        "image": "biohub-pipeline:latest",
        "imagePullPolicy": "IfNotPresent",
        "command": ["python", "/workspace/load_gen.py"],
        "env": [
          {"name": "GRPC_DNS_RESOLVER", "value": "native"},
          {"name": "DETECTION_ADDR", "value": "detection-service.default.svc.cluster.local:50051"},
          {"name": "TRACKING_ADDR", "value": "tracking-service.default.svc.cluster.local:50051"},
          {"name": "TARGET_QPS", "value": "4.0"},
          {"name": "TEST_DATA_DIR", "value": "/mnt/data/biohub_working/split_dataset/test"}
        ],
        "volumeMounts": [{"name": "data-vol", "mountPath": "/mnt/data", "readOnly": true}]
      }],
      "volumes": [{"name": "data-vol", "hostPath": {"path": "/mnt/data", "type": "Directory"}}]
    }
  }'

kubectl logs -f load-gen

```

### Step 6: Access Prometheus & Grafana

Forward both services to localhost:

```bash
kubectl port-forward svc/prometheus 9090:9090 >/dev/null 2>&1 &
kubectl port-forward svc/grafana 3000:3000 >/dev/null 2>&1 &

# Verify all scrape targets are healthy ("up")
curl -s http://127.0.0.1:9090/api/v1/targets | grep -o '"health":"[^"]*"'

```

If connecting over SSH from a local machine, forward port `3000`:

```bash
ssh -L 3000:127.0.0.1:3000 cc@<VM_IP>

```

Open `http://localhost:3000` (`admin` / `admin`), add `http://prometheus:9090` as a Prometheus data source, and query:

* **p50 / p95 / p99 Latency:** `histogram_quantile(0.95, sum(rate(request_latency_seconds_bucket[1m])) by (le, stage))`
* **Throughput (videos/sec):** `sum(rate(request_count_total[1m])) by (stage)`
* **Queue Length:** `sum(queue_length) by (stage)`
* **CPU Usage:** `sum(rate(container_cpu_usage_seconds_total{container="server"}[1m])) by (pod)`
* **Assertion Pass Rate:** `sum(rate(assertion_pass_total[1m])) / (sum(rate(assertion_pass_total[1m])) + sum(rate(assertion_fail_total[1m])))`

---

## 3. Kubernetes & Networking Fixes Applied

Below are the specific issues encountered while bringing up K3s on this VM and the exact fixes that resolved them:

| Issue Encountered | Root Cause | Working Fix |
| --- | --- | --- |
| **`StatusCode.UNAVAILABLE` / `No route to host (113)**` when `load-gen` connected to `10.42.0.x:50051` | Headless Services (`clusterIP: None`) forced gRPC to connect directly to pod IPs on the same `10.42.0.0/24` (`cni0`) subnet via L2 bridging, where K3s's `kube-router` firewall rejected bridged frames. | Changed `detection-service` and `tracking-service` from `clusterIP: None` to **`type: ClusterIP`**, routing gRPC over the `10.43.0.0/16` L3 service network. |
| **Prometheus unable to scrape pod `:8000` metrics** over `10.42.0.x` | K3s's embedded `kube-router` network policy controller continuously re-inserted `KUBE-ROUTER-FORWARD` at the top of `iptables FORWARD`, blocking direct pod-to-pod scrapes. | Added **`disable-network-policy: true`** to `/etc/rancher/k3s/config.yaml` and restarted K3s (`sudo systemctl restart k3s`). |
| **`C-ares status is not ARES_SUCCESS qtype=AAAA`** DNS errors in `grpcio` | `grpcio`'s default `c-ares` DNS resolver failed on IPv6 (`AAAA`) queries against Kubernetes internal service names. | Set **`GRPC_DNS_RESOLVER=native`** in the client pod environment so Python uses standard Linux glibc DNS resolution. |
| **`unable to do port forwarding: socat not found`** on `kubectl port-forward` | Running K3s with the external Docker runtime (`docker: true`) delegates port-forwarding tunnels to the host's `socat` binary, which is not pre-installed on minimal Ubuntu images. | Installed `socat` on the VM host via **`sudo apt-get install -y socat`**. |
| **Root disk (`/`) exhaustion & image pull errors** | Default K3s stores pod layers in `/var/lib/rancher/k3s` on `/` and defaults to `containerd` (which cannot see images built in `/mnt/data/docker`). | Configured K3s with **`data-dir: "/mnt/data/k3s"`** and **`docker: true`**, and set **`imagePullPolicy: IfNotPresent`** in `k8s/pipeline.yaml`. |
| **Container startup failures when model scripts are absent** | Importing an unimplemented `INFERENCE_CODE` module threw `ModuleNotFoundError` and crashed pods. | Updated `server.py` (`_load_inference_code`) to catch missing modules or `"null"` values and fall back cleanly to **`self.model = None`** (passthrough mode). |