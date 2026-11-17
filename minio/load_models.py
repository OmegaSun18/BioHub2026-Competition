import io
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from minio import Minio
from minio.error import S3Error

# All 20 trained model run directories (each containing `ckpt/last.pt` and `config.yaml`)
MODEL_RUN_DIRS: List[str] = [
    # 1. temporal_diff+dog
    "/mnt/data/biohub_working/split_dataset/biohub_runs/grid/temporal_diff+dog@small",
    "/mnt/data/biohub_working/split_dataset/biohub_runs/grid/temporal_diff+dog@medium",
    "/mnt/data/biohub_working/split_dataset/biohub_runs/long/temporal_diff+dog@medium-4000",
    "/mnt/data/biohub_working/split_dataset/biohub_runs/grid/temporal_diff+dog@large",
    "/mnt/data/biohub_working/split_dataset/biohub_runs/long/temporal_diff+dog@large-6000",
    # 2. temporal_stack+erode
    "/mnt/data/biohub_working/split_dataset/biohub_runs/grid/temporal_stack+erode@small",
    "/mnt/data/biohub_working/split_dataset/biohub_runs/grid/temporal_stack+erode@medium",
    "/mnt/data/biohub_working/split_dataset/biohub_runs/long/temporal_stack+erode@medium-4000",
    "/mnt/data/biohub_working/split_dataset/biohub_runs/grid/temporal_stack+erode@large",
    "/mnt/data/biohub_working/split_dataset/biohub_runs/long/temporal_stack+erode@large-6000",
    # 3. temporal_diff+erode
    "/mnt/data/biohub_working/split_dataset/biohub_runs/grid/temporal_diff+erode@small",
    "/mnt/data/biohub_working/split_dataset/biohub_runs/grid/temporal_diff+erode@medium",
    "/mnt/data/biohub_working/split_dataset/biohub_runs/long/temporal_diff+erode@medium-4000",
    "/mnt/data/biohub_working/split_dataset/biohub_runs/grid/temporal_diff+erode@large",
    "/mnt/data/biohub_working/split_dataset/biohub_runs/long/temporal_diff+erode@large-6000",
    # 4. raw_only+dog+edge+erode
    "/mnt/data/biohub_working/split_dataset/biohub_runs/grid/raw_only+dog+edge+erode@small",
    "/mnt/data/biohub_working/split_dataset/biohub_runs/grid/raw_only+dog+edge+erode@medium",
    "/mnt/data/biohub_working/split_dataset/biohub_runs/long/raw_only+dog+edge+erode@medium-4000",
    "/mnt/data/biohub_working/split_dataset/biohub_runs/grid/raw_only+dog+edge+erode@large",
    "/mnt/data/biohub_working/split_dataset/biohub_runs/long/raw_only+dog+edge+erode@large-6000",
]


def parse_run_dir(run_dir: str) -> Tuple[str, str, str]:
    """
    Parse `<...>/biohub_runs/<suite>/<name>@<version>` into `(name, version, source)`.
    Example:
      .../grid/temporal_diff+dog@small -> ("temporal_diff+dog", "small", "<full_path>")
    """
    path = Path(run_dir)
    folder = path.name  # e.g., "temporal_diff+dog@medium-4000"
    if "@" in folder:
        name, version = folder.split("@", 1)
    else:
        name, version = folder, "v1"
    return name, version, str(path)


def wait_for_minio(api_port: str, timeout_sec: int = 60) -> None:
    """Poll MinIO's liveness probe until the S3 API is ready to accept uploads."""
    health_url = f"http://127.0.0.1:{api_port}/minio/health/live"
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=2) as resp:
                if resp.status == 200:
                    print(f"[minio-init] MinIO server is live on port {api_port}.", flush=True)
                    return
        except (urllib.error.URLError, ConnectionResetError, TimeoutError):
            time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for MinIO to become healthy at {health_url}")


def get_existing_accuracy(client: Minio, bucket: str, metadata_key: str) -> Optional[float]:
    """Preserve previously populated `offline_accuracy` across container restarts."""
    try:
        resp = client.get_object(bucket, metadata_key)
        try:
            existing = json.loads(resp.read().decode("utf-8"))
            return existing.get("offline_accuracy", None)
        finally:
            resp.close()
            resp.release_conn()
    except S3Error:
        return None


def object_matches_size(client: Minio, bucket: str, object_key: str, expected_size: int) -> bool:
    """Return True if object already exists in MinIO with the exact same byte size."""
    try:
        stat = client.stat_object(bucket, object_key)
        return stat.size == expected_size
    except S3Error:
        return False


def ingest_models(
    client: Minio,
    bucket: str,
    force_reupload: bool = False,
) -> None:
    """Upload all model weights, config.yaml files, and metadata.json records into MinIO."""
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        print(f"[minio-init] Created bucket '{bucket}'.", flush=True)
    else:
        print(f"[minio-init] Using existing bucket '{bucket}'.", flush=True)

    uploaded_count = 0
    skipped_count = 0
    missing_count = 0

    for run_dir in MODEL_RUN_DIRS:
        name, version, source = parse_run_dir(run_dir)
        weights_path = Path(run_dir) / "ckpt" / "last.pt"
        config_path = Path(run_dir) / "config.yaml"

        if not weights_path.is_file() or not config_path.is_file():
            print(
                f"[minio-init] WARNING: Missing files for {name}@{version} "
                f"(weights={weights_path.exists()}, config={config_path.exists()}). Skipping.",
                flush=True,
            )
            missing_count += 1
            continue

        weights_size = weights_path.stat().st_size
        config_size = config_path.stat().st_size

        prefix = f"{name}/{version}"
        weights_key = f"{prefix}/last.pt"
        config_key = f"{prefix}/config.yaml"
        metadata_key = f"{prefix}/metadata.json"

        # Preserve any offline_accuracy score already recorded in MinIO; default is None (JSON null)
        offline_accuracy = get_existing_accuracy(client, bucket, metadata_key)

        s3_metadata = {
            "name": name,
            "version": version,
            "source": source,
            "size-on-disk": str(weights_size),
            "offline-accuracy": "null" if offline_accuracy is None else str(offline_accuracy),
        }

        # 1. Upload model checkpoint (`last.pt`) if new or changed
        if force_reupload or not object_matches_size(client, bucket, weights_key, weights_size):
            client.fput_object(
                bucket_name=bucket,
                object_name=weights_key,
                file_path=str(weights_path),
                content_type="application/octet-stream",
                metadata=s3_metadata,
            )
            action = "UPLOADED"
            uploaded_count += 1
        else:
            action = "CACHED"
            skipped_count += 1

        # 2. Upload corresponding `config.yaml`
        if force_reupload or not object_matches_size(client, bucket, config_key, config_size):
            client.fput_object(
                bucket_name=bucket,
                object_name=config_key,
                file_path=str(config_path),
                content_type="application/x-yaml",
                metadata=s3_metadata,
            )

        # 3. Write structured `metadata.json` sidecar (with `offline_accuracy: null` by default)
        metadata_doc: Dict[str, Any] = {
            "name": name,
            "version": version,
            "source": source,
            "size_on_disk": weights_size,
            "config_size_on_disk": config_size,
            "offline_accuracy": offline_accuracy,  # Serializes as `null` when None
            "weights_object": weights_key,
            "config_object": config_key,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        raw_meta = json.dumps(metadata_doc, indent=2).encode("utf-8")
        client.put_object(
            bucket_name=bucket,
            object_name=metadata_key,
            data=io.BytesIO(raw_meta),
            length=len(raw_meta),
            content_type="application/json",
        )

        print(
            f"[minio-init] [{action}] {name}@{version} | "
            f"size_on_disk={weights_size} B | offline_accuracy={offline_accuracy}",
            flush=True,
        )

    print(
        f"[minio-init] Ingestion complete: {uploaded_count} uploaded, "
        f"{skipped_count} already cached, {missing_count} missing.",
        flush=True,
    )


def main() -> None:
    data_dir = os.getenv("MINIO_DATA_DIR", "/data")
    api_port = os.getenv("MINIO_API_PORT", "9000")
    console_port = os.getenv("MINIO_CONSOLE_PORT", "9001")
    root_user = os.environ["MINIO_ROOT_USER"]
    root_password = os.environ["MINIO_ROOT_PASSWORD"]
    bucket = os.getenv("MINIO_BUCKET", "model-registry")
    force_reupload = os.getenv("FORCE_REUPLOAD", "false").lower() == "true"

    # Start MinIO server as a child process
    minio_cmd = [
        "/usr/bin/minio",
        "server",
        data_dir,
        "--address",
        f":{api_port}",
        "--console-address",
        f":{console_port}",
    ]
    print(f"[minio-init] Starting MinIO: {' '.join(minio_cmd)}", flush=True)
    minio_proc = subprocess.Popen(minio_cmd)

    # Forward SIGTERM/SIGINT from `docker stop` to MinIO for clean shutdown
    def _forward_signal(signum: int, _frame: Any) -> None:
        if minio_proc.poll() is None:
            minio_proc.send_signal(signum)

    signal.signal(signal.SIGTERM, _forward_signal)
    signal.signal(signal.SIGINT, _forward_signal)

    try:
        wait_for_minio(api_port)
        client = Minio(
            endpoint=f"127.0.0.1:{api_port}",
            access_key=root_user,
            secret_key=root_password,
            secure=False,
        )
        ingest_models(client=client, bucket=bucket, force_reupload=force_reupload)
    except Exception as exc:
        print(f"[minio-init] ERROR during startup model ingestion: {exc}", file=sys.stderr, flush=True)
        minio_proc.terminate()
        minio_proc.wait()
        sys.exit(1)

    # Keep MinIO running in the foreground
    sys.exit(minio_proc.wait())


if __name__ == "__main__":
    main()