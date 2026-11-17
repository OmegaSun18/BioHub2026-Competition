import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from minio import Minio
from minio.commonconfig import REPLACE, CopySource
from minio.error import S3Error

# Sentinel value meaning "leave whatever is currently stored in MinIO unchanged"
KEEP = object()

# ==============================================================================
# EDIT THIS LIST FOR EACH UPDATE RUN
# ==============================================================================
# Required keys per entry:
#   - "name":    Model family name in MinIO (e.g., "temporal_diff+dog")
#   - "version": Variant tag in MinIO (e.g., "large-6000")
#
# Optional keys (omit or set to KEEP to preserve existing values in MinIO):
#   - "weights_path":     Local/mounted path to a new `last.pt` file
#   - "config_path":      Local/mounted path to a new `config.yaml` file
#   - "source":           New source string
#   - "offline_accuracy": Float score (e.g., 0.894) or None (resets to JSON null)
# ==============================================================================
UPDATES: List[Dict[str, Any]] = [
    # Example 1: Populate offline_accuracy only (keeps existing last.pt & config.yaml)
    {
        "name": "temporal_diff+dog",
        "version": "large-6000",
        "offline_accuracy": 0.894,
    },
    # Example 2: Update model weights (and optionally config.yaml) for a variant
    # {
    #     "name": "temporal_stack+erode",
    #     "version": "medium-4000",
    #     "weights_path": "/mnt/data/biohub_working/split_dataset/biohub_runs/long/temporal_stack+erode@medium-4000/ckpt/last.pt",
    #     "config_path": "/mnt/data/biohub_working/split_dataset/biohub_runs/long/temporal_stack+erode@medium-4000/config.yaml",
    #     "offline_accuracy": None,  # Reset accuracy to null for newly trained weights
    # },
]


def load_existing_metadata(client: Minio, bucket: str, metadata_key: str) -> Dict[str, Any]:
    """Fetch existing metadata.json from MinIO, or return {} if registering a brand-new variant."""
    try:
        resp = client.get_object(bucket, metadata_key)
        try:
            return json.loads(resp.read().decode("utf-8"))
        finally:
            resp.close()
            resp.release_conn()
    except S3Error as exc:
        if exc.code == "NoSuchKey":
            return {}
        raise


def apply_update(client: Minio, bucket: str, entry: Dict[str, Any]) -> None:
    name = entry["name"]
    version = entry["version"]
    prefix = f"{name}/{version}"
    weights_key = f"{prefix}/last.pt"
    config_key = f"{prefix}/config.yaml"
    metadata_key = f"{prefix}/metadata.json"

    existing = load_existing_metadata(client, bucket, metadata_key)
    is_new = len(existing) == 0

    weights_path = entry.get("weights_path", KEEP)
    config_path = entry.get("config_path", KEEP)
    source = entry.get("source", KEEP)
    offline_accuracy = entry.get("offline_accuracy", KEEP)

    if is_new and (weights_path is KEEP or config_path is KEEP):
        raise ValueError(
            f"Model '{name}@{version}' does not exist in bucket '{bucket}' yet. "
            "Provide both 'weights_path' and 'config_path' to register a new variant."
        )

    # Resolve updated values (falling back to existing metadata values when KEEP is used)
    resolved_source = existing.get("source", "") if source is KEEP else source
    resolved_accuracy = (
        existing.get("offline_accuracy", None)
        if offline_accuracy is KEEP
        else offline_accuracy
    )

    if weights_path is not KEEP:
        w_file = Path(weights_path)
        if not w_file.is_file():
            raise FileNotFoundError(f"Weights file not found: {w_file}")
        resolved_weights_size = w_file.stat().st_size
        if source is KEEP and not resolved_source:
            resolved_source = str(w_file.parent.parent)
    else:
        resolved_weights_size = existing["size_on_disk"]

    if config_path is not KEEP:
        c_file = Path(config_path)
        if not c_file.is_file():
            raise FileNotFoundError(f"Config file not found: {c_file}")
        resolved_config_size = c_file.stat().st_size
    else:
        resolved_config_size = existing["config_size_on_disk"]

    s3_metadata = {
        "name": name,
        "version": version,
        "source": str(resolved_source),
        "size-on-disk": str(resolved_weights_size),
        "offline-accuracy": "null" if resolved_accuracy is None else str(resolved_accuracy),
    }

    # 1. Upload new weights if provided; otherwise sync S3 metadata headers in-place
    if weights_path is not KEEP:
        client.fput_object(
            bucket_name=bucket,
            object_name=weights_key,
            file_path=str(weights_path),
            content_type="application/octet-stream",
            metadata=s3_metadata,
        )
    else:
        client.copy_object(
            bucket_name=bucket,
            object_name=weights_key,
            source=CopySource(bucket, weights_key),
            metadata=s3_metadata,
            metadata_directive=REPLACE,
        )

    # 2. Upload new config.yaml if provided; otherwise sync S3 metadata headers in-place
    if config_path is not KEEP:
        client.fput_object(
            bucket_name=bucket,
            object_name=config_key,
            file_path=str(config_path),
            content_type="application/x-yaml",
            metadata=s3_metadata,
        )
    else:
        client.copy_object(
            bucket_name=bucket,
            object_name=config_key,
            source=CopySource(bucket, config_key),
            metadata=s3_metadata,
            metadata_directive=REPLACE,
        )

    # 3. Write updated metadata.json sidecar
    metadata_doc: Dict[str, Any] = {
        "name": name,
        "version": version,
        "source": resolved_source,
        "size_on_disk": resolved_weights_size,
        "config_size_on_disk": resolved_config_size,
        "offline_accuracy": resolved_accuracy,
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
        f"[update-model] Updated {name}@{version} | "
        f"weights_updated={weights_path is not KEEP} | "
        f"config_updated={config_path is not KEEP} | "
        f"size_on_disk={resolved_weights_size} B | "
        f"offline_accuracy={resolved_accuracy}"
    )


def main() -> None:
    if not UPDATES:
        print("[update-model] UPDATES list is empty. Nothing to update.")
        return

    client = Minio(
        endpoint=os.getenv("MINIO_ENDPOINT", "172.17.0.1:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "biohubadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "biohub-minio-secret-key-2026"),
        secure=False,
    )
    bucket = os.getenv("MINIO_BUCKET", "model-registry")

    for entry in UPDATES:
        apply_update(client, bucket, entry)


if __name__ == "__main__":
    main()