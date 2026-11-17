import json
import os
from minio import Minio

client = Minio(
    endpoint=os.getenv("MINIO_ENDPOINT", "172.17.0.1:9000"),
    access_key=os.getenv("MINIO_ACCESS_KEY", "biohubadmin"),
    secret_key=os.getenv("MINIO_SECRET_KEY", "biohub-minio-secret-key-2026"),
    secure=False,
)

bucket = "model-registry"
model_name = "temporal_diff+dog"
version = "large-6000"

# 1. Read metadata
resp = client.get_object(bucket, f"{model_name}/{version}/metadata.json")
metadata = json.loads(resp.read().decode("utf-8"))
resp.close()
resp.release_conn()
print("Metadata:", json.dumps(metadata, indent=2))

# 2. Download weights and config YAML
client.fget_object(bucket, f"{model_name}/{version}/last.pt", "/tmp/last.pt")
client.fget_object(bucket, f"{model_name}/{version}/config.yaml", "/tmp/config.yaml")

print(f"Downloaded /tmp/last.pt ({os.path.getsize('/tmp/last.pt')} bytes)")
print(f"Downloaded /tmp/config.yaml ({os.path.getsize('/tmp/config.yaml')} bytes)")