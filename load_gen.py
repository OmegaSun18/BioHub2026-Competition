import os
import glob
import time
import asyncio
import grpc
import numpy as np
from typing import List

import pipeline_pb2
import pipeline_pb2_grpc


def sample_zarr_datasets(
    data_dir: str,
    fraction: float = 0.20,
    seed: int = 8001,
) -> List[str]:
    """Deterministically sample a fraction of .zarr datasets using `seed`."""
    all_zarrs = sorted(glob.glob(os.path.join(data_dir, "*.zarr")))
    if not all_zarrs:
        raise FileNotFoundError(f"No .zarr files found in {data_dir}")

    n_sample = max(1, int(round(len(all_zarrs) * fraction)))
    rng = np.random.default_rng(seed)
    chosen_indices = rng.choice(len(all_zarrs), size=n_sample, replace=False)
    sampled_paths = sorted(all_zarrs[i] for i in chosen_indices)
    return sampled_paths


async def send_pipeline_request(det_stub, track_stub, zarr_path: str, results: list):
    t0 = time.time()
    req = pipeline_pb2.VideoRequest(zarr_path=zarr_path)
    try:
        det_resp = await det_stub.Detect(req)
        track_resp = await track_stub.Track(det_resp)
        elapsed = time.time() - t0
        results.append((True, elapsed, track_resp.assertion_passed))
        print(
            f"[OK] {os.path.basename(zarr_path)} | "
            f"det_out='{det_resp.detection_output_path or 'null'}' | "
            f"track_out='{track_resp.tracking_output_path or 'null'}' | "
            f"assertion={track_resp.assertion_passed} | {elapsed:.2f}s"
        )
    except Exception as exc:
        elapsed = time.time() - t0
        results.append((False, elapsed, False))
        print(f"[FAIL] {os.path.basename(zarr_path)} after {elapsed:.2f}s: {exc}")


async def open_loop_generator(data_dir: str, target_qps: float):
    files = sample_zarr_datasets(data_dir)
    det_addr = os.environ.get("DETECTION_ADDR", "detection-service.default.svc.cluster.local:50051")
    track_addr = os.environ.get("TRACKING_ADDR", "tracking-service.default.svc.cluster.local:50051")

    print(f"Sampled {len(files)} .zarr files from {data_dir}")
    print(f"Sending open-loop load at {target_qps} QPS -> Detection ({det_addr}) -> Tracking ({track_addr})")

    det_channel = grpc.aio.insecure_channel(
        det_addr, options=[("grpc.lb_policy_name", "round_robin")]
    )
    track_channel = grpc.aio.insecure_channel(
        track_addr, options=[("grpc.lb_policy_name", "round_robin")]
    )
    det_stub = pipeline_pb2_grpc.CellDetectionStub(det_channel)
    track_stub = pipeline_pb2_grpc.CellTrackingStub(track_channel)

    tasks = []
    results = []
    interval = 1.0 / target_qps

    for zarr_path in files:
        tasks.append(asyncio.create_task(send_pipeline_request(det_stub, track_stub, zarr_path, results)))
        await asyncio.sleep(interval)

    await asyncio.gather(*tasks)
    await det_channel.close()
    await track_channel.close()

    total = len(results)
    succeeded = sum(1 for ok, _, _ in results if ok)
    passed = sum(1 for ok, _, p in results if ok and p)
    latencies = [lat for ok, lat, _ in results if ok]
    p50 = np.percentile(latencies, 50) if latencies else 0.0
    p95 = np.percentile(latencies, 95) if latencies else 0.0
    print(f"\nCompleted: {succeeded}/{total} | Assertions Passed: {passed}/{succeeded} | p50={p50:.2f}s | p95={p95:.2f}s")


if __name__ == "__main__":
    test_dir = os.environ.get("TEST_DATA_DIR", "/mnt/data/biohub_working/split_dataset/test")
    target_qps = float(os.environ.get("TARGET_QPS", "2.0"))
    asyncio.run(open_loop_generator(test_dir, target_qps))
