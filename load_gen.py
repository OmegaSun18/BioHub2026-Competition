import os
import glob
import time
import asyncio
import grpc
import numpy as np
from typing import List

import pipeline_pb2
import pipeline_pb2_grpc

def sample_zarr_datasets(data_dir: str, fraction: float = 0.20, seed: int = 8001) -> List[str]:
    """Deterministically sample a fraction of .zarr datasets using `seed`."""
    all_zarrs = sorted(glob.glob(os.path.join(data_dir, "*.zarr")))
    if not all_zarrs:
        raise FileNotFoundError(f"No .zarr files found in {data_dir}")

    n_sample = max(1, int(round(len(all_zarrs) * fraction)))
    rng = np.random.default_rng(seed)
    chosen_indices = rng.choice(len(all_zarrs), size=n_sample, replace=False)
    return sorted(all_zarrs[i] for i in chosen_indices)

async def send_request(stub, zarr_path: str):
    start = time.time()
    req = pipeline_pb2.VideoRequest(zarr_path=zarr_path)
    try:
        # Calls Detection, assuming Detection pushes to Tracking or Load Gen manages the flow.
        # For an end-to-end load test, we trigger Detection, then trigger Tracking.
        det_resp = await stub.Detect(req)
        print(f"Detection done for {zarr_path}")
    except Exception as e:
        print(f"Request failed: {e}")

async def open_loop_generator(data_dir: str, target_qps: float):
    files = sample_zarr_datasets(data_dir)
    print(f"Targeting {target_qps} QPS over {len(files)} files.")
    
    channel = grpc.aio.insecure_channel('detection-service.default.svc.cluster.local:50051')
    stub = pipeline_pb2_grpc.CellDetectionStub(channel)
    
    tasks = []
    interval = 1.0 / target_qps
    
    for f in files:
        task = asyncio.create_task(send_request(stub, f))
        tasks.append(task)
        await asyncio.sleep(interval) # strict open-loop timing
        
    await asyncio.gather(*tasks)
    await channel.close()

if __name__ == "__main__":
    test_dir = "/mnt/data/biohub_working/split_dataset/test"
    target_qps = float(os.environ.get("TARGET_QPS", "2.0"))
    asyncio.run(open_loop_generator(test_dir, target_qps))