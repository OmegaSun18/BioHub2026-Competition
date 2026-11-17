import os
import time
import queue
import importlib
import threading
from concurrent import futures
import grpc
from prometheus_client import start_http_server, Counter, Histogram, Gauge

import pipeline_pb2
import pipeline_pb2_grpc

# --- Prometheus Metrics ---
REQ_COUNT = Counter('request_count_total', 'Total video requests processed', ['stage'])
LATENCY = Histogram(
    'request_latency_seconds',
    'Batch inference latency in seconds',
    ['stage'],
    buckets=[0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120]
)
QUEUE_LENGTH = Gauge('queue_length', 'Current number of requests waiting in queue', ['stage'])
BATCH_SIZE_GAUGE = Gauge('current_batch_size', 'Number of videos in the current active batch', ['stage'])
ASSERTION_PASS = Counter('assertion_pass_total', 'Passed tracking assertions', ['stage'])
ASSERTION_FAIL = Counter('assertion_fail_total', 'Failed tracking assertions', ['stage'])


class BatchingServer:
    def __init__(self):
        self.stage = os.environ.get("STAGE_TYPE", "detection").strip().lower()
        self.inference_code = os.environ.get("INFERENCE_CODE", "").strip()
        self.max_batch_size = int(os.environ.get("MAX_BATCH_SIZE", "4"))
        self.batch_timeout = float(os.environ.get("BATCH_TIMEOUT", "2.0"))

        self.model_module = None
        self.model = None
        self._load_inference_code()

        self.req_queue = queue.Queue()
        self.results = {}
        self.condition = threading.Condition()

        threading.Thread(target=self._process_batches, daemon=True).start()

    def _load_inference_code(self):
        """Load module from inference_codes, fiji_detectors, or fiji_trackers; default to None."""
        if not self.inference_code or self.inference_code.lower() in {"null", "none"}:
            print(f"[{self.stage}] INFERENCE_CODE is unset/null. Running in null passthrough mode.")
            self.model_module = None
            self.model = None
            return

        stage_pkg = "fiji_detectors" if self.stage == "detection" else "fiji_trackers"
        candidate_modules = [
            f"inference_codes.{self.inference_code}",
            f"{stage_pkg}.{self.inference_code}",
            self.inference_code,
        ]

        for mod_name in candidate_modules:
            try:
                mod = importlib.import_module(mod_name)
                if hasattr(mod, "load_model"):
                    self.model_module = mod
                    self.model = mod.load_model()
                    print(f"[{self.stage}] Loaded '{mod_name}' successfully.")
                    return
            except ModuleNotFoundError:
                continue
            except Exception as exc:
                print(f"[{self.stage}] WARNING: Error initializing '{mod_name}': {exc}")
                break

        print(
            f"[{self.stage}] WARNING: No runnable inference code found for '{self.inference_code}'. "
            f"Defaulting model to None (null mode)."
        )
        self.model_module = None
        self.model = None

    def _run_null_inference(self, inputs):
        if self.stage == "detection":
            return [{"output_path": ""} for _ in inputs]
        return [{"output_path": "", "assertion_passed": True} for _ in inputs]

    def _process_batches(self):
        while True:
            batch = []
            try:
                first_item = self.req_queue.get(timeout=1.0)
                batch.append(first_item)
                QUEUE_LENGTH.labels(stage=self.stage).dec()

                deadline = time.time() + self.batch_timeout
                while len(batch) < self.max_batch_size:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break
                    try:
                        item = self.req_queue.get(timeout=remaining)
                        batch.append(item)
                        QUEUE_LENGTH.labels(stage=self.stage).dec()
                    except queue.Empty:
                        break
            except queue.Empty:
                continue

            BATCH_SIZE_GAUGE.labels(stage=self.stage).set(len(batch))
            req_ids = [item['id'] for item in batch]
            inputs = [item['req'] for item in batch]

            start_time = time.time()
            try:
                if self.model is None:
                    outputs = self._run_null_inference(inputs)
                else:
                    outputs = self.model.predict_batch(inputs)
            except Exception as exc:
                outputs = [exc] * len(batch)

            latency = time.time() - start_time
            LATENCY.labels(stage=self.stage).observe(latency)
            REQ_COUNT.labels(stage=self.stage).inc(len(batch))

            with self.condition:
                for req_id, out in zip(req_ids, outputs):
                    self.results[req_id] = out
                self.condition.notify_all()

    def submit_request(self, req_id, payload):
        QUEUE_LENGTH.labels(stage=self.stage).inc()
        self.req_queue.put({'id': req_id, 'req': payload})

        with self.condition:
            while req_id not in self.results:
                self.condition.wait()
            res = self.results.pop(req_id)
            if isinstance(res, Exception):
                raise res
            return res


class CellDetectionService(pipeline_pb2_grpc.CellDetectionServicer):
    def __init__(self, backend):
        self.backend = backend

    def Detect(self, request, context):
        req_id = id(request)
        out_data = self.backend.submit_request(req_id, request.zarr_path)
        det_path = out_data.get('output_path') or ""
        return pipeline_pb2.DetectionResponse(
            zarr_path=request.zarr_path,
            detection_output_path=det_path
        )


class CellTrackingService(pipeline_pb2_grpc.CellTrackingServicer):
    def __init__(self, backend):
        self.backend = backend

    def Track(self, request, context):
        req_id = id(request)
        payload = {
            "zarr_path": request.zarr_path,
            "detection_output_path": request.detection_output_path,
        }
        out_data = self.backend.submit_request(req_id, payload)

        passed = bool(out_data.get('assertion_passed', True))
        if passed:
            ASSERTION_PASS.labels(stage='tracking').inc()
        else:
            ASSERTION_FAIL.labels(stage='tracking').inc()

        track_path = out_data.get('output_path') or ""
        return pipeline_pb2.TrackingResponse(
            zarr_path=request.zarr_path,
            tracking_output_path=track_path,
            assertion_passed=passed
        )


def serve():
    start_http_server(8000)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    backend = BatchingServer()

    if backend.stage == "detection":
        pipeline_pb2_grpc.add_CellDetectionServicer_to_server(CellDetectionService(backend), server)
    elif backend.stage == "tracking":
        pipeline_pb2_grpc.add_CellTrackingServicer_to_server(CellTrackingService(backend), server)
    else:
        raise ValueError(f"Unknown STAGE_TYPE '{backend.stage}'. Must be 'detection' or 'tracking'.")

    server.add_insecure_port('[::]:50051')
    print(f"[{backend.stage}] gRPC server listening on :50051 | Prometheus metrics on :8000")
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
