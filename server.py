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
LATENCY = Histogram('request_latency_seconds', 'Latency of requests', ['stage'], buckets=[0.5, 1, 5, 10, 30, 60, 120])
QUEUE_LENGTH = Gauge('queue_length', 'Current number of requests waiting in queue', ['stage'])
BATCH_SIZE_GAUGE = Gauge('current_batch_size', 'Number of videos in the current active batch', ['stage'])
ASSERTION_PASS = Counter('assertion_pass_total', 'Passed assertions', ['stage'])
ASSERTION_FAIL = Counter('assertion_fail_total', 'Failed assertions', ['stage'])

class BatchingServer:
    def __init__(self):
        self.stage = os.environ.get("STAGE_TYPE", "detection") # 'detection' or 'tracking'
        self.inference_code = os.environ.get("INFERENCE_CODE", "default_model")
        self.max_batch_size = int(os.environ.get("MAX_BATCH_SIZE", "4"))
        self.batch_timeout = float(os.environ.get("BATCH_TIMEOUT", "2.0"))
        
        # Dynamically load the correct inference module from the /inference_codes directory
        print(f"Loading {self.stage} model: {self.inference_code}")
        self.model_module = importlib.import_module(f"inference_codes.{self.inference_code}")
        self.model = self.model_module.load_model()
        
        self.req_queue = queue.Queue()
        self.results = {}
        self.condition = threading.Condition()
        
        # Start background batch processor
        threading.Thread(target=self._process_batches, daemon=True).start()

    def _process_batches(self):
        while True:
            batch = []
            try:
                # Wait for at least one item, then grab up to max_batch_size
                first_item = self.req_queue.get(timeout=1.0)
                batch.append(first_item)
                QUEUE_LENGTH.labels(stage=self.stage).dec()
                
                deadline = time.time() + self.batch_timeout
                while len(batch) < self.max_batch_size and time.time() < deadline:
                    try:
                        item = self.req_queue.get(timeout=deadline - time.time())
                        batch.append(item)
                        QUEUE_LENGTH.labels(stage=self.stage).dec()
                    except queue.Empty:
                        break
            except queue.Empty:
                continue
            
            BATCH_SIZE_GAUGE.labels(stage=self.stage).set(len(batch))
            
            # Extract paths for batch inference
            req_ids = [item['id'] for item in batch]
            inputs = [item['req'] for item in batch]
            
            start_time = time.time()
            try:
                # Execute dynamically loaded inference code
                outputs = self.model.predict_batch(inputs)
            except Exception as e:
                outputs = [e] * len(batch)
                
            latency = time.time() - start_time
            LATENCY.labels(stage=self.stage).observe(latency)
            REQ_COUNT.labels(stage=self.stage).inc(len(batch))
            
            with self.condition:
                for req_id, out in zip(req_ids, outputs):
                    self.results[req_id] = out
                self.condition.notify_all()

    def submit_request(self, req_id, request):
        QUEUE_LENGTH.labels(stage=self.stage).inc()
        self.req_queue.put({'id': req_id, 'req': request})
        
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
        return pipeline_pb2.DetectionResponse(
            zarr_path=request.zarr_path,
            detection_output_path=out_data['output_path']
        )

class CellTrackingService(pipeline_pb2_grpc.CellTrackingServicer):
    def __init__(self, backend):
        self.backend = backend
        
    def Track(self, request, context):
        req_id = id(request)
        out_data = self.backend.submit_request(req_id, request.detection_output_path)
        
        passed = out_data.get('assertion_passed', True)
        if passed:
            ASSERTION_PASS.labels(stage='tracking').inc()
        else:
            ASSERTION_FAIL.labels(stage='tracking').inc()
            
        return pipeline_pb2.TrackingResponse(
            zarr_path=request.zarr_path,
            tracking_output_path=out_data['output_path'],
            assertion_passed=passed
        )

def serve():
    # Start Prometheus metrics server on port 8000
    start_http_server(8000)
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    backend = BatchingServer()
    
    if backend.stage == "detection":
        pipeline_pb2_grpc.add_CellDetectionServicer_to_server(CellDetectionService(backend), server)
    else:
        pipeline_pb2_grpc.add_CellTrackingServicer_to_server(CellTrackingService(backend), server)
        
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()