# BioHub2026-Competition

Build the dockerfile with:
* For a public repository:
    ```bash
    docker build -t biohub2026-fiji-test:latest .
    ```
* For a private repository (pass a GitHub Personal Access Token):
    ```bash
    docker build --build-arg GITHUB_TOKEN=ghp_your_token_here -t biohub2026:latest .
    ```

Run the dockerfile with:

```bash
mkdir -p /mnt/data/biohub_working

# Start and drop directly into bash (CPU-only node; add --gpus all on GPU nodes)
docker run -it --gpus all --name biohub_dev_himanish \
    --shm-size=8g \
    -e _JAVA_OPTIONS="-Xmx16g" \
    -v /mnt/data:/mnt/data \
    -v /mnt/data/kaggle/biohub-cell-tracking-during-development:/kaggle/input/biohub-cell-tracking-during-development:ro \
    -v /mnt/data/biohub_working:/kaggle/working \
    -v ~/himanish/BioHub2026-Competition:/workspace \
    biohub2026-fiji:latest
```

Add the following snipped to all notebooks or python files before anything is run:
```python
import scyjava
import imagej

# Must be called BEFORE imagej.init() so scyjava uses the cached Java 21 JDK
scyjava.config.set_java_constraints(fetch=True, vendor='zulu', version='21')
scyjava.config.add_option('-Djava.awt.headless=true')

ij = imagej.init('/opt/Fiji.app', mode='headless')
TrackMate = scyjava.jimport('fiji.plugin.trackmate.TrackMate')
```

# Step 1: Compile gRPC Definitions

Generate the Python stubs required by the server
```
python -m grpc_tools.protoc -Iproto --python_out=. --grpc_python_out=. proto/pipeline.proto
```

# Step 2: Implement Inference Wrappers
Create an inference_codes/ directory inside your workspace. Add wrappers (e.g., detect_v1.py and track_v2.py) containing a load_model() function that returns a class with a predict_batch(inputs) method. The detection script must return a DetectionResponse dictionary containing the output_path, which tracking will subsequently read. Both stages read datasets directly from the mounted /mnt/data/ volume.

# Step 3: Deploy to Kubernetes
Ensure K3s/Minikube has access to /mnt/data locally on the VM.
```
kubectl apply -f k8s/pipeline.yaml
```

# Step 4: Run the Open-Loop Load Generator
Once pods are ready (kubectl get pods), execute the load generator inside the cluster to avoid external network bottlenecks.
```
kubectl run load-gen --image=your-registry/biohub-pipeline:latest --restart=Never \
  --env="TARGET_QPS=5.0" \
  --command -- python /workspace/load_gen.py
```
