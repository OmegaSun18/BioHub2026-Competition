# BioHub2026-Competition

Build the dockerfile with:
* For a public repository:
    ```bash
    docker build -t biohub2026-fiji:latest .
    ```
* For a private repository (pass a GitHub Personal Access Token):
    ```bash
    docker build --build-arg GITHUB_TOKEN=ghp_your_token_here -t biohub2026:latest .
    ```

Run the dockerfile with:

```bash
mkdir -p /mnt/data/biohub_working

# Start and drop directly into bash (CPU-only node; add --gpus all on GPU nodes)
docker run -it --name biohub_dev \
    --shm-size=8g \
    -e _JAVA_OPTIONS="-Xmx16g" \
    -v /mnt/data:/mnt/data \
    -v /mnt/data/kaggle/biohub-cell-tracking-during-development:/kaggle/input/biohub-cell-tracking-during-development:ro \
    -v /mnt/data/biohub_working:/kaggle/working \
    -v ~/shankar/BioHub2026-Competition:/workspace \
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