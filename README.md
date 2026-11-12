# BioHub2026-Competition

Build the dockerfile with:
* For a public repository:
    ```bash
    docker build -t biohub2026:latest .
    ```
* For a private repository (pass a GitHub Personal Access Token):
    ```bash
    docker build --build-arg GITHUB_TOKEN=ghp_your_token_here -t biohub2026:latest .
    ```

Run the dockerfile with:

```bash
docker run --gpus all -it --rm \
    --shm-size=16g \
    -p 8888:8888 \
    -v /mnt/data/kaggle/biohub-cell-tracking-during-development:/data/biohub-cell-tracking-during-development:ro \
    -v $(pwd)/outputs:/workspace/outputs \
    biohub2026:latest
```