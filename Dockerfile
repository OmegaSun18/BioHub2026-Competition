# Lean NVIDIA CUDA 12.4 runtime image on Ubuntu 22.04 (Python 3.10/3.11)
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install Python 3, pip, git, and minimal build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# WORKDIR /

# Clone the main branch of the repository
ARG GITHUB_TOKEN=""
RUN if [ -n "$GITHUB_TOKEN" ]; then \
        git clone --branch main --single-branch https://${GITHUB_TOKEN}@github.com/OmegaSun18/BioHub2026-Competition.git .; \
    else \
        git clone --branch main --single-branch https://github.com/OmegaSun18/BioHub2026-Competition.git .; \
    fi

# Copy and install exact notebook dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Create Kaggle-compatible directories so hardcoded notebook paths work seamlessly
# RUN mkdir -p /kaggle/input/biohub-cell-tracking-during-development /kaggle/working

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]