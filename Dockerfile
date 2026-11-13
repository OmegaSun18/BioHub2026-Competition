FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FIJI_PATH=/opt/Fiji.app

# 1. Install Python, Git, download tools, and Xvfb/X11 libs (for headless Java AWT/Swing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    wget \
    unzip \
    ca-certificates \
    xvfb \
    libxrender1 \
    libxtst6 \
    libxi6 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# 2. Download and unpack Fiji (Linux 64-bit) into /opt/Fiji.app
RUN curl -fsSL https://downloads.imagej.net/fiji/latest/fiji-latest-linux64-jdk.zip -o /tmp/fiji.zip && \
    unzip -q /tmp/fiji.zip -d /opt && \
    rm /tmp/fiji.zip && \
    if [ -d "/opt/Fiji" ] && [ ! -d "/opt/Fiji.app" ]; then mv /opt/Fiji /opt/Fiji.app; fi

WORKDIR /workspace

# 3. Clone the main branch of the repository
ARG GITHUB_TOKEN=""
RUN if [ -n "$GITHUB_TOKEN" ]; then \
        git clone --branch main --single-branch https://${GITHUB_TOKEN}@github.com/OmegaSun18/BioHub2026-Competition.git .; \
    else \
        git clone --branch main --single-branch https://github.com/OmegaSun18/BioHub2026-Competition.git .; \
    fi

# 4. Install Python dependencies (cupy, zarr, pyimagej, scyjava, etc.)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# 5. Provision Java 21 via scyjava/cjdk, pre-warm cache, and verify TrackMate loads
RUN python -c "\
import scyjava, imagej; \
scyjava.config.set_java_constraints(fetch=True, vendor='zulu', version='21'); \
scyjava.config.add_option('-Djava.awt.headless=true'); \
ij = imagej.init('/opt/Fiji.app', mode='headless'); \
TrackMate = scyjava.jimport('fiji.plugin.trackmate.TrackMate'); \
print('Fiji version:', ij.getVersion(), '| TrackMate loaded:', TrackMate)"

# 6. Create Kaggle-compatible mount points
RUN mkdir -p /kaggle/input/biohub-cell-tracking-during-development /kaggle/working

EXPOSE 8888

# Run JupyterLab inside Xvfb so TrackMate classes work headlessly
CMD ["xvfb-run", "-a", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]