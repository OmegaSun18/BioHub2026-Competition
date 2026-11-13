# Ubuntu 24.04 ships with Python 3.12 natively (required for zarr==3.2.1)
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FIJI_PATH=/opt/Fiji.app \
    DISPLAY=:99

# 1. Install Python 3.12, Git, download tools, CLI utilities, and Xvfb
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    wget \
    unzip \
    vim \
    less \
    tmux \
    htop \
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

# 4. Install Python dependencies (Ubuntu 24.04 allows --break-system-packages in containers)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --break-system-packages -r /tmp/requirements.txt

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

# 7. Auto-start Xvfb in the background whenever the container launches
RUN printf '#!/bin/bash\nXvfb :99 -screen 0 1024x768x24 >/dev/null 2>&1 &\nexec "$@"\n' > /usr/local/bin/entrypoint.sh && \
    chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]