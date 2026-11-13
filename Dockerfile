FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    FIJI_PATH=/opt/Fiji.app

# 1. Install Python, Git, OpenJDK 17, Maven, and Xvfb (for headless Java/Swing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    wget \
    unzip \
    ca-certificates \
    openjdk-17-jdk \
    maven \
    xvfb \
    libxrender1 \
    libxtst6 \
    libxi6 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# 2. Download and unpack Fiji (Linux 64-bit) into /opt/Fiji.app
RUN wget -q https://downloads.imagej.net/fiji/latest/fiji-linux64.zip -O /tmp/fiji.zip && \
    unzip -q /tmp/fiji.zip -d /opt && \
    rm /tmp/fiji.zip

# WORKDIR /workspace

# 3. Clone the main branch of the repository
ARG GITHUB_TOKEN=""
RUN if [ -n "$GITHUB_TOKEN" ]; then \
        git clone --branch main --single-branch https://${GITHUB_TOKEN}@github.com/OmegaSun18/BioHub2026-Competition.git .; \
    else \
        git clone --branch main --single-branch https://github.com/OmegaSun18/BioHub2026-Competition.git .; \
    fi

# 4. Install Python packages (including pyimagej & cupy)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

COPY sanity_test.py .

# 5. Pre-warm PyImageJ / scyjava Maven cache during build so startup is instant
RUN python -c "import imagej; ij = imagej.init('/opt/Fiji.app', mode='headless'); print('Fiji version:', ij.getVersion())"

# 6. Create Kaggle-compatible mount points
RUN mkdir -p /kaggle/input/biohub-cell-tracking-during-development /kaggle/working

EXPOSE 8888

# Run JupyterLab wrapped in xvfb-run so TrackMate GUI classes work headlessly
CMD ["xvfb-run", "-a", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]