# Multi-stage build for wakeword training environment.
#
# Base image pinned to a specific tag + digest so the build is
# reproducible across rebuilds. Bumping the python minor or the digest
# is a deliberate two-line change reviewed in PR.
FROM python:3.11.15-slim-bookworm@sha256:ee710afcfb733f4a750d9be683cf054b5cd247b6c5f5237a6849ea568b90ab15 AS base

# Install system dependencies (unset proxy for apt)
RUN unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    git \
    ffmpeg \
    sox \
    espeak-ng \
    libespeak-ng1 \
    libsndfile1 \
    libsndfile1-dev \
    libasound2-dev \
    libffi-dev \
    libssl-dev \
    curl \
    ca-certificates \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create workspace directory
RUN mkdir -p /workspace /workspace/custom_models /workspace/data

# Install Python dependencies. All pinned to exact versions for
# reproducibility. Hash-pinning (`--require-hashes`) is a follow-up: it
# requires generating a `requirements.lock` via `pip-compile` against
# this exact base image, which has to happen during the build itself.
# Tracked as `owe-sat: add hash-locked requirements.lock to wakeword-training`.
RUN pip install --no-cache-dir --upgrade pip==24.3.1 "setuptools==75.6.0" wheel==0.45.1 && \
    pip install --no-cache-dir \
    pyyaml==6.0.2 \
    numpy==1.26.4 \
    "scipy==1.12.0" \
    soundfile==0.12.1 \
    resampy==0.4.3 \
    tqdm==4.67.1 \
    matplotlib==3.9.3 \
    scikit-learn==1.5.2 \
    onnx==1.17.0 \
    onnxruntime==1.20.1 \
    onnxscript==0.1.0 \
    datasets==3.1.0 \
    speechbrain==1.0.2 \
    torch==2.8.0 \
    torchaudio==2.8.0 \
    espeak-phonemizer==1.3.1 \
    piper-tts==1.2.0 \
    pathvalidate==3.2.1 \
    flask==3.1.0 \
    torchinfo==1.8.0 \
    torchmetrics==1.6.0 \
    pronouncing==0.2.0 \
    mutagen==1.47.0 \
    acoustics==0.2.6 \
    audiomentations==0.38.0 \
    webrtcvad==2.0.10 \
    torch-audiomentations==0.12.0

# Clone openWakeWord repository pinned to a specific commit.
# Bumping the SHA is a deliberate one-line change reviewed in PR.
ARG OPENWAKEWORD_SHA=368c03716d1e
RUN git clone https://github.com/dscripka/openWakeWord.git /workspace/openWakeWord_upstream && \
    cd /workspace/openWakeWord_upstream && \
    git checkout "${OPENWAKEWORD_SHA}" && \
    pip install --no-cache-dir -e .

# Copy application files
COPY . .

# Make scripts executable
RUN chmod +x trainer.sh docker-train.sh generate_dataset.py generate_training_samples.py 2>/dev/null || true

# Set environment variables
ENV PYTHONPATH=/workspace/openWakeWord_upstream
ENV BASE_DIR=/workspace/data
ENV OWW_REPO_DIR=/workspace/openWakeWord_upstream
ENV PATH=/usr/local/bin:${PATH}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import openwakeword; print('OK')" || exit 1

CMD ["/bin/bash"]
