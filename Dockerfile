# Multi-stage build for wakeword training environment
FROM python:3.11-slim-bookworm AS base

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

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip "setuptools<81" wheel && \
    pip install --no-cache-dir \
    pyyaml \
    numpy \
    "scipy<1.17" \
    soundfile \
    resampy \
    tqdm \
    matplotlib \
    scikit-learn \
    onnx \
    onnxruntime \
    onnxscript \
    datasets \
    speechbrain \
    torch==2.8.0 \
    torchaudio==2.8.0 \
    espeak-phonemizer \
    piper-tts \
    flask \
    torchinfo \
    torchmetrics \
    pronouncing \
    mutagen \
    acoustics \
    audiomentations \
    webrtcvad \
    torch-audiomentations

# Clone openWakeWord repository
RUN git clone --depth 1 https://github.com/dscripka/openWakeWord.git /workspace/openWakeWord_upstream && \
    cd /workspace/openWakeWord_upstream && \
    pip install --no-cache-dir -e .

# Copy application files
COPY . .

# Make scripts executable
RUN chmod +x trainer.sh docker-train.sh generate_dataset.py generate_training_samples.py 2>/dev/null || true

# Set environment variables
ENV PYTHONPATH=/workspace/openWakeWord_upstream
ENV BASE_DIR=/workspace
ENV OWW_REPO_DIR=/workspace/openWakeWord_upstream
ENV PATH=/usr/local/bin:${PATH}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import openwakeword; print('OK')" || exit 1

CMD ["/bin/bash"]
