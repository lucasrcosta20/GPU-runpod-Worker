# GPU Worker — Runpod Serverless
# Base: CUDA 12.1 + Python 3.11
# Includes: Ollama + LLM models + rembg + Real-ESRGAN + runpod SDK
# Final image size: ~20-25GB (models baked in)

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    curl \
    zstd \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# ---- Python dependencies ----
WORKDIR /app

# Upgrade pip first (Ubuntu 22.04 ships old pip)
RUN python3 -m pip install --break-system-packages --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 first (separate step for caching)
# Using --extra-index-url to allow fallback to PyPI for torchvision deps
RUN pip install --no-cache-dir --break-system-packages \
    torch==2.1.2+cu121 torchvision==0.16.2+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# ---- Bake Ollama models ----
# Start Ollama temporarily to pull models, then stop
ENV OLLAMA_MODELS=/root/.ollama/models

RUN ollama serve & \
    sleep 5 && \
    ollama pull llama3.1:8b && \
    ollama pull qwen2.5vl:3b && \
    ollama pull qwen2.5vl:7b && \
    kill %1 || true

# ---- Bake image processing models ----
RUN mkdir -p /app/models

# Real-ESRGAN models
ADD https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth /app/models/
ADD https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth /app/models/

# Pre-download rembg BiRefNet model (triggers ONNX model download)
RUN python -c "from rembg.sessions import sessions_class; \
    [sc('birefnet-general', None) for sc in sessions_class if sc.name() == 'birefnet-general']" \
    || true

ENV MODELS_DIR=/app/models

# ---- Application code ----
COPY handler.py .
COPY operations/ ./operations/
COPY security/ ./security/

# ---- Entrypoint ----
COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]
