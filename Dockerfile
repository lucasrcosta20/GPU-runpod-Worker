# GPU Worker — Runpod Serverless / Pod
# Base: Runpod official PyTorch image (Python 3.11 + CUDA 12.4 + PyTorch 2.4 pre-installed)
# Same image used by the Pod template for consistency.
# Includes: Ollama + LLM models + rembg + Real-ESRGAN + runpod SDK
# Final image size: ~20-25GB (models baked in)
#
# Reference: https://docs.runpod.io/serverless/workers/create-dockerfile
# Base image: https://hub.docker.com/r/runpod/pytorch

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Extra system dependencies (curl for Ollama installer, zstd for Ollama, GL for OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    zstd \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# ---- Python dependencies ----
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
