#!/bin/bash
set -e

echo "=== GPU Worker Starting ==="

# 1. Auto-detect VRAM and configure Ollama parallelism BEFORE starting
# Formula: (VRAM_GB - model_size - overhead) / kv_cache_per_slot
# llama3.1:8b ~ 5GB model, ~5GB KV cache per parallel slot, 2GB overhead
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -n "$VRAM_MB" ]; then
    VRAM_GB=$((VRAM_MB / 1024))
    MODEL_GB=5
    OVERHEAD_GB=2
    KV_PER_SLOT_GB=5
    AVAILABLE=$((VRAM_GB - MODEL_GB - OVERHEAD_GB))
    PARALLEL=$((AVAILABLE / KV_PER_SLOT_GB))
    # Clamp between 1 and 8
    [ "$PARALLEL" -lt 1 ] && PARALLEL=1
    [ "$PARALLEL" -gt 8 ] && PARALLEL=8
    echo "Detected ${VRAM_GB}GB VRAM -> OLLAMA_NUM_PARALLEL=${PARALLEL}"
else
    echo "WARNING: nvidia-smi not found, defaulting to OLLAMA_NUM_PARALLEL=1"
    PARALLEL=1
fi

export OLLAMA_NUM_PARALLEL=$PARALLEL
export OLLAMA_MAX_LOADED_MODELS=1

# 2. Start Ollama in background
echo "Starting Ollama (OLLAMA_NUM_PARALLEL=$OLLAMA_NUM_PARALLEL)..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
for i in $(seq 1 30); do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama is ready."
        break
    fi
    if [ $i -eq 30 ]; then
        echo "ERROR: Ollama failed to start after 30s"
        exit 1
    fi
    sleep 1
done

# 3. List available models
echo "Available models:"
ollama list

# 4. Start worker (Serverless handler or Pod HTTP server)
if [ "$POD_MODE" = "1" ]; then
    echo "Starting Pod HTTP server on port ${POD_SERVER_PORT:-8000}..."
    python pod_server.py
else
    echo "Starting Runpod Serverless handler..."
    python handler.py
fi
