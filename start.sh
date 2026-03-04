#!/bin/bash
set -e

# Ensure ollama is in PATH (Pod mode: installed via curl to /usr/local/bin)
export PATH=$PATH:/usr/local/bin

echo "=== GPU Worker Starting ==="

# 1. Auto-detect VRAM and configure Ollama parallelism BEFORE starting
#
# CRITICAL: The old formula was wrong — it didn't account for KV cache scaling
# with context_length × num_parallel. With NUM_PARALLEL=3 and ctx=32768:
#   KV cache = 3 × 4GB = 12GB, model = 4.6GB → 16.6GB (offloads to CPU!)
#
# Correct formula considers:
#   - Model weights (Q4_K_M): ~5GB for 8B model
#   - KV cache per parallel slot (ctx=32768, GQA): ~4GB per slot
#   - Compute graph overhead: ~1GB
#   - Reserve for other GPU workloads (rembg, upscale): 2GB
#
# Priority: ALL model weights on GPU > more parallelism
# CPU offload kills performance — 1 parallel slot fully on GPU is faster
# than 3 slots with half the model on CPU.
#
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -n "$VRAM_MB" ]; then
    VRAM_GB=$((VRAM_MB / 1024))
    MODEL_GB=5          # llama3.1:8b Q4_K_M weights
    OVERHEAD_GB=3       # compute graph + buffers + Ollama internal overhead
    RESERVE_GB=0        # no reserve needed — Ollama manages its own memory
    KV_PER_SLOT_GB=6    # KV cache per slot: real-world ~4GB KV + 2GB compute/buffers per slot

    # Total Ollama needs = MODEL_GB + OVERHEAD_GB + (PARALLEL × KV_PER_SLOT_GB)
    # Must fit entirely in VRAM to avoid CPU offload (which kills performance)
    AVAILABLE_FOR_KV=$((VRAM_GB - MODEL_GB - OVERHEAD_GB - RESERVE_GB))
    PARALLEL=$((AVAILABLE_FOR_KV / KV_PER_SLOT_GB))

    # Clamp between 1 and 4
    [ "$PARALLEL" -lt 1 ] && PARALLEL=1
    [ "$PARALLEL" -gt 4 ] && PARALLEL=4

    echo "Detected ${VRAM_GB}GB VRAM -> OLLAMA_NUM_PARALLEL=${PARALLEL} (${AVAILABLE_FOR_KV}GB available for KV cache)"
else
    echo "WARNING: nvidia-smi not found, defaulting to OLLAMA_NUM_PARALLEL=1"
    PARALLEL=1
fi

export OLLAMA_NUM_PARALLEL=$PARALLEL
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_FLASH_ATTENTION=true
export OLLAMA_KEEP_ALIVE=5m

# Force Python to flush stdout/stderr immediately (so prints appear in Runpod logs)
export PYTHONUNBUFFERED=1

# cuDNN 9 is bundled with Ollama at a non-standard path.
# onnxruntime-gpu needs it in LD_LIBRARY_PATH to use CUDAExecutionProvider for rembg.
export LD_LIBRARY_PATH="/usr/local/lib/ollama/mlx_cuda_v13:${LD_LIBRARY_PATH:-}"

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

# 4. Warmup: pre-load default model into VRAM
# Skipped — with KEEP_ALIVE=5m the model loads on first LLM request
# and unloads after 5min idle. This avoids VRAM contention with
# rembg/upscale operations that need the full 24GB.
DEFAULT_MODEL=${DEFAULT_MODEL:-llama3.1:8b}
echo "Model $DEFAULT_MODEL will load on first LLM request (KEEP_ALIVE=5m)"

# 5. Start worker (Serverless handler or Pod HTTP server)
if [ "$POD_MODE" = "1" ]; then
    echo "Starting Pod HTTP server on port ${POD_SERVER_PORT:-8000}..."
    python pod_server.py
else
    echo "Starting Runpod Serverless handler..."
    python handler.py
fi
