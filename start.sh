#!/bin/bash
set -e

# Ensure ollama is in PATH (Pod mode: installed via curl to /usr/local/bin)
export PATH=$PATH:/usr/local/bin

echo "=== GPU Worker Starting ==="

# 0. Ensure Ollama is installed (template base may not include it)
if ! command -v ollama &> /dev/null; then
    echo "Ollama not found, installing..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "Ollama installed."
fi

# 1. Auto-detect VRAM and configure Ollama parallelism BEFORE starting
#
# CRITICAL: The old formula was wrong — it didn't account for KV cache scaling
# with context_length × num_parallel. With explicit num_ctx=2048 (set in llm.py),
# KV cache per slot is very small (~0.6GB vs ~4GB with default 32768).
#
# Correct formula considers:
#   - Model weights (Q4_K_M): ~5GB for 8B model
#   - KV cache per parallel slot (ctx=2048): ~0.6GB per slot
#   - Compute graph overhead + buffers: ~3GB
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
    KV_PER_SLOT_GB=1    # KV cache per slot with num_ctx=2048: ~0.6GB KV + ~0.4GB buffers

    # With num_ctx=2048 (set explicitly in llm.py), KV cache per slot is very
    # small compared to the Ollama default (32768). This allows many parallel
    # slots while keeping all model weights on GPU.
    #
    # A5000 24GB example: (24 - 5 - 3) / 1 = 16 → clamped to 6
    # Total VRAM: 5 (model) + 3 (overhead) + 6×1 (KV) = 14GB, 10GB headroom
    AVAILABLE_FOR_KV=$((VRAM_GB - MODEL_GB - OVERHEAD_GB - RESERVE_GB))
    PARALLEL=$((AVAILABLE_FOR_KV / KV_PER_SLOT_GB))

    # Clamp between 1 and 6
    [ "$PARALLEL" -lt 1 ] && PARALLEL=1
    [ "$PARALLEL" -gt 6 ] && PARALLEL=6

    echo "Detected ${VRAM_GB}GB VRAM -> OLLAMA_NUM_PARALLEL=${PARALLEL} (${AVAILABLE_FOR_KV}GB available for KV cache)"
else
    echo "WARNING: nvidia-smi not found, defaulting to OLLAMA_NUM_PARALLEL=1"
    PARALLEL=1
fi

export OLLAMA_NUM_PARALLEL=$PARALLEL
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_FLASH_ATTENTION=true
export OLLAMA_KEEP_ALIVE=30m

# Force Python to flush stdout/stderr immediately (so prints appear in Runpod logs)
export PYTHONUNBUFFERED=1

# cuDNN 9 is bundled with Ollama at a non-standard path.
# onnxruntime-gpu needs it in LD_LIBRARY_PATH to use CUDAExecutionProvider for rembg.
# Include both Ollama's bundled cuDNN and the pip-installed nvidia-cudnn path.
export LD_LIBRARY_PATH="/usr/local/lib/ollama/mlx_cuda_v13:/usr/local/lib/ollama/cuda_v13:${LD_LIBRARY_PATH:-}"

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

# 3. Ensure required models are available (pull if missing)
DEFAULT_MODEL=${DEFAULT_MODEL:-llama3.1:8b}
REQUIRED_MODELS="${DEFAULT_MODEL} qwen2.5vl:3b qwen2.5vl:7b"

for model in $REQUIRED_MODELS; do
    if ! ollama list 2>/dev/null | grep -q "^${model}"; then
        echo "Model $model not found, pulling..."
        ollama pull "$model"
    fi
done

echo "Available models:"
ollama list

# 4. Warmup: pre-load default model into VRAM
# Skipped — with KEEP_ALIVE=30m the model loads on first LLM request
# and unloads after 30min idle. Image operations (rembg, upscale)
# use hold_vram/release_vram for explicit VRAM management.
DEFAULT_MODEL=${DEFAULT_MODEL:-llama3.1:8b}
echo "Model $DEFAULT_MODEL will load on first LLM request (KEEP_ALIVE=30m)"

# 5. Start worker (Serverless handler or Pod HTTP server)
if [ "$POD_MODE" = "1" ]; then
    echo "Starting Pod HTTP server on port ${POD_SERVER_PORT:-8000}..."
    python pod_server.py
else
    echo "Starting Runpod Serverless handler..."
    python handler.py
fi
