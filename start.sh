#!/bin/bash
set -e

echo "=== GPU Worker Starting ==="

# 1. Configure Ollama parallelism BEFORE starting (read at startup)
# RTX 4090 (24GB VRAM) with llama3.1:8b (~5GB) supports 4 parallel requests
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_LOADED_MODELS=2

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

# 4. Start Runpod handler
echo "Starting Runpod handler..."
python handler.py
