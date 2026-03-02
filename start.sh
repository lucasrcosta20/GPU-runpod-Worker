#!/bin/bash
set -e

echo "=== GPU Worker Starting ==="

# 1. Start Ollama in background
echo "Starting Ollama..."
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

# 2. List available models
echo "Available models:"
ollama list

# 3. Start Runpod handler
echo "Starting Runpod handler..."
python handler.py
