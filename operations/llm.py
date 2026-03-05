"""
LLM operation — text generation via Ollama localhost.

Supports text-only and multimodal (images) generation.
"""

import time
from typing import Any, Dict, List, Optional

import requests


OLLAMA_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 300


def generate(
    model: str,
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.7,
    max_tokens: int = 2000,
    timeout: int = DEFAULT_TIMEOUT,
    images: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generate text via Ollama localhost.

    Args:
        model: Model name (e.g. 'llama3.1:8b', 'qwen2.5vl:7b').
        prompt: User prompt.
        system_prompt: System prompt.
        temperature: Generation temperature (0.0-2.0).
        max_tokens: Maximum tokens to generate.
        timeout: Request timeout in seconds.
        images: Optional list of base64-encoded images (multimodal).

    Returns:
        Dict with 'text', 'model', 'tokens_generated', 'processing_time_seconds'.

    Raises:
        RuntimeError: If Ollama returns an error.
    """
    start = time.time()

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    if system_prompt:
        payload["system"] = system_prompt

    if images:
        payload["images"] = images
        # Multimodal precisa de contexto maior para processar imagens + texto
        payload["options"]["num_ctx"] = 8192

    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json=payload,
        timeout=timeout,
    )

    if resp.status_code != 200:
        raise RuntimeError(
            f"Ollama error {resp.status_code}: {resp.text[:500]}"
        )

    data = resp.json()
    elapsed = time.time() - start

    return {
        "text": data.get("response", ""),
        "model": model,
        "tokens_generated": data.get("eval_count", 0),
        "processing_time_seconds": round(elapsed, 2),
    }
