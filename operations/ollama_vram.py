"""
Ollama VRAM management — unload/reload model to free GPU memory.

When rembg (onnxruntime) or upscale (PyTorch) need VRAM, the Ollama
model must be temporarily unloaded. After the operation completes,
the model reloads automatically on the next LLM request.

Usage:
    with ollama_vram_free():
        # VRAM is free here — run rembg/upscale
        ...
    # Model will auto-reload on next LLM call
"""

import os
import time
from contextlib import contextmanager

import requests


OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "llama3.1:8b")


def unload_model(model: str = DEFAULT_MODEL, timeout: int = 30) -> bool:
    """
    Unload model from VRAM by setting keep_alive=0.

    Args:
        model: Model name to unload.
        timeout: HTTP timeout in seconds.

    Returns:
        True if unload request succeeded.
    """
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": "",
                "keep_alive": 0,
            },
            timeout=timeout,
        )
        if resp.status_code == 200:
            # Give Ollama a moment to actually free VRAM
            time.sleep(2)
            print(f"[VRAM] Unloaded {model} from VRAM")
            return True
        else:
            print(f"[VRAM] Failed to unload {model}: HTTP {resp.status_code}")
            return False
    except Exception as e:
        print(f"[VRAM] Error unloading {model}: {e}")
        return False


@contextmanager
def ollama_vram_free(model: str = DEFAULT_MODEL):
    """
    Context manager that frees VRAM by unloading Ollama model.

    Model reloads automatically on next LLM request (keep_alive=24h),
    so we don't force a reload here — saves time if no LLM calls follow.

    Usage:
        with ollama_vram_free():
            # Full 24GB VRAM available for rembg/upscale
            run_rembg(...)
    """
    unload_model(model)
    try:
        yield
    finally:
        # Don't reload here — model auto-loads on next LLM request.
        # This avoids wasting 5-10s reloading if another image op follows.
        print("[VRAM] Ollama model will auto-reload on next LLM request")
