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
import threading
from contextlib import contextmanager

import requests


OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "llama3.1:8b")

# Reference counting for nested/concurrent ollama_vram_free() calls.
# Protected by lock since pod_server is threaded.
_lock = threading.Lock()
_active_count = 0


def _is_model_loaded(model: str = DEFAULT_MODEL, timeout: int = 5) -> bool:
    """
    Check if model is currently loaded in Ollama VRAM via /api/ps.

    Returns:
        True if model is loaded in VRAM.
    """
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/ps", timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("models", [])
            prefix = model.split(":")[0]
            return any(m.get("name", "").startswith(prefix) for m in models)
    except Exception:
        pass
    return False


def unload_model(model: str = DEFAULT_MODEL, timeout: int = 30) -> bool:
    """
    Unload model from VRAM by setting keep_alive=0.

    Checks /api/ps first — if model is not loaded, skips the
    expensive unload + sleep(2) cycle. This avoids wasting ~4s
    between consecutive batch chunks where the model is already gone.

    Args:
        model: Model name to unload.
        timeout: HTTP timeout in seconds.

    Returns:
        True if model was unloaded or already not loaded.
    """
    if not _is_model_loaded(model):
        print(f"[VRAM] {model} not loaded, skipping unload")
        return True

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

    Reference-counted: nested calls (upscale_batch → upscale_single)
    and concurrent requests skip redundant unload cycles.

    Between consecutive batch chunks (separate HTTP requests), checks
    /api/ps to see if model is actually loaded before doing the
    expensive unload + sleep(2) cycle. If model is already gone
    (common between chunks), the unload is a fast no-op.

    Usage:
        with ollama_vram_free():
            # Full 24GB VRAM available for rembg/upscale
            run_rembg(...)
    """
    global _active_count

    with _lock:
        _active_count += 1
        is_first = (_active_count == 1)

    if is_first:
        unload_model(model)
    else:
        print("[VRAM] Already in VRAM-free context, skipping unload")

    try:
        yield
    finally:
        with _lock:
            _active_count -= 1
            is_last = (_active_count == 0)

        if is_last:
            print("[VRAM] Ollama model will auto-reload on next LLM request")
