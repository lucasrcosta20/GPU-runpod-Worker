"""
Ollama VRAM management — unload/reload model to free GPU memory.

When rembg (onnxruntime) or upscale (PyTorch) need VRAM, the Ollama
model must be temporarily unloaded. After the operation completes,
the model reloads automatically on the next LLM request.

Two modes of operation:

1. Per-request (context manager):
    with ollama_vram_free():
        run_rembg(...)

2. Job-level hold (for multi-chunk image jobs):
    hold_vram()    # Called once at job start — unloads Ollama
    # ... multiple chunk requests run without re-checking ...
    release_vram() # Called at job end — allows Ollama to reload
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

# Job-level VRAM hold: when > 0, Ollama stays unloaded across
# multiple HTTP requests (avoids per-chunk unload/check overhead).
_hold_count = 0


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


def hold_vram(model: str = DEFAULT_MODEL) -> bool:
    """
    Hold VRAM free for the duration of a multi-chunk image job.

    Called once at job start. Unloads Ollama model and increments
    the hold counter. While held, ollama_vram_free() is a no-op
    (model is already unloaded and won't reload between chunks).

    Thread-safe: multiple concurrent holds are reference-counted.

    Args:
        model: Model name to unload.

    Returns:
        True if VRAM was freed (or already held).
    """
    global _hold_count

    with _lock:
        _hold_count += 1
        is_first = (_hold_count == 1)

    if is_first:
        print("[VRAM] Hold: unloading Ollama for multi-chunk job")
        return unload_model(model)
    else:
        print(f"[VRAM] Hold: already held (count={_hold_count})")
        return True


def release_vram() -> None:
    """
    Release VRAM hold after a multi-chunk image job completes.

    Decrements the hold counter. When it reaches 0, Ollama is
    free to reload on the next LLM request.

    Should be called when the job finishes, fails, or is cancelled.
    """
    global _hold_count

    with _lock:
        _hold_count = max(0, _hold_count - 1)
        is_last = (_hold_count == 0)

    if is_last:
        print("[VRAM] Release: Ollama free to reload on next LLM request")
    else:
        print(f"[VRAM] Release: still held (count={_hold_count})")


def is_vram_held() -> bool:
    """Check if VRAM is currently held by a job."""
    with _lock:
        return _hold_count > 0


@contextmanager
def ollama_vram_free(model: str = DEFAULT_MODEL):
    """
    Context manager that frees VRAM by unloading Ollama model.

    If VRAM is already held (via hold_vram), this is a no-op —
    the model is already unloaded and will stay unloaded until
    release_vram() is called.

    Reference-counted: nested calls (upscale_batch → upscale_single)
    skip redundant unload cycles.

    Usage:
        with ollama_vram_free():
            # Full 24GB VRAM available for rembg/upscale
            run_rembg(...)
    """
    global _active_count

    # If VRAM is held by a job, skip unload entirely
    if is_vram_held():
        print("[VRAM] Held by job, skipping per-request unload")
        yield
        return

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
