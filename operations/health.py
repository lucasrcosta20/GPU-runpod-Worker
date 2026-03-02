"""
Health check operation.

Returns GPU info, Ollama status, and loaded models.
"""

import requests

from operations.gpu_info import get_gpu_name, get_vram_total_mb


OLLAMA_URL = "http://localhost:11434"


def check_health() -> dict:
    """
    Check worker health: GPU device, VRAM, Ollama status, loaded models.

    Returns:
        Dict with health information.
    """
    result = {
        "status": "ok",
        "gpu_device": get_gpu_name(),
        "vram_total_mb": get_vram_total_mb(),
        "ollama_available": False,
        "models_loaded": [],
    }

    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        if resp.status_code == 200:
            result["ollama_available"] = True
            data = resp.json()
            result["models_loaded"] = [
                m["name"] for m in data.get("models", [])
            ]
    except Exception:
        result["ollama_available"] = False

    if not result["ollama_available"]:
        result["status"] = "degraded"

    return result
