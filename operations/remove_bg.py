"""
Background removal operation — rembg with GPU (onnxruntime-gpu).

Receives 1 image in base64, returns 1 image (PNG transparent) in base64.

VRAM strategy: Ollama model is unloaded before first rembg inference
to free ~13GB VRAM for onnxruntime CUDAExecutionProvider.
"""

import base64
import gc
import io
import time
from typing import Any, Dict

from PIL import Image

from operations.gpu_info import get_gpu_name
from operations.ollama_vram import ollama_vram_free


# Cache sessions to avoid reloading models per request
_sessions: Dict[str, Any] = {}


def remove_background(
    image_data: str,
    filename: str = "image.png",
    bg_model: str = "birefnet-general",
) -> Dict[str, Any]:
    """
    Remove background from a single image.

    Unloads Ollama model from VRAM first to make room for onnxruntime.

    Args:
        image_data: Base64-encoded image.
        filename: Original filename (for response).
        bg_model: rembg model name.

    Returns:
        Dict with 'filename', 'data' (base64 PNG), 'success', 'error',
        'processing_time_seconds', 'gpu_device'.
    """
    start = time.time()

    try:
        from rembg import remove

        # Decode input image
        raw = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(raw))

        # Free VRAM from Ollama, then run rembg on GPU
        with ollama_vram_free():
            session = _get_session(bg_model)
            result = remove(image, session=session)

        # Encode result as PNG (transparent)
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        result_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Cleanup
        del image, result, raw, buf
        gc.collect()

        elapsed = time.time() - start
        return {
            "filename": filename,
            "data": result_b64,
            "success": True,
            "error": None,
            "processing_time_seconds": round(elapsed, 2),
            "gpu_device": get_gpu_name(),
        }

    except Exception as e:
        elapsed = time.time() - start
        return {
            "filename": filename,
            "data": None,
            "success": False,
            "error": str(e),
            "processing_time_seconds": round(elapsed, 2),
            "gpu_device": get_gpu_name(),
        }


def _get_session(model_name: str) -> Any:
    """Get or create rembg session with GPU-optimized settings."""
    if model_name not in _sessions:
        import onnxruntime as ort
        from rembg.sessions import sessions_class

        session_class = None
        for sc in sessions_class:
            if sc.name() == model_name:
                session_class = sc
                break

        if session_class is None:
            raise ValueError(f"rembg model not found: '{model_name}'")

        # GPU-optimized session options
        sess_opts = ort.SessionOptions()
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_opts.intra_op_num_threads = 4
        sess_opts.inter_op_num_threads = 4

        _sessions[model_name] = session_class(model_name, sess_opts)

    return _sessions[model_name]
