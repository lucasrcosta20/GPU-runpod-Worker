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

# Track whether CUDA is available for onnxruntime (detected once, cached)
_cuda_available: bool | None = None


def clear_sessions():
    """
    Clear cached rembg sessions to free VRAM.

    Called after batch/single operations complete so onnxruntime
    releases GPU memory for Ollama to use on next LLM request.
    Does NOT reset _cuda_available — that's detected once at startup.
    """
    global _sessions
    if _sessions:
        print(f"[REMBG] Clearing {len(_sessions)} cached session(s) to free VRAM")
        _sessions.clear()
        gc.collect()


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

        # Free VRAM: clear onnxruntime session so Ollama can use full GPU
        clear_sessions()

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
    """
    Get or create rembg session with GPU support (CUDA fallback to CPU).

    On first call, tests if CUDAExecutionProvider works. If cuDNN is
    missing or incompatible, falls back to CPU-only and caches that
    decision so subsequent calls skip the slow CUDA init attempt.
    """
    if model_name not in _sessions:
        import onnxruntime as ort
        from rembg.sessions import sessions_class

        global _cuda_available

        session_class = None
        for sc in sessions_class:
            if sc.name() == model_name:
                session_class = sc
                break

        if session_class is None:
            raise ValueError(f"rembg model not found: '{model_name}'")

        # Session options (shared for both GPU and CPU)
        sess_opts = ort.SessionOptions()
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_opts.intra_op_num_threads = 4
        sess_opts.inter_op_num_threads = 4

        # Detect CUDA availability once (test real initialization, not just provider list)
        if _cuda_available is None:
            available_providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in available_providers:
                try:
                    # Real test: create a minimal ONNX session with CUDA
                    # This catches cuDNN missing/incompatible at detection time
                    import numpy as np
                    import tempfile
                    import onnx
                    from onnx import helper, TensorProto

                    # Create minimal ONNX model (identity op)
                    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])
                    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])
                    node = helper.make_node("Identity", ["X"], ["Y"])
                    graph = helper.make_graph([node], "test", [X], [Y])
                    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

                    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                        onnx.save(model, f.name)
                        test_session = ort.InferenceSession(
                            f.name,
                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                        )
                        # Check which provider was actually used
                        active = test_session.get_providers()
                        del test_session
                        import os
                        os.unlink(f.name)

                    if "CUDAExecutionProvider" in active:
                        _cuda_available = True
                        print("[REMBG] CUDAExecutionProvider tested OK — using GPU")
                    else:
                        _cuda_available = False
                        print("[REMBG] CUDAExecutionProvider fell back to CPU during test — using CPU only")
                except Exception as e:
                    _cuda_available = False
                    print(f"[REMBG] CUDAExecutionProvider test failed: {e} — using CPU only")
            else:
                _cuda_available = False
                print("[REMBG] CUDAExecutionProvider not installed — using CPU")

        # Build provider list based on detection
        if _cuda_available:
            cuda_provider_options = {
                "device_id": "0",
                "arena_extend_strategy": "kSameAsRequested",
                "gpu_mem_limit": str(16 * 1024 * 1024 * 1024),
            }
            ort_providers = [
                ("CUDAExecutionProvider", cuda_provider_options),
                "CPUExecutionProvider",
            ]
        else:
            ort_providers = ["CPUExecutionProvider"]

        try:
            session = session_class(model_name, sess_opts, providers=ort_providers)
        except TypeError:
            session = session_class(model_name, sess_opts)

        _sessions[model_name] = session

    return _sessions[model_name]
