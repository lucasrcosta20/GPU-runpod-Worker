"""
Background removal operation — rembg with GPU (onnxruntime-gpu).

Receives 1 image in base64, returns 1 image (PNG transparent) in base64.

VRAM strategy: Ollama model is unloaded before first rembg inference
to free ~13GB VRAM for onnxruntime CUDAExecutionProvider.
"""

import base64
import gc
import io
import os
import time
from typing import Any, Dict

from PIL import Image

from operations.gpu_info import get_gpu_name
from operations.ollama_vram import ollama_vram_free


# ── Ensure cuDNN is discoverable BEFORE onnxruntime is imported ──
# onnxruntime uses dlopen("libcudnn.so.9") which reads LD_LIBRARY_PATH
# at load time. If the path isn't set before the first import, CUDA
# initialization fails and gets cached permanently.
# This runs at module import time (before any onnxruntime import).
def _ensure_cudnn_path():
    """Add cuDNN library paths to LD_LIBRARY_PATH if not already present.

    IMPORTANT: Only add pip-installed nvidia-cudnn (9.1.0, compatible with CUDA 12.4).
    Do NOT add Ollama's bundled cuDNN (9.20.0) — it's incompatible and causes
    cudnnCreate() to fail with CUDNN_STATUS_NOT_INITIALIZED (error 1001).
    """
    current = os.environ.get("LD_LIBRARY_PATH", "")

    # pip-installed nvidia-cudnn (compatible version)
    try:
        import nvidia.cudnn
        pip_path = os.path.dirname(nvidia.cudnn.__file__) + "/lib"
        if os.path.isdir(pip_path) and pip_path not in current:
            os.environ["LD_LIBRARY_PATH"] = pip_path + ":" + current
            print(f"[REMBG] Added pip cuDNN to LD_LIBRARY_PATH: {pip_path}")
    except ImportError:
        pass

_ensure_cudnn_path()


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
                    # Real test: create a minimal ONNX model and run it with CUDA.
                    # This catches cuDNN missing/incompatible at detection time
                    # rather than failing on the first real rembg inference.
                    #
                    # The model is built as raw protobuf bytes — no `onnx` package needed.
                    # It's a trivial Identity op (X → Y, float[1]).
                    import tempfile

                    def _build_minimal_onnx() -> bytes:
                        """Build minimal valid ONNX model as raw protobuf bytes."""
                        def varint(n):
                            out = bytearray()
                            while n > 0x7F:
                                out.append((n & 0x7F) | 0x80)
                                n >>= 7
                            out.append(n & 0x7F)
                            return bytes(out)

                        def field(fnum, wire, data):
                            tag = (fnum << 3) | wire
                            if wire == 2:  # length-delimited
                                return varint(tag) + varint(len(data)) + data
                            if wire == 0:  # varint
                                return varint(tag) + varint(data)
                            return b""

                        # TensorTypeProto: elem_type=1 (FLOAT)
                        tensor_type = field(1, 0, 1)
                        # TypeProto: tensor_type (field 1)
                        type_proto = field(1, 2, tensor_type)
                        # ValueInfoProto for X and Y
                        vi_x = field(1, 2, b"X") + field(2, 2, type_proto)
                        vi_y = field(1, 2, b"Y") + field(2, 2, type_proto)
                        # NodeProto: input="X", output="Y", op_type="Identity"
                        node = field(1, 2, b"X") + field(2, 2, b"Y") + field(4, 2, b"Identity")
                        # GraphProto
                        graph = field(1, 2, node) + field(2, 2, b"test") + field(11, 2, vi_x) + field(12, 2, vi_y)
                        # OperatorSetIdProto: version=13
                        opset = field(2, 0, 13)
                        # ModelProto: ir_version=7, opset_import, graph
                        return field(1, 0, 7) + field(8, 2, opset) + field(7, 2, graph)

                    model_bytes = _build_minimal_onnx()
                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                            f.write(model_bytes)
                            tmp_path = f.name

                        test_session = ort.InferenceSession(
                            tmp_path,
                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                        )
                        active = test_session.get_providers()
                        del test_session
                    finally:
                        if tmp_path and os.path.exists(tmp_path):
                            os.unlink(tmp_path)

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
