"""
Background removal batch operation — sequential rembg with GPU.

Processes images SEQUENTIALLY (not parallel) to:
1. Avoid RAM accumulation from onnxruntime intermediate buffers
2. Share single VRAM allocation for the rembg model
3. Force gc.collect() between images to prevent memory leaks

Ollama model is unloaded once at the start of the batch,
and auto-reloads on the next LLM request after batch completes.
"""

import base64
import gc
import io
import time
from typing import Any, Dict, List, Optional

from PIL import Image

from operations.gpu_info import get_gpu_name
from operations.ollama_vram import ollama_vram_free


# Shared session cache (same as remove_bg.py)
_sessions: Dict[str, Any] = {}


def remove_background_batch(
    items: List[Dict[str, Any]],
    bg_model: str = "birefnet-general",
    max_parallel: int = 3,  # ignored — always sequential for memory safety
) -> Dict[str, Any]:
    """
    Remove background from multiple images sequentially.

    Unloads Ollama from VRAM once, processes all images, then
    Ollama auto-reloads on next LLM request.

    Args:
        items: List of dicts with 'data' and 'filename'.
        bg_model: rembg model name (shared across all items).
        max_parallel: Ignored (kept for API compat). Always sequential.

    Returns:
        Dict with 'results', 'total_time_seconds', 'successful', 'failed'.
    """
    start = time.time()

    if not items:
        return {
            "results": [],
            "total_time_seconds": 0,
            "successful": 0,
            "failed": 0,
            "total_items": 0,
        }

    results: List[Dict[str, Any]] = []
    successful = 0
    failed = 0

    print(f"[REMBG_BATCH] Starting batch: {len(items)} images, model={bg_model}")

    with ollama_vram_free():
        # Load session once (model stays in VRAM for entire batch)
        session = _get_session(bg_model)

        from rembg import remove

        for i, item in enumerate(items):
            item_start = time.time()
            data = item.get("data", "")
            filename = item.get("filename", f"image_{i}.png")
            item_model = item.get("bg_model", bg_model)

            if not data:
                results.append({
                    "filename": filename,
                    "data": None,
                    "success": False,
                    "error": "Empty image data",
                    "processing_time_seconds": 0,
                })
                failed += 1
                continue

            try:
                # Use per-item model if different from batch default
                if item_model != bg_model:
                    item_session = _get_session(item_model)
                else:
                    item_session = session

                # Decode
                raw = base64.b64decode(data)
                image = Image.open(io.BytesIO(raw))

                # Remove background
                result_img = remove(image, session=item_session)

                # Encode result
                buf = io.BytesIO()
                result_img.save(buf, format="PNG")
                result_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

                # Aggressive cleanup to prevent RAM leak
                del image, result_img, raw, buf

                elapsed_item = time.time() - item_start
                results.append({
                    "filename": filename,
                    "data": result_b64,
                    "success": True,
                    "error": None,
                    "processing_time_seconds": round(elapsed_item, 2),
                    "gpu_device": get_gpu_name(),
                })
                successful += 1

            except Exception as e:
                elapsed_item = time.time() - item_start
                results.append({
                    "filename": filename,
                    "data": None,
                    "success": False,
                    "error": str(e),
                    "processing_time_seconds": round(elapsed_item, 2),
                    "gpu_device": get_gpu_name(),
                })
                failed += 1

            # Force garbage collection every image to prevent RAM accumulation
            gc.collect()

            # Progress log every 10 images
            if (i + 1) % 10 == 0 or (i + 1) == len(items):
                elapsed_so_far = time.time() - start
                avg_per_image = elapsed_so_far / (i + 1)
                remaining = avg_per_image * (len(items) - i - 1)
                print(
                    f"[REMBG_BATCH] Progress: {i + 1}/{len(items)} "
                    f"({successful} ok, {failed} fail) "
                    f"avg={avg_per_image:.1f}s/img, ETA={remaining:.0f}s"
                )

    elapsed = time.time() - start
    print(
        f"[REMBG_BATCH] Complete: {len(items)} images in {elapsed:.1f}s "
        f"({successful} ok, {failed} fail)"
    )

    return {
        "results": results,
        "total_time_seconds": round(elapsed, 2),
        "successful": successful,
        "failed": failed,
        "total_items": len(items),
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

        sess_opts = ort.SessionOptions()
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_opts.intra_op_num_threads = 4
        sess_opts.inter_op_num_threads = 4

        _sessions[model_name] = session_class(model_name, sess_opts)

    return _sessions[model_name]
