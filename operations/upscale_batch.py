"""
Upscale batch operation — sequential Real-ESRGAN via spandrel + PyTorch CUDA.

Processes images SEQUENTIALLY to:
1. Avoid VRAM OOM from multiple concurrent PyTorch inferences
2. Force gc.collect() + torch.cuda.empty_cache() between images
3. Share single model load in VRAM

Ollama model is unloaded once at the start of the batch.
"""

import gc
import time
from typing import Any, Dict, List

from operations.gpu_info import get_gpu_name
from operations.ollama_vram import ollama_vram_free
from operations.upscale import upscale as upscale_single


def upscale_batch(
    items: List[Dict[str, Any]],
    upscale_factor: int = 2,
    denoise_strength: float = 0.0,
    sharpen_amount: float = 0.2,
    sharpen_radius: float = 1.0,
    max_parallel: int = 2,  # ignored — always sequential for VRAM safety
) -> Dict[str, Any]:
    """
    Upscale multiple images sequentially.

    Unloads Ollama from VRAM once, processes all images with
    gc.collect() between each, then Ollama auto-reloads on next LLM request.

    Args:
        items: List of dicts with 'data' and 'filename'.
        upscale_factor: Default upscale factor.
        denoise_strength: Default denoising strength.
        sharpen_amount: Default sharpening intensity.
        sharpen_radius: Default sharpening radius.
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

    print(f"[UPSCALE_BATCH] Starting batch: {len(items)} images, factor={upscale_factor}")

    # Note: upscale_single already calls ollama_vram_free() internally,
    # but for batch we want to unload ONCE and keep it unloaded for all images.
    # The nested ollama_vram_free() inside upscale_single will be a no-op
    # since the model is already unloaded.
    with ollama_vram_free():
        for i, item in enumerate(items):
            data = item.get("data", "")
            filename = item.get("filename", f"image_{i}.png")

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

            result = upscale_single(
                image_data=data,
                filename=filename,
                upscale_factor=item.get("upscale_factor", upscale_factor),
                denoise_strength=item.get("denoise_strength", denoise_strength),
                sharpen_amount=item.get("sharpen_amount", sharpen_amount),
                sharpen_radius=item.get("sharpen_radius", sharpen_radius),
            )
            results.append(result)

            if result.get("success"):
                successful += 1
            else:
                failed += 1

            # Force cleanup between images
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            # Progress log every 5 images
            if (i + 1) % 5 == 0 or (i + 1) == len(items):
                elapsed_so_far = time.time() - start
                avg_per_image = elapsed_so_far / (i + 1)
                remaining = avg_per_image * (len(items) - i - 1)
                print(
                    f"[UPSCALE_BATCH] Progress: {i + 1}/{len(items)} "
                    f"({successful} ok, {failed} fail) "
                    f"avg={avg_per_image:.1f}s/img, ETA={remaining:.0f}s"
                )

    elapsed = time.time() - start
    print(
        f"[UPSCALE_BATCH] Complete: {len(items)} images in {elapsed:.1f}s "
        f"({successful} ok, {failed} fail)"
    )

    return {
        "results": results,
        "total_time_seconds": round(elapsed, 2),
        "successful": successful,
        "failed": failed,
        "total_items": len(items),
    }
