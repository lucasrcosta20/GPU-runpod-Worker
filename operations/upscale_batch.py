"""
Upscale batch operation — parallel Real-ESRGAN via spandrel + PyTorch CUDA.

Receives a list of images, processes them concurrently using
ThreadPoolExecutor. Returns results in the same order.

Note: GPU VRAM is shared across threads. max_parallel should be
conservative (2-3) to avoid OOM on large images.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from operations.upscale import upscale as upscale_single


def upscale_batch(
    items: List[Dict[str, Any]],
    upscale_factor: int = 2,
    denoise_strength: float = 0.0,
    sharpen_amount: float = 0.2,
    sharpen_radius: float = 1.0,
    max_parallel: int = 2,
) -> Dict[str, Any]:
    """
    Upscale multiple images in parallel.

    Each item must have 'data' (base64) and 'filename'.
    Optional per-item overrides: 'upscale_factor', 'denoise_strength',
    'sharpen_amount', 'sharpen_radius'.

    Args:
        items: List of dicts with 'data' and 'filename'.
        upscale_factor: Default upscale factor (shared).
        denoise_strength: Default denoising strength.
        sharpen_amount: Default sharpening intensity.
        sharpen_radius: Default sharpening radius.
        max_parallel: Max concurrent threads (conservative for VRAM).

    Returns:
        Dict with 'results' (list in same order), 'total_time_seconds',
        'successful', 'failed'.
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

    results: List[Optional[Dict[str, Any]]] = [None] * len(items)

    def _process_item(index: int, item: Dict[str, Any]) -> tuple:
        """Process a single item and return (index, result_dict)."""
        data = item.get("data", "")
        filename = item.get("filename", f"image_{index}.png")

        if not data:
            return index, {
                "filename": filename,
                "data": None,
                "success": False,
                "error": "Empty image data",
                "processing_time_seconds": 0,
            }

        result = upscale_single(
            image_data=data,
            filename=filename,
            upscale_factor=item.get("upscale_factor", upscale_factor),
            denoise_strength=item.get("denoise_strength", denoise_strength),
            sharpen_amount=item.get("sharpen_amount", sharpen_amount),
            sharpen_radius=item.get("sharpen_radius", sharpen_radius),
        )
        return index, result

    successful = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(_process_item, i, item): i
            for i, item in enumerate(items)
        }

        for future in as_completed(futures):
            index, result = future.result()
            results[index] = result
            if result.get("success"):
                successful += 1
            else:
                failed += 1

    elapsed = time.time() - start

    return {
        "results": results,
        "total_time_seconds": round(elapsed, 2),
        "successful": successful,
        "failed": failed,
        "total_items": len(items),
    }
