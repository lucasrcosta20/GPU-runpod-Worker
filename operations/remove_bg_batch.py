"""
Background removal batch operation — parallel rembg with GPU.

Receives a list of images, processes them concurrently using
ThreadPoolExecutor. Returns results in the same order.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from operations.remove_bg import remove_background as remove_single


def remove_background_batch(
    items: List[Dict[str, Any]],
    bg_model: str = "birefnet-general",
    max_parallel: int = 3,
) -> Dict[str, Any]:
    """
    Remove background from multiple images in parallel.

    Each item must have 'data' (base64) and 'filename'.

    Args:
        items: List of dicts with 'data' and 'filename'.
        bg_model: rembg model name (shared across all items).
        max_parallel: Max concurrent processing threads.

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
        item_model = item.get("bg_model", bg_model)

        if not data:
            return index, {
                "filename": filename,
                "data": None,
                "success": False,
                "error": "Empty image data",
                "processing_time_seconds": 0,
            }

        result = remove_single(
            image_data=data,
            filename=filename,
            bg_model=item_model,
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
