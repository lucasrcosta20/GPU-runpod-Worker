"""
Resize operation — Pillow image resizing.

Receives 1 image in base64, returns 1 resized image in base64.
"""

import base64
import io
import os
import time
from typing import Any, Dict, List

from PIL import Image


def resize(
    image_data: str,
    filename: str = "image.png",
    target_size: List[int] = None,
    jpg_quality: int = 98,
) -> Dict[str, Any]:
    """
    Resize a single image.

    Args:
        image_data: Base64-encoded image.
        filename: Original filename.
        target_size: [width, height] target dimensions.
        jpg_quality: JPEG output quality.

    Returns:
        Dict with 'filename', 'data' (base64), 'success', 'error',
        'processing_time_seconds', 'gpu_device'.
    """
    if target_size is None:
        target_size = [1392, 1152]

    start = time.time()

    try:
        raw = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(raw))

        target_w, target_h = target_size[0], target_size[1]
        image = image.resize((target_w, target_h), Image.LANCZOS)

        buf = io.BytesIO()
        ext = os.path.splitext(filename)[1].lower()
        if ext in (".jpg", ".jpeg"):
            if image.mode == "RGBA":
                image = image.convert("RGB")
            image.save(buf, format="JPEG", quality=jpg_quality, optimize=True)
        else:
            image.save(buf, format="PNG")

        result_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        elapsed = time.time() - start
        return {
            "filename": filename,
            "data": result_b64,
            "success": True,
            "error": None,
            "processing_time_seconds": round(elapsed, 2),
            "gpu_device": "cpu",
        }

    except Exception as e:
        elapsed = time.time() - start
        return {
            "filename": filename,
            "data": None,
            "success": False,
            "error": str(e),
            "processing_time_seconds": round(elapsed, 2),
            "gpu_device": "cpu",
        }
