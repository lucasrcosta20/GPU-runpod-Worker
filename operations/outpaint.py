"""
Outpainting Operation — Extend image borders using FLUX.1 Fill Dev

Uses the diffusers FluxFillPipeline to extend an image in any direction,
filling new areas with contextually coherent content.

Requirements:
- diffusers>=0.31.0
- transformers>=4.44.0
- torch (pre-installed in RunPod image)
- ~20GB VRAM (A5000/A40/A100)
"""

import base64
import io
import time
from typing import Optional

import numpy as np
from PIL import Image


# Lazy-loaded pipeline (heavy model, load once)
_pipeline = None


def _get_pipeline():
    """Lazy-load the Flux Fill pipeline."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    import torch
    from diffusers import FluxFillPipeline

    print("Loading FLUX.1 Fill Dev pipeline...")
    start = time.time()

    _pipeline = FluxFillPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Fill-dev",
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    # Enable memory optimizations
    _pipeline.enable_model_cpu_offload()

    print(f"Pipeline loaded in {time.time() - start:.1f}s")
    return _pipeline


def outpaint(
    image_data: str,
    target_width: int = 928,
    target_height: int = 1152,
    prompt: str = "extend the scene naturally, maintaining the same style and lighting",
    num_inference_steps: int = 28,
    guidance_scale: float = 30.0,
    seed: Optional[int] = None,
) -> dict:
    """
    Extend an image to fill a target canvas size.

    The original image is placed centered on the target canvas,
    and the empty areas are filled by the AI model.

    Args:
        image_data: Base64-encoded input image
        target_width: Target canvas width
        target_height: Target canvas height
        prompt: Text prompt to guide the outpainting
        num_inference_steps: Number of diffusion steps (more = better quality)
        guidance_scale: How closely to follow the prompt
        seed: Random seed for reproducibility

    Returns:
        Dict with 'image' (base64 result), 'processing_time', 'success'
    """
    try:
        import torch

        start = time.time()

        # Decode input image
        img_bytes = base64.b64decode(image_data)
        source_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        src_w, src_h = source_image.size

        # Create canvas with the source image centered
        canvas = Image.new("RGB", (target_width, target_height), (255, 255, 255))
        paste_x = (target_width - src_w) // 2
        paste_y = (target_height - src_h) // 2

        # If source is larger than target, resize to fit first
        if src_w > target_width or src_h > target_height:
            scale = min(target_width / src_w, target_height / src_h) * 0.9
            new_w = int(src_w * scale)
            new_h = int(src_h * scale)
            source_image = source_image.resize((new_w, new_h), Image.LANCZOS)
            src_w, src_h = new_w, new_h
            paste_x = (target_width - src_w) // 2
            paste_y = (target_height - src_h) // 2

        canvas.paste(source_image, (paste_x, paste_y))

        # Create mask (white = areas to fill, black = keep original)
        mask = Image.new("L", (target_width, target_height), 255)
        mask.paste(0, (paste_x, paste_y, paste_x + src_w, paste_y + src_h))

        # Load pipeline
        pipe = _get_pipeline()

        # Generate
        generator = torch.Generator("cuda").manual_seed(seed) if seed else None

        result = pipe(
            prompt=prompt,
            image=canvas,
            mask_image=mask,
            width=target_width,
            height=target_height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        # Encode result
        buffer = io.BytesIO()
        result.save(buffer, format="JPEG", quality=95)
        result_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        processing_time = time.time() - start

        return {
            "success": True,
            "image": result_b64,
            "processing_time": processing_time,
            "dimensions": {"width": target_width, "height": target_height},
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"{type(e).__name__}: {str(e)[:300]}",
        }
