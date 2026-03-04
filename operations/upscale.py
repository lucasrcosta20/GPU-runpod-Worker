"""
Upscale operation — Real-ESRGAN via spandrel + PyTorch CUDA.

Receives 1 image in base64, returns 1 upscaled image in base64.
Pipeline: sRGB normalize → Denoise (optional) → Upscale → Sharpen (optional)
Replicates the exact same pipeline as ia-cadastro's QualityEnhancer.

VRAM strategy: Ollama model is unloaded before PyTorch inference
to free ~13GB VRAM for Real-ESRGAN.
"""

import base64
import gc
import io
import os
import time
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image, ImageCms

from operations.gpu_info import get_gpu_name
from operations.ollama_vram import ollama_vram_free


# Model cache
_models: Dict[int, Any] = {}
_device = None

MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")

MODEL_NAMES = {
    2: "RealESRGAN_x2plus.pth",
    4: "RealESRGAN_x4plus.pth",
}

# Tile processing to avoid OOM
TILE_SIZE = 512
OVERLAP = 32
MAX_INPUT_PIXELS = 2048 * 2048


def upscale(
    image_data: str,
    filename: str = "image.png",
    upscale_factor: int = 2,
    denoise_strength: float = 0.0,
    sharpen_amount: float = 0.2,
    sharpen_radius: float = 1.0,
) -> Dict[str, Any]:
    """
    Upscale a single image using Real-ESRGAN.

    Args:
        image_data: Base64-encoded image.
        filename: Original filename.
        upscale_factor: 2 or 4.
        denoise_strength: Denoising strength (0 = disabled).
        sharpen_amount: Sharpening intensity (0-1).
        sharpen_radius: Sharpening radius in pixels.

    Returns:
        Dict with 'filename', 'data' (base64), 'success', 'error',
        'processing_time_seconds', 'gpu_device'.
    """
    start = time.time()

    try:
        # Decode input
        raw = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(raw))
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 1. Normalize sRGB
        image = _normalize_srgb(image)

        # 2. Denoise (if enabled)
        if denoise_strength > 0:
            image = _denoise(image, denoise_strength)

        # 3. Upscale (free Ollama VRAM first for PyTorch)
        with ollama_vram_free():
            image = _upscale(image, upscale_factor)

        # 4. Sharpen (if enabled)
        if sharpen_amount > 0:
            image = _sharpen(image, sharpen_amount, sharpen_radius)

        # Encode result
        buf = io.BytesIO()
        ext = os.path.splitext(filename)[1].lower()
        if ext in (".jpg", ".jpeg"):
            image.save(buf, format="JPEG", quality=98, optimize=True)
        else:
            image.save(buf, format="PNG")

        result_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        del image, raw, buf
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


def _get_device():
    """Get torch device (cached)."""
    global _device
    if _device is None:
        import torch
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def _get_model(factor: int):
    """Load Real-ESRGAN model (lazy, cached)."""
    if factor not in _models:
        import spandrel
        import torch

        model_name = MODEL_NAMES.get(factor)
        if not model_name:
            raise ValueError(f"Unsupported upscale factor: {factor}")

        model_path = os.path.join(MODELS_DIR, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = spandrel.ModelLoader().load_from_file(model_path)
        model = model.to(_get_device())
        model.eval()
        _models[factor] = model

    return _models[factor]


def _normalize_srgb(image: Image.Image) -> Image.Image:
    """Normalize image to sRGB color space."""
    if "icc_profile" in image.info:
        try:
            srgb_profile = ImageCms.createProfile("sRGB")
            input_profile = ImageCms.ImageCmsProfile(
                io.BytesIO(image.info["icc_profile"])
            )
            image = ImageCms.profileToProfile(image, input_profile, srgb_profile)
        except Exception:
            pass
    return image


def _denoise(image: Image.Image, strength: float) -> Image.Image:
    """Apply denoising preserving details."""
    try:
        import cv2

        img_np = np.array(image)
        denoised = cv2.fastNlMeansDenoisingColored(
            img_np, None, strength, strength, 7, 21
        )
        return Image.fromarray(denoised)
    except Exception:
        return image


def _upscale(image: Image.Image, factor: int) -> Image.Image:
    """Upscale with Real-ESRGAN in tiles to avoid OOM."""
    import torch

    model = _get_model(factor)
    device = _get_device()

    # Limit input size to avoid OOM
    if image.width * image.height > MAX_INPUT_PIXELS:
        ratio = (MAX_INPUT_PIXELS / (image.width * image.height)) ** 0.5
        new_w = int(image.width * ratio)
        new_h = int(image.height * ratio)
        image = image.resize((new_w, new_h), Image.LANCZOS)

    img_np = np.array(image).astype(np.float32) / 255.0
    h, w, c = img_np.shape

    # Small image — process directly
    if h <= TILE_SIZE and w <= TILE_SIZE:
        return _upscale_tensor(img_np, model, device)

    # Tile processing
    out_h, out_w = h * factor, w * factor
    output_np = np.zeros((out_h, out_w, c), dtype=np.float32)

    for y in range(0, h, TILE_SIZE - OVERLAP):
        for x in range(0, w, TILE_SIZE - OVERLAP):
            y_end = min(y + TILE_SIZE, h)
            x_end = min(x + TILE_SIZE, w)
            y_start = max(0, y_end - TILE_SIZE)
            x_start = max(0, x_end - TILE_SIZE)

            tile = img_np[y_start:y_end, x_start:x_end, :]

            tile_tensor = (
                torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(device)
            )
            with torch.no_grad():
                tile_out = model(tile_tensor)
            tile_result = tile_out.squeeze(0).permute(1, 2, 0).cpu().numpy()
            del tile_tensor, tile_out

            paste_y = y * factor
            paste_x = x * factor
            crop_y = (y - y_start) * factor
            crop_x = (x - x_start) * factor
            paste_h = (min(y + TILE_SIZE, h) - y) * factor
            paste_w = (min(x + TILE_SIZE, w) - x) * factor

            output_np[
                paste_y : paste_y + paste_h, paste_x : paste_x + paste_w, :
            ] = tile_result[
                crop_y : crop_y + paste_h, crop_x : crop_x + paste_w, :
            ]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    output_np = (output_np * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(output_np)


def _upscale_tensor(img_np, model, device) -> Image.Image:
    """Upscale a small image directly (fits in memory)."""
    import torch

    img_tensor = (
        torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
    )
    with torch.no_grad():
        output = model(img_tensor)
    del img_tensor

    output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    del output
    output_np = (output_np * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(output_np)


def _sharpen(
    image: Image.Image, amount: float, radius: float
) -> Image.Image:
    """Apply sharpening (Unsharp Mask)."""
    try:
        import cv2

        img_np = np.array(image).astype(np.float32)
        kernel_size = int(radius * 2) * 2 + 1
        blurred = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), radius)
        sharpened = img_np + amount * (img_np - blurred)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        return Image.fromarray(sharpened)
    except Exception:
        from PIL import ImageEnhance

        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(1 + amount)
