"""
GPU Worker Handler — Runpod Serverless

Generic worker that processes atomic operations:
- health: GPU info, Ollama status, loaded models
- llm: Text generation via Ollama localhost (text-only + multimodal)
- remove_bg: Background removal via rembg (onnxruntime-gpu)
- upscale: Real-ESRGAN upscale via spandrel + PyTorch CUDA
- resize: Image resizing via Pillow

All pipeline logic stays in ia-cadastro. This worker only
executes individual operations.
"""

import runpod

from security.hmac_validator import validate_hmac
from operations.health import check_health
from operations.llm import generate as llm_generate
from operations.llm_batch import generate_batch as llm_generate_batch
from operations.remove_bg import remove_background
from operations.remove_bg_batch import remove_background_batch
from operations.upscale import upscale
from operations.upscale_batch import upscale_batch as upscale_batch_op
from operations.resize import resize


VALID_OPERATIONS = {
    "health", "llm", "llm_batch",
    "remove_bg", "remove_bg_batch",
    "upscale", "upscale_batch",
    "resize",
}


def handler(event: dict) -> dict:
    """
    Main handler for Runpod Serverless.

    Args:
        event: Runpod event with 'input' dict containing 'operation' and params.

    Returns:
        Dict with operation result or error.
    """
    try:
        input_data = event.get("input", {})
        operation = input_data.get("operation", "")

        # Validate operation
        if operation not in VALID_OPERATIONS:
            return {
                "error": f"Unknown operation: '{operation}'. "
                         f"Valid: {', '.join(sorted(VALID_OPERATIONS))}",
            }

        # Validate HMAC (skip for health — allows unauthenticated health checks)
        if operation != "health":
            signature = input_data.get("hmac_signature")
            if not validate_hmac(input_data, signature):
                return {"error": "Invalid HMAC signature"}

        # Route to operation
        if operation == "health":
            return check_health()

        if operation == "llm":
            return _handle_llm(input_data)

        if operation == "llm_batch":
            return _handle_llm_batch(input_data)

        if operation == "remove_bg":
            return _handle_remove_bg(input_data)

        if operation == "remove_bg_batch":
            return _handle_remove_bg_batch(input_data)

        if operation == "upscale":
            return _handle_upscale(input_data)

        if operation == "upscale_batch":
            return _handle_upscale_batch(input_data)

        if operation == "resize":
            return _handle_resize(input_data)

    except Exception as e:
        return {"error": f"Worker error: {type(e).__name__}: {str(e)[:500]}"}


def _handle_llm(data: dict) -> dict:
    """Route LLM operation."""
    model = data.get("model", "llama3.1:8b")
    prompt = data.get("prompt", "")
    system_prompt = data.get("system_prompt", "")
    temperature = data.get("temperature", 0.7)
    max_tokens = data.get("max_tokens", 2000)
    timeout = data.get("timeout", 300)
    images = data.get("images")

    if not prompt:
        return {"error": "Missing 'prompt' field"}

    return llm_generate(
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        images=images,
    )


def _handle_llm_batch(data: dict) -> dict:
    """Route LLM batch operation."""
    items = data.get("items", [])
    if not items:
        return {"error": "Missing 'items' field (list of prompts)"}

    model = data.get("model", "llama3.1:8b")
    system_prompt = data.get("system_prompt", "")
    temperature = data.get("temperature", 0.7)
    max_tokens = data.get("max_tokens", 2000)
    timeout = data.get("timeout", 300)
    max_parallel = data.get("max_parallel", 4)

    return llm_generate_batch(
        items=items,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_parallel=max_parallel,
    )


def _handle_remove_bg(data: dict) -> dict:
    """Route remove_bg operation."""
    image = data.get("image", {})
    image_data = image.get("data", "")
    filename = image.get("filename", "image.png")
    config = data.get("config", {})
    bg_model = config.get("bg_model", "birefnet-general")

    if not image_data:
        return {"error": "Missing 'image.data' field (base64)"}

    return remove_background(
        image_data=image_data,
        filename=filename,
        bg_model=bg_model,
    )


def _handle_upscale(data: dict) -> dict:
    """Route upscale operation."""
    image = data.get("image", {})
    image_data = image.get("data", "")
    filename = image.get("filename", "image.png")
    config = data.get("config", {})

    if not image_data:
        return {"error": "Missing 'image.data' field (base64)"}

    return upscale(
        image_data=image_data,
        filename=filename,
        upscale_factor=config.get("upscale_factor", 2),
        denoise_strength=config.get("denoise_strength", 0.0),
        sharpen_amount=config.get("sharpen_amount", 0.2),
        sharpen_radius=config.get("sharpen_radius", 1.0),
    )


def _handle_remove_bg_batch(data: dict) -> dict:
    """Route remove_bg_batch operation."""
    items = data.get("items", [])
    if not items:
        return {"error": "Missing 'items' field (list of images)"}

    config = data.get("config", {})
    bg_model = config.get("bg_model", "birefnet-general")
    max_parallel = data.get("max_parallel", 3)

    return remove_background_batch(
        items=items,
        bg_model=bg_model,
        max_parallel=max_parallel,
    )


def _handle_upscale_batch(data: dict) -> dict:
    """Route upscale_batch operation."""
    items = data.get("items", [])
    if not items:
        return {"error": "Missing 'items' field (list of images)"}

    config = data.get("config", {})
    max_parallel = data.get("max_parallel", 2)

    return upscale_batch_op(
        items=items,
        upscale_factor=config.get("upscale_factor", 2),
        denoise_strength=config.get("denoise_strength", 0.0),
        sharpen_amount=config.get("sharpen_amount", 0.2),
        sharpen_radius=config.get("sharpen_radius", 1.0),
        max_parallel=max_parallel,
    )


def _handle_resize(data: dict) -> dict:
    """Route resize operation."""
    image = data.get("image", {})
    image_data = image.get("data", "")
    filename = image.get("filename", "image.png")
    config = data.get("config", {})

    if not image_data:
        return {"error": "Missing 'image.data' field (base64)"}

    return resize(
        image_data=image_data,
        filename=filename,
        target_size=config.get("target_size", [1392, 1152]),
        jpg_quality=config.get("jpg_quality", 98),
    )


# Start Runpod Serverless handler (skip when running as Pod HTTP server)
import os
if os.environ.get("POD_MODE") != "1":
    runpod.serverless.start({"handler": handler})
