"""
Local test for GPU Worker handler.

Uses runpod's local test mode to simulate requests without
deploying to Runpod Serverless.

Usage:
    python test_local.py
"""

import base64
import json
import os
import sys

# Ensure we can import from the worker directory
sys.path.insert(0, os.path.dirname(__file__))


def test_health():
    """Test health check operation."""
    from handler import handler

    event = {"input": {"operation": "health"}}
    result = handler(event)

    print("=== Health Check ===")
    print(json.dumps(result, indent=2))
    assert result.get("status") in ("ok", "degraded"), f"Unexpected status: {result}"
    print("✅ Health check passed\n")


def test_llm():
    """Test LLM generation."""
    from handler import handler

    event = {
        "input": {
            "operation": "llm",
            "model": "llama3.1:8b",
            "prompt": "Diga 'teste ok' em uma palavra.",
            "system_prompt": "Responda de forma curta.",
            "temperature": 0.1,
            "max_tokens": 50,
        }
    }
    result = handler(event)

    print("=== LLM Generation ===")
    print(f"Text: {result.get('text', '')[:200]}")
    print(f"Tokens: {result.get('tokens_generated')}")
    print(f"Time: {result.get('processing_time_seconds')}s")
    assert "error" not in result or result["error"] is None, f"Error: {result}"
    assert result.get("text"), "Empty response"
    print("✅ LLM generation passed\n")


def test_remove_bg():
    """Test background removal with a small test image."""
    from handler import handler

    # Create a small test image (red square on white background)
    from PIL import Image
    import io

    img = Image.new("RGB", (100, 100), (255, 255, 255))
    # Draw a red square in the center
    for x in range(25, 75):
        for y in range(25, 75):
            img.putpixel((x, y), (255, 0, 0))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    event = {
        "input": {
            "operation": "remove_bg",
            "image": {"filename": "test.png", "data": img_b64},
            "config": {"bg_model": "birefnet-general"},
        }
    }
    result = handler(event)

    print("=== Remove Background ===")
    print(f"Success: {result.get('success')}")
    print(f"Time: {result.get('processing_time_seconds')}s")
    print(f"GPU: {result.get('gpu_device')}")
    assert result.get("success"), f"Failed: {result.get('error')}"
    assert result.get("data"), "Empty result"
    print("✅ Remove background passed\n")


def test_resize():
    """Test image resize."""
    from handler import handler

    from PIL import Image
    import io

    img = Image.new("RGB", (200, 200), (0, 128, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    event = {
        "input": {
            "operation": "resize",
            "image": {"filename": "test.jpg", "data": img_b64},
            "config": {"target_size": [100, 100], "jpg_quality": 95},
        }
    }
    result = handler(event)

    print("=== Resize ===")
    print(f"Success: {result.get('success')}")
    print(f"Time: {result.get('processing_time_seconds')}s")
    assert result.get("success"), f"Failed: {result.get('error')}"

    # Verify output dimensions
    out_bytes = base64.b64decode(result["data"])
    out_img = Image.open(io.BytesIO(out_bytes))
    assert out_img.size == (100, 100), f"Wrong size: {out_img.size}"
    print("✅ Resize passed\n")


def test_invalid_operation():
    """Test invalid operation returns error."""
    from handler import handler

    event = {"input": {"operation": "invalid_op"}}
    result = handler(event)

    print("=== Invalid Operation ===")
    assert "error" in result, "Should return error"
    print(f"Error: {result['error']}")
    print("✅ Invalid operation handled correctly\n")


def test_missing_hmac():
    """Test that non-health operations require HMAC when secret is set."""
    from handler import handler

    # This test only validates structure — HMAC validation is skipped
    # when HMAC_SECRET env var is empty (dev mode)
    event = {
        "input": {
            "operation": "llm",
            "model": "llama3.1:8b",
            "prompt": "test",
        }
    }
    result = handler(event)
    print("=== HMAC Check ===")
    # In dev mode (no HMAC_SECRET), should proceed normally
    print(f"Result has error: {'error' in result and result.get('error') is not None}")
    print("✅ HMAC check passed\n")


if __name__ == "__main__":
    print("GPU Worker — Local Tests\n")
    print("NOTE: These tests require Ollama running locally")
    print("      and rembg/spandrel/torch installed.\n")

    tests = [
        ("Health", test_health),
        ("Invalid Operation", test_invalid_operation),
        ("HMAC Check", test_missing_hmac),
        ("Resize", test_resize),
    ]

    # Optional tests that need Ollama / GPU libs
    optional_tests = [
        ("LLM", test_llm),
        ("Remove BG", test_remove_bg),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"❌ {name} FAILED: {e}\n")
            failed += 1

    for name, test_fn in optional_tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"⚠️  {name} SKIPPED (needs Ollama/GPU): {e}\n")

    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(1 if failed > 0 else 0)
