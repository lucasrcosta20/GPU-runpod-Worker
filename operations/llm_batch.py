"""
LLM Batch operation — parallel text generation via Ollama localhost.

Receives a list of prompts and processes them concurrently using
OLLAMA_NUM_PARALLEL. Returns results in the same order.

This eliminates per-request HTTP/HMAC/serialization overhead
when processing large batches (e.g. 500 product descriptions).
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from operations.llm import generate as generate_single


def generate_batch(
    items: List[Dict[str, Any]],
    model: str = "llama3.1:8b",
    system_prompt: str = "",
    temperature: float = 0.7,
    max_tokens: int = 2000,
    timeout: int = 300,
    max_parallel: int = 4,
) -> Dict[str, Any]:
    """
    Generate text for multiple prompts in parallel via Ollama.

    Each item in `items` must have at least a 'prompt' field.
    Optional per-item fields: 'system_prompt', 'images'.

    Args:
        items: List of dicts, each with 'prompt' and optional overrides.
        model: Model name (shared across all items).
        system_prompt: Default system prompt (item-level overrides allowed).
        temperature: Generation temperature.
        max_tokens: Maximum tokens per generation.
        timeout: Timeout per individual generation.
        max_parallel: Max concurrent Ollama requests (should match OLLAMA_NUM_PARALLEL).

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
        }

    results: List[Optional[Dict[str, Any]]] = [None] * len(items)

    def _process_item(index: int, item: Dict[str, Any]) -> tuple:
        """Process a single item and return (index, result_dict)."""
        prompt = item.get("prompt", "")
        if not prompt:
            return index, {
                "text": "",
                "success": False,
                "error": "Empty prompt",
                "tokens_generated": 0,
                "processing_time_seconds": 0,
            }

        item_system = item.get("system_prompt", system_prompt)
        item_images = item.get("images")

        try:
            result = generate_single(
                model=model,
                prompt=prompt,
                system_prompt=item_system,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                images=item_images,
            )
            result["success"] = True
            return index, result
        except Exception as e:
            return index, {
                "text": "",
                "success": False,
                "error": str(e)[:500],
                "tokens_generated": 0,
                "processing_time_seconds": 0,
            }

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
