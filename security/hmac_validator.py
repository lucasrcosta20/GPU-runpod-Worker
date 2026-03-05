"""
HMAC-SHA256 validation for incoming requests.

Validates that requests are signed by the ia-cadastro VPS
using a shared secret. Protects against replay and key compromise.
"""

import hashlib
import hmac
import json
import os
from typing import Optional


# Shared secret — injected via RunPod environment variables, never hardcoded
HMAC_SECRET = os.environ.get("HMAC_SECRET", "")

# Explicit opt-in for dev mode (must set HMAC_DEV_BYPASS=1)
HMAC_DEV_BYPASS = os.environ.get("HMAC_DEV_BYPASS", "") == "1"


def validate_hmac(payload: dict, signature: Optional[str]) -> bool:
    """
    Validate HMAC-SHA256 signature of a request payload.

    Args:
        payload: The request input dict (without hmac_signature field).
        signature: The HMAC signature from the request.

    Returns:
        True if signature is valid.

    Raises:
        RuntimeError: If HMAC_SECRET is not configured and dev bypass is off.
    """
    if not HMAC_SECRET:
        if HMAC_DEV_BYPASS:
            return True
        raise RuntimeError(
            "HMAC_SECRET not configured. Set HMAC_SECRET env var in RunPod "
            "or HMAC_DEV_BYPASS=1 for local development."
        )

    if not signature:
        return False

    # Build canonical payload (exclude hmac_signature itself)
    canonical = {k: v for k, v in payload.items() if k != "hmac_signature"}
    canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))

    expected = hmac.new(
        HMAC_SECRET.encode("utf-8"),
        canonical_json.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(expected, signature)
