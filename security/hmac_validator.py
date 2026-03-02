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


# Shared secret — must match GPU_HMAC_SECRET on the VPS
HMAC_SECRET = os.environ.get("HMAC_SECRET", "")


def validate_hmac(payload: dict, signature: Optional[str]) -> bool:
    """
    Validate HMAC-SHA256 signature of a request payload.

    Args:
        payload: The request input dict (without hmac_signature field).
        signature: The HMAC signature from the request.

    Returns:
        True if signature is valid.
    """
    if not HMAC_SECRET:
        # No secret configured — skip validation (dev mode)
        return True

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
