"""
GPU info utilities — shared across operations.
"""

import subprocess


def get_gpu_name() -> str:
    """Get GPU device name via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            timeout=5,
            text=True,
        )
        return out.strip().split("\n")[0]
    except Exception:
        return "unknown"


def get_vram_total_mb() -> int:
    """Get total VRAM in MB via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            timeout=5,
            text=True,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return 0


def get_vram_free_mb() -> int:
    """Get free VRAM in MB via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            timeout=5,
            text=True,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return 0

def get_vram_free_mb() -> int:
    """Get free VRAM in MB via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            timeout=5,
            text=True,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return 0

