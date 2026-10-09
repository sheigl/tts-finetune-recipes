"""
Cross-device detection utilities for TTS fine-tuning recipes.

Supports CUDA (NVIDIA), XPU (Intel Arc/iGPU), and CPU with automatic
fallback chain: CUDA > XPU > CPU.
"""
from __future__ import annotations


def _cuda_available() -> bool:
    """Check if NVIDIA CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except (ImportError, RuntimeError):
        return False


def _xpu_available() -> bool:
    """Check if Intel XPU is available.

    Note: torch.xpu may not exist in standard PyTorch builds.
    It requires the Intel Extension for PyTorch (IPEX).
    """
    try:
        import torch
        # torch.xpu only exists when IPEX is installed
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    except (ImportError, RuntimeError):
        return False


def get_device(preferred: str | None = None) -> str:
    """Return the best available device string.

    Priority chain when preferred is None or "auto":
      1. "cuda" — if NVIDIA CUDA is available
      2. "xpu"  — if Intel XPU (IPEX) is available
      3. "cpu"  — always available fallback

    When preferred is a specific device ("cuda", "xpu", "cpu"):
      - If that device is available, return it.
      - Otherwise, fall back through the chain and warn.

    Args:
        preferred: Device to prefer. One of {"auto", "cuda", "xpu", "cpu"}.
                   Defaults to "auto".

    Returns:
        A PyTorch device string: "cuda", "xpu", or "cpu".

    Examples:
        >>> get_device()                    # auto-detect, prefers CUDA
        'cuda'
        >>> get_device("xpu")              # force XPU if available
        'xpu'
        >>> get_device("auto")             # same as default
        'cuda'
    """
    import sys

    preferred = (preferred or "auto").lower().strip()

    if preferred == "cpu":
        return "cpu"

    if preferred in ("cuda", "xpu"):
        available = _cuda_available() if preferred == "cuda" else _xpu_available()
        if available:
            return preferred
        # Requested device not available — warn and fall back
        print(
            f"[WARN] Requested device '{preferred}' is not available. "
            f"Falling back to auto-detection.",
            file=sys.stderr,
        )

    # Auto-detect chain: CUDA > XPU > CPU
    if _cuda_available():
        return "cuda"
    if _xpu_available():
        return "xpu"
    return "cpu"


def get_device_type(device_str: str) -> str:
    """Return the device type string for use with torch.autocast().

    Maps device strings to their autocast-compatible device_type values.

    Args:
        device_str: A device string like "cuda", "xpu", or "cpu".

    Returns:
        The corresponding device_type string for torch.autocast(device_type=...).

    Examples:
        >>> get_device_type("cuda")
        'cuda'
        >>> get_device_type("xpu")
        'xpu'
        >>> get_device_type("cpu")
        'cpu'
    """
    # For "cuda:0", "xpu:1" etc., strip the index
    base = device_str.split(":")[0].lower()
    return base if base in ("cuda", "xpu", "cpu") else "cpu"


def empty_cache(device_str: str) -> None:
    """Clear the cache for the given device type.

    Args:
        device_str: A device string like "cuda", "xpu", or "cpu".
    """
    import torch
    base = get_device_type(device_str)
    if base == "cuda":
        torch.cuda.empty_cache()
    elif base == "xpu":
        try:
            torch.xpu.empty_cache()
        except AttributeError:
            pass  # IPEX version may not expose this
    # CPU has no cache to clear


def is_device_available(device_str: str) -> bool:
    """Check if a specific device type is available.

    Args:
        device_str: A device string like "cuda", "xpu", or "cpu".

    Returns:
        True if the device type is available on this system.
    """
    base = get_device_type(device_str)
    if base == "cuda":
        return _cuda_available()
    if base == "xpu":
        return _xpu_available()
    return True  # CPU is always available
