# Design: Step 1 — XPU Device Detection Helper

## Overview
Add Intel XPU (Intel Arc / iGPU) support to device detection across all TTS recipe scripts, establishing a CUDA > XPU > CPU priority chain that preserves existing behavior while enabling opt-in XPU usage.

## Architecture Decisions

### Decision 1: Top-level shared utility package (`tts_utils/`)
**Chosen**: Create a new top-level `tts_utils/` package at the repository root with a single `device.py` module containing `_get_device()`.

**Trade-offs considered**:
- **Inline per-file**: Simplest, but duplicates logic across 4+ files. Any bug fix or enhancement (e.g., adding MPS support later) requires touching every file. Rejected for maintainability.
- **Shared utility under one recipe** (e.g., `kokoro-recipe/utils/`): Creates a cross-recipe dependency that feels wrong — why would xttsv2 depend on kokoro's utils? Rejected for architectural cleanliness.
- **Top-level `tts_utils/` package**: Clean separation, no cross-recipe coupling, easy to import from any recipe (`from tts_utils.device import get_device`). This is the chosen approach.

### Decision 2: Function signature — `get_device(preferred=None)`
The function takes an optional `preferred` string argument and returns a device string. If `preferred` is `"auto"`, it runs the detection chain (CUDA > XPU > CPU). If `preferred` is a specific device like `"cuda"` or `"xpu"`, it validates availability and falls back gracefully.

### Decision 3: Backwards compatibility — CUDA remains default
The detection order is strictly **CUDA → XPU → CPU**. Existing scripts that don't specify `--device` get the same behavior as before (CUDA if available, else CPU). XPU users must explicitly request it via `--device xpu`.

### Decision 4: Argument parser pattern — add `"xpu"` and `"auto"` to choices
All `--device` arguments gain two new options:
- `"auto"` (new default for scripts that currently default to `"cuda"`) — runs the detection chain
- `"xpu"` — explicitly request XPU

The `xttsv2-recipe/infer.py` script uses a `--cpu` flag instead of `--device`. We add `--device` as an alternative and deprecate `--cpu` (keeping it for backwards compatibility).

## Files to Create/Modify

### New Files
| File | Purpose | Key Responsibilities |
|------|---------|---------------------|
| `tts_utils/__init__.py` | Package init | Exposes `get_device` at package level |
| `tts_utils/device.py` | Device detection module | Contains `get_device()`, `_cuda_available()`, `_xpu_available()` |

### Modified Files (Step 1 scope)
| File | Changes | Reason |
|------|---------|--------|
| `xttsv2-recipe/infer.py` | Replace inline device logic with `get_device()`. Add `--device` arg alongside existing `--cpu` flag. | Unified device detection, XPU support |
| `kokoro-recipe/scripts/05_infer.py` | Replace inline check with `get_device()`. Update `--device` choices to include `"xpu"` and `"auto"`. Change default from `"cuda"` to `"auto"`. | Unified device detection, XPU support |
| `kokoro-recipe/scripts/04_extract_voicepack.py` | Replace inline check in `extract_voicepack()` with `get_device()`. Update `--device` choices to include `"xpu"`. | Unified device detection, XPU support |
| `kokoro-recipe/scripts/03_eval_all_epochs.py` | Replace hardcoded default with `get_device()`. Update `--device` choices to include `"xpu"` and `"auto"`. Change default from `"cuda"` to `"auto"`. | Unified device detection, XPU support |

### Files NOT modified in Step 1 (future steps)
These contain `.to("cuda")`, `torch.cuda.empty_cache()`, or hardcoded `device = "cuda"` patterns that are out of scope for the device detection helper:
- `kokoro-recipe/framework/StyleTTS2/train_second.py` — hardcoded `device = "cuda"`, 3x `empty_cache()`
- `kokoro-recipe/framework/StyleTTS2/train_first.py` — 2x `.to("cuda")`
- `kokoro-recipe/framework/StyleTTS2/train_first_style_only.py` — 3x `.to("cuda")`
- `kokoro-recipe/framework/StyleTTS2/train_finetune.py` — `.to('cuda')`, `empty_cache()`
- `kokoro-recipe/framework/StyleTTS2/train_finetune_accelerate.py` — `.to('cuda')`, `empty_cache()`
- `kokoro-recipe/framework/StyleTTS2/Modules/hifigan.py` — 2x `.to("cuda")`
- `kokoro-recipe/framework/StyleTTS2/Modules/istftnet.py` — 2x `.to("cuda")`
- `voxcpm-recipe/train_voxcpm_finetune.py` — `torch.autocast(device_type="cuda")` + `is_available()` check
- `kokoro-recipe/framework/StyleTTS2/Configs/*.yml` — hardcoded `device: "cuda"`

## Data Models / Interfaces

### `tts_utils/device.py`

```python
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
```

### `tts_utils/__init__.py`

```python
"""Shared utilities for TTS fine-tuning recipes."""
from .device import (
    empty_cache,
    get_device,
    get_device_type,
    is_device_available,
)

__all__ = [
    "empty_cache",
    "get_device",
    "get_device_type",
    "is_device_available",
]
```

## Task Breakdown (Ordered by Dependency)

### Task 1: Create `tts_utils/` package
- **Files**: 
  - CREATE `tts_utils/__init__.py`
  - CREATE `tts_utils/device.py`
- **Description**: Create the shared device detection utility module with all helper functions. The module must handle missing IPEX gracefully (no ImportError when `torch.xpu` doesn't exist).
- **Acceptance Criteria**:
  - `python -c "from tts_utils.device import get_device; print(get_device())"` runs without error on any system
  - Returns `"cuda"` on CUDA systems, `"cpu"` on CPU-only systems
  - Does NOT raise ImportError when IPEX is not installed
  - `get_device("xpu")` returns `"xpu"` on XPU systems, falls back with warning otherwise

### Task 2: Update `xttsv2-recipe/infer.py`
- **Files**: MODIFY `xttsv2-recipe/infer.py`
- **Description**: 
  - Add `from tts_utils.device import get_device` import at top of file (after existing imports)
  - Replace line 64: `device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"` with `device = get_device("cpu") if args.cpu else get_device(args.device)`
  - Add `--device` argument to the parser alongside the existing `--cpu` flag. The new arg should have: `default="auto", choices=["auto", "cuda", "xpu", "cpu"]`. Mark `--cpu` as still supported but note in help text that `--device cpu` is equivalent.
- **Acceptance Criteria**:
  - Existing behavior preserved: no flags → CUDA if available, else CPU
  - `--cpu` flag still works (returns `"cpu"`)
  - `--device xpu` returns `"xpu"` on XPU systems
  - `--device auto` runs detection chain

### Task 3: Update `kokoro-recipe/scripts/05_infer.py`
- **Files**: MODIFY `kokoro-recipe/scripts/05_infer.py`
- **Description**:
  - Add `from tts_utils.device import get_device` import (inside the function, after existing imports at line 61)
  - Replace line 80: `device = args.device if torch.cuda.is_available() else "cpu"` with `device = get_device(args.device)`
  - Update line 43 argument parser: change `choices=["cuda", "cpu"]` to `choices=["auto", "cuda", "xpu", "cpu"]`, change `default="cuda"` to `default="auto"`
- **Acceptance Criteria**:
  - Default behavior unchanged (auto-detects CUDA if available)
  - `--device xpu` works on XPU systems
  - `--device cpu` forces CPU

### Task 4: Update `kokoro-recipe/scripts/04_extract_voicepack.py`
- **Files**: MODIFY `kokoro-recipe/scripts/04_extract_voicepack.py`
- **Description**:
  - Add `from tts_utils.device import get_device` import at top of file (after existing imports, around line 37)
  - Replace lines 150-151 in `extract_voicepack()`: 
    ```python
    # Before:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # After:
    device = get_device(device)
    ```
  - Update line 253 argument parser: change `choices=["auto", "cuda", "cpu"]` to `choices=["auto", "cuda", "xpu", "cpu"]`
- **Acceptance Criteria**:
  - Default `"auto"` behavior unchanged (CUDA > CPU → now CUDA > XPU > CPU)
  - `--device xpu` works on XPU systems

### Task 5: Update `kokoro-recipe/scripts/03_eval_all_epochs.py`
- **Files**: MODIFY `kokoro-recipe/scripts/03_eval_all_epochs.py`
- **Description**:
  - Add `from tts_utils.device import get_device` import (inside `_eval_one()` function, after existing imports at line 141)
  - Update line 55 argument parser: change `choices=["cuda", "cpu"]` to `choices=["auto", "cuda", "xpu", "cpu"]`, change `default="cuda"` to `default="auto"`
  - In `_eval_one()`, the device string is passed through from args. Since `get_device()` resolves `"auto"` at call time, add a resolution step: in `main()`, after parsing args, resolve the device early: `resolved_device = get_device(args.device)` and pass that to `_eval_one()`. This ensures consistent device across all epoch evaluations.
- **Acceptance Criteria**:
  - Default behavior unchanged (auto-detects CUDA if available)
  - Device is resolved once in main() and passed consistently through the evaluation loop
  - `--device xpu` works on XPU systems

## Testing Strategy

### Unit Tests for `tts_utils/device.py`
```python
# tts_utils/test_device.py (or tests/tts_utils/test_device.py)
import unittest
from unittest.mock import patch, MagicMock


class TestGetDevice(unittest.TestCase):
    """Test device detection logic."""

    def test_auto_returns_cuda_when_available(self):
        with patch("tts_utils.device._cuda_available", return_value=True):
            from tts_utils.device import get_device
            self.assertEqual(get_device(), "cuda")
            self.assertEqual(get_device("auto"), "cuda")

    def test_auto_falls_back_to_xpu_when_no_cuda(self):
        with patch("tts_utils.device._cuda_available", return_value=False), \
             patch("tts_utils.device._xpu_available", return_value=True):
            from tts_utils.device import get_device
            self.assertEqual(get_device(), "xpu")

    def test_auto_falls_back_to_cpu_when_nothing(self):
        with patch("tts_utils.device._cuda_available", return_value=False), \
             patch("tts_utils.device._xpu_available", return_value=False):
            from tts_utils.device import get_device
            self.assertEqual(get_device(), "cpu")

    def test_explicit_cpu_always_returns_cpu(self):
        with patch("tts_utils.device._cuda_available", return_value=True):
            from tts_utils.device import get_device
            self.assertEqual(get_device("cpu"), "cpu")

    def test_explicit_cuda_warns_and_falls_back_if_unavailable(self):
        with patch("tts_utils.device._cuda_available", return_value=False), \
             patch("tts_utils.device._xpu_available", return_value=True):
            from tts_utils.device import get_device
            # Should warn to stderr and fall back to xpu
            self.assertEqual(get_device("cuda"), "xpu")

    def test_none_and_empty_string_treated_as_auto(self):
        with patch("tts_utils.device._cuda_available", return_value=True):
            from tts_utils.device import get_device
            self.assertEqual(get_device(None), "cuda")
            self.assertEqual(get_device(""), "cuda")


class TestGetDeviceType(unittest.TestCase):
    def test_strips_index(self):
        from tts_utils.device import get_device_type
        self.assertEqual(get_device_type("cuda:0"), "cuda")
        self.assertEqual(get_device_type("xpu:1"), "xpu")

    def test_returns_base_for_simple_strings(self):
        from tts_utils.device import get_device_type
        self.assertEqual(get_device_type("cpu"), "cpu")


class TestEmptyCache(unittest.TestCase):
    def test_no_error_on_cpu(self):
        """Should not raise on CPU device."""
        from tts_utils.device import empty_cache
        # Should complete without error
        empty_cache("cpu")

    @patch("torch.cuda.empty_cache")
    def test_calls_cuda_empty_cache_for_cuda(self, mock_empty):
        from tts_utils.device import empty_cache
        empty_cache("cuda")
        mock_empty.assert_called_once()


class TestIsDeviceAvailable(unittest.TestCase):
    def test_cpu_always_available(self):
        from tts_utils.device import is_device_available
        self.assertTrue(is_device_available("cpu"))

    @patch("tts_utils.device._cuda_available", return_value=True)
    def test_cuda_reflects_availability(self, mock_cuda):
        from tts_utils.device import is_device_available
        self.assertTrue(is_device_available("cuda"))


if __name__ == "__main__":
    unittest.main()
```

### Integration Tests (manual verification per script)
For each modified script, verify:
1. `python <script> --help` shows updated choices including `"xpu"` and `"auto"`
2. Script runs without ImportError when IPEX is not installed
3. On CUDA systems, default behavior selects `"cuda"`

## Potential Risks

### Risk 1: Import error on systems without IPEX
**Problem**: `import torch.xpu` would fail if Intel Extension for PyTorch isn't installed.
**Mitigation**: `_xpu_available()` uses `hasattr(torch, "xpu")` check and wraps in try/except. The function never imports `torch.xpu` directly — it only accesses the attribute on the `torch` module itself.

### Risk 2: Cross-recipe import path issues
**Problem**: Scripts run from different working directories may fail to find `tts_utils`.
**Mitigation**: The `tts_utils/` package lives at the repository root. Users should either:
- Run scripts with the repo root in their PYTHONPATH, or
- Install the repo in development mode (`pip install -e .`) — but this requires a `pyproject.toml`, which is a future enhancement.

For Step 1, we document that users need to ensure the repo root is importable (e.g., by running from within the repo directory). This is consistent with how the kokoro scripts already use relative imports like `from g2p_helper import ...`.

### Risk 3: Breaking existing `--device cuda` explicit usage
**Problem**: Users who explicitly pass `--device cuda` on XPU-only systems will get a warning and fallback.
**Mitigation**: The warning is informative, not an error. The script continues with the best available device. This is the correct behavior — it's better to warn and work than to crash.

### Risk 4: CUDA preference may be undesirable for some users
**Problem**: A system with both CUDA and XPU might want to use XPU (e.g., to save CUDA memory).
**Mitigation**: `--device xpu` explicitly overrides the auto-detection chain. Users who want XPU can request it directly.

## Handoff to Developer

**Design Document**: Above
**Estimated Complexity**: Low — 2 new files, 4 file modifications, all changes are localized and non-breaking
**Key Files**: 
1. `tts_utils/device.py` (new) — core logic
2. `xttsv2-recipe/infer.py` — modify device detection + add `--device` arg
3. `kokoro-recipe/scripts/05_infer.py` — modify device detection + update choices
4. `kokoro-recipe/scripts/04_extract_voicepack.py` — modify device detection + update choices

**Start With**: Task 1 (create `tts_utils/device.py`) — all other tasks depend on it. Write the unit tests alongside the implementation to verify correctness before touching any recipe scripts.
