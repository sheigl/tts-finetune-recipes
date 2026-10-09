# AGENTS.md — TTS Fine-Tuning Recipes Project

## Project Overview

This repository contains fine-tuning recipes for three TTS (Text-to-Speech) models:

| Recipe | Model | Framework | Key Files |
|--------|-------|-----------|-----------|
| `xttsv2-recipe/` | Coqui XTTS v2 | Coqui TTS library | `train.py`, `infer.py` |
| `kokoro-recipe/` | Kokoro (StyleTTS2-based) | Custom StyleTTS2 fork + kokoro-deutsch | `scripts/01_*` through `scripts/05_*`, `framework/StyleTTS2/` |
| `voxcpm-recipe/` | VoxCPM 1.5 with LoRA | VoxCPM library | `train.py`, `infer.py`, `config.yaml` |

### Kokoro Recipe Pipeline (5 steps)
1. `scripts/01_prepare_dataset.py` — CSV → train_list.txt + val_list.txt
2. `scripts/02_train.py` — Launches StyleTTS2 Stage 1 or Stage 2 training
3. `scripts/03_eval_all_epochs.py` — Evaluates all checkpoints, generates audio
4. `scripts/04_extract_voicepack.py` — Extracts speaker voicepack from checkpoint
5. `scripts/05_infer.py` — Final inference with converted Kokoro model

### Directory Structure
```
tts-finetune-recipes/
├── tts_utils/                    # Shared utilities (Step 1 ✅)
│   ├── __init__.py               # Exports get_device, empty_cache, etc.
│   ├── device.py                 # Device detection: CUDA > XPU > CPU
│   └── test_device.py            # Unit tests for device detection
├── xttsv2-recipe/
│   ├── train.py                  # Training via env vars + Coqui Trainer
│   └── infer.py                  # Inference script
├── kokoro-recipe/
│   ├── scripts/                  # Pipeline scripts (01 through 05)
│   │   ├── g2p_helper.py         # G2P text-to-phoneme conversion
│   │   └── ...
│   └── framework/StyleTTS2/      # Vendored StyleTTS2 training code
│       ├── train_first.py        # Stage 1 training (acoustic warmup)
│       ├── train_second.py       # Stage 2 training (joint prosody+acoustic)
│       ├── train_finetune.py     # Fine-tuning variant
│       ├── Modules/              # Model modules (hifigan, istftnet, diffusion)
│       └── Configs/              # YAML configs with device settings
├── voxcpm-recipe/
│   ├── train.py                  # Thin launcher wrapper
│   ├── train_voxcpm_finetune.py  # Actual training logic (vendored from VoxCPM)
│   ├── infer.py                  # Inference script
│   └── config.yaml               # Training configuration
└── docs/
    ├── design_step1_device_detection.md
    └── design_step2_device_placement.md
```

## Coding Standards

### Python Style
- **Type hints**: Use `from __future__ import annotations` at top of new files. Use type hints for function signatures where practical (e.g., `def foo(x: int, y: str) -> Path:`).
- **Imports**: Standard library first, then third-party, then local — each group separated by a blank line.
- **Docstrings**: Google-style docstrings for public functions and classes. Use triple double quotes (`"""`).
- **Line length**: 100 characters max (consistent with existing codebase).
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.

### Device Handling (XPU Migration)
All device detection MUST go through `tts_utils.device.get_device()`:

```python
from tts_utils.device import get_device, empty_cache, is_device_available

# Auto-detect with priority: CUDA > XPU > CPU
device = get_device("auto")       # or just get_device()

# Explicit request (falls back if unavailable)
device = get_device("xpu")        # tries xpu first, falls back to cuda/cpu

# Force specific device
device = get_device("cpu")        # always returns "cpu"

# Check availability before conditional logic
if is_device_available("cuda"):
    do_cuda_thing()

# Device-aware cache clearing
empty_cache(device)               # calls torch.cuda.empty_cache() or torch.xpu.empty_cache() as needed
```

**NEVER write inline device detection like this:**
```python
# BAD — do not do this:
device = "cuda" if torch.cuda.is_available() else "cpu"
device = args.device if torch.cuda.is_available() else "cpu"
```

### Argument Parser Pattern for `--device`
```python
ap.add_argument(
    "--device",
    default="auto",
    choices=["auto", "cuda", "xpu", "cpu"],
    help="Device to use. 'auto' detects best available (CUDA > XPU > CPU). Default: auto"
)

# Then resolve in code:
from tts_utils.device import get_device
device = get_device(args.device)
```

### YAML Config Files
Config files that specify `device` should be updated in a future step. For now, the Python scripts override config values via `get_device()`.

## Backend Code Patterns

### Device Detection Helper (tts_utils/device.py)
```python
def get_device(preferred: str | None = None) -> str:
    """Return best available device string. Priority: CUDA > XPU > CPU."""
    preferred = (preferred or "auto").lower().strip()
    
    if preferred == "cpu":
        return "cpu"
    
    if preferred in ("cuda", "xpu"):
        available = _cuda_available() if preferred == "cuda" else _xpu_available()
        if available:
            return preferred
        # Fall back with warning
    
    # Auto-detect chain
    if _cuda_available():
        return "cuda"
    if _xpu_available():
        return "xpu"
    return "cpu"
```

### Safe XPU Check (no ImportError)
```python
def _xpu_available() -> bool:
    """Check Intel XPU availability without requiring IPEX."""
    try:
        import torch
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    except (ImportError, RuntimeError):
        return False
```

### StyleTTS2 Framework Device Refactoring Patterns (Step 2)

Three patterns for replacing hardcoded `.to("cuda")` in the StyleTTS2 framework:

**Pattern A — Accelerator-managed scripts** (`train_first.py`, `train_first_style_only.py`, `train_finetune_accelerate.py`):
```python
# device is already set via accelerator.device
device = accelerator.device

# Replace .to("cuda") with .to(device) on intermediate tensors:
mask = length_to_mask(mel_input_length // (2**n_down)).to(device)  # was .to("cuda")
wav.append(torch.from_numpy(y).to(device))                         # was .to("cuda")
```

**Pattern B — Standalone training scripts** (`train_finetune.py`, `train_second.py`):
```python
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from tts_utils.device import get_device, empty_cache

@click.command()
@click.option('-p', '--config_path', default='Configs/config_ft.yml', type=str)
@click.option(
    '-d', '--device', default='auto',
    choices=['auto', 'cuda', 'xpu', 'cpu'],
    help="Device to use. 'auto' detects best available (CUDA > XPU > CPU). Default: auto"
)
def main(config_path, device):
    # Replace hardcoded device = "cuda":
    device = get_device(device)

    # Replace .to('cuda') with .to(device):
    mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)

    # Replace torch.cuda.empty_cache():
    empty_cache(device)
```

**Pattern C — Module forward() methods** (`Modules/hifigan.py`, `Modules/istftnet.py`):
```python
# Use input tensor's device for temporary tensors:
torch.ones(1, 1, F0_down, device=F0_curve.device)   # was .to("cuda")
torch.ones(1, 1, N_down, device=N.device)            # was .to("cuda")
```

## Testing Standards

### Unit Test Framework: `unittest`
The project uses Python's built-in `unittest` framework. Tests live alongside the code or in a dedicated test directory.

### Sample Test Pattern
```python
import unittest
from unittest.mock import patch


class TestGetDevice(unittest.TestCase):
    def test_auto_returns_cuda_when_available(self):
        with patch("tts_utils.device._cuda_available", return_value=True):
            from tts_utils.device import get_device
            self.assertEqual(get_device(), "cuda")

    def test_fallback_chain_xpu_then_cpu(self):
        with patch("tts_utils.device._cuda_available", return_value=False), \
             patch("tts_utils.device._xpu_available", return_value=True):
            from tts_utils.device import get_device
            self.assertEqual(get_device(), "xpu")

    def test_explicit_cpu_ignores_detection(self):
        with patch("tts_utils.device._cuda_available", return_value=True):
            from tts_utils.device import get_device
            # Even if CUDA is available, --device cpu should respect the request
            self.assertEqual(get_device("cpu"), "cpu")


if __name__ == "__main__":
    unittest.main()
```

### Test Execution
```bash
# Run all tests in a module
python -m unittest tts_utils.test_device

# Run with verbose output
python -m unittest -v tts_utils.test_device

# Run specific test class
python -m unittest tts_utils.test_device.TestGetDevice
```

## XPU Migration Status (Step-by-Step Plan)

### Step 1: Device Detection Helper ✅ (COMPLETED)
- [x] Create `tts_utils/device.py` with `get_device()`, `_cuda_available()`, `_xpu_available()`, `get_device_type()`, `empty_cache()`, `is_device_available()`
- [x] Create `tts_utils/__init__.py` — package-level exports
- [x] Create `tts_utils/test_device.py` — 12 unit tests (all passing)
- [x] Update `xttsv2-recipe/infer.py` — add `--device` arg, use `get_device()`, preserve `--cpu` for backwards compat
- [x] Update `kokoro/scripts/05_infer.py` — update choices to include `auto`/`xpu`, use `get_device()`
- [x] Update `kokoro/scripts/04_extract_voicepack.py` — update choices to include `xpu`, use `get_device()`
- [x] Update `kokoro/scripts/03_eval_all_epochs.py` — update choices, resolve device once in `main()`, pass to `_eval_one()`

### Step 2: Hardcoded `.to("cuda")` in StyleTTS2 Framework ✅ (COMPLETED)
Replaced all hardcoded `.to("cuda")` / `.to('cuda')` calls across training scripts and model modules with dynamic device references. See `docs/design_step2_device_placement.md` for the full plan.

**Files modified (7 total):**
- `Modules/hifigan.py`: 2× `.to("cuda")` → `device=F0_curve.device` / `device=N.device` in forward()
- `Modules/istftnet.py`: 2× `.to("cuda")` → `device=F0_curve.device` / `device=N.device` in forward()
- `train_first.py`: 3× `.to("cuda")` → `.to(device)` (accelerator-managed)
- `train_first_style_only.py`: 3× `.to("cuda")` → `.to(device)` (accelerator-managed)
- `train_finetune_accelerate.py`: 1× `.to('cuda')` → `.to(device)`, added `empty_cache` import, replaced `torch.cuda.empty_cache()`

**Refactoring patterns used:**
- Accelerator-managed scripts: `.to("cuda")` → `.to(device)` (using existing `device = accelerator.device`)
- Standalone training scripts: add `--device` CLI flag + `get_device(args.device)`
- Module `forward()`: use input tensor's device (`torch.ones(..., device=F0_curve.device)`)

### Step 3: `torch.cuda.empty_cache()` Replacement ✅ (COMPLETED — merged into Step 2)
5 locations replaced with `empty_cache(device)` from `tts_utils` while modifying the same files for Step 2.

**Files modified:**
- `train_finetune.py`: added `--device` CLI option, `get_device()`, `empty_cache()` import + call, `.to('cuda')` → `.to(device)`, replaced `torch.cuda.empty_cache()`
- `train_second.py`: added `--device` CLI option, `get_device()`, `empty_cache()` import + 3× calls, replaced all `torch.cuda.empty_cache()`
- `train_finetune_accelerate.py`: added `empty_cache` import, replaced `torch.cuda.empty_cache()`

### Step 4: YAML Config Device Values ✅ (COMPLETED)
Changed `device: "cuda"` to `device: "auto"` in all 3 config files to reflect actual runtime behavior.

**Files modified:**
- `Configs/config.yml`: line 5, `device: "cuda"` → `device: "auto"`
- `Configs/config_ft.yml`: line 4, `device: "cuda"` → `device: "auto"`
- `Configs/config_libritts.yml`: line 5, `device: "cuda"` → `device: "auto"`

### Step 5: `torch.autocast(device_type="cuda")` Replacement ✅ (COMPLETED)
Replaced hardcoded `torch.autocast(device_type="cuda", ...)` + `torch.cuda.is_available()` guard in the VoxCPM recipe with `accelerator.autocast(dtype=torch.bfloat16)`. The HuggingFace Accelerator handles device detection internally, consistent with 2 other autocast calls already in this file.

**Files modified:**
- `voxcpm-recipe/train_voxcpm_finetune.py`: removed `autocast_ctx` variable + CUDA check, replaced with `accelerator.autocast(dtype=torch.bfloat16)` context manager

## Environment Notes
- **CUDA systems**: NVIDIA GPU + CUDA toolkit → `torch.cuda.is_available()` returns True
- **XPU systems**: Intel Arc/iGPU + IPEX (Intel Extension for PyTorch) → `torch.xpu.is_available()` returns True
- **CPU-only**: No GPU → falls back to `"cpu"`
- **Dual GPU systems** (CUDA + XPU): CUDA is preferred by default; use `--device xpu` to override
