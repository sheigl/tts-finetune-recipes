# Design: Step 2 — Replace Hardcoded `.to("cuda")` in StyleTTS2 Framework

## Overview
Replace all hardcoded `.to("cuda")` / `.to('cuda')` calls and `torch.cuda.empty_cache()` invocations across the StyleTTS2 framework with dynamic device references, enabling XPU support alongside CUDA.

## Architecture Decisions

### Decision 1: Three distinct refactoring patterns for three code shapes
The hardcoded cuda references fall into three categories requiring different fixes:

| Category | Files | Pattern | Fix |
|----------|-------|---------|-----|
| **Accelerator-managed** | `train_first.py`, `train_first_style_only.py`, `train_finetune_accelerate.py` | `device = accelerator.device` already exists; `.to("cuda")` on intermediate tensors only | Replace `.to("cuda")` → `.to(device)` using existing variable |
| **Hardcoded device** | `train_finetune.py`, `train_second.py` | `device = "cuda"` hardcoded string | Change to `get_device(args.device)`, add `--device` CLI option |
| **Module forward()** | `Modules/hifigan.py`, `Modules/istftnet.py` | `.to("cuda")` on temporary `torch.ones()` inside `forward()` | Use input tensor's device: `device=F0_curve.device` |

### Decision 2: Merge Step 3 (empty_cache) into this step
The 5 `torch.cuda.empty_cache()` calls live in the same files being modified for `.to("cuda")`. Handling them together avoids a second round of edits on these files.

### Decision 3: No changes to Accelerate-managed scripts' CLI
`train_first.py`, `train_first_style_only.py`, and `train_finetune_accelerate.py` use HuggingFace Accelerate, which manages device assignment internally via the accelerator launcher. Adding a `--device` flag would conflict with Accelerate's device management. These scripts only need `.to("cuda")` → `.to(device)` replacements.

### Decision 4: Module forward() uses input tensor device, not constructor parameter
Rather than passing `self.device` into module constructors (which would require changes to all callers), we use the device of existing input tensors (`F0_curve.device`, `N.device`). This is simpler and works regardless of how the model was placed on a device.

## Trade-offs Considered

1. **Passing device through config YAML**: Rejected for now — Step 4 covers this. Using CLI args + `get_device()` is sufficient for code-level correctness.
2. **Using `torch.device("cuda")` → `torch.device(device)`**: Not applicable — these are `.to("cuda")` calls, not `torch.device()` constructors.
3. **Adding `--device` to Accelerate scripts**: Rejected — Accelerate manages devices; adding a flag would create confusion about which device is authoritative.

## Files to Create/Modify

### New Files
| File | Purpose | Key Responsibilities |
|------|---------|---------------------|
| `docs/design_step2_device_placement.md` | This design document | Records decisions and task breakdown |

### Modified Files — Accelerator-Managed Scripts (`.to("cuda")` → `.to(device)`)

| File | Changes | Reason |
|------|---------|--------|
| `kokoro-recipe/framework/StyleTTS2/train_first.py` | 3× `.to("cuda")` → `.to(device)` at lines ~252, ~451, ~505 | Use existing `device = accelerator.device` variable |
| `kokoro-recipe/framework/StyleTTS2/train_first_style_only.py` | 3× `.to("cuda")` → `.to(device)` at lines ~272, ~458, ~512 | Same pattern as train_first.py |
| `kokoro-recipe/framework/StyleTTS2/train_finetune_accelerate.py` | 1× `.to('cuda')` → `.to(device)` at line ~587; 1× `torch.cuda.empty_cache()` → `empty_cache(device)` at line ~226 | Use existing `device = accelerator.device`; add import for `empty_cache` |

### Modified Files — Hardcoded Device Scripts (add `--device` + use `get_device()`)

| File | Changes | Reason |
|------|---------|--------|
| `kokoro-recipe/framework/StyleTTS2/train_finetune.py` | Add `import os, sys` and `tts_utils` import; add `--device` click option; change `device = 'cuda'` → `get_device(args.device)`; 1× `.to('cuda')` → `.to(device)` at line ~580; 1× `torch.cuda.empty_cache()` → `empty_cache(device)` at line ~222 | Full device dynamic resolution |
| `kokoro-recipe/framework/StyleTTS2/train_second.py` | Same pattern as train_finetune.py; 3× `torch.cuda.empty_cache()` → `empty_cache(device)` at lines ~281, ~572, ~597 | Full device dynamic resolution |

### Modified Files — Module Forward Methods (input tensor device)

| File | Changes | Reason |
|------|---------|--------|
| `kokoro-recipe/framework/StyleTTS2/Modules/hifigan.py` | 2× `.to("cuda")` → use `device=F0_curve.device` / `device=N.device` in `torch.ones()` at lines ~613, ~622 | Dynamic device from input tensor |
| `kokoro-recipe/framework/StyleTTS2/Modules/istftnet.py` | 2× `.to("cuda")` → use `device=F0_curve.device` / `device=N.device` in `torch.ones()` at lines ~688, ~697 | Same pattern as hifigan.py |

## Task Breakdown (Ordered by Dependency)

### Task 1: Module Forward Methods — hifigan.py and istftnet.py
- **Files**: `Modules/hifigan.py`, `Modules/istftnet.py`
- **Description**: Replace `.to("cuda")` on temporary `torch.ones()` tensors inside `forward()` with the device of existing input tensors. These modules are already placed on the correct device by training scripts; intermediate tensors must match.
- **Acceptance Criteria**: No `.to("cuda")` or `.to('cuda')` strings remain in either file.

**hifigan.py changes (lines ~609–626):**
```python
# BEFORE:
F0_curve = (
    nn.functional.conv1d(
        F0_curve.unsqueeze(1),
        torch.ones(1, 1, F0_down).to("cuda"),
        padding=F0_down // 2,
    ).squeeze(1)
    / F0_down
)

# AFTER:
F0_curve = (
    nn.functional.conv1d(
        F0_curve.unsqueeze(1),
        torch.ones(1, 1, F0_down, device=F0_curve.device),
        padding=F0_down // 2,
    ).squeeze(1)
    / F0_down
)
```

Same pattern for `N` (replace `.to("cuda")` with `, device=N.device`).

**istftnet.py changes**: Identical pattern but using `.view()` instead of `.unsqueeze()`:
```python
# BEFORE:
torch.ones(1, 1, F0_down).to("cuda"),
# AFTER:
torch.ones(1, 1, F0_down, device=F0_curve.device),

# BEFORE:
torch.ones(1, 1, N_down).to("cuda"),
# AFTER:
torch.ones(1, 1, N_down, device=N.device),
```

### Task 2: Accelerator-Managed Scripts — `.to("cuda")` → `.to(device)`
- **Files**: `train_first.py`, `train_first_style_only.py`, `train_finetune_accelerate.py`
- **Description**: These scripts already have a `device` variable (`accelerator.device`). Replace all remaining `.to("cuda")` / `.to('cuda')` calls with `.to(device)`. For `train_finetune_accelerate.py`, also replace `torch.cuda.empty_cache()` and add the import.
- **Acceptance Criteria**: No `.to("cuda")` or `.to('cuda')` strings remain in any of these files.

**train_first.py changes (3 locations):**
```python
# Line ~252: mask tensor in training loop
mask = length_to_mask(mel_input_length // (2**n_down)).to(device)

# Line ~451: mask tensor in validation loop
mask = length_to_mask(mel_input_length // (2**n_down)).to(device)

# Line ~505: wav tensor from numpy
wav.append(torch.from_numpy(y).to(device))
```

**train_first_style_only.py changes**: Identical 3 locations at lines ~272, ~458, ~512.

**train_finetune_accelerate.py changes (2 locations):**
```python
# Add import after existing imports section:
from tts_utils.device import empty_cache

# Line ~226: replace torch.cuda.empty_cache()
empty_cache(device)

# Line ~587: mask tensor in validation loop
mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
```

**Import path for tts_utils**: From `kokoro-recipe/framework/StyleTTS2/`, the repo root is three levels up. Add near top of file:
```python
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from tts_utils.device import empty_cache
```

### Task 3: Hardcoded Device Scripts — `train_finetune.py` and `train_second.py`
- **Files**: `train_finetune.py`, `train_second.py`
- **Description**: These scripts hardcode `device = "cuda"` / `'cuda'`. Add `--device` CLI option, import `get_device` and `empty_cache`, resolve device dynamically, replace all `.to('cuda')` and `torch.cuda.empty_cache()` calls.
- **Acceptance Criteria**: No `.to("cuda")`, `.to('cuda')`, or `torch.cuda.empty_cache()` strings remain in either file.

**train_finetune.py changes:**

1. Add imports (after existing stdlib imports, before third-party):
```python
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from tts_utils.device import get_device, empty_cache
```

2. Add `--device` click option:
```python
@click.command()
@click.option('-p', '--config_path', default='Configs/config_ft.yml', type=str)
@click.option(
    '-d', '--device', default='auto',
    choices=['auto', 'cuda', 'xpu', 'cpu'],
    help="Device to use. 'auto' detects best available (CUDA > XPU > CPU). Default: auto"
)
def main(config_path, device):
```

3. Replace hardcoded device assignment (~line 91):
```python
# BEFORE:
device = 'cuda'

# AFTER:
device = get_device(device)
```

4. Replace `.to('cuda')` in validation loop (~line 580):
```python
mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
```

5. Replace `torch.cuda.empty_cache()` (~line 222):
```python
empty_cache(device)
```

**train_second.py changes**: Same pattern as train_finetune.py, plus:
- 3× `torch.cuda.empty_cache()` at lines ~281, ~572, ~597 → `empty_cache(device)`
- No `.to("cuda")` calls in this file (only hardcoded device string and empty_cache)

### Task 4: Verification — Grep for Remaining Hardcoded CUDA References
- **Files**: All modified files from Tasks 1–3
- **Description**: Run grep to confirm no `.to("cuda")`, `.to('cuda')`, or `torch.cuda.empty_cache()` remain in the modified files.
- **Acceptance Criteria**: Zero matches for these patterns in the 7 modified files.

```bash
grep -rn '\.to("cuda")\|\.to('"'"'cuda'"'"')\|torch\.cuda\.empty_cache()' \
  kokoro-recipe/framework/StyleTTS2/train_first.py \
  kokoro-recipe/framework/StyleTTS2/train_first_style_only.py \
  kokoro-recipe/framework/StyleTTS2/train_finetune.py \
  kokoro-recipe/framework/StyleTTS2/train_finetune_accelerate.py \
  kokoro-recipe/framework/StyleTTS2/train_second.py \
  kokoro-recipe/framework/StyleTTS2/Modules/hifigan.py \
  kokoro-recipe/framework/StyleTTS2/Modules/istftnet.py
```

## Data Models / Interfaces

No new data models or interfaces. This is a refactoring of existing code paths.

The `tts_utils.device` interface used:
```python
from tts_utils.device import get_device, empty_cache

device = get_device("auto")  # returns "cuda", "xpu", or "cpu"
empty_cache(device)          # calls torch.cuda.empty_cache() or torch.xpu.empty_cache() as needed
```

## Testing Strategy

### Unit Tests (in `tts_utils/test_device.py`)
- Existing tests cover `get_device()` and `empty_cache()` — no new unit tests needed for the helper functions.

### Static Analysis Test
Add a test in `kokoro-recipe/framework/StyleTTS2/` that verifies no hardcoded `.to("cuda")` or `torch.cuda.empty_cache()` remains:

```python
import unittest
import os
import re


class TestNoHardcodedCuda(unittest.TestCase):
    """Verify Step 2 refactoring removed all hardcoded cuda references."""

    FORBIDDEN_PATTERNS = [
        r'\.to\("cuda"\)',
        r"\.to\('cuda'\)",
        r'torch\.cuda\.empty_cache\(\)',
    ]

    FILES_TO_CHECK = [
        'train_first.py',
        'train_first_style_only.py',
        'train_finetune.py',
        'train_finetune_accelerate.py',
        'train_second.py',
        'Modules/hifigan.py',
        'Modules/istftnet.py',
    ]

    def test_no_hardcoded_cuda_in_training_scripts(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        for filename in self.FILES_TO_CHECK:
            filepath = os.path.join(base_dir, filename)
            if not os.path.exists(filepath):
                continue  # skip if file doesn't exist yet
            with open(filepath, 'r') as f:
                content = f.read()
            for pattern in self.FORBIDDEN_PATTERNS:
                matches = re.findall(pattern, content)
                self.assertEqual(
                    len(matches), 0,
                    f"Found hardcoded cuda reference '{pattern}' in {filename}"
                )


if __name__ == "__main__":
    unittest.main()
```

### Integration Tests (manual / CI)
- Run `python train_finetune.py --help` to verify `--device` option appears.
- Run `python train_second.py --help` to verify `--device` option appears.
- On a CUDA system: run a single epoch of training with default args — should use CUDA as before.
- On an XPU system: run with `--device xpu` — should work without errors.

## Potential Risks

1. **Risk**: Accelerate's device string format may differ from `"cuda"`/`"xpu"` (e.g., `"cuda:0"`).
   → **Mitigation**: `.to(device)` works with both `"cuda"` and `"cuda:0"` strings in PyTorch. The `accelerator.device` returns the correct format for the environment.

2. **Risk**: `sys.path.insert()` may cause import conflicts if PYTHONPATH is already set differently.
   → **Mitigation**: Use `insert(0, ...)` which prepends to sys.path. If `tts_utils` is already on PYTHONPATH (common in CI), this is a no-op effect.

3. **Risk**: Module forward() method uses input tensor device, but if the model hasn't been `.to(device)` yet, inputs might be on CPU.
   → **Mitigation**: Training scripts always call `model[key].to(device)` before entering training loops. The forward() calls happen after placement, so input tensors are already on the correct device.

4. **Risk**: `train_finetune.py` and `train_second.py` use DataParallel (`MyDataParallel`) which expects all replicas on CUDA.
   → **Mitigation**: DataParallel works with any device string; it replicates to available devices of the same type. If only one XPU is available, it still works (single-device DP).

5. **Risk**: `torch.ones(1, 1, F0_down, device=F0_curve.device)` may fail if `F0_curve` is on a device that doesn't support `torch.ones()`.
   → **Mitigation**: All PyTorch devices (cuda, xpu, cpu) support `torch.ones()` with the `device=` keyword argument.
