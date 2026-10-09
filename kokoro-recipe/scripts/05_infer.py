#!/usr/bin/env python3
"""
Step 5: Inference with a Fine-Tuned Kokoro Model
=================================================
Synthesize speech from text using a trained checkpoint + its voicepack.

Usage:
    python scripts/05_infer.py \
        --voicepack eval/epoch08/voicepack.pt \
        --checkpoint eval/epoch08/kokoro_converted.pth \
        --config /path/to/kokoro-deutsch/training/config.json \
        --text "Hello, this is my custom voice." \
        --out  output.wav

    # Interactive mode (prompts for text):
    python scripts/05_infer.py \
        --voicepack eval/epoch08/voicepack.pt \
        --checkpoint eval/epoch08/kokoro_converted.pth

Note: --checkpoint expects the *converted* Kokoro-format .pth written by
      03_eval_all_epochs.py, NOT the raw StyleTTS2 checkpoint.
      If you only have the raw checkpoint, use 04_extract_voicepack.py first,
      then run 03_eval_all_epochs.py to get the converted weights.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voicepack",  required=True,
                    help="Voicepack .pt file (from eval/epochNN/voicepack.pt)")
    ap.add_argument("--checkpoint", required=True,
                    help="Converted Kokoro weights .pth (from eval/epochNN/kokoro_converted.pth)")
    ap.add_argument("--config",     default=None,
                    help="Kokoro config.json (auto-detected from kokoro-deutsch if omitted)")
    ap.add_argument("--text",       default=None,
                    help="Text to synthesize. Omit to enter interactive mode.")
    ap.add_argument("--out",        default="output.wav", help="Output WAV path")
    ap.add_argument("--speed",      type=float, default=1.0, help="Speaking speed (default: 1.0)")
    ap.add_argument("--device",     default="auto", choices=["auto", "cuda", "xpu", "cpu"],
                     help="Device to use. 'auto' detects best available (CUDA > XPU > CPU). Default: auto")
    ap.add_argument("--recipe-root", default=None,
                    help="Path to the kokoro-recipe directory (auto-detected if omitted)")
    args = ap.parse_args()

    recipe_root = Path(args.recipe_root) if args.recipe_root else Path(__file__).parent.parent
    config_path = _find_config(args.config, recipe_root)
    vp_path     = Path(args.voicepack)
    ckpt_path   = Path(args.checkpoint)

    for p, label in [(vp_path, "voicepack"), (ckpt_path, "checkpoint"), (config_path, "config.json")]:
        if not p.exists():
            print(f"[ERROR] {label} not found: {p}", file=sys.stderr)
            sys.exit(1)

    # Add kokoro-deutsch venv to path
    _setup_kokoro_path(recipe_root)

    import torch
    import numpy as np
    import soundfile as sf
    from tts_utils.device import get_device
    from kokoro import KModel, KPipeline

    scripts_dir = Path(__file__).parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from g2p_helper import g2p_with_lexicon, normalize_text

    # Load config to determine G2P language
    import yaml
    cfg_yml = recipe_root / "configs" / "config.yml"
    g2p_lang = "en-gb"
    if cfg_yml.exists():
        c = yaml.safe_load(cfg_yml.read_text())
        g2p_lang = c.get("g2p", {}).get("language", "en-gb")
    lang_code = "b" if "gb" in g2p_lang else "a"

    device = get_device(args.device)

    kmodel = KModel(repo_id="hexgrad/Kokoro-82M", config=str(config_path), model=str(ckpt_path))
    kmodel = kmodel.to(device).eval()
    pipeline = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M", model=kmodel)
    voice = torch.load(str(vp_path), map_location="cpu", weights_only=True)

    def synthesize(text: str, out_path: str):
        norm = normalize_text(text)
        ipa  = g2p_with_lexicon(norm, language=g2p_lang)
        print(f"Text : {text}")
        print(f"IPA  : {ipa[:80]}")
        gen   = pipeline.generate_from_tokens(ipa, voice=voice, speed=args.speed)
        parts = [audio for _, _, audio in gen]
        if not parts:
            print("[WARN] No audio generated. Check text and voicepack.")
            return
        combined = np.concatenate(parts)
        sf.write(out_path, combined, 24000)
        print(f"Saved: {out_path}  ({len(combined)/24000:.1f}s)")

    if args.text:
        synthesize(args.text, args.out)
    else:
        print("Interactive mode — type text and press Enter. Ctrl+C to exit.")
        i = 1
        while True:
            try:
                text = input(f"\n[{i}] Text> ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                break
            if not text:
                continue
            out_path = Path(args.out)
            if i > 1:
                out_path = out_path.with_name(f"{out_path.stem}_{i}{out_path.suffix}")
            synthesize(text, str(out_path))
            i += 1


def _find_config(config_arg: str | None, recipe_root: Path) -> Path:
    if config_arg:
        return Path(config_arg)
    import yaml
    cfg_yml = recipe_root / "configs" / "config.yml"
    if cfg_yml.exists():
        c = yaml.safe_load(cfg_yml.read_text())
        fw = c.get("framework", {})

        def resolve(raw: str, fallback: Path) -> Path:
            p = Path(raw) if raw else None
            if p:
                return p if p.is_absolute() else (recipe_root / p).resolve()
            return fallback

        training_dir = resolve(fw.get("training_dir", ""), recipe_root / "framework" / "training")
        # Legacy fallback
        if not training_dir.exists():
            kd = Path(fw.get("kokoro_deutsch_dir", ""))
            if kd.exists():
                training_dir = kd / "training"
        p = training_dir / "config.json"
        if p.exists():
            return p
    print("[ERROR] Cannot find Kokoro config.json. Pass --config explicitly.")
    sys.exit(1)


def _setup_kokoro_path(recipe_root: Path):
    # Most installs don't need this — kokoro is in the normal pip env.
    # Only needed if using a separate venv (legacy bundled-venv setup).
    pass


if __name__ == "__main__":
    main()
