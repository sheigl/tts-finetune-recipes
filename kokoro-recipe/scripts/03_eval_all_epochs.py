#!/usr/bin/env python3
"""
Step 3: Evaluate All Stage 2 Checkpoints
==========================================
For each epoch_2nd_NNNNN.pth in the log directory:
  1. Extracts a voicepack (style_encoder from first_stage.pth + predictor_encoder
     from that epoch's checkpoint)
  2. Synthesizes all sentences in training/eval_texts.txt
  3. Saves audio to eval/epochNN/{01.wav, 02.wav, ...}

Listen to the outputs and pick your best epoch. See README.md for selection tips.

Usage:
    python scripts/03_eval_all_epochs.py --config configs/config.yml
    python scripts/03_eval_all_epochs.py --config configs/config.yml --only 06 08 09
    python scripts/03_eval_all_epochs.py --config configs/config.yml --device cpu
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import yaml

from tts_utils.device import get_device


def load_config(config_path: str) -> dict:
    return yaml.safe_load(Path(config_path).read_text())


def find_styletts2_and_training(cfg: dict, recipe_root: Path) -> tuple[Path, Path]:
    fw = cfg.get("framework", {})

    def resolve(raw: str, fallback: Path) -> Path:
        p = Path(raw) if raw else None
        if p:
            return p if p.is_absolute() else (recipe_root / p).resolve()
        return fallback

    styletts2_dir = resolve(fw.get("styletts2_dir", ""), recipe_root / "framework" / "StyleTTS2")
    training_dir  = resolve(fw.get("training_dir", ""),  recipe_root / "framework" / "training")

    # Legacy fallback
    if not styletts2_dir.exists():
        kd = Path(fw.get("kokoro_deutsch_dir", ""))
        if kd.exists():
            styletts2_dir = kd / "StyleTTS2"
            training_dir  = kd / "training"

    return styletts2_dir, training_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/config.yml")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "xpu", "cpu"],
                     help="Device to use. 'auto' detects best available (CUDA > XPU > CPU). Default: auto")
    ap.add_argument("--only", nargs="+", default=None,
                    help="Only evaluate these epoch numbers (e.g. --only 06 08 09)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip epochs that already have eval audio")
    args = ap.parse_args()

    # Resolve device once so all epochs use the same device
    resolved_device = get_device(args.device)

    cfg = load_config(args.config)
    recipe_root = Path(args.config).parent.parent.resolve()
    styletts2_dir, training_dir = find_styletts2_and_training(cfg, recipe_root)

    model_cfg = cfg["model"]
    log_dir  = (recipe_root / model_cfg.get("log_dir", "output/kokoro-finetune")).resolve()
    run_name = model_cfg.get("run_name", "kokoro-custom-v1")
    ckpt_dir = log_dir / run_name

    first_stage = ckpt_dir / "first_stage.pth"
    if not first_stage.exists():
        first_stage = ckpt_dir / "epoch_1st_00001.pth"
    if not first_stage.exists():
        print(f"[ERROR] first_stage.pth not found in {ckpt_dir}")
        print("  Did Stage 1 complete? Check logs/stage1.log")
        sys.exit(1)

    ckpts = sorted(
        c for c in ckpt_dir.glob("epoch_2nd_[0-9]*.pth")
        if c.stem.split("_")[-1].isdigit()
    )
    if not ckpts:
        print(f"[ERROR] No epoch_2nd_*.pth found in {ckpt_dir}")
        print("  Did Stage 2 complete? Check logs/stage2.log")
        sys.exit(1)

    if args.only:
        wanted = {f"epoch_2nd_{int(n):05d}.pth" for n in args.only}
        ckpts = [c for c in ckpts if c.name in wanted]
        if not ckpts:
            print(f"[ERROR] No checkpoints matched --only {args.only}")
            sys.exit(1)

    print(f"Found {len(ckpts)} checkpoint(s) to evaluate:")
    for c in ckpts:
        print(f"  {c.name}")

    # Add scripts dir to path so we can import g2p_helper
    scripts_dir = Path(__file__).parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    # Add kokoro venv site-packages if available (legacy bundled setup)
    venv_site = styletts2_dir.parent.parent / ".venv" / "lib" / "python3.12" / "site-packages"
    if venv_site.exists() and str(venv_site) not in sys.path:
        sys.path.insert(0, str(venv_site))

    failures = []
    for ckpt in ckpts:
        epoch_num = int(ckpt.stem.split("_")[-1])
        tag = f"epoch{epoch_num:02d}"
        out_dir = recipe_root / "eval" / tag

        if args.skip_existing and (out_dir / "manifest.txt").exists():
            print(f"[skip] {tag} — already evaluated")
            continue

        print()
        print("=" * 70)
        print(f"[{tag}]  {ckpt.name}")
        print("=" * 70)

        try:
            _eval_one(ckpt, first_stage, out_dir, resolved_device, recipe_root, styletts2_dir, training_dir, cfg)
        except Exception as e:
            print(f"  ! {tag} failed: {e}")
            failures.append(tag)

    print()
    print("=" * 70)
    eval_dir = recipe_root / "eval"
    print(f"Done. Listen to outputs in: {eval_dir}")
    if failures:
        print(f"FAILED: {failures}")
        sys.exit(1)


def _eval_one(ckpt: Path, first_stage: Path, out_dir: Path,
              device: str, recipe_root: Path, styletts2_dir: Path, training_dir: Path, cfg: dict):
    import torch
    import numpy as np
    import soundfile as sf
    from g2p_helper import g2p_with_lexicon, normalize_text

    out_dir.mkdir(parents=True, exist_ok=True)
    voicepack = out_dir / "voicepack.pt"

    # Extract voicepack
    import importlib.util, sys as _sys
    _spec = importlib.util.spec_from_file_location("extract_voicepack", Path(__file__).parent / "04_extract_voicepack.py")
    _mod  = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
    extract_voicepack = _mod.extract_voicepack
    extract_voicepack(
        model_path=str(ckpt),
        audio_dir=str(recipe_root / cfg["dataset"].get("wavs_dir", "data/wavs")),
        output_path=str(voicepack),
        device=device,
        style_encoder_model=str(first_stage),
    )

    # Load eval sentences
    eval_texts_file = recipe_root / "training" / "eval_texts.txt"
    if not eval_texts_file.exists():
        eval_texts_file = training_dir / "OOD_texts.txt"
    sentences = [l.strip() for l in eval_texts_file.read_text().splitlines() if l.strip() and not l.startswith("#")]

    # Convert StyleTTS2 ckpt → Kokoro inference weights
    kokoro_pth = out_dir / "kokoro_converted.pth"
    raw = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    net = raw["net"]
    kw = {}
    for k in ("bert", "bert_encoder", "predictor", "text_encoder", "decoder"):
        if k in net:
            kw[k] = {("module." + k2 if not k2.startswith("module.") else k2): v
                     for k2, v in net[k].items()}
    torch.save(kw, str(kokoro_pth))

    from kokoro import KModel, KPipeline
    cfg_json = training_dir / "config.json"
    kmodel = KModel(repo_id="hexgrad/Kokoro-82M", config=str(cfg_json), model=str(kokoro_pth))
    kmodel = kmodel.to(device).eval()

    g2p_lang = cfg.get("g2p", {}).get("language", "en-gb")
    lang_code = "b" if "gb" in g2p_lang else "a"
    pipeline = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M", model=kmodel)
    voice = torch.load(str(voicepack), map_location="cpu", weights_only=True)

    manifest = out_dir / "manifest.txt"
    with open(manifest, "w") as mf:
        mf.write(f"# ckpt={ckpt.name}  g2p={g2p_lang}\n")
        for i, text in enumerate(sentences, start=1):
            norm_text = normalize_text(text)
            ipa = g2p_with_lexicon(norm_text, language=g2p_lang)
            print(f"  [{i}/{len(sentences)}] {text[:80]}")
            gen = pipeline.generate_from_tokens(ipa, voice=voice, speed=1)
            audio_parts = []
            for _, _, audio in gen:
                audio_parts.append(audio)
            if not audio_parts:
                print("    WARN: no audio generated")
                mf.write(f"{i:02d}\tFAILED\t{text}\n")
                continue
            combined = np.concatenate(audio_parts)
            wav_path = out_dir / f"{i:02d}.wav"
            sf.write(str(wav_path), combined, 24000)
            dur = len(combined) / 24000
            print(f"    -> {wav_path.name}  {dur:.1f}s")
            mf.write(f"{i:02d}\t{dur:.2f}s\t{text}\n")
    print(f"  Eval saved: {out_dir}")


if __name__ == "__main__":
    main()
