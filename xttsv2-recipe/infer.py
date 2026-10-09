#!/usr/bin/env python3
"""
XTTS v2 Inference — run synthesis with a fine-tuned checkpoint.

Usage:
    python infer.py \
        --text "Hello, this is my custom voice." \
        --speaker-wav dataset/reference.wav \
        --checkpoint output/<run_name>/best_model.pth \
        --config    output/<run_name>/config.json \
        --out       synthesized.wav
"""
import argparse
import sys
from pathlib import Path

import torch
import torchaudio

from tts_utils.device import get_device

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


def parse_args():
    ap = argparse.ArgumentParser(description="Synthesize speech with a fine-tuned XTTS v2 checkpoint.")
    ap.add_argument("--text", required=True, help="Text to synthesize.")
    ap.add_argument("--language", default="en", help="Language code (default: en).")
    ap.add_argument("--speaker-wav", required=True,
                    help="Reference speaker WAV (same speaker as training data).")
    ap.add_argument("--checkpoint", required=True,
                    help="Path to best_model.pth from the training run.")
    ap.add_argument("--config", default=None,
                    help="Path to config.json (auto-detected from checkpoint dir if omitted).")
    ap.add_argument("--vocab", default=None,
                    help="Path to vocab.json (auto-detected from config if omitted).")
    ap.add_argument("--out", default="output.wav", help="Output WAV path (default: output.wav).")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "xpu", "cpu"],
                    help="Device to use. 'auto' detects best available (CUDA > XPU > CPU). "
                         "--device cpu is equivalent to --cpu. Default: auto")
    ap.add_argument("--cpu", action="store_true",
                    help="(Deprecated) Force CPU inference. Use --device cpu instead.")
    return ap.parse_args()


def main():
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    ref_path  = Path(args.speaker_wav)
    out_path  = Path(args.out)

    # Auto-detect config.json from the checkpoint directory
    config_path = Path(args.config) if args.config else ckpt_path.parent / "config.json"

    for p, label in [(ckpt_path, "checkpoint"), (ref_path, "speaker-wav"), (config_path, "config")]:
        if not p.exists():
            print(f"[ERROR] {label} not found: {p}", file=sys.stderr)
            sys.exit(1)

    config = XttsConfig()
    config.load_json(str(config_path))

    vocab_path = args.vocab or config.model_args.tokenizer_file
    if not vocab_path:
        print("[ERROR] Cannot find vocab.json — pass --vocab explicitly.", file=sys.stderr)
        sys.exit(1)

    device = get_device("cpu") if args.cpu else get_device(args.device)
    print(f"[info] Device: {device}")
    print(f"[info] Checkpoint: {ckpt_path}")
    print(f"[info] Reference: {ref_path}")

    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=str(ckpt_path), vocab_path=str(vocab_path),
                          use_deepspeed=False)
    model.to(device)

    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=str(ref_path),
        gpt_cond_len=config.gpt_cond_len,
        max_ref_length=config.max_ref_len,
        sound_norm_refs=config.sound_norm_refs,
    )

    result = model.inference(
        text=args.text,
        language=args.language,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=config.temperature,
        length_penalty=config.length_penalty,
        repetition_penalty=config.repetition_penalty,
        top_k=config.top_k,
        top_p=config.top_p,
        enable_text_splitting=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wav = torch.tensor(result["wav"], dtype=torch.float32).unsqueeze(0).cpu()
    torchaudio.save(str(out_path), wav, config.audio.output_sample_rate)
    print(f"[done] Saved: {out_path}")


if __name__ == "__main__":
    main()
