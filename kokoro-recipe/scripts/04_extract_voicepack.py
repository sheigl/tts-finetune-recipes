#!/usr/bin/env python3
"""
Step 4 (helper): Extract Voicepack from a Checkpoint
=====================================================
A voicepack is a frozen [510, 1, 256] tensor capturing the speaker's acoustic
timbre (first 128 dims from style_encoder) and prosody (last 128 dims from
predictor_encoder). It's used at inference time via:
    pipeline.generate_from_tokens(ipa, voice=voicepack)

Usage:
    # After Stage 2 (recommended — use Stage 1 for style_encoder):
    python scripts/04_extract_voicepack.py \
        --model    output/.../epoch_2nd_00008.pth \
        --first-stage output/.../first_stage.pth \
        --audio-dir data/wavs/ \
        --out      voicepack.pt

    # Stage 1 only:
    python scripts/04_extract_voicepack.py \
        --model output/.../first_stage.pth \
        --audio-dir data/wavs/ \
        --out voicepack.pt

Voicepack shape: [510, 1, 256] float32
  - 510 = max phoneme sequence length
  - First 128 dims = acoustic/timbre (style_encoder)
  - Last  128 dims = prosody (predictor_encoder)

Credits: extract_voicepack.py from kokoro-deutsch by semidark
         https://github.com/semidark/kokoro-deutsch
"""
import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from tts_utils.device import get_device


# ── StyleTTS2 StyleEncoder (standalone, no external deps) ────────────────────

class LearnedDownSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "none":
            self.conv = nn.Identity()
        elif layer_type == "timepreserve":
            self.conv = spectral_norm(nn.Conv2d(dim_in, dim_in, (3, 1), (2, 1), groups=dim_in, padding=(1, 0)))
        elif layer_type == "half":
            self.conv = spectral_norm(nn.Conv2d(dim_in, dim_in, (3, 3), (2, 2), groups=dim_in, padding=1))
        else:
            raise RuntimeError(f"Unknown downsample: {layer_type}")

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "none":
            return x
        elif self.layer_type == "timepreserve":
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == "half":
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool2d(x, 2)
        raise RuntimeError(f"Unknown downsample: {self.layer_type}")


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2), normalize=False, downsample="none"):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.downsample_res = LearnedDownSample(downsample, dim_in)
        self.learned_sc = dim_in != dim_out
        self.conv1 = spectral_norm(nn.Conv2d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1))
        if normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = spectral_norm(nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample_res(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        return self.conv2(x)

    def forward(self, x):
        return (self._shortcut(x) + self._residual(x)) / math.sqrt(2)


class StyleEncoder(nn.Module):
    def __init__(self, dim_in=48, style_dim=48, max_conv_dim=384):
        super().__init__()
        blocks = [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]
        for _ in range(4):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks.append(ResBlk(dim_in, dim_out, downsample="half"))
            dim_in = dim_out
        blocks += [nn.LeakyReLU(0.2), spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0)),
                   nn.AdaptiveAvgPool2d(1), nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)
        self.unshared = nn.Linear(dim_out, style_dim)

    def forward(self, x):
        h = self.shared(x).view(self.shared(x).size(0), -1) if False else self.shared(x)
        h = h.view(h.size(0), -1)
        return self.unshared(h)


# ── Extraction logic ──────────────────────────────────────────────────────────

def extract_voicepack(
    model_path: str,
    audio_dir: str,
    output_path: str,
    num_samples: int = 200,
    device: str = "auto",
    style_encoder_model: str = None,
):
    import random
    import soundfile as sf
    import torchaudio

    device = get_device(device)
    print(f"Device: {device}")

    audio_files = sorted(Path(audio_dir).glob("*.wav"))
    if not audio_files:
        print(f"[ERROR] No WAV files in {audio_dir}")
        sys.exit(1)
    print(f"Found {len(audio_files)} WAV files")

    rng = random.Random(42)
    if len(audio_files) > num_samples:
        audio_files = rng.sample(audio_files, num_samples)
    print(f"Using {len(audio_files)} samples")

    def strip_prefix(sd, prefix="module."):
        if any(k.startswith(prefix) for k in sd):
            return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items()}
        return sd

    print(f"Loading: {model_path}")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    net  = ckpt["net"]

    # Kokoro-82M dims: dim_in=64, style_dim=128, max_conv_dim=512
    style_encoder     = StyleEncoder(dim_in=64, style_dim=128, max_conv_dim=512)
    predictor_encoder = StyleEncoder(dim_in=64, style_dim=128, max_conv_dim=512)

    if style_encoder_model:
        print(f"  style_encoder from: {style_encoder_model}")
        se_ckpt = torch.load(style_encoder_model, map_location="cpu", weights_only=False)
        style_encoder.load_state_dict(strip_prefix(se_ckpt["net"]["style_encoder"]))
    else:
        style_encoder.load_state_dict(strip_prefix(net["style_encoder"]))

    pe_trained = True
    try:
        predictor_encoder.load_state_dict(strip_prefix(net["predictor_encoder"]))
        with torch.no_grad():
            dummy = torch.randn(1, 1, 80, 200)
            pe_norm = predictor_encoder(dummy).norm().item()
            se_norm = style_encoder(dummy).norm().item()
            if pe_norm > 1e3 or pe_norm < se_norm * 0.5:
                pe_trained = False
    except Exception:
        pe_trained = False

    if not pe_trained:
        print("  predictor_encoder untrained — using style_encoder for both halves")
        src = se_ckpt["net"]["style_encoder"] if style_encoder_model else net["style_encoder"]
        predictor_encoder.load_state_dict(strip_prefix(src))

    style_encoder     = style_encoder.to(device).eval()
    predictor_encoder = predictor_encoder.to(device).eval()

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000, n_fft=2048, win_length=1200, hop_length=300, n_mels=80
    ).to(device)

    acoustic_styles, prosodic_styles = [], []
    print("Extracting style vectors...")
    with torch.no_grad():
        for i, wav_path in enumerate(audio_files):
            data, sr = sf.read(str(wav_path), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            waveform = torch.from_numpy(data).unsqueeze(0)
            if sr != 24000:
                waveform = torchaudio.functional.resample(waveform, sr, 24000)
            waveform = waveform.to(device)
            mel = mel_transform(waveform)
            mel = (torch.log(1e-5 + mel) - (-4)) / 4
            if mel.shape[-1] < 80:
                continue
            mel_in = mel.unsqueeze(1)
            acoustic_styles.append(style_encoder(mel_in).cpu())
            prosodic_styles.append(predictor_encoder(mel_in).cpu())
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(audio_files)}")

    if not acoustic_styles:
        print("[ERROR] No style vectors extracted. Check audio files.")
        sys.exit(1)

    avg_a = torch.cat(acoustic_styles).mean(0)
    avg_p = torch.cat(prosodic_styles).mean(0)
    combined  = torch.cat([avg_a, avg_p])
    voicepack = combined.unsqueeze(0).unsqueeze(0).expand(510, 1, 256).clone()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(voicepack, str(out))
    print(f"Saved voicepack: {out}  ({out.stat().st_size / 1024:.1f} KB)")
    print(f"  Acoustic norm: {avg_a.norm():.4f}   Prosodic norm: {avg_p.norm():.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model",        required=True, help="Stage 2 checkpoint (.pth)")
    ap.add_argument("--first-stage",  default=None,  help="Stage 1 checkpoint for style_encoder")
    ap.add_argument("--audio-dir",    required=True, help="Directory of speaker WAV files")
    ap.add_argument("--out",          required=True, help="Output voicepack path (.pt)")
    ap.add_argument("--num-samples",  type=int, default=200)
    ap.add_argument("--device",       default="auto", choices=["auto", "cuda", "xpu", "cpu"],
                     help="Device to use. 'auto' detects best available (CUDA > XPU > CPU). Default: auto")
    args = ap.parse_args()
    extract_voicepack(
        model_path=args.model,
        audio_dir=args.audio_dir,
        output_path=args.out,
        num_samples=args.num_samples,
        device=args.device,
        style_encoder_model=args.first_stage,
    )


if __name__ == "__main__":
    main()
