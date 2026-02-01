#!/usr/bin/env python3
import json
import sys
import numpy as np
import torch
from pathlib import Path
import soundfile as sf
import librosa

# add project root to python path so local modules import cleanly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.cnn import SimpleCNN


# these constants should match the training setup
SR = 4000
EPOCH_SEC = 30
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 256

def postprocess_events(timeline, epoch_sec=30, min_epochs=2):
    """
    timeline = list of dicts with keys:
      start, end, prob, label (0/1)
    Returns merged events + AHI
    """
    events = []
    in_event = False
    current = None

    for row in timeline:
        if row["label"] == 1:
            if not in_event:
                in_event = True
                current = {
                    "start": row["start"],
                    "end": row["end"],
                    "max_prob": row["prob"],
                    "epochs": 1
                }
            else:
                current["end"] = row["end"]
                current["max_prob"] = max(current["max_prob"], row["prob"])
                current["epochs"] += 1
        else:
            if in_event:
                if current["epochs"] >= min_epochs:
                    events.append(current)
                in_event = False
                current = None

    if in_event and current["epochs"] >= min_epochs:
        events.append(current)

    return events

def load_threshold():
    cfg = ROOT / "config.json"
    if cfg.exists():
        return json.load(open(cfg))["threshold"]
    return 0.5

def epoch_audio(audio, sr):
    samples_per_epoch = int(EPOCH_SEC * sr)
    epochs = []
    for i in range(0, len(audio), samples_per_epoch):
        seg = audio[i:i+samples_per_epoch]
        if len(seg) < samples_per_epoch:
            seg = np.pad(seg, (0, samples_per_epoch-len(seg)))
        epochs.append(seg.astype(np.float32))
    return epochs

def audio_to_mel(audio):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db.astype(np.float32)

@torch.no_grad()
def main():
    if len(sys.argv) < 2:
        print("Usage: python src/infer_apnea.py <audio.wav>")
        sys.exit(1)

    wav_path = Path(sys.argv[1])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print("Using device:", device)
    threshold = load_threshold()
    print("Using threshold:", threshold)

    model = SimpleCNN()
    ckpt = torch.load(ROOT / "checkpoints/best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != SR:
        print(f"WARNING: expected {SR}Hz, got {sr}Hz. Resampling...")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        sr = SR

    epochs = epoch_audio(audio, sr)

    print(f"Processing {len(epochs)} epochs...")

    timeline = []

    for i, ep in enumerate(epochs):
        mel = audio_to_mel(ep)
        X = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(device)
        logits = model(X)
        prob = torch.sigmoid(logits).item()

        label = 1 if prob >= threshold else 0
        start_sec = i * EPOCH_SEC
        end_sec = start_sec + EPOCH_SEC

        timeline.append({
            "start": start_sec,
            "end": end_sec,
            "prob": prob,
            "label": label
        })

    events = postprocess_events(timeline)

    sleep_hours = (len(epochs) * EPOCH_SEC) / 3600
    ahi = len(events) / sleep_hours

    if ahi < 5:
        severity = "Normal"
    elif ahi < 15:
        severity = "Mild"
    elif ahi < 30:
        severity = "Moderate"
    else:
        severity = "Severe"

    print("\n===== SLEEP APNEA REPORT =====")
    print(f"Total sleep time: {sleep_hours:.2f} hours")
    print(f"Detected events: {len(events)}")
    print(f"AHI: {ahi:.2f}")
    print(f"Severity: {severity}")

    print("\nEvents:")
    for i, ev in enumerate(events, 1):
        print(f"{i:3d}. {ev['start']:6.1f}s - {ev['end']:6.1f}s | peak p={ev['max_prob']:.3f}")

if __name__ == "__main__":
    main()