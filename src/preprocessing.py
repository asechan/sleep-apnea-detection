"""Feature contract shared by training-data generation and inference."""
import math
import librosa
import numpy as np


def compute_log_mel(epoch_wave, sr, n_mels, n_fft, hop_length):
    power = librosa.feature.melspectrogram(
        y=epoch_wave, sr=sr, n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, power=2.0)
    return librosa.power_to_db(power, ref=np.max).astype(np.float32)


def validate_preprocessing(meta):
    for key in ("sr", "n_mels", "n_fft", "hop_length"):
        value = meta[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"Invalid preprocessing {key}")
    for key in ("train_mean", "train_std", "epoch_sec"):
        if not math.isfinite(meta[key]):
            raise ValueError(f"Invalid preprocessing {key}")
    if meta["train_std"] <= 0 or meta["epoch_sec"] <= 0:
        raise ValueError("Training standard deviation and epoch duration must be positive")
    return meta


def normalized_mel(audio, meta):
    validate_preprocessing(meta)
    mel = compute_log_mel(audio, **{k: meta[k] for k in ("sr", "n_mels", "n_fft", "hop_length")})
    return ((mel - meta["train_mean"]) / meta["train_std"]).astype(np.float32)
