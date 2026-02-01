#!/usr/bin/env python3
"""
make_dataset.py

Creates a dataset of log-mel spectrograms from epoch files produced by epoch_labeler.py.

Outputs (into data/features/):
 - X.npy        : memmap float32 (N, 1, n_mels, T)  -- normalized using training stats
 - y.npy        : int8 (N,)
 - meta.json    : contains n_mels, n_fft, hop_length, sr, epoch_sec, splits (train/val/test indices), mean/std

Notes:
 - Uses memmap to avoid huge RAM usage.
 - Two-pass approach:
    1) compute & store raw mel -> memmap, and compute running mean/std over TRAIN set only
    2) normalize entire memmap in-place with computed mean/std
 - Default audio/mel params chosen by you:
    sr=4000, n_fft=512, hop_length=128, n_mels=64
"""
from pathlib import Path
import numpy as np
import librosa
import json
from tqdm import tqdm
import argparse
import csv
import math
import random
from typing import List, Tuple

# ---------------- Utility: running mean/std (Welford) ----------------
class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
    def push(self, x: np.ndarray):
        # x: numpy array of values (flattened)
        x = np.asarray(x, dtype=np.float64).ravel()
        for val in x:
            self.n += 1
            delta = val - self.mean
            self.mean += delta / self.n
            delta2 = val - self.mean
            self.M2 += delta * delta2
    def get(self) -> Tuple[float,float]:
        if self.n < 2:
            return float(self.mean), float(1.0)
        var = self.M2 / (self.n - 1)
        std = math.sqrt(var) if var > 0 else 1.0
        return float(self.mean), float(std)
    
def build_subject_splits(subject_dirs, seed=42, ratios=(0.8, 0.1, 0.1)):
    rng = random.Random(seed)
    subjects = [s.name for s in subject_dirs]
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    train_subs = set(subjects[:n_train])
    val_subs = set(subjects[n_train:n_train+n_val])
    test_subs = set(subjects[n_train+n_val:])

    return train_subs, val_subs, test_subs

# ---------------- Mel computation ----------------
def compute_log_mel(epoch_wave: np.ndarray, sr: int, n_mels: int, n_fft: int, hop_length: int) -> np.ndarray:
    # Returns float32 (n_mels, T) (log-power dB)
    S = librosa.feature.melspectrogram(y=epoch_wave, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.astype(np.float32)

# ---------------- Main workflow ----------------
def find_subject_dirs(processed_dir: Path) -> List[Path]:
    subs = [p for p in sorted(processed_dir.iterdir()) if p.is_dir()]
    return subs

def read_labels_csv(path: Path) -> np.ndarray:
    # expects header with label as last column (epoch_index,start_sec,end_sec,label)
    labels = []
    with open(path, "r") as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            # label assumed at index of 'label' column
            try:
                lbl = int(row[-1])
            except Exception:
                lbl = 0
            labels.append(lbl)
    return np.asarray(labels, dtype=np.int8)

def gather_index_map(processed_dir: Path) -> Tuple[List[Tuple[Path,int]], int]:
    """
    Build list mapping global_index -> (subject_dir, epoch_index)
    and also return total_n_epochs.
    """
    subs = find_subject_dirs(processed_dir)
    mapping = []
    for s in subs:
        meta_path = s / "epoch_meta.json"
        if not meta_path.exists():
            continue
        meta = json.load(open(meta_path))
        n_epochs = int(meta["n_epochs"])
        for i in range(n_epochs):
            mapping.append((s, i))
    return mapping, len(mapping)

def build_splits(total_n: int, seed: int=42, ratios=(0.8,0.1,0.1)) -> Tuple[List[int],List[int],List[int]]:
    rng = random.Random(seed)
    indices = list(range(total_n))
    rng.shuffle(indices)
    n_train = int(total_n * ratios[0])
    n_val = int(total_n * ratios[1])
    train = indices[:n_train]
    val = indices[n_train:n_train+n_val]
    test = indices[n_train+n_val:]
    return train, val, test

def main(processed_dir: str = "data/processed",
         out_dir: str = "data/features",
         sr: int = 4000,
         n_mels: int = 64,
         n_fft: int = 512,
         hop_length: int = 128,
         seed: int = 42):
    processed_dir = Path(processed_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) build index map
    mapping, total_n = gather_index_map(processed_dir)
    if total_n == 0:
        raise RuntimeError("No subjects/epochs found in processed directory.")
    print(f"Found {total_n} epochs across {len(set([m[0] for m in mapping]))} subjects.")

    # 2) build splits
    subject_dirs = find_subject_dirs(processed_dir)
    train_subs, val_subs, test_subs = build_subject_splits(subject_dirs, seed=seed)

    train_idx, val_idx, test_idx = [], [], []

    for g_idx, (subj_dir, _) in enumerate(mapping):
        sid = subj_dir.name
        if sid in train_subs:
            train_idx.append(g_idx)
        elif sid in val_subs:
            val_idx.append(g_idx)
        else:
            test_idx.append(g_idx)

    split_map = {
        "train_subjects": sorted(list(train_subs)),
        "val_subjects": sorted(list(val_subs)),
        "test_subjects": sorted(list(test_subs)),
        "train": train_idx,
        "val": val_idx,
        "test": test_idx
    }

    # 3) compute mel shape by reading first epoch
    # get first mapping that actually has valid epochs.npy
    sample_mel = None
    sample_shapes = None
    for subj_dir, eidx in mapping:
        epochs_mm = np.memmap(str(subj_dir / "epochs.npy"), mode="r", dtype="float32")
        meta = json.load(open(subj_dir / "epoch_meta.json"))
        n_epochs = int(meta["n_epochs"])
        samples_per_epoch = int(meta["samples_per_epoch"])
        if n_epochs <= 0:
            continue
        # reshape memmap if needed
        epochs_mm = epochs_mm.reshape((n_epochs, samples_per_epoch))
        wav = np.array(epochs_mm[eidx], dtype=np.float32)
        mel = compute_log_mel(wav, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        sample_mel = mel
        sample_shapes = (mel.shape[0], mel.shape[1])  # (n_mels, T)
        break
    if sample_mel is None:
        raise RuntimeError("Could not compute a sample mel (no epochs found?)")

    n_mels_det, T = sample_shapes
    assert n_mels_det == n_mels, "computed n_mels mismatch"

    print(f"Mel spectrogram shape per epoch: (n_mels={n_mels}, T={T})")
    # 4) create memmap containers
    X_path = out_dir / "X.npy"
    y_path = out_dir / "y.npy"

    # create memmap file for raw (unnormalized) mel data (N, 1, n_mels, T)
    X_shape = (total_n, 1, n_mels, T)
    X_mm = np.memmap(str(X_path), mode="w+", dtype="float32", shape=X_shape)
    y_mm = np.memmap(str(y_path), mode="w+", dtype="int8", shape=(total_n,))

    # 5) two-pass approach: write raw mels and compute running stats over train set
    stats = RunningStats()
    print("Pass 1/2: computing raw mels & training stats (running)...")
    for g_idx, (subj_dir, epoch_idx) in enumerate(tqdm(mapping, total=total_n)):
        meta = json.load(open(subj_dir / "epoch_meta.json"))
        n_epochs = int(meta["n_epochs"])
        samples_per_epoch = int(meta["samples_per_epoch"])
        # load memmap and reshape
        epochs_mm = np.memmap(str(subj_dir / "epochs.npy"), mode="r", dtype="float32", shape=(n_epochs, samples_per_epoch))
        wav = np.array(epochs_mm[epoch_idx], dtype=np.float32)
        mel = compute_log_mel(wav, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)  # (n_mels, T)
        # store raw mel
        X_mm[g_idx, 0, :, :] = mel
        # read label
        labels = read_labels_csv(subj_dir / "labels.csv")
        lbl = int(labels[epoch_idx]) if epoch_idx < len(labels) else 0
        y_mm[g_idx] = lbl
        # if this example is in training set, update running stats
        if g_idx in train_idx:
            stats.push(mel)  # flatten inside push

    # finalize train mean/std
    mean_train, std_train = stats.get()
    if std_train == 0.0:
        std_train = 1.0
    print(f"Train mean={mean_train:.6f} std={std_train:.6f}")

    # 6) Pass 2: normalize memmap in-place using train stats
    print("Pass 2/2: normalizing memmap in-place (train stats applied)...")
    chunk = 256  # process this many epochs at a time
    for start in tqdm(range(0, total_n, chunk)):
        end = min(total_n, start + chunk)
        block = X_mm[start:end].astype(np.float32)  # shape (B,1,n_mels,T)
        # apply normalization (elementwise)
        block = (block - mean_train) / std_train
        X_mm[start:end] = block

    # flush memmap
    del X_mm
    del y_mm

    # 7) save meta
    meta = {
        "n_mels": n_mels,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "sr": sr,
        "T": T,
        "total_epochs": total_n,
        "splits": split_map,
        "train_mean": mean_train,
        "train_std": std_train,
        "epoch_sec": float(meta.get("epoch_sec", 30.0))
    }
    json.dump(meta, open(out_dir / "meta.json", "w"), indent=2)
    print(f"Saved features to {out_dir}. X shape = {X_shape}, y shape = ({total_n},)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--out_dir", type=str, default="data/features")
    parser.add_argument("--sr", type=int, default=4000)
    parser.add_argument("--n_mels", type=int, default=64)
    parser.add_argument("--n_fft", type=int, default=512)
    parser.add_argument("--hop_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(processed_dir=args.processed_dir, out_dir=args.out_dir, sr=args.sr, n_mels=args.n_mels, n_fft=args.n_fft, hop_length=args.hop_length, seed=args.seed)