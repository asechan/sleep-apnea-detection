#!/usr/bin/env python3
import torch
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import sys

# make repo root importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.cnn import SimpleCNN
from train_cnn import MemmapDataset, collate_fn
from torch.utils.data import DataLoader

@torch.no_grad()
def main():
    features_dir = Path("data/features")
    ckpt_path = Path("checkpoints/best.pt")

    meta = json.load(open(features_dir / "meta.json"))
    val_idx = meta["splits"]["val"]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    dataset = MemmapDataset(
        str(features_dir / "X.npy"),
        str(features_dir / "y.npy"),
        val_idx
    )

    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    model = SimpleCNN().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    probs = []
    targets = []

    for X, y in tqdm(loader, desc="Collecting predictions"):
        X = X.to(device)
        logits = model(X)
        p = torch.sigmoid(logits).cpu().numpy()
        probs.append(p)
        targets.append(y.numpy())

    probs = np.concatenate(probs)
    targets = np.concatenate(targets)

    print("\nThreshold tuning results:")
    best_f1 = 0
    best_t = 0

    for t in np.linspace(0.05, 0.95, 19):
        preds = (probs >= t).astype(int)
        f1 = f1_score(targets, preds, zero_division=0)
        prec = precision_score(targets, preds, zero_division=0)
        rec = recall_score(targets, preds, zero_division=0)

        print(f"t={t:.2f} | F1={f1:.4f} | P={prec:.4f} | R={rec:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    print("\nBEST THRESHOLD")
    print(f"t = {best_t:.2f}")
    print(f"F1 = {best_f1:.4f}")

if __name__ == "__main__":
    main()