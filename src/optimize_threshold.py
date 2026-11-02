#!/usr/bin/env python3
import argparse
import hashlib
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=Path, default=ROOT / "data/features")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "config.json")
    args = parser.parse_args()
    features_dir = args.features_dir
    ckpt_path = args.checkpoint

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

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "preprocessing" in ckpt and any(meta.get(k) != v for k, v in ckpt["preprocessing"].items()):
        raise ValueError("Feature metadata conflicts with checkpoint preprocessing")
    model = SimpleCNN(**ckpt.get("model_config", {})).to(device)
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
    best_f1 = -1
    best_t = 0.5

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
    if best_f1 <= 0:
        raise ValueError("No threshold achieved positive validation F1; configuration was not changed")
    config = json.loads(args.output.read_text()) if args.output.exists() else {}
    config["threshold"] = float(best_t)
    config["checkpoint_sha256"] = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    config["validation_f1"] = float(best_f1)
    config["selection_split"] = "val"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Saved threshold to {args.output}")

if __name__ == "__main__":
    main()
