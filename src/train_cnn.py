#!/usr/bin/env python3
# src/train_cnn.py
"""
Train a convolutional model on the memmapped spectrogram dataset.

Saves:
 - checkpoints/best.pt  -- model with best validation F1
 - checkpoints/last.pt  -- most recent checkpoint
"""

#!/usr/bin/env python3

import sys
from pathlib import Path

# make repo root importable for local modules
ROOT = Path(__file__).resolve().parents[1]
print("DEBUG ROOT:", ROOT)
print("DEBUG models exists:", (ROOT / "models").exists())

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
import random
import os
from typing import List

from models.cnn import SimpleCNN
# Dataset: memmap wrapper
class MemmapDataset(Dataset):
    """
    Memory-mapped dataset wrapper for X.npy (N,1,n_mels,T) and y.npy (N,)
    """
    def __init__(self, features_path: str, labels_path: str, indices: List[int], dtype=np.float32):
        self.X = np.memmap(features_path, mode='r', dtype='float32')
        # reshape based on meta.json is required; we'll infer shape from meta passed externally
        self.meta = json.load(open(str(Path(features_path).parent / "meta.json")))
        total = int(self.meta["total_epochs"])
        n_mels = int(self.meta["n_mels"])
        T = int(self.meta["T"])
        self.shape = (total, 1, n_mels, T)
        self.X = self.X.reshape(self.shape)
        self.y = np.memmap(labels_path, mode='r', dtype='int8', shape=(total,))
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        gidx = self.indices[idx]
        mel = np.array(self.X[gidx], dtype=np.float32)  # shape (1,n_mels,T)
        label = int(self.y[gidx])
        # convert to torch tensors in training loop for performance reasons
        return mel, label

# Helper utilities
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def collate_fn(batch):
    xs = [torch.from_numpy(b[0]) for b in batch]
    ys = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    X = torch.stack(xs, dim=0)  # (B,1,n_mels,T)
    return X, ys

# Training & validation loops
def train_one_epoch(model, loader, optimizer, device, scaler, criterion, clip_grad=None):
    model.train()
    preds_all = []
    targets_all = []
    running_loss = 0.0
    for X, y in tqdm(loader, desc="train", leave=False):
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
            logits = model(X)
            loss = criterion(logits, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            if clip_grad is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
        running_loss += float(loss.item()) * X.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds_all.append(probs)
        targets_all.append(y.detach().cpu().numpy())
    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)
    loss_avg = running_loss / len(loader.dataset)
    preds_bin = (preds_all >= 0.5).astype(int)
    acc = accuracy_score(targets_all, preds_bin)
    f1 = f1_score(targets_all, preds_bin, zero_division=0)
    try:
        auc = roc_auc_score(targets_all, preds_all)
    except Exception:
        auc = float("nan")
    return loss_avg, acc, f1, auc

@torch.no_grad()
def validate(model, loader, device, criterion):
    model.eval()
    preds_all = []
    targets_all = []
    running_loss = 0.0
    for X, y in tqdm(loader, desc="val", leave=False):
        X = X.to(device)
        y = y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        running_loss += float(loss.item()) * X.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds_all.append(probs)
        targets_all.append(y.detach().cpu().numpy())
    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)
    loss_avg = running_loss / len(loader.dataset)
    preds_bin = (preds_all >= 0.5).astype(int)
    acc = accuracy_score(targets_all, preds_bin)
    f1 = f1_score(targets_all, preds_bin, zero_division=0)
    try:
        auc = roc_auc_score(targets_all, preds_all)
    except Exception:
        auc = float("nan")
    return loss_avg, acc, f1, auc

# Main training entrypoint
def main(args):
    set_seed(args.seed)
    # -------- Device selection: CUDA > MPS > CPU --------
    if not args.cpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    features_dir = Path(args.features_dir)
    X_path = features_dir / "X.npy"
    y_path = features_dir / "y.npy"
    meta = json.load(open(features_dir / "meta.json"))
    total = int(meta["total_epochs"])
    split_map = meta["splits"]
    train_idx = split_map["train"]
    val_idx = split_map["val"]
    test_idx = split_map["test"]

    # build datasets
    train_dataset = MemmapDataset(str(X_path), str(y_path), train_idx)
    val_dataset = MemmapDataset(str(X_path), str(y_path), val_idx)
    test_dataset = MemmapDataset(str(X_path), str(y_path), test_idx)

    # weighted sampler for training to balance classes
    y_all = np.memmap(str(y_path), mode='r', dtype='int8', shape=(total,))
    # compute per-sample weights (inverse freq)
    class_counts = np.bincount(y_all.astype(np.int64), minlength=2)
    class_weights = {0: 1.0, 1: 1.0}
    if class_counts[1] > 0:
        class_weights = {0: float(class_counts.sum())/class_counts[0], 1: float(class_counts.sum())/class_counts[1]}
    sample_weights = np.array([class_weights[int(y_all[idx])] for idx in train_idx], dtype=np.float32)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_idx), replacement=True)

    # dataloaders
    pin = (device.type == "cuda")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, sampler=sampler, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=pin)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=pin)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=pin)

    # model
    model = SimpleCNN(in_channels=1, n_classes=1, base_filters=args.base_filters, dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tmax, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    use_amp = (device.type == "cuda" and args.use_amp)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_val_f1 = -1.0
    epochs_no_improve = 0
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc, train_f1, train_auc = train_one_epoch(model, train_loader, optimizer, device, scaler, criterion, clip_grad=args.clip_grad)
        val_loss, val_acc, val_f1, val_auc = validate(model, val_loader, device, criterion)
        print(f" train: loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f} auc={train_auc:.4f}")
        print(f"  val : loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} auc={val_auc:.4f}")

        # scheduler step
        scheduler.step()

        # checkpoint last
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_f1": val_f1
        }, Path(args.checkpoint_dir) / "last.pt")

        # best-model logic
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_f1": val_f1
            }, Path(args.checkpoint_dir) / "best.pt")
            epochs_no_improve = 0
            print(" New best model saved.")
        else:
            epochs_no_improve += 1
            print(f" No improvement for {epochs_no_improve} epochs.")

        # early stopping
        if epochs_no_improve >= args.patience:
            print(f"Stopping early after {epoch} epochs (patience={args.patience})")
            break

    # final evaluation on test set with best model
    print("Loading best model for final test eval...")
    ckpt = torch.load(Path(args.checkpoint_dir) / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc, test_f1, test_auc = validate(model, test_loader, device, criterion)
    print(f"TEST : loss={test_loss:.4f} acc={test_acc:.4f} f1={test_f1:.4f} auc={test_auc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=str, default="data/features")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--tmax", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--base_filters", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()
    main(args)