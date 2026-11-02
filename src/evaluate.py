#!/usr/bin/env python3
"""Evaluate a frozen checkpoint and validation-selected threshold on test subjects."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.cnn import SimpleCNN
from artifacts import validate_features
from train_cnn import MemmapDataset, collate_fn


def metrics(targets, probabilities, threshold):
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)
    if targets.size == 0 or targets.shape != probabilities.shape:
        raise ValueError('Targets and probabilities must be nonempty and have matching shapes')
    if not np.isin(targets, [0, 1]).all() or not np.isfinite(probabilities).all():
        raise ValueError('Invalid labels or probabilities')
    if np.any((probabilities < 0) | (probabilities > 1)) or not 0 < threshold < 1:
        raise ValueError('Probabilities or threshold out of range')
    predicted = (probabilities >= threshold).astype(int)
    return dict(count=len(targets), positives=int(targets.sum()),
                accuracy=float(accuracy_score(targets, predicted)),
                precision=float(precision_score(targets, predicted, zero_division=0)),
                recall=float(recall_score(targets, predicted, zero_division=0)),
                f1=float(f1_score(targets, predicted, zero_division=0)),
                roc_auc=float(roc_auc_score(targets, probabilities)) if len(np.unique(targets)) == 2 else None,
                confusion_matrix=confusion_matrix(targets, predicted, labels=[0, 1]).tolist())


def subject_rows(processed_dir, meta):
    """Reconstruct and verify the feature writer's sorted-subject row ordering."""
    rows = []
    for directory in sorted(Path(processed_dir).iterdir()):
        path = directory / 'epoch_meta.json'
        if directory.is_dir() and path.exists():
            info = json.loads(path.read_text())
            rows.extend((directory.name, epoch) for epoch in range(info['n_epochs']))
    if len(rows) != meta['total_epochs']:
        raise ValueError('Processed subject counts do not match feature metadata')
    for split in ('train', 'val', 'test'):
        subjects = set(meta['splits'][split + '_subjects'])
        actual = {rows[i][0] for i in meta['splits'][split]}
        if actual != subjects or any(rows[i][0] not in subjects for i in meta['splits'][split]):
            raise ValueError(f'Subject ordering does not match {split} split')
    return rows


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features_dir', type=Path, required=True)
    parser.add_argument('--processed_dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Evaluation output directory is not empty; use a fresh directory')
    meta = validate_features(args.features_dir)
    mapping = subject_rows(args.processed_dir, meta)
    config = json.loads(args.config.read_text())
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if config.get('checkpoint_sha256') != checkpoint_hash:
        raise ValueError('Threshold configuration must be tuned for this exact checkpoint')
    threshold = config['threshold']
    if not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        raise ValueError('Invalid threshold')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if 'preprocessing' not in checkpoint or any(meta.get(k) != v for k, v in checkpoint['preprocessing'].items()):
        raise ValueError('Checkpoint preprocessing does not match features')
    device = torch.device('cpu' if args.cpu else 'cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = SimpleCNN(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    indices = meta['splits']['test']
    dataset = MemmapDataset(str(args.features_dir / 'X.npy'), str(args.features_dir / 'y.npy'), indices)
    probabilities, targets = [], []
    for x, y in DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn):
        probabilities.extend(torch.sigmoid(model(x.to(device))).cpu().tolist())
        targets.extend(y.int().tolist())
    groups = {}
    for offset, index in enumerate(indices):
        groups.setdefault(mapping[index][0], []).append(offset)
    targets, probabilities = np.asarray(targets), np.asarray(probabilities)
    per_subject = {subject: metrics(targets[offsets], probabilities[offsets], threshold)
                   for subject, offsets in groups.items()}
    report = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_hash,
                  checkpoint_epoch=checkpoint['epoch'], threshold=threshold, split='test',
                  confusion_matrix_order='[[TN, FP], [FN, TP]]',
                  overall=metrics(targets, probabilities, threshold), per_subject=per_subject,
                  subject_macro_f1=float(np.mean([m['f1'] for m in per_subject.values()])),
                  always_positive_baseline=metrics(targets, np.ones(len(targets)), threshold),
                  always_negative_baseline=metrics(targets, np.zeros(len(targets)), threshold))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    with (args.output_dir / 'predictions.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['global_index', 'subject', 'epoch_index', 'label', 'probability', 'prediction'])
        for index, target, probability in zip(indices, targets, probabilities):
            writer.writerow([index, *mapping[index], int(target), float(probability), int(probability >= threshold)])
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
