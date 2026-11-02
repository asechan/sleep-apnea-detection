#!/usr/bin/env python3
"""Report positive audio windows using the model's training feature contract."""
import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from models.cnn import SimpleCNN
from preprocessing import normalized_mel, validate_preprocessing


def postprocess_events(timeline, epoch_sec=30, min_epochs=1):
    """Merge adjacent positive windows into regions, not individual apnea events."""
    regions = []
    current = None
    for row in timeline:
        if row['label'] == 1:
            if current is None or row['start'] != current['end']:
                if current is not None and current['epochs'] >= min_epochs:
                    regions.append(current)
                current = dict(start=row['start'], end=row['end'], max_prob=row['prob'], epochs=1)
            else:
                current['end'] = row['end']
                current['max_prob'] = max(current['max_prob'], row['prob'])
                current['epochs'] += 1
        elif current is not None:
            if current['epochs'] >= min_epochs:
                regions.append(current)
            current = None
    if current is not None and current['epochs'] >= min_epochs:
        regions.append(current)
    return regions


def load_threshold(path=ROOT / 'config.json'):
    value = json.loads(Path(path).read_text())['threshold']
    if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 < value < 1:
        raise ValueError('Threshold must be finite and between 0 and 1')
    return value


def epoch_audio(audio, sr, epoch_sec=30):
    samples = int(round(epoch_sec * sr))
    if samples <= 0 or len(audio) == 0:
        raise ValueError('Audio must be nonempty and epoch duration positive')
    for i in range(0, len(audio), samples):
        segment = audio[i:i + samples]
        yield np.pad(segment, (0, samples - len(segment))).astype(np.float32)


def audio_to_mel(audio, metadata):
    return normalized_mel(audio, metadata)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/best.pt')
    parser.add_argument('--metadata', type=Path, help='Matching training meta.json required for legacy checkpoints')
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    try:
        info = sf.info(str(args.audio))
        if info.frames == 0:
            raise ValueError('Audio file is empty')
        device = torch.device('cpu' if args.cpu else 'cuda' if torch.cuda.is_available()
                              else 'mps' if torch.backends.mps.is_available() else 'cpu')
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        meta = checkpoint.get('preprocessing')
        if args.metadata:
            supplied = json.loads(args.metadata.read_text())
            if meta is not None and any(supplied.get(k) != v for k, v in meta.items()):
                raise ValueError('Supplied metadata conflicts with checkpoint preprocessing')
            meta = supplied
        if meta is None:
            raise ValueError('Legacy checkpoint lacks preprocessing; provide its matching --metadata meta.json')
        validate_preprocessing(meta)
        threshold = load_threshold(args.config)
        model = SimpleCNN(**checkpoint.get('model_config', {})).to(device)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        audio, sr = sf.read(str(args.audio), dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if not np.all(np.isfinite(audio)):
            raise ValueError('Audio contains nonfinite samples')
        duration = len(audio) / sr
        if sr != meta['sr']:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=meta['sr'])
        timeline = []
        for i, epoch in enumerate(epoch_audio(audio, meta['sr'], meta['epoch_sec'])):
            feature = torch.from_numpy(audio_to_mel(epoch, meta))[None, None].to(device)
            prob = torch.sigmoid(model(feature)).item()
            timeline.append(dict(start=i * meta['epoch_sec'],
                                 end=min((i + 1) * meta['epoch_sec'], duration),
                                 prob=prob, label=int(prob >= threshold)))
        regions = postprocess_events(timeline)
        positive_seconds = sum(row['end'] - row['start'] for row in timeline if row['label'])
        print('\n===== AUDIO WINDOW REPORT =====')
        print(f'Recording duration: {duration:.2f} seconds')
        print(f'Positive windows: {sum(row["label"] for row in timeline)} / {len(timeline)}')
        print(f'Time in positive windows: {positive_seconds:.2f} seconds ({positive_seconds / duration:.1%})')
        print(f'Positive regions: {len(regions)}')
        print('AHI and severity are not estimated: window labels do not provide individual event counts or sleep time.')
        for i, region in enumerate(regions, 1):
            print(f'{i:3d}. {region["start"]:.1f}s - {region["end"]:.1f}s | peak p={region["max_prob"]:.3f}')
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, f'Inference failed: {exc}\n')


if __name__ == '__main__':
    main()
