import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


class TrainingTests(unittest.TestCase):
    def test_training_persists_contract_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            meta = dict(total_epochs=6, n_mels=16, T=16, sr=4000, n_fft=128,
                        hop_length=128, train_mean=0., train_std=1., epoch_sec=.5,
                        splits=dict(train=[0, 1], val=[2, 3], test=[4, 5]))
            (path / 'meta.json').write_text(json.dumps(meta))
            np.random.default_rng(42).normal(size=(6, 1, 16, 16)).astype('float32').tofile(path / 'X.npy')
            np.array([0, 1, 0, 1, 0, 1], dtype='int8').tofile(path / 'y.npy')
            command = [sys.executable, '-B', str(ROOT / 'src/train_cnn.py'), '--features_dir', str(path),
                       '--checkpoint_dir', str(path / 'checkpoints'), '--epochs', '1', '--num_workers', '0',
                       '--batch_size', '2', '--base_filters', '4', '--cpu']
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn('TEST :', result.stdout)
            history = json.loads((path / 'checkpoints/history.json').read_text())
            self.assertEqual(len(history), 1)
            checkpoint_path = path / 'checkpoints/best.pt'
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            self.assertEqual(checkpoint['model_config']['base_filters'], 4)
            self.assertEqual(checkpoint['preprocessing']['train_std'], 1.)
            previous = checkpoint_path.read_bytes()
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('fresh --checkpoint_dir', result.stderr)
            self.assertEqual(checkpoint_path.read_bytes(), previous)


if __name__ == '__main__':
    unittest.main()
