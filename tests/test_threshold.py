import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.cnn import SimpleCNN


class ThresholdTests(unittest.TestCase):
    def test_persist_threshold_and_reject_conflicting_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.zeros((6, 1, 16, 16), dtype='float32').tofile(path / 'X.npy')
            np.array([0, 1, 0, 1, 0, 1], dtype='int8').tofile(path / 'y.npy')
            meta = dict(total_epochs=6, n_mels=16, T=16, sr=4000,
                        splits=dict(train=[0, 1], val=[2, 3], test=[4, 5]))
            (path / 'meta.json').write_text(json.dumps(meta))
            model = SimpleCNN(base_filters=4)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
                model.fc.bias.fill_(1)
            checkpoint = dict(model_state=model.state_dict(), model_config=dict(base_filters=4))
            torch.save(checkpoint, path / 'model.pt')
            command = [sys.executable, '-B', str(ROOT / 'src/optimize_threshold.py'),
                       '--features_dir', str(path), '--checkpoint', str(path / 'model.pt'),
                       '--output', str(path / 'config.json')]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            config = (path / 'config.json').read_text()
            self.assertGreater(json.loads(config)['threshold'], 0)
            checkpoint['preprocessing'] = {'sr': 8000}
            torch.save(checkpoint, path / 'model.pt')
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('conflicts with checkpoint', result.stderr)
            self.assertEqual((path / 'config.json').read_text(), config)


if __name__ == '__main__':
    unittest.main()
