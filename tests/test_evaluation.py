import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evaluate import metrics, subject_rows
from models.cnn import SimpleCNN


class EvaluationTests(unittest.TestCase):
    def test_known_confusion_and_threshold_boundary(self):
        result = metrics([0, 0, 1, 1], [.1, .8, .3, .5], .5)
        self.assertEqual(result['confusion_matrix'], [[1, 1], [1, 1]])
        self.assertEqual(result['precision'], .5)
        self.assertEqual(result['recall'], .5)
        self.assertEqual(result['f1'], .5)

    def test_single_class_auc_is_json_null(self):
        result = metrics([0, 0], [.1, .9], .5)
        self.assertIsNone(result['roc_auc'])
        json.dumps(result, allow_nan=False)

    def test_invalid_probabilities_rejected(self):
        for probabilities in ([float('nan')], [1.1], [-.1]):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                metrics([0], probabilities, .5)

    def test_cli_evaluates_only_test_and_checks_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = root / 'features'
            features.mkdir()
            processed = root / 'processed'
            for subject in ('a', 'b', 'c'):
                path = processed / subject
                path.mkdir(parents=True)
                (path / 'epoch_meta.json').write_text(json.dumps({'n_epochs': 2}))
            meta = dict(total_epochs=6, n_mels=16, T=16, sr=4000,
                        splits=dict(train=[0, 1], val=[2, 3], test=[4, 5],
                                    train_subjects=['a'], val_subjects=['b'], test_subjects=['c']))
            (features / 'meta.json').write_text(json.dumps(meta))
            np.zeros((6, 1, 16, 16), dtype='float32').tofile(features / 'X.npy')
            np.array([0, 0, 0, 0, 0, 1], dtype='int8').tofile(features / 'y.npy')
            model = SimpleCNN(base_filters=4)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
                model.fc.bias.fill_(1)
            checkpoint = root / 'model.pt'
            torch.save(dict(model_state=model.state_dict(), model_config={'base_filters': 4},
                            preprocessing={'sr': 4000}, epoch=1), checkpoint)
            config = root / 'config.json'
            config.write_text(json.dumps(dict(threshold=.5, checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())))
            command = [sys.executable, '-B', str(ROOT / 'src/evaluate.py'), '--features_dir', str(features),
                       '--processed_dir', str(processed), '--checkpoint', str(checkpoint), '--config', str(config),
                       '--output_dir', str(root / 'report'), '--cpu']
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root / 'report/metrics.json').read_text())
            self.assertEqual(report['overall']['confusion_matrix'], [[0, 1], [0, 1]])
            self.assertEqual(list(report['per_subject']), ['c'])
            self.assertEqual(report['overall']['count'], 2)
            self.assertEqual(len((root / 'report/predictions.csv').read_text().splitlines()), 3)
            config.write_text(json.dumps(dict(threshold=.5, checkpoint_sha256='wrong')))
            command[command.index('--output_dir') + 1] = str(root / 'bad-report')
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('exact checkpoint', result.stderr)
            self.assertFalse((root / 'bad-report').exists())
            meta['splits']['test_subjects'] = ['wrong']
            with self.assertRaisesRegex(ValueError, 'Subject ordering'):
                subject_rows(processed, meta)


if __name__ == '__main__':
    unittest.main()
