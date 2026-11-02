from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from infer_apnea import audio_to_mel, epoch_audio, postprocess_events
from make_dataset import compute_log_mel, build_subject_splits, RunningStats


class InferenceTests(unittest.TestCase):
    def test_running_statistics_match_numpy_across_batches(self):
        values = np.random.default_rng(12).normal(-45, 14, 101)
        stats = RunningStats()
        for batch in np.array_split(values, 7):
            stats.push(batch)
        mean, std = stats.get()
        self.assertAlmostEqual(mean, values.mean())
        self.assertAlmostEqual(std, values.std(ddof=1))

    def test_training_and_inference_features_match(self):
        audio = np.random.default_rng(42).normal(size=120000).astype('float32')
        meta = dict(sr=4000, n_mels=64, n_fft=512, hop_length=128,
                    train_mean=-45., train_std=14., epoch_sec=30.)
        expected = (compute_log_mel(audio, 4000, 64, 512, 128) + 45.) / 14.
        np.testing.assert_array_equal(audio_to_mel(audio, meta), expected)
        self.assertEqual(expected.shape, (64, 938))

    def test_single_positive_is_retained_and_gaps_not_merged(self):
        rows = [dict(start=0, end=30, prob=.8, label=1),
                dict(start=60, end=90, prob=.9, label=1)]
        self.assertEqual(len(postprocess_events(rows)), 2)
        rows[1]['start'] = 30
        self.assertEqual(len(postprocess_events(rows)), 1)

    def test_empty_audio_rejected_and_tail_padded(self):
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            list(epoch_audio(np.array([]), 4000))
        result = list(epoch_audio(np.ones(4000), 4000))
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 120000)
        self.assertEqual(np.count_nonzero(result[0]), 4000)

    def test_small_splits_stay_nonempty_and_disjoint(self):
        for n in (3, 9, 32):
            splits = build_subject_splits([Path(str(i)) for i in range(n)])
            self.assertTrue(all(splits))
            self.assertEqual(sum(map(len, splits)), len(set.union(*splits)))
        with self.assertRaises(ValueError):
            build_subject_splits([Path('a'), Path('b')])


if __name__ == '__main__':
    unittest.main()
