import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from annotations import read_events
from artifacts import validate_features
from epoch_labeler import choose_time_delay_sign, create_epochs_and_labels


class DataIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def csv(self, text):
        path = self.root / "annotations.csv"
        path.write_text(text)
        return path

    def test_apsaa_timestamp_and_manifest_agree(self):
        path = self.csv("Event_Name,Start_Time,Duration\nHypopnea,01:02:03,12.5\n")
        events = read_events(path)
        self.assertEqual(events, [{"onset": 3723., "duration": 12.5, "label": "Hypopnea"}])
        spec = importlib.util.spec_from_file_location("manifest", Path(__file__).resolve().parents[1] / "src/data-sync-apsaa.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.read_annotations_csv(path)["events"], events)

    def test_bad_annotations_are_errors(self):
        for text in ("foo,duration,event\n1,10,apnea\n",
                     "start,duration,event\n00:99:00,10,apnea\n",
                     "start,duration,event\n10,,apnea\n",
                     "start,duration,event\n10,0,apnea\n",
                     "start,duration,event\n10,10,\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                read_events(self.csv(text))

    def test_missing_annotations_do_not_create_labels(self):
        with self.assertRaisesRegex(ValueError, "Missing annotations"):
            create_epochs_and_labels({"subject": "a", "audio_path": "unused.wav",
                                      "audio_sr": 4000, "audio_duration_sec": 30}, self.root)
        self.assertFalse((self.root / "a/labels.csv").exists())

    def test_sync_never_guesses_from_vectors_or_sample_rate(self):
        for sync in ({}, {"fs": 4000}, {"t_sync": []}, {"time_delay": [699, 190]},
                     {"offset_seconds": [1, 2]}, {"offset_seconds": float("nan")}):
            with self.subTest(sync=sync), self.assertRaises(ValueError):
                choose_time_delay_sign(sync, [10], 60)
        self.assertEqual(choose_time_delay_sign({"offset_seconds": -5}, [10], 60)[:2], (-5, "+"))
        self.assertEqual(choose_time_delay_sign({"offset_seconds": 0}, [10], 60)[:2], (0, "+"))

    def test_manifest_alignment_overrides_ambiguous_mat(self):
        wav = self.root / "audio.wav"
        sf.write(wav, np.zeros(120000, dtype='float32'), 4000)
        sync = self.root / "sync.mat"
        savemat(sync, {"time_delay": [699, 190]})
        ann = self.csv("Event_Name,Start_Time,Duration\nHypopnea,00:00:10,12\n")
        out, meta = create_epochs_and_labels(
            dict(subject='aligned', audio_path=str(wav), audio_sr=4000,
                 audio_duration_sec=30, annotations_csv=str(ann), sync_mat=str(sync),
                 offset_seconds=0, alignment_source='original synchronized release'), self.root)
        self.assertEqual(meta['chosen_time_delay'], 0)
        self.assertEqual(meta['mapped_events_preview'][0]['audio_onset'], 10)
        self.assertTrue((out / 'labels.csv').read_text().strip().endswith(',1'))


    def features(self):
        meta = {"total_epochs": 3, "n_mels": 2, "T": 4,
                "splits": {"train": [0], "val": [1], "test": [2]}}
        (self.root / "meta.json").write_text(json.dumps(meta))
        np.zeros((3, 1, 2, 4), dtype="float32").tofile(self.root / "X.npy")
        np.array([0, 1, 0], dtype="int8").tofile(self.root / "y.npy")
        return meta

    def test_valid_features(self):
        meta = self.features()
        self.assertEqual(validate_features(self.root), meta)

    def test_truncated_features_explain_recovery(self):
        self.features()
        (self.root / "X.npy").write_bytes(b"abcd")
        with self.assertRaisesRegex(ValueError, "expected 96 bytes.*regenerate"):
            validate_features(self.root)

    def test_invalid_splits(self):
        for splits in ({"train": [0], "val": [], "test": [1, 2]},
                       {"train": [0], "val": [0], "test": [2]},
                       {"train": [0], "val": [1], "test": [3]}):
            meta = self.features()
            meta["splits"] = splits
            (self.root / "meta.json").write_text(json.dumps(meta))
            with self.subTest(splits=splits), self.assertRaises(ValueError):
                validate_features(self.root)


if __name__ == "__main__":
    unittest.main()
