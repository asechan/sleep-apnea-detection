# Fix plan and acceptance checks

## 1. Data integrity and reproducibility

Implemented: shared strict annotation parsing, explicit synchronization offsets,
missing-annotation rejection, nonzero exit on epoch failures, feature byte-size and
split validation, hidden macOS sidecar exclusion, dependency manifest, regression tests.

Acceptance: both readers parse the same 24,457 annotations from 32 subjects;
existing truncated features are rejected before training; unverified alignment
cannot silently create new labels.

## 2. Consistent training and inference

Implemented: shared mel extraction, saved preprocessing and architecture metadata
in new checkpoints, explicit matching metadata for legacy checkpoints, training-only
class weights, nonempty subject splits, sample-rate checks, output overwrite guard,
strict feature-label counts, persisted validation threshold.

Acceptance: identical training/inference features for the same window; temporary
synthetic pipeline trains, reloads its checkpoint and runs inference.

## 3. Honest reporting

Implemented: report positive windows and contiguous regions, retain isolated
positives, use actual recording duration, reject empty audio. Removed unsupported
AHI and severity estimates. Individual event counting and sleep-time estimation
require a separately validated model/evaluation design; region count is not AHI.

## 4. Recover and evaluate the real dataset (complete)

- Alignment established from the [official APSAA release](https://zenodo.org/records/14096541)
  and the owner's confirmation that WAV/CSV files are unchanged. The release is
  synchronized: use zero additional offset and ignore the added sync MAT files.
  This corrects the earlier assumption that empty MAT files imply missing alignment.
- New manifest records offset_seconds=0 and alignment provenance per subject.
- Completed: generated a new manifest, epochs and features into fresh output directories.
  The external-drive dataset and existing checkpoints must remain intact.
- Check labels against known annotated intervals, feature sizes, split isolation,
  class counts, and preprocessing metadata before training.
- Retrain, tune on validation, and evaluate the untouched test subjects. The
  repaired-v1 run selected threshold 0.50 on validation data (F1 0.7417).
  Held-out window metrics are accuracy 0.6955, precision 0.6160, recall 0.5528,
  F1 0.5827, and ROC-AUC 0.7170; the macro-average of per-subject F1 is 0.5410.
  The old checkpoint's stored score was not used as evidence.

Current external artifacts: X.npy is 28,311,552 bytes; metadata requires
6,773,770,752 bytes. Existing epoch byte sizes and label counts match metadata.

## Verification commands

```bash
.venv/bin/python -B -m unittest discover -s tests -v
.venv/bin/python src/artifacts.py '/Volumes/Ultra Touch/apsaa dataset/data/features'
```

The second command is expected to fail clearly until features are regenerated.

## Progress on 2026-11-02

- Owner confirmed raw WAV and annotation CSV files were extracted unchanged.
- Rebuilt `data/repaired-v1/manifest.csv`: 32 subjects, 24,457 annotations;
  all manifest diagnostics are empty.
- Rebuilt 28,209 epochs with explicit zero additional offset. Verified every
  epoch file's byte size and label count. Removing the unsupported -699-second
  shift changes 266 window labels for subject 20210919A; other subjects' labels
  match their previous outputs.
- 19 regression tests pass. A temporary three-subject pipeline completed epoch
  labeling, feature extraction, one training epoch, checkpoint reload, and
  inference with a nondefault architecture. Threshold tuning also persisted a
  valid configuration in a separate temporary fixture.
- Full-dataset feature regeneration completed in `data/repaired-v1/features`:
  28,209 rows of shape (1, 64, 938), totaling 6,773,770,752 feature bytes.
- Full retraining completed on Apple MPS. The best checkpoint was selected at
  epoch 11 using validation F1 at threshold 0.5; threshold tuning remained on
  validation data only. Held-out evaluation covers four subjects and 3,471
  windows. Results are also saved in the ignored
  `evaluation/repaired-v1/metrics.json` and `predictions.csv` artifacts.

After feature validation, run a new training experiment without replacing the old model:

```bash
.venv/bin/python src/train_cnn.py \
  --features_dir data/repaired-v1/features \
  --checkpoint_dir checkpoints/repaired-v1 --epochs 25
.venv/bin/python src/optimize_threshold.py \
  --features_dir data/repaired-v1/features \
  --checkpoint checkpoints/repaired-v1/best.pt \
  --output checkpoints/repaired-v1/config.json
.venv/bin/python src/evaluate.py \
  --features_dir data/repaired-v1/features \
  --processed_dir data/repaired-v1/processed \
  --checkpoint checkpoints/repaired-v1/best.pt \
  --config checkpoints/repaired-v1/config.json \
  --output_dir evaluation/repaired-v1
.venv/bin/python src/infer_apnea.py your_audio.wav \
  --checkpoint checkpoints/repaired-v1/best.pt \
  --config checkpoints/repaired-v1/config.json
```

Final artifact validation passed: all 6.77 GB of features contain finite values;
recomputed features for the first epoch of every subject exactly match the saved
arrays using inference preprocessing (maximum error 0). Subject splits remain
25/3/4 with no overlap. A real-data batch completed forward, backward and optimizer
steps. Detailed results are in `data/repaired-v1/validation.json` (ignored by Git).
The external originals and existing checkpoints were not modified.
