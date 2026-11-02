# Sleep Apnea Detection

Research pipeline for detecting apnea-related patterns in respiratory audio.
The project converts annotated recordings into fixed 30-second windows,
extracts log-mel spectrogram features, trains a compact PyTorch CNN, and
reports window-level predictions for new recordings.

> **Scope:** This is a research and engineering project, not a medical device.
> It does not diagnose sleep apnea, estimate clinical AHI, determine sleep
> stages, or provide a medical severity classification. Model outputs require
> clinical interpretation and independent validation before any real-world use.

## Contents

- [Highlights](#highlights)
- [System architecture](#system-architecture)
- [Technology stack](#technology-stack)
- [Data and feature contract](#data-and-feature-contract)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [End-to-end workflow](#end-to-end-workflow)
- [Running inference](#running-inference)
- [Training and evaluation](#training-and-evaluation)
- [Performance](#performance)
- [Data integrity and reproducibility](#data-integrity-and-reproducibility)
- [Limitations](#limitations)
- [Development and testing](#development-and-testing)
- [Responsible use](#responsible-use)
- [License](#license)

## Highlights

- Strict APSAA annotation parsing with explicit timestamp and duration handling.
- Explicit synchronization provenance instead of guessing undocumented offsets.
- Subject-level train/validation/test splits to reduce subject leakage.
- Shared preprocessing between feature generation, training, threshold tuning,
  evaluation, and inference.
- Memory-mapped feature storage for large datasets.
- Artifact validation for dimensions, file sizes, labels, and split coverage.
- Compact 2D CNN with configurable channel width and dropout.
- Validation-only threshold selection with checkpoint identity checks.
- Held-out evaluation with overall and per-subject metrics.
- Inference output based on positive windows and contiguous positive regions.
- Regression tests covering data integrity, inference, thresholding, evaluation,
  and training contracts.

## System architecture

```text
                  APSAA recording package
                 /          |             \
                /           |              \
        Audio WAV      Annotation CSV     Sync metadata
             |               |                 |
             +---------------+-----------------+
                             |
                    data-sync-apsaa.py
                             |
                    subject manifest.csv
                             |
                    epoch_labeler.py
                             |
              30-second epochs + binary labels
                             |
                     make_dataset.py
                             |
       log-mel features (X.npy) + labels (y.npy)
                             |
                  artifacts.py validation
                             |
                       train_cnn.py
                             |
                 PyTorch checkpoint (*.pt)
                    /                    \
                   /                      \
       optimize_threshold.py          evaluate.py
          validation threshold       held-out test report
                   |
             infer_apnea.py
                   |
       positive windows and regions
```

### Runtime data flow

1. The synchronization stage discovers subject audio, annotation, and optional
   synchronization files and writes a manifest.
2. The labeling stage maps annotation intervals onto fixed 30-second audio
   epochs. Missing or malformed annotations fail loudly rather than silently
   producing negative labels.
3. The feature stage computes log-mel spectrograms using the same parameters
   later used by inference.
4. The trainer reads memory-mapped arrays, trains on subject-isolated splits,
   and stores the model architecture and preprocessing contract in the
   checkpoint.
5. Threshold tuning selects a classification threshold using validation data
   only.
6. Evaluation applies the frozen checkpoint and threshold once to held-out
   subjects and writes reproducible JSON and CSV reports.
7. Inference resamples new audio when necessary, pads the final partial window,
   predicts each window, and merges adjacent positive windows into regions.

## Technology stack

| Area | Technology |
|---|---|
| Language | Python 3.13 (verified local environment) |
| Deep learning | PyTorch 2.10 |
| Model execution | CPU, Apple Metal Performance Shaders (MPS), or CUDA |
| Audio I/O | `soundfile` |
| Audio resampling and mel features | `librosa` |
| Numerical computing | NumPy |
| Tabular data | pandas |
| MATLAB synchronization files | SciPy |
| Metrics | scikit-learn |
| Progress reporting | tqdm |
| Tests | Python `unittest` |
| Artifact format | JSON, CSV, raw binary memory maps |

Installable versions are pinned in [requirements.txt](requirements.txt).

## Data and feature contract

The pipeline currently targets the APSAA dataset. Raw recordings and generated
data are intentionally excluded from Git because of their size and provenance.
The official APSAA release used for the repaired artifacts is already aligned;
the rebuilt manifest records zero additional offset with provenance.

### Audio and windows

- Default sample rate: **4,000 Hz**
- Window duration: **30 seconds**
- Window labels: binary, derived from annotation overlap
- Final inference windows: zero-padded when shorter than 30 seconds
- Multi-channel input: averaged to mono for inference
- Non-finite or empty audio: rejected

### Feature parameters

The shared preprocessing contract contains:

- 64 mel bands
- `n_fft=512`
- `hop_length=128`
- log-power mel spectrogram
- training-set mean and standard deviation normalization

The training contract is saved in each new checkpoint. Inference refuses a
missing or conflicting preprocessing contract. This prevents a model trained
with one spectrogram layout from silently receiving a different layout.

The generated feature arrays use raw binary memory maps despite the historical
`.npy` filenames:

```text
X.npy  float32 feature values: (total_epochs, 1, n_mels, T)
y.npy  int8 binary labels:     (total_epochs,)
meta.json feature dimensions, statistics, splits, and preprocessing metadata
```

Do not load these files with `numpy.load`; use the project metadata and loaders.
`src/artifacts.py` verifies expected byte sizes before training.

## Repository structure

```text
.
├── models/
│   └── cnn.py                  # Compact convolutional classifier
├── src/
│   ├── annotations.py          # Strict shared APSAA annotation reader
│   ├── artifacts.py            # Feature and split integrity validation
│   ├── data-sync-apsaa.py      # Manifest and subject metadata generation
│   ├── epoch_labeler.py        # Audio epoch creation and labeling
│   ├── evaluate.py             # Frozen held-out evaluation
│   ├── infer_apnea.py          # Audio inference and region reporting
│   ├── make_dataset.py         # Feature extraction and subject splits
│   ├── optimize_threshold.py   # Validation threshold selection
│   ├── preprocessing.py        # Shared mel and normalization contract
│   └── train_cnn.py            # Training loop and checkpoint management
├── tests/
│   ├── test_data_integrity.py
│   ├── test_evaluation.py
│   ├── test_inference.py
│   ├── test_threshold.py
│   └── test_training.py
├── requirements.txt
├── FIX_PLAN.md                 # Repair history and validation notes
└── README.md
```

Generated directories such as `data/repaired-v1/`, `checkpoints/`, and
`evaluation/` are ignored by Git. Keep raw recordings and large generated
artifacts outside the repository when possible.

## Installation

The verified development environment uses Python 3.13:

```bash
git clone https://github.com/asechan/sleep-apnea-detection.git
cd sleep-apnea-detection

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Apple Silicon users can use MPS when supported by their PyTorch installation.
The scripts automatically prefer CUDA, then MPS, then CPU unless `--cpu` is
provided where supported.

## End-to-end workflow

### 1. Build a manifest

Raw data is not included in this repository. The following example assumes the
official APSAA files are available locally:

```bash
python src/data-sync-apsaa.py \
  --raw_dir '/path/to/apsaa/data/raw' \
  --out_csv data/manifest_apsaa.csv \
  --diag data/diagnostics_apsaa.json \
  --processed_dir data/processed \
  --already_synchronized
```

Use `--already_synchronized` only when the WAV and annotation files are
unchanged from the synchronized official release. For modified recordings,
provide a verified signed `offset_seconds` for every subject. The project uses
the convention:

```text
audio_onset = annotation_onset + offset_seconds
```

Undocumented delay vectors, empty synchronization arrays, and missing
annotations are rejected rather than guessed.

### 2. Generate labeled epochs

```bash
python src/epoch_labeler.py \
  --manifest data/manifest_apsaa.csv \
  --processed_dir data/processed
```

### 3. Generate and validate features

```bash
python src/make_dataset.py \
  --processed_dir data/processed \
  --out_dir data/features

python src/artifacts.py data/features
```

Feature generation requires multiple subjects and matching sample rates. It
refuses to overwrite an existing output directory; use a new destination when
regenerating artifacts.

### 4. Train a model

```bash
python src/train_cnn.py \
  --features_dir data/features \
  --checkpoint_dir checkpoints/experiment-1 \
  --epochs 25
```

Training selects the best checkpoint using validation F1 at the default
threshold of 0.5. It writes `best.pt`, `last.pt`, `history.json`, and run
metadata. Existing checkpoints are protected from accidental overwrite.

### 5. Tune the threshold

```bash
python src/optimize_threshold.py \
  --features_dir data/features \
  --checkpoint checkpoints/experiment-1/best.pt \
  --output checkpoints/experiment-1/config.json
```

Threshold selection uses the validation split only and stores the SHA-256 hash
of the checkpoint. Evaluation and inference reject a configuration belonging to
a different checkpoint.

### 6. Evaluate held-out subjects

```bash
python src/evaluate.py \
  --features_dir data/features \
  --processed_dir data/processed \
  --checkpoint checkpoints/experiment-1/best.pt \
  --config checkpoints/experiment-1/config.json \
  --output_dir evaluation/experiment-1
```

The evaluator writes:

- `metrics.json`: overall metrics, per-subject metrics, baselines, threshold,
  checkpoint identity, and confusion matrices.
- `predictions.csv`: global index, subject, epoch index, label, probability,
  and thresholded prediction.

Confusion matrices use the order `[[TN, FP], [FN, TP]]`. Do not use the test
split for model selection, threshold tuning, or repeated experimentation.

## Running inference

For a checkpoint containing its preprocessing contract:

```bash
python src/infer_apnea.py recording.wav \
  --checkpoint checkpoints/experiment-1/best.pt \
  --config checkpoints/experiment-1/config.json
```

Use `--cpu` to force CPU execution:

```bash
python src/infer_apnea.py recording.wav \
  --checkpoint checkpoints/experiment-1/best.pt \
  --config checkpoints/experiment-1/config.json \
  --cpu
```

Legacy checkpoints that do not contain preprocessing metadata require the
matching training metadata:

```bash
python src/infer_apnea.py recording.wav \
  --checkpoint checkpoints/best.pt \
  --metadata /path/to/original/features/meta.json \
  --config config.json
```

The report includes recording duration, number of positive windows, time in
positive windows, and contiguous positive regions with their peak probability.
Regions are not individual apnea events. The output intentionally does not
calculate AHI or severity.

## Performance

### Verified repaired-v1 experiment

The repaired local artifacts contained:

- 32 subjects
- 28,209 labeled 30-second epochs
- 25 training subjects, 3 validation subjects, 4 test subjects
- Feature shape `(28,209, 1, 64, 938)`
- 6,773,770,752 bytes of finite feature data
- 24,457 parsed annotations

The model was trained on Apple MPS. The best checkpoint was selected at epoch
11. The threshold was selected on validation data and was **0.50**, producing
validation F1 **0.7417**.

Held-out test results:

| Metric | Value |
|---|---:|
| Windows | 3,471 |
| Accuracy | 0.6955 |
| Precision | 0.6160 |
| Recall | 0.5528 |
| F1 | 0.5827 |
| ROC-AUC | 0.7170 |
| Macro per-subject F1 | 0.5410 |

The held-out confusion matrix was `[[1676, 460], [597, 738]]`.
The detailed report and predictions are generated locally under
`evaluation/repaired-v1/` and are intentionally not committed because they
depend on large local artifacts.

These are window-level research metrics from one split. They are not estimates
of clinical sensitivity, patient-level generalization, or diagnostic accuracy.

### Computational characteristics

- Feature generation is the most storage-intensive stage.
- Features and labels are memory-mapped to avoid loading the complete dataset
  into RAM.
- Training and evaluation process data in batches.
- GPU acceleration is supported through CUDA and Apple MPS.
- Inference processes one fixed-size window at a time and pads only the final
  partial window.
- Reproducibility metadata records the seed, model configuration, preprocessing
  contract, feature metadata hash, and checkpoint hash.

Exact runtime depends on the processor, accelerator, storage, and dataset
location; this repository does not claim a fixed real-time latency benchmark.

## Data integrity and reproducibility

The pipeline is intentionally defensive:

- Annotation schemas, timestamps, durations, and event names are validated.
- Missing annotations cannot silently become all-negative labels.
- Subject splits are checked for empty, overlapping, out-of-range, or missing
  indices.
- Feature file sizes must match metadata exactly.
- Labels must be binary and aligned with feature rows.
- Checkpoints store preprocessing and model configuration.
- Threshold files are tied to the exact checkpoint hash.
- Existing outputs are not overwritten accidentally.
- Test data is excluded from training and threshold selection.

For a reproducible local run, preserve:

1. The raw dataset version and provenance.
2. The manifest and diagnostics files.
3. The feature metadata and validation output.
4. The checkpoint, run metadata, and threshold configuration.
5. The evaluation report and command-line arguments.

See [FIX_PLAN.md](FIX_PLAN.md) for the repair history and known decisions.

## Limitations

- The dataset and generated artifacts are not distributed in this repository.
- The reported model is trained and evaluated on a single subject split.
- The test set is small at four subjects.
- Window classification does not identify individual apnea events.
- No sleep/wake classifier is included, so sleep time is unknown.
- No clinical calibration, external validation, demographic analysis, or
  prospective study has been performed.
- Audio quality, microphone placement, environment, and sensor differences can
  affect predictions.
- A positive window is a model classification, not a diagnosis.
- Legacy checkpoints may not be compatible with the repaired preprocessing
  pipeline unless their matching metadata is supplied.

## Development and testing

Run the complete regression suite:

```bash
python -B -m unittest discover -s tests -v
```

Validate a generated feature directory:

```bash
python src/artifacts.py data/features
```

Check for whitespace errors before committing:

```bash
git diff --check
```

Tests use temporary fixtures and synthetic data where possible. Real-dataset
validation requires the external APSAA artifacts and is not part of the unit
test suite.

## Responsible use

Do not use this project as a substitute for polysomnography, a clinician, or
medical advice. Do not make treatment or diagnosis decisions from its output.
Any future clinical use would require validated labels, representative
multi-site data, calibration, prospective testing, privacy controls, and
appropriate regulatory review.

## License

See [LICENSE](LICENSE) if present in the checkout. If no license file is
included in a distribution, obtain the repository owner's permission before
redistributing or using the code commercially.
