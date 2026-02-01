#!/usr/bin/env python3
"""
epoch_labeler.py (final robust version)

- Parses HH:MM:SS annotation times
- Loads sync_output.mat and auto-detects whether to apply +time_delay or -time_delay
- Robust numeric coercion for manifest values (lists/tuples/ndarrays -> scalars)
- Epochs audio into fixed-length windows (default 30s) and labels epochs
- POSITIVE labels: Apnea + Hypopnea only (desaturation NOT considered positive)

Usage:
    python src/epoch_labeler.py --manifest data/manifest_apsaa.csv --processed_dir data/processed --epoch_sec 30 --overlap 0.0
"""
import argparse
from pathlib import Path
import json
import numpy as np
import soundfile as sf
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm
import math
import warnings

# ---------- small helper: safe numeric coercion ----------
import numpy as _np

def _coerce_to_scalar(val):
    """
    Convert val into a python float if possible.
    Accepts numbers, numpy scalars, 1-element lists/tuples/ndarrays, or string-convertible numbers.
    Returns (float or None).
    """
    if val is None:
        return None
    # numeric
    if isinstance(val, (int, float, _np.integer, _np.floating)):
        return float(val)
    # numpy scalar
    if isinstance(val, _np.generic):
        try:
            return float(val.item())
        except Exception:
            pass
    # list/tuple/ndarray: try first numeric element
    if isinstance(val, (list, tuple, _np.ndarray)):
        arr = _np.asarray(val)
        if arr.size == 0:
            return None
        # if single element
        if arr.size == 1:
            try:
                return float(arr.flatten()[0])
            except Exception:
                return None
        # multiple elements: pick first convertible element
        for x in arr.flatten():
            try:
                return float(x)
            except Exception:
                continue
        return None
    # string convertible?
    if isinstance(val, str):
        try:
            return float(val.strip())
        except Exception:
            return None
    return None

def to_number(val):
    """Alias to coercion - returns float or None."""
    return _coerce_to_scalar(val)

# ---------- config ----------
# POSITIVE: only apnea + hypopnea (user requested)
POSITIVE_KEYWORDS = ["apnea", "apnoea", "hypopnea", "hypopnoea"]

# ---------- Helpers ----------

def safe_load_mat(path: Path):
    """Load .mat and convert small arrays to python scalars/lists where reasonable."""
    try:
        m = loadmat(str(path))
    except Exception as e:
        return {"error": str(e)}
    keys = [k for k in m.keys() if not k.startswith("__")]
    out = {}
    for k in keys:
        v = m[k]
        try:
            if isinstance(v, np.ndarray):
                squeezed = np.squeeze(v)
                if np.isscalar(squeezed):
                    out[k] = float(squeezed)
                else:
                    try:
                        out[k] = [float(x) for x in np.atleast_1d(squeezed)]
                    except Exception:
                        try:
                            out[k] = squeezed.tolist()
                        except Exception:
                            out[k] = str(type(v))
            elif isinstance(v, (int, float, np.integer, np.floating)):
                out[k] = float(v)
            else:
                try:
                    out[k] = str(v)
                except Exception:
                    out[k] = "unreadable"
        except Exception:
            try:
                out[k] = str(v)
            except Exception:
                out[k] = "unreadable"
    return out

def parse_hhmmss_to_seconds(s):
    """Accept 'HH:MM:SS', 'MM:SS', or numeric strings; return seconds or None."""
    if pd.isna(s):
        return None
    if isinstance(s, (int, float, np.floating, np.integer)):
        return float(s)
    s = str(s).strip()
    if ":" in s:
        parts = s.split(":")
        try:
            parts = [float(p) for p in parts]
        except Exception:
            return None
        if len(parts) == 3:
            return parts[0]*3600 + parts[1]*60 + parts[2]
        elif len(parts) == 2:
            return parts[0]*60 + parts[1]
        else:
            try:
                return float(parts[0])
            except Exception:
                return None
    else:
        try:
            return float(s)
        except Exception:
            return None

def read_annotations_csv(csv_path: Path):
    """
    Robustly read an annotation CSV and return list of dicts:
      [{onset: seconds, duration: seconds, label: str}, ...]
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, encoding="latin1")
    cols = {c.lower(): c for c in df.columns}
    onset_col = None
    dur_col = None
    label_col = None
    for cand in ["start","start_time","onset","time","begin","start_time"]:
        if cand in cols:
            onset_col = cols[cand]; break
    for cand in ["duration","dur","length"]:
        if cand in cols:
            dur_col = cols[cand]; break
    for cand in ["event","event_name","type","label","annotation","description"]:
        if cand in cols:
            label_col = cols[cand]; break

    events = []
    for _, row in df.iterrows():
        try:
            raw_on = row[onset_col] if onset_col else None
            onset = parse_hhmmss_to_seconds(raw_on)
            if onset is None:
                continue
            dur = 0.0
            if dur_col:
                dv = row[dur_col]
                try:
                    dur = float(dv)
                except Exception:
                    dur = 0.0
            lbl = ""
            if label_col:
                lbl = str(row[label_col])
            else:
                for c in df.columns:
                    if df[c].dtype == object:
                        lbl = str(row[c]); break
            events.append({"onset": float(onset), "duration": float(dur), "label": lbl})
        except Exception:
            continue
    return events

def choose_time_delay_sign(sync_dict: dict, ann_onsets: list, audio_duration: float):
    """
    Find candidate numeric time_delay and choose + or - sign by maximizing onsets inside audio.
    Returns (td_number, sign, stats).
    sign = "+" or "-" or None.
    """
    if not sync_dict or not isinstance(sync_dict, dict):
        return 0.0, None, {"reason": "no_sync"}

    lower_map = {k.lower(): v for k,v in sync_dict.items()}

    # candidate keys to try
    candidates = []
    for key_name in ["time_delay", "timedelay", "timeoffset", "offset", "time_offset", "sync_offset", "audio_start", "annotation_start", "t0"]:
        if key_name.lower() in lower_map:
            candidates.append(lower_map[key_name.lower()])

    # fallback: any numeric scalar value
    if not candidates:
        for k, v in lower_map.items():
            num = to_number(v)
            if num is not None:
                candidates.append(v)
                break

    if not candidates:
        return 0.0, None, {"reason": "no_numeric_candidate"}

    td_num = to_number(candidates[0])
    if td_num is None:
        return 0.0, None, {"reason": "candidate_not_numeric"}

    def score_for(sign):
        mapped = [t + (td_num if sign=="+" else -td_num) for t in ann_onsets]
        inside = sum(1 for m in mapped if (0.0 <= m <= audio_duration))
        negative = sum(1 for m in mapped if m < 0.0)
        too_far = sum(1 for m in mapped if m > audio_duration)
        return {"inside": inside, "negative": negative, "too_far": too_far, "mapped_sample_count": len(mapped)}

    plus_stats = score_for("+")
    minus_stats = score_for("-")

    # prefer sign with more inside hits; tie-breakers: fewer negatives, fewer too_far
    if plus_stats["inside"] > minus_stats["inside"]:
        return td_num, "+", {"plus": plus_stats, "minus": minus_stats}
    if minus_stats["inside"] > plus_stats["inside"]:
        return td_num, "-", {"plus": plus_stats, "minus": minus_stats}
    if plus_stats["negative"] < minus_stats["negative"]:
        return td_num, "+", {"plus": plus_stats, "minus": minus_stats}
    if minus_stats["negative"] < plus_stats["negative"]:
        return td_num, "-", {"plus": plus_stats, "minus": minus_stats}
    # default to "+"
    return td_num, "+", {"plus": plus_stats, "minus": minus_stats, "note": "tie_default_plus"}

def map_ann_to_audio_time(ann_onset: float, td: float, sign):
    if sign is None:
        return float(ann_onset)
    if sign == "+":
        return float(ann_onset + td)
    else:
        return float(ann_onset - td)

def is_positive_event(label: str) -> bool:
    """Return True if label matches apnea/hypopnea (case-insensitive)"""
    if not label:
        return False
    lbl = str(label).lower()
    for k in POSITIVE_KEYWORDS:
        if k in lbl:
            return True
    return False

# ---------- Main processing (safe) ----------

def create_epochs_and_labels(subject_meta: dict, processed_dir: str, epoch_sec: float = 30.0, overlap: float = 0.0):
    subj = subject_meta.get("subject", "unknown")
    aud_p = Path(subject_meta.get("audio_path"))

    # ---- defensive coercion: ensure manifest numeric fields are scalars ----
    if "audio_sr" in subject_meta:
        coerced = _coerce_to_scalar(subject_meta["audio_sr"])
        if coerced is not None:
            subject_meta["audio_sr"] = coerced
    if "audio_duration_sec" in subject_meta:
        coerced = _coerce_to_scalar(subject_meta["audio_duration_sec"])
        if coerced is not None:
            subject_meta["audio_duration_sec"] = coerced
    # coerce any other tuple/list-like manifest fields to scalars defensively
    for k, v in list(subject_meta.items()):
        if isinstance(v, (list, tuple, _np.ndarray)):
            c = _coerce_to_scalar(v)
            if c is not None:
                subject_meta[k] = c

    # Defensive casts for sr/duration
    try:
        sr = int(float(subject_meta["audio_sr"]))
    except Exception:
        raise ValueError(f"Invalid sample rate for subject {subj}: {subject_meta.get('audio_sr')}")
    try:
        duration = float(subject_meta["audio_duration_sec"])
    except Exception:
        raise ValueError(f"Invalid audio_duration_sec for subject {subj}: {subject_meta.get('audio_duration_sec')}")

    out_dir = Path(processed_dir) / subj
    out_dir.mkdir(parents=True, exist_ok=True)

    # load sync info if available
    sync_info = {}
    if subject_meta.get("sync_mat") and (not pd.isna(subject_meta.get("sync_mat"))):
        sync_info = safe_load_mat(Path(subject_meta["sync_mat"]))

    # load annotations
    ann_events = []
    if subject_meta.get("annotations_csv") and (not pd.isna(subject_meta.get("annotations_csv"))):
        ann_events = read_annotations_csv(Path(subject_meta["annotations_csv"]))

    ann_onsets = [e["onset"] for e in ann_events]

    # choose time delay and sign
    chosen_td = 0.0
    chosen_sign = None
    stats = {}
    if sync_info and isinstance(sync_info, dict):
        chosen_td, chosen_sign, stats = choose_time_delay_sign(sync_info, ann_onsets, duration)
    else:
        chosen_td, chosen_sign, stats = 0.0, None, {"reason": "no_sync"}

    # map events
    mapped_events = []
    for e in ann_events:
        mapped_time = map_ann_to_audio_time(e["onset"], chosen_td, chosen_sign)
        mapped_flag = chosen_sign is not None
        mapped_events.append({"orig_onset": e["onset"], "duration": e["duration"], "label": e["label"], "audio_onset": mapped_time, "mapped": mapped_flag})

    # read audio
    data, file_sr = sf.read(str(aud_p), dtype="float32")
    if file_sr is not None:
        file_sr = int(file_sr)
    if file_sr != sr:
        # if mismatch, trust file sample rate
        sr = int(file_sr)

    if data is None:
        raise RuntimeError(f"Could not read audio for subject {subj}: {aud_p}")

    if data.ndim > 1:
        data = np.ascontiguousarray(data.mean(axis=1).astype("float32"))
    else:
        data = np.ascontiguousarray(data.astype("float32"))

    # epoch sizes (ints)
    samples_per_epoch = int(round(epoch_sec * sr))
    if samples_per_epoch <= 0:
        raise ValueError(f"Computed samples_per_epoch <= 0 for subj {subj}, epoch_sec={epoch_sec}, sr={sr}")
    hop = int(round(samples_per_epoch * (1.0 - float(overlap))))
    if hop <= 0:
        raise ValueError(f"Computed hop <= 0 for subj {subj}. Check overlap parameter (got overlap={overlap}).")

    # compute n_epochs
    data_len = int(len(data))
    if data_len < samples_per_epoch:
        n_epochs = 1
    else:
        n_epochs = int(math.ceil((data_len - samples_per_epoch) / hop)) + 1

    # create memmap
    epochs_path = out_dir / "epochs.npy"
    try:
        fp = np.memmap(str(epochs_path), dtype="float32", mode="w+", shape=(int(n_epochs), int(samples_per_epoch)))
    except Exception as e:
        raise RuntimeError(f"Failed to create memmap for subject {subj} with shape ({n_epochs},{samples_per_epoch}): {e}")

    labels = []
    for i in range(int(n_epochs)):
        start_sample = int(i * hop)
        end_sample = int(start_sample + samples_per_epoch)
        seg = data[start_sample: min(end_sample, data_len)]
        if seg.shape[0] < samples_per_epoch:
            pad_amt = int(samples_per_epoch - seg.shape[0])
            try:
                seg = np.pad(seg, (0, pad_amt), mode="constant", constant_values=0.0)
            except Exception as e:
                raise RuntimeError(f"Padding failed for subj {subj} epoch {i}: pad_amt={pad_amt}, seg_shape={seg.shape}, error={e}")
        if seg.shape[0] != samples_per_epoch:
            raise RuntimeError(f"Segment length mismatch after padding for subj {subj} epoch {i}: expected {samples_per_epoch}, got {seg.shape[0]}")
        fp[i, :] = seg.astype("float32")

        start_sec = float(start_sample) / float(sr)
        end_sec = float(end_sample) / float(sr)
        lbl = 0
        for me in mapped_events:
            try:
                ev_on = float(me["audio_onset"])
                ev_off = float(me["audio_onset"]) + float(me["duration"])
            except Exception:
                # skip badly-formed mapped event
                continue
            overlap_sec = max(0.0, min(end_sec, ev_off) - max(start_sec, ev_on))
            if overlap_sec >= 1.0 and is_positive_event(me["label"]):
                lbl = 1
                break
        labels.append({"epoch_index": int(i), "start_sec": float(start_sec), "end_sec": float(min(end_sec, duration)), "label": int(lbl)})

    # flush memmap
    del fp

    # save labels
    labels_df = pd.DataFrame(labels)
    labels_df.to_csv(out_dir / "labels.csv", index=False)

    # meta
    meta = {
        "subject": subj,
        "sr": sr,
        "n_epochs": int(n_epochs),
        "samples_per_epoch": int(samples_per_epoch),
        "overlap": float(overlap),
        "epoch_sec": float(epoch_sec),
        "total_annotations": len(ann_events),
        "mapped_events_preview": mapped_events[:10],
        "chosen_time_delay": chosen_td,
        "chosen_sign": chosen_sign,
        "chosen_sign_stats": stats
    }
    with open(out_dir / "epoch_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[{subj}] n_epochs={n_epochs} sr={sr} epoch_sec={epoch_sec} mapped_events_total={len(ann_events)} chosen_td={chosen_td} sign={chosen_sign} stats={stats}")

    return out_dir, meta

def main(manifest_csv: str, processed_dir: str, epoch_sec: float, overlap: float):
    df = pd.read_csv(manifest_csv)
    processed_dir = Path(processed_dir)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        subj_meta = row.to_dict()
        # top-level coercion before passing to worker:
        if "audio_sr" in subj_meta:
            c = _coerce_to_scalar(subj_meta["audio_sr"])
            if c is not None:
                subj_meta["audio_sr"] = c
        if "audio_duration_sec" in subj_meta:
            c = _coerce_to_scalar(subj_meta["audio_duration_sec"])
            if c is not None:
                subj_meta["audio_duration_sec"] = c
        # coerce any tuple/list/ndarray in manifest early
        for k, v in list(subj_meta.items()):
            if isinstance(v, (list, tuple, _np.ndarray)):
                c = _coerce_to_scalar(v)
                if c is not None:
                    subj_meta[k] = c

        try:
            create_epochs_and_labels(subj_meta, processed_dir, epoch_sec=epoch_sec, overlap=overlap)
        except Exception as e:
            warnings.warn(f"ERROR subject {subj_meta.get('subject')}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="data/manifest_apsaa.csv")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--epoch_sec", type=float, default=30.0)
    parser.add_argument("--overlap", type=float, default=0.0, help="fractional overlap between 0 and <1")
    args = parser.parse_args()
    main(args.manifest, args.processed_dir, args.epoch_sec, args.overlap)