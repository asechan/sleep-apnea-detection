#!/usr/bin/env python3
"""
data_sync_apsaa.py

APSAA-specific manifest + alignment builder.

Assumptions:
 - Raw dataset layout: data/raw/<subject_id>/* (one folder per subject)
 - Inside each subject folder there is:
     - one audio file (*.wav)
     - one Annotations CSV (name includes 'Annotation' or 'Annotations')
     - one sync_output.mat
     - other sensor CSVs (ignored here)
 - The script will create:
     - data/manifest_apsaa.csv  (one row per subject)
     - data/processed/<subject>/metadata.json
     - data/diagnostics_apsaa.json

Usage:
    python src/data_sync_apsaa.py --raw_dir data/raw --out_csv data/manifest_apsaa.csv --diag data/diagnostics_apsaa.json
"""

import argparse
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import soundfile as sf
import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm
import re

# ---------- Helpers ----------

AUDIO_EXTS = [".wav", ".flac", ".m4a", ".mp3"]

def md5_of_file(path: Path, block_size: int = 2**20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()

def get_audio_info(path: Path) -> Dict[str, Any]:
    try:
        with sf.SoundFile(str(path)) as f:
            frames = f.frames
            sr = f.samplerate
            channels = f.channels
            subtype = f.subtype
            duration = frames / float(sr) if sr > 0 else 0.0
            return {
                "sample_rate": int(sr),
                "duration_sec": float(duration),
                "frames": int(frames),
                "channels": int(channels),
                "subtype": subtype,
            }
    except Exception as e:
        return {"error": str(e)}

# Robust CSV annotation reader: tries common column name variants
POSSIBLE_ONSET_COLS = ["start", "onset", "time", "begin", "start_time", "startsec"]
POSSIBLE_DURATION_COLS = ["duration", "dur", "length", "stop_seconds"]
POSSIBLE_LABEL_COLS = ["event", "type", "label", "annotation", "description"]

def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_cols:
            return lower_cols[cand.lower()]
    # try fuzzy: contains substring
    for col in df.columns:
        lc = col.lower()
        for cand in candidates:
            if cand.lower() in lc:
                return col
    return None

def read_annotations_csv(csv_path: Path) -> Dict[str, Any]:
    """
    Read an annotations CSV and return a normalized list of events:
      [{"onset": float, "duration": float, "label": str, "raw_row": {...}}, ...]
    If parsing fails, returns an 'error' key.
    """
    out = {"events": [], "n_events": 0}
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        try:
            df = pd.read_csv(csv_path, encoding="latin1")
        except Exception as e2:
            return {"error": f"csv_read_error: {e} / {e2}"}

    if df.shape[0] == 0:
        return {"events": [], "n_events": 0}

    onset_col = find_column(df, POSSIBLE_ONSET_COLS)
    dur_col = find_column(df, POSSIBLE_DURATION_COLS)
    label_col = find_column(df, POSSIBLE_LABEL_COLS)

    # If no duration column, try to infer from consecutive onset or set to 0
    for idx, row in df.iterrows():
        try:
            onset = None
            dur = 0.0
            lbl = ""
            if onset_col is not None:
                onset = float(row[onset_col])
            else:
                # if there's a column named 'Event Time' or similar, we already tried; if none, skip
                onset = None

            if dur_col is not None:
                dur = float(row[dur_col]) if not pd.isna(row[dur_col]) else 0.0
            else:
                dur = 0.0

            if label_col is not None:
                lbl = str(row[label_col])
            else:
                # fallback: try to find an "Event" like column in any string columns
                # pick first non-numeric column
                for c in df.columns:
                    if df[c].dtype == object:
                        lbl = str(row[c])
                        break

            if onset is None:
                # skip rows we cannot parse onset for
                continue

            out["events"].append({"onset": float(onset), "duration": float(dur), "label": lbl, "raw_row": row.to_dict()})
        except Exception:
            # skip problematic row but record it later via diagnostics if needed
            continue

    out["n_events"] = len(out["events"])
    return out

def read_sync_mat(mat_path: Path) -> Dict[str, Any]:
    """
    Load sync_output.mat and try to extract plausible mapping fields.
    The contents vary; we try to return any numeric entries that look like start offsets or sample-rates.
    """
    res = {}
    try:
        mat = loadmat(str(mat_path))
    except Exception as e:
        return {"error": f"mat_read_error: {str(e)}"}

    # Flatten typical entries; look for keys that include 'audio', 'start', 'offset', 't0', 'time'
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        # convert small numpy arrays to python numbers when possible
        try:
            if isinstance(v, np.ndarray):
                if v.size == 1:
                    val = float(np.squeeze(v))
                    res[k] = val
                else:
                    # convert 1D arrays to lists
                    try:
                        res[k] = np.squeeze(v).tolist()
                    except Exception:
                        res[k] = str(type(v))
            else:
                res[k] = str(type(v))
        except Exception:
            res[k] = "unreadable"
    return res

# Map annotation times (in annotation timebase) to audio timebase:
# Use sync dict keys heuristics: many sync mats include fields like 'audio_start', 'annotation_start', or offsets
def map_annotation_time_to_audio_time(ann_time: float, sync_dict: Dict[str, Any]) -> Tuple[float, bool]:
    """
    Returns (audio_time_sec, mapped_flag)
    - If sync_dict contains fields that strongly indicate mapping, apply them.
    - Otherwise return (ann_time, False) meaning "no mapping applied".
    Heuristic order:
      1) If sync contains 'audio_start' and 'annotation_start' (or variants), do: audio_time = ann_time - annotation_start + audio_start
      2) If sync contains 'offset' or 'time_offset', do: audio_time = ann_time + offset
      3) If sync contains 'sample_offset' and sample_rate, convert units if necessary
    """
    if not sync_dict:
        return ann_time, False

    # Normalize keys to lowercase for lookup
    keys = {k.lower(): v for k, v in sync_dict.items()}

    # helper to fetch number-like
    def get_key_variants(klist):
        for k in klist:
            if k in keys:
                try:
                    return float(keys[k])
                except Exception:
                    pass
        return None

    # common variants:
    audio_start = get_key_variants(["audio_start", "audio_start_sec", "audio_start_time"])
    ann_start = get_key_variants(["annotation_start", "annotations_start", "annotation_t0", "ann_start"])
    offset = get_key_variants(["offset", "time_offset", "sync_offset", "offset_seconds"])
    sample_offset = get_key_variants(["sample_offset", "audio_sample_offset"])
    sample_rate = get_key_variants(["fs", "sample_rate", "sr", "audio_sr"])

    if audio_start is not None and ann_start is not None:
        mapped = ann_time - ann_start + audio_start
        return float(mapped), True
    if offset is not None:
        return float(ann_time + offset), True
    if sample_offset is not None and sample_rate is not None:
        # if sample_offset is in samples, convert to seconds
        return float(ann_time + (sample_offset / sample_rate)), True

    # no mapping found
    return ann_time, False

# ---------- Main manifest builder ----------

def build_manifest_apsaa(raw_dir: str, out_csv: str, diag_json: str, processed_dir: str = "data/processed", verbose: bool = True) -> str:
    raw = Path(raw_dir)
    if not raw.exists():
        raise FileNotFoundError(f"raw_dir not found: {raw}")

    subject_dirs = [p for p in raw.iterdir() if p.is_dir()]
    subject_dirs = sorted(subject_dirs)
    rows = []
    diagnostics = {"missing_audio": [], "missing_annotations": [], "missing_sync": [], "audio_errors": [], "annotation_errors": [], "sync_errors": []}

    for subj_dir in tqdm(subject_dirs, desc="Subjects", disable=not verbose):
        subj = subj_dir.name
        row: Dict[str, Any] = {
            "subject": subj,
            "audio_path": None,
            "audio_md5": None,
            "audio_sr": None,
            "audio_duration_sec": None,
            "audio_channels": None,
            "annotations_csv": None,
            "n_annotations": None,
            "n_apnea_like": None,
            "sync_mat": None,
            "sync_keys": None,
            "mapped_annotation_preview": [],  # small sample of mapped events
        }

        # find audio
        aud = None
        for ext in AUDIO_EXTS:
            cand = subj_dir / f"{subj}{ext}"
            if cand.exists():
                aud = cand
                break
        if aud is None:
            # fallback: any wav/m4a/mp3 in folder
            for p in subj_dir.iterdir():
                if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                    aud = p
                    break

        if aud is None:
            diagnostics["missing_audio"].append(str(subj_dir))
            rows.append(row)
            continue

        row["audio_path"] = str(aud)
        try:
            row["audio_md5"] = md5_of_file(aud)
        except Exception as e:
            diagnostics["audio_errors"].append({"subject": subj, "file": str(aud), "error": str(e)})

        ainfo = get_audio_info(aud)
        if "error" in ainfo:
            diagnostics["audio_errors"].append({"subject": subj, "file": str(aud), "error": ainfo["error"]})
        else:
            row["audio_sr"] = ainfo["sample_rate"]
            row["audio_duration_sec"] = ainfo["duration_sec"]
            row["audio_channels"] = ainfo["channels"]

        # find annotations CSV (look for *Annotation*.csv)
        ann_csv = None
        for p in subj_dir.iterdir():
            if p.is_file() and "annot" in p.name.lower() and p.suffix.lower() == ".csv":
                ann_csv = p
                break
        if ann_csv is None:
            # fallback: file that contains 'Annotation' or 'Annotations' or maybe 'events'
            for p in subj_dir.iterdir():
                if p.is_file() and p.suffix.lower() == ".csv":
                    name = p.name.lower()
                    if "annotation" in name or "event" in name or "annot" in name:
                        ann_csv = p
                        break

        if ann_csv is None:
            diagnostics["missing_annotations"].append(str(subj_dir))
        else:
            row["annotations_csv"] = str(ann_csv)
            ann_info = read_annotations_csv(ann_csv)
            if "error" in ann_info:
                diagnostics["annotation_errors"].append({"subject": subj, "file": str(ann_csv), "error": ann_info["error"]})
            else:
                row["n_annotations"] = ann_info["n_events"]
                # count apnea-like labels heuristically
                apnea_like = 0
                for e in ann_info["events"]:
                    lbl = str(e.get("label", "")).lower()
                    if ("apnea" in lbl) or ("apnoea" in lbl) or ("hypopnea" in lbl) or ("hypopnoea" in lbl):
                        apnea_like += 1
                row["n_apnea_like"] = int(apnea_like)

        # find sync_output.mat
        sync_mat = subj_dir / "sync_output.mat"
        if not sync_mat.exists():
            # try other mat files
            mats = [p for p in subj_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mat"]
            if mats:
                sync_mat = mats[0]
            else:
                sync_mat = None

        if sync_mat is None:
            diagnostics["missing_sync"].append(str(subj_dir))
            row["sync_mat"] = None
            row["sync_keys"] = None
        else:
            row["sync_mat"] = str(sync_mat)
            sync_info = read_sync_mat(sync_mat)
            if "error" in sync_info:
                diagnostics["sync_errors"].append({"subject": subj, "file": str(sync_mat), "error": sync_info["error"]})
                row["sync_keys"] = None
            else:
                # store only keys summary (first 10 keys)
                row["sync_keys"] = list(sync_info.keys())[:20]
                # If we have annotations, map first few to audio time using sync mapping
                if row["annotations_csv"] is not None and row["n_annotations"] and row["n_annotations"] > 0:
                    try:
                        ann_info_local = read_annotations_csv(Path(row["annotations_csv"]))
                        mapped_sample = []
                        for e in ann_info_local["events"][:10]:
                            mapped_time, mapped_flag = map_annotation_time_to_audio_time(e["onset"], sync_info)
                            mapped_sample.append({"orig_onset": e["onset"], "duration": e["duration"], "label": e["label"], "mapped_onset_audio": mapped_time, "mapped": bool(mapped_flag)})
                        row["mapped_annotation_preview"] = mapped_sample
                    except Exception:
                        row["mapped_annotation_preview"] = []
        # write per-subject processed metadata
        processed_subj_dir = Path(processed_dir) / subj
        processed_subj_dir.mkdir(parents=True, exist_ok=True)
        meta_out = processed_subj_dir / "metadata.json"
        with meta_out.open("w") as f:
            json.dump(row, f, indent=2)

        rows.append(row)

    # write global manifest
    df = pd.DataFrame(rows)
    outp = Path(out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False)

    # write diagnostics
    diagp = Path(diag_json)
    diagp.parent.mkdir(parents=True, exist_ok=True)
    with diagp.open("w") as f:
        json.dump(diagnostics, f, indent=2)

    if verbose:
        print(f"Processed {len(subject_dirs)} subject folders.")
        print(f"Manifest written to {outp}")
        print(f"Diagnostics written to {diagp}")
    return str(outp)

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APSAA manifest + sync builder")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Directory containing subject subfolders")
    parser.add_argument("--out_csv", type=str, default="data/manifest_apsaa.csv", help="Output manifest CSV path")
    parser.add_argument("--diag", type=str, default="data/diagnostics_apsaa.json", help="Diagnostics JSON output")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Processed output dir for per-subject metadata")
    parser.add_argument("--quiet", action="store_true", help="less stdout")
    args = parser.parse_args()

    build_manifest_apsaa(args.raw_dir, args.out_csv, args.diag, processed_dir=args.processed_dir, verbose=(not args.quiet))
