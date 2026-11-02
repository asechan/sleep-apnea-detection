"""Shared, strict annotation parsing for manifest creation and epoch labeling."""
import math
from pathlib import Path

import pandas as pd


def parse_time(value):
    parts = str(value).strip().split(":")
    if len(parts) not in (1, 2, 3):
        raise ValueError(f"Invalid timestamp: {value!r}")
    numbers = [float(part) for part in parts]
    if any(not math.isfinite(n) or n < 0 for n in numbers):
        raise ValueError(f"Invalid timestamp: {value!r}")
    if len(parts) > 1 and any(n >= 60 for n in numbers[1:]):
        raise ValueError(f"Invalid timestamp: {value!r}")
    return sum(n * 60 ** i for i, n in enumerate(reversed(numbers)))


def read_events(path: Path):
    try:
        frame = pd.read_csv(path)
    except UnicodeDecodeError:
        frame = pd.read_csv(path, encoding="latin1")
    columns = {str(c).strip().lower(): c for c in frame.columns}

    def column(names):
        for name in names:
            if name in columns:
                return columns[name]
        raise ValueError(f"{path}: missing required column; expected one of {names}")

    onset = column(("start_time", "start", "onset", "time", "begin", "startsec"))
    duration = column(("duration", "dur", "length"))
    label = column(("event_name", "event", "type", "label", "annotation", "description"))
    events = []
    for row_number, (_, row) in enumerate(frame.iterrows(), start=2):
        try:
            start = parse_time(row[onset])
            seconds = float(row[duration])
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("duration must be finite and positive")
            if pd.isna(row[label]) or not str(row[label]).strip():
                raise ValueError("event label is missing")
            events.append({"onset": start, "duration": seconds, "label": str(row[label]).strip()})
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}: invalid annotation on CSV row {row_number}: {exc}") from exc
    return events
