"""Validate the raw binary memmaps used by this project (despite .npy names)."""
import json
from pathlib import Path


def validate_features(features_dir):
    root = Path(features_dir)
    with (root / "meta.json").open() as handle:
        meta = json.load(handle)
    dimensions = [meta[k] for k in ("total_epochs", "n_mels", "T")]
    if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in dimensions):
        raise ValueError("Feature dimensions must be positive integers")
    total, mels, frames = dimensions
    for name, expected in (("X.npy", total * mels * frames * 4), ("y.npy", total)):
        path = root / name
        actual = path.stat().st_size
        if actual != expected:
            raise ValueError(
                f"{path}: expected {expected:,} bytes from meta.json, found {actual:,}. "
                "Feature artifacts are incomplete or inconsistent; regenerate them into a new directory."
            )
    seen = set()
    for name in ("train", "val", "test"):
        indices = meta["splits"][name]
        if not indices:
            raise ValueError(f"The {name} split is empty; provide more subjects or adjust splits")
        if any(not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < total for i in indices):
            raise ValueError(f"The {name} split contains invalid indices")
        if len(set(indices)) != len(indices) or seen.intersection(indices):
            raise ValueError(f"The {name} split contains duplicate or overlapping indices")
        seen.update(indices)
    if len(seen) != total:
        raise ValueError("Splits do not cover every feature row")
    return meta


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features_dir")
    args = parser.parse_args()
    try:
        meta = validate_features(args.features_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Artifact validation failed: {exc}\n")
    print(f"Validated {meta['total_epochs']} feature rows and all three splits.")
