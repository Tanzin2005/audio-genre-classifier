"""Prepare cached spectrograms, then split complete recordings."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from genre_cnn.audio import AudioConfig, GENRES, load_audio, spectrograms


def split_tracks(records, seed=42):
    counts = Counter(row["genre"] for row in records)
    if len(counts) < 2 or min(counts.values()) < 10:
        raise ValueError(
            "Need at least 10 valid, distinct recordings per genre before splitting"
        )
    train_val, test = train_test_split(
        records,
        test_size=0.2,
        random_state=seed,
        stratify=[r["genre"] for r in records],
    )
    train, validation = train_test_split(
        train_val,
        test_size=0.2,
        random_state=seed,
        stratify=[r["genre"] for r in train_val],
    )
    return {"train": train, "validation": validation, "test": test}


def prepare(data_dir, cache_dir, seed=42):
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    config = AudioConfig()
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Audio directory not found: {data_dir}. Point --data at genres_original."
        )
    for genre in GENRES:
        if not (data_dir / genre).is_dir():
            raise ValueError(
                f"Missing genre folder: {genre}. --data must point to genres_original."
            )
    if (cache_dir / "manifest.json").exists():
        raise FileExistsError(
            "A manifest already exists. Use a new cache directory to preserve the previous split."
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    records = []
    skipped = []
    seen = {}
    for genre in GENRES:
        files = sorted((data_dir / genre).glob("*.wav"))
        for path in files:
            if path.name.startswith("._"):
                continue  # macOS archive metadata, not an audio recording.
            try:
                y = load_audio(path, config)
                checksum = hashlib.sha256(y.tobytes()).hexdigest()
                if checksum in seen:
                    skipped.append(
                        {
                            "file": f"{genre}/{path.name}",
                            "reason": "Exact decoded-audio duplicate",
                            "duplicate_of": seen[checksum],
                        }
                    )
                    continue
                track = f"{genre}/{path.name}"
                features = spectrograms(y, config)
                filename = f"{genre}_{path.stem}.npy"
                np.save(
                    cache_dir / filename,
                    features.astype(np.float16),
                    allow_pickle=False,
                )
                seen[checksum] = track
                records.append(
                    {
                        "track": track,
                        "genre": genre,
                        "label": GENRES.index(genre),
                        "cache": filename,
                        "segments": len(features),
                        "sha256": checksum,
                    }
                )
            except (ValueError, RuntimeError, OSError) as error:
                skipped.append({"file": f"{genre}/{path.name}", "reason": str(error)})
        print(
            f"{genre}: {sum(r['genre'] == genre for r in records)} valid recordings",
            flush=True,
        )
    (cache_dir / "skipped_files.json").write_text(json.dumps(skipped, indent=2))
    insufficient = [
        genre for genre in GENRES if sum(r["genre"] == genre for r in records) < 10
    ]
    if insufficient:
        raise ValueError(
            f"Need at least 10 valid, distinct recordings in every genre. Check: {insufficient}"
        )
    splits = split_tracks(records, seed)
    for split, rows in splits.items():
        for row in rows:
            row["split"] = split
    manifest = {
        "format_version": 1,
        "seed": seed,
        "config": asdict(config),
        "classes": GENRES,
        "tracks": sorted(records, key=lambda r: r["track"]),
        "skipped": skipped,
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        "Recordings per split:",
        {key: len(value) for key, value in splits.items()},
        flush=True,
    )
    print("Corrupt/duplicate files are recorded in skipped_files.json.", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/genres_original")
    parser.add_argument("--cache", default="data/cnn_cache")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.data, args.cache, args.seed)


if __name__ == "__main__":
    main()
