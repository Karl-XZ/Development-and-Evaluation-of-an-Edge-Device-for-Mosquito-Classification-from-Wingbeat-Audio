#!/usr/bin/env python3
"""Create leakage-safe manifests for Ae / Cx / No_mosquito training.

The source WAV files remain in their species folders.  This script writes only
CSV/JSON metadata under ``mosquito_3class`` so later training can stream the
requested time windows without duplicating tens of gigabytes of audio.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


THREE_CLASS_SOURCES = {
    "Ae_aegypti_M": ("ae_aegypti", 0),
    "Cx_quin_M": ("cx_quin", 1),
}
SIX_CLASS_SOURCES = {
    "Ae_aegypti_M": ("ae_aegypti", 0),
    "Cx_quin_M": ("cx_quin", 1),
    "An_dirus_M": ("an_dirus", 2),
    "An_minimus_F": ("an_minimus_f", 3),
    "An_minimus_M": ("an_minimus_m", 4),
}
CLASSES = dict(THREE_CLASS_SOURCES)
NO_MOSQUITO = ("no_mosquito", len(CLASSES))
SPLITS = ("train", "validation", "test")


def configure_dataset_mode(mode: str) -> None:
    global CLASSES, NO_MOSQUITO
    CLASSES = dict(
        SIX_CLASS_SOURCES if mode == "6class" else THREE_CLASS_SOURCES
    )
    NO_MOSQUITO = ("no_mosquito", len(CLASSES))


def canonical_recording_name(name: str) -> str:
    """Collapse duplicate-download suffixes such as ``(1)`` and `` (1)``."""
    return re.sub(r" ?\(\d+\)", "", name)


@dataclass(frozen=True)
class Recording:
    category: str
    label: str
    label_id: int
    wav_path: Path
    label_path: Path
    duration_sec: float
    regions: tuple[tuple[float, float, float], ...]
    split: str
    cv_fold: int | None

    @property
    def recording_id(self) -> str:
        return self.wav_path.stem


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def find_recordings(root: Path, confidence_threshold: float) -> list[dict]:
    """Load paired WAV/JSON source records and reject malformed inputs."""
    records: list[dict] = []
    for category, (label, label_id) in CLASSES.items():
        category_dir = root / category
        candidates: dict[str, list[Path]] = defaultdict(list)
        for label_path in sorted(category_dir.glob("*.labels.json")):
            candidates[canonical_recording_name(label_path.name)].append(
                label_path
            )
        for canonical_name, duplicate_group in sorted(candidates.items()):
            label_path = min(
                duplicate_group,
                key=lambda path: (
                    path.name != canonical_name,
                    len(path.name),
                    path.name,
                ),
            )
            wav_path = label_path.with_name(
                label_path.name.removesuffix(".labels.json") + ".wav"
            )
            if not wav_path.exists():
                raise FileNotFoundError(f"Missing paired WAV for {label_path.name}")
            payload = json.loads(label_path.read_text(encoding="utf-8"))
            regions = []
            for region in payload.get("regions", []):
                if float(region["confidence"]) < confidence_threshold:
                    continue
                start = float(region["start_sec"])
                end = float(region["end_sec"])
                if end > start:
                    regions.append((start, end, float(region["confidence"])))
            records.append(
                {
                    "category": category,
                    "label": label,
                    "label_id": label_id,
                    "wav_path": wav_path,
                    "label_path": label_path,
                    "duration_sec": wav_duration(wav_path),
                    "regions": tuple(sorted(regions)),
                }
            )
    return records


def assign_splits_and_folds(
    raw_records: Iterable[dict], seed: int, test_per_class: int, val_per_class: int, folds: int
) -> list[Recording]:
    """Split by recording, never by window; reserve test recordings before CV."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in raw_records:
        grouped[record["category"]].append(record)

    result: list[Recording] = []
    for category, records in grouped.items():
        needed = test_per_class + val_per_class + folds
        if len(records) < needed:
            raise ValueError(f"{category} has {len(records)} recordings; need at least {needed}")
        shuffled = sorted(records, key=lambda item: item["wav_path"].name)
        random.Random(seed + CLASSES[category][1]).shuffle(shuffled)
        test_records = shuffled[:test_per_class]
        val_records = shuffled[test_per_class : test_per_class + val_per_class]
        train_records = shuffled[test_per_class + val_per_class :]
        development = train_records + val_records
        fold_map = {
            item["wav_path"].name: index % folds
            for index, item in enumerate(development)
        }
        for split, records_for_split in (
            ("test", test_records),
            ("validation", val_records),
            ("train", train_records),
        ):
            for record in records_for_split:
                result.append(
                    Recording(
                        **record,
                        split=split,
                        cv_fold=None if split == "test" else fold_map[record["wav_path"].name],
                    )
                )
    return result


def positive_windows(
    recording: Recording, window_sec: float, stride_sec: float
) -> list[dict]:
    rows: list[dict] = []
    for start, end, confidence in recording.regions:
        position = start
        while position + window_sec <= min(end, recording.duration_sec) + 1e-9:
            rows.append(
                manifest_row(
                    recording, recording.label, recording.label_id, position, window_sec,
                    "detector_positive", confidence,
                )
            )
            position += stride_sec
    return rows


def expanded_regions(recording: Recording, margin_sec: float) -> list[tuple[float, float]]:
    """Merge detector regions expanded by a safety margin for negative sampling."""
    intervals = sorted(
        (max(0.0, start - margin_sec), min(recording.duration_sec, end + margin_sec))
        for start, end, _ in recording.regions
    )
    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def background_windows(
    recording: Recording,
    window_sec: float,
    stride_sec: float,
    margin_sec: float,
) -> list[dict]:
    blocked = expanded_regions(recording, margin_sec)
    rows: list[dict] = []
    position = 0.0
    interval_index = 0
    while position + window_sec <= recording.duration_sec + 1e-9:
        while interval_index < len(blocked) and blocked[interval_index][1] <= position:
            interval_index += 1
        overlaps = (
            interval_index < len(blocked)
            and blocked[interval_index][0] < position + window_sec
            and blocked[interval_index][1] > position
        )
        if not overlaps:
            rows.append(
                manifest_row(
                    recording, NO_MOSQUITO[0], NO_MOSQUITO[1], position, window_sec,
                    "weak_detector_background", None,
                )
            )
        position += stride_sec
    return rows


def manifest_row(
    recording: Recording,
    label: str,
    label_id: int,
    start_sec: float,
    window_sec: float,
    label_source: str,
    confidence: float | None,
) -> dict:
    return {
        "split": recording.split,
        "cv_fold": "" if recording.cv_fold is None else recording.cv_fold,
        "recording_id": recording.recording_id,
        "source_category": recording.category,
        "wav_path": recording.wav_path.as_posix(),
        "label_path": recording.label_path.as_posix(),
        "start_sec": f"{start_sec:.6f}",
        "duration_sec": f"{window_sec:.6f}",
        "label": label,
        "label_id": label_id,
        "label_source": label_source,
        "detector_confidence": "" if confidence is None else f"{confidence:.6f}",
    }


def sample_rows(rows: list[dict], count: int, rng: random.Random, description: str) -> list[dict]:
    if len(rows) < count:
        raise ValueError(f"Not enough {description}: have {len(rows)}, need {count}")
    return rng.sample(rows, count)


def build_manifests(
    recordings: list[Recording],
    window_sec: float,
    stride_sec: float,
    margin_sec: float,
    seed: int,
) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    positives: dict[tuple[str, str], list[dict]] = defaultdict(list)
    backgrounds: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for recording in recordings:
        key = (recording.split, recording.category)
        positives[key].extend(
            positive_windows(recording, window_sec, stride_sec)
        )
        backgrounds[key].extend(
            background_windows(
                recording, window_sec, stride_sec, margin_sec
            )
        )

    manifests = {split: [] for split in SPLITS}
    for split in SPLITS:
        source_keys = [(split, category) for category in CLASSES]
        target = min(len(positives[key]) for key in source_keys)
        if target == 0:
            raise ValueError(f"No usable positive windows in {split}")
        for category in CLASSES:
            key = (split, category)
            manifests[split].extend(
                sample_rows(
                    positives[key],
                    target,
                    rng,
                    f"{category} positives in {split}",
                )
            )
        base_negative_count, remainder = divmod(target, len(CLASSES))
        for category_index, category in enumerate(CLASSES):
            negative_count = base_negative_count + (
                1 if category_index < remainder else 0
            )
            if negative_count:
                manifests[split].extend(
                    sample_rows(
                        backgrounds[(split, category)],
                        negative_count,
                        rng,
                        f"{category} background in {split}",
                    )
                )
        rng.shuffle(manifests[split])
    return manifests


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path("mosquito_3class"))
    parser.add_argument(
        "--dataset-mode",
        choices=("3class", "6class"),
        default="3class",
    )
    parser.add_argument("--window-sec", type=float, default=0.5)
    parser.add_argument(
        "--window-stride-sec",
        type=float,
        default=None,
        help="Window start stride; defaults to --window-sec",
    )
    parser.add_argument("--negative-margin-sec", type=float, default=1.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    stride_sec = (
        args.window_sec
        if args.window_stride_sec is None
        else args.window_stride_sec
    )
    if (
        args.window_sec <= 0
        or stride_sec <= 0
        or stride_sec > args.window_sec
        or args.negative_margin_sec < 0
    ):
        parser.error("window duration must be positive and negative margin cannot be negative")

    root = args.root.resolve()
    configure_dataset_mode(args.dataset_mode)
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    raw_records = find_recordings(root, args.confidence_threshold)
    recordings = assign_splits_and_folds(raw_records, args.seed, test_per_class=5, val_per_class=5, folds=5)
    manifests = build_manifests(
        recordings,
        args.window_sec,
        stride_sec,
        args.negative_margin_sec,
        args.seed,
    )

    record_rows = [
        {
            "recording_id": item.recording_id,
            "source_category": item.category,
            "label": item.label,
            "wav_path": item.wav_path.as_posix(),
            "label_path": item.label_path.as_posix(),
            "duration_sec": f"{item.duration_sec:.6f}",
            "detector_regions": len(item.regions),
            "split": item.split,
            "cv_fold": "" if item.cv_fold is None else item.cv_fold,
        }
        for item in sorted(recordings, key=lambda row: row.recording_id)
    ]
    summary = {
        "classes": {
            **{label: label_id for label, label_id in CLASSES.values()},
            NO_MOSQUITO[0]: NO_MOSQUITO[1],
        },
        "dataset_mode": args.dataset_mode,
        "window_sec": args.window_sec,
        "window_stride_sec": stride_sec,
        "negative_margin_sec": args.negative_margin_sec,
        "confidence_threshold": args.confidence_threshold,
        "seed": args.seed,
        "split_recordings": {
            split: Counter(row["source_category"] for row in record_rows if row["split"] == split)
            for split in SPLITS
        },
        "split_windows": {
            split: Counter(row["label"] for row in rows)
            for split, rows in manifests.items()
        },
        "five_fold_development_recordings": {
            str(fold): Counter(
                row["source_category"]
                for row in record_rows
                if row["split"] != "test" and row["cv_fold"] == fold
            )
            for fold in range(5)
        },
        "note": "No_mosquito uses detector-background weak labels and should be audited before final reporting.",
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, default=dict))
        return 0

    output.mkdir(parents=True, exist_ok=True)
    fields = list(manifests["train"][0])
    for split, rows in manifests.items():
        write_csv(output / f"{split}.csv", rows, fields)
    write_csv(output / "recording_splits.csv", record_rows, list(record_rows[0]))
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, default=dict) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
