#!/usr/bin/env python3
"""Config-driven, versioned end-to-end retraining entry point."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent / "src"
CATEGORIES = ("Ae_aegypti_M", "Cx_quin_M", "An_dirus_M", "An_minimus_F", "An_minimus_M")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def command_text(parts: list[str]) -> str:
    return subprocess.list2cmdline(parts)


def run(parts: list[str], dry_run: bool) -> None:
    print("\n$ " + command_text(parts), flush=True)
    if not dry_run:
        subprocess.run(parts, check=True)


def validate_training_folder(root: Path) -> dict:
    counts = {}
    problems = []
    for category in CATEGORIES:
        folder = root / category
        labels = sorted(folder.glob("*.labels.json")) if folder.is_dir() else []
        paired = 0
        for label in labels:
            wav = label.with_name(label.name.removesuffix(".labels.json") + ".wav")
            if wav.exists():
                paired += 1
            else:
                problems.append(f"Missing WAV paired with {label}")
        counts[category] = {"label_files": len(labels), "paired_wav_files": paired}
        if not labels:
            problems.append(f"No .labels.json files found in {folder}")
    if problems:
        raise ValueError("Training-folder validation failed:\n- " + "\n- ".join(problems))
    return counts


def dataset_fingerprint(root: Path) -> dict:
    entries = []
    for category in CATEGORIES:
        for path in sorted((root / category).glob("*")):
            if path.is_file() and (path.suffix.lower() == ".wav" or path.name.endswith(".labels.json")):
                stat = path.stat()
                item = {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
                if path.name.endswith(".labels.json"):
                    item["sha256"] = sha256(path)
                entries.append(item)
    canonical = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"algorithm": "metadata+label_sha256", "sha256": hashlib.sha256(canonical).hexdigest(), "files": entries}


def best_checkpoint(fold_results: Path) -> Path:
    with fold_results.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No completed folds in {fold_results}")
    best = max(rows, key=lambda row: float(row["best_validation_macro_f1"]))
    return Path(best["checkpoint"]).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-folder", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs" / "overlap025_logmel_pair_exact3s_v006.json")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print commands without writing training outputs")
    parser.add_argument("--smoke-test", action="store_true", help="Run one epoch on capped samples and fold 0")
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()

    training_folder = args.training_folder.resolve()
    config_path = args.config.resolve()
    config = load_json(config_path)
    validation = validate_training_folder(training_folder)
    fingerprint = dataset_fingerprint(training_folder)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or (PACKAGE_ROOT / "runs" / f"{config['experiment_name']}_{timestamp}")).resolve()
    dataset_dir = output / "dataset"
    model_dir = output / "model"
    export_dir = output / "export"
    version_match = re.search(r"v(\d+\.\d+\.\d+)", PACKAGE_ROOT.name, re.IGNORECASE)
    package_version = version_match.group(1) if version_match else PACKAGE_ROOT.name

    manifest = {
        "status": "planned" if args.dry_run else "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": package_version,
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "training_folder": str(training_folder),
        "training_folder_validation": validation,
        "dataset_fingerprint": fingerprint["sha256"],
        "python": sys.version,
        "platform": platform.platform(),
        "smoke_test": args.smoke_test,
    }
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=False)
        (output / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        (output / "dataset_fingerprint.json").write_text(json.dumps(fingerprint, indent=2) + "\n", encoding="utf-8")
        (output / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    d = config["dataset"]
    t = config["training"]
    prepare = [
        sys.executable, str(SRC / "prepare_dataset.py"),
        "--root", str(training_folder), "--output", str(dataset_dir),
        "--dataset-mode", d["mode"], "--window-sec", str(d["window_sec"]),
        "--window-stride-sec", str(d["window_stride_sec"]),
        "--negative-margin-sec", str(d["negative_margin_sec"]),
        "--confidence-threshold", str(d["confidence_threshold"]), "--seed", str(d["seed"]),
    ]
    if args.dry_run:
        prepare.append("--dry-run")
    run(prepare, False if args.dry_run else False)

    folds = "0" if args.smoke_test else str(t["folds"])
    epochs = 1 if args.smoke_test else int(t["epochs"])
    train = [
        sys.executable, str(SRC / "train_cv.py"), "--data-dir", str(dataset_dir), "--output", str(model_dir),
        "--class-names", ",".join(d["classes"]), "--epochs", str(epochs), "--batch-size", str(t["batch_size"]),
        "--workers", "0" if args.smoke_test else str(t["workers"]), "--learning-rate", str(t["learning_rate"]),
        "--weight-decay", str(t["weight_decay"]), "--patience", str(t["patience"]),
        "--sample-rate", str(t["sample_rate"]), "--resample-mode", t["resample_mode"],
        "--window-sec", str(d["window_sec"]), "--input-context-sec", str(t["input_context_sec"]),
        "--context-alignment", t["context_alignment"], "--n-mels", str(t["n_mels"]),
        "--n-fft", str(t["n_fft"]), "--hop-length", str(t["hop_length"]), "--folds", folds,
        "--frontend", t["frontend"], "--pcen-alpha", str(t["pcen_alpha"]),
        "--pcen-delta", str(t["pcen_delta"]), "--pcen-r", str(t["pcen_r"]),
        "--pcen-time-constant", str(t["pcen_time_constant"]),
        "--pcen-eps", str(t["pcen_eps"]),
        "--input-normalization", t["input_normalization"],
        "--device", t["device"], "--seed", str(d["seed"]),
    ]
    if args.smoke_test:
        train += ["--max-train-samples", "48", "--max-validation-samples", "24", "--max-test-samples", "24"]
    if args.dry_run:
        print("\n$ " + command_text(train))
        print("\nDry run complete. No training output was created.")
        return 0
    run(train, False)

    checkpoint = best_checkpoint(model_dir / "fold_results.csv")
    manifest["selected_checkpoint"] = str(checkpoint)
    manifest["selection_rule"] = "maximum validation Macro-F1 only; locked outer test is not used for selection"
    if not args.skip_export:
        e = config["export"]
        export = [
            sys.executable, str(SRC / "export_int8_tflite.py"), "--checkpoint", str(checkpoint),
            "--data-dir", str(dataset_dir), "--output-dir", str(export_dir), "--symbol", e["symbol"],
            "--representative-samples", str(e["representative_samples"]),
            "--verification-samples", str(e["verification_samples"]),
        ]
        run(export, False)
    manifest["status"] = "completed"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["output_files"] = [
        {"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(output.rglob("*")) if path.is_file() and path.name != "experiment_manifest.json"
    ]
    (output / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nCompleted: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
