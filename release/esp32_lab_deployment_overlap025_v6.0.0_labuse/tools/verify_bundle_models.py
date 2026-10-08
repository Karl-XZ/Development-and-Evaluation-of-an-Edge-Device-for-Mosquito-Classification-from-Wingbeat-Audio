#!/usr/bin/env python3
"""Check all four included TFLite/header/verification triplets before release."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER_DIR = ROOT / "source" / "lab_realtime_recorder_v006_quad_exact3s" / "models"
MODELS = (
    ("model_overlap025_logmel", "logmel", "esp32_pair"),
    ("model_overlap025", "pcen", "esp32_pair"),
    ("model_overlap025_logmel_lowpass", "logmel", "esp32_lowpass"),
    ("model_overlap025_pcen_lowpass", "pcen", "esp32_lowpass"),
)
CLASSES = ["ae_aegypti", "cx_quin", "an_dirus", "an_minimus_f", "an_minimus_m", "no_mosquito"]


def check(symbol: str, frontend: str, resample_mode: str) -> dict:
    tflite = ROOT / "model" / f"{symbol}.tflite"
    header = HEADER_DIR / f"{symbol}.h"
    verification = ROOT / "model" / f"{symbol}_verification.json"
    data = tflite.read_bytes()
    header_text = header.read_text(encoding="utf-8")
    found = re.search(rf"const unsigned char {symbol}\[\]\s*=\s*\{{(.*?)\}};",
                      header_text, re.DOTALL)
    if not found:
        raise ValueError(f"C array missing or incorrect: {header}")
    header_bytes = bytes(int(value, 16) for value in re.findall(r"0x([0-9a-fA-F]{2})", found.group(1)))
    length = re.search(rf"{symbol}_len\s*=\s*(\d+);", header_text)
    if header_bytes != data or length is None or int(length.group(1)) != len(data):
        raise ValueError(f"Model/header byte mismatch: {symbol}")
    report = json.loads(verification.read_text(encoding="utf-8"))
    if (report["frontend"] != frontend
            or report.get("resample_mode", "esp32_pair") != resample_mode
            or report["model_bytes"] != len(data)
            or report["input_shape"] != [1, 64, 151, 1]
            or report["class_names"] != CLASSES):
        raise ValueError(f"Model verification metadata mismatch: {symbol}")
    return {"symbol": symbol, "frontend": frontend, "resample_mode": resample_mode,
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "header_exact": True}


def main() -> None:
    result = [check(*item) for item in MODELS]
    print(json.dumps({"package": "v6.0.0", "models": result}, indent=2))


if __name__ == "__main__":
    main()
