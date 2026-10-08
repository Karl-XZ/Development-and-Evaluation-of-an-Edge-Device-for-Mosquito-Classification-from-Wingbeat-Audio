"""Check the v6 C++ low-pass taps and boundary rule against SciPy 1:2 resample_poly."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.signal import firwin, resample_poly


ROOT = Path(__file__).resolve().parents[1]
SKETCH = ROOT / "source" / "lab_realtime_recorder_v006_quad_exact3s" / "lab_realtime_recorder_v006_quad_exact3s.ino"


def coefficients() -> tuple[np.float32, np.ndarray]:
    source = SKETCH.read_text(encoding="utf-8")
    center_match = re.search(r"kLowpassCenter\s*=\s*([-+0-9.eE]+)f", source)
    taps_match = re.search(r"kLowpassOddTaps\[10\]\s*=\s*\{([^}]+)\}", source, re.S)
    if not center_match or not taps_match:
        raise RuntimeError("C++ low-pass coefficients not found")
    taps = np.array([float(value) for value in re.findall(r"([-+0-9.eE]+)f", taps_match.group(1))], dtype=np.float32)
    if taps.size != 10:
        raise RuntimeError(f"Expected ten half-band tap pairs, got {taps.size}")
    return np.float32(float(center_match.group(1))), taps


def mirrored_tap_kernel(center: np.float32, taps: np.ndarray) -> np.ndarray:
    kernel = np.zeros(41, dtype=np.float32)
    kernel[20] = center
    for index, tap in enumerate(taps):
        offset = 2 * index + 1
        kernel[20 - offset] = tap
        kernel[20 + offset] = tap
    return kernel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "validation" / "V006_LOWPASS_TAP_CHECK.json")
    args = parser.parse_args()
    center, taps = coefficients()
    kernel = mirrored_tap_kernel(center, taps)
    scipy_kernel = firwin(41, 0.5, window=("kaiser", 8.6)).astype(np.float32)
    coefficient_error = float(np.max(np.abs(kernel - scipy_kernel)))
    rng = np.random.default_rng(20260930)
    sample_index = np.arange(24000, dtype=np.float32)
    signals = {
        "random_pcm16": rng.integers(-32768, 32767, size=24000, dtype=np.int16).astype(np.float32) / 32768,
        "edge_impulse": np.eye(1, 24000, 0, dtype=np.float32).ravel(),
        "five_khz": np.sin(2 * np.pi * 5000 * sample_index / 16000).astype(np.float32),
        "wingbeat_420_hz": np.sin(2 * np.pi * 420 * sample_index / 16000).astype(np.float32),
    }
    errors = {}
    for name, signal in signals.items():
        expected = resample_poly(signal, 1, 2, window=("kaiser", 8.6)).astype(np.float32)
        actual = np.convolve(signal, kernel, mode="full")[20::2][:12000].astype(np.float32)
        errors[name] = float(np.max(np.abs(actual - expected)))
    passed = coefficient_error < 1e-6 and max(errors.values()) < 1e-6
    report = {"passed": passed, "scipy_reference": "resample_poly(x,1,2,window=('kaiser',8.6))",
              "taps": 41, "coefficient_max_error": coefficient_error, "waveform_max_errors": errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
