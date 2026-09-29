#!/usr/bin/env python3
"""Compare PC and ESP32 on identical PCM for log-Mel or PCEN."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import struct
import sys
import time
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

PACKAGE = Path(__file__).resolve().parents[1]
TRAIN_MODULE = PACKAGE / "training" / "src" / "train_cv.py"
spec = importlib.util.spec_from_file_location("v005_train_cv", TRAIN_MODULE)
train_cv = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = train_cv
spec.loader.exec_module(train_cv)

CLASSES = ["ae_aegypti", "cx_quin", "an_dirus", "an_minimus_f", "an_minimus_m", "no_mosquito"]
CONTEXT_OFFSETS = (0, 12000, 24000)


def read_exact(port, count: int, timeout: float) -> bytes:
    data = bytearray()
    deadline = time.time() + timeout
    while len(data) < count:
        chunk = port.read(min(8192, count - len(data)))
        if chunk:
            data.extend(chunk)
            deadline = time.time() + timeout
        elif time.time() > deadline:
            raise TimeoutError(f"expected {count} bytes, received {len(data)}")
    return bytes(data)


def read_until(port, token: bytes, timeout: float) -> bytes:
    data = bytearray()
    deadline = time.time() + timeout
    while token not in data:
        chunk = port.read(1024)
        if chunk:
            data.extend(chunk)
        elif time.time() > deadline:
            raise TimeoutError(f"missing {token!r}; received {bytes(data)!r}")
    return bytes(data)


def compare_int8(a: np.ndarray, b: np.ndarray) -> dict:
    delta = a.astype(np.int16) - b.astype(np.int16)
    return {
        "exact_equal_fraction": float(np.mean(delta == 0)),
        "mean_absolute_difference": float(np.mean(np.abs(delta))),
        "max_absolute_difference": int(np.max(np.abs(delta))),
    }


def load_module_checkpoint(path: Path):
    checkpoint = torch.load(path, map_location="cpu")
    cfg = checkpoint["config"]
    model = train_cv.LogMelCNN(
        cfg["sample_rate"], cfg["n_mels"], cfg["n_fft"], cfg["hop_length"],
        input_normalization=cfg["input_normalization"],
        rfn_lambda=cfg.get("rfn_lambda", 0.0), frontend=cfg.get("frontend", "logmel"),
        pcen_alpha=cfg.get("pcen_alpha", 0.6), pcen_delta=cfg.get("pcen_delta", 2.0),
        pcen_r=cfg.get("pcen_r", 0.25), pcen_time_constant=cfg.get("pcen_time_constant", 0.8),
        pcen_eps=cfg.get("pcen_eps", 1e-6),
    ).eval()
    return checkpoint, model


def pc_feature(pcm: np.ndarray, model, scale: float, zero: int) -> np.ndarray:
    waveform = (pcm[0::2].astype(np.float32) + pcm[1::2].astype(np.float32)) / (2.0 * 32768.0)
    wave = torch.from_numpy(waveform.copy()).float()
    wave = wave - wave.mean()
    wave = wave / wave.abs().amax().clamp_min(1e-6)
    with torch.inference_mode():
        spectrum = torch.stft(wave.unsqueeze(0), n_fft=model.n_fft, hop_length=model.hop_length,
                              window=model.stft_window.float(), return_complex=True, center=True)
        mel = torch.matmul(model.mel_filter.float(), spectrum.abs().pow(2.0)).clamp_min(1e-8)
        value = model._pcen(mel) if model.frontend == "pcen" else torch.log(mel)
        value = (value - value.mean()) / value.std().clamp_min(1e-5)
    quantized = np.round(value[0].numpy() / scale + zero)
    return np.clip(quantized, -128, 127).astype(np.int8).reshape(-1)


class TFLiteRunner:
    def __init__(self, path: Path):
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
        # TensorFlow imports JAX only for an optional conversion path.  A local
        # JAX installation with an incompatible ml_dtypes version must not stop
        # ordinary TFLite inference, which does not use that path.
        if "jax" not in sys.modules:
            jax_stub = types.ModuleType("jax")
            def _unused_xla_computation(*_args, **_kwargs):
                raise RuntimeError("JAX conversion is not used by parity validation")
            jax_stub.xla_computation = _unused_xla_computation
            sys.modules["jax"] = jax_stub
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=str(path))
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]

    def __call__(self, feature: np.ndarray) -> np.ndarray:
        self.interpreter.set_tensor(self.input["index"], feature.reshape(1, 64, 151, 1))
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output["index"])[0].astype(np.int8)


def probability(raw: np.ndarray, scale: float, zero: int) -> np.ndarray:
    logits = (raw.astype(np.float32) - zero) * scale
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()


def request_pcm(port, pcm: np.ndarray):
    port.reset_input_buffer()
    port.write(b"P")
    port.flush()
    read_until(port, b"HOST_PCM_READY bytes=48000\r\n", 5)
    port.write(pcm.astype("<i2", copy=False).tobytes())
    port.flush()
    if read_exact(port, 4, 20) != b"HPT1":
        raise RuntimeError("unexpected ESP32 parity packet")
    rate, pcm_count, feature_count, output_count, feature_ms, inference_us = struct.unpack(
        "<IIIIII", read_exact(port, 24, 5))
    if (rate, pcm_count, feature_count, output_count) != (16000, 24000, 9664, 6):
        raise RuntimeError(f"unexpected metadata {(rate, pcm_count, feature_count, output_count)}")
    feature = np.frombuffer(read_exact(port, feature_count, 10), dtype=np.int8).copy()
    output = np.frombuffer(read_exact(port, output_count, 5), dtype=np.int8).copy()
    return feature, output, {"feature_ms": feature_ms, "inference_ms": inference_us / 1000.0}


def select_rows(manifest: Path, per_class: int) -> list[dict]:
    grouped = {name: [] for name in CLASSES}
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["label"] in grouped:
                grouped[row["label"]].append(row)
    if any(not rows for rows in grouped.values()):
        raise ValueError("Manifest must contain all six classes")
    return [grouped[name][index] for name in CLASSES for index in
            np.linspace(0, len(grouped[name]) - 1, min(per_class, len(grouped[name])), dtype=int)]


def decision_pcm(row: dict) -> np.ndarray:
    info = sf.info(row["wav_path"])
    rate = int(info.samplerate)
    center = float(row["start_sec"]) + 0.5 * float(row["duration_sec"])
    start = max(0.0, center - 1.5)
    audio, _ = sf.read(row["wav_path"], start=int(round(start * rate)),
                       frames=int(round(3.0 * rate)), dtype="float32", always_2d=True)
    wave = audio.mean(axis=1)
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        wave = resample_poly(wave, 16000 // divisor, rate // divisor)
    wave = np.pad(wave, (0, max(0, 48000 - wave.size)))[:48000]
    return np.clip(np.round(wave * 32768.0), -32768, 32767).astype(np.int16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--frontend", choices=("logmel", "pcen"), default="logmel")
    parser.add_argument("--per-class", type=int, default=5)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--verification", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.per_class < 1:
        parser.error("--per-class must be at least 1")
    symbol = "model_overlap025" if args.frontend == "pcen" else "model_overlap025_logmel"
    if args.model is None:
        args.model = PACKAGE / "model" / f"{symbol}.tflite"
    if args.verification is None:
        args.verification = PACKAGE / "model" / f"{symbol}_verification.json"
    if args.output is None:
        args.output = PACKAGE / "validation" / f"{args.frontend.upper()}_EXACT3S_PARITY.json"

    verification = json.loads(args.verification.read_text(encoding="utf-8"))
    in_scale = float(verification["input_quantization"]["scale"])
    in_zero = int(verification["input_quantization"]["zero_point"])
    out_scale = float(verification["output_quantization"]["scale"])
    out_zero = int(verification["output_quantization"]["zero_point"])
    checkpoint, frontend = load_module_checkpoint(args.checkpoint)
    if frontend.frontend != args.frontend or verification.get("frontend") != args.frontend:
        raise ValueError("Checkpoint, verification report, and selected frontend must match")
    runner = TFLiteRunner(args.model)

    import serial
    reports = []
    with serial.Serial(args.port, 921600, timeout=0.1, write_timeout=10) as port:
        time.sleep(1)
        port.write(b"E")
        time.sleep(0.5)
        port.reset_input_buffer()
        port.write(b"B" if args.frontend == "pcen" else b"A")
        port.flush()
        response = read_until(port, b"FRONTEND_SELECTED", 5) + port.read(256)
        if f"frontend={args.frontend}".encode() not in response:
            raise RuntimeError(f"ESP32 frontend selection failed: {response!r}")
        for row in select_rows(args.manifest, args.per_class):
            pcm3 = decision_pcm(row)
            pc_probs, esp_probs, contexts = [], [], []
            for offset in CONTEXT_OFFSETS:
                pcm = pcm3[offset:offset + 24000]
                pc_int8 = pc_feature(pcm, frontend, in_scale, in_zero)
                esp_int8, esp_raw, timing = request_pcm(port, pcm)
                pc_raw = runner(pc_int8)
                pc_probs.append(probability(pc_raw, out_scale, out_zero))
                esp_probs.append(probability(esp_raw, out_scale, out_zero))
                context_report = {"feature": compare_int8(pc_int8, esp_int8),
                                  "raw_output": compare_int8(pc_raw, esp_raw),
                                  "prediction_equal": bool(pc_raw.argmax() == esp_raw.argmax()),
                                  "timing": timing}
                if not context_report["prediction_equal"]:
                    context_report["pc_raw"] = pc_raw.tolist()
                    context_report["esp32_raw"] = esp_raw.tolist()
                    context_report["pc_class"] = CLASSES[int(pc_raw.argmax())]
                    context_report["esp32_class"] = CLASSES[int(esp_raw.argmax())]
                contexts.append(context_report)
            pc_avg, esp_avg = np.mean(pc_probs, axis=0), np.mean(esp_probs, axis=0)
            reports.append({"source_label": row["label"], "source_wav": row["wav_path"],
                            "source_start_sec": row["start_sec"], "contexts": contexts,
                            "aggregate_pc": CLASSES[int(pc_avg.argmax())],
                            "aggregate_esp32": CLASSES[int(esp_avg.argmax())],
                            "aggregate_prediction_equal": bool(pc_avg.argmax() == esp_avg.argmax()),
                            "max_probability_difference": float(np.max(np.abs(pc_avg - esp_avg)))})

    contexts = [item for report in reports for item in report["contexts"]]
    summary = {
        "package_version": "v5.0.0",
        "frontend": checkpoint["config"].get("frontend"),
        "pcen": {key: checkpoint["config"].get(key) for key in
                 ("pcen_alpha", "pcen_delta", "pcen_r", "pcen_time_constant", "pcen_eps")},
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "tested_decisions": len(reports), "tested_subwindows": len(contexts),
        "feature_exact_equal_fraction_mean": float(np.mean([c["feature"]["exact_equal_fraction"] for c in contexts])),
        "feature_max_absolute_difference": int(max(c["feature"]["max_absolute_difference"] for c in contexts)),
        "subwindow_prediction_agreement": float(np.mean([c["prediction_equal"] for c in contexts])),
        "aggregate_prediction_agreement": float(np.mean([r["aggregate_prediction_equal"] for r in reports])),
        "mean_feature_ms_per_context": float(np.mean([c["timing"]["feature_ms"] for c in contexts])),
        "mean_inference_ms_per_context": float(np.mean([c["timing"]["inference_ms"] for c in contexts])),
        "speed_patch_runtime_target": "CNN inference below 1000 ms per 1.5 s context",
        "decisions": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "decisions"}, indent=2))


if __name__ == "__main__":
    main()
