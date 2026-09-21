"""Export a trained compact PCEN CNN checkpoint as a named full-int8 TFLite model.

Unlike the original three-class exporter, this version reads the class count,
frontend duration and resampling mode from the checkpoint.  It therefore supports
the six-class 1.5 s context models and the binary independent heads without
changing their inference graph.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import types
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

# TensorFlow imports JAX only for an optional JAX-to-TFLite path.  The machine's
# unrelated JAX installation currently expects a newer ml_dtypes and otherwise
# prevents TensorFlow itself from importing.  This local stub leaves the Python
# environment untouched; this exporter never calls the optional JAX path.
if "jax" not in sys.modules:
    jax_stub = types.ModuleType("jax")
    def _unused_xla_computation(*_args, **_kwargs):
        raise RuntimeError("JAX conversion is not used by this exporter")
    jax_stub.xla_computation = _unused_xla_computation
    sys.modules["jax"] = jax_stub

import numpy as np
import tensorflow as tf
import torch

import train_cv as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--representative-samples", type=int, default=192)
    parser.add_argument("--verification-samples", type=int, default=240)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def balanced_rows(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)
    selected = []
    cursor = {name: 0 for name in grouped}
    names = sorted(grouped)
    while len(selected) < min(count, len(rows)):
        changed = False
        for name in names:
            index = cursor[name]
            if index < len(grouped[name]) and len(selected) < count:
                selected.append(grouped[name][index])
                cursor[name] += 1
                changed = True
        if not changed:
            break
    return selected


def make_keras_model(frames: int, classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(64, frames, 1), dtype=tf.float32, name="logmel")
    x = tf.keras.layers.Conv2D(16, 3, padding="same", use_bias=True, name="conv1")(inputs)
    x = tf.keras.layers.ReLU(name="relu1")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, name="pool1")(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=True, name="conv2")(x)
    x = tf.keras.layers.ReLU(name="relu2")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, name="pool2")(x)
    x = tf.keras.layers.Conv2D(48, 3, padding="same", use_bias=True, name="conv3")(x)
    x = tf.keras.layers.ReLU(name="relu3")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average")(x)
    outputs = tf.keras.layers.Dense(classes, name="classifier")(x)
    return tf.keras.Model(inputs, outputs)


def copy_weights(torch_model: base.LogMelCNN, keras_model: tf.keras.Model) -> None:
    for keras_name, conv_index, bn_index in (("conv1", 0, 1), ("conv2", 4, 5), ("conv3", 8, 9)):
        conv = torch_model.features[conv_index]
        bn = torch_model.features[bn_index]
        scale = bn.weight.detach().numpy() / np.sqrt(bn.running_var.detach().numpy() + bn.eps)
        weight = conv.weight.detach().numpy() * scale[:, None, None, None]
        bias = bn.bias.detach().numpy() + (conv.bias.detach().numpy() - bn.running_mean.detach().numpy()) * scale
        keras_model.get_layer(keras_name).set_weights([weight.transpose(2, 3, 1, 0), bias])
    linear = torch_model.classifier[2]
    keras_model.get_layer("classifier").set_weights(
        [linear.weight.detach().numpy().T, linear.bias.detach().numpy()]
    )


@torch.inference_mode()
def frontend_features(model: base.LogMelCNN, waveform: torch.Tensor) -> np.ndarray:
    waveform = waveform.unsqueeze(0)
    spectrum = torch.stft(
        waveform.float(), n_fft=model.n_fft, hop_length=model.hop_length,
        window=model.stft_window.float(), return_complex=True, center=True,
    )
    mel = torch.matmul(model.mel_filter.float(), spectrum.abs().pow(2.0)).clamp_min(1e-8)
    values = model._pcen(mel) if model.frontend == "pcen" else torch.log(mel)
    layer_normalized = (
        values - values.mean((1, 2), keepdim=True)
    ) / values.std((1, 2), keepdim=True).clamp_min(1e-5)
    if model.input_normalization == "rfn":
        frequency_normalized = (
            values - values.mean(dim=2, keepdim=True)
        ) / values.std(dim=2, keepdim=True).clamp_min(1e-5)
        values = (
            model.rfn_lambda * layer_normalized
            + (1.0 - model.rfn_lambda) * frequency_normalized
        )
    else:
        values = layer_normalized
    return values.unsqueeze(1).permute(0, 2, 3, 1).numpy().astype(np.float32)


def make_features(
    model: base.LogMelCNN, rows: list[dict[str, str]], config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = base.AudioWindowDataset(
        rows,
        int(config["sample_rate"]),
        float(config["window_sec"]),
        str(config["resample_mode"]),
        float(config.get("input_context_sec", config["window_sec"])),
        str(config.get("context_alignment", "centered")),
    )
    features, labels = [], []
    for index in range(len(dataset)):
        waveform, label = dataset[index]
        features.append(frontend_features(model, waveform)[0])
        labels.append(label)
    return np.stack(features), np.asarray(labels)


def write_header(payload: bytes, path: Path, symbol: str) -> None:
    lines = ["#pragma once", "#include <cstdint>", "", f"alignas(16) const unsigned char {symbol}[] = {{"]
    for offset in range(0, len(payload), 12):
        lines.append("  " + ", ".join(f"0x{x:02x}" for x in payload[offset:offset + 12]) + ",")
    lines += ["};", f"constexpr unsigned int {symbol}_len = {len(payload)};", ""]
    path.write_text("\n".join(lines), encoding="ascii")


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    config = dict(payload["config"])
    names = list(payload["class_names"])
    base.configure_class_names(names)
    model = base.LogMelCNN(
        int(config["sample_rate"]), int(config["n_mels"]),
        int(config["n_fft"]), int(config["hop_length"]),
        input_normalization=str(config.get("input_normalization", "global")),
        rfn_lambda=float(config.get("rfn_lambda", 0.5)),
        frontend=str(config.get("frontend", "logmel")),
        pcen_alpha=float(config.get("pcen_alpha", 0.60)),
        pcen_delta=float(config.get("pcen_delta", 2.0)),
        pcen_r=float(config.get("pcen_r", 0.25)),
        pcen_time_constant=float(config.get("pcen_time_constant", 0.8)),
        pcen_eps=float(config.get("pcen_eps", 1e-6)),
    )
    model.load_state_dict(payload["model_state"])
    model.eval()

    train = balanced_rows(read_rows(args.data_dir / "train.csv"), args.representative_samples)
    test = balanced_rows(read_rows(args.data_dir / "test.csv"), args.verification_samples)
    representative, _ = make_features(model, train, config)
    verification, labels = make_features(model, test, config)
    frames = int(representative.shape[2])
    keras = make_keras_model(frames, len(names))
    keras(np.zeros((1, 64, frames, 1), np.float32), training=False)
    copy_weights(model, keras)
    keras_logits = keras(verification, training=False).numpy()
    torch_logits = np.concatenate([
        model.classifier(model.features(torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0))).detach().numpy()
        for x in verification
    ])
    mapping_error = float(np.max(np.abs(torch_logits - keras_logits)))
    if mapping_error > 2e-4:
        raise RuntimeError(f"PyTorch/Keras mapping mismatch: {mapping_error}")

    def representative_dataset():
        for feature in representative:
            yield [feature[None, ...]]

    with tempfile.TemporaryDirectory(prefix="mosquito_tflite_") as saved:
        tf.saved_model.save(keras, saved)
        converter = tf.lite.TFLiteConverter.from_saved_model(saved)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite = converter.convert()

    model_path = output / f"{args.symbol}.tflite"
    header_path = output / f"{args.symbol}.h"
    model_path.write_bytes(tflite)
    write_header(tflite, header_path, args.symbol)
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    predicted, max_error = [], 0.0
    in_scale, in_zero = input_info["quantization"]
    out_scale, out_zero = output_info["quantization"]
    for feature, reference in zip(verification, keras_logits):
        quantized = np.clip(np.round(feature[None, ...] / in_scale + in_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(input_info["index"], quantized)
        interpreter.invoke()
        raw = interpreter.get_tensor(output_info["index"])[0]
        logits = (raw.astype(np.float32) - out_zero) * out_scale
        predicted.append(int(np.argmax(logits)))
        max_error = max(max_error, float(np.max(np.abs(reference - logits))))
    predicted = np.asarray(predicted)
    confusion = np.zeros((len(names), len(names)), dtype=np.int64)
    for target, prediction in zip(labels, predicted):
        confusion[int(target), int(prediction)] += 1
    per_class_f1 = []
    for class_index in range(len(names)):
        true_positive = int(confusion[class_index, class_index])
        false_positive = int(confusion[:, class_index].sum()) - true_positive
        false_negative = int(confusion[class_index, :].sum()) - true_positive
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        per_class_f1.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "frontend": str(config.get("frontend", "logmel")),
        "input_normalization": str(config.get("input_normalization", "global")),
        "class_names": names,
        "model_bytes": len(tflite),
        "input_shape": input_info["shape"].tolist(),
        "input_quantization": {"scale": in_scale, "zero_point": in_zero},
        "output_quantization": {"scale": out_scale, "zero_point": out_zero},
        "verification_samples": len(labels),
        "pytorch_keras_max_abs_logit_error": mapping_error,
        "keras_tflite_max_abs_logit_error": max_error,
        "prediction_agreement": float(np.mean(keras_logits.argmax(1) == predicted)),
        "tflite_accuracy_on_verification_subset": float(np.mean(predicted == labels)),
        "tflite_macro_f1_on_verification_subset": float(np.mean(per_class_f1)),
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_axes": {"rows": "true labels", "columns": "predicted labels"},
    }
    (output / f"{args.symbol}_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
