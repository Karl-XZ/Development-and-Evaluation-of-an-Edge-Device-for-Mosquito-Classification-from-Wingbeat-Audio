"""Five-fold training for the initial mosquito-wingbeat classifier.

The script deliberately keeps recordings separate across folds: windows from one
recording never occur in both a fold's training and validation sets.  The held
out ``test.csv`` is never used for model selection; after all folds finish it is
evaluated once with an ensemble of the five best fold checkpoints.

Example (run from this folder)::

    python train_3class_cv.py --epochs 30 --batch-size 64 --workers 4

Outputs are placed under ``runs/3class_cv`` by default.  The terminal progress
bars show per-batch loss, accuracy, throughput and remaining time.  Each run
also saves machine-readable histories, per-fold confusion matrices/checkpoints,
and a compact cross-validation/test summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm


CLASS_NAMES = ["ae_aegypti", "cx_quin", "no_mosquito"]
NUM_CLASSES = len(CLASS_NAMES)


def configure_class_names(names: Iterable[str]) -> None:
    global CLASS_NAMES, NUM_CLASSES
    configured = [str(name).strip() for name in names if str(name).strip()]
    if len(configured) < 2 or len(set(configured)) != len(configured):
        raise ValueError("Class names must contain at least two unique values")
    CLASS_NAMES = configured
    NUM_CLASSES = len(CLASS_NAMES)


@dataclass
class RunConfig:
    data_dir: str
    output: str
    sample_rate: int
    window_sec: float
    n_mels: int
    n_fft: int
    hop_length: int
    epochs: int
    batch_size: int
    workers: int
    learning_rate: float
    weight_decay: float
    patience: int
    seed: int
    device: str
    folds: list[int]
    resample_mode: str
    channel_response_npy: str | None
    channel_augment_probability: float
    channel_augment_scale_min: float
    channel_augment_scale_max: float
    freq_mix_probability: float
    freq_mix_alpha: float
    input_normalization: str
    rfn_lambda: float
    input_context_sec: float
    context_alignment: str
    class_names: list[str]
    presence_background_mix_probability: float
    presence_background_snr_min_db: float
    presence_background_snr_max_db: float
    balanced_sampling: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Five-fold mosquito audio classifier training")
    parser.add_argument("--data-dir", type=Path, default=Path("mosquito_3class"),
                        help="Folder containing train.csv, validation.csv and test.csv")
    parser.add_argument(
        "--class-names",
        type=str,
        default="",
        help=(
            "Optional comma-separated class order. If omitted, read the "
            "classes mapping from data-dir/dataset_summary.json."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("runs/3class_cv"),
                        help="Output folder (created if needed)")
    parser.add_argument(
        "--init-run",
        type=Path,
        default=None,
        help=(
            "Optional completed five-fold run used only to initialise the matching "
            "fold before training. The source checkpoints are read-only and the new "
            "run still trains and validates independently."
        ),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4,
                        help="Audio loading worker processes; use 0 to debug")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8,
                        help="Early stopping patience, measured in epochs")
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument(
        "--resample-mode",
        choices=["polyphase", "esp32_pair"],
        default="polyphase",
        help=(
            "polyphase keeps the original training frontend; esp32_pair first "
            "converts source audio to 16 kHz and then applies the deployed "
            "ESP32 adjacent-sample average decimator"
        ),
    )
    parser.add_argument("--window-sec", type=float, default=0.5)
    parser.add_argument(
        "--input-context-sec",
        type=float,
        default=0.5,
        help=(
            "Audio duration presented to the CNN for each labelled window. "
            "May exceed --window-sec without changing the prediction cadence."
        ),
    )
    parser.add_argument(
        "--context-alignment",
        choices=["centered", "causal"],
        default="centered",
        help=(
            "centered splits extra context before/after the labelled window; "
            "causal places all extra context before it"
        ),
    )
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--n-fft", type=int, default=256)
    parser.add_argument("--hop-length", type=int, default=80)
    parser.add_argument("--folds", type=str, default="0,1,2,3,4",
                        help="Comma-separated CV fold IDs. Default trains all five; e.g. --folds 0 for a smoke test")
    parser.add_argument(
        "--resume-completed-folds",
        action="store_true",
        help=(
            "Reuse folds that already completed all epochs or satisfied early "
            "stopping. Incomplete folds restart from epoch 1."
        ),
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--max-train-samples", type=int, default=0,
                        help="Optional balanced cap for a quick smoke test (0 means all)")
    parser.add_argument("--max-validation-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument(
        "--channel-response-npy",
        type=Path,
        default=None,
        help=(
            "Optional [num_responses, n_mels] mean-centred log-power response "
            "curves used only for training augmentation"
        ),
    )
    parser.add_argument(
        "--channel-augment-probability",
        type=float,
        default=0.0,
        help="Per-training-window probability of empirical channel augmentation",
    )
    parser.add_argument("--channel-augment-scale-min", type=float, default=0.25)
    parser.add_argument("--channel-augment-scale-max", type=float, default=1.25)
    parser.add_argument(
        "--freq-mix-probability",
        type=float,
        default=0.0,
        help=(
            "Training-only probability of BioDCASE-style per-frequency "
            "MixStyle; adds no inference parameters or operations"
        ),
    )
    parser.add_argument(
        "--freq-mix-alpha",
        type=float,
        default=0.4,
        help="Beta-distribution alpha used by frequency MixStyle",
    )
    parser.add_argument(
        "--input-normalization",
        choices=["global", "rfn"],
        default="global",
        help=(
            "global reproduces the original per-window normalization; rfn "
            "uses Relaxed Instance Frequency-wise Normalization"
        ),
    )
    parser.add_argument(
        "--rfn-lambda",
        type=float,
        default=0.5,
        help=(
            "RFN relaxation in [0,1]: 1 is the original global layer "
            "normalization and 0 is full per-Mel temporal normalization"
        ),
    )
    parser.add_argument(
        "--presence-background-mix-probability",
        type=float,
        default=0.0,
        help=(
            "Training-only presence augmentation. For mosquito windows, mix a "
            "random no_mosquito training window at a random SNR. This is only "
            "valid for the binary mosquito/no_mosquito task."
        ),
    )
    parser.add_argument("--presence-background-snr-min-db", type=float, default=-5.0)
    parser.add_argument("--presence-background-snr-max-db", type=float, default=15.0)
    parser.add_argument(
        "--balanced-sampling",
        action="store_true",
        help=(
            "Draw an equal expected number of examples from every class during "
            "training. When enabled, loss weights remain uniform to avoid "
            "double-correcting class imbalance."
        ),
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.patience < 1:
        parser.error("epochs, batch-size and patience must be positive; workers must be non-negative")
    if args.input_context_sec + 1e-9 < args.window_sec:
        parser.error("--input-context-sec must be at least --window-sec")
    try:
        args.fold_ids = [int(value.strip()) for value in args.folds.split(",") if value.strip()]
    except ValueError as exc:
        parser.error(f"--folds must be comma-separated integers: {exc}")
    if not args.fold_ids:
        parser.error("at least one fold is required")
    if not 0.0 <= args.channel_augment_probability <= 1.0:
        parser.error("--channel-augment-probability must be between 0 and 1")
    if args.channel_augment_scale_max < args.channel_augment_scale_min:
        parser.error(
            "--channel-augment-scale-max must be >= --channel-augment-scale-min"
        )
    if args.channel_augment_probability > 0 and args.channel_response_npy is None:
        parser.error(
            "--channel-response-npy is required when channel augmentation is enabled"
        )
    if not 0.0 <= args.freq_mix_probability <= 1.0:
        parser.error("--freq-mix-probability must be between 0 and 1")
    if args.freq_mix_alpha <= 0:
        parser.error("--freq-mix-alpha must be positive")
    if not 0.0 <= args.rfn_lambda <= 1.0:
        parser.error("--rfn-lambda must be between 0 and 1")
    if not 0.0 <= args.presence_background_mix_probability <= 1.0:
        parser.error("--presence-background-mix-probability must be between 0 and 1")
    if args.presence_background_snr_max_db < args.presence_background_snr_min_db:
        parser.error(
            "--presence-background-snr-max-db must be >= "
            "--presence-background-snr-min-db"
        )
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Reproducibility is more useful here than the small throughput gain from benchmark selection.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"recording_id", "wav_path", "start_sec", "duration_sec", "label", "label_id", "cv_fold"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    return rows


def cap_balanced(rows: list[dict[str, str]], maximum: int, seed: int) -> list[dict[str, str]]:
    """Return at most ``maximum`` rows while preserving an equal class mix."""
    if maximum <= 0 or len(rows) <= maximum:
        return rows
    per_class = maximum // NUM_CLASSES
    if per_class == 0:
        raise ValueError(
            f"--max-*-samples must be at least the number of classes ({NUM_CLASSES})"
        )
    grouped: dict[str, list[dict[str, str]]] = {name: [] for name in CLASS_NAMES}
    for row in rows:
        grouped[row["label"]].append(row)
    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    for name in CLASS_NAMES:
        if len(grouped[name]) < per_class:
            raise ValueError(f"Cannot make a balanced sample: only {len(grouped[name])} rows for {name}")
        selected.extend(rng.sample(grouped[name], per_class))
    rng.shuffle(selected)
    return selected


class AudioWindowDataset(Dataset[tuple[torch.Tensor, int]]):
    """Loads a manifest window lazily, downmixes and resamples it to a fixed tensor."""

    def __init__(
        self,
        rows: list[dict[str, str]],
        target_rate: int,
        window_sec: float,
        resample_mode: str = "polyphase",
        input_context_sec: float | None = None,
        context_alignment: str = "centered",
    ) -> None:
        self.rows = rows
        self.target_rate = target_rate
        self.window_sec = window_sec
        self.input_context_sec = (
            window_sec if input_context_sec is None else input_context_sec
        )
        if self.input_context_sec + 1e-9 < window_sec:
            raise ValueError("input_context_sec must be at least window_sec")
        if context_alignment not in {"centered", "causal"}:
            raise ValueError(f"Unknown context alignment: {context_alignment}")
        self.context_alignment = context_alignment
        self.target_frames = int(round(target_rate * self.input_context_sec))
        self.resample_mode = resample_mode
        if resample_mode == "esp32_pair" and target_rate != 8000:
            raise ValueError("esp32_pair currently requires --sample-rate 8000")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        wav_path = row["wav_path"]
        start_sec = float(row["start_sec"])
        duration_sec = float(row["duration_sec"])
        extra_context = max(0.0, self.input_context_sec - duration_sec)
        if self.context_alignment == "centered":
            read_start_sec = start_sec - 0.5 * extra_context
        else:
            read_start_sec = start_sec - extra_context
        try:
            info = sf.info(wav_path)
            source_rate = int(info.samplerate)
            desired_start_frame = int(round(read_start_sec * source_rate))
            left_padding = max(0, -desired_start_frame)
            start_frame = max(0, desired_start_frame)
            frames = max(
                1,
                int(round(self.input_context_sec * source_rate)) - left_padding,
            )
            audio, _ = sf.read(wav_path, start=start_frame, frames=frames, dtype="float32", always_2d=True)
        except Exception as exc:  # pragma: no cover - message is useful with a damaged source recording
            raise RuntimeError(f"Could not read sample {index} from {wav_path}: {exc}") from exc

        waveform = audio.mean(axis=1) if audio.size else audio.reshape(-1)
        if left_padding:
            waveform = np.pad(waveform, (left_padding, 0))
        if self.resample_mode == "esp32_pair" and waveform.size:
            # The deployed INMP441 path captures at 16 kHz and reduces it to
            # 8 kHz by averaging adjacent samples.  Original training files are
            # 44.1 kHz, so first convert them to the board capture rate, then
            # reproduce that exact low-cost decimator.
            recorder_rate = 16000
            if source_rate != recorder_rate:
                divisor = math.gcd(source_rate, recorder_rate)
                waveform = resample_poly(
                    waveform, recorder_rate // divisor, source_rate // divisor
                ).astype("float32")
            recorder_frames = self.target_frames * 2
            if waveform.size < recorder_frames:
                waveform = np.pad(waveform, (0, recorder_frames - waveform.size))
            waveform = waveform[:recorder_frames]
            waveform = (
                waveform[0::2].astype(np.float32)
                + waveform[1::2].astype(np.float32)
            ) / 2.0
        elif source_rate != self.target_rate and waveform.size:
            divisor = math.gcd(source_rate, self.target_rate)
            waveform = resample_poly(
                waveform, self.target_rate // divisor, source_rate // divisor
            ).astype("float32")
        if waveform.size < self.target_frames:
            # A final short detector region is padded rather than discarded; manifests normally hold 0.5 s exactly.
            waveform = np.pad(waveform, (0, self.target_frames - waveform.size))
        waveform = waveform[:self.target_frames]
        if not np.isfinite(waveform).all():
            raise RuntimeError(f"Non-finite samples found in {wav_path} at {start_sec:.3f} seconds")
        tensor = torch.from_numpy(waveform.copy()).float()
        tensor = tensor - tensor.mean()
        tensor = tensor / tensor.abs().amax().clamp_min(1e-6)
        return tensor, int(row["label_id"])


class PresenceBackgroundMixDataset(Dataset[tuple[torch.Tensor, int]]):
    """Training-only acoustic-domain augmentation for binary presence models.

    A labelled mosquito waveform is mixed with a randomly selected
    no_mosquito waveform from the same training fold. The mixture therefore
    simulates a source recording being replayed into a different room/background
    without using any physical playback test recording for training.
    """

    def __init__(
        self,
        base: AudioWindowDataset,
        rows: list[dict[str, str]],
        probability: float,
        snr_min_db: float,
        snr_max_db: float,
    ) -> None:
        self.base = base
        self.probability = float(probability)
        self.snr_min_db = float(snr_min_db)
        self.snr_max_db = float(snr_max_db)
        background_rows = [row for row in rows if row["label"] == "no_mosquito"]
        if not background_rows:
            raise ValueError("Presence background mixing requires no_mosquito rows")
        self.background = AudioWindowDataset(
            background_rows,
            base.target_rate,
            base.window_sec,
            base.resample_mode,
            base.input_context_sec,
            base.context_alignment,
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        waveform, label = self.base[index]
        if label != 0 or torch.rand(()) >= self.probability:
            return waveform, label
        background_index = int(torch.randint(len(self.background), (1,)).item())
        background, _ = self.background[background_index]
        signal_rms = waveform.square().mean().sqrt().clamp_min(1e-5)
        background_rms = background.square().mean().sqrt().clamp_min(1e-5)
        snr_db = torch.empty(()).uniform_(self.snr_min_db, self.snr_max_db)
        background_scale = signal_rms / (
            background_rms * torch.pow(torch.tensor(10.0), snr_db / 20.0)
        )
        mixed = waveform + background * background_scale
        mixed = mixed - mixed.mean()
        mixed = mixed / mixed.abs().amax().clamp_min(1e-6)
        return mixed, label


def hz_to_mel(frequency: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + frequency / 700.0)


def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


def build_mel_filter(sample_rate: int, n_fft: int, n_mels: int) -> torch.Tensor:
    """Minimal Slaney-style triangular mel filter bank, avoiding a torchaudio dependency."""
    n_freqs = n_fft // 2 + 1
    mel_points = torch.linspace(hz_to_mel(torch.tensor(0.0)), hz_to_mel(torch.tensor(sample_rate / 2)), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = torch.floor((n_fft + 1) * hz_points / sample_rate).long().clamp(0, n_freqs - 1)
    filters = torch.zeros(n_mels, n_freqs)
    for mel_index in range(n_mels):
        left, center, right = bins[mel_index : mel_index + 3].tolist()
        if center <= left:
            center = min(left + 1, n_freqs - 1)
        if right <= center:
            right = min(center + 1, n_freqs)
        if center > left:
            filters[mel_index, left:center] = torch.linspace(0.0, 1.0, center - left)
        if right > center:
            filters[mel_index, center:right] = torch.linspace(1.0, 0.0, right - center)
    return filters


class LogMelCNN(nn.Module):
    """A compact log-mel CNN suitable as a PC baseline before ESP32 optimisation."""

    def __init__(
        self,
        sample_rate: int,
        n_mels: int,
        n_fft: int,
        hop_length: int,
        channel_responses: torch.Tensor | None = None,
        channel_augment_probability: float = 0.0,
        channel_augment_scale_min: float = 0.25,
        channel_augment_scale_max: float = 1.25,
        freq_mix_probability: float = 0.0,
        freq_mix_alpha: float = 0.4,
        input_normalization: str = "global",
        rfn_lambda: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        if channel_responses is None:
            channel_responses = torch.empty((0, n_mels), dtype=torch.float32)
        if channel_responses.ndim != 2 or channel_responses.shape[1] != n_mels:
            raise ValueError(
                f"channel_responses must have shape [N, {n_mels}], "
                f"received {tuple(channel_responses.shape)}"
            )
        self.channel_augment_probability = float(channel_augment_probability)
        self.channel_augment_scale_min = float(channel_augment_scale_min)
        self.channel_augment_scale_max = float(channel_augment_scale_max)
        self.freq_mix_probability = float(freq_mix_probability)
        self.freq_mix_alpha = float(freq_mix_alpha)
        if input_normalization not in {"global", "rfn"}:
            raise ValueError(f"Unknown input normalization: {input_normalization}")
        if not 0.0 <= rfn_lambda <= 1.0:
            raise ValueError("rfn_lambda must be between 0 and 1")
        self.input_normalization = input_normalization
        self.rfn_lambda = float(rfn_lambda)
        # Training-only asset: deliberately excluded from checkpoints so the
        # inference architecture and existing ESP32 export stay unchanged.
        self.register_buffer(
            "channel_responses",
            channel_responses.float(),
            persistent=False,
        )
        self.register_buffer("stft_window", torch.hann_window(n_fft), persistent=False)
        self.register_buffer("mel_filter", build_mel_filter(sample_rate, n_fft, n_mels), persistent=False)
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 48, kernel_size=3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(48, NUM_CLASSES))

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        # STFT/mel stays float32 even when the CNN itself uses CUDA AMP.  In FP16,
        # low-energy mel bins can underflow before log(), producing -inf/NaN.
        with torch.amp.autocast(device_type=waveforms.device.type, enabled=False):
            spectrum = torch.stft(waveforms.float(), n_fft=self.n_fft, hop_length=self.hop_length,
                                  window=self.stft_window.float(), return_complex=True, center=True)
            power = spectrum.abs().pow(2.0)
            mel = torch.matmul(self.mel_filter.float(), power)
            log_mel = torch.log(mel.clamp_min(1e-8))
            if (
                self.training
                and self.channel_augment_probability > 0.0
                and self.channel_responses.shape[0] > 0
            ):
                batch = log_mel.shape[0]
                response_count = self.channel_responses.shape[0]
                first = self.channel_responses[
                    torch.randint(response_count, (batch,), device=log_mel.device)
                ]
                second = self.channel_responses[
                    torch.randint(response_count, (batch,), device=log_mel.device)
                ]
                mixture = torch.rand(
                    (batch, 1), device=log_mel.device, dtype=log_mel.dtype
                )
                response = mixture * first + (1.0 - mixture) * second
                scale = torch.empty(
                    (batch, 1), device=log_mel.device, dtype=log_mel.dtype
                ).uniform_(
                    self.channel_augment_scale_min,
                    self.channel_augment_scale_max,
                )
                apply_mask = (
                    torch.rand((batch, 1), device=log_mel.device)
                    < self.channel_augment_probability
                )
                response = response * scale * apply_mask
                log_mel = log_mel + response.unsqueeze(2)
            if (
                self.training
                and self.freq_mix_probability > 0.0
                and log_mel.shape[0] > 1
                and torch.rand((), device=log_mel.device)
                < self.freq_mix_probability
            ):
                # Adapted from the official BioDCASE 2026 baseline's
                # apply_freq_mixstyle: exchange per-Mel temporal statistics
                # across examples while preserving normalized content.
                frequency_mean = log_mel.mean(dim=2)
                frequency_std = log_mel.std(dim=2).clamp_min(1e-6)
                permutation = torch.randperm(log_mel.shape[0], device=log_mel.device)
                beta = torch.distributions.Beta(
                    self.freq_mix_alpha, self.freq_mix_alpha
                )
                mix = beta.sample((log_mel.shape[0], 1)).to(
                    device=log_mel.device, dtype=log_mel.dtype
                )
                mixed_mean = (
                    mix * frequency_mean
                    + (1.0 - mix) * frequency_mean[permutation]
                )
                mixed_std = (
                    mix * frequency_std
                    + (1.0 - mix) * frequency_std[permutation]
                )
                log_mel = (
                    (log_mel - frequency_mean.unsqueeze(2))
                    / frequency_std.unsqueeze(2)
                    * mixed_std.unsqueeze(2)
                    + mixed_mean.unsqueeze(2)
                )
            layer_normalized = (
                log_mel - log_mel.mean(dim=(1, 2), keepdim=True)
            ) / log_mel.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)
            if self.input_normalization == "rfn":
                frequency_normalized = (
                    log_mel - log_mel.mean(dim=2, keepdim=True)
                ) / log_mel.std(dim=2, keepdim=True).clamp_min(1e-5)
                log_mel = (
                    self.rfn_lambda * layer_normalized
                    + (1.0 - self.rfn_lambda) * frequency_normalized
                )
            else:
                log_mel = layer_normalized
        return self.classifier(self.features(log_mel.unsqueeze(1)))


def make_loader(rows: list[dict[str, str]], args: argparse.Namespace, shuffle: bool, seed: int) -> DataLoader:
    dataset: Dataset[tuple[torch.Tensor, int]] = AudioWindowDataset(
        rows,
        args.sample_rate,
        args.window_sec,
        args.resample_mode,
        args.input_context_sec,
        args.context_alignment,
    )
    if shuffle and args.presence_background_mix_probability > 0.0:
        dataset = PresenceBackgroundMixDataset(
            dataset,
            rows,
            args.presence_background_mix_probability,
            args.presence_background_snr_min_db,
            args.presence_background_snr_max_db,
        )
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if shuffle and args.balanced_sampling:
        counts = Counter(int(row["label_id"]) for row in rows)
        sample_weights = [1.0 / counts[int(row["label_id"])] for row in rows]
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(rows),
            replacement=True,
            generator=generator,
        )
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle and sampler is None,
                      sampler=sampler, num_workers=args.workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=args.workers > 0,
                      generator=generator, drop_last=False)


def confusion_and_metrics(targets: list[int], predictions: list[int]) -> dict[str, Any]:
    matrix = [[0 for _ in range(NUM_CLASSES)] for _ in range(NUM_CLASSES)]
    for target, prediction in zip(targets, predictions):
        matrix[target][prediction] += 1
    total = sum(sum(row) for row in matrix)
    per_class: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        tp = matrix[class_id][class_id]
        fp = sum(matrix[row][class_id] for row in range(NUM_CLASSES)) - tp
        fn = sum(matrix[class_id]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[class_name] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(matrix[class_id])}
        f1_values.append(f1)
    return {
        "accuracy": sum(matrix[index][index] for index in range(NUM_CLASSES)) / total if total else 0.0,
        "macro_f1": sum(f1_values) / NUM_CLASSES,
        "confusion_matrix": matrix,
        "confusion_matrix_axes": {"rows": "true labels", "columns": "predicted labels", "class_names": CLASS_NAMES},
        "per_class": per_class,
        "n_samples": total,
    }


def run_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer | None,
              scaler: Any, device: torch.device, fold: int, epoch: int, total_epochs: int,
              class_weights: torch.Tensor) -> dict[str, Any]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    amp_enabled = device.type == "cuda"
    label = "train" if is_training else "valid"
    progress = tqdm(loader, desc=f"Fold {fold + 1}/5 | {label} {epoch}/{total_epochs}", unit="batch", dynamic_ncols=True, leave=False)
    for waveforms, labels in progress:
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if is_training:
            optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(waveforms)
            loss = F.cross_entropy(logits, labels, weight=class_weights)
        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite model output/loss encountered. The run was stopped before saving an invalid checkpoint."
            )
        if is_training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        batch_size = labels.size(0)
        total_loss += loss.detach().item() * batch_size
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        running_accuracy = sum(a == b for a, b in zip(targets, predictions)) / len(targets)
        progress.set_postfix(loss=f"{total_loss / len(targets):.4f}", acc=f"{running_accuracy:.3f}")
    metrics = confusion_and_metrics(targets, predictions)
    metrics["loss"] = total_loss / len(targets) if targets else float("nan")
    return metrics


@torch.inference_mode()
def evaluate_ensemble(checkpoint_paths: Iterable[Path], loader: DataLoader, args: argparse.Namespace,
                      device: torch.device) -> dict[str, Any]:
    models: list[nn.Module] = []
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = LogMelCNN(args.sample_rate, args.n_mels, args.n_fft, args.hop_length).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        models.append(model)
    if not models:
        raise RuntimeError("No checkpoints available for outer-test ensemble")
    targets: list[int] = []
    predictions: list[int] = []
    progress = tqdm(loader, desc=f"Outer test | ensemble of {len(models)} fold model(s)", unit="batch", dynamic_ncols=True)
    for waveforms, labels in progress:
        waveforms = waveforms.to(device, non_blocking=True)
        averaged_logits = torch.stack([model(waveforms) for model in models]).mean(dim=0)
        targets.extend(labels.tolist())
        predictions.extend(averaged_logits.argmax(dim=1).cpu().tolist())
        progress.set_postfix(acc=f"{sum(a == b for a, b in zip(targets, predictions)) / len(targets):.3f}")
    return confusion_and_metrics(targets, predictions)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_fold_table(path: Path, results: list[dict[str, Any]]) -> None:
    fields = ["fold", "best_epoch", "best_validation_loss", "best_validation_accuracy", "best_validation_macro_f1", "epochs_completed", "checkpoint"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: result[field] for field in fields} for result in results])


def print_metrics(title: str, metrics: dict[str, Any]) -> None:
    print(f"\n{title}")
    print(f"  accuracy={metrics['accuracy']:.4f} | macro-F1={metrics['macro_f1']:.4f} | n={metrics['n_samples']}")
    print("  confusion matrix (rows=true, columns=predicted):")
    for class_name, row in zip(CLASS_NAMES, metrics["confusion_matrix"]):
        print(f"    {class_name:12s} {row}")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    if args.class_names.strip():
        configure_class_names(args.class_names.split(","))
    else:
        dataset_summary_path = data_dir / "dataset_summary.json"
        if dataset_summary_path.is_file():
            dataset_summary = json.loads(
                dataset_summary_path.read_text(encoding="utf-8")
            )
            class_mapping = dataset_summary.get("classes", {})
            configure_class_names(
                name
                for name, _ in sorted(
                    class_mapping.items(), key=lambda item: int(item[1])
                )
            )
    set_seed(args.seed)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    init_run = args.init_run.resolve() if args.init_run is not None else None
    if init_run is not None:
        init_checkpoints = [init_run / f"fold_{fold + 1}" / "best.pt" for fold in args.fold_ids]
        missing = [str(path) for path in init_checkpoints if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing initialisation checkpoint(s): {missing}")
        write_json(
            output_dir / "initialization.json",
            {
                "mode": "matching-fold warm start",
                "source_run": str(init_run),
                "source_checkpoints_read_only": True,
            },
        )
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")

    train_rows = read_manifest(data_dir / "train.csv")
    validation_rows = read_manifest(data_dir / "validation.csv")
    test_rows = read_manifest(data_dir / "test.csv")
    development_rows = train_rows + validation_rows
    for row in development_rows + test_rows:
        label_id = int(row["label_id"])
        if (
            label_id < 0
            or label_id >= NUM_CLASSES
            or row["label"] != CLASS_NAMES[label_id]
        ):
            raise ValueError(
                "Manifest class mapping does not match configured classes: "
                f"label={row['label']!r}, label_id={label_id}, "
                f"classes={CLASS_NAMES}"
            )
    available_folds = sorted({int(row["cv_fold"]) for row in development_rows})
    unknown_folds = sorted(set(args.fold_ids).difference(available_folds))
    if unknown_folds:
        raise ValueError(f"Requested folds {unknown_folds} are not in development manifests; available: {available_folds}")
    if len(args.fold_ids) == 5 and sorted(args.fold_ids) != available_folds:
        raise ValueError(f"Expected five folds {available_folds}, received {args.fold_ids}")

    channel_responses = None
    channel_response_path = None
    if args.channel_response_npy is not None:
        channel_response_path = args.channel_response_npy.resolve()
        if not channel_response_path.is_file():
            raise FileNotFoundError(
                f"Channel-response array not found: {channel_response_path}"
            )
        channel_response_array = np.load(channel_response_path)
        if (
            channel_response_array.ndim != 2
            or channel_response_array.shape[1] != args.n_mels
            or not np.isfinite(channel_response_array).all()
        ):
            raise ValueError(
                "Channel-response array must be finite with shape "
                f"[N, {args.n_mels}], received {channel_response_array.shape}"
            )
        channel_responses = torch.from_numpy(
            channel_response_array.astype(np.float32, copy=False)
        )

    config = RunConfig(str(data_dir), str(output_dir), args.sample_rate, args.window_sec, args.n_mels,
                       args.n_fft, args.hop_length, args.epochs, args.batch_size, args.workers,
                       args.learning_rate, args.weight_decay, args.patience, args.seed, str(device), args.fold_ids,
                       args.resample_mode,
                       str(channel_response_path) if channel_response_path else None,
                       args.channel_augment_probability,
                       args.channel_augment_scale_min,
                       args.channel_augment_scale_max,
                       args.freq_mix_probability,
                       args.freq_mix_alpha,
                       args.input_normalization,
                       args.rfn_lambda,
                       args.input_context_sec,
                       args.context_alignment,
                       list(CLASS_NAMES),
                       args.presence_background_mix_probability,
                       args.presence_background_snr_min_db,
                       args.presence_background_snr_max_db,
                       args.balanced_sampling)
    write_json(output_dir / "run_config.json", asdict(config))
    print("=" * 72)
    print(f"Mosquito {NUM_CLASSES}-class training: {' / '.join(CLASS_NAMES)}")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"Development windows: {len(development_rows):,}; outer test windows: {len(test_rows):,}; folds: {args.fold_ids}")
    print(f"Resampling frontend: {args.resample_mode}")
    if args.presence_background_mix_probability > 0.0:
        print(
            "Presence background mixing: "
            f"p={args.presence_background_mix_probability:.2f}, "
            f"SNR=[{args.presence_background_snr_min_db:.1f}, "
            f"{args.presence_background_snr_max_db:.1f}] dB"
        )
    if args.balanced_sampling:
        print("Training sampler: class-balanced; loss weights: uniform")
    if channel_responses is not None:
        print(
            "Empirical channel augmentation: "
            f"{len(channel_responses)} responses, "
            f"p={args.channel_augment_probability:.2f}, "
            f"scale=[{args.channel_augment_scale_min:.2f}, "
            f"{args.channel_augment_scale_max:.2f}]"
        )
    print(f"Output: {output_dir}")
    print("=" * 72)

    def completed_fold_result(fold: int, fold_dir: Path) -> dict[str, Any] | None:
        checkpoint_path = fold_dir / "best.pt"
        history_path = fold_dir / "history.json"
        marker_path = fold_dir / "completed.json"
        if not checkpoint_path.is_file() or not history_path.is_file():
            return None
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if not history:
            return None
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        best_metrics = checkpoint.get("validation_metrics")
        if not isinstance(best_metrics, dict):
            return None
        best_epoch = int(checkpoint.get("epoch", 0))
        epochs_completed = len(history)
        inferred_complete = epochs_completed >= args.epochs
        if not inferred_complete and best_epoch > 0:
            inferred_complete = epochs_completed - best_epoch >= args.patience
        if not marker_path.is_file() and not inferred_complete:
            return None
        result = {
            "fold": fold + 1,
            "best_epoch": best_epoch,
            "best_validation_loss": best_metrics["loss"],
            "best_validation_accuracy": best_metrics["accuracy"],
            "best_validation_macro_f1": best_metrics["macro_f1"],
            "epochs_completed": epochs_completed,
            "checkpoint": str(checkpoint_path),
            "validation_metrics": best_metrics,
        }
        if not marker_path.is_file():
            write_json(
                marker_path,
                {
                    "fold": fold + 1,
                    "epochs_completed": epochs_completed,
                    "best_epoch": best_epoch,
                    "completion": "inferred_from_existing_history",
                },
            )
        return result

    fold_results: list[dict[str, Any]] = []
    checkpoint_paths: list[Path] = []
    for fold in args.fold_ids:
        fold_seed = args.seed + fold
        set_seed(fold_seed)
        fold_dir = output_dir / f"fold_{fold + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        if args.resume_completed_folds:
            reused = completed_fold_result(fold, fold_dir)
            if reused is not None:
                print(
                    f"\nFold {fold + 1}/5 — reusing completed fold "
                    f"({reused['epochs_completed']} epochs, "
                    f"best epoch {reused['best_epoch']})."
                )
                fold_results.append(reused)
                checkpoint_paths.append(Path(reused["checkpoint"]))
                continue
        fold_train = [row for row in development_rows if int(row["cv_fold"]) != fold]
        fold_validation = [row for row in development_rows if int(row["cv_fold"]) == fold]
        fold_train = cap_balanced(fold_train, args.max_train_samples, fold_seed)
        fold_validation = cap_balanced(fold_validation, args.max_validation_samples, fold_seed)
        print(f"\nFold {fold + 1}/5 — training windows={len(fold_train):,}, validation windows={len(fold_validation):,}")
        print(f"  train labels: {dict(sorted(Counter(row['label'] for row in fold_train).items()))}")
        print(f"  valid labels: {dict(sorted(Counter(row['label'] for row in fold_validation).items()))}")
        train_counts = Counter(int(row["label_id"]) for row in fold_train)
        if args.balanced_sampling:
            class_weights = torch.ones(NUM_CLASSES, dtype=torch.float32, device=device)
        else:
            class_weights = torch.tensor(
                [len(fold_train) / (NUM_CLASSES * train_counts[class_id]) for class_id in range(NUM_CLASSES)],
                dtype=torch.float32,
                device=device,
            )
        print(f"  class-loss weights: {[round(value, 3) for value in class_weights.detach().cpu().tolist()]}")
        train_loader = make_loader(fold_train, args, shuffle=True, seed=fold_seed)
        validation_loader = make_loader(fold_validation, args, shuffle=False, seed=fold_seed)
        model = LogMelCNN(
            args.sample_rate,
            args.n_mels,
            args.n_fft,
            args.hop_length,
            channel_responses=channel_responses,
            channel_augment_probability=args.channel_augment_probability,
            channel_augment_scale_min=args.channel_augment_scale_min,
            channel_augment_scale_max=args.channel_augment_scale_max,
            freq_mix_probability=args.freq_mix_probability,
            freq_mix_alpha=args.freq_mix_alpha,
            input_normalization=args.input_normalization,
            rfn_lambda=args.rfn_lambda,
        ).to(device)
        initialization_checkpoint = None
        if init_run is not None:
            initialization_checkpoint = init_run / f"fold_{fold + 1}" / "best.pt"
            initialization_payload = torch.load(
                initialization_checkpoint, map_location="cpu", weights_only=False
            )
            initialization_names = list(initialization_payload.get("class_names", []))
            if initialization_names != list(CLASS_NAMES):
                raise RuntimeError(
                    f"Initialisation class order mismatch for fold {fold + 1}: "
                    f"expected {CLASS_NAMES}, found {initialization_names}"
                )
            model.load_state_dict(initialization_payload["model_state"])
            print(f"  warm start: {initialization_checkpoint}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
        try:
            scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
        except (AttributeError, TypeError):  # compatibility with older PyTorch releases
            scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
        best_f1 = -1.0
        best_epoch = 0
        epochs_without_improvement = 0
        history: list[dict[str, Any]] = []
        checkpoint_path = fold_dir / "best.pt"
        start_time = time.perf_counter()

        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, fold, epoch, args.epochs, class_weights)
            validation_metrics = run_epoch(model, validation_loader, None, scaler, device, fold, epoch, args.epochs, class_weights)
            scheduler.step(validation_metrics["macro_f1"])
            record = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "validation": validation_metrics,
                "elapsed_sec": time.perf_counter() - start_time,
            }
            history.append(record)
            write_json(fold_dir / "history.json", history)
            print(f"Fold {fold + 1} epoch {epoch:02d}/{args.epochs}: "
                  f"train loss={train_metrics['loss']:.4f}, F1={train_metrics['macro_f1']:.4f} | "
                  f"valid loss={validation_metrics['loss']:.4f}, F1={validation_metrics['macro_f1']:.4f}, "
                  f"acc={validation_metrics['accuracy']:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}")
            if validation_metrics["macro_f1"] > best_f1 + 1e-6:
                best_f1 = validation_metrics["macro_f1"]
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save({"model_state": model.state_dict(), "fold": fold, "epoch": epoch,
                            "validation_metrics": validation_metrics, "config": asdict(config), "class_names": CLASS_NAMES,
                            "initialization_checkpoint": str(initialization_checkpoint) if initialization_checkpoint else None}, checkpoint_path)
                write_json(fold_dir / "best_validation.json", validation_metrics)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    print(f"Fold {fold + 1}: early stopping after {epoch} epochs (best epoch {best_epoch}).")
                    break

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        best_metrics = checkpoint["validation_metrics"]
        print_metrics(f"Fold {fold + 1} best validation result (epoch {best_epoch})", best_metrics)
        fold_results.append({
            "fold": fold + 1,
            "best_epoch": best_epoch,
            "best_validation_loss": best_metrics["loss"],
            "best_validation_accuracy": best_metrics["accuracy"],
            "best_validation_macro_f1": best_metrics["macro_f1"],
            "epochs_completed": len(history),
            "checkpoint": str(checkpoint_path),
            "validation_metrics": best_metrics,
        })
        checkpoint_paths.append(checkpoint_path)
        write_json(
            fold_dir / "completed.json",
            {
                "fold": fold + 1,
                "epochs_completed": len(history),
                "best_epoch": best_epoch,
                "completion": "training_finished",
            },
        )

    write_fold_table(output_dir / "fold_results.csv", fold_results)
    f1_values = [row["best_validation_macro_f1"] for row in fold_results]
    accuracy_values = [row["best_validation_accuracy"] for row in fold_results]
    cv_summary = {
        "fold_results": fold_results,
        "validation_macro_f1_mean": sum(f1_values) / len(f1_values),
        "validation_macro_f1_std": (sum((item - sum(f1_values) / len(f1_values)) ** 2 for item in f1_values) / len(f1_values)) ** 0.5,
        "validation_accuracy_mean": sum(accuracy_values) / len(accuracy_values),
        "validation_accuracy_std": (sum((item - sum(accuracy_values) / len(accuracy_values)) ** 2 for item in accuracy_values) / len(accuracy_values)) ** 0.5,
    }
    write_json(output_dir / "cv_summary.json", cv_summary)
    print("\n" + "=" * 72)
    print(f"Cross-validation: macro-F1={cv_summary['validation_macro_f1_mean']:.4f} ± {cv_summary['validation_macro_f1_std']:.4f}; "
          f"accuracy={cv_summary['validation_accuracy_mean']:.4f} ± {cv_summary['validation_accuracy_std']:.4f}")

    test_rows = cap_balanced(test_rows, args.max_test_samples, args.seed)
    test_loader = make_loader(test_rows, args, shuffle=False, seed=args.seed)
    outer_test = evaluate_ensemble(checkpoint_paths, test_loader, args, device)
    outer_test["ensemble_folds"] = [fold + 1 for fold in args.fold_ids]
    outer_test["warning"] = "This is the single locked outer-test result. Do not tune hyperparameters from it."
    write_json(output_dir / "outer_test_ensemble.json", outer_test)
    print_metrics("Locked outer-test ensemble result", outer_test)
    print(f"\nCompleted. Results are in: {output_dir}")


if __name__ == "__main__":
    # Required on Windows because DataLoader workers use spawn.
    main()
