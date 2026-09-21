# Development and Evaluation of an Edge Device for Mosquito Classification from Wingbeat Audio

This repository contains the reproducible software artifact for a six-class mosquito wingbeat-audio classifier running on an Arduino Nano ESP32 (ESP32-S3). The current release implements an 8 kHz, 64-band Mel/PCEN frontend, a compact INT8 convolutional neural network, exact 3.0-second decision aggregation, continuous WAV/CSV recording, and recovery of historical SD-card sessions.

## Current release

The validated laboratory package is [`release/v004.0.0`](release/v004.0.0/README.md).

It contains:

- a pretrained INT8 TFLite model and matching embedded C header;
- complete Arduino firmware and precompiled binaries;
- a five-fold retraining pipeline from WAV/JSON pairs;
- automatic INT8 export, model-header replacement, speed-patch installation, compilation, and flashing;
- PC--ESP32 same-PCM parity checks;
- recording/session download tools and CSV field definitions;
- the INMP441 and SD-card wiring table;
- machine-readable validation reports.

## Task and signal path

The classifier predicts five mosquito classes and a no-mosquito class:

1. *Aedes aegypti*
2. *Culex quinquefasciatus*
3. *Anopheles dirus*
4. female *Anopheles minimus*
5. male *Anopheles minimus*
6. no mosquito

The deployed signal path is:

```text
INMP441, 16 kHz mono PCM16
-> paired-sample reduction to 8 kHz
-> 256-point STFT, 80-sample hop
-> 64-band Mel power
-> context-local PCEN
-> INT8 CNN
-> probability average over contexts [0.00,1.50], [0.75,2.25], [1.50,3.00] s
-> one exact 3.0-second decision
-> timestamped CSV plus the corresponding raw WAV
```

The fixed PCEN parameters are `alpha=0.6`, `delta=2.0`, `r=0.25`, `time_constant=0.8 s`, and `epsilon=1e-6`.

## Quick start

Open PowerShell in `release/v004.0.0` and follow one of the two supported paths.

### Flash the included model

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\build_flash_included.ps1 -Port COM12
```

Replace `COM12` with the port shown by `arduino-cli board list`. The wrapper verifies and installs the three-file CMSIS-NN speed patch before compiling and flashing.

### Retrain and deploy

Prepare five class folders containing paired `.wav` and `.labels.json` files, then run:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\retrain_build_flash.ps1 `
  -TrainingFolder "D:\mosquito_training_audio" `
  -Port COM12
```

This command performs dataset checks, recording-level splits, five-fold PCEN training, validation-only fold selection, locked-test evaluation, full INT8 export, model/header installation, speed-patch verification, firmware compilation, and flashing. The release [README](release/v004.0.0/README.md) gives the complete data layout, manual commands, serial controls, download procedure, and CSV schema.

## Verified implementation properties

The release records the following hardware and software checks:

- 18 subwindow predictions and six exact-3-second aggregate decisions agreed between PC and ESP32 on identical PCM;
- 99.9983% of quantized PCEN input values were identical, with a maximum difference of one INT8 unit;
- mean CNN inference time was 667.48 ms per 1.5-second context with the verified speed patch;
- the embedded model header matched the TFLite file byte for byte;
- a 14.304-second hardware smoke session produced valid 16 kHz mono PCM16 WAV and CSV files with no dropped audio blocks or skipped decisions.

The source reports for these checks are stored in [`release/v004.0.0/validation`](release/v004.0.0/validation/).

## Data availability

The repository provides the data format, label example, split logic, training code, model, firmware, and validation artifacts. The original mosquito recordings remain subject to their ownership and sharing conditions. Researchers with authorized access can place the recordings in the documented folder structure and reproduce the training pipeline.

## Reproducibility convention

Each experiment should record the model version and hash, training domain, input recording, evaluation platform, aggregation geometry, and boundary-window rule. PC offline evaluation, same-PCM ESP32 replay, loudspeaker replay, and live laboratory recordings represent distinct evaluation conditions and should be reported separately.

## License

The software is released under the [MIT License](LICENSE). Data and third-party library licenses remain governed by their original terms.
