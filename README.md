# Development and Evaluation of an Edge Device for Mosquito Classification from Wingbeat Audio

This repository contains the reproducible software artifact for a six-class mosquito wingbeat-audio classifier running on an Arduino Nano ESP32 (ESP32-S3). The current release provides matched log-Mel or PCEN frontends with adjacent-average or low-pass 16-to-8 kHz decimation, compact INT8 convolutional neural networks, exact 3.0-second decision aggregation, continuous WAV/CSV recording, and recovery of historical SD-card sessions.

## Current release

The current laboratory package is **v6.0.0**: [`release/esp32_lab_deployment_overlap025_v6.0.0_labuse`](release/esp32_lab_deployment_overlap025_v6.0.0_labuse/README.md).

It contains:

- four matched INT8 TFLite models and embedded C headers for log-Mel or PCEN with adjacent-average or low-pass decimation;
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
-> matched adjacent-average or low-pass decimation to 8 kHz
-> 256-point STFT, 80-sample hop
-> 64-band Mel power
-> log-Mel by default, or context-local PCEN when selected
-> INT8 CNN
-> probability average over contexts [0.00,1.50], [0.75,2.25], [1.50,3.00] s
-> one exact 3.0-second decision
-> timestamped CSV plus the corresponding raw WAV
```

The optional PCEN frontend uses `alpha=0.6`, `delta=2.0`, `r=0.25`, `time_constant=0.8 s`, and `epsilon=1e-6`. Stop the current session with `E`, select `A` (log-Mel/adjacent average), `B` (PCEN/adjacent average), `F` (log-Mel/low-pass), or `H` (PCEN/low-pass), and start a new session with `4`. Power-on defaults to `A`.

## Quick start

On Windows, use a short checkout path such as `C:\mosquito` to avoid path-length limits. The repository preserves the exact CMSIS-NN patch bytes across Git line-ending settings.

Open PowerShell in `release/esp32_lab_deployment_overlap025_v6.0.0_labuse` and follow one of the two supported paths.

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

This command performs dataset checks, recording-level splits, five-fold log-Mel/adjacent-average training by default, validation-only fold selection, locked-test evaluation, full INT8 export, model/header installation, speed-patch verification, firmware compilation, flashing, and a same-PCM PC/ESP32 check. Use `-Frontend pcen` and/or `-Decimator lowpass` for another matched option. The release [README](release/esp32_lab_deployment_overlap025_v6.0.0_labuse/README.md) gives the complete data layout, commands, serial controls, download procedure, and CSV schema.

## Verified implementation properties

The v6.0.0 checks recorded:

- all four frontend/decimator options matched PC aggregate classes on 30/30 strict three-second decisions each; the PCEN/adjacent-average option had one differing subwindow class among 90, while all other options matched 90/90;
- four approximately 18-second microphone-and-SD sessions, one per option, saved valid WAV/CSV files with zero recorded dropped blocks or skipped decisions;
- the default log-Mel/adjacent-average configuration recorded 3,600 seconds and 1,200 three-second predictions in session LAB095, with zero recording errors, dropped blocks, or skipped decisions;
- the PC and ESP32 agreed on all 1,200 aggregate classes when processing LAB095's saved PCM; this unlabelled environmental recording does not provide classification accuracy or Macro-F1.

The corresponding reports are in [`release/esp32_lab_deployment_overlap025_v6.0.0_labuse/validation`](release/esp32_lab_deployment_overlap025_v6.0.0_labuse/validation/). The one-hour report covers the default option; the other three options have short hardware sessions and same-PCM checks.

## Historical v5.0.0 implementation audit

The previous dual-frontend release is preserved in [`history/esp32_lab_deployment_overlap025_v5.0.0_labuse`](history/esp32_lab_deployment_overlap025_v5.0.0_labuse/README.md), including its original model bytes, firmware, validation reports, and short WAV/CSV sessions. Its longest recorded hardware session in that release was 264.064 seconds.

## Historical v4.0.0 implementation audit

The historical release and the supplementary thesis audit record the following hardware and software checks:

- an expanded same-PCM audit covered 540 subwindow predictions and 180 exact-3-second decisions; 536 subwindow classes and all 180 aggregate classes agreed between PC and ESP32;
- 99.9978% of quantized PCEN input values were identical, with a maximum difference of one INT8 unit;
- mean PCEN feature extraction and CNN inference times were 110.86 ms and 667.50 ms per 1.5-second context, respectively;
- the selected-fold FP32 and INT8 models differed by 0.165 Macro-F1 percentage points on the same 2,934 source-domain contexts;
- the embedded model header matched the TFLite file byte for byte;
- a 14.304-second hardware smoke session produced valid 16 kHz mono PCM16 WAV and CSV files with no dropped audio blocks or skipped decisions;
- a separate one-hour session contained exactly 57,600,000 PCM frames and all 1,200 expected non-overlapping three-second decisions, with zero skipped hops.

The source reports for these historical checks are stored in [`history/esp32_lab_deployment_overlap025_v4.0.0_labuse/validation`](history/esp32_lab_deployment_overlap025_v4.0.0_labuse/validation/), including the expanded reports in [`supplementary_20260921`](history/esp32_lab_deployment_overlap025_v4.0.0_labuse/validation/supplementary_20260921/).

## Historical versions

Deployment versions v1.0.0 through v5.0.0 are stored in [`history/`](history/README.md). Folder names follow `esp32_lab_deployment_overlap025_vX.0.0_labuse`; the current version is v6.0.0. Internal firmware filenames and saved report filenames are retained so the archived and tested artifacts remain identifiable.

## Data availability

The repository provides the data format, label example, split logic, training code, model, firmware, and validation artifacts. The original mosquito recordings remain subject to their ownership and sharing conditions. Researchers with authorized access can place the recordings in the documented folder structure and reproduce the training pipeline.

## Reproducibility convention

Each experiment should record the model version and hash, training domain, input recording, evaluation platform, aggregation geometry, and boundary-window rule. PC offline evaluation, same-PCM ESP32 replay, loudspeaker replay, and live laboratory recordings represent distinct evaluation conditions and should be reported separately.

## License

The software is released under the [MIT License](LICENSE). Data and third-party library licenses remain governed by their original terms.
