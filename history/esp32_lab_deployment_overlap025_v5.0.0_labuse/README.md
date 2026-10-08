# Six-Class Mosquito Sound Model and ESP32 Deployment Package

Version: **v5.0.0** Laboratory Version

## Package Contents

- `model/model_overlap025_logmel.tflite`: default log-Mel INT8 model; `model/model_overlap025.tflite`: optional PCEN INT8 model. Both use the same original-audio training data and split.
- `precompiled/`: verified firmware that can be flashed directly.
- `source/`: complete Arduino source code and embedded model header.
- `speed_patch/`: three CMSIS-NN files and an installation script; the recommended workflows install and verify them automatically.
- `tools/build_flash_included.ps1`: automatically installs and verifies the speed patch, compiles, refreshes `precompiled/`, and flashes the included model.
- `tools/retrain_build_flash.ps1`: automatically performs complete training, model/header replacement, speed-patch installation, compilation, and flashing.
- `tools/download_lab_session.py`: downloads a specified historical WAV and CSV session.
- `tools/verify_pc_esp32_parity.py`: compares PC and ESP32 features, CNN outputs, and exact-3-second decisions on identical PCM for the selected frontend.
- `PINOUT.csv`: wiring table for the INMP441 and SD card.
- `DEPENDENCIES.txt`: hardware, Arduino, and Python dependency versions.
- `training/`: complete retraining program, fixed configuration, dependencies, and label example.
- `validation/`: v5 same-PCM parity reports and short WAV/CSV hardware sessions. The copied v4 reports remain historical PCEN evidence; use the `V005_` reports for this firmware.

The target hardware is **Arduino Nano ESP32**, with an **ESP32-S3** main chip and the FQBN `arduino:esp32:nano_nora`.

Log-Mel is the default frontend. PCEN is an optional matched model/frontend pair; its effect varies across recording conditions, so choose between them using validation audio from the intended deployment setting.

## Run the Existing Model Directly

1. Connect the INMP441 and SD card according to `PINOUT.csv`, and insert a FAT32 SD card.
2. Install the versions listed in `DEPENDENCIES.txt`.
3. Close the Arduino Serial Monitor and query the port:

```powershell
arduino-cli board list
```

4. Replace `COM12` with the actual port and flash the firmware:

```powershell
arduino-cli upload -p COM12 --fqbn arduino:esp32:nano_nora --input-dir ".\precompiled" ".\source\lab_realtime_recorder_v005_dual_exact3s"
```

After power-on, the firmware defaults to the included log-Mel model, waits for 10 seconds, and then automatically starts six-class classification. Sending `E` through the serial port during the countdown cancels the automatic start. To use PCEN, stop the current session with `E`, select `B`, and start a new session with `4`. Selection returns to log-Mel after power loss.

The serial baud rate is 115200:

- `?`: show status
- `A`: select log-Mel (default); `B`: select PCEN. Select only after stopping the current session with `E`. A new CSV records the selected frontend and model.
- `4`: manually start six-class classification while saving `LABnnn.WAV` and `LABnnn.CSV`
- `E`: stop and safely close the files
- `C4`: record 10 seconds of no-mosquito background and calculate temporary calibration
- `C4S`: start mode 4 with the current calibration
- `R`: clear calibration from RAM
- `N`: list historical sessions on the SD card
- `Jnnn`: select a historical session, for example `J010`
- `D`: download the selected WAV
- `L`: download the selected CSV
- `G`: run one strict 3-second timing-geometry diagnostic

Calibration values are stored only in RAM and are automatically cleared after power loss. During a normal stop, wait until `SESSION_RECORD_DONE` and `LIVE_STOPPED` appear in the serial output.

## Download Recordings and Prediction Results

Enter `N` in the Serial Monitor to identify the intended completed session, then close the Serial Monitor and download it by number:

```powershell
python .\tools\download_lab_session.py --port COM12 --session LAB010 --output-dir .\downloaded_sessions\LAB010
```

Every complete CSV prediction row is immediately synchronized to the SD card. After an unexpected power loss, the download tool repairs the WAV header according to the saved PCM length. The row or audio block being written at the instant of power loss may be lost; previously synchronized data remain available.

`LAB010` is an example session number. Before downloading, enter uppercase `N` in the Arduino Serial Monitor to view the session list stored on the SD card, and then replace `LAB010` in the command with the session number that must be downloaded. The session with the largest number is usually the most recent recording; in our actual use, the recording that must be downloaded is generally the second most recent one. `COM12` must also be replaced with the actual serial port currently recognized by the computer.

## Rebuild from the Arduino Source Code

**Important:** The precompiled firmware already contains the verified speed optimization. For every source rebuild, use the wrapper below. It installs and verifies all three CMSIS-NN patch files before compilation, refreshes the binaries in `precompiled/`, and uploads the firmware:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\build_flash_included.ps1 -Port COM12
```

For a compile-only check:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\build_flash_included.ps1 -NoUpload
```

The underlying manual sequence remains available for diagnosis:

```powershell
powershell -ExecutionPolicy Bypass -File .\speed_patch\install_speed_patch.ps1
powershell -ExecutionPolicy Bypass -File .\tools\verify_speed_patch.ps1
arduino-cli compile --fqbn arduino:esp32:nano_nora .\source\lab_realtime_recorder_v005_dual_exact3s
arduino-cli upload -p COM12 --fqbn arduino:esp32:nano_nora .\source\lab_realtime_recorder_v005_dual_exact3s
```

## Retrain the Model and Update the ESP32 Firmware

### 1. Prepare the Training Data

The training-data root directory must contain the following five class folders:

```text
D:\mosquito_training_audio\
  Ae_aegypti_M\
  Cx_quin_M\
  An_dirus_M\
  An_minimus_F\
  An_minimus_M\
```

Each recording must contain a pair of files with the same base name:

```text
recording_name.wav
recording_name.labels.json
```

The `regions` field in each JSON file must contain `start_sec`, `end_sec`, and `confidence`. See the example at:

```text
training\example\example.labels.json
```

### 2. Install the Training Environment

Open PowerShell in the root directory of this package:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\training\requirements-training.txt
```

### 3. Check the Training Data

Replace `D:\mosquito_training_audio` with the actual training-data directory:

```powershell
python .\training\run_training.py --training-folder "D:\mosquito_training_audio" --dry-run
```

If the data structure and labels are valid, the program displays the number of recordings in each class, the training, validation, and test sets, and the five-fold split information.

### 4. Run Full Training

The recommended one-command workflow performs five-fold log-Mel training by default, selects a checkpoint only by validation Macro-F1, exports full INT8 TFLite, updates the corresponding package model and firmware header, installs and verifies the speed patch, compiles, and flashes:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\retrain_build_flash.ps1 -TrainingFolder "D:\mosquito_training_audio" -Port COM12
```

For PCEN training, add `-Frontend pcen`, for example:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\retrain_build_flash.ps1 -TrainingFolder "D:\mosquito_training_audio" -Port COM12 -Frontend pcen
```

Only the selected model/header pair is replaced; the other included model remains available. The flashed device still boots in log-Mel mode; enter `B` before starting a PCEN session.

To run the complete training and export stage without immediately updating the ESP32:

```powershell
python .\training\run_training.py --training-folder "D:\mosquito_training_audio" --config .\training\configs\overlap025_logmel_exact3s_v005.json --output .\runs\overlap025_logmel_exact3s_v005
```

For a PCEN-only training/export run, replace the config with `training\configs\overlap025_pcen_exact3s_v005.json` and use a separate output directory.

The program performs:

```text
WAV and JSON checks
→ 0.5-second label windows with a 0.25-second stride
→ recording-level data split
→ 8 kHz paired-sample resampling
→ STFT and 64-band Mel power
→ log-Mel by default, or context-local PCEN with `-Frontend pcen` (alpha 0.6, delta 2.0, r 0.25, time constant 0.8 s)
→ five-fold CNN training
→ locked test-set evaluation
→ selection of the fold with the highest validation Macro-F1
→ full INT8 TFLite quantization
→ generation of the TFLite model and C header
```

After training, the main output files are located at:

```text
runs\overlap025_logmel_exact3s_v005\export\
  model_overlap025_logmel.tflite
  model_overlap025_logmel.h
  model_overlap025_logmel_verification.json
```

Their purposes are:

- `model_overlap025_logmel.tflite`: full INT8 six-class log-Mel model. PCEN export uses `model_overlap025.tflite`.
- `model_overlap025_logmel.h`: C array embedded in the Arduino firmware. PCEN export uses `model_overlap025.h`.
- `model_overlap025_logmel_verification.json`: quantization parameters, model size, conversion agreement, accuracy, Macro-F1, and confusion matrix. PCEN has the corresponding `model_overlap025_verification.json`.

### 5. Back Up the Current Model

Run:

```powershell
Copy-Item .\model\model_overlap025_logmel.tflite .\model\model_overlap025_logmel_previous.tflite
Copy-Item .\source\lab_realtime_recorder_v005_dual_exact3s\models\model_overlap025_logmel.h .\source\lab_realtime_recorder_v005_dual_exact3s\models\model_overlap025_logmel_previous.h
```

### 6. Update the TFLite Model in the Package

```powershell
Copy-Item .\runs\overlap025_logmel_exact3s_v005\export\model_overlap025_logmel.tflite .\model\model_overlap025_logmel.tflite -Force
```

### 7. Insert the New Model into the ESP32 Source Code

This step is automatic when `tools/retrain_build_flash.ps1` is used. For a manual update, copy the generated TFLite model, verification report, and C header:

```powershell
Copy-Item .\runs\overlap025_logmel_exact3s_v005\export\model_overlap025_logmel.tflite .\model\model_overlap025_logmel.tflite -Force
Copy-Item .\runs\overlap025_logmel_exact3s_v005\export\model_overlap025_logmel_verification.json .\model\model_overlap025_logmel_verification.json -Force
Copy-Item .\runs\overlap025_logmel_exact3s_v005\export\model_overlap025_logmel.h .\source\lab_realtime_recorder_v005_dual_exact3s\models\model_overlap025_logmel.h -Force
```

Verify that the TFLite byte count equals the generated header length and that `model_overlap025_logmel` is defined once:

```powershell
(Select-String -Path .\source\lab_realtime_recorder_v005_dual_exact3s\models\model_overlap025_logmel.h -Pattern "const unsigned char model_overlap025_logmel\[\]").Count
```

The correct result is:

```text
1
```

For manual PCEN replacement, use the PCEN config and the `model_overlap025` files in the same locations. Do not change the PCEN parameters in only one place: the training configuration and Arduino constants must both remain alpha 0.6, delta 2.0, r 0.25, time constant 0.8 s, and epsilon 1e-6. Any frontend or parameter change requires a new same-PCM PC/ESP32 parity check.

### 8. Install the Speed Optimization Patch

This step is automatic in both recommended wrapper scripts. For a manual diagnostic installation:

```powershell
powershell -ExecutionPolicy Bypass -File .\speed_patch\install_speed_patch.ps1
powershell -ExecutionPolicy Bypass -File .\tools\verify_speed_patch.ps1
```

The installer accepts only the known original or already-patched CMSIS-NN files and verifies all three installed SHA-256 hashes. If an unknown library version is found, it stops before changing the library.

### 9. Find the ESP32 Serial Port

Connect the ESP32 and run:

```powershell
arduino-cli board list
```

Record the serial port corresponding to the Arduino Nano ESP32, for example `COM12`.

### 10. Compile and Flash the New Model

Replace `COM12` with the actual serial port:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\build_flash_included.ps1 -Port COM12
```

This command automatically installs and verifies the speed patch, compiles the firmware, refreshes the precompiled binaries, and uploads to the ESP32. Mode 4 defaults to the newly trained log-Mel model. If PCEN was retrained, enter `B` after stopping the session and then enter `4` to run that model.

### 11. Check the Flashing Result

Open the Arduino Serial Monitor and set it to:

```text
115200 baud
```

During a normal start, the following should appear:

```text
BOOT lab_realtime_recorder_v005_dual_frontend_exact3s frontend=logmel
READY commands:
```

Enter:

```text
?
```

Check the SD card, microphone, recording state, and model state.

Enter:

```text
4
```

This starts the updated overlap025 six-class model with the currently selected frontend. The ESP32 simultaneously performs:

```text
real-time audio capture
→ STFT, Mel, and log-Mel (or selected PCEN) feature calculation
→ six-class INT8 CNN inference
→ three contexts at 0.00–1.50, 0.75–2.25, and 1.50–3.00 seconds
→ probability averaging into one exact 3.0-second decision
→ classification-result output
→ saving LABnnn.WAV
→ saving LABnnn.CSV
```

Enter:

```text
E
```

Stop safely and wait for:

```text
SESSION_RECORD_DONE
LIVE_STOPPED
```

---

## CSV Output Column Descriptions

Each real-time recognition session creates a pair of files on the ESP32 SD card:

```text
LABnnn.WAV
LABnnn.CSV
```

The WAV file stores the complete raw recording, and the CSV file stores prediction results with audio timestamps.

The beginning of the CSV file contains firmware and recording parameters, for example:

```text
# firmware=lab_realtime_recorder_v005_dual_frontend_exact3s
# frontend=logmel
# deployed_model=logmel_selected_fold_int8
# speed_patch=cmsis_nn_verified_3file
# sample_rate_hz=16000,channels=1,bits_per_sample=16
# maximum_recording_seconds=3600
# mode=4,calibrated=0
```

PCEN sessions instead show `# frontend=pcen`, `# deployed_model=pcen_selected_fold_int8`, and a line with the PCEN parameters. The included models were trained on the original recordings; after retraining, the `model/` files and their verification reports identify the replaced model.

The columns have the following meanings:

| Column | Meaning |
|---|---|
| `result_index` | Prediction index in the current session, increasing from 0. |
| `session_elapsed_ms` | Actual elapsed time from the beginning of the session until this prediction is written, in milliseconds. |
| `audio_end_ms` | End time of the audio region corresponding to the current prediction, based on the WAV audio timeline, in milliseconds. |
| `support_start_ms` | Start point of the audio support region actually used by the current prediction, in milliseconds. |
| `support_end_ms` | End point of the audio support region actually used by the current prediction, in milliseconds. |
| `mode` | Model mode currently in use. Mode 4 is the overlap025 six-class model. |
| `class` | Final predicted class. |
| `ae_aegypti` | Model score for Ae. aegypti. In mode 4, this is the aggregated class probability. |
| `cx_quin` | Model score for Cx. quin. In mode 4, this is the aggregated class probability. |
| `an_dirus` | Model score for An. dirus. In mode 4, this is the aggregated class probability. |
| `an_minimus_f` | Model score for An. minimus female. In mode 4, this is the aggregated class probability. |
| `an_minimus_m` | Model score for An. minimus male. In mode 4, this is the aggregated class probability. |
| `no_mosquito` | Model score for No mosquito. In mode 4, this is the aggregated class probability. |
| `calibrated` | Whether background calibration is enabled. `0` means disabled and `1` means enabled. |
| `no_margin` | The `No mosquito` score minus the highest score among the five mosquito classes. A larger value indicates a stronger tendency to classify the window as No mosquito. |
| `no_threshold` | No-mosquito decision threshold used by the current background calibration. It is `0` when calibration is disabled. |
| `snapshot_wait_ms` | Time required to obtain the current audio-context snapshot, in milliseconds. |
| `feature_ms` | Total time required to calculate resampling, STFT, Mel filtering, and the selected log-Mel or PCEN features for the three contexts, in milliseconds. |
| `infer_ms` | Total time required to execute the INT8 TFLite Micro CNN for the three contexts, in milliseconds. |
| `audio_lag_ms` | Amount of time by which the predicted audio endpoint is behind the real-time capture position when the result is output, in milliseconds. |
| `latency_ms` | Total latency from the beginning of the current real-time processing cycle until the final result is obtained, in milliseconds. |
| `skipped_hops_total` | Cumulative number of skipped detection hops since the start of the session. This value increases when the system cannot keep up with the audio stream. |

In mode 4, the six class scores are normally between `0–1`, and their sum is approximately `1`. Each decision captures exactly 3.0 seconds, evaluates the three 1.5-second contexts `[0.00,1.50]`, `[0.75,2.25]`, and `[1.50,3.00]`, and averages their probability vectors. Consecutive decisions use a 3.0-second hop.

Every completed CSV row is immediately synchronized to the SD card. After an unexpected power loss, complete rows that have already been synchronized remain available; the final row being written at the instant of power loss may be lost.

The program automatically performs WAV/JSON checking, recording-level partitioning, 0.5-second label windows with a 0.25-second stride, 1.5-second centered contexts, ESP32-equivalent resampling, 64-band log-Mel extraction by default (or PCEN when selected), five-fold CNN training, locked-test evaluation, and full INT8 TFLite export.

Fixed frontend parameters: 8 kHz, 256-point FFT, STFT hop of 80 points, 64 Mel bands, and input tensor `1×64×151×1`. The training output directory saves a copy of the configuration, data fingerprint, five-fold results, locked-test results, exported model, and `experiment_manifest.json`. It records the v5.0.0 version from this package folder, the model hash, and the output-file hashes.

The current real-time mode uses the same strict 3.0-second geometry as PC evaluation. Command `G` runs one diagnostic decision with the same three context ranges.

## Power and Storage

- Use a stable 5 V USB power supply rated for at least 1 A.
- Keep the INMP441 and SD module at 3.3 V logic levels.
- WAV format is 16 kHz, 16-bit, mono, using approximately 115.2 MB per hour.
- Check the remaining SD-card capacity before long-duration experiments.
