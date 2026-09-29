# Six-Class Mosquito Sound Model and ESP32 Deployment Package

Version: **v3.0.0** Laboratory Version

## Package Contents

- `model/model_overlap025.tflite`: INT8 model used by mode 4.
- `precompiled/`: verified firmware that can be flashed directly.
- `source/`: complete Arduino source code and embedded model header.
- `speed_patch/`: three CMSIS-NN files and an installation script required to rebuild the optimized firmware.
- `tools/download_lab_session.py`: downloads the latest or a specified historical WAV and CSV session.
- `PINOUT.csv`: wiring table for the INMP441 and SD card.
- `DEPENDENCIES.txt`: hardware, Arduino, and Python dependency versions.
- `training/`: complete retraining program, fixed configuration, dependencies, and label example.

The target hardware is **Arduino Nano ESP32**, with an **ESP32-S3** main chip and the FQBN `arduino:esp32:nano_nora`.

## Run the Existing Model Directly

1. Connect the INMP441 and SD card according to `PINOUT.csv`, and insert a FAT32 SD card.
2. Install the versions listed in `DEPENDENCIES.txt`.
3. Close the Arduino Serial Monitor and query the port:

```powershell
arduino-cli board list
```

4. Replace `COM12` with the actual port and flash the firmware:

```powershell
arduino-cli upload -p COM12 --fqbn arduino:esp32:nano_nora --input-dir ".\precompiled" ".\source\lab_realtime_recorder_v1_speedopt"
```

After power-on, the firmware waits for 10 seconds and then automatically starts six-class classification. Sending `E` through the serial port during the countdown cancels the automatic start.

The serial baud rate is 115200:

- `?`: show status
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

Close the Serial Monitor and run:

```powershell
python .\tools\download_lab_session.py --port COM12 --output-dir .\downloaded_sessions
```

Download a specified historical session:

```powershell
python .\tools\download_lab_session.py --port COM12 --session LAB010 --output-dir .\downloaded_sessions\LAB010
```

Every complete CSV prediction row is immediately synchronized to the SD card. After an unexpected power loss, the download tool repairs the WAV header according to the saved PCM length. The row or audio block being written at the instant of power loss may be lost; previously synchronized data remain available.

`LAB010` is an example session number. Before downloading, enter uppercase `N` in the Arduino Serial Monitor to view the session list stored on the SD card, and then replace `LAB010` in the command with the session number that must be downloaded. The session with the largest number is usually the most recent recording; in our actual use, the recording that must be downloaded is generally the second most recent one. `COM12` must also be replaced with the actual serial port currently recognized by the computer.

## Rebuild from the Arduino Source Code

**Important:** Install `speed_patch/install_speed_patch.ps1` before compiling from source; the precompiled firmware already includes this patch, while an unpatched rebuild can increase CNN inference time substantially.

```powershell
powershell -ExecutionPolicy Bypass -File .\speed_patch\install_speed_patch.ps1
arduino-cli compile --fqbn arduino:esp32:nano_nora .\source\lab_realtime_recorder_v1_speedopt
arduino-cli upload -p COM12 --fqbn arduino:esp32:nano_nora .\source\lab_realtime_recorder_v1_speedopt
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

```powershell
python .\training\run_training.py --training-folder "D:\mosquito_training_audio" --config .\training\configs\overlap025_v2.json --output .\runs\overlap025_v2.0.0
```

The program performs:

```text
WAV and JSON checks
→ 0.5-second label windows with a 0.25-second stride
→ recording-level data split
→ five-fold CNN training
→ locked test-set evaluation
→ selection of the fold with the highest validation Macro-F1
→ full INT8 TFLite quantization
→ generation of the TFLite model and C header
```

After training, the main output files are located at:

```text
runs\overlap025_v2.0.0\export\
  model_overlap025.tflite
  model_overlap025.h
  model_overlap025_verification.json
```

Their purposes are:

- `model_overlap025.tflite`: full INT8 six-class model.
- `model_overlap025.h`: C array that can be embedded in the Arduino firmware.
- `model_overlap025_verification.json`: quantization parameters, model size, and conversion-verification results.

### 5. Back Up the Current Model

Run:

```powershell
Copy-Item .\model\model_overlap025.tflite .\model\model_overlap025_previous.tflite
Copy-Item .\source\lab_realtime_recorder_v1_speedopt\models\models_embedded.h .\source\lab_realtime_recorder_v1_speedopt\models\models_embedded_previous.h
```

### 6. Update the TFLite Model in the Package

```powershell
Copy-Item .\runs\overlap025_v2.0.0\export\model_overlap025.tflite .\model\model_overlap025.tflite -Force
```

### 7. Insert the New Model into the ESP32 Source Code

Open the file generated by training:

```text
runs\overlap025_v2.0.0\export\model_overlap025.h
```

Also open the firmware model file:

```text
source\lab_realtime_recorder_v1_speedopt\models\models_embedded.h
```

In `models_embedded.h`, find:

```cpp
// ---- model_overlap025.h ----
```

Delete the complete old model block beginning with that line and ending immediately before:

```cpp
// ---- model_ovr_ae.h ----
```

At the same location, paste:

```cpp
// ---- model_overlap025.h ----
```

Then paste the complete contents of the newly generated `model_overlap025.h`.

Keep all other model blocks in `models_embedded.h`, including:

```text
model_main050
model_window025
model_ovr_ae
model_ovr_cx
model_ovr_dirus
model_ovr_minf
model_ovr_minm
model_ovr_no
```

After saving the file, verify that `model_overlap025` is defined only once:

```powershell
(Select-String -Path .\source\lab_realtime_recorder_v1_speedopt\models\models_embedded.h -Pattern "const unsigned char model_overlap025\[\]").Count
```

The correct result is:

```text
1
```

### 8. Install the Speed Optimization Patch

```powershell
powershell -ExecutionPolicy Bypass -File .\speed_patch\install_speed_patch.ps1
```

### 9. Find the ESP32 Serial Port

Connect the ESP32 and run:

```powershell
arduino-cli board list
```

Record the serial port corresponding to the Arduino Nano ESP32, for example `COM12`.

### 10. Compile and Flash the New Model

Replace `COM12` with the actual serial port:

```powershell
arduino-cli compile --upload -p COM12 --fqbn arduino:esp32:nano_nora .\source\lab_realtime_recorder_v1_speedopt
```

After compilation and flashing complete successfully, mode 4 on the ESP32 uses the newly trained `model_overlap025` six-class model.

### 11. Check the Flashing Result

Open the Arduino Serial Monitor and set it to:

```text
115200 baud
```

During a normal start, the following should appear:

```text
BOOT lab_realtime_recorder_v1
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

This starts the updated overlap025 six-class model. The ESP32 simultaneously performs:

```text
real-time audio capture
→ log-Mel feature calculation
→ six-class CNN inference
→ aggregation of the three most recent window scores
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
# firmware=lab_realtime_recorder_v1
# sample_rate_hz=16000,channels=1,bits_per_sample=16
# mode=4,calibrated=0
```

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
| `feature_ms` | Time required to calculate resampling, FFT, Mel filtering, and log-Mel features, in milliseconds. |
| `infer_ms` | Time required to execute TFLite Micro CNN inference, in milliseconds. |
| `audio_lag_ms` | Amount of time by which the predicted audio endpoint is behind the real-time capture position when the result is output, in milliseconds. |
| `latency_ms` | Total latency from the beginning of the current real-time processing cycle until the final result is obtained, in milliseconds. |
| `skipped_hops_total` | Cumulative number of skipped detection hops since the start of the session. This value increases when the system cannot keep up with the audio stream. |

In mode 4, the six class scores are normally between `0–1`, and their sum is approximately `1`. The system calculates one CNN score from a 1.5-second context every second and averages the three most recent scores, resulting in an effective audio support range of approximately 3.5 seconds.

Every completed CSV row is immediately synchronized to the SD card. After an unexpected power loss, complete rows that have already been synchronized remain available; the final row being written at the instant of power loss may be lost.

The program automatically performs WAV/JSON checking, recording-level partitioning, 0.5-second label windows with a 0.25-second stride, 1.5-second centered contexts, ESP32-equivalent resampling, 64-band log-Mel extraction, five-fold CNN training, locked-test evaluation, and full INT8 TFLite export.

Fixed frontend parameters: 8 kHz, 256-point FFT, STFT hop of 80 points, 64 Mel bands, and input tensor `1×64×151×1`. The training output directory saves a copy of the configuration, data fingerprint, five-fold results, locked-test results, exported model, and `experiment_manifest.json`. It records the v2.0.0 version from this package folder, the model hash, and the output-file hashes.

The current real-time mode calculates one 1.5-second context every second and averages the three most recent scores, giving an effective audio support range of 3.5 seconds. Command `G` runs a strict 3.0-second alignment diagnostic.

## Power and Storage

- Use a stable 5 V USB power supply rated for at least 1 A.
- Keep the INMP441 and SD module at 3.3 V logic levels.
- WAV format is 16 kHz, 16-bit, mono, using approximately 115.2 MB per hour.
- Check the remaining SD-card capacity before long-duration experiments.
