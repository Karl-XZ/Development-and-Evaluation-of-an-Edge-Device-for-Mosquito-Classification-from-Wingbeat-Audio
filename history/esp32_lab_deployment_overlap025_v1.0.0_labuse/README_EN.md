# Mosquito Six-Class ESP32 Deployment Package

Version: **v1.0.0** (historical archive of the original unversioned deployment)

This package reproduces the laboratory real-time classifier on an Arduino Nano ESP32. It performs six-class inference, records the complete microphone stream, and writes timestamped predictions to CSV. Mode `4` (overlap025 aggregation) is the default mode.

## Package contents

- `source/`: complete Arduino source and embedded model headers.
- `model/model_overlap025.tflite`: INT8 TFLite model used by mode 4.
- `precompiled/`: ready-to-upload firmware.
- `speed_patch/`: CMSIS-NN patch required to rebuild the optimized firmware.
- `tools/download_lab_session.py`: downloads the latest WAV and CSV through the serial port.
- `PINOUT.csv`: microphone and SD-card wiring.
- `DEPENDENCIES.txt`: required software, board package, and library versions.

## Quick reproduction

1. Wire the device according to `PINOUT.csv` and insert a FAT32 SD card.
2. Install the software and libraries listed in `DEPENDENCIES.txt`.
3. Close the Arduino Serial Monitor and identify the port:

```powershell
arduino-cli board list
```

4. Replace `COM12` with the actual port and upload the precompiled firmware:

```powershell
arduino-cli upload -p COM12 --fqbn arduino:esp32:nano_nora --input-dir ".\precompiled" ".\source\lab_realtime_recorder_v1_speedopt"
```

After upload or power-on, the firmware initializes the SD card, microphone, and models. It then waits ten seconds and starts mode 4 automatically. Send `E` during the countdown to cancel automatic startup.

## Serial commands

Use `115200` baud.

- `?`: show status.
- `4`: start mode 4 and create `LABnnn.WAV` plus `LABnnn.CSV`.
- `E`: stop inference and close the WAV and CSV safely.
- `C4`: record 10 seconds of local No-mosquito background and calculate a temporary calibration.
- `C4S`: start mode 4 with the current calibration.
- `R`: clear the calibration stored in RAM.
- `D`: send the latest WAV through the serial port.
- `L`: send the latest CSV through the serial port.

Calibration is stored in RAM only and disappears after restart or power loss.

Before removing power or the SD card, wait for:

```text
SESSION_RECORD_DONE ...
LIVE_STOPPED reason=user
```

## Download recordings and predictions

Close the Arduino Serial Monitor and run:

```powershell
python ".\tools\download_lab_session.py" --port COM12 --output-dir ".\downloaded_sessions"
```

The tool receives data into temporary files and creates the final WAV and CSV only after validation. Existing files are not overwritten.

## Rebuild from source

After installing the dependencies, run:

```powershell
powershell -ExecutionPolicy Bypass -File ".\speed_patch\install_speed_patch.ps1"
arduino-cli compile --fqbn arduino:esp32:nano_nora ".\source\lab_realtime_recorder_v1_speedopt"
arduino-cli upload -p COM12 --fqbn arduino:esp32:nano_nora ".\source\lab_realtime_recorder_v1_speedopt"
```

The patch installer verifies the target library files and creates backups before modification. It stops when the installed version does not match.

## Power and storage

- Use a stable 5 V USB supply capable of at least 1 A.
- INMP441 and the SD module use 3.3 V logic.
- WAV output is 16 kHz, 16-bit mono and uses about 115.2 MB per hour.
- Use mode 4 for normal experiments; modes `1/2/3` remain available for comparison.
