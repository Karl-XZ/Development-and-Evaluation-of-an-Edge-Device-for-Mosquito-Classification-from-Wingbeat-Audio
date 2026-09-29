from __future__ import annotations

import argparse
import csv
import re
import struct
import time
from pathlib import Path

import serial


def read_status(port: serial.Serial) -> str:
    port.reset_input_buffer()
    port.write(b"?")
    port.flush()
    deadline = time.monotonic() + 4.0
    lines: list[str] = []
    while time.monotonic() < deadline:
        raw = port.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", "replace").strip()
        lines.append(line)
        if line.startswith("STATUS "):
            return line
    raise RuntimeError(f"No STATUS response. Received: {lines}")


def select_session(port: serial.Serial, session: str) -> tuple[str, str]:
    match = re.fullmatch(r"(?:LAB)?(\d{1,3})", session.strip(), re.IGNORECASE)
    if not match:
        raise ValueError("Session must look like LAB008 or 8")
    index = int(match.group(1))
    if not 1 <= index <= 999:
        raise ValueError("Session number must be between 1 and 999")
    port.reset_input_buffer()
    port.write(f"J{index:03d}".encode("ascii"))
    port.flush()
    deadline = time.monotonic() + 4.0
    lines: list[str] = []
    while time.monotonic() < deadline:
        raw = port.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", "replace").strip()
        lines.append(line)
        selected = re.search(
            r"SESSION_SELECTED wav=(LAB\d{3}\.WAV) csv=(LAB\d{3}\.CSV)", line
        )
        if selected:
            return selected.group(1), selected.group(2)
        if line.startswith("SESSION_SELECT_ERROR"):
            raise RuntimeError(line)
    raise RuntimeError(f"No SESSION_SELECTED response. Received: {lines}")


def receive_file(port: serial.Serial, command: bytes, magic: bytes, target: Path) -> int:
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite: {target}")
    partial = target.with_suffix(target.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"Inspect or move the existing partial file first: {partial}")

    port.reset_input_buffer()
    port.write(command)
    port.flush()
    header = port.read(8)
    if len(header) != 8 or header[:4] != magic:
        raise RuntimeError(f"Unexpected transfer header for {target.name}: {header!r}")
    total = struct.unpack("<I", header[4:])[0]
    if total <= 0:
        raise RuntimeError(f"Board reported an empty file for {target.name}")

    received = 0
    last_report = -1
    with partial.open("xb") as output:
        while received < total:
            block = port.read(min(8192, total - received))
            if not block:
                raise TimeoutError(f"Transfer stopped at {received}/{total} bytes")
            output.write(block)
            received += len(block)
            percent = int(received * 100 / total)
            if percent != last_report:
                print(f"{target.name}: {percent:3d}% ({received}/{total} bytes)", end="\r", flush=True)
                last_report = percent
        output.flush()
    print()
    partial.replace(target)
    return total


def validate_wav(path: Path) -> None:
    data = path.read_bytes()
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise RuntimeError(f"Invalid RIFF/WAVE header: {path}")
    declared = struct.unpack("<I", data[4:8])[0] + 8
    if declared != len(data):
        raise RuntimeError(f"WAV length mismatch: declared={declared}, actual={len(data)}")


def validate_csv(path: Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    header_index = next((index for index, line in enumerate(lines) if line.startswith("result_index,")), None)
    if header_index is None:
        raise RuntimeError(f"Prediction CSV header is missing: {path}")
    reader = csv.DictReader(lines[header_index:])
    rows = list(reader)
    required = {"result_index", "audio_end_ms", "class", "latency_ms"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"Prediction CSV has no complete result row: {path}")
    field_count = len(reader.fieldnames or [])
    malformed = [index for index, row in enumerate(rows, 1)
                 if None in row or len(row) != field_count or any(value is None for value in row.values())]
    if malformed:
        raise RuntimeError(f"Prediction CSV contains malformed rows {malformed[:5]}: {path}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and validate the latest LABnnn WAV/CSV pair")
    parser.add_argument("--port", required=True, help="Arduino Nano ESP32 serial port, for example COM12")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--session", help="Historical session to download, for example LAB008")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with serial.Serial(args.port, 115200, timeout=2.0, write_timeout=2.0) as port:
        time.sleep(1.0)
        status = read_status(port)
        if "live=0" not in status or "recording=0" not in status:
            raise RuntimeError(f"Stop and finalize the session before downloading: {status}")
        if args.session:
            wav_name, csv_name = select_session(port, args.session)
        else:
            wav_match = re.search(r"\bwav=(LAB\d{3}\.WAV)\b", status)
            csv_match = re.search(r"\bcsv=(LAB\d{3}\.CSV)\b", status)
            if not wav_match or not csv_match:
                raise RuntimeError(f"No completed LAB session is reported: {status}")
            wav_name, csv_name = wav_match.group(1), csv_match.group(1)
        wav_target = args.output_dir / wav_name
        csv_target = args.output_dir / csv_name
        receive_file(port, b"D", b"WAV1", wav_target)
        receive_file(port, b"L", b"CSV1", csv_target)

    validate_wav(wav_target)
    rows = validate_csv(csv_target)
    print(f"VALIDATED wav={wav_target} csv={csv_target} prediction_rows={rows}")


if __name__ == "__main__":
    main()
