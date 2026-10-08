param(
    [string]$Port = "",
    [switch]$NoUpload
)

$ErrorActionPreference = 'Stop'
$packageRoot = Split-Path -Parent $PSScriptRoot
$sketch = Join-Path $packageRoot 'source\lab_realtime_recorder_v005_dual_exact3s'
$patchScript = Join-Path $packageRoot 'speed_patch\install_speed_patch.ps1'

if (-not $Port -and -not $NoUpload) {
    $boardRows = arduino-cli board list --format json | ConvertFrom-Json
    $match = $boardRows.detected_ports | Where-Object {
        $_.matching_boards -and ($_.matching_boards | Where-Object { $_.fqbn -eq 'arduino:esp32:nano_nora' })
    } | Select-Object -First 1
    if (-not $match) { throw 'Arduino Nano ESP32 serial port was not detected.' }
    $Port = $match.port.address
}

& powershell -ExecutionPolicy Bypass -File $patchScript
if ($LASTEXITCODE -ne 0) { throw "Speed-patch installation failed with exit code $LASTEXITCODE" }

arduino-cli compile --fqbn arduino:esp32:nano_nora --export-binaries $sketch
if ($LASTEXITCODE -ne 0) { throw "Compilation failed with exit code $LASTEXITCODE" }

$binary = Get-ChildItem -LiteralPath $sketch -Recurse -Filter '*.ino.bin' -File |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $binary) { throw 'Compiled application binary was not found.' }
$buildDirectory = $binary.Directory.FullName
$precompiled = Join-Path $packageRoot 'precompiled'
New-Item -ItemType Directory -Force -Path $precompiled | Out-Null
foreach ($suffix in @('.ino.bin', '.ino.bootloader.bin', '.ino.partitions.bin')) {
    $artifact = Get-ChildItem -LiteralPath $buildDirectory -Filter "*$suffix" -File | Select-Object -First 1
    if (-not $artifact) { throw "Missing compiled artifact: *$suffix" }
    Copy-Item -LiteralPath $artifact.FullName -Destination (Join-Path $precompiled $artifact.Name) -Force
}
if (Test-Path -LiteralPath (Join-Path $buildDirectory 'partitions.csv')) {
    Copy-Item -LiteralPath (Join-Path $buildDirectory 'partitions.csv') -Destination (Join-Path $precompiled 'partitions.csv') -Force
}

if (-not $NoUpload) {
    arduino-cli upload -p $Port --fqbn arduino:esp32:nano_nora $sketch
    if ($LASTEXITCODE -ne 0) { throw "Upload failed with exit code $LASTEXITCODE" }
    Write-Host "Compiled with the verified speed patch and uploaded to $Port."
} else {
    Write-Host 'Compiled with the verified speed patch; upload was skipped.'
}
