param(
    [Parameter(Mandatory = $true)]
    [string]$TrainingFolder,
    [string]$Port = "",
    [string]$Output = "",
    [ValidateSet('logmel', 'pcen')]
    [string]$Frontend = 'logmel',
    [ValidateSet('pair', 'lowpass')]
    [string]$Decimator = 'pair',
    [switch]$SmokeTest,
    [switch]$ReuseCompletedRun,
    [switch]$NoUpload
)

$ErrorActionPreference = 'Stop'
$packageRoot = Split-Path -Parent $PSScriptRoot
$config = Join-Path $packageRoot "training\configs\overlap025_${Frontend}_${Decimator}_exact3s_v006.json"
$trainingScript = Join-Path $packageRoot 'training\run_training.py'
$modelDirectory = Join-Path $packageRoot 'model'
$symbol = if ($Decimator -eq 'lowpass') { "model_overlap025_${Frontend}_lowpass" } elseif ($Frontend -eq 'pcen') { 'model_overlap025' } else { 'model_overlap025_logmel' }
$firmwareModel = Join-Path $packageRoot "source\lab_realtime_recorder_v006_quad_exact3s\models\$symbol.h"
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
if (-not $Output) { $Output = Join-Path $packageRoot "runs\overlap025_${Frontend}_${Decimator}_exact3s_v006_$stamp" }
if ($ReuseCompletedRun -and ($SmokeTest -or -not $Output)) { throw '-ReuseCompletedRun requires an existing -Output and cannot be combined with -SmokeTest.' }

if ($ReuseCompletedRun) {
    $runManifest = Join-Path $Output 'experiment_manifest.json'
    if (-not (Test-Path -LiteralPath $runManifest -PathType Leaf)) { throw "Completed-run manifest missing: $runManifest" }
    $manifest = Get-Content -LiteralPath $runManifest -Raw | ConvertFrom-Json
    if ($manifest.status -ne 'completed' -or $manifest.smoke_test) { throw 'Run is incomplete or only a smoke test.' }
    if ($manifest.config_sha256 -ne (Get-FileHash -LiteralPath $config -Algorithm SHA256).Hash.ToLowerInvariant()) {
        throw 'Completed-run config does not match the selected package config.'
    }
} else {
    $trainingArguments = @(
        '-u', $trainingScript,
        '--training-folder', $TrainingFolder,
        '--config', $config,
        '--output', $Output
    )
    if ($SmokeTest) { $trainingArguments += '--smoke-test' }
    python @trainingArguments
    if ($LASTEXITCODE -ne 0) { throw "Training/export failed with exit code $LASTEXITCODE" }
}

$export = Join-Path $Output 'export'
$tflite = Join-Path $export "$symbol.tflite"
$header = Join-Path $export "$symbol.h"
$verification = Join-Path $export "${symbol}_verification.json"
foreach ($required in @($tflite, $header, $verification)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Missing training output: $required" }
}
$report = Get-Content -LiteralPath $verification -Raw | ConvertFrom-Json
$expectedMode = if ($Decimator -eq 'lowpass') { 'esp32_lowpass' } else { 'esp32_pair' }
if ($report.frontend -ne $Frontend -or $report.resample_mode -ne $expectedMode) {
    throw "Exported model/frontend/decimator mismatch: frontend=$($report.frontend) mode=$($report.resample_mode)"
}
if ($SmokeTest) {
    Write-Host "Smoke training/export passed for $Frontend/$Decimator. Included models and firmware were not changed."
    Write-Host "Run output: $Output"
    return
}

$backup = Join-Path $Output "previous_package_model_$stamp"
New-Item -ItemType Directory -Path $backup | Out-Null
$packageTflite = Join-Path $modelDirectory "$symbol.tflite"
$packageVerification = Join-Path $modelDirectory "${symbol}_verification.json"
Copy-Item -LiteralPath $packageTflite -Destination $backup
Copy-Item -LiteralPath $firmwareModel -Destination $backup
Copy-Item -LiteralPath $tflite -Destination $packageTflite -Force
Copy-Item -LiteralPath $verification -Destination $packageVerification -Force
Copy-Item -LiteralPath $header -Destination $firmwareModel -Force

$buildScript = Join-Path $PSScriptRoot 'build_flash_included.ps1'
if ($NoUpload) {
    & powershell -ExecutionPolicy Bypass -File $buildScript -NoUpload
} elseif ($Port) {
    & powershell -ExecutionPolicy Bypass -File $buildScript -Port $Port
} else {
    & powershell -ExecutionPolicy Bypass -File $buildScript
}
if ($LASTEXITCODE -ne 0) { throw "Build/flash failed with exit code $LASTEXITCODE" }

if (-not $NoUpload) {
    Start-Sleep -Seconds 8
    if (-not $Port) {
        $boardRows = arduino-cli board list --format json | ConvertFrom-Json
        $match = $boardRows.detected_ports | Where-Object {
            $_.matching_boards -and ($_.matching_boards | Where-Object { $_.fqbn -eq 'arduino:esp32:nano_nora' })
        } | Select-Object -First 1
        if (-not $match) { throw 'ESP32 serial port not found for mandatory parity check.' }
        $Port = $match.port.address
    }
    $parityOutput = Join-Path $Output 'pc_esp32_same_pcm_parity.json'
    python -u (Join-Path $PSScriptRoot 'verify_pc_esp32_parity.py') `
        --port $Port --frontend $Frontend --decimator $Decimator `
        --checkpoint $report.checkpoint --manifest (Join-Path $Output 'dataset\test.csv') `
        --model $packageTflite --verification $packageVerification --output $parityOutput
    if ($LASTEXITCODE -ne 0) { throw "PC/ESP32 same-PCM parity failed for $Frontend/$Decimator" }
    Write-Host "PC/ESP32 same-PCM parity report: $parityOutput"
}

if ($NoUpload) {
    Write-Host "Retraining, $Frontend/$Decimator model insertion, automatic speed patch, and compilation completed; upload was skipped."
} else {
    Write-Host "Retraining, $Frontend/$Decimator model insertion, automatic speed patch, compilation, and deployment completed."
}
Write-Host "Run output: $Output"
Write-Host "Previous model backup: $backup"
