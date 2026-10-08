param(
    [Parameter(Mandatory = $true)]
    [string]$TrainingFolder,
    [string]$Port = "",
    [string]$Output = "",
    [ValidateSet('logmel', 'pcen')]
    [string]$Frontend = 'logmel',
    [switch]$SmokeTest,
    [switch]$NoUpload
)

$ErrorActionPreference = 'Stop'
$packageRoot = Split-Path -Parent $PSScriptRoot
$config = Join-Path $packageRoot "training\configs\overlap025_${Frontend}_exact3s_v005.json"
$trainingScript = Join-Path $packageRoot 'training\run_training.py'
$modelDirectory = Join-Path $packageRoot 'model'
$symbol = if ($Frontend -eq 'pcen') { 'model_overlap025' } else { 'model_overlap025_logmel' }
$firmwareModel = Join-Path $packageRoot "source\lab_realtime_recorder_v005_dual_exact3s\models\$symbol.h"
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
if (-not $Output) { $Output = Join-Path $packageRoot "runs\overlap025_${Frontend}_exact3s_v005_$stamp" }

$trainingArguments = @(
    '-u', $trainingScript,
    '--training-folder', $TrainingFolder,
    '--config', $config,
    '--output', $Output
)
if ($SmokeTest) { $trainingArguments += '--smoke-test' }
python @trainingArguments
if ($LASTEXITCODE -ne 0) { throw "Training/export failed with exit code $LASTEXITCODE" }

$export = Join-Path $Output 'export'
$tflite = Join-Path $export "$symbol.tflite"
$header = Join-Path $export "$symbol.h"
$verification = Join-Path $export "${symbol}_verification.json"
foreach ($required in @($tflite, $header, $verification)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Missing training output: $required" }
}

$backup = Join-Path $packageRoot "model_backup_$stamp"
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

if ($NoUpload) {
    Write-Host "Retraining, $Frontend model insertion, automatic speed patch, and compilation completed; upload was skipped."
} else {
    Write-Host "Retraining, $Frontend model insertion, automatic speed patch, compilation, and deployment completed."
}
Write-Host "Run output: $Output"
Write-Host "Previous model backup: $backup"
