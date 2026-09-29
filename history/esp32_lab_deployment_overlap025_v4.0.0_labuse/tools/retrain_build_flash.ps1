param(
    [Parameter(Mandatory = $true)]
    [string]$TrainingFolder,
    [string]$Port = "",
    [string]$Output = "",
    [switch]$SmokeTest,
    [switch]$NoUpload
)

$ErrorActionPreference = 'Stop'
$packageRoot = Split-Path -Parent $PSScriptRoot
$config = Join-Path $packageRoot 'training\configs\overlap025_pcen_exact3s_v004.json'
$trainingScript = Join-Path $packageRoot 'training\run_training.py'
$modelDirectory = Join-Path $packageRoot 'model'
$firmwareModel = Join-Path $packageRoot 'source\lab_realtime_recorder_v004_pcen_exact3s\models\model_overlap025.h'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
if (-not $Output) { $Output = Join-Path $packageRoot "runs\overlap025_pcen_exact3s_v004_$stamp" }

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
$tflite = Join-Path $export 'model_overlap025.tflite'
$header = Join-Path $export 'model_overlap025.h'
$verification = Join-Path $export 'model_overlap025_verification.json'
foreach ($required in @($tflite, $header, $verification)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Missing training output: $required" }
}

$backup = Join-Path $packageRoot "model_backup_$stamp"
New-Item -ItemType Directory -Path $backup | Out-Null
Copy-Item -LiteralPath (Join-Path $modelDirectory 'model_overlap025.tflite') -Destination $backup
Copy-Item -LiteralPath $firmwareModel -Destination $backup
Copy-Item -LiteralPath $tflite -Destination (Join-Path $modelDirectory 'model_overlap025.tflite') -Force
Copy-Item -LiteralPath $verification -Destination (Join-Path $modelDirectory 'model_overlap025_verification.json') -Force
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

Write-Host "Retraining, model insertion, automatic speed patch, compilation and deployment completed."
Write-Host "Run output: $Output"
Write-Host "Previous model backup: $backup"
