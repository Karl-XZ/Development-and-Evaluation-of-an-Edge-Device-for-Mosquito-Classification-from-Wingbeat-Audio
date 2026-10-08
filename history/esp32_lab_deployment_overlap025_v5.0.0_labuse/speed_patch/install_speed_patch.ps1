param(
    [string]$LibraryRoot = "$env:USERPROFILE\Documents\Arduino\libraries\Chirale_TensorFLowLite"
)

$ErrorActionPreference = 'Stop'
$patchDirectory = $PSScriptRoot
$timeStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$backupDirectory = Join-Path $LibraryRoot "codex_speed_patch_backup_$timeStamp"

$items = @(
    @{
        Name = 'arm_convolve_s8.c'
        Relative = 'src\third_party\cmsis_nn\Source\ConvolutionFunctions\arm_convolve_s8.c'
        OriginalSha256 = '99993EF01C67BD262E241F21D8AB8FDA5B9BEA98B9387CC396D737BE3B1017E1'
        PatchedSha256 = '5704CC2E94D6ADDFB44DCB6A8E796F77C0D0EEF17FBF9CBD5689A9E49B82553B'
    },
    @{
        Name = 'arm_nn_mat_mult_kernel_s8_s16.c'
        Relative = 'src\third_party\cmsis_nn\Source\ConvolutionFunctions\arm_nn_mat_mult_kernel_s8_s16.c'
        OriginalSha256 = '34E8ED9E9EF13E7C668303CF79CCCD39445830D8A2D34D5A6E569E75330343E7'
        PatchedSha256 = 'B163658071F9382F6D0C1432820E590EDD73F30DD7601BE57B9F162A37E4BB51'
    },
    @{
        Name = 'arm_q7_to_q15_with_offset.c'
        Relative = 'src\third_party\cmsis_nn\Source\NNSupportFunctions\arm_q7_to_q15_with_offset.c'
        OriginalSha256 = 'F4B868B91CCA553EA64DC0F842A554A765A8DAF1AF4694F274E7FF032C31C9AE'
        PatchedSha256 = 'A61D5A4B28A7CBCA1479169DCD8AD27B1EA00C596A3BBCF3490461225F4CD98F'
    }
)

if (-not (Test-Path -LiteralPath $LibraryRoot -PathType Container)) {
    throw "Chirale_TensorFLowLite library not found: $LibraryRoot"
}

foreach ($item in $items) {
    $target = Join-Path $LibraryRoot $item.Relative
    $patch = Join-Path $patchDirectory $item.Name
    if (-not (Test-Path -LiteralPath $target -PathType Leaf)) { throw "Missing library file: $target" }
    if (-not (Test-Path -LiteralPath $patch -PathType Leaf)) { throw "Missing patch file: $patch" }
    $targetHash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash
    $patchHash = (Get-FileHash -LiteralPath $patch -Algorithm SHA256).Hash
    if ($patchHash -ne $item.PatchedSha256) { throw "Patch hash mismatch: $patch" }
    if ($targetHash -ne $item.OriginalSha256 -and $targetHash -ne $item.PatchedSha256) {
        throw "Installed file has an unknown version; no files were changed: $target"
    }
}

New-Item -ItemType Directory -Path $backupDirectory | Out-Null
foreach ($item in $items) {
    $target = Join-Path $LibraryRoot $item.Relative
    Copy-Item -LiteralPath $target -Destination (Join-Path $backupDirectory $item.Name)
}

foreach ($item in $items) {
    $target = Join-Path $LibraryRoot $item.Relative
    $patch = Join-Path $patchDirectory $item.Name
    Copy-Item -LiteralPath $patch -Destination $target -Force
    $installedHash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash
    if ($installedHash -ne $item.PatchedSha256) { throw "Verification failed after copying: $target" }
}

Write-Host "Speed patch installed successfully."
Write-Host "Backup: $backupDirectory"
