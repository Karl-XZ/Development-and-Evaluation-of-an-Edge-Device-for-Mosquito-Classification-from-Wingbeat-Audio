param(
    [string]$LibraryRoot = "$env:USERPROFILE\Documents\Arduino\libraries\Chirale_TensorFLowLite"
)

$ErrorActionPreference = 'Stop'
$expected = @{
    'src\third_party\cmsis_nn\Source\ConvolutionFunctions\arm_convolve_s8.c' = '5704CC2E94D6ADDFB44DCB6A8E796F77C0D0EEF17FBF9CBD5689A9E49B82553B'
    'src\third_party\cmsis_nn\Source\ConvolutionFunctions\arm_nn_mat_mult_kernel_s8_s16.c' = 'B163658071F9382F6D0C1432820E590EDD73F30DD7601BE57B9F162A37E4BB51'
    'src\third_party\cmsis_nn\Source\NNSupportFunctions\arm_q7_to_q15_with_offset.c' = 'A61D5A4B28A7CBCA1479169DCD8AD27B1EA00C596A3BBCF3490461225F4CD98F'
}

foreach ($relative in $expected.Keys) {
    $path = Join-Path $LibraryRoot $relative
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing patched file: $path" }
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    if ($actual -ne $expected[$relative]) { throw "Speed patch is not active: $path" }
}
Write-Host 'Verified: all three CMSIS-NN speed-patch files are active.'

