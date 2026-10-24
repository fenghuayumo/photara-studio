[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$Stage)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $Stage).Path
$forbidden = @(Get-ChildItem -LiteralPath $root -Recurse -Force | Where-Object {
    ($_.PSIsContainer -and $_.Name -match '^(models|model-agreements)$') -or
    (-not $_.PSIsContainer -and ($_.Name -match '\.(ggml|gguf|safetensors|onnx|pt|pth|ckpt|tflite|bin|agreement|verified)(\.|$)' -or
        $_.Name -match '(license-accepted|licence-accepted|^pytorch_model.*\.bin$|^grounding-dino-vocab\.txt$)'))
})
if ($forbidden.Count) {
    throw ("Release package contains model weights, caches or user agreements: " + ($forbidden.FullName -join ', '))
}
Write-Host 'Release guard passed: no model weights or user agreement records.'
