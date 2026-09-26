param(
    [string]$Dataset = 'D:/ScanVideo/antman_nomask',
    [int]$Iterations = 220,
    [int]$LogInterval = 20,
    [int]$MaxResolution = 1920,
    [ValidateSet('cuda', 'vulkan')][string[]]$Backends = @('cuda', 'vulkan'),
    [switch]$Densification,
    [switch]$ProfileVulkanBackward
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root 'build/photara/Release/photara.exe'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$resultDir = Join-Path $root "artifacts/splat_backend_$stamp"
New-Item -ItemType Directory -Force -Path $resultDir | Out-Null
$sparse = Join-Path $Dataset 'sparse/0'
if (-not (Test-Path $sparse)) {
    $sparse = Join-Path $Dataset 'sparse'
}
if (-not (Test-Path $sparse)) {
    throw "No COLMAP sparse dataset found below $Dataset"
}

$common = @(
    '--mode', 'global',
    '--images', (Join-Path $Dataset 'images'),
    '--splat-dataset', $sparse,
    '--splat',
    '--splat-iterations', "$Iterations",
    '--splat-log-interval', "$LogInterval",
    '--splat-strategy', 'adc_igs',
    "--splat-densification=$($Densification.IsPresent.ToString().ToLowerInvariant())",
    '--splat-densification-cap', '1000000',
    '--splat-max-resolution', "$MaxResolution",
    '--splat-progressive-resolution=false',
    '--splat-eval-split-every', '0',
    '--splat-preview-interval', '0',
    '--splat-use-mask=false',
    '--splat-cache-auto',
    '--splat-prefetch-views', '4'
)

foreach ($backend in $Backends) {
    $output = Join-Path $resultDir "$backend.ply"
    $stdout = Join-Path $resultDir "$backend.log"
    $stderr = Join-Path $resultDir "$backend.err.log"
    $arguments = $common + @('--backend', $backend, '--output', $output)
    if ($backend -eq 'cuda') {
        $arguments += @('--splat-profile-cuda', '--splat-profile-interval', '100')
    }
    if ($backend -eq 'vulkan' -and $ProfileVulkanBackward) {
        $env:SPLAT_DRENDER_PROFILE_BACKWARD = '1'
        $env:SPLAT_DRENDER_PROFILE_BACKWARD_INTERVAL = '100'
    }
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        & $exe @arguments 1> $stdout 2> $stderr
        if ($LASTEXITCODE -ne 0) {
            throw "$backend training failed with exit code $LASTEXITCODE"
        }
    } finally {
        $watch.Stop()
        Remove-Item Env:SPLAT_DRENDER_PROFILE_BACKWARD -ErrorAction SilentlyContinue
        Remove-Item Env:SPLAT_DRENDER_PROFILE_BACKWARD_INTERVAL -ErrorAction SilentlyContinue
    }
    "backend=$backend elapsed_s=$([math]::Round($watch.Elapsed.TotalSeconds, 3)) log=$stdout"
}

"result_dir=$resultDir"
