# Runs one splat training pass and reports the peak per-process device memory
# that nvidia-smi attributes to it, so a VRAM change can be checked without
# relying on the training build's own instrumentation.
#
#   scripts/measure_splat_vram.ps1 -Tag after -Output artifacts/vram_probe/after.ply
param(
    [Parameter(Mandatory = $true)][string]$Tag,
    [Parameter(Mandatory = $true)][string]$Output,
    # Dataset root holding images/ and sparse/0; defaults to the antman capture.
    [string]$Dataset = 'D:/ScanVideo/antman_nomask',
    [int]$Iterations = 1200,
    [int]$LogInterval = 300,
    # 1 restores the pre-optimization behavior of always allocating the
    # depth/normal channels, for an A/B inside a single binary.
    [string]$ForceGeometryChannels = ''
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $root 'artifacts/vram_probe_20260916'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$arguments = @(
    '--mode', 'global',
    '--images', "$Dataset/images",
    '--splat-dataset', "$Dataset/sparse/0",
    '--output', $Output,
    '--splat',
    '--splat-iterations', "$Iterations",
    '--splat-log-interval', "$LogInterval",
    '--splat-profile-cuda',
    '--splat-profile-interval', '100',
    '--splat-strategy', 'adc_igs',
    '--splat-densification-cap', '1000000',
    '--splat-max-resolution', '1920',
    '--splat-progressive-resolution=false',
    '--splat-eval-split-every', '0',
    '--splat-cache-auto',
    '--splat-prefetch-views', '4'
)

$exe = Join-Path $root 'build/photara/Release/photara.exe'
$log = Join-Path $logDir "$Tag.log"
if ($ForceGeometryChannels -ne '') {
    $env:PHOTARA_SPLAT_FORCE_GEOMETRY_CHANNELS = $ForceGeometryChannels
    # Must be set together: the two knobs cover the caller-owned gradient
    # images and the rasterizer-owned per-Gaussian workspace.
    $env:PHOTARA_SPLAT_FORCE_GEOMETRY_WORKSPACE = $ForceGeometryChannels
}
$process = Start-Process -FilePath $exe -ArgumentList $arguments -PassThru `
    -NoNewWindow -RedirectStandardOutput $log -RedirectStandardError "$log.err"
Remove-Item Env:PHOTARA_SPLAT_FORCE_GEOMETRY_CHANNELS -ErrorAction SilentlyContinue
Remove-Item Env:PHOTARA_SPLAT_FORCE_GEOMETRY_WORKSPACE -ErrorAction SilentlyContinue

$peak = 0
$floor = [int]::MaxValue
while (-not $process.HasExited) {
    $rows = & nvidia-smi --query-compute-apps=pid,used_memory `
        --format=csv,noheader,nounits 2>$null
    # Windows/WDDM does not report per-process used_memory (nvidia-smi answers
    # [N/A]), so sample the device total and report it together with the idle
    # floor observed over the same loop. Both numbers are measured the same way
    # for every run, so before/after comparisons stay meaningful.
    $rows = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
    foreach ($row in $rows) {
        $used = 0
        if ([int]::TryParse($row.Trim(), [ref]$used)) {
            if ($used -gt $peak) { $peak = $used }
            if ($used -lt $floor) { $floor = $used }
        }
    }
    Start-Sleep -Milliseconds 100
}

"PEAK_VRAM_MIB tag=$Tag peak=$peak floor=$floor delta=$($peak - $floor)"
