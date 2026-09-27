# Runs one splat training pass and samples device-wide VRAM and utilization.
# On Windows/WDDM nvidia-smi cannot attribute VRAM to this process, so compare
# runs made while other GPU workloads are quiet.
#
#   scripts/measure_splat_vram.ps1 -Tag after -Output artifacts/vram_probe/after.ply
param(
    [Parameter(Mandatory = $true)][string]$Tag,
    [Parameter(Mandatory = $true)][string]$Output,
    # Dataset root holding images/ and sparse/ or sparse/0.
    [string]$Dataset = 'D:/ScanVideo/antman_nomask',
    [int]$Iterations = 1200,
    [int]$LogInterval = 300,
    [ValidateSet('cuda', 'vulkan')][string]$Backend = 'cuda',
    [switch]$Densification,
    [switch]$ProfileCuda,
    # 1 restores the pre-optimization behavior of always allocating the
    # depth/normal channels, for an A/B inside a single binary.
    [string]$ForceGeometryChannels = ''
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $root 'artifacts/vram_probe_20260916'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$sparse = Join-Path $Dataset 'sparse/0'
if (-not (Test-Path $sparse)) { $sparse = Join-Path $Dataset 'sparse' }
if (-not (Test-Path $sparse)) { throw "No COLMAP sparse dataset found below $Dataset" }
$arguments = @(
    '--mode', 'global',
    '--images', "$Dataset/images",
    '--splat-dataset', $sparse,
    '--output', $Output,
    '--splat',
    '--splat-iterations', "$Iterations",
    '--splat-log-interval', "$LogInterval",
    '--backend', $Backend,
    '--splat-strategy', 'adc_igs',
    '--splat-densification-cap', '1000000',
    '--splat-max-resolution', '1920',
    '--splat-progressive-resolution=false',
    '--splat-eval-split-every', '0',
    '--splat-cache-auto',
    '--splat-prefetch-views', '4'
)
if ($Backend -eq 'cuda' -and $ProfileCuda) {
    $arguments += @('--splat-profile-cuda', '--splat-profile-interval', '100')
}
if ($Densification) { $arguments += '--splat-densification=true' }

$exe = Join-Path $root 'build/photara/Release/photara.exe'
$log = Join-Path $logDir "$Tag.log"
if ($ForceGeometryChannels -ne '') {
    $env:PHOTARA_SPLAT_FORCE_GEOMETRY_CHANNELS = $ForceGeometryChannels
    # Must be set together: the two knobs cover the caller-owned gradient
    # images and the rasterizer-owned per-Gaussian workspace.
    $env:PHOTARA_SPLAT_FORCE_GEOMETRY_WORKSPACE = $ForceGeometryChannels
}
$baselineRow = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null | Select-Object -First 1
$baseline = [int]$baselineRow.Trim()
$process = Start-Process -FilePath $exe -ArgumentList $arguments -PassThru `
    -NoNewWindow -RedirectStandardOutput $log -RedirectStandardError "$log.err"
Remove-Item Env:PHOTARA_SPLAT_FORCE_GEOMETRY_CHANNELS -ErrorAction SilentlyContinue
Remove-Item Env:PHOTARA_SPLAT_FORCE_GEOMETRY_WORKSPACE -ErrorAction SilentlyContinue

$peak = 0
$floor = [int]::MaxValue
$samples = [System.Collections.Generic.List[string]]::new()
$samples.Add('timestamp,memory_mib,util_pct')
while (-not $process.HasExited) {
    # Windows/WDDM does not report per-process used_memory (nvidia-smi answers
    # [N/A]), so sample the device total and report it together with the idle
    # floor observed over the same loop. Both numbers are measured the same way
    # for every run, so before/after comparisons stay meaningful.
    $rows = & nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>$null
    foreach ($row in $rows) {
        $used = 0
        $columns = $row.Split(',')
        if ([int]::TryParse($columns[0].Trim(), [ref]$used)) {
            if ($used -gt $peak) { $peak = $used }
            if ($used -lt $floor) { $floor = $used }
        }
        $util = 0
        if ($columns.Length -gt 1) { [void][int]::TryParse($columns[1].Trim(), [ref]$util) }
        $samples.Add("$((Get-Date).ToString('o')),$used,$util")
    }
    Start-Sleep -Milliseconds 100
}
$samples | Set-Content -Path "$log.gpu.csv"

if ($process.ExitCode -ne 0) { throw "$Backend training failed with exit code $($process.ExitCode); see $log" }
$trainingLines = Get-Content $log | Where-Object { $_ -match 'splat iteration=' }
$trainingSamples = @()
if ($trainingLines.Count -ge 2) {
    $start = [datetime]$trainingLines[0].Substring(0, 23)
    $end = [datetime]$trainingLines[-1].Substring(0, 23)
    $trainingSamples = @(Import-Csv "$log.gpu.csv" | Where-Object {
        [datetime]$_.timestamp -ge $start -and [datetime]$_.timestamp -le $end
    })
}
$trainingPeak = if ($trainingSamples.Count) {
    ($trainingSamples | Measure-Object memory_mib -Maximum).Maximum
} else { 0 }
$trainingUtil = if ($trainingSamples.Count -ge 10) {
    [math]::Round(($trainingSamples | Measure-Object util_pct -Average).Average, 1)
} else { 'n/a' }
"PEAK_VRAM_MIB tag=$Tag backend=$Backend baseline=$baseline peak=$peak delta=$($peak - $baseline) training_peak=$trainingPeak training_delta=$($trainingPeak - $baseline) training_util_avg_pct=$trainingUtil samples=$($trainingSamples.Count) trace=$log.gpu.csv"
