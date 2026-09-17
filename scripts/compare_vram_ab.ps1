# Compares two artifacts/vram_probe logs produced by measure_splat_vram.ps1:
# windowed device bytes (from the training build's own probe), step time and
# image quality. Windows are aligned by iteration so the two runs can be
# differenced even though the device-wide probe is noisy.
#
#   scripts/compare_vram_ab.ps1 -Before artifacts/.../iphone_before.log `
#                               -After  artifacts/.../iphone_after.log
param(
    [Parameter(Mandatory = $true)][string]$Before,
    [Parameter(Mandatory = $true)][string]$After,
    [int]$TailWindows = 30
)

function Get-Windows([string]$path) {
    $windows = @()
    foreach ($line in Get-Content $path) {
        if ($line -notmatch 'splat_cuda_vram') { continue }
        if ($line -match 'sampled_iteration=(\d+) vram_start_mib=(\d+) ' +
                         'vram_peak_mib=(\d+) ') {
            $windows += [pscustomobject]@{
                Iter = [int]$Matches[1]
                Start = [int]$Matches[2]
                Peak = [int]$Matches[3]
            }
        }
    }
    return $windows
}

function Get-Steps([string]$path) {
    $steps = @{}
    foreach ($line in Get-Content $path) {
        if ($line -match 'splat iteration=(\d+)/\d+ .*gaussians=(\d+).*step_ms=([\d.]+)') {
            $steps[[int]$Matches[1]] = [pscustomobject]@{
                Gaussians = [int]$Matches[2]
                StepMs = [double]$Matches[3]
            }
        }
    }
    return $steps
}

function Summarize($windows, [string]$label) {
    $tail = $windows | Select-Object -Last $TailWindows
    $startMean = ($tail | Measure-Object Start -Average).Average
    $peakMean = ($tail | Measure-Object Peak -Average).Average
    "{0}: windows={1} tail={2} start_mean={3:N1} MiB start_min={4} peak_mean={5:N1} MiB" -f `
        $label, $windows.Count, $tail.Count, $startMean,
        ($tail | Measure-Object Start -Minimum).Minimum, $peakMean
}

$beforeWindows = Get-Windows $Before
$afterWindows = Get-Windows $After
Summarize $beforeWindows 'before'
Summarize $afterWindows 'after'

$beforeByIter = @{}
foreach ($w in $beforeWindows) { $beforeByIter[$w.Iter] = $w }
$deltas = @()
foreach ($w in $afterWindows) {
    if ($beforeByIter.ContainsKey($w.Iter)) {
        $deltas += $w.Start - $beforeByIter[$w.Iter].Start
    }
}
if ($deltas.Count -gt 0) {
    $matched = $deltas | Select-Object -Last $TailWindows
    "paired_delta_mib: mean={0:N1} min={1} max={2} n={3}" -f `
        (($matched | Measure-Object -Average).Average),
        ($matched | Measure-Object -Minimum).Minimum,
        ($matched | Measure-Object -Maximum).Maximum, $matched.Count
}

foreach ($pair in @(@('before', $Before), @('after', $After))) {
    $steps = Get-Steps $pair[1]
    $keys = $steps.Keys | Sort-Object
    if ($keys.Count -eq 0) { continue }
    $tailKeys = $keys | Select-Object -Last 5
    $ms = ($tailKeys | ForEach-Object { $steps[$_].StepMs } | Measure-Object -Average).Average
    "{0}: gaussians_last={1} step_ms_mean_last5={2:N3}" -f `
        $pair[0], $steps[$keys[-1]].Gaussians, $ms
    Get-Content $pair[1] |
        Select-String -Pattern 'average_psnr|splat_model=' |
        Select-Object -Last 2 |
        ForEach-Object { "  " + ($_.Line -replace '^.*\[info\] ', '') }
}
