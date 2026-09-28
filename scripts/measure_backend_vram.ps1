param(
    [string]$Exe = "build\photara\Release\photara.exe",
    [string]$OutDir = "artifacts\vram_backend_compare",
    [int]$Iterations = 400,
    [string]$Images = "D:\ScanVideo\glass\images",
    [string]$Dataset = "D:\ScanVideo\glass\sparse",
    [string[]]$Runs = @("cuda", "vulkan", "vulkan_pool64")
)

$ErrorActionPreference = "Stop"
$Runs = $Runs | ForEach-Object { $_ -split "," } | Where-Object { $_ -ne "" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Sample-MiB {
    $line = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
    return [int]($line | Select-Object -First 1)
}

foreach ($run in $Runs) {
    $backend = $run -replace "_pool.*", ""
    $baseline = Sample-MiB
    $dir = Join-Path $OutDir $run
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $log = Join-Path $dir "stdout.log"
    $args = @(
        "--images", $Images,
        "--output", (Join-Path $dir "splat.ply"),
        "--splat-dataset", $Dataset,
        "--splat", "--backend", $backend,
        "--splat-strategy", "adc_igs",
        "--splat-iterations", "$Iterations",
        "--splat-densification-cap", "1000000",
        "--splat-max-resolution", "1920",
        "--splat-progressive-resolution=false",
        "--splat-eval-split-every", "0"
    )
    $env:SPLAT_IGS_PROFILE = "1"
    if ($run -match "_pool(\d+)") {
        $env:TINYTENSOR_VULKAN_POOL_BYTES = [string]([int]$Matches[1] * 1MB)
    } else {
        Remove-Item Env:TINYTENSOR_VULKAN_POOL_BYTES -ErrorAction SilentlyContinue
    }
    $proc = Start-Process -FilePath $Exe -ArgumentList $args -PassThru `
        -WindowStyle Hidden -RedirectStandardOutput $log `
        -RedirectStandardError (Join-Path $dir "stderr.log")
    $samples = New-Object System.Collections.Generic.List[int]
    while (-not $proc.HasExited) {
        $samples.Add((Sample-MiB) - $baseline)
        Start-Sleep -Milliseconds 400
    }
    $proc.WaitForExit()
    $peak = ($samples | Measure-Object -Maximum).Maximum
    $median = ($samples | Sort-Object)[[int]($samples.Count / 2)]
    [pscustomobject]@{
        run = $run
        baseline_mib = $baseline
        peak_above_baseline_mib = $peak
        median_above_baseline_mib = $median
        samples = $samples.Count
        exit_code = $proc.ExitCode
    } | Format-Table -AutoSize
    $samples | Set-Content (Join-Path $dir "vram_trace_mib.txt")
}
