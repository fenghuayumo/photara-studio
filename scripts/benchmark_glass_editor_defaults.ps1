param(
    [string]$Dataset = 'D:/ScanVideo/glass',
    [ValidateSet('cuda', 'vulkan')][string]$Backend = 'vulkan',
    [string]$Name = 'glass_editor_default_probe',
    [int]$Iterations = 30000,
    [int]$StopAtIteration = 0
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root 'build/photara/Release/photara.exe'
$artifactDir = Join-Path $root 'artifacts'
New-Item -ItemType Directory -Force -Path $artifactDir | Out-Null
$stem = Join-Path $artifactDir $Name
$args = @(
    '--mode', 'global',
    '--images', (Join-Path $Dataset 'images'),
    '--splat-dataset', (Join-Path $Dataset 'sparse'),
    '--splat', '--backend', $Backend,
    '--splat-iterations', "$Iterations",
    '--splat-log-interval', '100',
    '--splat-strategy', 'adc_igs',
    '--splat-densification-cap', '1000000',
    '--splat-max-resolution', '1920',
    '--splat-progressive-resolution=true',
    '--splat-eval-split-every', '0',
    '--splat-use-mask=false',
    '--splat-cache-auto', '--splat-prefetch-views', '4',
    '--gui', '--mesh=false', '--output', "$stem.ply"
)
$trace = "$stem.gpu.csv"
Set-Content -LiteralPath $trace -Value 'timestamp,memory_mib,util_pct'
$process = Start-Process -FilePath $exe -ArgumentList $args -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput "$stem.log" -RedirectStandardError "$stem.err.log"
$stoppedAt = 0
try {
    do {
        $sample = & nvidia-smi --query-gpu=memory.used,utilization.gpu `
            --format=csv,noheader,nounits -i 0
        if ($LASTEXITCODE -eq 0 -and $sample) {
            Add-Content -LiteralPath $trace -Value "$(Get-Date -Format o),$sample"
        }
        Start-Sleep -Milliseconds 200
        $process.Refresh()
        if ($StopAtIteration -gt 0 -and -not $process.HasExited) {
            $recent = Get-Content -LiteralPath "$stem.log" -Tail 8 -ErrorAction SilentlyContinue
            foreach ($line in $recent) {
                if ($line -match 'splat iteration=(\d+)/') {
                    $iteration = [int]$Matches[1]
                    if ($iteration -ge $StopAtIteration) {
                        $stoppedAt = $iteration
                        Stop-Process -Id $process.Id
                        break
                    }
                }
            }
        }
    } while (-not $process.HasExited)
    $process.WaitForExit()
    if ($stoppedAt -gt 0) {
        Write-Output "stopped_at_iteration=$stoppedAt"
        exit 0
    }
    Write-Output "exit=$($process.ExitCode)"
    if ($process.ExitCode -ne 0) { exit $process.ExitCode }
} finally {
    if (-not $process.HasExited) { $process.WaitForExit() }
}
