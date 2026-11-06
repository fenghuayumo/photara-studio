param(
    [Parameter(Mandatory = $true)][string]$Model,
    [Parameter(Mandatory = $true)][string]$Sfm,
    [Parameter(Mandatory = $true)][string]$Images,
    [string]$Output = 'build/popspa-review',
    [string]$Executable = 'build/photara/Release/photara_splat_popspa_review.exe',
    [string[]]$Modes = @('baseline', 'prune20', 'popspa20', 'popspa40', 'unpruned', 'codec20', 'codec40', 'codecbase'),
    [uint32]$Steps = 5000
)
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Path $Output -Force | Out-Null
foreach ($mode in $Modes) {
    $destination = Join-Path $Output $mode
    if (Test-Path -LiteralPath (Join-Path $destination 'result.json')) {
        Write-Output "Completed result already exists: $mode"
        continue
    }
    $inputModel = $Model
    $runMode = $mode
    if ($mode -eq 'codecbase') { $runMode = 'codec' }
    if ($mode -eq 'codec20' -or $mode -eq 'codec40') {
        $sourceMode = if ($mode -eq 'codec20') { 'popspa20' } else { 'popspa40' }
        $inputModel = Join-Path (Join-Path $Output $sourceMode) 'model.ply'
        $runMode = 'codec'
    }
    if ($mode -eq 'codecpair') {
        $inputModel = Join-Path (Join-Path $Output 's3pair20') 'model.ply'
        $runMode = 'codec'
    }
    Write-Output "Starting $mode"
    & $Executable $inputModel $Sfm $Images $destination $runMode $Steps *> (Join-Path $Output "$mode.log")
    if ($LASTEXITCODE -ne 0) { throw "$mode failed; see $Output/$mode.log" }
    Write-Output "Completed $mode"
}
python scripts/analyze_popspa_review.py $Output
if ($LASTEXITCODE -ne 0) { throw 'Result analysis failed' }
