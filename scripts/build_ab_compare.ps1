param(
    [string]$VsRoot = "D:\ProgramTool\VS2022",
    [string]$VcpkgInclude = "D:\ProgramCode\vcpkg\installed\x64-windows\include",
    [ValidateSet("reference_compare", "backward_probe")]
    [string]$Target = "reference_compare",
    [switch]$Rebuild
)

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent

# Import the MSVC environment so nvcc can host-compile and link.
$vcvars = Join-Path $VsRoot "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found: $vcvars" }
cmd /c "`"$vcvars`" >nul && set" | ForEach-Object {
    if ($_ -match "^([^=]+)=(.*)$") {
        Set-Item -Path "env:$($Matches[1])" -Value $Matches[2]
    }
}

$src = @(
    "third_party\splat_drender\src\pipeline.cu",
    "third_party\splat_drender\src\render_forward.cu",
    "third_party\splat_drender\src\render_backward.cu",
    "third_party\splat_drender\src\point_sampling.cu",
    "aetherscan\third_party\gggs_reference\src\rasterizer_impl.cu",
    "aetherscan\third_party\gggs_reference\src\render_forward.cu",
    "aetherscan\third_party\gggs_reference\src\render_backward.cu",
    "aetherscan\third_party\gggs_reference\src\sample_forward.cu",
    "aetherscan\third_party\gggs_reference\src\sample_backward.cu",
    "third_party\splat_drender\tests\$Target.cu"
) | ForEach-Object { Join-Path $repo $_ }

New-Item -ItemType Directory -Force (Join-Path $repo "build") | Out-Null
$out = Join-Path $repo "build\$Target.exe"

Push-Location $repo
try {
    $flags = @('-O2', '-arch=native', '-std=c++20', '--extended-lambda',
        '--expt-relaxed-constexpr', '-use_fast_math', '-DNOMINMAX',
        '-I', (Join-Path $repo "third_party\splat_drender\include"),
        '-I', (Join-Path $repo "third_party\splat_drender\src"),
        '-I', (Join-Path $repo "aetherscan\third_party\gggs_reference\include"),
        '-I', $VcpkgInclude, '-DAETHERSCAN_GGGS_ACCUTILE=1', '-Xcompiler=/EHsc')
    # A conservative header timestamp invalidates every object after layout changes.
    $headerRoots = @('third_party\splat_drender', 'aetherscan\third_party\gggs_reference')
    $dependencyTime = (Get-Item -LiteralPath $PSCommandPath).LastWriteTimeUtc
    foreach ($root in $headerRoots) {
        Get-ChildItem -LiteralPath (Join-Path $repo $root) -Recurse -File |
            Where-Object { $_.Extension -in '.h', '.hpp', '.cuh' } |
            ForEach-Object { if ($_.LastWriteTimeUtc -gt $dependencyTime) { $dependencyTime = $_.LastWriteTimeUtc } }
    }
    $objectDir = Join-Path $repo 'build\ab_objects'
    New-Item -ItemType Directory -Force $objectDir | Out-Null
    $config = ($flags -join "`n") + "`n" + $VsRoot
    $configPath = Join-Path $objectDir 'flags.txt'
    if (!(Test-Path -LiteralPath $configPath) -or (Get-Content -Raw -LiteralPath $configPath) -ne $config) { $Rebuild = $true }
    $objects = @()
    foreach ($source in $src) {
        $relative = $source.Substring($repo.Length + 1)
        $object = Join-Path $objectDir (($relative -replace '[\\/.:]', '_') + '.obj')
        $objects += $object
        $objectInfo = Get-Item -LiteralPath $object -ErrorAction SilentlyContinue
        if ($Rebuild -or !$objectInfo -or $objectInfo.LastWriteTimeUtc -lt (Get-Item -LiteralPath $source).LastWriteTimeUtc -or $objectInfo.LastWriteTimeUtc -lt $dependencyTime) {
            & nvcc @flags -c $source -o $object
            if ($LASTEXITCODE -ne 0) { throw "nvcc compile failed: $source" }
        }
    }
    & nvcc @flags $objects -o $out
    if ($LASTEXITCODE -ne 0) { throw "nvcc link failed with $LASTEXITCODE" }
    [System.IO.File]::WriteAllText($configPath, $config)

} finally {
    Pop-Location
}
Write-Host "built $out"
