# Packages the Release build into a portable directory and an Inno Setup
# installer. Runtime DLLs are collected automatically by walking the import
# tables of photara_studio.exe / photara.exe with dumpbin, so only binaries
# that are actually needed reach the package. A copy-all fallback keeps the
# script usable on machines without the VS developer tools.

[CmdletBinding()]
param(
    # CMake's Release output directory, not the CMake build root.
    [string]$RuntimeDir = "",
    [string]$OutputRoot = "",
    [string]$Version = "",
    # An explicit ffmpeg.exe takes precedence over PATH discovery.
    [string]$Ffmpeg = "",
    # An explicit ISCC.exe takes precedence over automatic discovery.
    [string]$InnoSetup = "",
    # Delighter weights are large and have a non-commercial license; opt in.
    [switch]$IncludeIntrinsicModels,
    # Creates only the portable directory and ZIP, without invoking Inno Setup.
    [switch]$NoInstaller,
    # Required to replace an existing directory for the same version.
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'

# $PSScriptRoot can be unavailable while parameter defaults are evaluated.
$scriptRoot = if ($PSScriptRoot) { $PSScriptRoot }
    else { Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $RuntimeDir) { $RuntimeDir = Join-Path $scriptRoot "..\build\photara\Release" }
if (-not $OutputRoot) { $OutputRoot = Join-Path $scriptRoot "..\dist" }

# DLLs provided by Windows, the GPU driver, or the VC++ redistributable. They
# are never bundled; README.txt documents the end-user requirements instead.
$script:SystemDlls = [System.Collections.Generic.HashSet[string]]::new(
    [StringComparer]::OrdinalIgnoreCase)
foreach ($name in @(
    'ntdll.dll', 'kernel32.dll', 'kernelbase.dll', 'ucrtbase.dll',
    'msvcrt.dll',
    'user32.dll', 'gdi32.dll', 'gdiplus.dll', 'shell32.dll', 'shlwapi.dll',
    'shcore.dll', 'advapi32.dll', 'ole32.dll', 'oleaut32.dll', 'oleacc.dll',
    'comctl32.dll', 'comdlg32.dll', 'uxtheme.dll', 'dwmapi.dll', 'imm32.dll',
    'ws2_32.dll', 'wsock32.dll', 'mswsock.dll', 'iphlpapi.dll', 'dnsapi.dll',
    'netapi32.dll', 'secur32.dll', 'sspicli.dll', 'crypt32.dll',
    'wintrust.dll', 'bcrypt.dll', 'bcryptprimitives.dll', 'ncrypt.dll',
    'userenv.dll', 'dbghelp.dll', 'psapi.dll', 'version.dll', 'winmm.dll',
    'winhttp.dll', 'wininet.dll', 'urlmon.dll', 'setupapi.dll',
    'cfgmgr32.dll', 'powrprof.dll', 'msimg32.dll', 'opengl32.dll',
    'glu32.dll', 'd3d11.dll', 'd3d12.dll', 'dxgi.dll', 'd3dcompiler_47.dll',
    'vulkan-1.dll', 'mfplat.dll', 'mfreadwrite.dll', 'mf.dll', 'mfuuid.dll',
    # VC++ 2015-2022 redistributable, provided by the machine.
    'vcruntime140.dll', 'vcruntime140_1.dll', 'vcruntime140d.dll',
    'vcruntime140_1d.dll', 'msvcp140.dll', 'msvcp140_1.dll', 'msvcp140_2.dll',
    'msvcp140d.dll', 'msvcp140d_1.dll', 'msvcp140d_2.dll', 'concrt140.dll',
    'vccorlib140.dll', 'vcomp140.dll', 'msvcp140_atomic_wait.dll',
    'msvcp140_codecs.dll',
    # NVIDIA driver / CUDA driver components.
    'nvcuda.dll', 'nvapi64.dll', 'nvml.dll'
)) { [void]$script:SystemDlls.Add($name) }

function Resolve-ExistingFile([string]$Path, [string]$Description) {
    if (-not $Path) { return $null }
    $item = Get-Item -LiteralPath $Path -ErrorAction Stop
    if ($item.PSIsContainer) { throw "$Description must be a file: $Path" }
    return $item.FullName
}

function Find-Ffmpeg([string]$ExplicitPath) {
    if ($ExplicitPath) { return Resolve-ExistingFile $ExplicitPath 'FFmpeg' }
    $command = Get-Command ffmpeg.exe -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($command) { return $command.Source }
    foreach ($candidate in @(
        'C:\ffmpeg\bin\ffmpeg.exe',
        'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
        'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe'
    )) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    }
    return $null
}

function Find-Iscc([string]$ExplicitPath) {
    if ($ExplicitPath) { return Resolve-ExistingFile $ExplicitPath 'Inno Setup compiler' }
    $command = Get-Command ISCC.exe -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($command) { return $command.Source }
    foreach ($candidate in @(
        'C:\Program Files (x86)\Inno Setup 6\ISCC.exe',
        'C:\Program Files\Inno Setup 6\ISCC.exe'
    )) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    }
    # Winget / custom installations may use any directory; ask the registry.
    $registryPaths = @(
        'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*',
        'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*',
        'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*'
    )
    $install = Get-ItemProperty $registryPaths -ErrorAction SilentlyContinue |
        Where-Object { $_.DisplayName -like 'Inno Setup*' -and $_.InstallLocation } |
        Select-Object -First 1
    if ($install) {
        $candidate = Join-Path $install.InstallLocation 'ISCC.exe'
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    }
    return $null
}

function Find-Dumpbin {
    $vswhere = Join-Path (${env:ProgramFiles(x86)}) `
        'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path -LiteralPath $vswhere -PathType Leaf) {
        $root = & $vswhere -latest -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath 2>$null | Select-Object -First 1
        if ($root) {
            $hit = Get-ChildItem -LiteralPath (Join-Path $root 'VC\Tools\MSVC') `
                -Directory -ErrorAction SilentlyContinue |
                Sort-Object Name -Descending |
                ForEach-Object { Join-Path $_.FullName 'bin\Hostx64\x64\dumpbin.exe' } |
                Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
                Select-Object -First 1
            if ($hit) { return $hit }
        }
    }
    $command = Get-Command dumpbin.exe -CommandType Application -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    return $null
}

function Find-CudaRuntimeDirs([string]$BuildRoot) {
    # CUDA runtimes (cudart, cublas, cusolver, curand, ...) live in the
    # toolkit installation, not in the CMake output directory.
    $dirs = New-Object 'System.Collections.Generic.HashSet[string]'
    $candidates = @()
    if ($env:CUDA_PATH) { $candidates += (Join-Path $env:CUDA_PATH 'bin') }
    $cache = Join-Path $BuildRoot 'CMakeCache.txt'
    if (Test-Path -LiteralPath $cache -PathType Leaf) {
        $match = Select-String -LiteralPath $cache -Pattern `
            '^CMAKE_CUDA_COMPILER:FILEPATH=(.+)$' | Select-Object -First 1
        if ($match) {
            $compiler = $match.Matches[0].Groups[1].Value
            if ($compiler) { $candidates += (Split-Path -Parent $compiler) }
        }
    }
    $candidates += Get-ChildItem `
        'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*' `
        -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        ForEach-Object { Join-Path $_.FullName 'bin' }
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Container)) {
            [void]$dirs.Add((Resolve-Path -LiteralPath $candidate).Path)
        }
    }
    return @($dirs)
}

function Get-BinaryImports([string]$Dumpbin, [string]$File) {
    $imports = @()
    $inside = $false
    foreach ($line in (& $Dumpbin /nologo /dependents "$File" 2>$null)) {
        if ($line -match 'Image has the following (delay load )?dependencies') {
            $inside = $true
            continue
        }
        if (-not $inside) { continue }
        $trimmed = $line.Trim()
        if (-not $trimmed) {
            if ($imports.Count -gt 0) { break }
            continue
        }
        if ($trimmed -match '\.dll$') { $imports += $trimmed.ToLowerInvariant() }
    }
    return , $imports
}

function Test-SystemImport([string]$Name) {
    if ($Name -match '^(api-ms-|ext-ms-)') { return $true }
    return $script:SystemDlls.Contains($Name)
}

function Copy-RuntimeDependencies([string]$Dumpbin, [string]$Stage, [string[]]$SearchDirs, [string[]]$Roots) {
    # Index every DLL in the search directories (Release output + CUDA
    # toolkit); prefer shallow copies when the same DLL name appears more
    # than once.
    $available = @{}
    foreach ($dir in $SearchDirs) {
        Get-ChildItem -LiteralPath $dir -Recurse -File -Filter '*.dll' |
            Sort-Object { $_.FullName.Length } |
            ForEach-Object {
                $key = $_.Name.ToLowerInvariant()
                if (-not $available.ContainsKey($key)) { $available[$key] = $_.FullName }
            }
    }

    $staged = [System.Collections.Generic.HashSet[string]]::new(
        [StringComparer]::OrdinalIgnoreCase)
    $unresolved = [System.Collections.Generic.HashSet[string]]::new(
        [StringComparer]::OrdinalIgnoreCase)
    $queue = [System.Collections.Queue]::new()
    foreach ($root in $Roots) { $queue.Enqueue((Get-Item -LiteralPath $root).FullName) }

    while ($queue.Count -gt 0) {
        $file = $queue.Dequeue()
        foreach ($import in (Get-BinaryImports $Dumpbin $file)) {
            if (Test-SystemImport $import) { continue }
            if ($staged.Contains($import)) { continue }
            if (-not $available.ContainsKey($import)) {
                [void]$unresolved.Add($import)
                continue
            }
            $source = $available[$import]
            Copy-Item -LiteralPath $source -Destination `
                (Join-Path $Stage (Split-Path -Leaf $source))
            [void]$staged.Add($import)
            $queue.Enqueue($source)
        }
    }

    if ($unresolved.Count -gt 0) {
        throw (
            "dumpbin reported imports that are not in the search directories " +
            "and not system DLLs: " + ($unresolved -join ', ') +
            ". If these are CUDA runtime DLLs, install the CUDA toolkit used " +
            "for the build or set CUDA_PATH.")
    }
    return $staged.Count
}

function Copy-AllDlls([string]$SourceDir, [string]$Stage) {
    $count = 0
    Get-ChildItem -LiteralPath $SourceDir -Recurse -File -Filter '*.dll' |
        Sort-Object { $_.FullName.Length } |
        ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination `
                (Join-Path $Stage $_.Name) -ErrorAction SilentlyContinue
            $count++
        }
    return $count
}

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$runtime = (Resolve-Path -LiteralPath $RuntimeDir).Path
foreach ($required in 'photara_studio.exe', 'photara.exe') {
    if (-not (Test-Path -LiteralPath (Join-Path $runtime $required) -PathType Leaf)) {
        throw "Release output is missing $required : $runtime"
    }
}

if (-not $Version) {
    $Version = (Get-Content -LiteralPath (Join-Path $repositoryRoot 'VERSION') -Raw).Trim()
}
if ($Version -notmatch '^\d+\.\d+\.\d+([-.][0-9A-Za-z.]+)?$') {
    throw "Version must be a release-like value (for example 0.4.0): $Version"
}

$output = [System.IO.Path]::GetFullPath($OutputRoot)
$stage = Join-Path $output "Photara-Studio-$Version-win64"
if (Test-Path -LiteralPath $stage) {
    if (-not $Clean) {
        throw "Staging directory already exists: $stage. Re-run with -Clean to replace it."
    }
    Remove-Item -LiteralPath $stage -Recurse -Force
}
New-Item -ItemType Directory -Path $stage -Force | Out-Null

Copy-Item -LiteralPath (Join-Path $runtime 'photara_studio.exe') -Destination $stage
Copy-Item -LiteralPath (Join-Path $runtime 'photara.exe') -Destination $stage

$dumpbin = Find-Dumpbin
if ($dumpbin) {
    $cudaDirs = Find-CudaRuntimeDirs (Split-Path -Parent $runtime)
    foreach ($cudaDir in $cudaDirs) {
        Write-Host "CUDA runtime directory: $cudaDir"
    }
    $searchDirs = @($runtime) + $cudaDirs
    $dllCount = Copy-RuntimeDependencies $dumpbin $stage $searchDirs @(
        (Join-Path $runtime 'photara_studio.exe'),
        (Join-Path $runtime 'photara.exe'))
    Write-Host "Collected $dllCount runtime DLL(s) via dumpbin dependency walk."
} else {
    Write-Warning 'dumpbin.exe not found; falling back to copying every DLL in the Release output.'
    $dllCount = Copy-AllDlls $runtime $stage
    Write-Host "Copied $dllCount runtime DLL(s) (copy-all fallback)."
}

foreach ($asset in 'icon.png') {
    $source = Join-Path $runtime $asset
    if (Test-Path -LiteralPath $source -PathType Leaf) { Copy-Item -LiteralPath $source -Destination $stage }
}

$ffmpeg = Find-Ffmpeg $Ffmpeg
if (-not $ffmpeg) {
    throw 'ffmpeg.exe was not found. Install FFmpeg, add it to PATH, or pass -Ffmpeg <path-to-ffmpeg.exe>.'
}
Copy-Item -LiteralPath $ffmpeg -Destination (Join-Path $stage 'ffmpeg.exe')
# Shared FFmpeg builds (gyan "shared", BtbN "shared", ...) need their DLLs.
$ffmpegFolder = Split-Path -Parent $ffmpeg
Get-ChildItem -LiteralPath $ffmpegFolder -File -Filter '*.dll' -ErrorAction SilentlyContinue |
    Copy-Item -Destination $stage

foreach ($document in 'LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md') {
    Copy-Item -LiteralPath (Join-Path $repositoryRoot $document) -Destination $stage
}
New-Item -ItemType Directory -Path (Join-Path $stage 'licenses') -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $repositoryRoot 'docs\LICENSE-Intrinsic.md') `
    -Destination (Join-Path $stage 'licenses\LICENSE-Intrinsic.md')

if ($IncludeIntrinsicModels) {
    $models = Join-Path $repositoryRoot 'Models\Intrinsic'
    if (-not (Test-Path -LiteralPath $models -PathType Container)) {
        throw "Intrinsic models were requested but not found: $models"
    }
    Copy-Item -LiteralPath $models -Destination (Join-Path $stage 'Models') -Recurse
}

$readme = @"
Photara Studio $Version (Windows x64)

Run photara_studio.exe. FFmpeg is bundled beside the application for video input.

Requirements: 64-bit Windows, a Vulkan-capable graphics driver, the Microsoft
Visual C++ 2015-2022 redistributable, and (for CUDA / 3DGS features) a
compatible NVIDIA driver. The Vulkan loader, graphics drivers, and the CUDA
driver are provided by the operating system / GPU vendor and are intentionally
not bundled.

SAM 3 model weights are not included. Download them from Studio only after accepting
Meta's model license. Intrinsic delight models are included only in packages built
with -IncludeIntrinsicModels; see licenses\LICENSE-Intrinsic.md before redistribution.

See LICENSE, NOTICE, and THIRD_PARTY_NOTICES.md for distribution terms.
"@
# WriteAllText keeps the file UTF-8 without BOM on both Windows PowerShell 5.1
# and PowerShell 7+.
[System.IO.File]::WriteAllText(
    (Join-Path $stage 'README.txt'),
    $readme,
    [System.Text.UTF8Encoding]::new($false))

$zip = Join-Path $output "Photara-Studio-$Version-win64.zip"
if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
Compress-Archive -LiteralPath $stage -DestinationPath $zip -CompressionLevel Optimal
Write-Host "Portable package: $zip"

if (-not $NoInstaller) {
    $iscc = Find-Iscc $InnoSetup
    if (-not $iscc) {
        throw 'Inno Setup 6 was not found. Install it, add ISCC.exe to PATH, pass -InnoSetup <path-to-ISCC.exe>, or use -NoInstaller.'
    }
    $installerOutput = Join-Path $output 'installers'
    New-Item -ItemType Directory -Path $installerOutput -Force | Out-Null
    $iss = Join-Path $repositoryRoot 'installer\PhotaraStudio.iss'
    & $iscc "/DMyAppVersion=$Version" `
        "/DMySourceDir=$stage" `
        "/DMyOutputDir=$installerOutput" `
        "/DMySetupIconFile=$(Join-Path $repositoryRoot 'apps\icons\icon.ico')" `
        $iss
    if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed with exit code $LASTEXITCODE." }
    Write-Host "Installer: $(Join-Path $installerOutput "Photara-Studio-$Version-win64-setup.exe")"
}
