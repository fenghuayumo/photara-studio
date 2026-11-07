$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pkg = Join-Path $root "photara_render"
$dest = Join-Path $root "photara_render-1.1.0.zip"

if (-not (Test-Path $pkg)) { throw "missing $pkg" }
if (Test-Path $dest) { Remove-Item -LiteralPath $dest -Force }

$tmp = Join-Path $env:TEMP ("photara_pack_" + [guid]::NewGuid().ToString("N"))
$tmp = [System.IO.Path]::GetFullPath($tmp)
$tempRoot = [System.IO.Path]::GetFullPath($env:TEMP).TrimEnd('\') + '\'
if (-not $tmp.StartsWith($tempRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
    (Split-Path -Leaf $tmp) -notmatch '^photara_pack_[a-f0-9]{32}$') {
    throw "Unsafe packaging directory: $tmp"
}
New-Item -ItemType Directory -Path $tmp | Out-Null
Copy-Item -LiteralPath $pkg -Destination $tmp -Recurse
Get-ChildItem -LiteralPath $tmp -Recurse -Directory -Filter "__pycache__" |
    ForEach-Object {
        $cachePath = [System.IO.Path]::GetFullPath($_.FullName)
        if (-not $cachePath.StartsWith($tmp + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Unsafe cache directory: $cachePath"
        }
        Remove-Item -LiteralPath $cachePath -Recurse -Force
    }

Compress-Archive -Path (Join-Path $tmp "photara_render") -DestinationPath $dest
Remove-Item -LiteralPath $tmp -Recurse -Force
Write-Host "打包完成: $dest"
