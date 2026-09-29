# Windows Packaging

`scripts/package_windows.ps1` turns an already-built Release directory into two
artifacts:

- `dist/Photara-Studio-<version>-win64.zip` - portable, unzip-and-run package;
- `dist/installers/Photara-Studio-<version>-win64-setup.exe` - Inno Setup
  installer.

The script does not blindly copy the build directory. It walks the PE import
tables of `photara_studio.exe` and `photara.exe` with `dumpbin.exe` and copies
exactly the DLLs that are actually needed (including transitive dependencies)
from the Release output, plus a bundled `ffmpeg.exe`. Test executables, PDB
files, static libraries, and intermediate build files never reach the package.
If `dumpbin.exe` is unavailable the script falls back to copying every DLL
found under the Release output directory.

## Prerequisites

1. Build the Release configuration. The default runtime directory is
   `build/photara/Release`.
2. Visual Studio Build Tools (for `dumpbin.exe`) or any environment where
   `dumpbin.exe` is on `PATH`. This is only needed for dependency-driven
   collection; without it the script uses the copy-all fallback.
3. [Inno Setup 6](https://jrsoftware.org/isinfo.php), or `ISCC.exe` on `PATH`.
   Required unless `-NoInstaller` is passed.
4. FFmpeg on `PATH`, or its full path passed explicitly. A shared (DLL-based)
   FFmpeg build works too: the script copies the DLLs sitting next to
   `ffmpeg.exe` automatically.

Target machines still need 64-bit Windows, a Vulkan-capable GPU driver, the
Microsoft Visual C++ 2015-2022 redistributable, and - for CUDA / 3DGS features
- a compatible NVIDIA driver. Graphics drivers, the Vulkan loader, and the
CUDA driver are intentionally not bundled.

## Creating a Release Package

From the repository root:

```powershell
.\scripts\package_windows.ps1
```

Repackaging the same version requires explicitly replacing the old artifacts:

```powershell
.\scripts\package_windows.ps1 -Clean
```

When FFmpeg or Inno Setup are not on `PATH`:

```powershell
.\scripts\package_windows.ps1 -Clean `
  -Ffmpeg 'D:\tools\ffmpeg\bin\ffmpeg.exe' `
  -InnoSetup 'C:\Program Files (x86)\Inno Setup 6\ISCC.exe'
```

To validate the portable package only, without compiling the installer:

```powershell
.\scripts\package_windows.ps1 -Clean -NoInstaller
```

## Models and Licensing

SAM 3 weights are never packaged; end users download them from Studio after
accepting Meta's model license. Intrinsic delight models are excluded by
default because their license is academic / non-commercial. Include them only
when redistribution is confirmed:

```powershell
.\scripts\package_windows.ps1 -Clean -IncludeIntrinsicModels
```
