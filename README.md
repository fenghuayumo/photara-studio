<h1 align="center">Photara</h1>

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
  <a href="https://github.com/fenghuayumo/photara-studio/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/fenghuayumo/photara-studio?display_name=tag&sort=semver"></a>
  <img alt="Platform" src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux-informational">
</p>

**Photara** is an open-source C++20 photogrammetry and 3D Gaussian Splatting
system with CUDA and Vulkan training backends. Built for fast training on
Windows and Linux, it turns photographs or video into Gaussian scenes, meshes,
and textured assets through **Photara Studio** or the `photara` CLI.

<p align="center">
  <a href="https://github.com/fenghuayumo/photara-studio/releases/latest"><b>Download latest release</b></a>
  · <a href="https://github.com/fenghuayumo/photara-studio/releases">All releases</a>
  · <a href="https://github.com/fenghuayumo/photara-studio/issues">Report an issue</a>
</p>

<p align="center">
  <img src="docs/media/teaser-preview.gif" alt="Photara Studio training preview">
</p>

## Features

- **CUDA and Vulkan training** — choose a GPU backend for 3D Gaussian Splatting;
  train and preview scenes in Photara Studio.
- **Windows and Linux** — build and train on both platforms with the desktop
  editor or the command-line pipeline.
- **Designed for fast training** — GPU-accelerated rasterization and optimized
  training pipelines reduce work per iteration. Performance depends on hardware,
  scene, and settings.
- **From images to 3D assets** — reconstruct cameras and geometry with SfM and
  MVS, train Gaussians, extract meshes, and bake textures.
- **Flexible workflows** — use SAM 3 masks, optional learned feature matching,
  the CLI and Python API, and exports including COLMAP, PLY, OBJ, GLB, SOG,
  SPZ, and USDC.

See the [training backend documentation](docs/SPLAT_CPP.md) for supported
features and backend-specific limitations.

## Get started

Download the latest [Windows installer or portable build](https://github.com/fenghuayumo/photara-studio/releases/latest),
or [build from source](docs/BUILD_AND_USAGE.md#build). Source builds require
CMake 3.28+, a C++20 compiler, and the dependencies listed in the
[build guide](docs/BUILD_AND_USAGE.md#requirements). CUDA and Vulkan are needed
for Studio and GPU features.

For a Windows CUDA build with vcpkg:

```powershell
git submodule update --init --recursive
cmake -S . -B build `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
  -DPHOTARA_ENABLE_CUDA=ON
cmake --build build --config Release --parallel
```

Run `build/photara/Release/photara_studio.exe` to open Studio, or use the CLI:

```powershell
.\build\photara\Release\photara.exe `
  --images D:\captures\object\images `
  --output object.ascan
```

For Linux, other build configurations, and more CLI examples, see the
[build and usage reference](docs/BUILD_AND_USAGE.md).

## Documentation

- [Build and usage reference](docs/BUILD_AND_USAGE.md) — requirements, configuration, CLI examples, and output formats.
- [Pipeline overview](docs/PIPELINE.md) — reconstruction stages and artifacts.
- [Python API](docs/api/python.md) · [C++ API](docs/api/cpp.md) · [API index](docs/api/README.md).
- [Windows packaging](docs/PACKAGING_WINDOWS.md) · [Agent control](docs/AGENT_CONTROL.md).
- [SfM](docs/SFM_ARCHITECTURE.md) · [MVS](docs/MVS_ARCHITECTURE.md) · [Gaussian Splatting](docs/SPLAT_CPP.md) · [Feature backends](docs/FEATURES_BACKENDS.md).

## License

Photara is licensed under [Apache-2.0](LICENSE); see also [NOTICE](NOTICE).
Third-party components and model weights have their own terms, including the
[intrinsic/delight model license](docs/LICENSE-Intrinsic.md).
