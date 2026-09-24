# Photara

Photara is a C++20 photogrammetry and 3D Gaussian Splatting system for turning
image sequences or video into calibrated cameras, sparse and dense geometry,
Gaussian scenes, meshes, and textured assets. The repository contains both the
`photara` command-line pipeline and **Photara Studio**, a Vulkan and Dear ImGui
desktop editor.

The project version has a single source of truth: [`VERSION`](VERSION). CMake,
the command-line application, Photara Studio, and the Python module all read the
same value. The current version is **0.3.0**.

## What the code implements

Photara is organized as a set of independently usable reconstruction stages:

```text
images or video
    │
    ├─ optional SAM 3 masks
    ▼
feature extraction and matching
    ▼
Structure from Motion (SfM)
    ├─ sparse scene / camera export
    ├─ Multi-View Stereo (MVS) ──► dense cloud ──► Delaunay or TSDF mesh
    └─ CUDA 3D Gaussian training ──► Gaussian model ──► TSDF or PAM mesh
                                                       ▼
                                             UV unwrap and texture bake
```

The implementation currently includes:

- global, incremental, and hierarchical SfM;
- pinhole, radial/tangential, OpenCV fisheye, and equirectangular camera paths;
- Eigen/PoseLib multi-view geometry, including E/F/H estimation, PnP, and
  triangulation;
- robust bundle adjustment with CPU and CUDA backends;
- SIFT/RootSIFT and SiftGPU, plus optional DISK, SuperPoint, ALIKED, and
  LightGlue inference through ONNX Runtime;
- CPU and CUDA mutual-ratio feature matching, vocabulary-based retrieval, and
  a selective LightGlue rescue matcher;
- CUDA PatchMatch MVS with a CPU fallback and multi-view depth fusion;
- CGAL global Delaunay graph-cut meshing and sparse TSDF extraction;
- CUDA Gaussian Splatting with ADC+, ADC-IGS, and EMC densification,
  progressive-resolution training, multi-view geometry, NCC, and normal
  supervision;
- Gaussian-to-mesh extraction through median-depth TSDF or the
  GaussianWrapping-inspired PAM path;
- optional Instant Meshes remeshing, CGAL repair/decimation, UVAtlas unwrap,
  Vulkan texture baking, and ONNX-based intrinsic-image delighting;
- SAM 3 text-prompted masks and video mask propagation when the bundled SAM
  component and a compatible checkpoint are available;
- `.ascan` project archives and exports for OpenMVS, COLMAP, Nerfstudio, PLY,
  OBJ, GLB, SOG, and SPZ, depending on the selected stage.

OpenCV is intentionally not part of the default dependency graph. JPEG and PNG
use their native libraries, FreeImage handles the remaining supported image
formats, the built-in SIFT path uses the vendored VLFeat sources, and the core
geometry code uses Eigen, PoseLib, and Ceres.

## Repository layout

```text
.
├── apps/editor/                 Photara Studio
├── docs/                        Architecture and implementation notes
├── photara/
│   ├── include/                 Public C++ headers
│   ├── src/                     BA, features, SfM, MVS, splat, texture, and tools
│   ├── tests/                   Correctness and regression tests
│   └── third_party/             Small source dependencies such as VLFeat
├── python/                      nanobind module, utilities, and experiments
├── third_party/
│   ├── photara_drender/         Texture and mesh-processing submodule
│   ├── sam3/                    Native SAM implementation
│   └── splat_drender/           Differentiable Gaussian rasterizer
├── CMakeLists.txt
├── LICENSE
├── NOTICE
└── VERSION
```

The public CMake targets are `Photara::BA`, `Photara::Features`,
`Photara::SfM`, `Photara::MVS`, `Photara::Splat`, and `Photara::Texture` when
their corresponding components are enabled.

## Requirements

The build requires CMake 3.28 or newer and a C++20 compiler. The default
Windows configuration is designed for MSVC with vcpkg and expects the packages
requested by CMake, including Eigen3, Ceres, PoseLib, FreeImage, JPEG, PNG,
hnswlib, Boost, and GLFW for Photara Studio.

Optional or feature-specific requirements are:

- CUDA Toolkit 12 or newer for CUDA BA, CUDA MVS, and all splat training;
- Vulkan SDK 1.2 or newer for Photara Studio and texture baking;
- CGAL for global Delaunay meshing, PAM extraction, and optional mesh tools;
- ONNX Runtime for learned feature backends, LightGlue, and delighting;
- FFmpeg on `PATH`, or an explicit `--ffmpeg` executable, for video input;
- a DXC-capable Vulkan SDK when the TinyTensor Vulkan compute backend is used.

SiftGPU is enabled by default by the build configuration, but its upstream
license has non-commercial restrictions. Review that license before
distribution or commercial use. The intrinsic/delight model weights also have
their own licensing boundary; see
[`docs/LICENSE-Intrinsic.md`](docs/LICENSE-Intrinsic.md).

## Build

Initialize the texture/mesh-processing submodule first:

```powershell
git submodule update --init --recursive
```

A typical CUDA build with vcpkg is:

```powershell
cmake -S . -B build `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
  -DPHOTARA_ENABLE_CUDA=ON
cmake --build build --config Release --parallel
```

The default build produces:

```text
build/photara/Release/photara.exe
build/photara/Release/photara_studio.exe
```

Paths vary with the generator and platform. With a single-config generator,
the executables normally appear directly below `build/photara/`.

For a CPU-only CLI build, disable the CUDA-only splat component and the Studio
target, which currently links the splat library:

```powershell
cmake -S . -B build-cpu `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
  -DPHOTARA_ENABLE_CUDA=OFF `
  -DPHOTARA_ENABLE_SPLAT=OFF `
  -DPHOTARA_BUILD_STUDIO=OFF
cmake --build build-cpu --config Release --parallel
```

Enable tests and auxiliary tools only when needed; they are excluded from the
default build:

```powershell
cmake -S . -B build `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
  -DPHOTARA_BUILD_TESTS=ON `
  -DPHOTARA_BUILD_BENCHMARKS=ON
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

### Important CMake options

| Option | Default | Purpose |
|---|---:|---|
| `PHOTARA_BUILD_STUDIO` | `ON` | Build the Vulkan/Dear ImGui desktop editor |
| `PHOTARA_BUILD_TESTS` | `OFF` | Configure correctness and regression tests |
| `PHOTARA_BUILD_BENCHMARKS` | `OFF` | Configure auxiliary tools and benchmarks |
| `PHOTARA_BUILD_PYTHON` | `OFF` | Build the experimental nanobind module |
| `PHOTARA_ENABLE_CUDA` | `ON` | Enable CUDA BA and MVS acceleration |
| `PHOTARA_CUDA_ARCHITECTURES` | auto | CUDA targets; auto builds RTX 20/30/40 cubins, RTX 50 with CUDA 12.8+, and PTX fallback |
| `PHOTARA_ENABLE_SPLAT` | `ON` | Build CUDA Gaussian training and extraction |
| `PHOTARA_ENABLE_FEATURES` | `ON` | Build image features, SfM, MVS, and the CLI |
| `PHOTARA_ENABLE_SIFTGPU` | `ON` | Enable the optional SiftGPU adapter |
| `PHOTARA_ENABLE_SAM` | `ON` | Build in-process SAM 3 mask generation with ggml |
| `PHOTARA_ENABLE_SAM_VULKAN` | `ON` | Build the ggml Vulkan backend when a Vulkan SDK with `glslc` is available |
| `PHOTARA_ENABLE_ONNX` | `OFF` | Enable ONNX Runtime feature and delight models |
| `PHOTARA_FETCH_ONNX` | `ON` | Download ONNX Runtime when no local SDK is set |
| `PHOTARA_ONNXRUNTIME_ROOT` | empty | Path to a local ONNX Runtime SDK |
| `PHOTARA_ENABLE_TEXTURE` | `ON` | Build UV unwrap and Vulkan texture baking |
| `PHOTARA_ENABLE_MESH_TOOLS` | `ON` | Use CGAL repair and decimation from `photara_drender` |
| `PHOTARA_ENABLE_INSTANT_REMESH` | `ON` | Run Instant Meshes before CGAL decimation |
| `PHOTARA_ENABLE_VULKAN_COMPUTE` | `ON` | Build TinyTensor's Vulkan compute backend |
| `PHOTARA_ENABLE_ACCUTILE` | `ON` | Use opacity-aware SnugBox/AccuTile enumeration |
| `PHOTARA_ENABLE_NATIVE_ARCH` | `ON` | Optimize CPU code for the build host |
| `PHOTARA_DRENDER_ROOT` | auto | Override the `photara_drender` source directory |

The automatic CUDA release set contains native cubins for Turing (`sm_75`),
Ampere (`sm_86`), Ada (`sm_89`), and, with CUDA 12.8 or newer, Blackwell
(`sm_120`). It also embeds `compute_75` PTX for forward compatibility. Override
the complete set when producing a specialized build:

```powershell
cmake -S . -B build `
  -DPHOTARA_CUDA_ARCHITECTURES="75-real;86-real;89-real;120-real;75-virtual"
```

To use a local ONNX Runtime SDK instead of `FetchContent`:

```powershell
cmake -S . -B build `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
  -DPHOTARA_ENABLE_ONNX=ON `
  -DPHOTARA_FETCH_ONNX=OFF `
  -DPHOTARA_ONNXRUNTIME_ROOT="D:/sdk/onnxruntime-win-x64-gpu-1.20.1"
```

## Command-line usage

The CLI requires `--images` and `--output`. Input may be an image directory or
a video file. Run `photara --help` for the complete set of options.

### Sparse reconstruction

```powershell
.\build\photara\Release\photara.exe `
  --images D:\captures\object\images `
  --output object.ascan `
  --cache-dir cache
```

The default mapping mode is `global`, which is the recommended starting point
for turntable captures, 360-degree loops, and other sequences with loop
closure. `--mode incremental` and `--mode hierarchical` select the other
implemented mapping backends.

The camera model defaults to `auto`; it can be set explicitly to `pinhole`,
`opencv_fisheye`, or `equirectangular`. If `--focal` is omitted or zero,
Photara initializes it from the image dimensions and then refines calibration.
For already calibrated, zero-distortion pinhole images, use
`--trust-focal --focal <pixels>` to keep the supplied focal length fixed.

Photara writes an `*_sfm_diagnostics.csv` report for normal CLI runs. Check its
per-image reprojection errors, camera centers, pose steps, and alignment status
before interpreting failures in MVS or splat training as dense-stage problems.

### Video input

```powershell
.\build\photara\Release\photara.exe `
  --images D:\captures\object.mp4 `
  --video-fps 2 `
  --video-sharp-window 3 `
  --output object.ascan
```

The CLI extracts frames with FFmpeg. `--video-max-frames`, `--video-quality`,
`--video-scale`, `--video-rotate`, `--video-frames-dir`, and `--video-redo`
control frame generation.

### Dense MVS and mesh reconstruction

```powershell
.\build\photara\Release\photara.exe `
  --images D:\captures\object\images `
  --output object.ascan `
  --dense --mesh `
  --dense-quality default `
  --mesh-method auto
```

`--dense-quality` accepts `preview`, `default`, or `high`. The presets change
the complete MVS configuration rather than only image resolution.
`--dense-resolution-level` can override the preset's working scale.

For MVS-only reconstruction, `auto` selects the global CGAL Delaunay path.
TSDF remains available explicitly, and the splat mesh path uses median-depth
TSDF by default. The CPU PatchMatch scheduler can be tuned with
`--patchmatch-concurrent-views` and `--patchmatch-tile-rows`.

### 3D Gaussian Splatting

```powershell
.\build\photara\Release\photara.exe `
  --images D:\captures\object\images `
  --output object.sog `
  --splat `
  --splat-iterations 30000
```

Sparse SfM points initialize Gaussian training unless `--dense` or
`--dense-ply` supplies a dense initializer. The default densification strategy
is `adc_igs`; the CLI also exposes `adc_plus` and `emc`. Training supports SOG,
SPZ, GLB, and the native project workflow. Use `--mesh` to extract a surface
after training:

```powershell
.\build\photara\Release\photara.exe `
  --images D:\captures\object\images `
  --output object.ascan `
  --splat --mesh `
  --mesh-method tsdf
```

`--mesh-method pam` enables the learned-normal PAM path. It turns on normal
field training, constructs a tetrahedral seed surface, refines samples against
Gaussian occupancy, and performs a final CGAL Delaunay classification. PAM
therefore requires both splat support and a CGAL-enabled build.

### Texture baking

```powershell
.\build\photara\Release\photara.exe `
  --images D:\captures\object\images `
  --output object.ascan `
  --dense --mesh --texture `
  --atlas-resolution 2048
```

`--texture` implies mesh generation unless an editor working mesh is supplied.
It creates UVs and bakes projective color through `photara_drender`. Add
`--delight` to run the optional intrinsic-image model before baking; this
requires ONNX Runtime and the licensed `stage_0.onnx` through `stage_3.onnx`
weights in the configured intrinsic-model directory.

### Feature backends

Extractors and matchers are configured independently:

```powershell
# Default CUDA-oriented path
--extractor siftgpu --matcher gpu_mutual_ratio

# SuperPoint with LightGlue
--extractor superpoint `
--extractor-model D:\models\superpoint.onnx `
--matcher lightglue `
--lightglue-model D:\models\superpoint_lightglue_fused.onnx `
--max-features 2048

# SiftGPU with selective LightGlue rescue
--extractor siftgpu `
--matcher hybrid_lightglue `
--lightglue-model D:\models\sift_lightglue.onnx `
--hybrid-lightglue-max-features 2048
```

The fused `--pipeline lightglue_end2end` route is also available. Learned
extractors and LightGlue require a build with `PHOTARA_ENABLE_ONNX=ON` and
compatible model files.

## Outputs

The primary output extension selects the scene format:

| Extension | Meaning |
|---|---|
| `.ascan` | Photara project archive containing pipeline state and artifacts |
| `.asfm` | Compact Photara SfM scene |
| `.mvs` | OpenMVS Interface scene |
| `.ply` | Sparse point cloud unless a stage-specific working path is used |
| `.sog`, `.spz`, `.glb` | Trained Gaussian model |

Additional stages write sidecar artifacts next to the primary output. Common
names include `*_dense.ply`, `*_mesh.ply`, `*_mvs_mesh.ply`,
`*_splat_mesh.ply`, and `*_textured.obj/.mtl/_albedo.png`. A `.ply` primary
output does **not** implicitly create an OpenMVS file; pass `--export-mvs`
when both formats are required. `--export-colmap <directory>` writes a COLMAP
text model.

## Python experiments

The optional nanobind module exposes SfM, MVS, splat training, and TSDF as
separate calls:

```powershell
python -m pip install nanobind
cmake -S . -B build -DPHOTARA_BUILD_PYTHON=ON
cmake --build build --config Release --target photara_python
$env:PYTHONPATH = "$PWD\python"
```

See [`python/README.md`](python/README.md) for the API example and native-module
lookup rules.

## Documentation

| Document | Scope |
|---|---|
| [`docs/PIPELINE.md`](docs/PIPELINE.md) | Stage ordering, triggers, and artifacts |
| [`docs/SFM_ARCHITECTURE.md`](docs/SFM_ARCHITECTURE.md) | Front end, mapping modes, BA, cameras, and diagnostics |
| [`docs/MVS_ARCHITECTURE.md`](docs/MVS_ARCHITECTURE.md) | PatchMatch, fusion, meshing, cleanup, and quality presets |
| [`docs/DENSE_RECONSTRUCTION.md`](docs/DENSE_RECONSTRUCTION.md) | Dense, splat, mesh, and texture design |
| [`docs/SPLAT_CPP.md`](docs/SPLAT_CPP.md) | CUDA splat trainer and TinyTensor backend |
| [`docs/FEATURES_BACKENDS.md`](docs/FEATURES_BACKENDS.md) | Extractor/matcher combinations and compatibility |
| [`LICENSE`](LICENSE) | Apache-2.0 terms for Photara's own source |
| [`NOTICE`](NOTICE) | Copyright attribution required by Apache-2.0 |
| [`docs/LICENSE-Intrinsic.md`](docs/LICENSE-Intrinsic.md) | Delight model licensing and placement |

## Logging and validation

Set `PHOTARA_LOG_LEVEL` to `error`, `warning`, `info`, `debug`, `trace`, or
`off`. For development builds, enable `PHOTARA_BUILD_TESTS` and use CTest.
Benchmark and diagnostic executables are available behind
`PHOTARA_BUILD_BENCHMARKS` and are intentionally not part of the default build.

## License

Photara's own source in this repository is licensed under the Apache License,
Version 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

Third-party code, optional backends, and model weights keep their upstream
terms. Delight/Intrinsic weights are documented in
[`docs/LICENSE-Intrinsic.md`](docs/LICENSE-Intrinsic.md). Review every enabled
optional dependency before distribution.
