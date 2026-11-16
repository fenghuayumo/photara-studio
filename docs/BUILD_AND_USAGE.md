# Build and usage

Build Photara from source on Windows, Linux, or macOS, then run Photara Studio
or the reconstruction CLI. Run all commands from the repository root unless
stated otherwise.

- [Requirements](#requirements)
- [Build](#build): [Windows](#windows), [Linux](#linux), [macOS](#macos-vulkan-release), [CPU-only CLI](#cpu-only-cli)
- [Build options](#build-options)
- [Packaging](#packaging)
- [Studio usage](#studio-usage)
- [Command-line usage](#command-line-usage)
- [Outputs](#outputs) and [Python API](#python-api)
- [Logging and validation](#logging-and-validation)
- [Project reference](#project-reference) and [documentation](#documentation)


## Requirements

The build requires CMake 3.28 or newer and a C++20 compiler. The default
Windows configuration is designed for MSVC with vcpkg and expects the packages
requested by CMake, including Eigen3, Ceres, PoseLib, FreeImage, JPEG, PNG,
hnswlib, gflags, glog, Boost, and GLFW for Photara Studio. Gaussian model
I/O also requires nlohmann-json, yaml-cpp, WebP, and zstd. PoseLib is fetched
by CMake if no installed package is found.

Optional or feature-specific requirements are:

- CUDA Toolkit 12 or newer for CUDA BA, CUDA MVS, and the CUDA splat backend;
- Vulkan SDK 1.2 or newer for Photara Studio and texture baking;
- CGAL for global Delaunay meshing, PAM extraction, and optional mesh tools;
- ONNX Runtime for learned feature backends, LightGlue, and delighting;
- FFmpeg on `PATH`, or an explicit `--ffmpeg` executable, for video input;
- a DXC-capable Vulkan SDK when the TinyTensor Vulkan compute backend is used.

## Build

Initialize the submodules before configuring any platform:

```sh
git submodule update --init --recursive
```

Use a separate build directory for each toolchain or generator. The examples
below use `build/` independently on each platform.

### Windows

Use a CUDA-compatible MSVC toolchain, vcpkg, and the Vulkan SDK with DXC.
Set `VCPKG_ROOT` to the vcpkg checkout and run in PowerShell from the repository
root. The following uses the Visual Studio multi-config generator:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
  -DPHOTARA_ENABLE_CUDA=ON
cmake --build build --config Release --parallel
```

The default build produces:

```text
build/photara/Release/photara.exe
build/photara/Release/photara_studio.exe
```

Launch Studio or check the CLI:

```powershell
.\build\photara\Release\photara_studio.exe
.\build\photara\Release\photara.exe --help
```

### Linux

Install Ninja and CUDA Toolkit 12+ for the commands below.
For a Vulkan-only build, set `-DPHOTARA_ENABLE_CUDA=OFF`.

Studio uses GLFW, so the same target builds on Linux. Dependencies are a C++20
compiler, CMake 3.28 or newer, the Vulkan SDK (`dxc` included), GLFW, and the
same libraries the CLI already needs (Eigen, Ceres, FreeImage, and so on).
GLFW may come from the system (`libglfw3-dev`) or from vcpkg. File dialogs use
`zenity`, then `kdialog`. Live training preview needs a driver that can export
Vulkan opaque file descriptors; CUDA training also needs CUDA external memory
on that same GPU.

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPHOTARA_ENABLE_CUDA=ON
cmake --build build --target photara_studio --parallel
./build/photara/photara_studio
./build/photara/photara --help
```

Single-config generators write `build/photara/photara` and
`build/photara/photara_studio`. Editor preferences, language, and the cache
root are stored in `$XDG_CONFIG_HOME/Photara`, or `~/.config/Photara` when
`XDG_CONFIG_HOME` is unset.

### macOS Vulkan Release

Use an arm64 C++20 toolchain, CMake 3.28+, Ninja, Vulkan SDK (including DXC
and MoltenVK), Eigen, PoseLib (fetched by CMake), Ceres, gflags, glog,
FreeImage, JPEG, PNG, Boost, hnswlib headers, GLFW, nlohmann-json, yaml-cpp,
WebP, zstd, and optional CGAL.

From the repository root, set the SDK and dependency paths, then run:

```sh
export VULKAN_SDK="/path/to/vulkan-sdk/macOS"
export PHOTARA_DEPS_PREFIX="/path/to/dependency-prefix"
export PHOTARA_HNSWLIB_INCLUDE="/path/to/hnswlib"
scripts/build_macos_release.sh
```

`PHOTARA_DEPS_PREFIX` is the prefix containing dependency libraries and build
tools (for example, the Homebrew prefix). `PHOTARA_HNSWLIB_INCLUDE` must contain
the `hnswlib/` header directory. The script adds the SDK and dependency `bin/`
directories to `PATH`, disables CUDA, and selects the Vulkan Studio backend.
The default CPU target is Apple M2 (`-mcpu=apple-m2`); host-native tuning is
disabled for both Photara and GGML. DotProd, I8MM and BF16 optimizations remain
available, while SME and SVE are not enabled. The default minimum OS is macOS
27.0. `PHOTARA_BUILD_DIR` selects an independent build directory.
It builds Release only and creates a local
`build/Photara Studio.app` launcher with a macOS application icon generated from
`apps/icons/icon.png` and a local ad-hoc signature. Keep it alongside the build
folder: dependencies are not bundled for distribution.

All generated files live under `build/`. The main project builds its modules
through CMake targets in that directory. Independent runtime checks use
`build/checks/tinytensor-vulkan` and `build/checks/splat-vulkan`.

The macOS release script disables texture baking and USDC export and uses CPU
inference. Launch Studio with:

```sh
open "build/Photara Studio.app"
```

### CPU-only CLI

For a CPU-only CLI build, disable the splat component and the Studio
target, which currently links the splat library:

```powershell
cmake -S . -B build-cpu `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
  -DPHOTARA_ENABLE_CUDA=OFF `
  -DPHOTARA_ENABLE_SPLAT=OFF `
  -DPHOTARA_BUILD_STUDIO=OFF
cmake --build build-cpu --config Release --parallel
```

On Linux/macOS, use the same three `PHOTARA_*` switches with a separate
build directory and `-DCMAKE_BUILD_TYPE=Release`; omit the vcpkg toolchain
argument unless using vcpkg. These switches disable CUDA, splat and Studio;
Vulkan support is configured independently.

## Build options

| Option | Default | Purpose |
|---|---:|---|
| `PHOTARA_BUILD_STUDIO` | `ON` | Build the Vulkan/Dear ImGui desktop editor |
| `PHOTARA_ENABLE_AGENT_CONTROL` | `ON` | Build the Studio control server and `photara-ctl` client |
| `PHOTARA_BUILD_TESTS` | `OFF` | Configure correctness and regression tests |
| `PHOTARA_BUILD_BENCHMARKS` | `OFF` | Configure auxiliary tools and benchmarks |
| `PHOTARA_BUILD_PYTHON` | `OFF` | Build the nanobind Python module (`import photara`) |
| `PHOTARA_ENABLE_CUDA` | `OFF` on macOS, `ON` elsewhere | Enable CUDA BA and MVS acceleration |
| `PHOTARA_CUDA_ARCHITECTURES` | auto | CUDA targets; auto builds RTX 20/30/40 cubins, RTX 50 with CUDA 12.8+, and PTX fallback |
| `PHOTARA_ENABLE_SPLAT` | `ON` | Build Gaussian training and extraction with the enabled GPU backends |
| `PHOTARA_ENABLE_FEATURES` | `ON` | Build image features, SfM, MVS, and the CLI |
| `PHOTARA_ENABLE_SAM` | `ON` | Build in-process SAM 3 mask generation with ggml |
| `PHOTARA_ENABLE_INFERENCE` | `ON` | Build shared ggml inference support independently of SAM |
| `PHOTARA_INFERENCE_VULKAN` | `ON` | Build the shared ggml Vulkan backend when a Vulkan SDK with `glslc` is available |
| `PHOTARA_INFERENCE_CUDA` | `OFF` | Build the optional ggml CUDA backend for existing SAM usage |
| `PHOTARA_ENABLE_SAM_VULKAN` | `ON` | Legacy initial default for `PHOTARA_INFERENCE_VULKAN` |
| `PHOTARA_ENABLE_ONNX` | `OFF` | Enable ONNX Runtime feature and delight models |
| `PHOTARA_FETCH_ONNX` | `ON` | Download ONNX Runtime when no local SDK is set |
| `PHOTARA_ONNXRUNTIME_ROOT` | empty | Path to a local ONNX Runtime SDK |
| `PHOTARA_ENABLE_USDC` | `ON` | Enable native USDC Gaussian export with LightUSD |
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

## Packaging

Build Release binaries before creating distribution artifacts.

### Windows packaging

The repository includes a repeatable Windows release script that gathers the
Studio executable, CLI worker, app-local DLLs, FFmpeg, and notices, then builds
both a portable ZIP and an Inno Setup installer. See
[`docs/PACKAGING_WINDOWS.md`](PACKAGING_WINDOWS.md) for prerequisites and
the release command.

### macOS arm64 DMG

After `scripts/build_macos_release.sh`, create an independently relocatable
application and disk image (FFmpeg must be installed on the packaging machine).
Rebuild runtime dependencies first to avoid bottles requiring a newer OS:

```sh
MACOSX_DEPLOYMENT_TARGET=27.0 scripts/build_macos_runtime_deps.sh
python3 scripts/package_macos.py --sdk "$VULKAN_SDK" --ffmpeg "$(command -v ffmpeg)" \
  --deployment-target 27.0
```

Outputs are in `build/release/`: `Photara-Studio-<version>-macOS-arm64.dmg`,
its SHA-256 checksum, and a dependency manifest. The DMG includes an Applications
shortcut and installation instructions. The packager recursively copies all
non-system Mach-O dependencies, bundles MoltenVK and its driver manifest, rewrites
library references to bundle-relative paths, collects dependency license files,
and validates the final ad-hoc signature. Its native launcher selects the bundled
Vulkan driver and adds the bundled FFmpeg to PATH. Optional model weights remain
separate downloads. The CPU baseline is Apple M2 or later; packaging rejects a build still using
host-native CPU settings. The default deployment target is macOS 27.0, recorded in
the application and README. Packaging fails if any included arm64 binary requires
a newer OS. The launcher is also compiled for the selected target. OpenMP and
SVT-AV1 are rebuilt from checksum-verified upstream sources into
`build/macos-deps/prefix`; system Homebrew installations are left unchanged.
Use `--dependency-prefix` to select a different rebuilt dependency prefix.
Building for earlier macOS releases requires rebuilding the main application
and all dependencies for that deployment target, then testing on that release;
changing only the plist or Mach-O version declaration is insufficient.

This package does not require an Apple developer account and is not notarized.
Users may need to allow the downloaded app under System Settings → Privacy &
Security → Open Anyway. Validate releases on a clean target Mac before publishing.

## Studio usage

### Studio compute GPU selection

Open **Preferences > Compute > Compute GPU** to choose a physical GPU, or
**Automatic** to use Studio's default. The choice is saved per user by device
UUID, separately from project settings. Restart Studio to apply it; the bottom
status bar shows the device actually active in this session. Pending selections
are identified in Preferences. If a saved GPU is disconnected, Studio falls
back to the default and reports the missing device.

An explicit choice pins independently-created Vulkan compute devices, AI Vulkan
backends, CUDA visibility and child processes to the selected adapter. CUDA is
available only for NVIDIA adapters; other adapters use Vulkan. The viewport
uses the same physical adapter so training/preview shared resources remain
compatible. `--compute-gpu UUID` overrides the saved choice for one launch
(32 lowercase hex digits), including `--project-transition-smoke` runs.

### Demo camera shots

In **Camera Path**, open **Demo shots** (also available under **Path tools**).
Frame the subject and set its orbit target in the viewport, then choose:

- **Energy wave reveal**: a cyan luminous wave grows outwards from the orbit
  target as the camera makes a quarter orbit and gently pushes in.
- **Micro fly-through**: pulls back from a close particle view through Gaussian
  rings to the fully rendered scene.
- **Spiral ascent**: makes a full orbit while rising and widening the view.

Set duration (15 seconds by default), distance multiplier and clockwise direction,
then select **Append demo shot**. Each shot adds 17 editable perspective
keyframes after the existing path. Reveal shots require a loaded Gaussian model
and add a bounded effect clip without replacing earlier clips; spiral shots
retain existing effects. Use **Effect clip** to select a previous clip for editing.
Outside each shot's clip interval, ordinary rendering is preserved. New demo
shots use monotone cubic interpolation with continuous velocity; existing keys
can opt into **Cinematic** interpolation in the keyframe inspector or use
**Path tools > Use cinematic interpolation** to convert the path while retaining
hold/cut segments. For an older effect with global pre-roll, enable
**Limit effect to clip** in **Effect settings** and set its clip start.
The generated shots do not perform collision avoidance, so preview the path
before exporting, especially for interiors and close particle shots.

Use **Effect settings** to adjust wave radius, point density, highlight, growth
and timing. The wave centre follows the orbit target at the effect start time;
editing that target updates the reveal. Wave radius is a minimum reach: the wave
automatically expands to cover all geometry without making outer regions arrive
at the same time. Preview, still-image export and video
export share the same reconstruction animation and material settings. Interactive
reconstruction preview displays GPU textures directly, and point-cloud preview
samples up to 120,000 source points. Image/video export retains the 600,000-point
budget and full CPU bloom; preview retains the luminous material without that
additional bloom pass. Projects using bounded clips or cinematic interpolation
require settings reader version 32; existing projects remain readable.

### Agent control (experimental)

Photara Studio can expose a local semantic control API for coding agents such
as Codex, Claude Code, DeepSeek-based clients, and zcode. The feature is off
unless Studio is started explicitly with `--enable-agent-control`.

The build also produces `photara-ctl`, a small console client that discovers
the running Studio process. It is written beside `photara_studio.exe`; add that
folder to `PATH` or invoke it by full path:

```powershell
photara_studio.exe --enable-agent-control
photara-ctl doctor
photara-ctl state --json
photara-ctl schema --json
photara-ctl params set iterations=30000 strategy=adc_igs
photara-ctl job start train --json
photara-ctl job wait --timeout 30 --json
photara-ctl job wait --op op_1 --timeout 30 --json
photara-ctl view set yaw=1 pitch=0.3 distance=6 target=[0,0,0]
photara-ctl project save D:/data/project/result.ascan
photara-ctl export model --out D:/data/project/result.ply --format ply
photara-ctl capture --out D:/data/agent/view.png
```

The control endpoint binds only to `127.0.0.1`, uses a per-session bearer token
stored in the discovery file under `%APPDATA%\Photara\control`, and is disabled
by default. See [`docs/AGENT_CONTROL.md`](AGENT_CONTROL.md) for the
protocol, client integration paths, and security model.

Ready-to-copy Codex and Claude Code instructions are in
[`tools/photara_agent/`](../tools/photara_agent/). Any other client that can run
terminal commands can use the same `photara-ctl` commands.

## Command-line usage

The CLI requires `--images` and `--output`. Input may be an image directory or
a video file. Run `photara --help` for the complete set of options.

Examples below use Windows PowerShell and the Release executable. On
Linux/macOS, use `./build/photara/photara`, replace PowerShell continuation
backticks with `\`, and supply local input paths. For example:

```sh
./build/photara/photara \
  --images /path/to/captures/object/images \
  --output object.ascan \
  --cache-dir cache
```

### Sparse reconstruction

```powershell
.\build\photara\Release\photara.exe `
  --images D:\captures\object\images `
  --output object.ascan `
  --cache-dir cache
```

The default mapping mode is `global`, which is the recommended starting point
for turntable captures, 360-degree loops, and other sequences with loop
closure. `--mode hierarchical` selects clustered reconstruction with Sim(3)
submap alignment and merging.

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

### SfM-guided SAM 3 masks

```powershell
.\build\photara\Release\photara.exe `
  --images D:\captures\object\images `
  --output object.ascan `
  --sam-text "transparent glass cup" `
  --sam-backend vulkan
```

Mask generation is entirely native C++/ggml and does not require Python,
PyTorch, OpenCV, or NumPy. A first text/video pass creates coarse masks, but
camera alignment uses full-image features by default. Opt in with
`--sam-use-mask-for-sfm=true` to restrict alignment to those masks. In Studio,
enable **Generate SAM3 masks**, then use **Camera Alignment → Use SAM3 masks
for alignment** (off by default). This setting is saved with the project and
does not change downstream mask use. After SfM succeeds, the default
`--sam-sfm-guided=true` pass finds
the sparse points nearest the registered-camera centroid, projects them into
each view, and uses their location to select and clean the matching SAM 3
instance. Projected point pixels are guidance only; the final silhouette still
comes from SAM 3.

This central-point prior is intended for orbit/turntable object captures. For
a general scene or an off-centre subject, disable it with
`--sam-sfm-guided=false`. `--sam-sfm-point-fraction` (default `0.01`),
`--sam-min-area`, `--sam-max-area`, `--sam-guided-threshold` (default `0.2`),
and `--sam-close-kernel` expose the main selection and cleanup controls.
Existing mask directories are still preserved unless `--sam-refresh` is
requested.

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
It creates UVs and bakes projective color through `photara_drender`.

An existing mesh supplied with `--working-mesh` is checked for manifold topology
before UV unwrapping. CGAL repairs non-manifold edges and vertex fans without a
face reduction target; vertex duplication can preserve the original surfaces.
Baking preserves disconnected components and defaults to no simplification or
remeshing. Degenerate or duplicate faces may be removed during repair.
Explicitly pass `--mesh-target-faces N` to request simplification, and optionally
`--mesh-remesh=true` to run Instant Meshes first. `--mesh-target-faces 0` disables
simplification while retaining the manifold repair. The source PLY is preserved;
an explicitly processed baking mesh is written to `<output_stem>_bake_mesh.ply`.

Studio's Mesh panel has a separate **Mesh postprocessing** section, available
even when Build mesh is off so imported meshes can be prepared for baking.
It applies during mesh extraction and Bake Texture. Simplify mesh and Remesh
surface are independent switches and default off, including for older projects.
The postprocessing preset changes budgets without enabling either operation:

| Postprocessing quality | Simplification cap / remesh triangle budget | Remesh smoothing passes |
| --- | ---: | ---: |
| Fast | 500,000 | 0 |
| Standard | 1,000,000 | 1 |
| High | 2,000,000 | 2 |

Both face budgets and smoothing passes can be overridden and are saved in the
project independently of extraction and texture quality. Simplification only
reduces meshes above the cap. Remeshing changes geometry and may affect thin
structures; its target is approximate and it runs even when simplification is
off or the input is below the simplification cap. Enabling both operations runs
remeshing first, followed by CGAL manifold repair and optional decimation.

The CLI equivalent is `--mesh-postprocess-quality fast|standard|high`, which
opts into the preset simplification cap unless `--mesh-target-faces` overrides
it. Remeshing is enabled separately with `--mesh-remesh=true`.
`--mesh-remesh-target-faces` selects an independent triangle budget, and
`--mesh-remesh-smooth-iterations` accepts 0 through 10. For remesh-only work,
use `--mesh-target-faces 0 --mesh-remesh=true`. Without an explicit remesh
budget, the target follows the reduction cap or the input face count.

`--uv-parallel-partitions` defaults to 8. Increasing it adds spatial cuts and
can fragment charts; prefer the default when texture continuity matters.
`--uv-max-stretch` defaults to one third. Higher values allow more UV distortion
in exchange for fewer chart cuts; it does not change mesh geometry.
Large chart sets are packed concurrently into separate atlas tiles.
`--texture-fill-unobserved` extrapolates measured face colours along shared mesh
edges, including UV seams, before gutter padding. It preserves measured texels
and leaves disconnected components without observations empty. Filled regions
contain estimated colour rather than recovered photographic detail.

Blending defaults to sRGB so the atlas matches the photographs. Pass
`--texture-color-space linear` to blend in scene-linear light instead. Add
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
| `.sog`, `.sog-xt`, `.spz`, `.glb` | Trained Gaussian model |

Additional stages write sidecar artifacts next to the primary output. Common
names include `*_dense.ply`, `*_mesh.ply`, `*_mvs_mesh.ply`,
`*_splat_mesh.ply`, and `*_textured.obj/.mtl/_albedo.png`. A `.ply` primary
output does **not** implicitly create an OpenMVS file; pass `--export-mvs`
when both formats are required. `--export-colmap <directory>` writes a COLMAP
text model.

See [SOG and SOG-XT](SOG.md) for export/import behavior, container
compatibility, POPSpa compaction, encoding-aware adaptation, and measurements.

## Python API

The optional nanobind module is imported as `photara`. It exposes SfM, MVS,
splat training, and TSDF as separate calls:

```powershell
python -m pip install nanobind
cmake -S . -B build -DPHOTARA_BUILD_PYTHON=ON
cmake --build build --config Release --target photara_python
$env:PYTHONPATH = "$PWD\python"
```

See [`python/README.md`](../python/README.md) for the API example and native-module
lookup rules. The complete [Python API guide](api/python.md) documents
functions, result handles, callbacks and failures. Native consumers can use the
[C++ API guide](api/cpp.md) for CMake integration and corresponding calls.
Both share a generated [type and enum reference](api/types.md) with bound
fields, defaults and declaration links.

## Logging and validation

Set `PHOTARA_LOG_LEVEL` to `error`, `warning`, `info`, `debug`, `trace`, or
`off`. Tests and diagnostic tools are excluded from the default build even when
enabled. Configure tests, explicitly build the test targets, then run CTest:

```sh
cmake -S . -B build -DPHOTARA_BUILD_TESTS=ON
cmake --build build --config Release --target photara_tests --parallel
ctest --test-dir build -C Release --output-on-failure
```

Reuse the build directory configured for your platform. For single-config
builds, the configuration remains the `CMAKE_BUILD_TYPE` selected at configure
time. Some tests require a GPU, model weights, or fixtures; inspect test output
for those prerequisites.

Set `PHOTARA_BUILD_BENCHMARKS=ON` to configure auxiliary tools and benchmarks,
then build the required target explicitly. SOG-XT-specific checks are documented
in [SOG validation](SOG.md#validation).

## Project reference

### Pipeline and implementation details

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

- global and hierarchical SfM;
- pinhole, radial/tangential, OpenCV fisheye, and equirectangular camera paths;
- Eigen/PoseLib multi-view geometry, including E/F/H estimation, PnP, and
  triangulation;
- robust bundle adjustment with CPU and CUDA backends;
- SIFT/RootSIFT, including a native CUDA SIFT extractor, plus optional DISK, SuperPoint, ALIKED, and
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
- SAM 3 text-prompted masks, video propagation, and SfM sparse-point-guided
  instance selection when the bundled SAM component and a compatible
  checkpoint are available;
- `.ascan` project archives and exports for OpenMVS, COLMAP, Nerfstudio, PLY,
  OBJ, GLB, SOG, SPZ, and USDC, depending on the selected stage.
- native C++ [Streamed SOG LOD and voxel collision exports](../photara/include/splat/deployment.hpp),
  with Vulkan merge-cost evaluation and Gaussian voxelization.

OpenCV is intentionally not part of the default dependency graph. JPEG and PNG
use their native libraries, FreeImage handles the remaining supported image
formats, and the core geometry code uses Eigen, PoseLib, and Ceres.

### Repository layout and CMake targets

```text
.
├── apps/editor/                 Photara Studio
├── docs/                        Architecture and implementation notes
├── photara/
│   ├── include/                 Public C++ headers
│   ├── src/                     BA, features, SfM, MVS, splat, texture, and tools
│   ├── tests/                   Correctness and regression tests
│   └── third_party/             Small source dependencies
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
`Photara::SfM`, `Photara::MVS`, `Photara::Splat`, `Photara::Texture`,
`Photara::Inference`, `Photara::SAM`, and `Photara::SAMCache` when
their corresponding components are enabled.

Shared Vulkan/ggml inference and geometry-prior contracts are documented in
[`docs/INFERENCE_ARCHITECTURE.md`](INFERENCE_ARCHITECTURE.md).

## Documentation

| Document | Scope |
|---|---|
| [`docs/api/README.md`](api/README.md) | Library API documentation index and maintenance |
| [`docs/api/python.md`](api/python.md) | Python third-party integration and function reference |
| [`docs/api/cpp.md`](api/cpp.md) | C++ third-party integration and native API reference |
| [`docs/api/types.md`](api/types.md) | Shared bound configuration/result fields, defaults and enums |
| [`docs/SOG.md`](SOG.md) | SOG/SOG-XT exports, compatibility, compaction, adaptation, and validation |
| [`docs/PIPELINE.md`](PIPELINE.md) | Stage ordering, triggers, and artifacts |
| [`docs/SFM_ARCHITECTURE.md`](SFM_ARCHITECTURE.md) | Front end, mapping modes, BA, cameras, and diagnostics |
| [`docs/MVS_ARCHITECTURE.md`](MVS_ARCHITECTURE.md) | PatchMatch, fusion, meshing, cleanup, and quality presets |
| [`docs/DENSE_RECONSTRUCTION.md`](DENSE_RECONSTRUCTION.md) | Dense, splat, mesh, and texture design |
| [`docs/SPLAT_CPP.md`](SPLAT_CPP.md) | CUDA splat trainer and TinyTensor backend |
| [`docs/FEATURES_BACKENDS.md`](FEATURES_BACKENDS.md) | Extractor/matcher combinations and compatibility |
| [`LICENSE`](../LICENSE) | Apache-2.0 terms for Photara's own source |
| [`NOTICE`](../NOTICE) | Copyright attribution required by Apache-2.0 |
| [`docs/LICENSE-Intrinsic.md`](LICENSE-Intrinsic.md) | Delight model licensing and placement |

## License

Photara's own source in this repository is licensed under the Apache License,
Version 2.0. See [`LICENSE`](../LICENSE) and [`NOTICE`](../NOTICE).

Third-party code, optional backends, and model weights keep their upstream
terms. The intrinsic/delight model weights have their own licensing boundary;
see [`docs/LICENSE-Intrinsic.md`](LICENSE-Intrinsic.md). Review every
enabled optional dependency before distribution.
