# Build and usage reference

Detailed build instructions, CLI examples, and project reference for [Photara](../README.md).

## Pipeline and implementation details

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
  OBJ, GLB, SOG, SPZ, and [USDC](splat-usdc.md), depending on the selected stage.
- native C++ [Streamed SOG LOD and voxel collision exports](splat-deployment.md),
  with Vulkan merge-cost evaluation and Gaussian voxelization.

OpenCV is intentionally not part of the default dependency graph. JPEG and PNG
use their native libraries, FreeImage handles the remaining supported image
formats, and the core geometry code uses Eigen, PoseLib, and Ceres.


## Studio demo camera shots

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

## Repository layout and CMake targets

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

### Linux

Studio uses GLFW, so the same target builds on Linux. Dependencies are a C++20
compiler, CMake 3.28 or newer, the Vulkan SDK (`dxc` included), GLFW, and the
same libraries the CLI already needs (Eigen, Ceres, FreeImage, and so on).
GLFW may come from the system (`libglfw3-dev`) or from vcpkg. File dialogs use
`zenity`, then `kdialog`. Live training preview needs a driver that can export
Vulkan opaque file descriptors; CUDA training also needs CUDA external memory
on that same GPU.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target photara_studio --parallel
```

Single-config generators write `build/photara/photara` and
`build/photara/photara_studio`. Editor preferences, language, and the cache
root are stored in `$XDG_CONFIG_HOME/Photara`, or `~/.config/Photara` when
`XDG_CONFIG_HOME` is unset.

### Windows packaging

The repository includes a repeatable Windows release script that gathers the
Studio executable, CLI worker, app-local DLLs, FFmpeg, and notices, then builds
both a portable ZIP and an Inno Setup installer. See
[`docs/PACKAGING_WINDOWS.md`](PACKAGING_WINDOWS.md) for prerequisites and
the release command.

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
| `PHOTARA_ENABLE_AGENT_CONTROL` | `ON` | Build the Studio control server and `photara-ctl` client |
| `PHOTARA_BUILD_TESTS` | `OFF` | Configure correctness and regression tests |
| `PHOTARA_BUILD_BENCHMARKS` | `OFF` | Configure auxiliary tools and benchmarks |
| `PHOTARA_BUILD_PYTHON` | `OFF` | Build the nanobind Python module (`import photara`) |
| `PHOTARA_ENABLE_CUDA` | `ON` | Enable CUDA BA and MVS acceleration |
| `PHOTARA_CUDA_ARCHITECTURES` | auto | CUDA targets; auto builds RTX 20/30/40 cubins, RTX 50 with CUDA 12.8+, and PTX fallback |
| `PHOTARA_ENABLE_SPLAT` | `ON` | Build CUDA Gaussian training and extraction |
| `PHOTARA_ENABLE_FEATURES` | `ON` | Build image features, SfM, MVS, and the CLI |
| `PHOTARA_ENABLE_SAM` | `ON` | Build in-process SAM 3 mask generation with ggml |
| `PHOTARA_ENABLE_INFERENCE` | `ON` | Build shared ggml inference support independently of SAM |
| `PHOTARA_INFERENCE_VULKAN` | `ON` | Build the shared ggml Vulkan backend when a Vulkan SDK with `glslc` is available |
| `PHOTARA_INFERENCE_CUDA` | `OFF` | Build the optional ggml CUDA backend for existing SAM usage |
| `PHOTARA_ENABLE_SAM_VULKAN` | `ON` | Legacy initial default for `PHOTARA_INFERENCE_VULKAN` |
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


## Agent control (experimental)

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
| `.sog`, `.spz`, `.glb` | Trained Gaussian model |

Additional stages write sidecar artifacts next to the primary output. Common
names include `*_dense.ply`, `*_mesh.ply`, `*_mvs_mesh.ply`,
`*_splat_mesh.ply`, and `*_textured.obj/.mtl/_albedo.png`. A `.ply` primary
output does **not** implicitly create an OpenMVS file; pass `--export-mvs`
when both formats are required. `--export-colmap <directory>` writes a COLMAP
text model.

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


## Documentation

| Document | Scope |
|---|---|
| [`docs/api/README.md`](api/README.md) | Library API documentation index and maintenance |
| [`docs/api/python.md`](api/python.md) | Python third-party integration and function reference |
| [`docs/api/cpp.md`](api/cpp.md) | C++ third-party integration and native API reference |
| [`docs/api/types.md`](api/types.md) | Shared bound configuration/result fields, defaults and enums |
| [`docs/PIPELINE.md`](PIPELINE.md) | Stage ordering, triggers, and artifacts |
| [`docs/SFM_ARCHITECTURE.md`](SFM_ARCHITECTURE.md) | Front end, mapping modes, BA, cameras, and diagnostics |
| [`docs/MVS_ARCHITECTURE.md`](MVS_ARCHITECTURE.md) | PatchMatch, fusion, meshing, cleanup, and quality presets |
| [`docs/DENSE_RECONSTRUCTION.md`](DENSE_RECONSTRUCTION.md) | Dense, splat, mesh, and texture design |
| [`docs/SPLAT_CPP.md`](SPLAT_CPP.md) | CUDA splat trainer and TinyTensor backend |
| [`docs/FEATURES_BACKENDS.md`](FEATURES_BACKENDS.md) | Extractor/matcher combinations and compatibility |
| [`LICENSE`](../LICENSE) | Apache-2.0 terms for Photara's own source |
| [`NOTICE`](../NOTICE) | Copyright attribution required by Apache-2.0 |
| [`docs/LICENSE-Intrinsic.md`](LICENSE-Intrinsic.md) | Delight model licensing and placement |

## Logging and validation

### SOG-XT Gaussian exports

Select **SOG-XT** in Studio's **Export Splat** panel to write a `.sog-xt`
file. The file can be reopened or dropped into Studio like `.sog` and `.spz`.
Studio also exports PLY, SOG, SPZ, GLB and USDC in a background task with a
progress display. PLY reports processed Gaussians, SOG reports encoded attribute
images, SPZ reports compressed streams, and disk writes report completed bytes.
Opaque preparation/encoding/copy stages show an activity animation rather than
a fixed percentage. Successful completion remains visible at 100%. Closing the dialog keeps
the export running. The C++ `save_gaussians_with_progress` API reports the same
stage completion events. Existing `save_gaussians` callers remain supported.

SOG-XT export preserves the original quantization, layout search rounds and
lossless WebP settings (`quality=100`, `method=6`, `exact=1`). The C++
`save_sog_xt` API accepts an optional stage progress callback through
`SogXtExportOptions`.
CPU layout, box filtering and PRAS use OpenMP when available. At most two WebP
images encode concurrently. Export logs report stage and per-image times
(`sog_xt_export`); WebP work overlaps SH clustering, so those durations are not
additive. Progress reports stage completion rather than an estimated finish time.

Export runs in the background. The control API accepts `format: "sog-xt"`;
when it returns `completed: false`, poll `studio.job.status` and inspect
`splat_export.running` and `splat_export.error` before using the output.
The reconstruction CLI recognizes `.sog-xt` output paths. The C++ API supports
`GaussianFormat::sog_xt` and extension-based detection; Python exposes
`GaussianFormat.SOG_XT` when the Python module is built.

This is Photara's native implementation of the image-coding ideas in
[KISS-GS](https://arxiv.org/abs/2608.26948), not the complete KISS-GS training
pipeline. It keeps the Gaussian count, performs coarse-to-fine local assignment
sorting of both position byte planes and SH codebooks, selects equivalent
rotation/scale representatives with PRAS, and quantizes the attributes before
lossless WebP encoding. Non-DC SH clustering uses the existing Vulkan backend;
non-Vulkan builds have a bounded-memory hierarchical two-means CPU path.
0-3 SH degrees, image-grid padding, and Mip2D metadata are supported. Bake
Mip3D before exporting, as for SOG. Loaded attributes use the existing renderer;
the smaller file does not imply smaller runtime GPU storage.

The `.sog-xt` container is a ZIP with `container_meta.yaml` and WebP attribute
images. Version 2 exports use the paper reference decoder's global position
range, scalar opacity range, and channel-major SH range ordering. Optional
native fields preserve SH degree, active count, and Mip2D mode; reference
decoders can ignore them. The reader accepts both JSON and ordinary block YAML,
old native v1 files, and unmarked reference-style WebP directories (including a
direct `container_meta.yaml` path). Missing degree defaults to 3, missing count
is inferred from the active mask, and missing quantization levels use the
reference's observed-image-range normalization. Positions and SH remain in
Photara's PLY coordinate frame. PNG/JPEG-XL containers are not supported.

The build now requires `yaml-cpp` (`libyaml-cpp-dev` on Debian/Ubuntu, or the
`yaml-cpp` vcpkg port). An independent NumPy decoder using the paper's reference
layout cross-checks exported degree-3 files against the C++ decoder. This tests
the reference conventions, not the authors' unreleased encoder. The CPU sorter
and hierarchical clustering remain independent implementations. POPSpa pruning
and encoding-aware fine-tuning are not performed by this export.

POPSpa compaction is a separate opt-in operation. Studio exposes a `POPSpa`
checkbox in training settings and in the model simplification panel. The latter
requires the project's registered SfM reconstruction and its source images;
with POPSpa disabled the existing training-free NanoGS merge remains available.
CLI training accepts `--splat-popspa=true --splat-popspa-keep-ratio 0.2`.
The default retains 20% of the trained input count, rounded to a whole primitive.
Speedy-Splat remains an independent training-time option.

The native implementation builds on KISS-GS Section 3.2: GaussianPOP RGB deletion
scores across training cameras, first pruning, 5,000 optimize-sparsify steps
with GaussianSpa auxiliary opacity/dual variables and effective-rank shape
regularization, a new scoring/pruning stage, and 5,000 fine-tuning steps. Held-out
cameras are excluded from both scoring and optimization. Scoring loads camera
metadata only. The first pass keeps up to twice the final target; this budget
split is an implementation choice because the paper does not specify it.
The GaussianSpa penalty is 0.0005 and its projection/penalty cadence is 50 steps;
projection keeps the largest auxiliary magnitudes with deterministic tie breaks.
Each pruning stage now uses four partial removals with fresh scores between
them, addressing GaussianPOP's batch-removal approximation. Use
`--splat-popspa-prune-cycles 1` for a single removal per stage. The existing
normalized minimum-eigenvalue shape term is retained; it differs from the
paper's physical minimum-scale term and remains an independent choice.
Scale/rotation learning rates keep their original multiplier of 1. The optional
`--splat-popspa-geometry-lr-scale` can reduce them, but the indoor iPhone sweep
did not support changing the default to 0.2: some close-ups improved while
aggregate quality declined.
Final fine-tuning drops the shape prior,
continues the terminal position rate, and scales the other Gaussian learning
rates by 0.2 (`--splat-popspa-finetune-lr-scale`). Compaction honors the selected
bilateral-grid and PPISP settings; these remain training-only corrections and
are excluded from the exported canonical RGB and SOG-XT decoder.
This is an independent implementation, not a claim to reproduce the authors'
rate-distortion results. The standalone C++ entry point is
`Trainer::compact_popspa(model, scene, progress)`.

Compaction does not densify or initialize from the point cloud. It uses the
existing RGB/SSIM optimizer with all SH bands active, no progressive resolution,
and a reduced position learning rate. The reconstruction checkpoint remains
intact; interrupted compaction restarts from that checkpoint rather than
resuming partially saved ADMM state. CLI stage lengths can be overridden with
`--splat-popspa-refine-steps` and `--splat-popspa-finetune-steps` for diagnostics.
SOG-XT encoding-aware adaptation is a separate CUDA-only opt-in stage. For a
`.sog-xt` output, pass `--splat-sog-xt-finetune=true`; the default is 4,000
optimizer steps with eight views accumulated per step. Override with
`--splat-sog-xt-finetune-steps` and `--splat-sog-xt-batch`. It preserves the
selected PPISP/bilateral-grid settings, excludes evaluation-split cameras,
freezes the image order and SH assignments, renders the quantized model, and
propagates STE gradients through the codec. PRAS runs at initialization and at
60%/80%, resetting covariance moments and updating only covariance ranges.
Export writes the trained assets directly without a second clustering pass.
`Trainer::finetune_sog_xt(model, scene, output, progress)` exposes the same
operation for an existing model; Python exposes `finetune_sog_xt` as well.

The native adaptation additionally uses a canonical input-render preservation
loss (weight 2) and conservative Gaussian learning-rate multiplier 0.02; these
are quality-oriented implementation choices, not the paper's exact training
recipe. CLI controls are `--splat-sog-xt-reference-weight` and
`--splat-sog-xt-lr-scale`. Twelve fixed **training** cameras select the best
canonical-MSE snapshot every 500 steps, including the unadapted baseline.
Held-out RGB does not participate in optimization or snapshot selection.
Cancellation returns the input model without writing a partially adapted file.
Plain export and Studio's existing export action remain training-free.

SH clustering now spends the exact requested codebook budget, including
non-power-of-two budgets, then performs three global Lloyd updates across all
initialization batches. The bounded GPU point batches share the same final
centers; no N-by-K distance matrix is allocated. Empty codewords are filled
from non-singleton clusters and their means are recomputed. Constant/repeated
SH vectors retain deterministic tie membership. This corrects the former
power-of-two truncation and batch-local-only optimization.

On the indoor iPhone POPSpa output (347,718 degree-3 Gaussians), the former
encoder used 30,837 codewords instead of the requested 43,264. The new encoder
uses all 43,264. A fixed-model sweep over all 739 registered cameras, at a
960-pixel maximum dimension with Mip2D rendering, measured:

| Stage | Mean PSNR | Mean SSIM | Compaction-holdout PSNR | SOG-XT size |
| --- | ---: | ---: | ---: | ---: |
| Input PLY | 23.6808 dB | 0.957596 | 23.1082 dB | — |
| Former encoder | 23.2682 dB | 0.952129 | 22.7835 dB | 5.797 MB |
| Exact budget + global Lloyd | 23.4161 dB | 0.953797 | 22.9260 dB | 6.135 MB |
| Encoding-aware adaptation | 23.5948 dB | 0.956209 | 23.0532 dB | 6.120 MB |

The final row uses 4,000 steps, eight views per step, both PPISP and the
bilateral grid, and selects step 4,000. All 93 compaction-holdout cameras are
excluded from adaptation and snapshot choice. Average encoding loss relative
to the same input PLY falls from 0.413 dB to 0.086 dB; views losing more than
0.5 dB decrease from 293 to 2, and the worst decrease improves from 1.822 dB to
0.561 dB. The final file grows about 5.6%. These figures isolate encoding;
they do not undo earlier POPSpa errors or establish a multi-scene guarantee.
The first adaptation experiment with higher rates and no input-render anchor
improved SSIM but reduced PSNR; it is not the selected recipe.

Measured on an RTX 5090 D v2, the former/new encoding passes took about
25.2/32.8 seconds, and the quality-preserving adaptation including codec setup,
selection and export took about 315 seconds. Evaluation renders are additional.
Use `photara_splat_popspa_review ... OUTPUT codec` for a fixed-model encode and
`photara_splat_popspa_review ... OUTPUT adapt 4000` for adaptation. Compare the
four sweeps with `scripts/analyze_sog_xt_refinement.py`.

Fixed training ranges can lose an observed code endpoint during adaptation.
The frozen export algebraically rebases codes and metadata ranges, without
introducing a new quantization, to match the paper decoder's observed-range
normalization. Native and independent NumPy reference decoding were compared
across all 347,718 rows and 59 PLY fields; the external-style YAML reader was
also checked. The decoder accepts explicit precisions up to the format's field
limit, including smaller effective code ranges from this rebasing.

A check on the supplied indoor iPhone scene, with 1,738,589 degree-3 Gaussians,
gave the following results (decimal MB, same count, no pruning):

| Format | Size | Encode | Decode | Mean PSNR vs original render |
| --- | ---: | ---: | ---: | ---: |
| PLY | 458.99 MB | 0.37 s | 0.41 s | Reference |
| SOG | 30.41 MB | 4.67 s | 0.67 s | 35.31 dB |
| SOG-XT | 26.17 MB | 74.13 s | 0.92 s | 35.22 dB |

Eight evenly spaced registered cameras from the neighboring `sfm.bin` were
rendered at an 800-pixel maximum dimension using the same CUDA raster settings.
PSNR measures deviation from the original PLY render, not fidelity to source
photographs. SOG-XT is 13.95% smaller, averaging 0.096 dB below SOG; its worst
view is 0.639 dB below SOG. This is one scene, not a universal rate-distortion
claim. Encoding is substantially slower and does not reduce GPU storage.
Do not apply the paper's full-pipeline compression factors to this export.
To repeat the measurement and retain files/screenshots after building tests:

```powershell
build/photara/Release/photara_splat_formats_test.exe --sog-xt-benchmark model.ply output-directory
python scripts/test_sog_xt_reference.py output-directory/benchmark.sog-xt output-directory/decoded-sog-xt.ply output-directory/reference-container
build/photara/Release/photara_splat_formats_test.exe --sog-xt-compare output-directory/reference-container output-directory/decoded-sog-xt.ply
```

The format tests also cover SH/position association after sorting, covariance
preservation before PRAS quantization, opacity, constant attributes, single-point
models, padding, ordinary reference-style YAML, and rejection of corrupt assets.
The CLI RPC fixture tests cover waiting for asynchronous exports, returning
their errors, and ignoring unrelated historical export failures when waiting
for a training operation.

### General validation

Set `PHOTARA_LOG_LEVEL` to `error`, `warning`, `info`, `debug`, `trace`, or
`off`. For development builds, enable `PHOTARA_BUILD_TESTS` and use CTest.
Benchmark and diagnostic executables are available behind
`PHOTARA_BUILD_BENCHMARKS` and are intentionally not part of the default build.


## License

Photara's own source in this repository is licensed under the Apache License,
Version 2.0. See [`LICENSE`](../LICENSE) and [`NOTICE`](../NOTICE).

Third-party code, optional backends, and model weights keep their upstream
terms. The intrinsic/delight model weights have their own licensing boundary;
see [`docs/LICENSE-Intrinsic.md`](LICENSE-Intrinsic.md). Review every
enabled optional dependency before distribution.
