# C++ API

Photara's native C++20 libraries can be linked into another program. Public
headers are under [`photara/include`](../../photara/include); namespaces are
`photara::sfm`, `photara::mvs`, `photara::splat`, `photara::ba`, and other modules.
Python wraps a subset of these APIs; it is not required by a native caller.

This guide covers the reconstruction APIs corresponding to the
[Python guide](python.md), plus entry points for other native components.
[Types and enumerations](types.md) lists bound fields and defaults with their
native type names and source links. C++ callers can additionally access members
and functions declared in the linked headers.

## Integrate using CMake

Use the source tree as a CMake subdirectory and link the component targets.
The main project currently does not install/export a complete SDK package with
`PhotaraConfig.cmake`; do not assume `find_package(Photara CONFIG REQUIRED)` is
available. Dependencies and feature switches are in the
[repository build guide](../../README.md#build).

For a consumer directory containing `CMakeLists.txt`, `main.cpp` and a checked-out
`photara-studio/` subdirectory:

```cmake
cmake_minimum_required(VERSION 3.28)
project(PhotaraConsumer LANGUAGES C CXX)

set(PHOTARA_BUILD_STUDIO OFF CACHE BOOL "" FORCE)
set(PHOTARA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(PHOTARA_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(PHOTARA_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
# For a CPU-only SfM/MVS consumer, also set these before add_subdirectory:
# set(PHOTARA_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
# set(PHOTARA_ENABLE_SPLAT OFF CACHE BOOL "" FORCE)
add_subdirectory(photara-studio)

add_executable(my_reconstruction main.cpp)
target_compile_features(my_reconstruction PRIVATE cxx_std_20)
target_link_libraries(my_reconstruction PRIVATE Photara::SfM Photara::MVS)
# For training/extraction, with the splat component enabled:
# target_link_libraries(my_reconstruction PRIVATE Photara::Splat)
```

Configure the consumer with the same dependency toolchain used to build Photara
(for example, `-DCMAKE_TOOLCHAIN_FILE=.../vcpkg.cmake`), then explicitly build
`my_reconstruction`. Initialize the source submodules first. Link targets carry
include directories, compile definitions and dependency requirements; copying
only an archive file or manually adding one include path is insufficient.
Deploy dependent DLLs/shared libraries alongside the consumer or on its runtime
search path. The library does not provide the editor's deployment bundle to a
consumer automatically.

| Target | Typical headers | Availability |
|---|---|---|
| `Photara::BA` | `ba/problem.hpp`, `ba/optimizer.hpp` | Core bundle adjustment |
| `Photara::Features` | `features/registry.hpp`, `features/features.hpp` | Feature component |
| `Photara::SfM` | `sfm/frontend.hpp`, `sfm/reconstruct.hpp`, `sfm/bundle.hpp` | Feature/SfM component |
| `Photara::MVS` | `mvs/densify.hpp`, `mvs/export.hpp` | Feature/MVS component |
| `Photara::Splat` | `splat/dataset.hpp`, `splat/trainer.hpp`, `splat/formats.hpp` | Splat enabled; current build requires CUDA |
| `Photara::Texture` | `texture/bake.hpp`, `texture/export.hpp` | Texture backend enabled and found |
| `Photara::Inference` | `inference/runtime.hpp`, `inference/geometry_predictor.hpp` | Shared inference enabled |
| `Photara::SAM` | `sam/masks.hpp` | SAM wrapper; use `built_with_sam()` to inspect implementation support |
| `Photara::SAMCache` | `sam/model_cache.hpp` | Model cache/licence helpers |

Use `if(TARGET Photara::Splat)` and similar checks for optional components.
No umbrella header or single monolithic `Photara::Photara` target is provided.

## Minimal SfM consumer

Save this as the consumer's `main.cpp`. Arguments are explicit input image
paths; the library has no direct equivalent of Python's directory-discovery
wrapper. Callers control ordering, including any sequence-neighbor semantics.

```cpp
#include "sfm/reconstruct.hpp"
#include "sfm/asfm.hpp"
#include "sfm/export_colmap.hpp"
#include <exception>
#include <filesystem>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: my_reconstruction image1 image2 [image3 ...]\n";
        return 2;
    }
    try {
        std::vector<std::filesystem::path> images;
        for (int i = 1; i < argc; ++i) images.emplace_back(argv[i]);
        photara::sfm::ReconstructionConfig options;
        options.mode = photara::sfm::ReconstructionMode::global;
        photara::sfm::Scene scene;
        const auto summary = photara::sfm::reconstruct(scene, images, options);
        if (!summary.valid) {
            std::cerr << "SfM did not produce a usable reconstruction\n";
            return 1;
        }
        std::filesystem::create_directories("output");
        photara::sfm::save_asfm(scene, "output/scene.asfm");
        photara::sfm::save_colmap_text(scene, "output/sparse");
        photara::sfm::save_sparse_ply(scene, "output/sparse.ply");
        std::cout << "Registered views: " << summary.registered_views << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
```

## SfM function reference

Names in this section are under `photara::sfm`. Configuration arguments with
`={}` use the public struct's default initialization.

| Function | Arguments, return and effects | Python counterpart |
|---|---|---|
| `reconstruct(scene_out, image_paths, config={})` | `Scene&`, `const vector<filesystem::path>&`, `const ReconstructionConfig&` → `ReconstructionSummary`. Populates output scene using selected mapping mode and final audits. | `run_sfm`; `run_sfm_directory` adds Python-only discovery |
| `run_frontend(image_paths, options={})` | Paths and `const FrontEndOptions&` → `FrontEndResult` containing `scene`, `timing`, `tracks_checkpoint_key`; extracts, matches, verifies and builds tracks. | `run_sfm_frontend` wraps scene/timing |
| `reconstruct_from_frontend(scene, config={})` | `Scene&`, config → `ReconstructionSummary`; maps and audits an existing frontend scene in place. | `run_sfm_mapping` additionally stores summary on handle |
| `run_global_mapping(scene, rotation={}, positioning={}, fallback_resection={})` | `Scene&`, corresponding options → `ReconstructionSummary`; global rotation/positioning, triangulation and BA. | No direct binding; selected via mapping mode |
| `estimate_global_rotations(scene, options={})` | `Scene&`, `const GlobalRotationOptions&` → `GlobalRotationSummary`; updates rotations using verified pairs. | Same name |
| `solve_global_positions(scene, options={})` | `Scene&`, `const GlobalPositioningOptions&` → `GlobalPositioningSummary`; updates centers/points from rotations and bearing constraints. | Same name |
| `run_bundle_adjustment(scene, options={})` | `Scene&`, `const BundleOptions&` → `BundleSummary`; updates selected poses/points and optionally shared intrinsics. | `bundle_adjust` |
| `set_bundle_backend_preference(preference)` | `BundleBackendPreference` → `void`; process-wide setting. | `set_bundle_backend` |
| `bundle_backend_preference()` | → `BundleBackendPreference`; requested policy, not last used backend. | `bundle_backend` |

Declarations: [reconstruct.hpp](../../photara/include/sfm/reconstruct.hpp),
[frontend.hpp](../../photara/include/sfm/frontend.hpp),
[global_rotation.hpp](../../photara/include/sfm/global_rotation.hpp),
[global_positioning.hpp](../../photara/include/sfm/global_positioning.hpp),
[bundle.hpp](../../photara/include/sfm/bundle.hpp).

`ReconstructionSummary.valid` means usable, not converged or metrically accurate.
Inspect registration, reliability, position-audit and reprojection fields.
`GlobalPositioningSummary` separates `success` from `termination`. BA returns
`BundleSummary.backend` and an `OptimizerSummary` with costs, timing and
termination; `brief_report()` returns a diagnostic string.

BA automatic policy prefers CUDA, then Vulkan, then CPU where applicable.
Explicit GPU policies can retain/fall back to CPU for unsupported parameter
layouts, problem sizes or unavailable implementations; they do not switch to
the other GPU backend. Positioning also has CPU paths for unsupported solve
sizes or camera-only initialization. Inspect actual results and diagnostics.

For independent stages:

```cpp
photara::sfm::ReconstructionConfig options;
auto frontend = photara::sfm::run_frontend(images, options.frontend);
auto summary = photara::sfm::reconstruct_from_frontend(frontend.scene, options);
```

The `Scene` is native data: cameras, images, pairs and tracks are accessible in
[scene.hpp](../../photara/include/sfm/scene.hpp). Maintain IDs, camera references
and track observations if editing it. `.asfm` stores a product subset, not all
matching/resection working state. Camera conventions and projection types are in
[sfm/types.hpp](../../photara/include/sfm/types.hpp) and
[core/camera_projection.hpp](../../photara/include/core/camera_projection.hpp).
Scene reconstruction has no automatic metric-scale guarantee; keep clouds,
camera poses and model data in the same world frame.

## SfM serialization and exports

All functions below use `photara::sfm`.

| Signature | Behavior |
|---|---|
| `save_asfm(const Scene&, const filesystem::path&, const AsfmOptions& = {}) -> void` | Writes native scene snapshot; `compact` and `path_base` control storage/path handling. |
| `load_asfm(const filesystem::path&, const AsfmOptions& = {}) -> Scene` | Reads native snapshot; resolves relative paths using the supplied base. Does not rerun mapping or construct a new complete summary. |
| `encode_asfm(const Scene&, const AsfmOptions& = {}) -> vector<uint8_t>` | In-memory snapshot bytes. |
| `decode_asfm(span<const uint8_t>, const AsfmOptions& = {}) -> Scene` | Reads snapshot bytes. |
| `save_sparse_ply(const Scene&, const filesystem::path&) -> void` | ASCII XYZRGB triangulated-track export. |
| `save_colmap_text(const Scene&, const filesystem::path& directory, const filesystem::path& image_path_base = {}, bool write_points = true) -> void` | Writes COLMAP text files; points file can be empty except header. |
| `export_openmvs_interface(const Scene&, const filesystem::path&, const ExportMvsOptions& = {}) -> void` | OpenMVS interface export of registered views/landmarks. Python uses default export options. |

Headers: [asfm.hpp](../../photara/include/sfm/asfm.hpp),
[export_colmap.hpp](../../photara/include/sfm/export_colmap.hpp),
[export_mvs.hpp](../../photara/include/sfm/export_mvs.hpp).

## MVS function reference

All functions use `photara::mvs`, declared in
[densify.hpp](../../photara/include/mvs/densify.hpp). Unlike the Python wrappers,
the native stage functions require an explicit `const DensifyOptions&` argument.

| Function | Return / effects | Python counterpart |
|---|---|---|
| `build_mvs_scene(const sfm::Scene&, const DensifyOptions&)` | New `MvsScene`; loads, undistorts and scales registered views. Does not fill a fused dense cloud. | Same name |
| `prepare_imported_scene(MvsScene&, const DensifyOptions&)` | `void`; prepares imported working resolutions, preserving poses/source calibration; discards previous maps. | No direct binding |
| `select_neighbors(MvsScene&, const DensifyOptions&)` | `void`; sets covisibility/baseline neighbors. | `mvs_select_neighbors` |
| `estimate_depth_maps(MvsScene&, const DensifyOptions&)` | `void`; estimates depth/normal/confidence. Default CUDA if available, otherwise CPU. | `mvs_estimate_depth_maps` |
| `fuse_depth_maps(MvsScene&, const DensifyOptions&)` | `void`; fills `dense_cloud`. | `mvs_fuse_depth_maps` |
| `reconstruct_mesh(MvsScene&, const DensifyOptions&)` | `void`; fills mesh from dense products. Method and dependency support matter. | `mvs_reconstruct_mesh` |
| `densify(MvsScene&, const DensifyOptions&)` | `void`; neighbors → depth → fusion → optional mesh. | No single in-place wrapper |
| `densify_from_sfm(const sfm::Scene&, const DensifyOptions&)` | New `MvsScene`; scene preparation and densification. | `run_mvs` |
| `apply_quality_preset(DensifyOptions&, DensifyQuality)` | `void`; modifies preset-controlled fields. Apply before your overrides. | `apply_mvs_quality_preset` |

Use the same options for scene construction and subsequent stages. To change
working resolution, prepare/rebuild scene images and maps rather than only
changing a field between stages.

```cpp
#include "mvs/densify.hpp"
#include "mvs/export.hpp"

// Within a function, with a reconstructed sfm_scene:
photara::mvs::DensifyOptions options;
photara::mvs::apply_quality_preset(
    options, photara::mvs::DensifyQuality::default_quality);
auto mvs_scene = photara::mvs::build_mvs_scene(sfm_scene, options);
photara::mvs::densify(mvs_scene, options);
photara::mvs::save_dense_ply(mvs_scene.dense_cloud, "output/dense.ply");
options.build_mesh = true;
photara::mvs::reconstruct_mesh(mvs_scene, options);
photara::mvs::save_mesh_ply(mvs_scene.mesh, "output/mesh.ply");
```

`MvsScene`, `MvsView`, `DenseCloud`, `DepthMap` and `Mesh` are native types in
[mvs/types.hpp](../../photara/include/mvs/types.hpp).
[mvs/export.hpp](../../photara/include/mvs/export.hpp) provides
`load_dense_ply(path) -> DenseCloud`, `load_mesh_ply(path) -> Mesh`,
`save_dense_ply(cloud, path)`, `save_mesh_ply(mesh, path)` and
`save_mesh_obj(mesh, path)`. Save functions return `void`; they select format by
function, not extension. View-indexed clouds must use the same ordered cameras
and coordinate frame. Binary mesh encode/decode and depth-map cache functions
are also declared there; `load_depth_map` returns `bool`, so inspect it.

## Dataset import and initialization

Under `photara::splat`, [dataset.hpp](../../photara/include/splat/dataset.hpp):

| Function | Contract |
|---|---|
| `load_splat_dataset(const DatasetLoadRequest&) -> DatasetLoadResult` | Imports COLMAP, RealityCapture or OpenMVS through default adapters. Request includes source, optional image root, initial PLY override, format, random-point count and seed. Result includes scene, detected format, resolved paths, generated/dense initialization flags and warnings. Python `load_dataset` retains only scene/warnings internally plus detected format. |
| `initialize_scene_from_sparse_points(mvs::MvsScene&) -> void` | Fills initialization cloud from sparse landmarks without marking input as dense. Call on an SfM-built MVS scene before sparse-initialized training. No direct Python binding. |
| `make_default_dataset_loader() -> DatasetLoader` | Returns loader with built-in adapters. Native callers may `register_reader(unique_ptr<DatasetReader>)` and call `load(request)` for custom adapters. |

```cpp
photara::splat::DatasetLoadRequest request;
request.source = "dataset/sparse/0";
request.image_directory = "dataset/images";
request.format = photara::splat::DatasetFormat::colmap;
auto loaded = photara::splat::load_splat_dataset(request);
// Inspect loaded.warnings and initialization flags before training.
```

When using SfM directly, `build_mvs_scene` creates sparse landmarks but does not
fill `dense_cloud`. The trainer requires that cloud to be nonempty. Either run
MVS fusion, assign a loaded cloud, or call the sparse-initialization helper.
Dataset import already finalizes initialization. It can generate random points
if source metadata has none; these are not measured geometry.

## Gaussian training and callbacks

Under `photara::splat`, [trainer.hpp](../../photara/include/splat/trainer.hpp):

```cpp
explicit Trainer(TrainingOptions options = {});
GaussianModel Trainer::train(
    const mvs::MvsScene& scene,
    ProgressCallback progress = {},
    EvaluationCallback evaluate = {},
    PreviewCallback preview = {},
    DevicePreviewCallback device_preview = {}) const;
```

The constructor takes an options value; configure it before constructing the
trainer. `train` is synchronous and returns an owning `GaussianModel`. Scene
must contain calibrated views and nonempty initialization points, unless a valid
resume path supplies the model. Enable depth/normal supervision only with the
required maps. There is no CPU training backend. Vulkan is a compiled option
with a restricted feature set; current native validation permits ADC-IGS
densification and rejects unsupported settings such as external MVS targets or
transparent-alpha supervision. Inspect backend errors rather than assuming all
CUDA training settings are portable.

```cpp
#include "splat/dataset.hpp"
#include "splat/trainer.hpp"
#include "splat/formats.hpp"

// For an SfM-built working scene that has not undergone dense fusion:
photara::splat::initialize_scene_from_sparse_points(mvs_scene);
photara::splat::TrainingOptions training;
training.backend = photara::splat::TrainingBackend::cuda;
training.densification_strategy = photara::splat::DensificationStrategy::emc;
photara::splat::apply_strategy_defaults(training);
training.iterations = 30'000; // Apply overrides after strategy defaults.
auto model = photara::splat::Trainer(training).train(
    mvs_scene, [](const photara::splat::TrainingProgress& stats) {
        return stats.iteration < 30'000; // false requests stopping.
    });
photara::splat::save_gaussians(model, "output/model.ply");
```

`apply_strategy_defaults(TrainingOptions&)` returns `void` natively; Python's
wrapper returns the modified options object. `TrainingOptions` fields and
initializers are listed in [the shared reference](types.md#trainingoptions).

| Callback | Signature and lifetime |
|---|---|
| `ProgressCallback` | `bool(const TrainingProgress&)`; return `false` to stop after this progress callback and finalize the current model. Copy values needed beyond the callback. |
| `EvaluationCallback` | `void(unsigned iteration, const GaussianModel&)`; do not retain the borrowed model reference beyond the callback. |
| `PreviewCallback` | `void(TrainingPreview)`; receives ownership of top-left-origin, interleaved 8-bit RGB plus dimensions and iteration. |
| `DevicePreviewCallback` | `void(unsigned, size_t view_index, const Camera&, const tinytensor::Tensor& color)`; planar float RGB `[3,H,W]`, valid only during the callback. |

Keep objects captured by callbacks alive for their use. Do not assume callbacks
run on the UI thread or that another thread can mutate the training scene.
Python only exposes the progress callback. A returned early-stop model is not
proof of completed training. Resume checkpoints and ordinary model exports have
different requirements; see [training resume](../training-resume.md).

## Model I/O and mesh extraction

All names below use `photara::splat`.

| Function | Return / effects | Python counterpart |
|---|---|---|
| `save_gaussians(model, path, format=GaussianFormat::auto_detect)` | `void`; writes PLY/SOG/SPZ/GLB/USDC subject to compiled support. | `GaussianModel.save` |
| `load_gaussians(path, format=GaussianFormat::auto_detect, device=tinytensor::Device::CUDA)` | Owning `GaussianModel`; format inferred from source when automatic. Native overload also accepts `(path, device)`. | `load_3dgs` uses default format/device |
| `extract_splat_mesh(model, scene, training_options={}, mesh_options={})` | `SplatMeshResult` with `mesh`, `surface_cloud`, `valid_depth_pixels`; options are `SplatMeshOptions`. | `extract_tsdf`, with forced TSDF/build-mesh settings |
| `extract_pam_mesh(model, scene, seed_mesh, training_options={}, pam_options={})` | `PamMeshResult` with seed/final mesh, candidate cloud and tetrahedron counts. Seed is `const mvs::Mesh&`; an empty mesh requests native pivot-seed generation. | `extract_pam`, requiring a `MeshResult` seed |
| `render_evaluation_png(model, view, path, options={})` | `RenderMetrics`; writes evaluation render and reports MAE, PSNR variants, SSIM and alpha metrics. | Not bound |

Headers: [formats.hpp](../../photara/include/splat/formats.hpp) and
[trainer.hpp](../../photara/include/splat/trainer.hpp). The native model uses
TinyTensor device storage; keeping many model copies/handles alive can retain
device memory. Use explicit device selection when loading if needed. Model
data, cameras, masks and extraction settings must describe the same scene.

Unlike Python's `extract_tsdf`, native `extract_splat_mesh` does not force TSDF:

```cpp
photara::splat::SplatMeshOptions mesh_options;
mesh_options.fusion.mesh_method = photara::mvs::MeshMethod::tsdf;
mesh_options.fusion.build_mesh = true;
mesh_options.fusion.mesh_tsdf_truncation_voxels = 6.F;
auto extracted = photara::splat::extract_splat_mesh(
    model, mvs_scene, training, mesh_options);
photara::mvs::save_mesh_ply(extracted.mesh, "output/tsdf.ply");
// Native callers may request automatic pivot-seed generation with an empty mesh:
// auto pam = photara::splat::extract_pam_mesh(model, mvs_scene, {}, training);
```

PAM needs native meshing dependencies such as CGAL. A PLY/OBJ mesh writer does
not infer Gaussian format. Gaussian exports are model interchange; they do not
embed all source images, camera metadata or resume state.

## Discovery equivalents

Python's `available_extractors`, `available_matchers` and
`available_pair_pipelines` call `photara::features::ensure_builtin_feature_backends()`
before `list_extractors()`, `list_matchers()` and `list_pair_pipelines()`.
Do the same in native consumers before querying/creating a backend. The registry
also provides factories and custom registrations in
[registry.hpp](../../photara/include/features/registry.hpp).

Python's `available_bundle_backends` combines build flags with
`ba::CudaOptimizer::is_available()` and `ba::VulkanOptimizer::is_available()`
where compiled. `compiled_training_backends` reports the compiled set; it is not
a live device/memory check. There is no single identically named native discovery
function for either list; account for component compile definitions in C++.

## Ownership, errors and concurrency

Native scene/result structs are ordinary C++ values. Functions taking mutable
references update the caller's object; functions taking const references borrow
it for the call. Keep referenced data alive, and respect tensor/callback-specific
lifetimes. Python handles wrap these values and need not have the same name or
return shape as native functions.

Invalid arguments can throw `std::invalid_argument`; backend, format and I/O
failures commonly throw `std::runtime_error` or filesystem exceptions. Check
returned success/validity flags and boolean cache-load results as well as
catching exceptions. An algorithm can return an unusable result without throwing.
Exceptions do not guarantee rollback of scenes or output files. Error strings
are diagnostics, not a stable error-code interface.

Configure process-wide BA preferences and registries before starting concurrent
work. No general guarantee is made here that callers may mutate/share the same
scene, trainer resources or GPU state across simultaneous calls. UI integration
should dispatch blocking computation appropriately and marshal owned preview
data back to the UI.

## Additional native interfaces

These entry points are not currently Python-bound:

| Module | Native entry points and documentation |
|---|---|
| Raw BA | `ba::Problem`, `optimize_cpu(problem, options={}) -> OptimizerSummary`, optional GPU optimizer classes; [problem.hpp](../../photara/include/ba/problem.hpp), [optimizer.hpp](../../photara/include/ba/optimizer.hpp). Populate observations and maintain index/layout invariants. |
| Texture | `texture::bake_mesh_texture(scene, options={}) -> TexturedMesh`, `bake_and_export(scene, output_stem, options={}) -> void`; [bake.hpp](../../photara/include/texture/bake.hpp), [options.hpp](../../photara/include/texture/options.hpp), [export.hpp](../../photara/include/texture/export.hpp). Requires a mesh and calibrated imagery; export writes OBJ/MTL/PNG. |
| SAM | `sam::built_with_sam() -> bool`, `generate_masks(const GenerateOptions&) -> GenerateResult`; [masks.hpp](../../photara/include/sam/masks.hpp). Supply existing model path, input images, prompt and output directory; white means keep. Throws on model/frame failures. |
| Shared inference | `inference::available_devices() -> vector<DeviceInfo>` and geometry predictor contracts; [runtime.hpp](../../photara/include/inference/runtime.hpp), [geometry_predictor.hpp](../../photara/include/inference/geometry_predictor.hpp), [architecture](../INFERENCE_ARCHITECTURE.md). Memory/device data are snapshots. |
| Project archives | `project::Archive` and stage/settings serialization; [archive.hpp](../../photara/include/project/archive.hpp), [document.hpp](../../photara/include/project/document.hpp). Keep archive format/version and stage consistency rules. |
| Renderer | `splat::Rasterizer`; [rasterizer.hpp](../../photara/include/splat/rasterizer.hpp), [splat implementation guide](../SPLAT_CPP.md). Tensor layouts and device support are part of the native contract. |
| Deployment | Native SOG LOD/collision exports; [deployment.hpp](../../photara/include/splat/deployment.hpp), [deployment guide](../splat-deployment.md). |

These links document additional native access; they do not imply every native
module is available in every build or has a Python equivalent.
