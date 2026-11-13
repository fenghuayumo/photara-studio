# Python API

The native `photara` module provides SfM, MVS, Gaussian training, dataset import,
and mesh extraction. Configuration objects use writable attributes. Scenes and
models are opaque handles with statistics and export methods; raw camera,
feature and tensor arrays are not exposed by this binding.

See [types and enumerations](types.md) for the complete bound field inventory,
defaults and C++ names, and [C++ API](cpp.md) for native equivalents.

## Build and import

Use Python 3.9 or newer, CMake 3.28 or newer, a C++20 compiler and the native
dependencies in the [build guide](../../README.md#build). The current Python
target requires the splat component and CUDA build support, including for a
caller that only wants SfM. There is no separate CPU-only Python target.

From the repository root, using a configured dependency toolchain:

```powershell
python -m pip install nanobind
cmake -S . -B build -DPHOTARA_BUILD_PYTHON=ON -DPHOTARA_BUILD_STUDIO=OFF
cmake --build build --config Release --target photara_python
$env:PYTHONPATH = "$PWD\python"
python -c "import photara; print(photara.__version__)"
```

Visual Studio may require `--target photara/photara_python`; Ninja uses
`photara_python`. Use the same Python interpreter for configuration and import
(set CMake's `Python_EXECUTABLE` explicitly when multiple interpreters exist).

The source package loader checks `PHOTARA_NATIVE_DIR`, then
`build-codex-verify/photara/Release`, `build/photara/Release`, and `build/photara`.
For another build location, including a Linux directory:

```powershell
$env:PHOTARA_NATIVE_DIR = "D:\sdk-build\photara\Release"
```

On Windows, the loader adds the native directory and `$env:CUDA_PATH\bin` to
DLL search paths. Other dependent DLLs must also be discoverable. Import failure
raises `ImportError` with the paths attempted. This checkout uses a local native
build; it does not contain a wheel installation workflow.

## Conventions

All names below are in `photara`. `Path` means a filesystem path accepted by the
binding (`str` or `pathlib.Path`). Relative paths are relative to the process
working directory. Create output parent directories before saving.

In signatures below, `options=Type()` means an optional native default value.
Construct an explicit options object for each experiment, especially when using
presets. Enum members are case-sensitive; notably `GlobalPositioningTermination`
members are lowercase. See [enum mappings](types.md#enumerations).

Calls are synchronous. Functions with a native `gil_scoped_release` guard
release the Python GIL while computing. Training, compaction and SOG-XT adaptation
release it during native computation and reacquire it for Python callbacks.
This is not an asynchronous job API. Do not mutate a scene or its options
from another thread while a call is using them. Bundle backend preference is
process-wide; configure it before starting reconstruction.

## Minimal reconstruction

```python
from pathlib import Path
import photara as p

output = Path("output")
output.mkdir(parents=True, exist_ok=True)
options = p.SfmOptions()
options.mode = p.ReconstructionMode.GLOBAL
scene = p.run_sfm_directory("images", options)
if not scene.summary.valid:
    raise RuntimeError("SfM did not produce a usable reconstruction")
print(scene.registered_count, scene.summary.rms_reprojection_error_pixels)
scene.save_asfm(output / "scene.asfm")
scene.save_colmap(output / "sparse", image_path_base="images")
scene.save_sparse_ply(output / "sparse.ply")
```

`summary.valid` indicates usability, not a precision guarantee or convergence.
Inspect registration counts, reprojection errors, alignment reliability and
`position_unverified_views` for the capture. A freshly loaded ASFM does not
recompute a complete reconstruction summary; use handle counts and stored audit
counts rather than treating `summary.valid` as a load-success flag.

## Discovery and backend functions

| Function | Return and behavior |
|---|---|
| `available_extractors()` | `list[str]`; registers and lists compiled feature extractors |
| `available_matchers()` | `list[str]`; registers and lists compiled matchers |
| `available_pair_pipelines()` | `list[str]`; registers and lists compiled pair recipes |
| `available_bundle_backends()` | `list[str]`; CPU plus compiled GPU BA implementations whose availability checks succeed |
| `compiled_training_backends()` | `list[str]`; compiled training backends, not a runtime device or memory test |
| `set_bundle_backend(backend: BundleBackend)` | `None`; sets process-wide BA preference |
| `bundle_backend()` | `BundleBackend`; returns that preference, not necessarily the backend used by a solve |

BA automatic selection prefers GPUs where applicable. Explicit CUDA/Vulkan BA
preferences can still use CPU for unsupported problem layouts, small problems
or unavailable implementations; inspect `BundleSummary.backend`. A GPU preference
does not switch to the other GPU implementation. Training has no CPU backend;
requesting an unavailable training backend raises a native error.

## SfM functions

| Signature | Return / effects |
|---|---|
| `discover_images(directory: Path)` | `list[Path]`; sorted files in that directory, non-recursive. Extensions: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.bmp`, case-insensitive. Missing directory or no supported files raises `ValueError`. |
| `run_sfm(image_paths: list[Path], options=SfmOptions())` | New `SfmScene`; extraction, matching, mapping and final audits. Input order is caller-controlled. |
| `run_sfm_directory(image_directory: Path, options=SfmOptions())` | New `SfmScene`; `discover_images` followed by `run_sfm`. |
| `run_sfm_frontend(image_paths: list[Path], options=FrontEndOptions())` | New `SfmScene`; extraction, verified matches and tracks. Does not perform mapping; its reconstruction summary is not yet a mapping result. |
| `run_sfm_mapping(scene: SfmScene, options=SfmOptions())` | `SfmSummary`; mutates existing frontend scene and updates its stored `summary`. Frontend extraction settings do not rerun extraction here. |
| `estimate_global_rotations(scene: SfmScene, options=GlobalRotationOptions())` | `GlobalRotationSummary`; mutates scene rotations. Requires verified pair geometry. |
| `solve_global_positions(scene: SfmScene, options=GlobalPositioningOptions())` | `GlobalPositioningSummary`; mutates positions/points. Requires suitable rotations and tracks. |
| `bundle_adjust(scene: SfmScene, options=BundleOptions())` | `BundleSummary`; mutates poses, points and optionally intrinsics. Requires a populated reconstruction. |
| `load_sfm(path: Path, options=AsfmOptions())` | New `SfmScene` from native `.asfm`; not a COLMAP importer. |

Separate rotation, positioning and BA calls do not refresh the handle's complete
mapping summary. Inspect each returned summary. For the complete mapping and
audit sequence, use `run_sfm_mapping`:

```python
options = p.SfmOptions()
frontend = p.run_sfm_frontend(p.discover_images("images"), options.frontend)
summary = p.run_sfm_mapping(frontend, options)
if not summary.valid:
    raise RuntimeError("Mapping failed")
```

`SfmOptions` groups `frontend`, `star`, `resection`, `hierarchical`,
`global_rotation`, `global_positioning` and final audit thresholds. Model-backed
extractors/matchers need a build with their inference support and model paths.
Inspect the registry functions before selecting a recipe. Thresholds ending in
`_px`/`_pixels` use pixels; `_deg` uses degrees; timing fields state seconds or
milliseconds in their names. Full defaults are in [SfmOptions](types.md#sfmoptions)
and the nested type tables.

## MVS and dataset import

| Signature | Return / effects |
|---|---|
| `build_mvs_scene(sfm_scene: SfmScene, options=MvsOptions())` | New `MvsScene`; loads/undistorts/downscales registered views. Does not estimate depth maps. |
| `run_mvs(sfm_scene: SfmScene, options=MvsOptions())` | New `MvsScene`; build, neighbors, depth estimation, fusion and optional mesh (`options.build_mesh`). |
| `mvs_select_neighbors(scene: MvsScene, options=MvsOptions())` | `None`; mutates neighbor lists based on shared geometry and baseline. |
| `mvs_estimate_depth_maps(scene: MvsScene, options=MvsOptions())` | `None`; mutates depth/normal/confidence maps; call after neighbor selection. |
| `mvs_fuse_depth_maps(scene: MvsScene, options=MvsOptions())` | `None`; populates dense cloud from estimated maps. |
| `mvs_reconstruct_mesh(scene: MvsScene, options=MvsOptions())` | `None`; populates mesh from dense products. |
| `apply_mvs_quality_preset(options: MvsOptions, quality: DensifyQuality)` | `None`; modifies the supplied options. Apply before custom overrides. |
| `load_dataset(request: DatasetRequest)` | New `MvsScene`; imports external cameras/images/initial points. Inspect `warnings`. Does not train or run PatchMatch. |
| `initialize_scene_from_sparse_points(scene: MvsScene)` | `None`; fills an empty initialization cloud from SfM sparse landmarks and marks it sparse. An existing cloud and its initialization metadata are preserved. |

Use the same MVS options through each stage. Changing resolution after scene
construction does not recreate working images; rebuild the working scene.
Default PatchMatch uses CUDA when a suitable device is detected and otherwise
uses CPU. Meshing may require CGAL depending on the method/build.

```python
options = p.MvsOptions()
p.apply_mvs_quality_preset(options, p.DensifyQuality.DEFAULT)
mvs = p.build_mvs_scene(scene, options)
p.mvs_select_neighbors(mvs, options)
p.mvs_estimate_depth_maps(mvs, options)
p.mvs_fuse_depth_maps(mvs, options)
mvs.save_dense_cloud("output/dense.ply")
options.build_mesh = True
p.mvs_reconstruct_mesh(mvs, options)
mvs.save_mesh("output/mvs_mesh.ply")
```

Import an existing dataset without repeating SfM:

```python
request = p.DatasetRequest()
request.source = "dataset/sparse/0"
request.image_directory = "dataset/images"
request.format = p.DatasetFormat.COLMAP
# Optional PLY replacing source initial points:
# request.initial_point_cloud = "dataset/points.ply"
mvs = p.load_dataset(request)
print(mvs.warnings)
```

Supported adapters are COLMAP, RealityCapture, OpenMVS and Nerfstudio/Blender
transforms JSON. `source` is a dataset directory or an adapter-specific
CSV/MVS/JSON file; `AUTO` probes it. Empty
`image_directory` uses conventional locations. If metadata has no initial
points, the loader can generate `random_initial_point_count` points using `seed`.
The handle exposes `dataset_format`, `resolved_source`, `initial_point_cloud`,
`generated_initial_points`, `initial_points_dense` and `warnings`.

## Training and extraction

| Signature | Return / effects |
|---|---|
| `apply_strategy_defaults(options: TrainingOptions)` | Same `TrainingOptions` object; mutates strategy-dependent settings. Also available as `options.apply_strategy_defaults()`. |
| `training_options_for_scene(scene: MvsScene)` | New `TrainingOptions`; sets `input_is_dense` from the scene's initialization metadata and applies default ADC-IGS strategy settings. Apply custom strategy defaults and overrides afterward. |
| `train_3dgs(scene: MvsScene, options=None, progress=None, preview=None)` | New `GaussianModel`; `None` options use `training_options_for_scene(scene)`. Explicit options are copied and respected, including `input_is_dense`; they are not automatically rewritten. |
| `load_3dgs(path: Path, format=GaussianFormat.AUTO, device=Device.CUDA)` | New `GaussianModel`; choose inferred/explicit format and CPU, CUDA or Vulkan storage, subject to native reader/device support. CPU storage does not enable CPU training. |
| `compact_popspa(model: GaussianModel, scene: MvsScene, options=TrainingOptions(), progress=None, preview=None)` | New compacted `GaussianModel`; uses registered views and source images, preserving the input model. Configure `popspa_*` fields; no initialization or densification occurs. |
| `finetune_sog_xt(model: GaussianModel, scene: MvsScene, output: Path, options=TrainingOptions(), progress=None)` | New decoded `GaussianModel`; CUDA encoding-aware adaptation writes fixed-layout SOG-XT assets to `output`. Configure `sog_xt_*` fields. Plain SOG-XT export does not perform this adaptation. |
| `transform_gaussians(model: GaussianModel, transform: GaussianTransform)` | New model with positive uniform scale, Euler XYZ rotation in degrees and translation. Preserves the input model; place its cameras consistently before training/extraction/evaluation. |
| `merge_gaussians(models: list[GaussianModel])` | New concatenated model; requires a nonempty list of nonempty compatible models. SH bands are zero-padded to the widest input degree. |
| `restrict_sh_degree(model: GaussianModel, degree: int)` | `None`; truncates the model's SH bands in place, degree 0–3. A degree at least as high as its current degree has no effect. |
| `render_evaluation_png(model: GaussianModel, scene: MvsScene, view_index: int, path: Path, options=TrainingOptions())` | `RenderMetrics`; renders the zero-based scene view to PNG and reports MAE, PSNR variants, SSIM and alpha metrics. An out-of-range view raises `IndexError`. |
| `load_mesh(path: Path)` | New `MeshResult` containing a PLY mesh, usable as a PAM seed. |
| `extract_tsdf(model: GaussianModel, scene: MvsScene, training_options=TrainingOptions(), tsdf_options=TsdfOptions())` | New `MeshResult`; renders splat depth and fuses TSDF. Wrapper forces `fusion.mesh_method=TSDF` and `fusion.build_mesh=True` on its local options copy. |
| `extract_pam(model: GaussianModel, scene: MvsScene, seed_mesh=None, training_options=TrainingOptions(), pam_options=PamOptions())` | New `MeshResult`; `None` requests automatic native pivot-seed generation. Supply a mesh from `load_mesh` or extraction to use an explicit seed. Requires a CGAL-enabled build. |

Training requires calibrated views and a nonempty initialization cloud in the
native `MvsScene.dense_cloud` storage (that storage may hold sparse initial
points). `build_mvs_scene` alone does not fill it. Use MVS fusion as above,
`mvs.load_dense_cloud(path)`, or `load_dataset`, which prepares initial points.
For sparse-only training directly from SfM, call
`initialize_scene_from_sparse_points(mvs)` after `build_mvs_scene`.
`load_dataset` already initializes its cloud. Both paths avoid PatchMatch.
MVS fusion and `load_dense_cloud` mark the initialization dense; sparse/random
dataset initialization is marked sparse. Use `training_options_for_scene` when
constructing explicit options so sparse inputs enable ADC-IGS growth. A directly
constructed `TrainingOptions()` retains the native `input_is_dense=True` default.
If enabling MVS depth/normal supervision, estimate the corresponding maps.

Select the densification strategy, apply its defaults, then apply your overrides.
Extraction needs the matching scene coordinate system and relevant training
rendering settings. A saved Gaussian model alone does not replace its cameras,
source images, masks or checkpoint progress metadata. `render_method` selects
3DGS or 3DGUT; 3DGUT is for rendering and cannot be used for mesh extraction.
`rasterization`, `speedy_*` and all native `TrainingOptions` fields are exposed.
SOG-XT adaptation requires a nonempty CUDA 3DGS model with SH degree greater
than zero and Classic or Mip2D filtering; invalid combinations raise `ValueError`.

```python
training = p.training_options_for_scene(mvs)
training.backend = p.TrainingBackend.CUDA
training.densification_strategy = p.DensificationStrategy.EMC
training.apply_strategy_defaults()
training.iterations = 30_000

def progress(stats):
    print(stats.iteration, stats.loss, stats.gaussian_count)
    return True

model = p.train_3dgs(mvs, training, progress=progress)
model.save("output/model.ply")
tsdf = p.TsdfOptions()
tsdf.fusion.mesh_tsdf_truncation_voxels = 6.0
mesh = p.extract_tsdf(model, mvs, training, tsdf)
mesh.save("output/mesh.ply")
# Optional refinement; requires the native PAM dependencies:
# pam = p.extract_pam(model, mvs, mesh, training, p.PamOptions())
# pam.save("output/pam.ply")
```

Callback return `None` or `True` continues training; `False` requests stopping
after the current progress callback. The call returns the model accumulated so
far after native finalization. Keep callbacks short. Progress receives an owning
`TrainingProgress` copy that can be retained. `TrainingProgress` includes iteration, losses, growth/pruning,
resolution, timings and device memory counters; total VRAM of zero means the
probe failed. Callback exceptions propagate to the caller; they are not a
successful cancellation result.

The optional `preview(frame)` callback receives an owning `TrainingPreview`.
Set `options.preview_interval` to a nonzero cadence to enable it. `frame.rgb`
is immutable bytes with interleaved RGB8 pixels, top-left origin, of length
`frame.width * frame.height * 3`. Callbacks execute on the calling thread;
their progress/frame objects may be retained. No device tensors are exposed.

## Training checkpoint recovery

`inspect_training_checkpoint(path: Path)` returns `TrainingCheckpointInfo` or
`None` when no supported checkpoint exists. `training_checkpoint_in_use(path: Path)`
returns whether another trainer holds its lease. Invalid/corrupt checkpoint
data can raise a native exception. Info exposes `iteration`, `total_iterations`,
`gaussian_count`, `fingerprint` and the referenced `model_path`.

```python
training = p.training_options_for_scene(mvs)
training.iterations = 30_000
training.checkpoint_path = "output/training.checkpoint"
training.checkpoint_interval_iterations = 1_000
model = p.train_3dgs(mvs, training, progress=lambda stats: stats.iteration < 1_000)

info = p.inspect_training_checkpoint(training.checkpoint_path)
if info is None:
    raise RuntimeError("No checkpoint was saved")
training.resume_checkpoint = True
model = p.train_3dgs(mvs, training)
```

Recovery restores the saved model and iteration; the optimizer is reinitialized,
so it is not bit-exact optimizer-state continuation. Keep source images, the
manifest and its referenced model together, and reuse matching scene/options:
the native fingerprint rejects incompatible settings. The total iteration
budget is the original target, not an additional count. Native checkpoint
leases prevent concurrent writers. Checkpoints are published at the configured
iteration/time cadence, on a progress-requested stop, and on completion;
callback exceptions do not request a graceful checkpoint save. Optional
`checkpoint_control_file` uses the native control-file protocol.

For an existing trained model and dataset, see the runnable
[`tsdf_experiment.py`](../../python/examples/tsdf_experiment.py).

## Result handles

These handles are returned by functions; none has a bound Python constructor.
They own native data for their lifetime. Counts are read-only. They do not
provide NumPy arrays or unrestricted access to the C++ scene/tensors.

### SfmScene

Properties: `summary: SfmSummary`, `frontend_timing: FrontEndTiming`, and integer
`camera_count`, `image_count`, `registered_count`, `track_count`.
`summary` and `frontend_timing` are returned as values, not writable views.

| Method | Behavior (returns `None`) |
|---|---|
| `save_asfm(path, options=AsfmOptions())` | Native scene snapshot. `path_base` controls relative image paths; `compact` controls observation storage. Does not store all frontend descriptors/pairs/resection state. |
| `save_colmap(directory, image_path_base="", write_points=True)` | COLMAP **text** model (`cameras.txt`, `images.txt`, `points3D.txt`); base controls image names. `False` writes an empty points file with a header. |
| `save_sparse_ply(path)` | ASCII XYZRGB PLY of triangulated tracks. |
| `save_openmvs(path)` | OpenMVS interface export. |

### MvsScene

Integer properties: `view_count`, `dense_point_count`, `mesh_vertex_count`,
`mesh_face_count`; `warnings: list[str]` records dataset-loader warnings.
Read-only import metadata: `dataset_format: DatasetFormat`, `resolved_source: Path`,
`initial_point_cloud: Path`, `generated_initial_points: bool` and
`initial_points_dense: bool`. `initial_point_cloud` is empty when no cloud file
is used; fusion clears it. For scenes built directly from SfM the import format
remains `AUTO` and resolved source is empty.

| Method | Behavior (returns `None`) |
|---|---|
| `load_dense_cloud(path)` | Replaces current dense cloud with an ASCII or binary-little-endian PLY and marks initialization dense. Preserve the scene's world frame and camera ordering for view-indexed point data. |
| `save_dense_cloud(path)` | PLY with available normals/colors and view-support data. |
| `save_mesh(path)` | Writes PLY; changing the filename extension does not select OBJ/GLB. |

### GaussianModel

Properties: `size: int` and `sh_degree: int`.
`save(path, format=GaussianFormat.AUTO) -> None` exports the model. `AUTO` infers
the format from the filename; explicit formats are PLY, SOG, SOG_XT, SPZ, GLB and USDC.
Availability depends on the build (USDC uses optional LightUSD support).
Not every export is a reloadable training checkpoint; readers and writers may
support different formats. See [native format implementation](../../photara/src/splat/formats.cpp).

### MeshResult

Integer properties: `vertex_count`, `face_count`, `surface_point_count`,
`valid_depth_pixels`, `tetrahedron_count`, `occupied_tetrahedron_count`.
TSDF fills the depth-pixel count; PAM fills the tetrahedron counts and uses
its candidate cloud as `surface_point_count`. Unused counters remain zero.

`save(path) -> None` writes mesh PLY. `save_surface_cloud(path) -> None` writes
the surface/candidate cloud PLY. Neither infers a different format by extension.

### Solver summaries

`SfmSummary.valid`, `GlobalRotationSummary.success`,
`GlobalPositioningSummary.success` and `BundleSummary.success` describe outcomes;
they are not interchangeable. Positioning also reports `termination`, including
time/iteration limits and camera-only fallback. BA reports the actual backend
and `optimizer: OptimizerSummary`; `optimizer.brief_report() -> str` produces a
short diagnostic report. Inspect its termination and cost history where needed.
All bound summary fields are listed in [the type reference](types.md).

## Failure handling and limitations

Invalid arguments generally surface as `ValueError`; native runtime failures
generally surface as `RuntimeError`. Filesystem failures may also surface as
native runtime exceptions. Binding conversion errors surface as `TypeError`.
Error text is diagnostic, not a stable machine-readable status code.

```python
try:
    scene = p.run_sfm_directory("images", p.SfmOptions())
except (ValueError, RuntimeError) as error:
    raise RuntimeError(f"Reconstruction could not run: {error}") from error
if not scene.summary.valid:
    raise RuntimeError("Reconstruction ran but did not produce a usable result")
```

An exception does not promise rollback of a mutated scene or files already
written. Use separate output directories for independent runs. Python does not
currently bind texture baking, SAM mask generation, the generic inference API,
raw BA problems, renderer tensors, archive editing or device-preview callbacks.
The [C++ guide](cpp.md#additional-native-interfaces) links to these interfaces.
