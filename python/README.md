# Photara Python API

The Python API calls the same C++ libraries as the desktop application and CLI.
It does not start `photara.exe` or translate Python options into CLI arguments.

Build the native nanobind module with Visual Studio:

```powershell
python -m pip install nanobind
cmake -S . -B build -DPHOTARA_BUILD_PYTHON=ON
cmake --build build --config Release --target photara/photara_python
$env:PYTHONPATH = "$PWD\python"
```

With a single-configuration generator such as Ninja, the target name is
`photara_python`.

The import name is `photara`. The package searches common local build
directories for the compiled extension. For a custom build:

```powershell
$env:PHOTARA_NATIVE_DIR = "D:\path\to\build\photara\Release"
```

## SfM

Feature algorithms and compiled recipes can be inspected at runtime:

```python
import photara

print(photara.available_extractors())
print(photara.available_matchers())
print(photara.available_pair_pipelines())
print(photara.available_bundle_backends())
```

SfM options mirror the public C++ configuration, including the frontend,
relative-pose verification, retrieval, mapping, resection, global rotation,
global positioning, and bundle-adjustment controls:

```python
import photara as aes

options = aes.SfmOptions()
options.mode = aes.ReconstructionMode.GLOBAL
options.frontend.camera_model = aes.CameraModel.AUTOMATIC
options.frontend.extractor = "superpoint"
options.frontend.matcher = "lightglue"
options.frontend.extractor_model_path = "models/superpoint.onnx"
options.frontend.lightglue_model_path = "models/superpoint_lightglue.onnx"
options.frontend.relative.max_epipolar_error_px = 3.0
options.global_positioning.backend = aes.PositioningBackend.CUDA

aes.set_bundle_backend(aes.BundleBackend.CUDA)
scene = aes.run_sfm_directory("images", options)
scene.save_asfm("scene.asfm")
scene.save_colmap("sparse/0", image_path_base="images")
```

The frontend and mapping can be reused independently. Global rotation,
positioning, and bundle adjustment are also callable separately for solver
experiments:

```python
scene = aes.run_sfm_frontend(aes.discover_images("images"), options.frontend)
aes.estimate_global_rotations(scene, options.global_rotation)
aes.solve_global_positions(scene, options.global_positioning)
aes.bundle_adjust(scene, aes.BundleOptions())

# Or run the selected mapping mode with all nested mapping options.
scene = aes.run_sfm_frontend(aes.discover_images("images"), options.frontend)
summary = aes.run_sfm_mapping(scene, options)
```

## MVS and 3DGS

MVS and training are independently callable stages. `TrainingOptions` exposes
the public C++ training fields, including backend, optimization, loss,
densification, cache, evaluation, and preview controls:

```python
import photara as aes

sfm = aes.run_sfm_directory("images", aes.SfmOptions())
mvs = aes.build_mvs_scene(sfm, aes.MvsOptions())
aes.mvs_select_neighbors(mvs)
aes.mvs_estimate_depth_maps(mvs)
aes.mvs_fuse_depth_maps(mvs)

training = aes.TrainingOptions()
training.backend = aes.TrainingBackend.CUDA
training.iterations = 30_000
training.densification_strategy = aes.DensificationStrategy.EMC
training.apply_strategy_defaults()
gaussians = aes.train_3dgs(
    mvs, training, progress=lambda stats: print(stats.iteration, stats.loss)
)

tsdf = aes.TsdfOptions()
tsdf.fusion.mesh_tsdf_truncation_voxels = 6
mesh = aes.extract_tsdf(gaussians, mvs, training, tsdf)
mesh.save("mesh.ply")
```

Use `compiled_training_backends()` to inspect which training backends were
compiled into the current native module. Selecting an unavailable device raises
an exception from the C++ backend rather than silently launching another
process or changing the requested backend.

`examples/tsdf_experiment.py` loads an existing COLMAP model and trained splat
PLY, so TSDF parameters can be tested without rerunning SfM or 3DGS training.
