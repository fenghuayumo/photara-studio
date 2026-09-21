# Photara Python experiments

Build the native nanobind module:

```powershell
python -m pip install nanobind
cmake -S . -B build -DPHOTARA_BUILD_PYTHON=ON
cmake --build build --config Release --target photara_python
$env:PYTHONPATH = "$PWD\python"
```

The package searches common local build directories. For a custom build:

```powershell
$env:PHOTARA_NATIVE_DIR = "D:\path\to\build\photara\Release"
```

The API exposes independently callable stages:

```python
import photara as aes

sfm = aes.run_sfm_directory("images", aes.SfmOptions())
mvs = aes.build_mvs_scene(sfm, aes.MvsOptions())
aes.mvs_select_neighbors(mvs)
aes.mvs_estimate_depth_maps(mvs)
aes.mvs_fuse_depth_maps(mvs)

training = aes.TrainingOptions()
training.iterations = 30_000
training.densification_strategy = aes.DensificationStrategy.ADC_PLUS
gaussians = aes.train_3dgs(mvs, training)

tsdf = aes.TsdfOptions()
tsdf.fusion.mesh_tsdf_truncation_voxels = 6
mesh = aes.extract_tsdf(gaussians, mvs, training, tsdf)
mesh.save("mesh.ply")
```

`examples/tsdf_experiment.py` loads an existing COLMAP model and trained splat
PLY, so TSDF parameters can be tested without rerunning SfM or 3DGS training.
