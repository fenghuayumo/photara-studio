# Photara Splat C++ and TinyTensor backend

## 1. Current implementation

Photara provides a compiled, differentiable Gaussian Splat training path that
is available through the CLI. The implementation combines ADC-style adaptive
density control, learned Gaussian normal fields, Mip-Splatting filtering,
TSDF/PAM mesh extraction, and the repository's `splat_drender` CUDA rasterizer.

```text
images -> SfM or imported cameras/sparse points
       -> Gaussian initialization
       -> CUDA rasterization and backward
       -> fused L1 + SSIM and optional geometry/mask losses
       -> TinyTensor Adam
       -> PLY / SOG / SPZ / GLB
```

The default path does not run MVS. SfM sparse points initialize Gaussians and
logs report `splat_input=sfm_sparse` and `patchmatch=false`. `--dense` is the
explicit MVS path.

Imported alignment data is selected with `--splat-dataset`; COLMAP,
RealityCapture, and OpenMVS layouts are detected automatically. Output format
follows the requested suffix. Other suffixes produce `<stem>_splat.ply` next
to the requested output.

The path has no LibTorch, PyBind, or Python runtime dependency. Projection,
sorting, blending, losses, backward, and optimizer updates execute on CUDA.

## 2. Source layout

- `photara/include/splat/`: public model, camera, training, and mesh APIs.
- `src/splat/rasterizer.cu`: TinyTensor to `splat_drender` bridge.
- `src/splat/cuda_ops.cu`: parameter activation, chain-rule gradients,
  fused supervised losses, and fused Adam.
- `src/splat/fused_ssim.cu`: 11×11 valid-window SSIM forward/backward.
- `src/splat/bilateral_grid.cu`: spatially varying affine color correction.
- `src/splat/ppisp.cu`: exposure, white balance, vignetting, and CRF models.
- `src/splat/trainer.cpp`: initialization, topology changes, training, and
  model I/O.
- `src/splat/training_data_loader.cpp`: bounded decode, host/device caches,
  masks, alpha, depth, and neighbor data.
- `src/splat/colmap.cpp`: imported camera and sparse-model loading.
- `src/splat/mesh.cpp`: rendered-frame TSDF extraction.
- `src/splat/pam_mesh.cpp`: learned-normal occupancy and PAM extraction.
- `photara/third_party/tinytensor/`: tensor storage and CUDA/Vulkan support.

## 3. Model and optimization

Each Gaussian stores a trainable world-space mean, log scale, quaternion,
opacity logit, and spherical harmonics up to degree three. Sparse or dense
input normals initialize orientation, pixel footprint initializes scale, and
point color initializes SH0.

The baseline photometric objective is compatible with the common
`0.8 * L1 + 0.2 * SSIM` formulation. SSIM uses an explicit separable 11×11
CUDA implementation with valid-window boundary semantics. Optional mask/alpha,
depth-normal, multi-view reprojection, planar NCC, and regularization terms are
added by configuration.

All gradients are explicit CUDA backward implementations and are applied by
TinyTensor Adam. Clone, split, prune, and recycle operations update parameters
and optimizer state together.

## 4. Masks and alpha

Splat masks are independent of the SfM feature-mask threshold:

- `--splat-use-mask` enables subject-only training.
- `--masks` selects the shared external mask directory.
- If no external mask exists, Splat may use source-image alpha.
- Projected MVS validity masks can also contribute where the selected path
  provides them.
- Source and projected coverage are resampled to the training camera.
- Subject-only training requires every training view to have a real foreground
  mask or source alpha; validity-only masks do not satisfy that requirement.

Two alpha modes are available:

- `transparent` (default): foreground RGB plus alpha BCE supervision.
- `masked`: foreground RGB plus an opacity-leak penalty outside the mask.

Soft coverage is preserved. Panorama training can retain a configurable
background alpha target so one view does not erase geometry required by
another view.

SfM uses the same mask directory earlier in the pipeline but interprets zero
as invalid and every non-zero value as valid. Therefore binary lossless PNG
masks are the safest shared format.

## 5. Color correction

Two optional training-time models compensate for exposure and white-balance
variation without changing the exported Gaussian model:

- PPISP: `--splat-ppisp` with `no_crf_no_vig`, `no_crf`, or `original`.
- Affine bilateral grid: `--splat-bilateral-grid`.

PPISP is anchored to an identity mean across views. The bilateral grid is
projected back to identity mean after updates so it represents spatial
variation while PPISP handles global exposure. When both are enabled, PPISP is
applied first.

## 6. Geometry supervision

Mesh-oriented training can enable:

- splat depth-normal consistency;
- multi-view depth reprojection with gradients through reference depth,
  projected query points, and neighboring-view Gaussians;
- half-pixel planar homography NCC with robust confidence;
- learned normal-field alignment;
- Mip-Splatting 3D filtering.

Depth-normal consistency back-projects the center neighborhood of median
depth, computes a normal from finite differences, and minimizes
`mean(1 - dot(rendered_normal, depth_normal))`. Gradients flow to both rendered
normal and median depth.

The 3D filter expands sub-pixel Gaussians according to visible camera distance
and compensates opacity by the scale-determinant ratio. It is recomputed after
topology changes and periodically during training and is stored as
`filter_3D` in PLY output.

## 7. Learned normal field and PAM

The trainer can learn four GaussianWrapping-style channels named
`gaussian_features_0..3`. The first three normalize to a direction; the fourth
uses `tanh` to represent orientation sign. The field has independent optimizer
state and follows clone, split, prune, and recycle operations.

PAM (`--mesh --mesh-method pam`) is available in CGAL-enabled builds and does
not use TSDF:

1. Create center/normal-offset pivots from selected Gaussians.
2. Evaluate multi-view occupancy and build a Delaunay tetrahedralization.
3. Extract a pivot mesh with Marching Tetrahedra.
4. Sample refinement candidates by projected face importance.
5. Project candidates toward the occupancy isosurface with the learned normal
   field.
6. Tetrahedralize again and extract the occupied/free boundary.

Foreground mask views contribute occupancy; explicit background votes empty;
views outside the frustum abstain. PAM processes the full scene and does not
apply camera focus, automatic ROI, `SubjectBounds`, or an external convex-hull
crop. Intermediate outputs include `*_pam_pivot_mesh.ply` and
`*_pam_candidates.ply`.

Important options include `--pam-pivot-max-points`,
`--pam-pivot-std-factor`, `--pam-max-points`, `--pam-refinement-steps`,
`--pam-neighbors`, `--pam-points-per-tetrahedron`, and
`--pam-occupancy-iso-value`.

## 8. Densification strategies

Photara exposes three strategies:

- **ADC-IGS**: default adaptive density control.
- **ADC+**: aggressive budget-aware growth and recycling.
- **EMC**: deterministic topology schedule useful for controlled comparisons.

Strategies share the same renderer, losses, optimizer, masks, and export
formats. Topology changes preserve all active model fields and optimizer state.
An initialization point budget can cap the sparse seed without changing the
source SfM scene.

## 9. Training data and memory

The data loader uses bounded host caches, pinned staging buffers, asynchronous
upload, view prefetch, and adaptive cache sizing. It records source dimensions
and camera calibration without decoding every full-resolution image during
metadata discovery.

`--splat-max-resolution 0` keeps the selected source files at their actual
resolution. Positive limits downscale the working camera and targets while
preserving calibration and source-to-working coordinate mapping.

Evaluation views can be held out with `--splat-eval-split-every`. Masked and
foreground metrics are reported on explicitly defined pixel sets so masked and
unmasked runs are not compared on incompatible support.

## 10. Mesh extraction

The default Splat mesh path renders median depth, normals, and alpha from each
registered view and integrates them into the shared sparse-block TSDF backend.
Marching Cubes output is cleaned and optionally repaired/decimated.

TSDF quality depends on source resolution, pixel sampling step, alpha support,
grazing-angle weights, voxel size, truncation, and minimum accumulated weight.
Thin structures can be lost when any of those gates are too coarse. TSDF is
not guaranteed to be watertight.

PAM is the alternative learned-normal occupancy path described above.

## 11. Build and run

```powershell
cmake -S . -B build `
  -DPHOTARA_ENABLE_CUDA=ON `
  -DPHOTARA_ENABLE_SPLAT=ON
cmake --build build --config Release --target photara --parallel
```

Sparse-initialized training:

```powershell
build\photara\Release\photara.exe `
  --images data\images `
  --output output\scene.ply `
  --splat `
  --splat-iterations 10000
```

Masked object training and TSDF mesh:

```powershell
build\photara\Release\photara.exe `
  --images data\images `
  --masks data\masks `
  --output output\object.ply `
  --capture-mode object
```

PAM:

```powershell
build\photara\Release\photara.exe `
  --images data\images `
  --splat-dataset data\sparse\0 `
  --output output\scene.ply `
  --splat --mesh --mesh-method pam
```

## 12. Performance design

- Keep rasterization, losses, backward, optimizer, and topology operations on
  the GPU.
- Use compact parameter and optimizer layouts.
- Bound image and target caches by memory budget.
- Prefetch and upload asynchronously.
- Reuse temporary storage and avoid per-iteration allocation.
- Use CUDA events and profiler categories for stage timing.
- Treat GPU model, driver/toolkit, build type, image resolution, Gaussian
  count, strategy, and iteration schedule as part of every benchmark.

Dataset-specific Nsight captures and A/B tables belong under `artifacts/`.
Architecture documentation records the stable conclusions: sample-depth
backward is generally more expensive than forward, occupancy varies strongly
with scene coverage, and scheduling changes must be validated for geometry
quality rather than timing alone.

## 13. Validation

`photara.splat.rasterizer` covers renderer forward/backward behavior, fused
loss gradients, masks, model serialization, topology operations, and training
data paths. Additional smoke tests validate imported datasets, short training
runs, mesh extraction, and finite exported model values.

Quality acceptance must include held-out rendering, mask support, thin
structures, connected components, topology diagnostics, and visual inspection;
iteration speed and triangle count are insufficient.

## 14. Current delivery and remaining work

Delivered:

- sparse SfM and imported-dataset initialization;
- CUDA forward/backward and TinyTensor Adam;
- ADC-IGS, ADC+, and EMC;
- external masks and source-alpha fallback;
- PPISP and bilateral-grid correction;
- geometry losses, learned normal field, and 3D filtering;
- TSDF and PAM mesh extraction;
- PLY/SOG/SPZ/GLB model I/O;
- editor preview and project archive integration.

Remaining quality work includes automatic subject-mask bootstrap, stronger
thin-structure preservation, broader end-to-end dataset gates, topology
quality thresholds, and resumable warmup/mask/final-training stages.

## 15. License boundaries

- This repository's own source follows the repository license.
- SiftGPU has upstream non-commercial restrictions and is optional.
- Delight/Intrinsic model weights are not distributed here and have a
  separate academic/non-commercial boundary; see
  [LICENSE-Intrinsic.md](LICENSE-Intrinsic.md).
- CUDA, ONNX Runtime, CGAL, FreeImage, VLFeat, cxxopts, TinyTensor, and
  `photara_drender` retain their respective upstream licenses.
- Before commercial distribution, review every enabled optional dependency and
  every externally supplied model weight.
