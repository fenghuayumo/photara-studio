# Rasterization and filtering ablations

The defaults are `--splat-raster-profile standard` and
`--splat-filter-mode classic`. These controls change Photara's rasterizer; they
do not select or execute the Brush application.

## Raster policy

| Control | Standard | Mip Splatting numerical preset |
|---|---|---|
| Sort key | Camera-center Euclidean distance | Camera-space Z |
| Per-contributor alpha ceiling | 0.99 | 0.999 |
| Pinhole near plane | 0.2 scene units | 0.01 scene units |
| Pixel sample location | Integer | Half-integer |

Select the preset with `--splat-raster-profile mip-splatting`. Each control can be
overridden independently with `--splat-depth-sort radial|z`,
`--splat-alpha-clip`, `--splat-near-plane`, and `--splat-pixel-center 0|0.5`.
The preset is experimental; filter modes do not select it. Compare filter
modes under the default standard raster policy before attributing changes to
filtering. The benchmark defaults to `classic mip2d mip3d`; select
`mip_splatting_*` variants explicitly to study the experimental preset.
The near plane is expressed in the SfM scene's units, not metres.
Equirectangular rendering retains radial sorting because camera Z is not a
valid ordering over the full sphere. Native fisheye retains its existing
projection visibility gate.

The implementation uses integer kernel pixel indices and subtracts the sample
offset from the principal point. Projection, ray construction, backward,
point queries, geometry losses and TSDF back-projection share that convention.
CUDA and Vulkan propagate the alpha ceiling and compensation switch through
the corresponding backward and point-query passes. Changing the alpha ceiling
does not otherwise replace Photara's gradient-clipping approximations with
Brush's complete backward implementation.

Near-plane clipping is tested for the forward rasterizer on CUDA and Vulkan.
CUDA depth-point queries also accept a sample just beyond the preset near plane.
The current Vulkan median-depth query still reports that same synthetic sample
as outside even though its forward render contains the surface. Treat near
plane changes on Vulkan geometry supervision as incomplete until that query
path is corrected; the ORI raster and mesh benchmark uses CUDA.

In an 8,000-step, single-seed room-scene diagnosis, individual held-out
views showed strong sensitivity to the preset even when mean PSNR changed
little. On one fixed Classic model, changing the near plane from 0.01 to 0.2
improved view 104 by 7.51 dB; on a different model, switching from Z to radial
sorting improved view 224 by 4.60 dB. These are fixed-model interventions, not
evidence that either setting is universally better after retraining. Inspect
per-view results and near-camera Gaussian occlusion, and investigate sorting
artifacts before changing default raster settings.

## Primary filter modes: choose 2D or 3D

The 2D Mip filter is distinct from Classic's covariance dilation. Classic's
0.3 variance remains present in `mip3d`; “3D only” means no **2D Mip energy
compensation**, not the absence of the Classic footprint.

| `--splat-filter-mode` | Screen covariance variance | 2D opacity compensation | 3D filter |
|---|---:|---|---|
| `classic` | 0.3 px² | Off | Off |
| `mip2d` | 0.1 px² | On | Off |
| `mip3d` | 0.3 px² | Off | On |

The primary product comparison is Classic versus Mip2D versus Mip3D, with
the two Mip filters mutually exclusive. The existing `mip` value (0.1 px²,
both compensations) is retained only for the completed combined-filter
ablation and reproducibility; it is not a recommended primary mode and is
excluded from the benchmark script's default variant list.

The selected mode controls both the screen kernel and the 3D filter. The old
independent kernel and 3D-filter CLI controls are no longer available.
Explicit 3D modes use strength 0.1 and
`sqrt(0.1) * min_camera(distance / that_camera_fx)` over training cameras.
The radius is frozen between refreshes, never optimized. It expands each
scale to `sqrt(scale² + radius²)` and compensates opacity by the product of
raw/filtered scales. Its scale/opacity cross-derivative is retained.

Photara continues to keep the 3D floor separate, refresh after refinement and
resolution changes, periodically refresh after growth ends, and serialize
`filter_3D`. This is **not** Brush's bake-before-refine/export lifecycle.
Native non-pinhole training retains the pre-existing 3D-filter restriction.

PLY and project model data store the filter mode; untagged models use Classic.
Studio uses the stored screen kernel and 2D compensation. Non-Classic modes
require PLY export to preserve the mode. Third-party readers must support
`filter_3D` to reproduce the 3D floor.

The ordinary pinhole 2D Mip opacity compensation remains detached from the
covariance gradient. Native fisheye differentiates its exact compensation.
Determinant guards and other numerical safeguards remain Photara's existing
ones; the Mip Splatting numerical preset is not a claim of complete bitwise parity.

## Validation and experiments

`photara_splat_test --raster-policy-only` checks CUDA and Vulkan against
independent analytic Gaussian/compositing oracles. Tests distinguish the two
sort orders, sample positions and near planes, check both alpha ceilings,
Classic opacity backward, 2D/3D compensation, and mixed-focal 3D radii.

`scripts/benchmark_raster_policy.py` records immutable input/model executable
hashes, exact commands, serial training times, held-out renders, synchronized
device-resident forward latency, depth maps, alpha and normal maps. The script
explicitly disables mesh training unless `--geometry` is selected; an explicit
capture mode would otherwise enable meshing in the product CLI.

`scripts/analyze_raster_policy.py` computes valid-region appearance metrics,
spatial LPIPS-Squeeze, sampled cross-view depth consistency, depth/normal
agreement, near-camera Gaussian counts and optional TSDF topology statistics.
Near-camera counts and geometric consistency are diagnostics, not ground-truth
accuracy. Shared SfM calibration and sparse initialization can include held-out
images. The split therefore measures appearance generalization with shared
initialization, not an independently reconstructed test scene.

Brush reference used to derive the numerical preset:
[`1388f74`](https://github.com/ArthurBrussee/brush/tree/1388f74c6fe0236f68ee4915564bf00e9d2e3747).
