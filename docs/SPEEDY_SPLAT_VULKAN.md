# Speedy-Splat pruning on Vulkan

Vulkan supports the same opt-in Speedy-Splat pruning as CUDA through
`Rasterizer::pruning_scores()`, `TrainingOptions`, the reconstruction CLI and
Studio's Pruning checkbox. Start/interval/exclusive stop, full-resolution
gating, soft/hard ratios, training-camera selection, fixed-count pruning and
row-index tie breaking use the existing shared trainer implementation.
Surviving model rows, Adam moments, quantized SH state, normal features,
3D filters and densification evidence are compacted through the same GPU
row-selection path.

Enable it by adding `--backend vulkan --splat-speedy-pruning` to a splat
training command, or select Vulkan and enable Pruning in Studio.

The HLSL scorer uses exactly the CUDA scorer's per-pixel recurrence:
`sum_pixels (opacity * d(sum RGB)/d alpha)^2`. It consumes the forward draw
list and last accepted contributor, uses the same alpha floor/clip and
panorama wrapping, and includes background and occluded colour. Scoring
ignores precomputed feature colours, loss residuals and exposure correction;
it never dispatches backward or updates model parameters or Adam state.
The formula follows the existing CUDA adaptation of
[Speedy-Splat](https://speedysplat.github.io/) and
[Faster-GS](https://github.com/nerficg-project/faster-gaussian-splatting).

## Implementation and performance

The score render keeps attachments internal, omits depth/normal rendering,
backward snapshots and their prefix scan, and performs no per-camera host
readback. Shared memory stages 256 primitive attributes per tile. Subgroup
reduction combines pixel scores before global accumulation, reducing global
atomics from one per pixel/contributor to one per subgroup/contributor. Devices
with enabled Float32 atomic-add support use native atomic additions. Other
devices use integer compare/exchange; a shared-memory reduction also handles
devices without subgroup arithmetic. Scoring submits asynchronously on the
same queue as TinyTensor; the trainer downloads only the summed multi-camera
scores before ranking and compaction.

Floating-point summation order differs across the CUDA atomic, Vulkan subgroup
and portable reductions. Numerical agreement is tested with tolerance;
bitwise score equality and identical entire training trajectories are not
promised. The pruning policy and tie handling are identical.

## Validation

Build `photara_splat_test`, `photara` and `photara_studio` in Release, then run:

```powershell
./build/photara/Release/photara_splat_test.exe --speedy-only
./build/photara/Release/photara_splat_test.exe --speedy-vulkan-only
$env:SPLAT_DRENDER_DISABLE_ATOMIC_BACKWARD = '1'
./build/photara/Release/photara_splat_test.exe --speedy-vulkan-only
Remove-Item Env:SPLAT_DRENDER_DISABLE_ATOMIC_BACKWARD
$env:SPLAT_DRENDER_DISABLE_SUBGROUP_BACKWARD = '1'
./build/photara/Release/photara_splat_test.exe --speedy-vulkan-only
Remove-Item Env:SPLAT_DRENDER_DISABLE_SUBGROUP_BACKWARD
```

The existing diagnostic environment switches also select scorer fallbacks.
Tests cover independent CPU reverse-compositing oracles, non-black backgrounds,
occlusion, alpha clipping, transmittance early stopping, partial edge tiles,
pinhole/fisheye/panorama cameras (including the panorama seam), more than one
staged primitive batch, degree-3 SH, anisotropy, 3D filters, changing views,
all-culled views, repeated scoring, asynchronous multi-view accumulation and
topology changes. A cross-backend fixture checks every score and the exact
30% prune-row set. Fixed-model and ADC-IGS trainer tests perform three pruning
events, then continue optimizer/densification updates with compacted SH state.

## Measured CUDA comparison

2026-10-04, RTX 3060 Laptop GPU, driver 616.64, CUDA 12.8, Release build.
The same saved bike models and three evenly spaced COLMAP cameras were scored
at 960x540 with degree-3 SH and a black background. Each backend had three
warmup calls and 20 measured calls per camera. Times include forward rendering,
scoring and a GPU completion wait, and exclude loading/transferring the model,
score readback, ranking and compaction. These measure the complete per-camera
`pruning_scores()` API, not overall training speed.

| Gaussians | View | CUDA ms | Vulkan ms | CUDA/Vulkan | Relative score L1 | 30% prune-row symmetric difference |
|---:|---:|---:|---:|---:|---:|---:|
| 156832 | 0 | 6.43383 | 3.89464 | 1.65x | 2.07651e-6 | 0 |
| 156832 | 1 | 5.33932 | 3.64888 | 1.46x | 8.43612e-6 | 0 |
| 156832 | 2 | 5.01323 | 3.28053 | 1.53x | 3.00858e-6 | 0 |
| 815880 | 0 | 20.5288 | 11.4206 | 1.80x | 4.17560e-6 | 0 |
| 815880 | 1 | 19.6379 | 11.4430 | 1.72x | 5.34762e-6 | 0 |
| 815880 | 2 | 17.6040 | 9.75155 | 1.81x | 1.85317e-6 | 0 |

Relative L1 is `sum(abs(CUDA-Vulkan))/sum(CUDA)`. All six measured views
selected exactly the same rows for 30% pruning, including zero-score ties.
The synthetic 320-row, three-view fixture's maximum per-row relative error
was 1.35645e-5. These are finite model/view comparisons, not a universal
guarantee about near-tied nonzero scores in every possible scene.

Nine queued scoring views (three cameras repeated three times) agreed with
synchronized Vulkan scoring to relative L1 6.04103e-8 and 4.46071e-8 for the
156832- and 815880-row models respectively. Command-ring retirement happens
before reusing mapped camera data so overlapping views cannot overwrite
camera constants still consumed by an earlier preprocessing dispatch.

Reproduce against local saved models (image paths are used for COLMAP scene
metadata; the benchmark does not load training pixels):

```powershell
./build/photara/Release/photara_splat_test.exe --speedy-bench artifacts/speedy-bike-30000-gentle/gentle_42/model_splat.ply artifacts/speedy-bike-30000-gentle/colmap D:/ScanData/bike/images 20
./build/photara/Release/photara_splat_test.exe --speedy-bench artifacts/speedy-bike-30000-audit/baseline_42/model_splat.ply artifacts/speedy-bike-30000-audit/colmap D:/ScanData/bike/images 20
```
