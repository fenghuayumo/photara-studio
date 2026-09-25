# splat_drender

Differentiable 3D Gaussian splatting backends, built for Photara: a CUDA
backend for training and a Vulkan backend for drawing a trained model without a
CUDA context.

Both forward paths render RGB, alpha, accumulated surface normals and median
depth. The CUDA backend also provides analytically matching backward gradients
for all of them, including the median-depth bisection chain (T_p vacancy
products, dT/dtm accumulation, dL/dt_peak, dL/drsigma), which is the most
error-prone part of the GGGS-style geometry losses.

`splat_drender::splat_drender` is the CUDA library; `splat_drender::vulkan` is
the Vulkan one (built when a Vulkan SDK and dxc are found, see
`SPLAT_DRENDER_ENABLE_VULKAN`). The Vulkan backend runs the same EWA forward as
compute passes and shares the CUDA numerics deliberately: tile size,
opacity-aware AccuTile enumeration, alpha/transmittance thresholds, the 131072
single-sort crossover, median-depth bisection and the camera models.
`tests/vulkan_forward_compare.cpp` asserts that parity channel by channel
against the CUDA backend. The per-Gaussian state, the sorted instances and the
pixel snapshots of the last forward stay resident, which is where the
differentiable Vulkan pass will attach.

### Vulkan frame cost

The working set (Gaussian state, keys, values, the blended image and the
snapshots) lives in device memory; the host only touches it through the
context's staging buffer. Per-pixel bucket snapshots are off unless
`SplatSettings::pixel_snapshots` asks for them, because only the backward pass
reads them and at a million instances they are hundreds of megabytes per frame.
The radix sort sizes its passes from the meaningful key bits and builds its
histogram with one thread per key instead of one thread per digit.

Measured on an RTX 5090 at 1000x1000 with a 213k-Gaussian model (997k
instances), median of 20 frames:

| pass | Vulkan | CUDA |
|---|---:|---:|
| preview frame (RGBA8, stays on device) | 1.5 ms | - |
| preview frame (RGBA8 to host) | 2.5 ms | 4.1 ms |
| color + full channel readback | 5.7 ms | 0.75 ms |
| color, alpha, depth and normals | 12.2 ms | 3.6 ms |

The device-resident preview is the path the editor uses: it uploads nothing
back to the host and copies the RGBA buffer straight into its own image on the
shared device (`SplatRasterizer::render_rgba_device`).

## Scope

- Spherical harmonics view-dependent color (degrees 0..3) only; no
  Spherical Gaussians.
- Camera models: pinhole, OpenCV fisheye (k1..k4, native splat geometry with
  forward-mode dual-number backward), equirectangular (with seam wrap).
- Loss channels exposed by backward(): RGB, alpha, median depth, normals,
  plus a Brush-style refine_weight densification signal.
- World-space point queries: median-depth sampling (sample_depth /
  sample_depth_backward) and ray occupancy (evaluate_occupancy) used by the
  multi-view geometry/photometric losses.

## Architecture

```text
include/splat_drender/
  api.h          public entry points and POD parameter views (CUDA)
  buffers.h      workspace pool layouts (explicit, no pointer arenas)
  camera.h       projection models + analytic Jacobians + pixel rays
  config.h       compile-time numerics (tile size, bisection, thresholds)
  vulkan_api.h   public Vulkan context and forward rasterizer
src/
  device/     device-only headers
    matrix.cuh    row-major Mat3 (+ glm-style at(col,row) accessor)
    sh.cuh        SH evaluation, forward and backward
    tiles.cuh     opacity-aware SnugBox/AccuTile enumeration
    fisheye.cuh   native fisheye geometry with dual-number autodiff
    geometry.cuh  EWA splat projection forward/backward (fused)
  render_forward.cu   fused preprocessing + instance emission + tile blender
  render_backward.cu  bucket-parallel backward + fused per-Gaussian backward
  point_sampling.cu   unified point evaluation (occupancy / median depth)
  pipeline.cu         host orchestration
  vulkan/
    context_internal.hpp  mapped buffer, compute pipeline, device owner
    context.cpp           instance/device/queue and embedded SPIR-V lookup
    splat_rasterizer.cpp  Vulkan equivalent of render_forward.cu
shaders/
  splat_math.hlsli        projection, EWA, SH and camera math
  splat_enum.hlsli        opacity-aware tile walk
  splat_preprocess.hlsl   per-Gaussian state
  splat_scan.hlsl         block scan for instance offsets
  splat_emit.hlsl         instance keys in depth order
  splat_radix_hist.hlsl   radix histogram
  splat_radix_scatter.hlsl  stable radix scatter
  splat_ranges.hlsl       per-tile instance ranges
  splat_blend.hlsl        tile blender (color, alpha, depth, normals)
```

## Performance design (FasterGS-derived, order preserving)

1. Adaptive stable radix sort: lists with at most 131,072 instances use one
   64-bit (tile | depth-bits) sort to avoid the additional scans and launches
   of a depth pre-sort. Larger lists use the FasterGS double sort: visible
   Gaussians are sorted by 32-bit depth, then instances are stably sorted by
   tile id. Both paths preserve Gaussian-index tie order. Workspace layout
   and backward reconstruction select the same path from instance count.
2. Per-pixel early termination in forward and median-depth traversal, with
   median batches bounded by the last contributing instance. Depth probes
   use separate compile-time first/refinement paths.
3. Fused kernels: preprocessing computes projection + conic + SH color +
   tile enumeration + depth key + screen bounds in one launch; the backward
   per-Gaussian stage merges the EWA geometry backward, mean2d->mean3d and
   SH backward; occupancy and median-depth point queries share one
   templated evaluation kernel with in-kernel point rounds (no
   tile-duplication buffers or auxiliary count/expand kernels).
4. Warp-aggregated preprocessing counters on SM80+, a single adjacent-counter
   readback, 32-bit keys for large lists, and __restrict__ on hot pointers.
5. Bucket-parallel backward: the forward blender snapshots per-pixel
   (color, transmittance[, accumulated normal]) at every 32-instance bucket
   boundary together with the fully accumulated color, and one warp per
   bucket then walks the bucket's 32 instances against all 256 tile pixels.
   Per-lane gradient accumulators live in registers for the whole tile and
   commit once, replacing the classic per-instance warp reductions, shared
   memory staging and per-warp atomics of the thread-per-pixel walk. The
   per-pixel (T, color-after, normal-after) state enters the warp at lane 0
   from the bucket snapshot and flows diagonally through warp shuffles;
   the reference's accum_rec chains are computed with the equivalent
   "state-after" formulation. Bucket offsets are derived on device from the
   sorted tile ranges; the launch grid uses a deterministic upper bound so
   no second host readback is needed. The geometry median walk
   (dT_dtm -> dL_dmt per pixel) runs as a separate thread-per-pixel kernel
   before the bucket walk reads it.

## Reproducible timing

Build with `scripts/build_ab_compare.ps1`. The harness compares identical
inputs before timing warmed forward, backward, and forward/backward pairs.
`SPLAT_BENCH_ITERS` enables timing. Captured Gaussian counts are inferred from
`.inputmeans`; optional trailing width/height arguments specify image shape.

`experiments/benchmark_splat_rasterizer.py` runs serial repeats, rejects
non-finite/mismatched comparison channels, saves raw logs and reports median
wall time. Use `--geometry` to test depth/normal gradients. CUDA event timings
include CPU submission gaps, not just kernel execution. The 131,072 crossover
is a measured heuristic, not a hardware-independent optimum. See
`docs/SPLAT_RENDERER_PERFORMANCE_20260911.md` for the measurements and limits.

## Correctness policy for the depth path

The median-depth forward (seed + bisection) and backward (dT_dtm
accumulation, dL_dmt_dT_dtm, per-Gaussian dL_dGt / dL_dopa_sigma /
dL_drsigma / ray-plane gradients) are reimplemented formula-for-formula and
verified with central finite differences in photara_splat_test
("Depth Gradients" case).

## Reference-parity conventions

Four gggs_reference numerical conventions are reproduced deliberately so
training matches the original library bit-for-bit where the arithmetic
allows; each differs from exact calculus:

1. Footprint-normal backward keeps the normalization Jacobian factor
   evaluated on the already-normalized normal (1, not 1/|cam_normal|).
2. The Mip opacity compensation coef = sqrt(det0/det1) is detached from
   covariance gradients (relevant only for kernel_size > 0; training
   defaults to kernel_size = 0 where coef == 1).
3. The fminf(0.99, alpha) clamp gradient is zeroed only for fisheye.
4. Pixels whose median-depth bracket fails keep their upstream depth-loss
   gradient, which lands in the 1/1e-7 dL_dmt_dT_dtm penalty.

Point queries additionally use the reference's wide bracket: a
SAMPLE_RANGE_TESTING*2 = +/-200 seed window with 8 bisection refinements
(vs the pixel renderer's +/-0.4 and 5).

The A/B harness (tests/reference_compare.cu) links both rasterizers over
identical inputs and reports every forward/backward channel at ~1e-6
relative L2 (random scene) and ~1e-7 (captured 22k-Gaussian scene); the
residual is atomic summation order.
Bucket-snapshot backward changes floating-point association (front-to-back
"state-after" products instead of back-to-front transmittance division), so
gradients match the reference at ~1e-6 relative L2 rather than bit-exactly,
well inside the harness tolerance (1e-4).

### Thin-splat regression coverage

Pixel forward, median traversal and bucket backward use an explicitly rounded
Gaussian quadratic form. Identical source expressions alone are insufficient:
CUDA contraction/common-subexpression choices in backward changed the alpha
of a thin captured splat, yielding a 0.9% refine-gradient discrepancy despite
matching forward images. The RGB path also skips the inverse-covariance
depth/normal chain when its upstream gradients are all zero; otherwise an
ill-conditioned splat can introduce NaNs through a mathematically unused branch.
`test_thin_splat_rgb_backward` checks the exact single-splat color derivative
against forward alpha and finite geometry gradients on captured fixtures.

The small-scene tolerance above is not a guarantee for arbitrarily thin trained
Gaussians. A 671k-Gaussian capture exposes non-finite gradients in the reference
itself and larger residual relative errors. See
`docs/SPLAT_QUALITY_20260913.md` in the Photara repository for the matched
30k-step training comparison, fixes, and remaining numerical limits.
