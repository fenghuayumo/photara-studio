# splat_drender

Standalone CUDA backend for differentiable 3D Gaussian splatting, built for
AetherScan. It renders RGB, alpha, accumulated surface normals and median
depth, and provides analytically matching backward gradients for all of
them, including the median-depth bisection chain (T_p vacancy products,
dT/dtm accumulation, dL/dt_peak, dL/drsigma), which is the most
error-prone part of the GGGS-style geometry losses.

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
  api.h       public entry points and POD parameter views
  buffers.h   workspace pool layouts (explicit, no pointer arenas)
  camera.h    projection models + analytic Jacobians + pixel rays
  config.h    compile-time numerics (tile size, bisection, thresholds)
src/
  device/     device-only headers
    matrix.cuh    row-major Mat3 (+ glm-style at(col,row) accessor)
    sh.cuh        SH evaluation, forward and backward
    tiles.cuh     opacity-aware SnugBox/AccuTile enumeration
    fisheye.cuh   native fisheye geometry with dual-number autodiff
    geometry.cuh  EWA splat projection forward/backward (fused)
  render_forward.cu   fused preprocessing + instance emission + tile blender
  render_backward.cu  tile backward + fused per-Gaussian backward
  point_sampling.cu   unified point evaluation (occupancy / median depth)
  pipeline.cu         host orchestration
```

## Performance design (FasterGS-derived, order preserving)

1. Double radix sort replaces the classic 64-bit (tile | depth-bits) key
   sort: visible Gaussians are first stably sorted by their ordered 32-bit
   depth key, then (gaussian, tile) instances are stably sorted by tile id
   only. Cub radix sorts are stable, so the final within-tile depth order
   (including tie order) is bit-identical to the reference while sorting
   roughly half the key bits.
2. Warp-level sub-tile culling in the forward/backward blenders: each warp
   covers two tile rows and skips Gaussians whose screen bounds cannot
   overlap them. The skip never changes the contributor numbering, so
   n_contrib / max_contributor / last_contributor semantics (which the
   median-depth bisection and backward traversal depend on) are unchanged.
3. Fused kernels: preprocessing computes projection + conic + SH color +
   tile enumeration + depth key + screen bounds in one launch; the backward
   per-Gaussian stage merges the EWA geometry backward, mean2d->mean3d and
   SH backward; occupancy and median-depth point queries share one
   templated evaluation kernel with in-kernel point rounds (no
   tile-duplication buffers or auxiliary count/expand kernels).
4. Compact instance data: ushort4 screen bounds, 32-bit keys and
   __restrict__ on all hot pointers.

## Correctness policy for the depth path

The median-depth forward (seed + bisection) and backward (dT_dtm
accumulation, dL_dmt_dT_dtm, per-Gaussian dL_dGt / dL_dopa_sigma /
dL_drsigma / ray-plane gradients) are reimplemented formula-for-formula and
verified with central finite differences in aetherscan_splat_test
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
