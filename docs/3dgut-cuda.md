# 3DGUT CUDA rasterization

Select **Rendering method → 3DGUT (CUDA)** in advanced training settings, or
pass `--backend cuda --splat-render-method 3dgut` to the `photara` CLI.
The default remains `--splat-render-method 3dgs`.

`TrainingOptions::render_method` selects the reconstruction method. The method
then belongs to `GaussianModel`, so CUDA rasterization, training previews,
evaluation, densification, cloning, transformations and PLY reloads use the
model's method. A resumed checkpoint must use the same method.

## Primitive and differentiation

`splat_drender::Gaussians::primitive` accepts `gaussian_3dgs` or
`gaussian_3dgut`. Both use activated opacities, positive scales, scalar-first
unit quaternions, and the existing SH or precomputed RGB layout. The GUT path
requires scale/quaternion inputs; precomputed covariance input is unsupported.

The CUDA implementation uses a seven-point unscented transform (lambda=1) to
estimate the projected footprint for tile binning. Projected sigma points wrap
across the panorama seam. Bins expand the estimated footprint to accommodate
nonlinear tails; sigma points crossing the projection boundary fall back to
full-image bins. This remains approximate culling, as in other UT rasterizers.
It reuses the existing stable tile/depth sort.

Pixel blending evaluates the maximum 3D Gaussian response along the camera
ray, in Gaussian principal-axis coordinates. The 2D conic is used only for
binning. GUT does not apply the EWA `kernel_size` opacity compensation;
the model's existing optional 3D scale filter still applies. Depth is the
camera-space peak depth of the contributor crossing 50% transmittance (zero
if there is no crossing). Normals are alpha-normalized, camera-facing minimum
scale axes in camera coordinates.

The backward pass replays the accepted contributions and differentiates the
3D response, peak depth and normal, then reuses the SH derivative, parameter
activation chain and fused SH/structure Adam updates. Tile membership, sort
order, clipping, median-contributor selection and normal-axis selection are
piecewise constant, like the corresponding discrete choices in 3DGS.
This first implementation uses forward-mode geometry derivatives and
conservative bins to prioritize correctness; performance parity with the
optimized 3DGS path or NVIDIA's implementation has not been established.

## Persistence

PLY keeps the existing vertex properties and adds one of these header comments:

```text
comment photara_render_method 3dgut
comment photara_render_method 3dgs
```

Only one is written per file. Untagged legacy PLY files load as 3DGS. Unknown
method tags are rejected. PLY is currently the supported export format for
3DGUT; other export formats are rejected rather than losing the method.
The embedded Gaussian archive uses payload v2 for GUT and retains v1 for GS.
Project settings v21 preserve the method selection; older settings default to
3DGS. Merging models with different rendering methods is rejected.

## Current scope and checks

The implementation targets CUDA. The editor Vulkan viewport still uses its
existing 3DGS preview; CUDA training previews use GUT. Vulkan GUT rendering/training, point-depth
queries, occupancy queries and Speedy pruning are unsupported. GUT training
logs that multi-view point-depth losses are disabled. Occupancy/PAM extraction
and point-depth queries report an explicit error for a GUT model.

With `SPLAT_DRENDER_BUILD_TESTS=ON`, `splat_gut_gradient_test` checks all geometry,
opacity and color derivatives against finite differences, a double-precision
CPU ray oracle, and a descending gradient step. It covers pinhole, OpenCV
fisheye, equirectangular and orthographic cameras, geometry attachments,
precomputed colors, SH and compact draw lists.

The main project's `photara_splat_gut_test` exercises CUDA training with fused
Adam, densification and GS/GUT PLY/archive roundtrips, plus legacy PLY loading
and rejection of Vulkan GUT training. `photara_project_io_test` covers project
settings persistence and legacy versions.

Algorithm reference: [NVIDIA 3DGUT project and paper](https://research.nvidia.com/labs/toronto-ai/3DGUT/).
