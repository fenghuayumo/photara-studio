# 3DGUT rasterization

Select **Rendering method → 3DGUT** in advanced training settings, or
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
The CUDA RGB-only path caches world-to-Gaussian rows and camera-relative
origins during preprocessing. Forward blending stages tile batches in shared
memory. Training records color/transmittance snapshots every 32 Gaussians and
caches world rays. Backward streams pixels through 32-Gaussian warp buckets,
accumulates analytic gradients in registers, then transforms geometry gradients
once per Gaussian before the existing SH/activation/fused Adam updates. The
floating-point ray orthogonality residual is retained in the VJP for thin
Gaussians. Snapshot storage trades additional memory and forward writes for
much faster backward; its size follows conservative tile instance counts.
Inference does not allocate snapshots or cached pixel rays.

Depth/normal attachments retain the original forward-mode derivative path.
For diagnostic comparisons, `PHOTARA_GUT_REFERENCE=1` selects the original
pixel implementation for RGB as well; set it consistently across forward and
backward. Reordered floating-point transforms can differ locally for severely
ill-conditioned, extremely thin Gaussians; bitwise equivalence is not promised.
Conservative UT bins remain unchanged. Performance parity with optimized 3DGS
or NVIDIA's implementation has not been established.

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

Training currently targets CUDA. 3DGUT uses appearance-only optimization: depth,
normal, multi-view and AI geometry supervision are disabled even if requested.
TSDF/PAM extraction rejects GUT models and recommends training with 3DGS.

Training and display methods remain separate from backend selection. The editor
viewport renders GUT models using
Vulkan 3DGUT forward, including RGB, depth, normal and rings shading. The
rendering method follows the loaded model (including PLY metadata and embedded
models), and is preserved after edits, undo/redo, reload and diagnostic shading
changes. Selecting a training method does not alter the selected training
backend. Vulkan GUT training, point-depth queries, occupancy queries and Speedy
pruning are unsupported. GUT training logs that geometry losses are disabled.
Occupancy/PAM extraction and point-depth queries report an explicit error for
a GUT model.

With `SPLAT_DRENDER_BUILD_TESTS=ON`, `splat_gut_gradient_test` checks all geometry,
opacity and color derivatives against finite differences, a double-precision
CPU ray oracle, and a descending gradient step. It covers pinhole, OpenCV
fisheye, equirectangular and orthographic cameras, geometry attachments,
precomputed colors, SH and compact draw lists. RGB cases also compare against
the original pixel derivative path. Odd image dimensions and 300 overlapping
Gaussians cover partial tiles, multiple warp buckets and shared-memory batches.
A separate 1e-8-scale stress case checks finite outputs and gradients; it does
not assert finite-difference or original-path agreement at that condition number.

The main project's `photara_splat_gut_test` exercises CUDA training with fused
Adam, densification and GS/GUT PLY/archive roundtrips, plus legacy PLY loading
and rejection of Vulkan GUT training. `photara_project_io_test` covers project
settings persistence and legacy versions.

`splat_render::GaussianCloud::render_method` selects the viewport method.
The shared Vulkan renderer accepts
`SplatSettings::primitive = SplatPrimitive::gaussian_3dgut` for forward-only
rendering; recording backward snapshots and point queries is rejected. Its
seven-point UT and ray response match CUDA, including panorama seam wrapping,
orthographic ray origins, SH/precomputed colors, optional 3D scale filtering,
alpha compositing, peak median depth and camera-facing normals. Projected
2D conics only select tiles, while per-pixel opacity comes from the 3D ray.
RGB preview remains on the editor's Vulkan device without CPU image readback.

`splat_gut_forward_test` compares Vulkan RGB/alpha/depth/normals with an
independent double-precision CPU ray oracle across four camera models and
transformed cameras. It also exercises packed RGBA, panorama seams, method
switches, empty frames, and viewport cache, diagnostic shading, rings, center
edits and reloads. Its optional `--validation` flag also checks Vulkan validation
and synchronization errors. `splat_display_test` retains the 3DGS display regression.

Algorithm reference: [NVIDIA 3DGUT project and paper](https://research.nvidia.com/labs/toronto-ai/3DGUT/).


`photara_splat_vulkan_parity --bench-cuda-methods MODEL IMAGES OUTPUT SFM
[MAX_RESOLUTION] [ITERS]` measures resident-parameter CUDA rasterizer timings
for GS and GUT, using three warmups and synchronized median wall times.
Backward includes gradient allocation and the activation chain, but excludes
optimizer and losses. `--compare-gut-paths` with the same arguments compares
RGB, alpha and all five model-gradient groups against the original GUT path;
it rejects nonfinite results and relative L2 above 1e-3 for images or 5e-3 for
gradients. These comparisons use the same parameters and do not establish
quality equivalence between independently trained GS and GUT models.


## Vulkan viewport forward optimization

The RGB/rings GPU preview uses a dedicated `splat_blend_gut_rgba` pipeline.
It stages 64 candidates at a time, omits geometry/training-statistics storage,
reduces live-pixel detection to one shared atomic per live subgroup, and packs
RGBA directly in the blend shader. This avoids the intermediate four-plane
float image and separate RGBA packing dispatch. Non-orthographic cameras also
skip the zero ray-origin-offset matrix products.

Before evaluating a candidate for each pixel, the tile tests its finite alpha
support against camera-origin halfspaces. Pinhole tiles use four frustum planes;
full-domain equirectangular tiles use two exact meridian planes, including seam
tiles. Fisheye/orthographic cameras do not use this tile rejection. Normals are
transformed into Gaussian principal coordinates with cofactors, avoiding a
matrix inverse. Tests use padded support and cancellation margins; transforms
with principal-axis condition ratio above 1e4 retain the pixel evaluation. The
accepted candidates retain the original ordering, alpha test, transmittance
termination, 3D response and rings contour calculation. UT bins/sorting are
unchanged; the optimization reduces blend work within those bins.

Depth/normal diagnostic attachments retain the original pipeline. Devices
without the required compute subgroup operations also use the original RGBA
path. `SPLAT_DRENDER_GUT_REFERENCE=1` selects it at rasterizer construction for
A/B measurement. The editor still receives a GPU buffer on its shared device.

`--bench-viewport MODEL IMAGES OUTPUT SFM WIDTH HEIGHT ITERS [GAUSSIANS]
[VIEW] [pano|pinhole]` measures synchronized median latency after five warmups.
The pinhole option uses the selected real pose with a 70-degree vertical FOV.
Each RGB/rings measurement also compares packed output with the original float
blender quantized using the same byte conversion, rejecting differences above
one byte level. It excludes model upload, UI/compositing and presentation.
The viewport forward test additionally exercises 300 overlapping Gaussians,
partial batches/tiles, rings, and 1e-8 thin scales against the original path.
