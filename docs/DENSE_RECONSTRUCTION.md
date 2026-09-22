# Photara dense reconstruction and texturing architecture

## 1. Product boundary

After SfM, the default product path initializes Gaussian Splatting directly
from registered cameras and sparse landmarks. Splat optimizes multi-view
appearance and geometry, then rendered median depth, normals, and alpha feed a
TSDF mesh pipeline.

```text
SfM -> sparse points -> Splat -> TSDF -> Clean -> Texture / Delight
```

The default path deliberately skips PatchMatch, MVS cloud fusion, MVS
Delaunay/Poisson meshing, and masks derived from MVS depth or mesh. Those
stages can discard spokes, wires, thin sheets, and occlusion boundaries before
Splat has a chance to recover them. MVS remains available as an explicit
diagnostic and compatibility backend.

The default build does not require OpenCV. Splat is CUDA-oriented; TSDF and
mesh cleanup use multithreaded CPU code.

## 2. Pipeline

```text
Images
  -> SfM
  -> registered cameras + sparse landmarks
  -> sparse-point filtering and optional SubjectBounds
  -> Gaussian initialization and adaptive densification
  -> photometric and multi-view geometry optimization
  -> median depth / normal / alpha frames
  -> sparse-block TSDF
  -> Marching Cubes -> topology audit -> Clean
  -> optional UV unwrap, texture baking, and Delight
```

Object and scene modes share the geometry path:

- **Object mode** uses conservative spatial bounds and foreground constraints.
  External masks are consumed directly when present. Automatic subject-mask
  bootstrapping remains incomplete and must not silently turn the entire image
  into foreground.
- **Scene mode** does not separate foreground from background; spatial bounds,
  visibility, and opacity control reconstruction extent.

Public product options are intentionally small:

```text
--capture-mode object|scene
--masks <directory>            optional
```

Explicit `--capture-mode` enables Splat and TSDF mesh extraction. Omitting it
preserves low-level SfM-only workflows. Object mode estimates `SubjectBounds`
from SfM sparse points; scene mode disables subject cropping.

## 3. User-visible capabilities

| Capability | Meaning | Dependency |
|---|---|---|
| `capture_mode=object` | object reconstruction with bounds and foreground constraints | SfM |
| `capture_mode=scene` | scene reconstruction without foreground separation | SfM |
| Splat | sparse-point Gaussian initialization and training | SfM + CUDA |
| Mesh | rendered geometry -> TSDF -> Clean | trained Splat |
| Texture | UV unwrap, projection, seam resolve, dilation | final mesh |
| Delight | image-space intrinsic decomposition and albedo | Texture + external ONNX weights |

```powershell
photara --images images --output object.ply --capture-mode object
photara --images images --output object.ply --capture-mode object --masks masks
photara --images images --output scene.ply --capture-mode scene
```

## 4. Core data model

The orchestration state can be viewed as:

```text
RebuildScene
  cameras[] / views[]
  sparse_points[] / observations[]
  subject_bounds
  input_masks[]
  gaussians
  rendered_depth[] / rendered_normal[] / rendered_alpha[]
  mesh
```

The actual implementation passes `sfm::Scene`, MVS/Splat adapters, project
archive chunks, and working files rather than exposing a single monolithic
public type.

Checkpoints are split by responsibility:

```text
SfM --cache-dir: features / matches / geometry / tracks / reconstruction
.ascan chunks:    settings / sfm / gaussians / mesh
editor files:     --working-sfm / splat / mesh / dense / texture
```

## 5. Stage A: sparse initialization and subject constraints

### Input gates

Before training, Photara validates registered cameras, sparse landmark
observations, finite coordinates, source-image availability, camera/image
dimensions, and subject mask availability when required.

SfM masks and Splat masks share the same directory but have different
semantics. The composable SfM frontend uses zero/non-zero keypoint validity;
Splat preserves soft mask coverage and may fall back to source alpha. Masked
alignment therefore requires mask files to exist before SfM starts.

### Subject bounds

Object mode estimates a conservative oriented region from SfM sparse points,
using outlier rejection and padding. `SubjectBounds` constrains initialization,
TSDF extent, and cleanup behavior. It is a 3D spatial constraint, not a 2D
mask, and does not delete SfM points by itself.

### Subject-mask bootstrap

The intended automatic path is:

```text
sparse initialization -> short warmup -> subject Gaussian selection
  -> source-resolution soft masks -> confidence gate -> full training
```

Soft alpha, holes, and thin structures must be preserved. Low-confidence
selection must request user input or external masks rather than silently
falling back to the full frame. This automatic path is not yet complete.

### Splat training

Sparse points initialize Gaussian means, scales, rotations, opacity, and SH
color. Training combines photometric loss with optional alpha/mask loss,
depth-normal consistency, multi-view depth reprojection, planar NCC, learned
normal fields, and Mip-Splatting filtering. Densification strategy is selected
from ADC-IGS, ADC+, or EMC.

## 6. Stage B: TSDF and mesh

For each registered view, the trained model renders median depth, normals, and
alpha. The TSDF backend:

1. allocates sparse voxel blocks around observed surfaces;
2. integrates depth with alpha, normal, grazing-angle, and view weights;
3. applies truncation and minimum-support gates;
4. extracts the zero level set with Marching Cubes;
5. audits topology and runs source-aware cleanup.

TSDF must preserve thin structures when source resolution and sampling permit,
but it is not guaranteed to produce a watertight manifold. Cleanup removes
degenerates, tiny components, spikes, non-manifold defects, and small holes
without closing intentional `SubjectBounds` crop boundaries.

PAM is an alternative Splat mesh backend. It uses learned Gaussian normal
fields, occupancy, Delaunay tetrahedralization, and Marching Tetrahedra instead
of rendered-depth TSDF.

## 7. Stage C: Texture and Delight

Texture consumes only the final mesh and registered source views:

```text
mesh -> UVAtlas -> view selection -> projective bake
     -> seam optimization -> dilation -> OBJ/MTL/albedo
```

Foreground masks limit visibility and projection support. Final rasterized mesh
masks are texture artifacts; they do not feed back into Splat or camera
alignment.

`--delight` applies image-space intrinsic decomposition before projection.
Model weights are external and have a separate license boundary documented in
[LICENSE-Intrinsic.md](LICENSE-Intrinsic.md).

## 8. Orchestration state machine

```text
require and validate SfM cameras + sparse landmarks
if object mode:
    estimate SubjectBounds
    if external masks exist:
        load soft masks
    else:
        run subject bootstrap when available
        require confidence gate

train sparse-initialized Splat

if mesh requested:
    render depth/normal/alpha
    extract TSDF or PAM mesh
    audit and clean mesh

if texture requested:
    optionally delight source views
    unwrap, project, optimize, and dilate texture
```

No stage may silently substitute MVS depth or mesh for a missing subject mask.

## 9. Integration with SfM and MVS

- Default in-process path: `sfm::Scene -> sparse Splat -> TSDF`.
- SfM cameras, sparse colors, and observation tracks remain intact through
  Gaussian initialization.
- OpenMVS export is an interoperability artifact, not an internal dependency.
- Imported COLMAP, RealityCapture, and OpenMVS datasets can bypass internal
  SfM through `--splat-dataset`.
- Explicit `--dense` remains available for PatchMatch comparison and MVS mesh
  diagnostics.

## 10. Performance principles

1. Keep images, masks, depth, and training targets in bounded host/device
   caches instead of loading the entire dataset onto the GPU.
2. Use multithreaded CPU work for SfM, preprocessing, TSDF, Marching Cubes,
   cleanup, and export.
3. Keep Splat rasterization, loss, backward, densification, and Adam on CUDA.
4. Avoid synchronization in the per-iteration hot path.
5. Use source-resolution decode only when a stage genuinely consumes it.
6. Treat benchmark datasets, GPU model, build type, and iteration schedule as
   part of every performance result.

The editor can receive CUDA-backed previews through the Vulkan interop path.
Preview failures must not corrupt training state; host preview remains a
fallback.

## 11. Current implementation status

Implemented:

- internal SfM sparse initialization;
- optional prompt-driven SAM 3 mask generation before SfM;
- external COLMAP/RealityCapture/OpenMVS dataset loading;
- CUDA Splat forward/backward and training;
- ADC-IGS, ADC+, and EMC densification;
- soft external masks and alpha supervision;
- depth-normal, multi-view geometry, NCC, normal-field, and Mip-Splatting
  geometry terms;
- TSDF and PAM mesh extraction;
- mesh cleanup, export, texture baking, and optional Delight;
- `.ascan` project chunks and editor working files.

Incomplete or intentionally separate:

- fully automatic, prompt-free object subject bootstrapping;
- arbitrary resume points between warmup, generated masks, and final training;
- automatic promotion of MVS artifacts into the product Splat path;
- guaranteed watertight topology for thin or heavily occluded objects.

## 12. Validation and acceptance

Acceptance must examine more than loss or triangle count:

- SfM registration coverage and `*_sfm_diagnostics.csv`;
- mask coverage and boundary stability;
- held-out RGB/alpha metrics;
- thin structures, holes, and occlusion boundaries;
- TSDF support and topology diagnostics;
- connected-component coverage and non-manifold defects;
- texture seams and albedo consistency;
- finite exported model values and successful format round trips.

Historical performance and dataset-specific experiments should live under
`artifacts/` rather than in this architecture specification.
