# Photara SfM architecture

This document describes the current `Photara::SfM` implementation in
`photara/include/sfm` and `photara/src/sfm`: the frontend, mapping modes,
bundle adjustment, camera models, exports, checkpoints, and quality gates.
See [MVS architecture](MVS_ARCHITECTURE.md),
[Splat C++ backend](SPLAT_CPP.md), and [Pipeline overview](PIPELINE.md) for
downstream stages.

## 1. Scope

- Input is an image directory or video frames.
- Output is a set of registered camera poses, sparse landmarks, and
  observations.
- The default build does not require OpenCV. Image I/O uses FreeImage,
  libjpeg, and libpng; geometry uses Eigen.
- The default frontend is `siftgpu × gpu_mutual_ratio`. ONNX-backed
  LightGlue, SuperPoint, DISK, and ALIKED are optional alternatives.
- SfM produces geometry and track colors. It does not produce dense depth.
  The CLI can optionally generate foreground masks with SAM 3 before it calls
  the SfM library.

## 2. Data model

`sfm::Scene` is the single reconstruction-state container:

```text
Scene
  cameras[]      intrinsics, distortion, camera model, focal prior
  images[]       path, features, camera identity, pose, registered flag
  pairs[]        verified matches, relative pose, E/F/H, pair weights
  tracks[]       3D position, observations, color, inlier count
  image_tracks[] image_id -> observed tracks
  resection_progress / registration_generation / thread_count
```

The projection convention is `P = K R [I | -C]`, where `R` maps world to
camera and `C` is the camera center in world coordinates. `Pose3D` provides
world-to-camera and camera-to-world transforms, composition, and inversion.
Equirectangular residuals and Jacobians are evaluated in a local tangent plane.

## 3. Frontend

Entry point: `sfm::run_frontend` in `src/sfm/frontend.cpp`, logged as stage
`sfm.frontend`.

```text
1. Fingerprint source images and masks
2. Extract features
3. Propose pairs: sequential window + BoW retrieval
4. Match descriptors
5. Verify geometry with E/F/H RANSAC
6. Compute pair weights, refresh relative poses, and build tracks
```

### 3.1 Feature extraction

| Backend | Description |
|---|---|
| `siftgpu` (default) | CUDA SiftGPU; `--sift-contrast` controls the peak threshold |
| `sift` | Built-in VLFeat SIFT/RootSIFT fallback |
| `superpoint`, `disk`, `aliked` | ONNX extractors configured by `--extractor-model` |

`--max-features` defaults to 27,000. A 3×3 grid selection prevents features
from concentrating in a small image region. Descriptors are compressed to
`uint8` by default. SiftGPU uses a thread-affine coordinator; other extractors
are cloned per worker when required.

#### SfM valid-region masks

`--masks <directory>` and `FrontEndOptions::mask_dir` provide optional
per-image masks. `--masks auto` uses a sibling `masks/` directory when it
exists; `--masks -` disables discovery.

Masks are matched by exact image filename first, then by image stem with
`.png`, `.jpg`, or `.jpeg` in lowercase or uppercase form. Zero is invalid;
every non-zero value is valid. Keypoint centers are mapped proportionally when
mask and source resolutions differ.

Extraction still runs on the source image. Photara then removes invalid
keypoints and their descriptor rows before grid selection, descriptor
compression, BoW retrieval, matching, geometric verification, and track
construction. Weak-view SiftGPU augmentation applies the same filter.

Every composable extractor/matcher combination consumes the filtered
`FeatureSet`, including the default `siftgpu × gpu_mutual_ratio` path. The
fused `lightglue_end2end` pipeline does not yet filter its final pair-wise
detections and matches and must not be used for masked alignment.

SfM requires separate mask files and does not fall back to source-image alpha.
Missing masks produce a warning and those images are processed without a mask.
Mask paths and contents are included in the feature checkpoint fingerprint.
Prefer lossless binary PNG masks because the non-zero rule can retain JPEG
compression noise or small soft-matte values.

See [Feature backends](FEATURES_BACKENDS.md#sfm-valid-region-masks) for the
support matrix.

#### Optional SAM 3 mask generation

When Photara is built with `PHOTARA_ENABLE_SAM=ON`, a non-empty
`--sam-text` runs in-process SAM 3 before SfM and writes one
source-resolution `<image-stem>.png` keep-mask per image. The generated mask
directory becomes `--masks` for SfM and downstream stages. The default output
is the sibling `masks/` directory; an explicit `--masks` path overrides it.

`--sam-neg-text` removes named distractors, `--sam-keep-prompted=false`
interprets the main prompt as the content to remove, `--sam-video` enables
ordered-frame tracking, and `--sam-max-size` controls inference resolution.
Existing masks are reused unless `--sam-refresh` is set.

SAM support is compiled from `third_party/sam3` with ggml. CUDA is the GPU
backend when the toolkit is available; otherwise inference stays on CPU.
Model weights are not bundled. `--sam-model` selects a local GGML checkpoint;
otherwise Photara searches its user cache and `PHOTARA_SAM_MODEL`. The editor
requires explicit acceptance of Meta's SAM 3 license before downloading the
model.

### 3.2 Pair proposal

- The sequential window proposes `(i, i+k)` pairs. The CLI default is 3.
- For at least 50 images, hierarchical BoW retrieval can augment the
  sequential window. Defaults are `top_k=50`,
  `max_descriptors_per_image=2000`, `sample_grid=3`,
  `stop_word_ratio=0.5`, and `max_posting_images=64`.
- A learned vocabulary can be persisted as `<cache-dir>/vocabulary.bin`.
- Byte-identical pairs are marked as zero-baseline and use an identity
  constraint instead of ordinary two-view pose estimation.

### 3.3 Matching and geometry verification

| Matcher | Description |
|---|---|
| `gpu_mutual_ratio` (default; alias `siftgpu`) | CUDA mutual ratio matching; default ratio 0.8 |
| `mutual_ratio` | CPU/AVX2 mutual ratio matching |
| `lightglue` | ONNX descriptor matching |
| `hybrid_lightglue` | SiftGPU first, followed by selective LightGlue rescue |
| `--pipeline lightglue_end2end` | Fused pair-wise detection and matching |

`verify_pair_geometry` estimates essential, fundamental, and homography
models and records relative pose, `E/F/H`, mean ray angle, homography ratio,
and planar degeneracy. Planar or low-parallax pairs are down-weighted instead
of deleted so they may still bridge the view graph.

Weak-view augmentation is enabled by default for `gpu_mutual_ratio`. Images
with insufficient verified degree or fewer than three non-planar neighbors are
re-extracted with a larger feature budget and selectively rematched.
Structural pair expansion remains experimental and disabled by default.

### 3.4 Pair weights and tracks

`sfm::compute_pair_weights` combines the inlier count with spatial proximity,
geometry quality, endpoint connectivity, triplet support, and cycle
consistency. `build_tracks` merges observations above `min_pair_weight`.
`filter_tracks` applies reprojection, triangulation-angle, and depth-ratio
gates. `colour_triangulated_tracks` samples source images for Splat and MVS.

### 3.5 Camera models and intrinsics

| `--camera-model` | Model | Notes |
|---|---|---|
| `auto` (default) | `automatic` | Resolved for the scene |
| `pinhole` | `pinhole` | fx, fy, cx, cy + k1, k2, p1, p2 |
| `fisheye` / `opencv_fisheye` | `opencv_fisheye` | k1..k4 angular coefficients |
| `equirectangular` and aliases | `equirectangular` | fixed panorama intrinsics |

When `--focal` is zero, the initial focal is `1.2 * max(W,H)`, or
`0.5 * max(W,H)` for fisheye, followed by view-graph consensus and BA.
`--trust-focal` locks externally calibrated intrinsics. EXIF metadata can
provide focal and physical-camera grouping priors.

## 4. Mapping modes

Entry point: `sfm::reconstruct` in `src/sfm/reconstruct.cpp`.

### 4.1 `global` (default)

```text
rotation averaging: MST -> LAD/ADMM L1 -> robust IRLS
  -> reject inconsistent rotations and rebuild tracks
  -> global positioning with fixed rotations
  -> repair position outliers, triangulate, and filter tracks
  -> BA: structure/translation, then rotation/focal, then distortion
  -> optional orbit regularization for strict turntable sequences
  -> incremental resection fallback for unregistered views
  -> enforce zero-baseline pose groups and retriangulate
```

### 4.2 `incremental`

`star_initialize` creates a reference-centered seed with up to 36 views.
`register_images` performs wave-parallel PnP, periodic local and full BA, and a
final polish pass. If a large reconstruction stalls, Photara can roll back to
the frontend state and use hierarchical submap recovery.

### 4.3 `hierarchical`

The scene is split by weighted covisibility, each cluster is reconstructed
incrementally, submaps are aligned with Sim(3) RANSAC plus Umeyama refinement,
and protected tracks are merged before optional final BA.

All modes finish with a `ReconstructionSummary`, observability analysis, and
`prune_unsupported_registrations`.

## 5. Bundle adjustment

`Photara::BA` provides Levenberg-Marquardt optimization with analytic
Jacobians, Huber IRLS, Schur elimination, matrix-free Schur products,
block-preconditioned PCG, gauge fixing, and independent intrinsic groups.

`CudaOptimizer` keeps linearization, Schur assembly, preconditioning, PCG,
state updates, and cost evaluation on the GPU. `sfm::run_bundle_adjustment`
builds full and local problems, freezes poorly observable intrinsic groups,
and falls back to CPU if CUDA fails. `--ba-backend automatic|cpu|cuda` selects
the process-level preference.

## 6. Outputs

| API / CLI | Output |
|---|---|
| `save_asfm` / `.asfm` | Native scene with cameras, poses, keypoints, and tracks |
| `project::Archive` / `.ascan` | Project container with SfM and optional Gaussians/mesh |
| `export_openmvs_interface` / `.mvs` | OpenMVS cameras and colored landmarks |
| `save_sparse_ply` | Sparse XYZRGB PLY |
| `save_colmap_text` | COLMAP text model |
| `save_nerfstudio_transforms` | Nerfstudio/Blender `transforms.json` |

## 7. Checkpoints and diagnostics

- `CheckpointStore` supports `features`, `matches`, `geometry`, `tracks`, and
  `reconstruction` stages.
- Keys include build identity, frontend parameters, source images, and masks.
- `--cache-dir` enables reuse; a tracks hit skips the entire frontend.
- `--working-sfm` writes editor sidecars: `*.live` and periodic
  `*.preview.asfm` snapshots.
- `*_sfm_diagnostics.csv` reports intrinsics, camera centers, quaternions,
  observations, reprojection statistics, pose deltas, reliability, pair
  weights, ray angles, and camera models.

## 8. Common options

| Option | Default | Description |
|---|---|---|
| `--mode` | `global` | `global`, `incremental`, or `hierarchical` |
| `--window` | 3 | Sequential pair window |
| `--max-features` | 27,000 | Limit after 3×3 grid selection |
| `--extractor` × `--matcher` | `siftgpu` × `gpu_mutual_ratio` | Default frontend |
| `--masks` | `auto` | Sibling `masks/`; `-` disables discovery |
| `--match-ratio` | 0.8 | Descriptor ratio test |
| `--camera-model` | `auto` | See section 3.5 |
| `--focal` | 0 | Automatic initial focal |
| `--trust-focal` | false | Lock supplied intrinsics |
| `--ba-backend` | `automatic` | May be forced to `cpu` or `cuda` |
| `--cache-dir` | empty | Enable checkpoints |

## 9. Tests

SfM CTests cover two-view geometry, equirectangular residuals, incremental and
global mapping, vocabulary retrieval, checkpoint equivalence, submap recovery,
hierarchical Sim(3) merge, OpenMVS export, and project I/O. `photara.features`
also covers feature storage, CPU/CUDA matcher consistency, and mask filtering.

## 10. Known limitations

- Degenerate self-calibration can produce a low reprojection error but an
  incorrect trajectory. Inspect `*_sfm_diagnostics.csv` and
  `alignment_reliable` on real datasets.
- Structural pair expansion remains experimental.
- The SfM library does not generate dense depth or masks; optional SAM mask
  generation is CLI/editor preprocessing.
- `lightglue_end2end` is not yet mask-aware for final pair-wise matches.
