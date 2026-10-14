# Photara SfM architecture

This document describes the current `Photara::SfM` implementation in
`photara/include/sfm` and `photara/src/sfm`: the frontend, mapping modes,
bundle adjustment, camera models, exports, checkpoints, and quality gates.
See [MVS architecture](MVS_ARCHITECTURE.md),
[Splat C++ backend](SPLAT_CPP.md), and [Pipeline overview](PIPELINE.md) for
downstream stages.
Key experiment findings and priorities for further improvement are recorded
in [SfM improvement notes](SFM_VALIDATION.md).

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
| `siftgpu` (default) | Native CUDA SIFT; `--sift-contrast` controls the peak threshold |
| `sift` | Native CPU SIFT/RootSIFT (same pipeline as CUDA / Vulkan) |
| `superpoint`, `disk`, `aliked` | ONNX extractors configured by `--extractor-model` |

`--max-features` defaults to 27,000. Descriptors are compressed to `uint8` by
default. The CUDA extractor uses a thread-affine coordinator; other extractors
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
ordered-frame tracking, and `--sam-max-size=0` uses the checkpoint's native
inference resolution (the default). Existing masks are reused unless
`--sam-refresh` is set.

SAM support is compiled from `third_party/sam3` with ggml. Builds always have
the CPU backend, add CUDA when the CUDA toolkit is available, and add Vulkan
when a Vulkan SDK with `glslc` is available. `--sam-backend` accepts `auto`,
`cpu`, `cuda`, `vulkan`, or `metal`; `auto` prefers CUDA, then Vulkan/Metal,
and finally CPU. An explicitly requested unavailable backend is an error and
never silently falls back.
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

CLI, Studio, and default Python/C++ frontend configurations use automatic camera
selection, a sequential window of 3, and retrieval augmentation. Explicit
`neighbor_window=0` / `--window 0` selects automatic pair selection; CLI retrieval
augmentation can be disabled with `--retrieval-augmentation=false`.
`ReconstructionConfig.global_rotation` and `global_positioning` apply to both
Global and Hierarchical, including single-cluster fast paths and cluster retries.
Direct `run_hierarchical_mapping` callers configure these in `HierarchicalConfig`.

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

### 4.2 `hierarchical`

The scene is split by weighted covisibility. A single cluster uses the Global
fast path; the default capacity is 2048 views. Multiple clusters use Star
initialization and PnP registration, followed by Sim(3) RANSAC, Umeyama
refinement and protected track merging.
Before extracting multiple clusters, physical camera identities and measured
focal lengths partition the parent calibration groups. Local refinements then
merge into those groups instead of collapsing distinct lenses into one camera.

A failed local initialization, or coverage below 98%, triggers a cluster-local
Global retry with the original intrinsics. A supported candidate replaces a
failed seed or a valid seed with fewer registered views. Retry candidates pass
the final observation-support audit and retain the caller's rotation and
positioning options. Logs identify both attempts and accepted results.
After merged BA and densification, a bounded polish reconciles the newly added
observations before the final audit. Its budget uses
`resection.final_ba_additional_iterations` (60 by default; zero disables it).
The polish restores its input poses, calibration, tracks and track index if BA
fails or introduces a new rotation conflict supported by two distinct strong,
nonplanar, cycle-consistent neighbors. Existing conflicts remain eligible for
the independent pose recovery below; a smaller pixel cost alone does not justify
introducing a new verified graph conflict.

After a multi-cluster merge, independent global rotation averaging screens
camera poses against a dominant common rotation gauge. A rotation discrepancy
only nominates a candidate: replacement poses require PnP against stable-only
depths, held-out pixel validation and verified baseline checks. Validated
repairs can anchor the next recovery pass; rejected candidates retain their
geometry. The audit copies only weighted edges, not feature-match arrays.
After all registration and weak-branch recovery, multi-cluster results also
exclude poses still contradicted by a dominant weight of strong, verified
neighbors. Their observations leave the inlier prefix before summary/export.
The CLI and Python `run_sfm_mapping` share `reconstruct_from_frontend`, including
camera metadata, mapping options, recovery and final support/rotation audits.

All modes finish with a `ReconstructionSummary`, observability analysis, and
`prune_unsupported_registrations`.

The complete reconstruction also screens position support using strong,
nonplanar pairs with measured bearing parallax, without optimized depths.
Views outside the largest bridge-free block are marked `unverified`; this
conservative screen preserves geometry and registration. Summary counts, CLI
warnings and diagnostic CSV expose the status. ASFM v4 stores it; files with
unverified registered poses require a v4 reader. Graph support and `valid`
describe available evidence and usability, not guaranteed position accuracy.

Reprojection summaries, support audits and recovery validation use the camera
model's checked residual. Equirectangular cameras include the rear hemisphere
with a seam- and pole-safe tangent residual in pixels. An oriented-ray check
rejects antipodal observations, whose tangent residual alone can be zero.

## 5. Bundle adjustment

`Photara::BA` provides Levenberg-Marquardt optimization with analytic
Jacobians, Huber IRLS, Schur elimination, matrix-free Schur products,
block-preconditioned PCG, gauge fixing, and independent intrinsic groups.

`CudaOptimizer` and `VulkanOptimizer` keep linearization, Schur assembly,
preconditioning, PCG, state updates, and cost evaluation on the GPU. The
Vulkan matrix-free Schur product uses observation-parallel point accumulation
and workgroup-parallel camera/intrinsic reduction to avoid global atomic
contention. `sfm::run_bundle_adjustment` builds full and local problems,
freezes poorly observable intrinsic groups, and falls back to CPU if the
requested GPU backend cannot solve the problem.

`--ba-backend automatic|cpu|cuda|vulkan` selects the process-level preference;
automatic uses CUDA when a device is present, otherwise Vulkan, then CPU.
Global positioning has the matching `--positioning-backend automatic|cpu|cuda|vulkan`
selector. CUDA and
Vulkan share the bearing-Schur formulation, robust IRLS passes, gauge
constraints, and stopping rules so backend comparisons preserve the mapping
semantics.

Camera warm-start selection remains internal to CLI and Studio with the existing
512 registered-camera default. Research comparisons use the dedicated positioning
benchmark and existing low-level library fields, not new product switches.
Before accepting the warm-start bearing refinement, compare its geometry to the
camera initializer using unweighted original relative-translation directions,
including the worst per-camera mean angle. Independently sample up to four tracks
per registered camera (minimum sample target 256, cap 4096 tracks)
excluded from positioning (up to 32 observations each), fit two separated rays
and evaluate observations excluded from each two-ray fit. Alternate pairs also
validate the two anchors, so endpoint cameras retain independent scores. No optimized point coordinates,
IRLS weights or external reference enter this screen. Behind-camera and missing
predictions remain in the score. Reject nonfinite/collapsed centers, changed
validation coverage, a positive-depth drop greater than 2%, or held-out angular
degradation greater than the larger of half a degree and 25% of the initializer's
error. Check per-camera means as well as whole-scene P90. Two-view translation
disagreement alone need not veto an independently supported refinement. With
at least 16 eligible held-out tracks, reject observed independent regressions;
direction changes at cameras with fewer than three held-out predictions are
logged as unverified rather than triggering an expensive full retry without
evidence of degradation. Without sufficient held-out tracks, retain the
conservative relative-direction screen. Missing evidence is logged and
does not establish small-scene eligibility. Rejected refinements use the existing
primary fallback; state restoration also restores track inlier metadata. Cache
fingerprints invalidate pre-screening reconstructions. These checks detect
material regressions, not absolute position accuracy; the 512 default is retained
pending cross-scene validation.

CPU positioning reports IRLS convergence, iteration limit, time limit, stall
or failure separately from solution usability. Evaluation iteration zero is
excluded from accepted updates. A pass without an accepted update cannot
change robust weights unless Ceres genuinely converged. Subsequent CPU slices
reserve measured preprocessing and an attempted step, within the shared budget;
time-limited passes cannot report IRLS convergence from stable weights alone.
An unchanged timed-out pass retries with a larger slice only if the measured
step fits the remaining shared budget; its robust weights remain unchanged.
Ceres checks deadlines between steps, so actual wall time can exceed a slice.
Rotation averaging scopes OpenMP to one thread for its small sparse CG products
and restores the caller's thread setting before positioning and BA.

## 6. Outputs

Studio's photo-alignment compute preference selects both positioning and BA.
Advanced CLI and Python/C++ callers can still choose their backends separately.
Studio displays position-risk counts and per-camera status from native scenes
and diagnostic CSV, with warning markers on flagged image thumbnails. Imported
or older scenes without audit metadata remain explicitly unaudited.

New Studio projects reuse compatible alignment caches by default; saved projects
retain their choice. Turning reuse off sends `--cache-dir -` and recomputes
alignment stages, while working cameras remain available for training/preview.
CLI and library caching requires an explicit directory. Image fingerprints,
parameters, and implementation versions invalidate affected cached stages;
final recovery and position audits still run on reconstruction-cache hits.

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
- Keys include build identity, frontend parameters, source images, masks, and
  the selected positioning/BA backend, so A/B runs do not reuse each other's
  reconstruction checkpoint.
- `--cache-dir` enables reuse; a tracks hit skips the entire frontend.
- `--working-sfm` writes editor sidecars: `*.live` and periodic
  `*.preview.asfm` snapshots.
- `*_sfm_diagnostics.csv` reports intrinsics, camera centers, quaternions,
  observations, reprojection statistics, pose deltas, reliability, pair
  weights, ray angles, and camera models.

## 8. Common options

| Option | Default | Description |
|---|---|---|
| `--mode` | `global` | `global` or `hierarchical` |
| `--window` | 3 | Sequential pair window |
| `--max-features` | 27,000 | Limit after 3×3 grid selection |
| `--extractor` × `--matcher` | `siftgpu` × `gpu_mutual_ratio` | Default frontend |
| `--masks` | `auto` | Sibling `masks/`; `-` disables discovery |
| `--match-ratio` | 0.8 | Descriptor ratio test |
| `--camera-model` | `auto` | See section 3.5 |
| `--focal` | 0 | Automatic initial focal |
| `--trust-focal` | false | Lock supplied intrinsics |
| `--positioning-backend` | `automatic` | May be forced to `cpu`, `cuda`, or `vulkan` |
| `--ba-backend` | `automatic` | May be forced to `cpu`, `cuda`, or `vulkan` |
| `--cache-dir` | empty | Enable checkpoints |

## 9. Tests

SfM CTests cover two-view geometry, equirectangular residuals, PnP registration and
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
