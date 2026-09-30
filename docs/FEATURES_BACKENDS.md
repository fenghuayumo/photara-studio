# Feature backends

Photara treats **extraction** and **matching** as independent, swappable
backends. Mix any compatible pair; do not treat SuperPoint+LightGlue as one
bundled algorithm.

## Model

```text
Default:   --extractor × --matcher
Optional:  --pipeline <fused recipe>   (mutually exclusive with composition)
```

| Interface | Role |
|-----------|------|
| `FeatureExtractor` | image → `FeatureSet` |
| `FeatureMatcher` | two `FeatureSet`s → `MatchSet` |
| `PairFeaturePipeline` | fused end2end recipes only |
| `compat.hpp` | which extractor×matcher combos are valid |

## Built-in names

```text
extractors:  siftgpu (default) | vulkan_sift | sift | superpoint | disk | aliked
matchers:    gpu_mutual_ratio (default) | vulkan_mutual_ratio | mutual_ratio | lightglue | hybrid_lightglue
pipelines:   none (default) | lightglue_end2end
```

Default behavior is unchanged: `siftgpu` × `gpu_mutual_ratio`
(`--matcher siftgpu` is a legacy alias for `gpu_mutual_ratio`).

## Vulkan translation of the default path

`vulkan_sift` × `vulkan_mutual_ratio` is a Vulkan compute (HLSL/SPIR-V)
translation of the same algorithms the default path runs on CUDA. The default
`siftgpu` extractor is Photara's CUDA implementation of that pipeline
(Gaussian pyramid with 2x upsampling, DoG, sub-pixel extrema, multi-orientation,
UBC descriptors). It does not link an external SiftGPU library. Matching stays
on photara's CUDA mutual-ratio matcher (32x32 descriptor dot-product tiles,
angular distance `acos(dot / 262144) < 0.7`, ratio test, mutual check).
Parameters map one-to-one (`--max-features`, `--sift-contrast-threshold`,
`--match-ratio`, `--no-mutual-check`).

When CUDA is unavailable but a Vulkan compute device is available, the
frontend canonicalizes the default `siftgpu` × `gpu_mutual_ratio` selection
to `vulkan_sift` × `vulkan_mutual_ratio`, so reconstruction still runs.
Checkpoints record the canonicalized backend names.

The Vulkan matcher keeps the CUDA matcher's integer top-two reductions exact;
accepted matches can differ only at `acos` rounding boundaries. On the pairs
measured by `photara_vulkan_features_parity` it returns *identical* match sets
to the native CUDA matcher.

The Vulkan extractor reproduces the SiftGPU CUDA pipeline step for step: the
same Gaussian pyramid (2x bilinear upsampling, truncated-width input rows,
separably filtered levels), the same DoG extrema test with the partial-pivot
sub-pixel solve, the same histogram-pyramid keypoint compaction, the same
multi-orientation split and UBC/RootSIFT descriptors. On identical inputs the
two backends return the same feature counts and the same keypoint positions
(1 px / 2% scale agreement is 97-100% with a mean offset of ~0.0002 px; the
`photara.features.vulkan` test asserts count parity plus 80%+ position
agreement, and `photara_vulkan_features_parity` prints the full comparison).

End-to-end agreement was checked on a 72-frame 1080x1920 scan video
(`photara --images ... --max-features 12000 --window 3 --export-colmap ...`).
The default CUDA pair and `vulkan_sift` × `vulkan_mutual_ratio` both register
72/72 images with 100% of the reconstructed points in front of their cameras,
and after a similarity alignment their camera centres agree with the CUDA
baseline to 0.007-0.062% of the trajectory extent with rotations within
0.05 degrees (the residual is the usual SIFT feature jitter, not a systematic
difference).

Two Vulkan-side defects surfaced on that run and are fixed here:

- descriptor sets came from one 1024-set pool shared by every worker, so
  parallel extraction, which records a few hundred dispatches per image,
  exhausted it. Each recording session now owns growable pools that are reset
  between submissions;
- parallel extraction of 1080x1920 frames (one worker per thread, each holding
  ~600 MB of pyramid buffers) failed with `VK_ERROR_OUT_OF_DEVICE_MEMORY`.
  Extractions now reserve their estimated bytes against a device-memory budget
  before allocating, and the matcher's descriptor mirror grows on demand instead
  of reserving its whole 768 MB budget per clone.

The CUDA path is deliberately left as it was: the baseline `siftgpu` ×
`gpu_mutual_ratio` behaviour is the reference these numbers are compared
against, and the measurements below show nothing on the CUDA side that the
dataset justifies changing.

Performance on that dataset: extraction is 2x faster on Vulkan (1.16 s vs
2.30 s) because whole images are extracted in parallel, while matching is much
faster on CUDA (1.34 s vs 42.6 s for 1978 pairs) because the native matcher uses
int8 tensor cores while the Vulkan shader evaluates byte dot products scalar.
That dot product is the remaining gap for large feature counts; small feature
counts are dominated by host round trips instead, where the two are within a
factor of two (`photara_vulkan_features_parity` reports both). Everything else
on the CUDA side is untouched: reverting the intermediate experiment reproduces
the baseline frontend statistics exactly (features 880167, raw matches 291303,
tracks 83917, 72/72 registered, reprojection 0.413002 px).

Two caveats:

- SiftGPU's upsampling pass indexes slightly past its own textures, so its
  pyramid border reads undefined memory and its output is *not* run-to-run
  reproducible: on one real photo pair `photara --images` reported 7877-8296
  features and 317-489 raw matches across three runs, while the Vulkan backend
  reported the same 8526 features / 616 matches / 594 inliers every time. The
  Vulkan translation returns the zero that tex1Dfetch documents for those
  reads, matches the CUDA result on reproducible inputs, and is deterministic.
- SiftGPU truncates the input width to a multiple of four before building the
  pyramid (`TruncateWidthCU`); the Vulkan backend does the same, so up to three
  trailing columns never contribute.

## SfM valid-region masks

The composable frontend accepts optional per-image valid-region masks through
`--masks <directory>` or `FrontEndOptions::mask_dir`. The CLI default,
`--masks auto`, uses a sibling `masks/` directory next to the image directory
when it exists. Pass `--masks -` to disable mask discovery.

Mask files are resolved by exact image filename first, then by image stem with
one of these extensions: `.png`, `.jpg`, or `.jpeg` (lowercase and uppercase
forms are supported). A missing mask does not abort reconstruction: Photara
emits a warning and processes that image without an SfM mask.

SfM uses the following mask convention:

- pixel value `0` is invalid and is excluded;
- every non-zero value is valid and is retained;
- mask and source-image resolutions may differ; keypoint centers are mapped
  proportionally into mask coordinates;
- SfM does not fall back to the source image alpha channel. Provide a separate
  mask file when camera alignment must be masked.

Extraction still runs on the source image. Immediately after extraction,
Photara removes masked keypoints and their matching descriptor rows before
3×3 grid selection, descriptor compression, BoW retrieval, descriptor
matching, geometric verification, and track construction. The weak-view
SiftGPU augmentation pass applies the same filter before appending new
features. Consequently, masked features never reach the composable matcher,
although the detector's GPU workload is not reduced.

All current extractor/matcher compositions use the filtered `FeatureSet` and
therefore support SfM masks:

| extractor | mask-aware matchers |
|-----------|---------------------|
| `siftgpu` | `gpu_mutual_ratio`, `mutual_ratio`, `lightglue`, `hybrid_lightglue` |
| `vulkan_sift` | `vulkan_mutual_ratio`, `mutual_ratio`, `lightglue` |
| `sift` | `gpu_mutual_ratio`, `mutual_ratio`, `lightglue` |
| `superpoint`, `disk`, `aliked` | `lightglue` |

The fused `--pipeline lightglue_end2end` path is not mask-aware yet. Its
temporary SiftGPU retrieval features are filtered, but its final pair-wise
LightGlue detections and matches are not. Use the default composable path when
masked camera alignment is required.

Resolved mask paths and mask file contents contribute to the feature-stage
checkpoint key. Adding, removing, renaming, or editing a mask invalidates the
dependent feature, match, geometry, and track checkpoints.

For reliable binary segmentation, prefer lossless PNG masks. Because SfM
currently treats every non-zero pixel as valid, JPEG compression noise and
small non-zero matte values can retain pixels that appear visually black.

### Compose freely

```text
# Classic
--extractor siftgpu --matcher gpu_mutual_ratio

# Learned extract + learned match
--extractor superpoint --extractor-model superpoint.onnx \
--matcher lightglue --lightglue-model superpoint_lightglue_fused.onnx

--extractor disk --extractor-model disk.onnx \
--matcher lightglue --lightglue-model disk_lightglue_fused.onnx

--extractor aliked --extractor-model aliked-n16rot.onnx \
--matcher lightglue --lightglue-model aliked-lightglue.onnx

# SIFT and SiftGPU use their existing extractor; only the matcher needs a model.
--extractor siftgpu \
--matcher lightglue --lightglue-model sift-lightglue.onnx

# Quality/performance mode: fast SiftGPU matching first, then LightGlue only
# for nearby or weakly connected pairs that failed geometric verification.
--extractor siftgpu --matcher hybrid_lightglue \
--lightglue-model sift-lightglue.onnx --lightglue-min-score 0.1 \
--hybrid-lightglue-max-features 2048
```

Extractor knobs (`--extractor-model/width/height/cpu`) belong to the extractor.
Matcher knobs (`--lightglue-model/min-score/cpu`) belong to the LightGlue
matcher (or the optional end2end pipeline). They are not a joint “package”.
`--hybrid-lightglue-max-features` bounds only the selective rescue pass; the
primary SiftGPU path still uses `--max-features`.

Compatibility today (extend in `compat.hpp`):

| extractor | matcher |
|-----------|---------|
| siftgpu | gpu_mutual_ratio / mutual_ratio / lightglue / hybrid_lightglue |
| vulkan_sift | vulkan_mutual_ratio / mutual_ratio / lightglue |
| sift | gpu_mutual_ratio / mutual_ratio / lightglue |
| superpoint / disk / aliked | lightglue |

### Optional fused recipe

`--pipeline lightglue_end2end` re-detects per pair (slower). Prefer composition,
especially when SfM masks are enabled.

## Adding a backend

1. Implement extractor or matcher.
2. Register / construct from `run_frontend`.
3. Update `compat.hpp` for allowed pairs.
4. Keep CLI options namespaced to that backend.
