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
extractors:  siftgpu (default) | sift | superpoint | disk | aliked
matchers:    gpu_mutual_ratio (default) | mutual_ratio | lightglue | hybrid_lightglue
pipelines:   none (default) | lightglue_end2end
```

Default behavior is unchanged: `siftgpu` × `gpu_mutual_ratio`
(`--matcher siftgpu` is a legacy alias for `gpu_mutual_ratio`).

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
