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
Parameters map one-to-one (`--max-features`, `--sift-contrast`,
`--match-ratio`, `--no-mutual-check`).

When CUDA is unavailable but a Vulkan compute device is available, the
frontend canonicalizes the default `siftgpu` × `gpu_mutual_ratio` selection
to `vulkan_sift` × `vulkan_mutual_ratio`, so reconstruction still runs.
Checkpoints record the canonicalized backend names.

The Vulkan matcher keeps the CUDA matcher's integer top-two reductions exact;
accepted matches can differ only at `acos` rounding boundaries. On the pairs
measured by `photara_vulkan_features_parity` it returns *identical* match sets
to the native CUDA matcher.

### Current SIFT implementation and parity

The native CUDA engine (`src/features/sift_cuda.cu`) and Vulkan extractor
(`src/features/vulkan/sift_vk.cpp` and its shaders) implement the same Gaussian
pyramid, DoG extrema tests, partial-pivot sub-pixel solve, multi-orientation
split, and UBC/RootSIFT descriptors. Default options agree, including
`peak_threshold=0.005`, `edge_threshold=20`, `first_octave=-1`, three octave
layers, two orientations, and RootSIFT enabled. Floating-point evaluation
across CUDA and SPIR-V need not be bit-identical.

Keypoint collection differs: CUDA compacts detections with an atomic counter,
downloads them, sorts by integer y/x coordinates on the CPU, and uploads the
sorted list. Vulkan uses histogram-pyramid compaction. Both apply the same
scale-level feature-budget selection. One API edge case remains different:
Vulkan accepts `maximum_orientations=0` for upright descriptors, whereas the
CUDA constructor rejects zero; the shared default of two is unaffected.

The textured-image fixture in `tests/vulkan_features_test.cpp` currently
requires equal feature counts and descriptor sizes, at least 99% indexed
keypoint agreement within 0.01 pixels per coordinate and 2% relative scale,
and mean descriptor cosine similarity of at least 0.999. It also requires
two CUDA extractions of that fixture to reproduce keypoints and descriptors
exactly. These are fixture-specific test requirements, not guarantees for
every image or GPU. `photara_vulkan_features_parity` reports comparisons on
synthetic inputs or an image directory.

Both current backends explicitly return zero for out-of-range linear reads
in the 2x upsampling pass. Reports of undefined border reads and nondeterminism
in the former external SiftGPU implementation do not describe the current
native CUDA engine. Both retain input-width truncation to a multiple of four
for ordinary images, so up to three trailing columns do not contribute.

### Memory optimizations and performance evidence

The CUDA extractor uses shared-memory separable Gaussian filtering, reusable
device allocations, two rolling Gaussian buffers and three rolling DoG buffers.
For descriptor generation it rebuilds the pyramid rather than retaining all
level gradients, and reuses inactive DoG buffers for gradient magnitude and
orientation. This trades recomputation for lower peak device memory.

Optimization commit `f814881` records a Gingy extractor VRAM delta reduction
from 1736 MiB to 1368 MiB, 65.5% below the original native implementation's
3962 MiB. The commit records bit-identical keypoints and quantized descriptors,
deterministic extraction on Gingy and ori_img, CUDA/Vulkan parity checks, and
a full Gingy SfM validation. These are historical validation results for that
commit and dataset, not a new benchmark of the current checkout.

Vulkan now uses three rolling Gaussian and three rolling DoG buffers, one
keypoint image and one gradient image. It downloads compact oriented keypoints
and rebuilds Gaussian levels for descriptors, preserving the shader math and
feature selection order. Extractor clones share a serialized, reusable GPU
workspace; CPU image decoding can still run across workers. Matching releases
that workspace and uses a shared descriptor cache across matcher clones.
Idle device/staging buffer pools are bounded at 256/128 MiB and two buffers
per size bucket. Host-visible allocations prefer non-device-local memory.

Current CUDA extraction still includes per-level host sorting, transfers and
synchronization. The frontend uses a thread-affine CUDA coordinator; Vulkan
uses the shared workspace and batches matching through the GPU coordinator.
Single-image latency and multi-image throughput therefore need separate
measurements. CUDA matcher's tensor-core acceleration concerns matching, not
SIFT extraction.

An earlier 72-frame scan experiment recorded extraction at 1.16 s on Vulkan
and 2.30 s on the then-current CUDA path, with matching at 42.6 s and 1.34 s
respectively for 1978 pairs. It predates the native CUDA replacement
(`4db48d1`) and subsequent memory optimization. Those timings do not rank the
current backends, and the old scalar-only Vulkan matcher description is no
longer current.

The [2026-10-21 indoor dataset comparison](SFM_CUDA_VULKAN_IPHONE_COMPARISON.md)
records the full pipeline before the Vulkan memory changes on 739 images, with Blender paused
and both configurations limited to four frontend workers. CUDA extraction took
26.284 s versus 29.359 s on Vulkan; complete SfM took 343.746 s versus
425.810 s. Vulkan scored better on the common held-out reference observations
(mean 1.661 px versus 2.694 px). These are single-run, dataset-specific results;
they do not isolate SIFT's contribution to the final quality difference.
The [Vulkan memory optimization record](SFM_VULKAN_MEMORY_OPTIMIZATION.md)
tracks the subsequent implementation and validation separately.

To measure the current checkout, build the tools and run, for example:

```powershell
photara_vulkan_features_parity IMAGE_DIRECTORY
photara_cuda_sift_benchmark IMAGE_DIRECTORY 2 27000
```

The parity harness feeds both backends the same decoded pixels and reports
extraction separately from matching. The CUDA benchmark checks repeated
extraction and reports timing and device-memory usage. Keep image dimensions,
feature budgets, hardware and build configuration consistent, and separately
measure frontend worker scheduling for batch-throughput comparisons. Debug
logs expose CUDA wait, transfer and CPU sorting times; trace logs add per-level
timings. Documentation corrections alone do not rerun these benchmarks.

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
