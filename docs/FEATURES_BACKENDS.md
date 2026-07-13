# Feature backends

AetherScan treats detection and matching as swappable backends behind stable
interfaces so product code can add SuperPoint, DISK, ALIKED, SuperGlue, etc.
without touching the SfM mapper.

## Model

```text
Default (composable):   --extractor × --matcher
Optional fused recipe:  --pipeline <name>   (mutually exclusive with composition)
```

| Interface | Header | Role |
|-----------|--------|------|
| `FeatureExtractor` | `features/extractor.hpp` | image → `FeatureSet` |
| `FeatureMatcher` | `features/matcher.hpp` | two `FeatureSet`s → `MatchSet` |
| `PairFeaturePipeline` | `features/matcher.hpp` | fused pair recipes (end2end LightGlue) |
| Compat | `features/compat.hpp` | extractor×matcher compatibility checks |
| Registry | `features/registry.hpp` | name → factory |

`FeatureSet` carries `metric` and `extractor_name` so matchers can validate
descriptor space (L2 / RootSIFT / inner-product).

## Built-in names

```text
extractors:  siftgpu (default) | sift | superpoint(stub)
matchers:    gpu_mutual_ratio (default) | mutual_ratio
             legacy alias: siftgpu → gpu_mutual_ratio
pipelines:   none (default) | lightglue_end2end
```

**Default algorithm is unchanged in behavior:** `extractor=siftgpu` ×
`matcher=gpu_mutual_ratio` (GPU SIFT extract + GPU descriptor mutual-ratio).
`pipeline` empty.

### Composable path

```cpp
features::ensure_builtin_feature_backends();
auto extractor = features::create_extractor("siftgpu");
auto matcher = features::create_matcher("gpu_mutual_ratio");
```

```cpp
sfm::FrontEndOptions options;
options.extractor = "siftgpu";
options.matcher = "gpu_mutual_ratio";
// options.pipeline left empty
```

Valid combinations today:

| extractor | matcher | notes |
|-----------|---------|-------|
| `siftgpu` | `gpu_mutual_ratio` | **default** (SiftMatchGPU) |
| `siftgpu` | `mutual_ratio` | CPU mutual-ratio on SiftGPU descriptors |
| `sift` | `mutual_ratio` | VLFeat SIFT |
| `sift` | `gpu_mutual_ratio` | allowed if descriptors are L2/RootSIFT |

Incompatible combos (e.g. future `superpoint` × `gpu_mutual_ratio`) are rejected by
`features::validate_extractor_matcher_combo`.
### Fused pair pipeline

`lightglue_end2end` is a **PairFeaturePipeline**, not an extract×matcher pair.
It re-detects features per image pair via a fused ONNX model
(`images → keypoints, matches, mscores`). Requires `AETHERSCAN_ENABLE_ONNX=ON`.

```cpp
options.pipeline = "lightglue_end2end";
options.lightglue_model_path = "disk-lightglue.onnx";
options.lightglue_extractor = "disk";  // or "superpoint"
// extractor / matcher are ignored for matching (BoW retrieval may still use SiftGPU)
```

CLI:

```text
aetherscan --images ... --mode incremental --output scene.mvs \
  --pipeline lightglue_end2end --lightglue-model path/to/model.onnx \
  [--lightglue-extractor disk|superpoint] [--lightglue-width 1024] \
  [--lightglue-height 1024] [--lightglue-min-score 0] [--lightglue-cpu]
```

Legacy alias (still works): `--matcher lightglue` normalizes to
`--pipeline lightglue_end2end`.

Because fused LightGlue re-detects keypoints per pair, the frontend merges
detections into stable per-image feature sets (1.5 px radius) before geometry
verification and track building.

## Adding SuperPoint (or any new extractor)

1. Implement `class SuperPointExtractor final : public FeatureExtractor`.
2. Fill `extract_gray` / `extract_file` with ONNX/TensorRT inference.
3. Set `info().metric = DescriptorMetric::inner_product`.
4. Register `"superpoint"` and extend `features/compat.hpp` so only matching
   matchers (e.g. future `lightglue` descriptor matcher) accept it.
5. Prefer a dedicated `FeatureMatcher`; do **not** special-case the frontend.

## Adding a descriptor LightGlue matcher (future)

Implement `LightGlueMatcher : public FeatureMatcher` that takes two
`FeatureSet`s (keypoints + descriptors). Register as `"lightglue"` matcher
and allow combos like `disk × lightglue` / `superpoint × lightglue` in
`compat.hpp`. Keep `lightglue_end2end` as the fused recipe name.

## Adding a fused pair model

`LightGluePipeline` implements `PairFeaturePipeline` (`name() ==
lightglue_end2end`). Product code builds it with options (model path required):

```cpp
features::LightGlueOptions options;
options.model_path = "disk-lightglue.onnx";
options.extractor = features::LightGlueExtractor::disk;
auto pipeline = std::make_unique<features::LightGluePipeline>(options);
auto result = pipeline->match_files(image0, image1);
```
