# Feature backends

AetherScan treats detection and matching as swappable backends behind stable
interfaces so product code can add SuperPoint, DISK, ALIKED, SuperGlue, etc.
without touching the SfM mapper.

## Interfaces

| Interface | Header | Role |
|-----------|--------|------|
| `FeatureExtractor` | `features/extractor.hpp` | image → `FeatureSet` |
| `FeatureMatcher` | `features/matcher.hpp` | two `FeatureSet`s → `MatchSet` |
| `PairFeaturePipeline` | `features/matcher.hpp` | fused pair pipelines (LightGlue) |
| Registry | `features/registry.hpp` | name → factory |

`FeatureSet` carries `metric` and `extractor_name` so matchers can validate
descriptor space (L2 / RootSIFT / inner-product).

## Built-in names

```text
extractors:  siftgpu (default) | sift | superpoint(stub)
matchers:    siftgpu (default) | mutual_ratio | lightglue
```

`lightglue` requires a build with `AETHERSCAN_ENABLE_ONNX=ON` (see root
`CMakeLists.txt` / `cmake/FetchOnnxRuntime.cmake`). It is a fused **image-pair**
pipeline (not a descriptor matcher):
`images[2,C,H,W] → keypoints, matches, mscores`. Constructed with an ONNX model
path at runtime.

```cpp
features::ensure_builtin_feature_backends();
auto extractor = features::create_extractor("siftgpu");
auto matcher = features::create_matcher("siftgpu");
```

Front-end selection:

```cpp
sfm::FrontEndOptions options;
options.extractor = "siftgpu";
options.matcher = "siftgpu";
// Fused LightGlue path:
options.matcher = "lightglue";
options.lightglue_model_path = "disk-lightglue.onnx";
options.lightglue_extractor = "disk";  // or "superpoint"
```

CLI:

```text
aetherscan --images ... --focal ... --mode incremental --output scene.mvs \
  --matcher lightglue --lightglue-model path/to/model.onnx \
  [--lightglue-extractor disk|superpoint] [--lightglue-width 1024] \
  [--lightglue-height 1024] [--lightglue-min-score 0] [--lightglue-cpu]
```

Because fused LightGlue re-detects keypoints per pair, the frontend merges
detections into stable per-image feature sets (1.5 px radius) before geometry
verification and track building. BoW retrieval is disabled on this path
(no descriptors).

## Adding SuperPoint (or any new extractor)

1. Implement `class SuperPointExtractor final : public FeatureExtractor` in
   `src/features/superpoint_extractor.cpp` (stub already exists).
2. Fill `extract_gray` / `extract_file` with ONNX/TensorRT inference.
3. Set `info().metric = DescriptorMetric::inner_product` (or cosine).
4. Keep `register_superpoint_feature_backends()` registering `"superpoint"`.
5. Prefer a dedicated matcher (`ip_mutual_ratio` / SuperGlue) if L2 ratio is wrong
   for the descriptor; register it with `register_matcher`.

```cpp
void register_superpoint_feature_backends() {
    register_extractor("superpoint", [] {
        SuperPointOptions options;
        options.model_path = ".../superpoint.onnx";
        return std::make_unique<SuperPointExtractor>(options);
    });
}
```

## Adding a fused pair model (LightGlue-style)

`LightGluePipeline` already implements `PairFeaturePipeline`. Product code
builds it with options (model path is required at construction):

```cpp
features::LightGlueOptions options;
options.model_path = "disk-lightglue.onnx";
options.extractor = features::LightGlueExtractor::disk;
auto pipeline = std::make_unique<features::LightGluePipeline>(options);
auto result = pipeline->match_files(image0, image1);
```
