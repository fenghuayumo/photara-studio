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
matchers:    siftgpu (default) | mutual_ratio
```

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
// or inject prototypes:
options.extractor = std::make_shared<features::SiftExtractor>(sift_opts);
options.matcher = std::make_shared<features::MutualRatioMatcher>(match_opts);
```

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

Implement `PairFeaturePipeline` and either construct it directly with options or
bind a factory:

```cpp
register_pair_pipeline("lightglue", [options] {
    return std::make_unique<LightGluePipeline>(options);
});
```
