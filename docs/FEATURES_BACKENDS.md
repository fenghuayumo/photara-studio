# Feature backends

AetherScan treats **extraction** and **matching** as independent, swappable
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
extractors:  siftgpu (default) | sift | superpoint | disk
matchers:    gpu_mutual_ratio (default) | mutual_ratio | lightglue
pipelines:   none (default) | lightglue_end2end
```

Default behavior is unchanged: `siftgpu` × `gpu_mutual_ratio`.

### Compose freely

```text
# Classic
--extractor siftgpu --matcher gpu_mutual_ratio

# Learned extract + learned match
--extractor superpoint --extractor-model superpoint.onnx \
--matcher lightglue --lightglue-model superpoint_lightglue_fused.onnx

--extractor disk --extractor-model disk.onnx \
--matcher lightglue --lightglue-model disk_lightglue_fused.onnx
```

Extractor knobs (`--extractor-model/width/height/cpu`) belong to the extractor.
Matcher knobs (`--lightglue-model/min-score/cpu`) belong to the LightGlue
matcher (or the optional end2end pipeline). They are not a joint “package”.

Compatibility today (extend in `compat.hpp`):

| extractor | matcher |
|-----------|---------|
| siftgpu / sift | gpu_mutual_ratio / mutual_ratio |
| superpoint / disk | lightglue |

### Optional fused recipe

`--pipeline lightglue_end2end` re-detects per pair (slower). Prefer compose.

## Adding a backend

1. Implement extractor or matcher.
2. Register / construct from `run_frontend`.
3. Update `compat.hpp` for allowed pairs.
4. Keep CLI options namespaced to that backend.
