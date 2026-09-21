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

`--pipeline lightglue_end2end` re-detects per pair (slower). Prefer compose.

## Adding a backend

1. Implement extractor or matcher.
2. Register / construct from `run_frontend`.
3. Update `compat.hpp` for allowed pairs.
4. Keep CLI options namespaced to that backend.
