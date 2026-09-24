# Intrinsic / Delight model license boundary

`--delight` performs image-space intrinsic decomposition with the Intrinsic
`stage_0.onnx` through `stage_3.onnx` weights. Inference is implemented with
the C++ ONNX Runtime path in `photara/src/texture/delight.cpp`.

## Current status

- This repository does not distribute the model weights. Obtain them
  separately and place them in `PHOTARA_INTRINSIC_MODELS_DIR`, which defaults
  to `${CMAKE_BINARY_DIR}/Models/Intrinsic`; see
  `cmake/FetchIntrinsicModels.cmake`.
- Internal project records classify the upstream `compphoto/Intrinsic`
  project and its weights as academic/non-commercial. This statement is not
  legal advice. Before any commercial release, the responsible owner or legal
  counsel must verify the upstream license and model card directly.
- `--delight` fails when the weights are unavailable. Plain `--texture`
  remains available for projection-based texture baking.

## Scope

- This boundary applies only to the Delight weights and inference path. It
  does not change the Apache-2.0 license of Photara's own source, or the
  license boundaries of SfM, MVS, Splat, or `photara_drender` texture baking.
- Exported albedo is influenced by these weights only when `--delight` is
  explicitly enabled.
- Other third-party components, including VLFeat, cxxopts, TinyTensor, and the
  `photara_drender` submodule, retain their own licenses. See
  [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
