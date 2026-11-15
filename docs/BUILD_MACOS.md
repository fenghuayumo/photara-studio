# macOS Vulkan Release

Use an arm64 C++20 toolchain, CMake 3.28+, Ninja, Vulkan SDK (including DXC
and MoltenVK), Eigen, PoseLib (fetched by CMake), Ceres, gflags, glog,
FreeImage, JPEG, PNG, Boost, hnswlib headers, GLFW, nlohmann-json, yaml-cpp,
WebP, zstd, and optional CGAL.

Set `VULKAN_SDK`, `PHOTARA_DEPS_PREFIX`, and `PHOTARA_HNSWLIB_INCLUDE`, then run
`scripts/build_macos_release.sh`. This builds Release only and creates a local
`build/Photara Studio.app` launcher. Keep it alongside the build
folder: dependencies are not bundled for distribution.

All generated files live under `build/`. The main project builds its modules
through CMake targets in that directory. Independent runtime checks use
`build/checks/tinytensor-vulkan` and `build/checks/splat-vulkan`.

`scripts/check_macos_vulkan.sh` checks independent runtime initialization.
`photara/tests/vulkan_backend_smoke.cpp` checks Vulkan tensor addition and a Gaussian render through
the public rasterizer API. Full training and CUDA regression tests have not
been run on macOS. CUDA-only entry points fail explicitly when CUDA is absent;
this does not imply feature parity. Texture baking and USDC are disabled by
this build script. Inference uses CPU. Training previews share an IOSurface-backed
GPU image through `VK_EXT_metal_objects`. The producer packs and copies pixels
on the GPU; Studio copies the shared image into its sampled viewport texture on
the GPU. Only completed producer/consumer sequence numbers cross an anonymous
shared-memory page, after each GPU submission completes. A busy viewport drops
updates instead of blocking training. No image readback, PNG codec, or preview
files are involved. Completed models use the in-process Vulkan renderer.

To check live preview, camera updates, completion and a second training run
with an existing ASFM scene and its sibling `images/` folder:

```sh
VK_ICD_FILENAMES="$VULKAN_SDK/share/vulkan/icd.d/MoltenVK_icd.json" \
  python3 scripts/studio_training_preview_smoke.py --asfm /data/scene.asfm
```

The check uses isolated preferences and a temporary working copy of the ASFM;
it leaves captures and a JSON report in the printed artifact directory.

Changes under the photara_vk and photara_drender git submodules must be retained
along with the parent repository changes; updating/resetting these submodules
would discard the portability fixes.

## SOG-XT encoding-aware fine-tuning

The Vulkan path supports frozen-layout field quantization, shared SH codebook
STE gradients, signed-log position Jacobians, and image-space TV regularization.
It uses gather reductions rather than optional floating-point atomic adds.
Frozen cluster/neighbor topology is prepared once on the CPU; codec field and
gradient passes execute on the Vulkan device. PPISP and bilateral-grid correction
use their existing Vulkan implementations.

Build and run the Release regression test on macOS:

```sh
cmake --build build --target photara_sog_xt_vulkan_test --parallel 8
VK_ICD_FILENAMES="$VULKAN_SDK/share/vulkan/icd.d/MoltenVK_icd.json" \
  build/photara/photara_sog_xt_vulkan_test --sog-xt-vulkan-training-only
```

This tests export/decode parity, shared-codeword gradients, a CPU oracle for
STE/TV derivatives, an Adam parameter update, and four end-to-end fine-tuning
steps with both appearance corrections, held-out image exclusion, and cancellation.
It is a synthetic regression test, not a large-scene quality or speed benchmark.

## ADC-IGS training and points visualization

Build `photara_vulkan_training_features_test` in the same Release build directory
and run it with the MoltenVK ICD selected:

```sh
cmake --build build --target photara_vulkan_training_features_test --parallel 8
VK_ICD_FILENAMES="$VULKAN_SDK/share/vulkan/icd.d/MoltenVK_icd.json" \
  build/photara/photara_vulkan_training_features_test
```

The regression checks points pixels against CPU camera projection/compositing
for pinhole, fisheye, panorama and orthographic views, overlapping points and
SH0–3 shading. Revised ADC-IGS noise and MVS depth/normal losses have CPU numeric
oracles. Short training runs cover ADC-IGS densification, PPISP/bilateral grids,
revised noise, MVS supervision, transparent masks, normal fields, multi-view
losses, Mip3D filtering, opacity resets and 3DGUT. Only ADC-IGS is covered;
other densification strategies remain disabled on Vulkan.

To enable SDK validation, additionally set
`VK_LAYER_PATH="$VULKAN_SDK/share/vulkan/explicit_layer.d"` and
`TINYTENSOR_VULKAN_VALIDATION=1`. These are synthetic correctness regressions;
large-scene quality/performance and CUDA regression tests remain separate work.
