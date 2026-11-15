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
this build script. Inference uses CPU. macOS preview uses the ordinary texture
handoff because Linux opaque FD exports are unavailable on MoltenVK.

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
