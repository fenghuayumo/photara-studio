# macOS Vulkan Release

Use an arm64 C++20 toolchain, CMake 3.28+, Ninja, Vulkan SDK (including DXC
and MoltenVK), Eigen, PoseLib (fetched by CMake), Ceres, gflags, glog,
FreeImage, JPEG, PNG, Boost, hnswlib headers, GLFW, nlohmann-json, yaml-cpp,
WebP, zstd, and optional CGAL.

Set `VULKAN_SDK`, `PHOTARA_DEPS_PREFIX`, and `PHOTARA_HNSWLIB_INCLUDE`, then run
`scripts/build_macos_release.sh`. This builds Release only and creates a local
`build-studio-release/Photara Studio.app` launcher. Keep it alongside the build
folder: dependencies are not bundled for distribution.

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
