#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${VULKAN_SDK:?Set VULKAN_SDK to the macOS SDK directory}"
export PATH="$VULKAN_SDK/bin:$PATH"
cmake -S third_party/splat_drender -B build-vulkan-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSPLAT_DRENDER_ENABLE_CUDA=OFF -DSPLAT_DRENDER_ENABLE_VULKAN=ON
cmake --build build-vulkan-release --parallel 8
cmake -S photara/third_party/tinytensor -B build-tensor-vulkan-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DTINYTENSOR_BUILD_VULKAN_RUNTIME_ONLY=ON
cmake --build build-tensor-vulkan-release --parallel 8
c++ -std=c++20 photara/tests/vulkan_runtime_smoke.cpp \
  -Iphotara/third_party/tinytensor -Ithird_party/splat_drender/include \
  -Ithird_party/photara_vk/include -I"$VULKAN_SDK/include" \
  build-tensor-vulkan-release/libtinytensor_vulkan_runtime.a \
  build-vulkan-release/libsplat_drender_vulkan.a \
  build-vulkan-release/photara_vk/libphotara_vk.a \
  -L"$VULKAN_SDK/lib" -lvulkan -Wl,-rpath,"$VULKAN_SDK/lib" \
  -o build-vulkan-release/vulkan-smoke
export VK_ICD_FILENAMES="$VULKAN_SDK/share/vulkan/icd.d/MoltenVK_icd.json"
build-vulkan-release/vulkan-smoke
