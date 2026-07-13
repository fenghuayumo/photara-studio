cmake_minimum_required(VERSION 3.28)

project(AetherScan
    VERSION 0.2.0
    DESCRIPTION "High-performance GPU-first Structure from Motion"
    LANGUAGES C CXX)

option(AETHERSCAN_BUILD_BENCHMARKS "Build performance benchmarks" ON)
option(AETHERSCAN_BUILD_TESTS "Build correctness tests" ON)
option(AETHERSCAN_ENABLE_CUDA "Build CUDA acceleration" ON)
option(AETHERSCAN_ENABLE_NATIVE_ARCH "Optimize CPU code for the build host" ON)
option(AETHERSCAN_ENABLE_FEATURES "Build image feature extraction and matching" ON)
option(AETHERSCAN_ENABLE_SIFTGPU "Enable the optional non-commercial SiftGPU adapter" ON)
# COLMAP-style ONNX gate (LightGlue). Off by default; turn on to compile
# learned matchers. With FETCH_ONNX=ON, CMake downloads the official SDK.
option(AETHERSCAN_ENABLE_ONNX
    "Enable ONNX Runtime (LightGlue / learned matchers)" OFF)
option(AETHERSCAN_FETCH_ONNX
    "Download ONNX Runtime via FetchContent when ENABLE_ONNX is ON" ON)
set(AETHERSCAN_ONNX_VERSION "1.20.1" CACHE STRING
    "ONNX Runtime release version used by FETCH_ONNX")
set(AETHERSCAN_ONNXRUNTIME_ROOT "" CACHE PATH
    "Optional local ONNX Runtime SDK root (skips FetchContent when valid)")
# OpenCV is intentionally not a dependency. Image IO uses FreeImage; SIFT uses
# vendored VLFeat; multi-view geometry uses Eigen.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FetchOnnxRuntime)
aetherscan_setup_onnxruntime()

include(CTest)
add_subdirectory(aetherscan)
