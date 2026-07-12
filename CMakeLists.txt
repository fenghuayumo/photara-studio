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
# OpenCV is intentionally not a dependency. Image IO uses FreeImage; SIFT uses
# vendored VLFeat; multi-view geometry uses Eigen.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
add_subdirectory(aetherscan)
