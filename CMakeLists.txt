cmake_minimum_required(VERSION 3.28)

project(AetherScan
    VERSION 0.2.0
    DESCRIPTION "High-performance GPU-first Structure from Motion"
    LANGUAGES C CXX)

option(AETHERSCAN_BUILD_BENCHMARKS "Build performance benchmarks" ON)
option(AETHERSCAN_BUILD_TESTS "Build correctness tests" ON)
option(AETHERSCAN_BUILD_EDITOR "Build the Vulkan + Dear ImGui reconstruction editor" ON)
option(AETHERSCAN_BUILD_PYTHON
    "Build the nanobind Python experiment module" OFF)
option(AETHERSCAN_ENABLE_CUDA "Build CUDA acceleration" ON)
option(AETHERSCAN_ENABLE_SPLAT
    "Build CUDA Gaussian splat training (TinyTensor + GGGS-derived reference rasterizer)" ON)
option(AETHERSCAN_ENABLE_ACCUTILE
    "Use opacity-aware SnugBox/AccuTile Gaussian-to-tile enumeration" ON)
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

# Texture baking (aether_drender / UVAtlas / Vulkan) + optional Intrinsic delight.
# Default source is the git submodule at third_party/aether_drender.
option(AETHERSCAN_ENABLE_TEXTURE
    "Enable mesh UV unwrap + projective texture baking (aether_drender)" ON)
option(AETHERSCAN_ENABLE_AETHER_MESH
    "Use optional CGAL aether_drender mesh repair/decimation" ON)
option(AETHERSCAN_ENABLE_INSTANT_REMESH
    "Use Instant Meshes field-aligned remeshing before CGAL decimation" ON)
set(AETHERSCAN_AETHER_DRENDER_ROOT "" CACHE PATH
    "Path to aether_drender sources (empty = auto-detect submodule / sibling)")

function(aetherscan_resolve_aether_drender_root out_var)
    if(AETHERSCAN_AETHER_DRENDER_ROOT AND
       EXISTS "${AETHERSCAN_AETHER_DRENDER_ROOT}/CMakeLists.txt")
        set(${out_var} "${AETHERSCAN_AETHER_DRENDER_ROOT}" PARENT_SCOPE)
        return()
    endif()
    set(_candidates
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/aether_drender"
        "${CMAKE_CURRENT_SOURCE_DIR}/../aether_drender")
    foreach(_cand IN LISTS _candidates)
        get_filename_component(_cand_abs "${_cand}" ABSOLUTE)
        if(EXISTS "${_cand_abs}/CMakeLists.txt")
            set(${out_var} "${_cand_abs}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${out_var} "" PARENT_SCOPE)
endfunction()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FetchOnnxRuntime)
aetherscan_setup_onnxruntime()

set(AETHERSCAN_HAS_TEXTURE OFF)
if(AETHERSCAN_ENABLE_TEXTURE)
    aetherscan_resolve_aether_drender_root(_aetherscan_drender_root)
    if(NOT _aetherscan_drender_root)
        message(WARNING
            "AETHERSCAN_ENABLE_TEXTURE=ON but aether_drender not found. "
            "Initialize the git submodule at third_party/aether_drender "
            "(git submodule update --init --recursive), "
            "or set -DAETHERSCAN_AETHER_DRENDER_ROOT=. Disabling texture module.")
        set(AETHERSCAN_ENABLE_TEXTURE OFF)
    else()
        set(AETHERSCAN_AETHER_DRENDER_ROOT "${_aetherscan_drender_root}"
            CACHE PATH "Path to aether_drender sources" FORCE)
        message(STATUS "aether_drender: ${AETHERSCAN_AETHER_DRENDER_ROOT}")
        find_package(Vulkan 1.2 QUIET)
        if(NOT Vulkan_FOUND)
            message(WARNING
                "Vulkan SDK 1.2+ not found; disabling AETHERSCAN_ENABLE_TEXTURE")
            set(AETHERSCAN_ENABLE_TEXTURE OFF)
        else()
            set(AETHER_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
            set(AETHER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
            find_package(CGAL CONFIG QUIET)
            if(AETHERSCAN_ENABLE_AETHER_MESH AND CGAL_FOUND)
                set(AETHER_BUILD_MESH_TOOLS ON CACHE BOOL "" FORCE)
                set(AETHER_ENABLE_INSTANT_MESHES
                    ${AETHERSCAN_ENABLE_INSTANT_REMESH} CACHE BOOL "" FORCE)
                message(STATUS
                    "Enabling aether::mesh repair/decimation; Instant Meshes="
                    "${AETHERSCAN_ENABLE_INSTANT_REMESH}")
            else()
                set(AETHER_BUILD_MESH_TOOLS OFF CACHE BOOL "" FORCE)
            endif()
            add_subdirectory(
                "${AETHERSCAN_AETHER_DRENDER_ROOT}"
                "${CMAKE_BINARY_DIR}/_deps/aether_drender")
            set(AETHERSCAN_HAS_TEXTURE ON)
            include(FetchIntrinsicModels)
            aetherscan_setup_intrinsic_models()
        endif()
    endif()
endif()

include(CTest)
add_subdirectory(aetherscan)
