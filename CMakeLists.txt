cmake_minimum_required(VERSION 3.28)

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" PHOTARA_VERSION LIMIT_COUNT 1)
string(STRIP "${PHOTARA_VERSION}" PHOTARA_VERSION)
if(NOT PHOTARA_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
    message(FATAL_ERROR
        "VERSION must contain a MAJOR.MINOR.PATCH number (got '${PHOTARA_VERSION}')")
endif()

project(Photara
    VERSION ${PHOTARA_VERSION}
    DESCRIPTION "Structure from motion, dense reconstruction, and 3D Gaussian splatting"
    LANGUAGES C CXX)

option(PHOTARA_BUILD_BENCHMARKS
    "Build extra CLI tools and performance benchmarks" OFF)
option(PHOTARA_BUILD_TESTS "Build correctness tests" OFF)
option(PHOTARA_BUILD_STUDIO "Build Photara Studio (Vulkan + Dear ImGui reconstruction editor)" ON)

# Keep extra tools/tests out of ALL_BUILD and Visual Studio "Build Solution".
# Build them explicitly, e.g. cmake --build build --target photara_splat_test.
function(photara_exclude_from_default)
    foreach(_target IN LISTS ARGN)
        if(TARGET "${_target}")
            set_target_properties("${_target}" PROPERTIES
                EXCLUDE_FROM_ALL TRUE
                EXCLUDE_FROM_DEFAULT_BUILD TRUE)
        endif()
    endforeach()
endfunction()

option(PHOTARA_BUILD_PYTHON
    "Build the nanobind Python experiment module" OFF)
option(PHOTARA_ENABLE_CUDA "Build CUDA acceleration" ON)
set(PHOTARA_CUDA_ARCHITECTURES "" CACHE STRING
    "CUDA architectures for Photara targets (empty = RTX 20+ release set)")

# Resolve one architecture set for every in-tree and bundled CUDA target.
# The default ships native cubins for the desktop RTX generations supported by
# the active toolkit, plus compute_75 PTX for forward compatibility. CUDA 12.8
# is the first toolkit in this project that can emit sm_120 for RTX 50 cards.
function(photara_resolve_cuda_architectures out_var)
    if(PHOTARA_CUDA_ARCHITECTURES)
        set(_photara_cuda_architectures
            ${PHOTARA_CUDA_ARCHITECTURES})
    else()
        set(_photara_cuda_architectures
            75-real   # Turing: RTX 20
            86-real   # Ampere: RTX 30
            89-real)  # Ada: RTX 40
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8")
            list(APPEND _photara_cuda_architectures
                120-real) # Blackwell: RTX 50
        endif()
        list(APPEND _photara_cuda_architectures
            75-virtual) # PTX fallback for later architectures
    endif()
    set(CMAKE_CUDA_ARCHITECTURES
        "${_photara_cuda_architectures}" CACHE STRING
        "CUDA architectures used by Photara and bundled dependencies" FORCE)
    set(${out_var} "${_photara_cuda_architectures}" PARENT_SCOPE)
endfunction()
option(PHOTARA_ENABLE_SPLAT
    "Build CUDA Gaussian splat training (TinyTensor + splat_drender)" ON)
option(PHOTARA_ENABLE_VULKAN_COMPUTE
    "Build the TinyTensor Vulkan compute backend (HLSL/DXC, independent VkDevice)" ON)
option(PHOTARA_ENABLE_ACCUTILE
    "Use opacity-aware SnugBox/AccuTile Gaussian-to-tile enumeration" ON)
option(PHOTARA_ENABLE_NATIVE_ARCH "Optimize CPU code for the build host" ON)
option(PHOTARA_ENABLE_FEATURES "Build image feature extraction and matching" ON)
option(PHOTARA_ENABLE_SIFTGPU "Enable the optional non-commercial SiftGPU adapter" ON)
# COLMAP-style ONNX gate (LightGlue). Off by default; turn on to compile
# learned matchers. With FETCH_ONNX=ON, CMake downloads the official SDK.
option(PHOTARA_ENABLE_ONNX
    "Enable ONNX Runtime (LightGlue / learned matchers)" OFF)
option(PHOTARA_FETCH_ONNX
    "Download ONNX Runtime via FetchContent when ENABLE_ONNX is ON" ON)
set(PHOTARA_ONNX_VERSION "1.20.1" CACHE STRING
    "ONNX Runtime release version used by FETCH_ONNX")
set(PHOTARA_ONNXRUNTIME_ROOT "" CACHE PATH
    "Optional local ONNX Runtime SDK root (skips FetchContent when valid)")
# OpenCV is intentionally not a dependency. JPEG/PNG decoding uses libjpeg and
# libpng directly; FreeImage remains the fallback for other formats; SIFT uses
# vendored VLFeat; multi-view geometry uses Eigen.

# Texture baking (photara_drender / UVAtlas / Vulkan) + optional Intrinsic delight.
# Default source is the git submodule at third_party/photara_drender, which
# exports the photara_drender library (photara::render, photara::mesh).
option(PHOTARA_ENABLE_TEXTURE
    "Enable mesh UV unwrap + projective texture baking (photara_drender)" ON)
option(PHOTARA_ENABLE_MESH_TOOLS
    "Use optional CGAL mesh repair/decimation from photara_drender" ON)
option(PHOTARA_ENABLE_INSTANT_REMESH
    "Use Instant Meshes field-aligned remeshing before CGAL decimation" ON)
set(PHOTARA_DRENDER_ROOT "" CACHE PATH
    "Path to photara_drender sources (empty = auto-detect submodule / sibling)")

function(photara_resolve_drender_root out_var)
    if(PHOTARA_DRENDER_ROOT AND
       EXISTS "${PHOTARA_DRENDER_ROOT}/CMakeLists.txt")
        set(${out_var} "${PHOTARA_DRENDER_ROOT}" PARENT_SCOPE)
        return()
    endif()
    set(_candidates
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/photara_drender"
        "${CMAKE_CURRENT_SOURCE_DIR}/../photara_drender")
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
photara_setup_onnxruntime()

set(PHOTARA_HAS_TEXTURE OFF)
if(PHOTARA_ENABLE_TEXTURE)
    photara_resolve_drender_root(_photara_drender_root)
    if(NOT _photara_drender_root)
        message(WARNING
            "PHOTARA_ENABLE_TEXTURE=ON but photara_drender not found. "
            "Initialize the git submodule at third_party/photara_drender "
            "(git submodule update --init --recursive), "
            "or set -DPHOTARA_DRENDER_ROOT=. Disabling texture module.")
        set(PHOTARA_ENABLE_TEXTURE OFF)
    else()
        set(PHOTARA_DRENDER_ROOT "${_photara_drender_root}"
            CACHE PATH "Path to photara_drender sources" FORCE)
        message(STATUS "photara_drender: ${PHOTARA_DRENDER_ROOT}")
        find_package(Vulkan 1.2 QUIET)
        if(NOT Vulkan_FOUND)
            message(WARNING
                "Vulkan SDK 1.2+ not found; disabling PHOTARA_ENABLE_TEXTURE")
            set(PHOTARA_ENABLE_TEXTURE OFF)
        else()
            set(PHOTARA_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
            set(PHOTARA_BUILD_NATIVE_TESTS OFF CACHE BOOL "" FORCE)
            find_package(CGAL CONFIG QUIET)
            if(PHOTARA_ENABLE_MESH_TOOLS AND CGAL_FOUND)
                set(PHOTARA_BUILD_MESH_TOOLS ON CACHE BOOL "" FORCE)
                set(PHOTARA_ENABLE_INSTANT_MESHES
                    ${PHOTARA_ENABLE_INSTANT_REMESH} CACHE BOOL "" FORCE)
                message(STATUS
                    "Enabling photara::mesh repair/decimation; Instant Meshes="
                    "${PHOTARA_ENABLE_INSTANT_REMESH}")
            else()
                set(PHOTARA_BUILD_MESH_TOOLS OFF CACHE BOOL "" FORCE)
            endif()
            add_subdirectory(
                "${PHOTARA_DRENDER_ROOT}"
                "${CMAKE_BINARY_DIR}/_deps/photara_drender")
            set(PHOTARA_HAS_TEXTURE ON)
            include(FetchIntrinsicModels)
            photara_setup_intrinsic_models()
        endif()
    endif()
endif()

include(CTest)
include(PhotaraSam)
add_subdirectory(photara)
