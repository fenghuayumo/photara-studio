# ============================================================================
# TinyTensor - Standalone Tensor Library
# ============================================================================

cmake_minimum_required(VERSION 3.20)
project(tinytensor LANGUAGES CXX CUDA)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Disable RDC to avoid MSBuild path issues
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

# CUDA sources
set(TINYTENSOR_CUDA_SOURCES
    tensor_ops.cu
    tensor_warp_reduce.cu
    tensor_matrix_ops.cu
    tensor_broadcast_ops.cu
    tensor_masking_ops.cu
    tensor_random_ops.cu
    tensor_strided_ops.cu
    tensor_dot_optimized.cu
    tensor_nn_ops.cu
    tensor_fused_pointwise.cu
    tensor_exports.cu
)

# C++ sources
set(TINYTENSOR_CPP_SOURCES
    tensor.cpp
    tensor_advanced_ops.cpp
    tensor_broadcast.cpp
    tensor_factory.cpp
    tensor_masking_ops.cpp
    tensor_matrix_ops.cpp
    tensor_movement_ops.cpp
    tensor_nn_ops.cpp
    tensor_random_ops.cpp
    tensor_row_proxy.cpp
    tensor_shape_ops.cpp
    tensor_unified_ops.cpp
    tensor_utils.cpp
    lazy_config.cpp
    lazy_executor.cpp
    lazy_ir.cpp
    cuda_stream_context.cpp
    gpu_slab_allocator.cpp
    offset_allocator.cpp
    pinned_memory_allocator.cpp
    core/vram_profiler.cpp
)

# Create CUDA static library directly
add_library(tinytensor_cuda OBJECT ${TINYTENSOR_CUDA_SOURCES})
target_include_directories(tinytensor_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Set CUDA architectures explicitly for the target
# Use the configured project architecture. Native keeps local developer builds
# fast while allowing release builds to pass an explicit architecture list.
set_target_properties(tinytensor_cuda PROPERTIES
    CUDA_ARCHITECTURES "native"
    FOLDER "external"
)

# Create C++ static library
add_library(tinytensor_cpp OBJECT ${TINYTENSOR_CPP_SOURCES})
target_include_directories(tinytensor_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(tinytensor_cpp PROPERTIES FOLDER "external")

# Create the main static library that merges both
add_library(tinytensor STATIC $<TARGET_OBJECTS:tinytensor_cuda> $<TARGET_OBJECTS:tinytensor_cpp>)

# Include directories
target_include_directories(tinytensor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link CUDA runtime
target_link_libraries(tinytensor PUBLIC CUDA::cudart CUDA::curand)

# Set CUDA properties
set_target_properties(tinytensor PROPERTIES FOLDER "external")

# Compile options
if(MSVC)
    target_compile_options(tinytensor_cuda PRIVATE --extended-lambda --expt-relaxed-constexpr)
    target_compile_options(tinytensor_cpp PRIVATE /EHsc /utf-8)
    target_compile_definitions(tinytensor_cuda PRIVATE FMT_UNICODE=0)
endif()

# Alias for linking
add_library(tinytensor::tinytensor ALIAS tinytensor)
