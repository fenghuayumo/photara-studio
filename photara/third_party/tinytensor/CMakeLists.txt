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
# Use the architecture set resolved by the parent project so every CUDA object
# in the final executable has the same deployment coverage.
set_target_properties(tinytensor_cuda PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
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

# ============================================================================
# Optional Vulkan compute backend (HLSL -> SPIR-V via DXC)
# ============================================================================
option(TINYTENSOR_ENABLE_VULKAN "Build the TinyTensor Vulkan compute backend" OFF)
if(TINYTENSOR_ENABLE_VULKAN)
    find_package(Vulkan 1.2 QUIET)
    find_program(TINYTENSOR_DXC_EXECUTABLE
        NAMES dxc
        HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
    if(NOT Vulkan_FOUND OR NOT TINYTENSOR_DXC_EXECUTABLE)
        message(WARNING
            "TINYTENSOR_ENABLE_VULKAN=ON but Vulkan SDK 1.2+ or dxc was not found; "
            "disabling the Vulkan backend")
        set(TINYTENSOR_ENABLE_VULKAN OFF)
    endif()
endif()

if(TINYTENSOR_ENABLE_VULKAN)
    set(TINYTENSOR_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vulkan/shaders")
    set(TINYTENSOR_SHADER_SOURCES
        adam_f32.hlsl
        elementwise.hlsl
        fused_pointwise.hlsl
        strided_copy.hlsl
        index_select.hlsl
        index_fill.hlsl
        mask_flags.hlsl
        scan_block.hlsl
        scan_add.hlsl
        compact.hlsl
        multinomial.hlsl
        reduce.hlsl
        reduce_all_f32.hlsl
        matmul.hlsl
        random.hlsl
        cumsum.hlsl
        pool.hlsl
        scatter.hlsl
        select_compact.hlsl
        cat.hlsl)
    set(TINYTENSOR_SHADER_BINARIES)
    set(TINYTENSOR_SHADER_EMBEDDED_HEADERS)
    foreach(shader IN LISTS TINYTENSOR_SHADER_SOURCES)
        set(shader_binary "${CMAKE_CURRENT_BINARY_DIR}/vulkan/shaders/${shader}.spv")
        set(shader_embedded "${CMAKE_CURRENT_BINARY_DIR}/vulkan/generated/${shader}.embedded.hpp")
        string(MAKE_C_IDENTIFIER "${shader}_spv" shader_variable)
        add_custom_command(
            OUTPUT "${shader_binary}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory
                    "${CMAKE_CURRENT_BINARY_DIR}/vulkan/shaders"
            COMMAND "${TINYTENSOR_DXC_EXECUTABLE}"
                    -spirv -T cs_6_0 -E main -O3
                    -fspv-target-env=vulkan1.2
                    -fvk-use-dx-layout
                    -I "${TINYTENSOR_SHADER_DIR}"
                    "${TINYTENSOR_SHADER_DIR}/${shader}"
                    -Fo "${shader_binary}"
            DEPENDS
                "${TINYTENSOR_SHADER_DIR}/${shader}"
                "${TINYTENSOR_SHADER_DIR}/common.hlsli"
            VERBATIM)
        add_custom_command(
            OUTPUT "${shader_embedded}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory
                    "${CMAKE_CURRENT_BINARY_DIR}/vulkan/generated"
            COMMAND "${CMAKE_COMMAND}"
                    -DINPUT_FILE=${shader_binary}
                    -DOUTPUT_FILE=${shader_embedded}
                    -DVARIABLE_NAME=${shader_variable}
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/vulkan/cmake/embed_spir_v.cmake"
            DEPENDS
                "${shader_binary}"
                "${CMAKE_CURRENT_SOURCE_DIR}/vulkan/cmake/embed_spir_v.cmake"
            VERBATIM)
        list(APPEND TINYTENSOR_SHADER_BINARIES "${shader_binary}")
        list(APPEND TINYTENSOR_SHADER_EMBEDDED_HEADERS "${shader_embedded}")
    endforeach()

    add_custom_target(tinytensor_shaders
        DEPENDS ${TINYTENSOR_SHADER_BINARIES} ${TINYTENSOR_SHADER_EMBEDDED_HEADERS})

    add_library(tinytensor_vulkan OBJECT
        vulkan/runtime/runtime.cpp
        vulkan/ops.cpp)
    add_dependencies(tinytensor_vulkan tinytensor_shaders)
    target_include_directories(tinytensor_vulkan
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
        PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/vulkan/generated)
    target_compile_features(tinytensor_vulkan PUBLIC cxx_std_20)
    target_compile_definitions(tinytensor_vulkan PUBLIC TINYTENSOR_HAS_VULKAN=1)
    target_link_libraries(tinytensor_vulkan PUBLIC Vulkan::Vulkan)
    set_target_properties(tinytensor_vulkan PROPERTIES FOLDER "external")
    if(MSVC)
        target_compile_options(tinytensor_vulkan PRIVATE /EHsc /utf-8)
    endif()

    target_sources(tinytensor PRIVATE $<TARGET_OBJECTS:tinytensor_vulkan>)
    target_link_libraries(tinytensor PUBLIC Vulkan::Vulkan)
    target_compile_definitions(tinytensor PUBLIC TINYTENSOR_HAS_VULKAN=1)
    target_compile_definitions(tinytensor_cpp PUBLIC TINYTENSOR_HAS_VULKAN=1)
    target_compile_definitions(tinytensor_cuda PUBLIC TINYTENSOR_HAS_VULKAN=1)
    message(STATUS "TinyTensor Vulkan backend: ON (dxc=${TINYTENSOR_DXC_EXECUTABLE})")
    set(TINYTENSOR_HAS_VULKAN_BACKEND ON PARENT_SCOPE)
endif()
