cmake_minimum_required(VERSION 3.28)

project(splat_drender VERSION 1.0.0 LANGUAGES CXX CUDA)

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "splat_drender requires CUDA.")
endif()

find_package(CUDAToolkit 12.0 REQUIRED)

add_library(splat_drender STATIC
    src/pipeline.cu
    src/render_forward.cu
    src/render_backward.cu
    src/point_sampling.cu)
add_library(splat_drender::splat_drender ALIAS splat_drender)

target_include_directories(splat_drender
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
target_compile_features(splat_drender PUBLIC cxx_std_20)
target_link_libraries(splat_drender PUBLIC CUDA::cudart)
set_target_properties(splat_drender PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    target_compile_options(splat_drender PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr -use_fast_math -Xcompiler=/EHsc>)
else()
    target_compile_options(splat_drender PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr -use_fast_math>)
endif()

# ---------------------------------------------------------------------------
# Vulkan backend: the same forward rasterizer as Vulkan compute passes, so a
# trained model can be drawn and edited without a CUDA context, and so the
# differentiable pass has a device-side home. The CUDA and Vulkan libraries are
# independent: a consumer can link either one.
# ---------------------------------------------------------------------------
option(SPLAT_DRENDER_ENABLE_VULKAN "Build the Vulkan splat rasterizer" ON)
option(SPLAT_DRENDER_ENABLE_VALIDATION "Enable Vulkan validation by default" OFF)

if(SPLAT_DRENDER_ENABLE_VULKAN)
    find_package(Vulkan 1.2 QUIET)
    find_program(SPLAT_DRENDER_DXC_EXECUTABLE
        NAMES dxc
        HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")

    if(Vulkan_FOUND AND SPLAT_DRENDER_DXC_EXECUTABLE)
        set(SPLAT_DRENDER_VULKAN_SHADERS
            shaders/splat_preprocess.hlsl
            shaders/splat_scan.hlsl
            shaders/splat_emit.hlsl
            shaders/splat_indirect_control.hlsl
            shaders/splat_emit_indirect.hlsl
            shaders/splat_radix_hist.hlsl
            shaders/splat_radix_hist_indirect.hlsl
            shaders/splat_radix_scatter.hlsl
            shaders/splat_radix_scatter_indirect.hlsl
            shaders/splat_ranges.hlsl
            shaders/splat_ranges_indirect.hlsl
            shaders/splat_blend.hlsl
            shaders/splat_blend_no_geometry.hlsl
            shaders/splat_blend_training.hlsl
            shaders/splat_median_backward.hlsl
            shaders/splat_blend_backward.hlsl
            shaders/splat_blend_backward_no_geometry.hlsl
            shaders/splat_blend_backward_no_geometry_subgroup.hlsl
            shaders/splat_blend_backward_no_geometry_atomic.hlsl
            shaders/splat_sample_depth.hlsl
            shaders/splat_sample_depth_backward.hlsl
            shaders/splat_multi_view.hlsl
            shaders/splat_clear.hlsl
            shaders/splat_project_backward.hlsl
            shaders/splat_pack_rgba.hlsl)
        # Photometric training shaders belong to Photara's splat backend, not
        # the reusable rasterizer source tree. They are still embedded in the
        # Vulkan library because its command recorder owns these pipelines.
        set(SPLAT_DRENDER_PHOTARA_SHADER_DIR
            "${CMAKE_CURRENT_LIST_DIR}/../../photara/src/splat"
            CACHE PATH "Photara splat training shader directory")
        list(APPEND SPLAT_DRENDER_VULKAN_SHADERS
            "${SPLAT_DRENDER_PHOTARA_SHADER_DIR}/splat_ssim.hlsl"
            "${SPLAT_DRENDER_PHOTARA_SHADER_DIR}/splat_loss_reduce.hlsl"
            "${SPLAT_DRENDER_PHOTARA_SHADER_DIR}/splat_color_correction.hlsl"
            "${SPLAT_DRENDER_PHOTARA_SHADER_DIR}/splat_color_correction_atomic.hlsl"
            "${SPLAT_DRENDER_PHOTARA_SHADER_DIR}/splat_preview_pack.hlsl")

        if(NOT TARGET photara_vk::photara_vk)
            add_subdirectory(
                "${CMAKE_CURRENT_SOURCE_DIR}/../photara_vk"
                "${CMAKE_CURRENT_BINARY_DIR}/photara_vk")
        endif()
        if(NOT PHOTARA_VK_DXC)
            set(PHOTARA_VK_DXC "${SPLAT_DRENDER_DXC_EXECUTABLE}")
        endif()
        include("${PHOTARA_VK_CMAKE_DIR}/PhotaraVkShaders.cmake")

        add_library(splat_drender_vulkan STATIC
            src/vulkan/context.cpp
            src/vulkan/external_preview.cpp
            src/vulkan/splat_rasterizer.cpp)
        add_library(splat_drender::vulkan ALIAS splat_drender_vulkan)
        photara_vk_embed_hlsl(
            TARGET splat_drender_vulkan
            SHADERS ${SPLAT_DRENDER_VULKAN_SHADERS}
            INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
            OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated"
            DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/splat_math.hlsli"
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/splat_enum.hlsli"
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/splat_blend.hlsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/splat_blend_backward.hlsl"
                "${SPLAT_DRENDER_PHOTARA_SHADER_DIR}/splat_color_correction.hlsl")

        target_include_directories(splat_drender_vulkan
            PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan"
                "${CMAKE_CURRENT_BINARY_DIR}/generated"
            PUBLIC
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:include>)
        target_compile_features(splat_drender_vulkan PUBLIC cxx_std_20)
        target_link_libraries(splat_drender_vulkan PUBLIC Vulkan::Vulkan photara_vk::photara_vk)
        target_compile_definitions(splat_drender_vulkan
            PRIVATE SPLAT_DRENDER_ENABLE_VALIDATION=$<BOOL:${SPLAT_DRENDER_ENABLE_VALIDATION}>)
        if(MSVC)
            target_compile_options(splat_drender_vulkan PRIVATE /W4 /permissive- /EHsc)
        else()
            target_compile_options(splat_drender_vulkan PRIVATE -Wall -Wextra -Wpedantic)
        endif()
        message(STATUS "splat_drender: Vulkan backend enabled (${SPLAT_DRENDER_DXC_EXECUTABLE})")
    else()
        message(STATUS "splat_drender: Vulkan backend disabled (Vulkan SDK or dxc not found)")
    endif()
endif()
