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
            shaders/splat_radix_hist.hlsl
            shaders/splat_radix_scatter.hlsl
            shaders/splat_ranges.hlsl
            shaders/splat_blend.hlsl
            shaders/splat_median_backward.hlsl
            shaders/splat_blend_backward.hlsl
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
            "${SPLAT_DRENDER_PHOTARA_SHADER_DIR}/splat_loss_reduce.hlsl")

        set(SPLAT_DRENDER_SHADER_BINARIES)
        set(SPLAT_DRENDER_SHADER_HEADERS)
        foreach(shader IN LISTS SPLAT_DRENDER_VULKAN_SHADERS)
            if(IS_ABSOLUTE "${shader}")
                set(shader_source "${shader}")
            else()
                set(shader_source "${CMAKE_CURRENT_SOURCE_DIR}/${shader}")
            endif()
            get_filename_component(shader_name "${shader}" NAME)
            set(shader_binary "${CMAKE_CURRENT_BINARY_DIR}/shaders/${shader_name}.spv")
            add_custom_command(
                OUTPUT "${shader_binary}"
                COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders"
                COMMAND "${SPLAT_DRENDER_DXC_EXECUTABLE}"
                        -spirv -T cs_6_0 -E main -O3
                        -fspv-target-env=vulkan1.2
                        -fvk-use-dx-layout
                        -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
                        "${shader_source}"
                        -Fo "${shader_binary}"
                DEPENDS "${shader_source}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/splat_math.hlsli"
                        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/splat_enum.hlsli"
                VERBATIM)
            list(APPEND SPLAT_DRENDER_SHADER_BINARIES "${shader_binary}")
            string(MAKE_C_IDENTIFIER "${shader_name}_spv" shader_variable)
            set(shader_embedded_header "${CMAKE_CURRENT_BINARY_DIR}/generated/${shader_name}.embedded.hpp")
            add_custom_command(
                OUTPUT "${shader_embedded_header}"
                COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
                COMMAND "${CMAKE_COMMAND}"
                        -DINPUT_FILE=${shader_binary}
                        -DOUTPUT_FILE=${shader_embedded_header}
                        -DVARIABLE_NAME=${shader_variable}
                        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spir_v.cmake"
                DEPENDS "${shader_binary}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spir_v.cmake"
                VERBATIM)
            list(APPEND SPLAT_DRENDER_SHADER_HEADERS "${shader_embedded_header}")
        endforeach()

        add_custom_target(splat_drender_shaders
            DEPENDS ${SPLAT_DRENDER_SHADER_BINARIES} ${SPLAT_DRENDER_SHADER_HEADERS})

        add_library(splat_drender_vulkan STATIC
            src/vulkan/context.cpp
            src/vulkan/splat_rasterizer.cpp)
        add_library(splat_drender::vulkan ALIAS splat_drender_vulkan)
        add_dependencies(splat_drender_vulkan splat_drender_shaders)

        target_include_directories(splat_drender_vulkan
            PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan"
                "${CMAKE_CURRENT_BINARY_DIR}/generated"
            PUBLIC
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:include>)
        target_compile_features(splat_drender_vulkan PUBLIC cxx_std_20)
        target_link_libraries(splat_drender_vulkan PUBLIC Vulkan::Vulkan)
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
