find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(_splat_render_dxc_hints "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
if(Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
    get_filename_component(_splat_render_vulkan_bin
        "${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}" DIRECTORY)
    list(APPEND _splat_render_dxc_hints "${_splat_render_vulkan_bin}")
endif()
find_program(SPLAT_RENDER_DXC NAMES dxc HINTS ${_splat_render_dxc_hints} REQUIRED)

set(_splat_render_shader_dir "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(_splat_render_generated "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(_splat_render_header "${_splat_render_generated}/preview_spv.hpp")
set(_splat_render_shaders
    "${_splat_render_shader_dir}/sort_keys.cs.hlsl"
    "${_splat_render_shader_dir}/sort_hist.cs.hlsl"
    "${_splat_render_shader_dir}/sort_scan.cs.hlsl"
    "${_splat_render_shader_dir}/sort_scatter.cs.hlsl"
    "${_splat_render_shader_dir}/ring_prepare.cs.hlsl"
    "${_splat_render_shader_dir}/ring_draw.vs.hlsl"
    "${_splat_render_shader_dir}/ring_draw.ps.hlsl")

add_custom_command(
    OUTPUT "${_splat_render_header}"
    COMMAND Python3::Interpreter
        "${_splat_render_shader_dir}/embed_spv.py"
        "${SPLAT_RENDER_DXC}"
        "${_splat_render_header}"
        "${_splat_render_shader_dir}"
        ${_splat_render_shaders}
    DEPENDS
        "${_splat_render_shader_dir}/embed_spv.py"
        "${_splat_render_shader_dir}/ring_math.hlsl"
        ${_splat_render_shaders}
    VERBATIM)

add_library(splat_render STATIC
    src/renderer.cpp
    "${_splat_render_header}")
add_library(photara::splat_render ALIAS splat_render)
target_include_directories(splat_render
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${_splat_render_generated}")
target_link_libraries(splat_render PUBLIC Vulkan::Vulkan)
if(TARGET splat_drender::vulkan)
    target_link_libraries(splat_render PUBLIC splat_drender::vulkan)
    target_compile_definitions(splat_render PRIVATE SPLAT_DRENDER_HAS_VULKAN=1)
endif()
target_compile_features(splat_render PUBLIC cxx_std_20)
if(MSVC)
    target_compile_options(splat_render PRIVATE /utf-8)
endif()
source_group("shaders" FILES ${_splat_render_shaders}
    "${_splat_render_shader_dir}/ring_math.hlsl")
