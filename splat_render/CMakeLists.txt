find_package(Python3 COMPONENTS Interpreter REQUIRED)
if(NOT Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
    find_program(Vulkan_GLSLANG_VALIDATOR_EXECUTABLE NAMES glslangValidator REQUIRED)
endif()

set(_splat_render_shader_dir "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(_splat_render_generated "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(_splat_render_header "${_splat_render_generated}/gut_spv.hpp")
set(_splat_render_shaders
    "${_splat_render_shader_dir}/gut_keys.comp"
    "${_splat_render_shader_dir}/gut_hist.comp"
    "${_splat_render_shader_dir}/gut_scan.comp"
    "${_splat_render_shader_dir}/gut_scatter.comp"
    "${_splat_render_shader_dir}/gut_prepare.comp"
    "${_splat_render_shader_dir}/gut_splat.vert"
    "${_splat_render_shader_dir}/gut_splat.frag")

add_custom_command(
    OUTPUT "${_splat_render_header}"
    COMMAND Python3::Interpreter
        "${_splat_render_shader_dir}/embed_gut_spv.py"
        "${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}"
        "${_splat_render_header}"
        "${_splat_render_shader_dir}"
        ${_splat_render_shaders}
    DEPENDS
        "${_splat_render_shader_dir}/embed_gut_spv.py"
        "${_splat_render_shader_dir}/gut_frame.glsl"
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
target_compile_features(splat_render PUBLIC cxx_std_20)
if(MSVC)
    target_compile_options(splat_render PRIVATE /utf-8)
endif()
source_group("shaders" FILES ${_splat_render_shaders}
    "${_splat_render_shader_dir}/gut_frame.glsl")
