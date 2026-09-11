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
    CUDA_ARCHITECTURES native
    POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    target_compile_options(splat_drender PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr -use_fast_math -Xcompiler=/EHsc>)
else()
    target_compile_options(splat_drender PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr -use_fast_math>)
endif()
