# In-tree SAM 3 on ggml. CUDA is the default GPU backend when the CUDA
# toolkit is available. Vulkan is left off so a GPU run does not pick it.

option(PHOTARA_ENABLE_SAM "Build in-process SAM 3 mask generation" ON)

set(PHOTARA_HAS_SAM OFF)

if(PHOTARA_ENABLE_SAM)
    set(GGML_CCACHE OFF CACHE BOOL "" FORCE)
    set(GGML_STATIC OFF CACHE BOOL "" FORCE)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared CACHE STRING "" FORCE)
    set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GGML_METAL OFF CACHE BOOL "" FORCE)
    set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
    set(GGML_BLAS OFF CACHE BOOL "" FORCE)
    if(PHOTARA_ENABLE_CUDA)
        include(CheckLanguage)
        check_language(CUDA)
        if(CMAKE_CUDA_COMPILER)
            enable_language(CUDA)
            set(GGML_CUDA ON CACHE BOOL "" FORCE)
        else()
            message(WARNING
                "PHOTARA_ENABLE_SAM: CUDA compiler not found; "
                "SAM 3 will use the ggml CPU backend")
            set(GGML_CUDA OFF CACHE BOOL "" FORCE)
        endif()
    else()
        set(GGML_CUDA OFF CACHE BOOL "" FORCE)
    endif()

    set(PHOTARA_SAM3_ROOT "${CMAKE_SOURCE_DIR}/third_party/sam3")
    if(NOT EXISTS "${PHOTARA_SAM3_ROOT}/sam3.cpp")
        message(WARNING
            "PHOTARA_ENABLE_SAM=ON but third_party/sam3 is missing. "
            "SAM mask generation is disabled.")
    else()
        add_subdirectory(
            "${PHOTARA_SAM3_ROOT}/ggml"
            "${CMAKE_BINARY_DIR}/ggml")
        add_library(photara_sam3 STATIC
            "${PHOTARA_SAM3_ROOT}/sam3.cpp"
            "${PHOTARA_SAM3_ROOT}/sam3.h")
        target_include_directories(photara_sam3 PUBLIC
            "${PHOTARA_SAM3_ROOT}"
            "${PHOTARA_SAM3_ROOT}/stb")
        target_link_libraries(photara_sam3 PUBLIC ggml)
        target_compile_features(photara_sam3 PRIVATE cxx_std_17)
        if(MSVC)
            target_compile_options(photara_sam3 PRIVATE /bigobj /utf-8)
            target_compile_definitions(photara_sam3 PRIVATE NOMINMAX)
        endif()
        set(PHOTARA_HAS_SAM ON)
        if(GGML_CUDA)
            message(STATUS "SAM 3: ggml CUDA backend")
        else()
            message(STATUS "SAM 3: ggml CPU backend")
        endif()
    endif()
endif()
