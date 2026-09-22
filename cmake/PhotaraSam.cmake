# In-tree SAM 3 on ggml. Runtime selection prefers CUDA, then Vulkan, then
# Metal, and falls back to CPU.

option(PHOTARA_ENABLE_SAM "Build in-process SAM 3 mask generation" ON)
option(PHOTARA_ENABLE_SAM_VULKAN "Build the SAM 3 ggml Vulkan backend" ON)

set(PHOTARA_HAS_SAM OFF)

if(PHOTARA_ENABLE_SAM)
    set(GGML_CCACHE OFF CACHE BOOL "" FORCE)
    set(GGML_STATIC OFF CACHE BOOL "" FORCE)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared CACHE STRING "" FORCE)
    set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GGML_METAL OFF CACHE BOOL "" FORCE)
    if(PHOTARA_ENABLE_SAM_VULKAN)
        find_package(Vulkan QUIET COMPONENTS glslc)
        if(Vulkan_FOUND AND Vulkan_GLSLC_EXECUTABLE)
            set(GGML_VULKAN ON CACHE BOOL "" FORCE)
        else()
            message(STATUS
                "SAM 3: Vulkan SDK/glslc not found; Vulkan backend disabled")
            set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
        endif()
    else()
        set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
    endif()
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
        if(TARGET ggml-cuda)
            # The repository may carry an old global CUDA architecture cache
            # (for example sm_52). SAM 3 needs the attention kernels compiled
            # for the GPU that will execute them, matching Photara's other CUDA
            # targets.
            set_target_properties(ggml-cuda PROPERTIES
                CUDA_ARCHITECTURES native)
        endif()
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
        set(_photara_sam_backends "CPU")
        if(GGML_CUDA)
            list(APPEND _photara_sam_backends "CUDA")
        endif()
        if(GGML_VULKAN)
            list(APPEND _photara_sam_backends "Vulkan")
        endif()
        message(STATUS "SAM 3 ggml backends: ${_photara_sam_backends}")
    endif()
endif()
