# ONNX Runtime discovery / FetchContent download.
# Expects:
#   PHOTARA_ENABLE_ONNX
#   PHOTARA_FETCH_ONNX
#   PHOTARA_ONNX_VERSION
#   PHOTARA_ONNXRUNTIME_ROOT (optional manual SDK)
#   PHOTARA_ENABLE_CUDA (selects GPU vs CPU package when fetching)
#
# On success sets:
#   PHOTARA_HAS_ONNXRUNTIME=TRUE
#   PHOTARA_ONNXRUNTIME_ROOT
#   imported target: onnxruntime::onnxruntime

function(photara_setup_onnxruntime)
    set(PHOTARA_HAS_ONNXRUNTIME FALSE PARENT_SCOPE)

    if(NOT PHOTARA_ENABLE_ONNX)
        message(STATUS "ONNX Runtime disabled (PHOTARA_ENABLE_ONNX=OFF)")
        return()
    endif()

    set(_ort_root "${PHOTARA_ONNXRUNTIME_ROOT}")

    # 1) Prefer an explicit local SDK when provided.
    if(_ort_root AND EXISTS "${_ort_root}/include/onnxruntime_cxx_api.h")
        message(STATUS "Using ONNX Runtime from PHOTARA_ONNXRUNTIME_ROOT=${_ort_root}")
    elseif(PHOTARA_FETCH_ONNX)
        include(FetchContent)
        include(GNUInstallDirs)

        set(_ver "${PHOTARA_ONNX_VERSION}")
        message(STATUS "Fetching ONNX Runtime ${_ver}...")

        set(_use_gpu FALSE)
        if(PHOTARA_ENABLE_CUDA)
            set(_use_gpu TRUE)
        endif()

        if(WIN32)
            if(_use_gpu)
                set(_ort_url
                    "https://github.com/microsoft/onnxruntime/releases/download/v${_ver}/onnxruntime-win-x64-gpu-${_ver}.zip")
            else()
                set(_ort_url
                    "https://github.com/microsoft/onnxruntime/releases/download/v${_ver}/onnxruntime-win-x64-${_ver}.zip")
            endif()
        elseif(APPLE)
            if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
                set(_ort_url
                    "https://github.com/microsoft/onnxruntime/releases/download/v${_ver}/onnxruntime-osx-arm64-${_ver}.tgz")
            else()
                message(WARNING
                    "ONNX Runtime FetchContent: x86_64 macOS package not configured; "
                    "set PHOTARA_ONNXRUNTIME_ROOT or PHOTARA_FETCH_ONNX=OFF")
                set(PHOTARA_ENABLE_ONNX OFF CACHE BOOL
                    "Enable ONNX Runtime (LightGlue / learned matchers)" FORCE)
                return()
            endif()
        else()
            if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
                set(_ort_url
                    "https://github.com/microsoft/onnxruntime/releases/download/v${_ver}/onnxruntime-linux-aarch64-${_ver}.tgz")
            elseif(_use_gpu)
                set(_ort_url
                    "https://github.com/microsoft/onnxruntime/releases/download/v${_ver}/onnxruntime-linux-x64-gpu-${_ver}.tgz")
            else()
                set(_ort_url
                    "https://github.com/microsoft/onnxruntime/releases/download/v${_ver}/onnxruntime-linux-x64-${_ver}.tgz")
            endif()
        endif()

        FetchContent_Declare(photara_onnxruntime
            URL "${_ort_url}"
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        FetchContent_MakeAvailable(photara_onnxruntime)

        set(_ort_root "${photara_onnxruntime_SOURCE_DIR}")
        if(NOT EXISTS "${_ort_root}/include/onnxruntime_cxx_api.h")
            message(WARNING
                "Fetched ONNX Runtime is missing headers under ${_ort_root}/include; "
                "disabling ONNX support")
            set(PHOTARA_ENABLE_ONNX OFF CACHE BOOL
                "Enable ONNX Runtime (LightGlue / learned matchers)" FORCE)
            return()
        endif()
        set(PHOTARA_ONNXRUNTIME_ROOT "${_ort_root}" CACHE PATH
            "ONNX Runtime SDK root" FORCE)
        message(STATUS "Fetched ONNX Runtime -> ${_ort_root}")
    else()
        # 2) find_package / path search without FetchContent.
        find_path(_ort_inc onnxruntime_cxx_api.h
            HINTS
                "${PHOTARA_ONNXRUNTIME_ROOT}/include"
                ENV onnxruntime_ROOT
                ENV ONNXRUNTIME_ROOT
            PATH_SUFFIXES include)
        find_library(_ort_lib onnxruntime
            HINTS
                "${PHOTARA_ONNXRUNTIME_ROOT}/lib"
                ENV onnxruntime_ROOT
                ENV ONNXRUNTIME_ROOT
            PATH_SUFFIXES lib lib64)
        if(_ort_inc AND _ort_lib)
            get_filename_component(_ort_lib_dir "${_ort_lib}" DIRECTORY)
            get_filename_component(_ort_root "${_ort_lib_dir}" DIRECTORY)
            set(PHOTARA_ONNXRUNTIME_ROOT "${_ort_root}" CACHE PATH
                "ONNX Runtime SDK root" FORCE)
            message(STATUS "Found ONNX Runtime: ${_ort_root}")
        else()
            message(WARNING
                "PHOTARA_ENABLE_ONNX=ON but ONNX Runtime was not found. "
                "Set PHOTARA_FETCH_ONNX=ON or PHOTARA_ONNXRUNTIME_ROOT. "
                "Disabling ONNX support.")
            set(PHOTARA_ENABLE_ONNX OFF CACHE BOOL
                "Enable ONNX Runtime (LightGlue / learned matchers)" FORCE)
            return()
        endif()
    endif()

    find_path(PHOTARA_ONNXRUNTIME_INCLUDE onnxruntime_cxx_api.h
        HINTS "${_ort_root}/include"
        NO_DEFAULT_PATH
        REQUIRED)
    find_library(PHOTARA_ONNXRUNTIME_LIBRARY onnxruntime
        HINTS "${_ort_root}/lib" "${_ort_root}/lib64"
        NO_DEFAULT_PATH
        REQUIRED)

    if(NOT TARGET onnxruntime::onnxruntime)
        add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${PHOTARA_ONNXRUNTIME_INCLUDE}")
        if(WIN32)
            set_target_properties(onnxruntime::onnxruntime PROPERTIES
                IMPORTED_IMPLIB "${PHOTARA_ONNXRUNTIME_LIBRARY}"
                IMPORTED_LOCATION "${_ort_root}/lib/onnxruntime.dll")
        else()
            set_target_properties(onnxruntime::onnxruntime PROPERTIES
                IMPORTED_LOCATION "${PHOTARA_ONNXRUNTIME_LIBRARY}")
        endif()
    endif()

    set(PHOTARA_ONNXRUNTIME_ROOT "${_ort_root}" PARENT_SCOPE)
    set(PHOTARA_ONNXRUNTIME_INCLUDE "${PHOTARA_ONNXRUNTIME_INCLUDE}" PARENT_SCOPE)
    set(PHOTARA_ONNXRUNTIME_LIBRARY "${PHOTARA_ONNXRUNTIME_LIBRARY}" PARENT_SCOPE)
    set(PHOTARA_HAS_ONNXRUNTIME TRUE PARENT_SCOPE)
    message(STATUS "Enabling ONNX Runtime support (LightGlue)")
endfunction()
