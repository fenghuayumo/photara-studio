# Resolve Photara third-party packages. vcpkg CONFIG exports are preferred;
# on Linux we also accept distro packages and, when PoseLib is missing,
# fetch it from upstream.

function(photara_imported_library target include_dir library)
    if(TARGET "${target}")
        return()
    endif()
    add_library("${target}" UNKNOWN IMPORTED)
    set_target_properties("${target}" PROPERTIES
        IMPORTED_LOCATION "${library}"
        INTERFACE_INCLUDE_DIRECTORIES "${include_dir}")
endfunction()

function(photara_find_freeimage)
    find_package(freeimage CONFIG QUIET)
    if(TARGET freeimage::FreeImage)
        return()
    endif()
    find_path(PHOTARA_FREEIMAGE_INCLUDE FreeImage.h)
    find_library(PHOTARA_FREEIMAGE_LIBRARY NAMES freeimage FreeImage)
    if(NOT PHOTARA_FREEIMAGE_INCLUDE OR NOT PHOTARA_FREEIMAGE_LIBRARY)
        message(FATAL_ERROR
            "FreeImage not found. Install libfreeimage-dev or a vcpkg freeimage port.")
    endif()
    photara_imported_library(
        freeimage::FreeImage
        "${PHOTARA_FREEIMAGE_INCLUDE}"
        "${PHOTARA_FREEIMAGE_LIBRARY}")
    # Ubuntu FreeImage is linked against JPEG-XR (libjxrglue); pull it in when present.
    find_library(PHOTARA_JXRGLUE_LIBRARY NAMES jxrglue)
    if(PHOTARA_JXRGLUE_LIBRARY)
        set_property(TARGET freeimage::FreeImage APPEND PROPERTY
            INTERFACE_LINK_LIBRARIES "${PHOTARA_JXRGLUE_LIBRARY}")
        message(STATUS "FreeImage JPEG-XR: ${PHOTARA_JXRGLUE_LIBRARY}")
    endif()
    message(STATUS "FreeImage: ${PHOTARA_FREEIMAGE_LIBRARY}")
endfunction()

function(photara_find_hnswlib)
    find_package(hnswlib CONFIG QUIET)
    if(TARGET hnswlib::hnswlib)
        return()
    endif()
    find_path(PHOTARA_HNSWLIB_INCLUDE hnswlib/hnswlib.h)
    if(NOT PHOTARA_HNSWLIB_INCLUDE)
        message(FATAL_ERROR
            "hnswlib not found. Install libhnswlib-dev or a vcpkg hnswlib port.")
    endif()
    if(NOT TARGET hnswlib::hnswlib)
        add_library(hnswlib::hnswlib INTERFACE IMPORTED)
        set_target_properties(hnswlib::hnswlib PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${PHOTARA_HNSWLIB_INCLUDE}")
    endif()
    message(STATUS "hnswlib headers: ${PHOTARA_HNSWLIB_INCLUDE}")
endfunction()

function(photara_find_poselib)
    find_package(PoseLib CONFIG QUIET)
    if(TARGET PoseLib::PoseLib)
        return()
    endif()
    include(FetchContent)
    FetchContent_Declare(photara_poselib
        GIT_REPOSITORY https://github.com/PoseLib/PoseLib.git
        GIT_TAG v2.0.5
        GIT_SHALLOW TRUE)
    set(POSELIB_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(photara_poselib)
    if(NOT TARGET PoseLib::PoseLib)
        if(TARGET PoseLib)
            add_library(PoseLib::PoseLib ALIAS PoseLib)
        elseif(TARGET poselib)
            add_library(PoseLib::PoseLib ALIAS poselib)
        else()
            message(FATAL_ERROR "Fetched PoseLib but PoseLib::PoseLib was not created")
        endif()
    endif()
    message(STATUS "PoseLib fetched from upstream")
endfunction()

function(photara_find_webpzstd)
    # vcpkg's exported targets select the matching Debug/Release binaries.
    # A plain find_library can pick debug/lib first and use it for all configs.
    if(WIN32)
        find_package(WebP CONFIG QUIET)
        find_package(zstd CONFIG QUIET)
    endif()
    # Prefer distro libs over conda CONFIG packages that can poison the link
    # line (mismatched libtiff/libjpeg).
    set(_photara_webp_hints /usr/lib/x86_64-linux-gnu /usr/lib)
    set(_photara_zstd_hints /usr/lib/x86_64-linux-gnu /usr/lib)

    find_path(PHOTARA_WEBP_INCLUDE webp/decode.h
        HINTS /usr/include
        NO_CMAKE_ENVIRONMENT_PATH NO_SYSTEM_ENVIRONMENT_PATH)
    find_library(PHOTARA_WEBP_LIBRARY NAMES webp
        HINTS ${_photara_webp_hints}
        NO_CMAKE_ENVIRONMENT_PATH NO_SYSTEM_ENVIRONMENT_PATH)
    find_library(PHOTARA_WEBPDEMUX_LIBRARY NAMES webpdemux
        HINTS ${_photara_webp_hints}
        NO_CMAKE_ENVIRONMENT_PATH NO_SYSTEM_ENVIRONMENT_PATH)
    find_library(PHOTARA_WEBPMUX_LIBRARY NAMES webpmux
        HINTS ${_photara_webp_hints}
        NO_CMAKE_ENVIRONMENT_PATH NO_SYSTEM_ENVIRONMENT_PATH)
    if(NOT TARGET WebP::webp)
        if(NOT PHOTARA_WEBP_INCLUDE OR NOT PHOTARA_WEBP_LIBRARY)
            find_package(WebP CONFIG QUIET)
        endif()
        if(NOT TARGET WebP::webp)
            if(NOT PHOTARA_WEBP_INCLUDE OR NOT PHOTARA_WEBP_LIBRARY)
                message(FATAL_ERROR
                    "WebP not found. Install libwebp-dev or a vcpkg libwebp port.")
            endif()
            photara_imported_library(
                WebP::webp "${PHOTARA_WEBP_INCLUDE}" "${PHOTARA_WEBP_LIBRARY}")
            if(PHOTARA_WEBPDEMUX_LIBRARY AND NOT TARGET WebP::webpdemux)
                photara_imported_library(
                    WebP::webpdemux "${PHOTARA_WEBP_INCLUDE}" "${PHOTARA_WEBPDEMUX_LIBRARY}")
            endif()
            if(PHOTARA_WEBPMUX_LIBRARY AND NOT TARGET WebP::webpmux)
                photara_imported_library(
                    WebP::webpmux "${PHOTARA_WEBP_INCLUDE}" "${PHOTARA_WEBPMUX_LIBRARY}")
            endif()
            message(STATUS "WebP: ${PHOTARA_WEBP_LIBRARY}")
        endif()
    endif()

    if(NOT TARGET zstd::libzstd AND NOT TARGET zstd::libzstd_shared AND NOT TARGET zstd::libzstd_static)
        find_path(PHOTARA_ZSTD_INCLUDE zstd.h
            HINTS /usr/include
            NO_CMAKE_ENVIRONMENT_PATH NO_SYSTEM_ENVIRONMENT_PATH)
        find_library(PHOTARA_ZSTD_LIBRARY NAMES zstd
            HINTS ${_photara_zstd_hints}
            NO_CMAKE_ENVIRONMENT_PATH NO_SYSTEM_ENVIRONMENT_PATH)
        if(NOT PHOTARA_ZSTD_INCLUDE OR NOT PHOTARA_ZSTD_LIBRARY)
            find_package(zstd CONFIG QUIET)
        endif()
    endif()
    if(NOT TARGET zstd::libzstd AND NOT TARGET zstd::libzstd_shared AND NOT TARGET zstd::libzstd_static)
        if(NOT PHOTARA_ZSTD_INCLUDE OR NOT PHOTARA_ZSTD_LIBRARY)
            message(FATAL_ERROR
                "zstd not found. Install libzstd-dev or a vcpkg zstd port.")
        endif()
        photara_imported_library(
            zstd::libzstd "${PHOTARA_ZSTD_INCLUDE}" "${PHOTARA_ZSTD_LIBRARY}")
        add_library(zstd::libzstd_shared ALIAS zstd::libzstd)
        message(STATUS "zstd: ${PHOTARA_ZSTD_LIBRARY}")
    elseif(TARGET zstd::libzstd_shared AND NOT TARGET zstd::libzstd)
        add_library(zstd::libzstd ALIAS zstd::libzstd_shared)
    elseif(TARGET zstd::libzstd_static AND NOT TARGET zstd::libzstd)
        add_library(zstd::libzstd ALIAS zstd::libzstd_static)
    endif()
endfunction()

function(photara_find_glfw)
    find_package(glfw3 CONFIG QUIET)
    if(TARGET glfw)
        return()
    endif()
    find_package(glfw3 QUIET)
    if(TARGET glfw)
        return()
    endif()
    find_path(PHOTARA_GLFW_INCLUDE GLFW/glfw3.h)
    find_library(PHOTARA_GLFW_LIBRARY NAMES glfw glfw3)
    if(NOT PHOTARA_GLFW_INCLUDE OR NOT PHOTARA_GLFW_LIBRARY)
        message(FATAL_ERROR
            "GLFW not found. Install libglfw3-dev or a vcpkg glfw3 port.")
    endif()
    photara_imported_library(glfw "${PHOTARA_GLFW_INCLUDE}" "${PHOTARA_GLFW_LIBRARY}")
    message(STATUS "GLFW: ${PHOTARA_GLFW_LIBRARY}")
endfunction()
