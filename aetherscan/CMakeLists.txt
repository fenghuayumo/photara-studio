find_package(OpenMP)
find_package(CGAL QUIET)

set(AETHERSCAN_BA_SOURCES
    src/ba/linearizer_cpu.cpp
    src/ba/optimizer_cpu.cpp)

if(AETHERSCAN_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit 12.0 REQUIRED)
        list(APPEND AETHERSCAN_BA_SOURCES
            src/ba/linearizer_cuda.cu
            src/ba/optimizer_cuda.cu)
    else()
        message(WARNING "CUDA compiler not found; building CPU-only AetherScan")
        set(AETHERSCAN_ENABLE_CUDA OFF)
        set(AETHERSCAN_ENABLE_SPLAT OFF)
    endif()
endif()

add_library(aetherscan_ba ${AETHERSCAN_BA_SOURCES})
add_library(AetherScan::BA ALIAS aetherscan_ba)

target_include_directories(aetherscan_ba
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
target_compile_features(aetherscan_ba PUBLIC cxx_std_20)

if(OpenMP_CXX_FOUND)
    target_link_libraries(aetherscan_ba PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(aetherscan_ba PUBLIC AETHERSCAN_HAS_OPENMP=1)
endif()

if(AETHERSCAN_ENABLE_CUDA)
    target_link_libraries(aetherscan_ba PUBLIC CUDA::cudart)
    target_compile_definitions(aetherscan_ba PUBLIC AETHERSCAN_HAS_CUDA=1)
    set_target_properties(aetherscan_ba PROPERTIES
        CUDA_ARCHITECTURES native
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

if(MSVC)
    target_compile_options(aetherscan_ba PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /EHsc>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4,-EHsc>)
    if(AETHERSCAN_ENABLE_NATIVE_ARCH)
        target_compile_options(aetherscan_ba PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
    endif()
else()
    target_compile_options(aetherscan_ba PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
    if(AETHERSCAN_ENABLE_NATIVE_ARCH)
        target_compile_options(aetherscan_ba PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
    endif()
endif()

set_target_properties(aetherscan_ba PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(AETHERSCAN_ENABLE_FEATURES)
    find_package(freeimage CONFIG REQUIRED)
    find_package(Eigen3 CONFIG REQUIRED)
    find_package(PoseLib CONFIG REQUIRED)
    # Some Ceres/glog vcpkg exports retain gflags::gflags in their link
    # interface without declaring the dependency themselves.
    set(GFLAGS_USE_TARGET_NAMESPACE TRUE)
    find_package(gflags CONFIG REQUIRED)
    if(MSVC AND gflags_DIR)
        get_filename_component(_gflags_share_dir "${gflags_DIR}" DIRECTORY)
        get_filename_component(_gflags_prefix "${_gflags_share_dir}" DIRECTORY)
        link_directories("${_gflags_prefix}/lib")
    endif()
    find_package(Ceres CONFIG REQUIRED)
    find_package(hnswlib CONFIG REQUIRED)
    find_package(Boost REQUIRED)

    set(AETHERSCAN_VLFEAT_SOURCES
        third_party/vlfeat/vl/generic.c
        third_party/vlfeat/vl/host.c
        third_party/vlfeat/vl/imopv.c
        third_party/vlfeat/vl/mathop.c
        third_party/vlfeat/vl/random.c
        third_party/vlfeat/vl/sift.c)
    set(AETHERSCAN_VLFEAT_HAS_SSE2 FALSE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i[3-6]86)$")
        list(APPEND AETHERSCAN_VLFEAT_SOURCES
            third_party/vlfeat/vl/imopv_sse2.c
            third_party/vlfeat/vl/mathop_sse2.c)
        set(AETHERSCAN_VLFEAT_HAS_SSE2 TRUE)
    endif()
    add_library(aetherscan_vlfeat STATIC ${AETHERSCAN_VLFEAT_SOURCES})
    target_include_directories(aetherscan_vlfeat PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/vlfeat)
    # Outer image-level parallelism owns threading; keep VLFeat itself serial.
    target_compile_definitions(aetherscan_vlfeat PUBLIC VL_DISABLE_THREADS)
    if(AETHERSCAN_VLFEAT_HAS_SSE2)
        # MSVC x64 always has SSE2 but does not define the GCC-style macro that
        # VLFeat's SSE2 translation units require.
        if(MSVC)
            target_compile_definitions(aetherscan_vlfeat PRIVATE __SSE2__)
        endif()
    else()
        target_compile_definitions(aetherscan_vlfeat PUBLIC VL_DISABLE_SSE2)
    endif()
    if(MSVC)
        target_compile_definitions(aetherscan_vlfeat PRIVATE _CRT_SECURE_NO_WARNINGS)
        target_compile_options(aetherscan_vlfeat PRIVATE /W1)
    endif()
    set_target_properties(aetherscan_vlfeat PROPERTIES
        C_STANDARD 99
        POSITION_INDEPENDENT_CODE ON
        LINKER_LANGUAGE C)

    add_library(aetherscan_features
        src/io/image_freeimage.cpp
        src/features/extractor.cpp
        src/features/registry.cpp
        src/features/sift_features.cpp
        src/features/siftgpu_features.cpp
        src/features/superpoint_extractor.cpp
        src/features/aliked_extractor.cpp
        src/features/lightglue_matcher.cpp
        src/features/lightglue_pipeline.cpp)
    add_library(AetherScan::Features ALIAS aetherscan_features)
    target_include_directories(aetherscan_features PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    target_compile_features(aetherscan_features PUBLIC cxx_std_20)
    target_link_libraries(aetherscan_features PUBLIC
        freeimage::FreeImage
        Eigen3::Eigen
        hnswlib::hnswlib
        aetherscan_vlfeat)
    target_compile_definitions(aetherscan_features PUBLIC AETHERSCAN_HAS_FREEIMAGE=1)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(aetherscan_features PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(aetherscan_features PUBLIC AETHERSCAN_HAS_OPENMP=1)
    endif()
    if(MSVC AND AETHERSCAN_ENABLE_NATIVE_ARCH)
        target_compile_options(aetherscan_features PRIVATE /arch:AVX2)
    elseif(AETHERSCAN_ENABLE_NATIVE_ARCH)
        target_compile_options(aetherscan_features PRIVATE -march=native)
    endif()
    if(AETHERSCAN_ENABLE_SIFTGPU)
        find_package(siftgpu CONFIG REQUIRED)
        target_link_libraries(aetherscan_features PRIVATE siftgpu::siftgpu)
        target_compile_definitions(aetherscan_features PRIVATE AETHERSCAN_HAS_SIFTGPU=1)
        if(WIN32)
            add_custom_command(TARGET aetherscan_features POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "$<TARGET_FILE:siftgpu::siftgpu>"
                    "$<TARGET_FILE_DIR:aetherscan_features>")
        endif()
    endif()
    if(AETHERSCAN_HAS_ONNXRUNTIME AND TARGET onnxruntime::onnxruntime)
        target_link_libraries(aetherscan_features PRIVATE onnxruntime::onnxruntime)
        target_compile_definitions(aetherscan_features PRIVATE AETHERSCAN_HAS_ONNXRUNTIME=1)
        if(WIN32)
            set(_ort_lib_dir "${AETHERSCAN_ONNXRUNTIME_ROOT}/lib")
            foreach(_ort_dll IN ITEMS
                    onnxruntime.dll
                    onnxruntime_providers_shared.dll
                    onnxruntime_providers_cuda.dll)
                if(EXISTS "${_ort_lib_dir}/${_ort_dll}")
                    add_custom_command(TARGET aetherscan_features POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                            "${_ort_lib_dir}/${_ort_dll}"
                            "$<TARGET_FILE_DIR:aetherscan_features>/${_ort_dll}")
                endif()
            endforeach()
        endif()
    elseif(AETHERSCAN_ENABLE_ONNX)
        message(WARNING
            "AETHERSCAN_ENABLE_ONNX=ON but onnxruntime::onnxruntime is unavailable; "
            "LightGlue will not be compiled in")
    endif()
    if(MSVC)
        target_compile_options(aetherscan_features PRIVATE /W4 /permissive- /EHsc)
    endif()

    add_library(aetherscan_sfm
        src/sfm/scene.cpp
        src/sfm/checkpoint.cpp
        src/sfm/retrieval.cpp
        src/sfm/vocabulary.cpp
        src/sfm/geometry.cpp
        src/sfm/pair_weighting.cpp
        src/sfm/triangulation.cpp
        src/sfm/tracks.cpp
        src/sfm/bundle.cpp
        src/sfm/star_init.cpp
        src/sfm/resection.cpp
        src/sfm/global_rotation.cpp
        src/sfm/global_positioning.cpp
        src/sfm/hierarchical.cpp
        src/sfm/submap_recovery.cpp
        src/sfm/frontend.cpp
        src/sfm/camera_selection.cpp
        src/sfm/reconstruct.cpp
        src/sfm/export_mvs.cpp
        src/sfm/asfm.cpp
        src/sfm/preview.cpp
        src/project/archive.cpp
        src/project/document.cpp)

    # Derive cache invalidation from the implementation itself. This removes
    # the need to manually bump ad-hoc "vN"/dated ABI strings whenever an SfM,
    # feature, or BA algorithm changes.
    file(GLOB_RECURSE AETHERSCAN_FRONTEND_CACHE_ID_FILES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/include/features/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/sfm/checkpoint.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/sfm/frontend.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/sfm/geometry.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/sfm/pair_weighting.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/sfm/retrieval.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/sfm/scene.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/sfm/tracks.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/sfm/types.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/sfm/vocabulary.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/features/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/io/image_freeimage.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/sfm/checkpoint.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/sfm/frontend.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/sfm/camera_selection.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/sfm/camera_selection.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/sfm/submap_recovery.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/sfm/submap_recovery.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/sfm/reconstruct.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/sfm/geometry.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/sfm/pair_weighting.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/sfm/retrieval.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/sfm/tracks.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/sfm/vocabulary.cpp")
    file(GLOB_RECURSE AETHERSCAN_RECONSTRUCTION_CACHE_ID_FILES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/include/ba/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/sfm/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/ba/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/ba/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/sfm/*.cpp")
    list(SORT AETHERSCAN_FRONTEND_CACHE_ID_FILES)
    list(SORT AETHERSCAN_RECONSTRUCTION_CACHE_ID_FILES)
    set(_aetherscan_cache_build_options
        "cuda=${AETHERSCAN_ENABLE_CUDA};siftgpu=${AETHERSCAN_ENABLE_SIFTGPU};onnx=${AETHERSCAN_HAS_ONNXRUNTIME}\n")
    function(aetherscan_source_fingerprint output files_variable)
        set(_material "${_aetherscan_cache_build_options}")
        foreach(_cache_id_file IN LISTS ${files_variable})
            file(RELATIVE_PATH _cache_id_relative
                "${CMAKE_CURRENT_SOURCE_DIR}" "${_cache_id_file}")
            file(SHA256 "${_cache_id_file}" _cache_id_file_hash)
            string(APPEND _material
                "${_cache_id_relative}:${_cache_id_file_hash}\n")
        endforeach()
        string(SHA256 _fingerprint "${_material}")
        set(${output} "${_fingerprint}" PARENT_SCOPE)
    endfunction()
    aetherscan_source_fingerprint(
        AETHERSCAN_FRONTEND_CACHE_BUILD_ID AETHERSCAN_FRONTEND_CACHE_ID_FILES)
    aetherscan_source_fingerprint(
        AETHERSCAN_RECONSTRUCTION_CACHE_BUILD_ID
        AETHERSCAN_RECONSTRUCTION_CACHE_ID_FILES)

    add_library(AetherScan::SfM ALIAS aetherscan_sfm)
    target_include_directories(aetherscan_sfm PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    target_compile_features(aetherscan_sfm PUBLIC cxx_std_20)
    target_link_libraries(aetherscan_sfm PUBLIC
        AetherScan::Features
        AetherScan::BA
        Eigen3::Eigen
        PoseLib::PoseLib
        Ceres::ceres)
    target_compile_definitions(aetherscan_sfm PRIVATE
        AETHERSCAN_HAS_POSELIB=1
        AETHERSCAN_FRONTEND_CACHE_BUILD_ID="${AETHERSCAN_FRONTEND_CACHE_BUILD_ID}"
        AETHERSCAN_RECONSTRUCTION_CACHE_BUILD_ID="${AETHERSCAN_RECONSTRUCTION_CACHE_BUILD_ID}")
    if(OpenMP_CXX_FOUND)
        # Link explicitly instead of relying on transitive BA usage
        # requirements: triangulation and mapping own OpenMP regions.
        target_link_libraries(aetherscan_sfm PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(aetherscan_sfm PUBLIC AETHERSCAN_HAS_OPENMP=1)
    endif()
    if(MSVC)
        target_compile_options(aetherscan_sfm PRIVATE /W4 /permissive- /EHsc)
        if(AETHERSCAN_ENABLE_NATIVE_ARCH)
            target_compile_options(aetherscan_sfm PRIVATE /arch:AVX2)
        endif()
    else()
        target_compile_options(aetherscan_sfm PRIVATE -Wall -Wextra -Wpedantic)
        if(AETHERSCAN_ENABLE_NATIVE_ARCH)
            target_compile_options(aetherscan_sfm PRIVATE -march=native)
        endif()
    endif()

    add_library(aetherscan_mvs
        src/mvs/scene_build.cpp
        src/mvs/neighbors.cpp
        src/mvs/patchmatch.cpp
        src/mvs/subject_bounds.cpp
        src/mvs/fuse.cpp
        src/mvs/mesh.cpp
        src/mvs/mesh_cgal.cpp
        src/mvs/mesh_clean.cpp
        src/mvs/mesh_tsdf.cpp
        src/mvs/export.cpp
        src/mvs/densify.cpp)
    if(AETHERSCAN_ENABLE_CUDA)
        # Keep CUDA compilation isolated from image-codec targets whose
        # imported MSVC-only options (for example /EHsc) are invalid for nvcc.
        add_library(aetherscan_mvs_cuda STATIC
            src/mvs/patchmatch_cuda.cu)
        target_compile_features(aetherscan_mvs_cuda PUBLIC cxx_std_20)
        target_link_libraries(aetherscan_mvs_cuda PUBLIC CUDA::cudart)
        if(MSVC)
            target_compile_options(aetherscan_mvs_cuda PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4,-EHsc>)
        endif()
        set_target_properties(aetherscan_mvs_cuda PROPERTIES
            CUDA_ARCHITECTURES native
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            POSITION_INDEPENDENT_CODE ON)
        target_link_libraries(aetherscan_mvs PRIVATE aetherscan_mvs_cuda)
        target_compile_definitions(aetherscan_mvs PUBLIC
            AETHERSCAN_MVS_HAS_CUDA=1)
    endif()
    add_library(AetherScan::MVS ALIAS aetherscan_mvs)
    target_include_directories(aetherscan_mvs PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    target_compile_features(aetherscan_mvs PUBLIC cxx_std_20)
    target_link_libraries(aetherscan_mvs PUBLIC
        AetherScan::SfM
        Eigen3::Eigen
        Boost::headers)
    if(CGAL_FOUND)
        target_link_libraries(aetherscan_mvs PUBLIC CGAL::CGAL)
        target_compile_definitions(aetherscan_mvs PUBLIC AETHERSCAN_HAS_CGAL=1)
        message(STATUS "Enabling CGAL global Delaunay MVS meshing")
    else()
        message(WARNING
            "CGAL not found: default/high global MVS meshing is unavailable; "
            "splat TSDF meshing remains supported, but PAM and global "
            "Delaunay meshing require a CGAL-enabled build")
    endif()

    if(AETHERSCAN_ENABLE_SPLAT AND AETHERSCAN_ENABLE_CUDA)
        find_package(glm CONFIG REQUIRED)
        find_package(WebP CONFIG REQUIRED)
        find_package(ZLIB REQUIRED)
        find_package(zstd CONFIG REQUIRED)
        add_subdirectory(
            third_party/tinytensor
            ${CMAKE_CURRENT_BINARY_DIR}/third_party/tinytensor)

        add_library(aetherscan_gggs_reference STATIC
            third_party/gggs_reference/src/rasterizer_impl.cu
            third_party/gggs_reference/src/render_backward.cu
            third_party/gggs_reference/src/render_forward.cu
            third_party/gggs_reference/src/sample_backward.cu
            third_party/gggs_reference/src/sample_forward.cu)
        target_include_directories(aetherscan_gggs_reference PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gggs_reference/include)
        target_link_libraries(aetherscan_gggs_reference PUBLIC
            CUDA::cudart
            glm::glm)
        target_compile_features(aetherscan_gggs_reference PUBLIC cxx_std_20)
        target_compile_definitions(aetherscan_gggs_reference PRIVATE
            AETHERSCAN_GGGS_ACCUTILE=$<BOOL:${AETHERSCAN_ENABLE_ACCUTILE}>)
        target_compile_options(aetherscan_gggs_reference PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr -use_fast_math -lineinfo>)
        set_target_properties(aetherscan_gggs_reference PROPERTIES
            CUDA_ARCHITECTURES native
            CUDA_SEPARABLE_COMPILATION OFF
            POSITION_INDEPENDENT_CODE ON
            FOLDER "external")

        add_library(aetherscan_splat_cuda STATIC
            src/splat/rasterizer.cu
            src/splat/cuda_ops.cu
            src/splat/cuda_vulkan_preview.cu
            src/splat/fused_ssim.cu
            src/splat/visualize.cu)
        target_include_directories(aetherscan_splat_cuda
            PUBLIC
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:include>
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/src/splat)
        target_compile_features(aetherscan_splat_cuda PUBLIC cxx_std_20)
        target_link_libraries(aetherscan_splat_cuda PUBLIC
            tinytensor::tinytensor
            aetherscan_gggs_reference
            CUDA::cudart)
        target_compile_definitions(aetherscan_splat_cuda PUBLIC AETHERSCAN_HAS_SPLAT=1)
        target_compile_options(aetherscan_splat_cuda PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr -use_fast_math>)
        set_target_properties(aetherscan_splat_cuda PROPERTIES
            CUDA_ARCHITECTURES native
            CUDA_SEPARABLE_COMPILATION OFF
            POSITION_INDEPENDENT_CODE ON)

        add_library(aetherscan_splat
            src/splat/colmap.cpp
            src/splat/dataset.cpp
            src/splat/dataset_colmap.cpp
            src/splat/dataset_openmvs.cpp
            src/splat/dataset_reality_capture.cpp
            src/splat/densification.cpp
            src/splat/densification_igs.cpp
            src/splat/densification_adc_plus.cpp
            src/splat/densification_strategies.cpp
            src/splat/formats.cpp
            src/splat/formats_glb.cpp
            src/splat/mesh.cpp
            src/splat/pam_mesh.cpp
            src/splat/training_data_loader.cpp
            src/splat/trainer.cpp)
        add_library(AetherScan::Splat ALIAS aetherscan_splat)
        target_include_directories(aetherscan_splat
            PUBLIC
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:include>
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/src/splat)
        target_compile_features(aetherscan_splat PUBLIC cxx_std_20)
        target_link_libraries(aetherscan_splat PUBLIC
            AetherScan::MVS
            aetherscan_splat_cuda
            WebP::webp
            ZLIB::ZLIB
            zstd::libzstd)
        target_compile_definitions(aetherscan_splat PUBLIC AETHERSCAN_HAS_SPLAT=1)
    endif()
    if(OpenMP_CXX_FOUND)
        target_link_libraries(aetherscan_mvs PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(aetherscan_mvs PUBLIC AETHERSCAN_HAS_OPENMP=1)
    endif()
    if(MSVC)
        target_compile_options(aetherscan_mvs PRIVATE /W4 /permissive- /EHsc)
        if(AETHERSCAN_ENABLE_NATIVE_ARCH)
            target_compile_options(aetherscan_mvs PRIVATE /arch:AVX2)
        endif()
    else()
        target_compile_options(aetherscan_mvs PRIVATE -Wall -Wextra -Wpedantic)
        if(AETHERSCAN_ENABLE_NATIVE_ARCH)
            target_compile_options(aetherscan_mvs PRIVATE -march=native)
        endif()
    endif()

    if(AETHERSCAN_HAS_TEXTURE AND TARGET aether::render)
        add_library(aetherscan_texture
            src/texture/bake.cpp
            src/texture/delight.cpp
            src/texture/export.cpp
            src/texture/mask.cpp)
        add_library(AetherScan::Texture ALIAS aetherscan_texture)
        target_include_directories(aetherscan_texture PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>)
        target_compile_features(aetherscan_texture PUBLIC cxx_std_20)
        target_link_libraries(aetherscan_texture PUBLIC
            AetherScan::MVS
            aether::render)
        target_compile_definitions(aetherscan_texture PUBLIC AETHERSCAN_HAS_TEXTURE=1)
        file(TO_CMAKE_PATH "${AETHERSCAN_INTRINSIC_MODELS_DIR}"
            AETHERSCAN_INTRINSIC_MODELS_DIR)
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/include/texture/paths.hpp.in"
            "${CMAKE_CURRENT_BINARY_DIR}/generated/texture/paths.hpp"
            @ONLY)
        target_include_directories(aetherscan_texture PRIVATE
            "${CMAKE_CURRENT_BINARY_DIR}/generated")
        if(AETHERSCAN_HAS_ONNXRUNTIME AND TARGET onnxruntime::onnxruntime)
            target_link_libraries(aetherscan_texture PRIVATE onnxruntime::onnxruntime)
            target_compile_definitions(aetherscan_texture
                PRIVATE AETHERSCAN_HAS_ONNXRUNTIME=1)
            message(STATUS
                "Enabling texture baking + C++ ONNX Intrinsic delighter")
        else()
            message(STATUS
                "Enabling texture baking; delighter disabled until "
                "AETHERSCAN_ENABLE_ONNX=ON (stage_*.onnx + ORT)")
        endif()
        if(MSVC)
            target_compile_options(aetherscan_texture PRIVATE /W4 /permissive- /EHsc)
            if(AETHERSCAN_ENABLE_NATIVE_ARCH)
                target_compile_options(aetherscan_texture PRIVATE /arch:AVX2)
            endif()
        else()
            target_compile_options(aetherscan_texture PRIVATE -Wall -Wextra -Wpedantic)
            if(AETHERSCAN_ENABLE_NATIVE_ARCH)
                target_compile_options(aetherscan_texture PRIVATE -march=native)
            endif()
        endif()
        endif()
    endif()

    if(AETHERSCAN_BUILD_PYTHON)
        if(NOT TARGET aetherscan_splat)
            message(FATAL_ERROR
                "AETHERSCAN_BUILD_PYTHON requires CUDA splat support")
        endif()
        find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
            RESULT_VARIABLE _aetherscan_nanobind_result
            OUTPUT_VARIABLE nanobind_ROOT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(_aetherscan_nanobind_result EQUAL 0)
            find_package(nanobind CONFIG QUIET)
        endif()
        if(NOT nanobind_FOUND)
            include(FetchContent)
            FetchContent_Declare(
                nanobind
                GIT_REPOSITORY https://github.com/wjakob/nanobind.git
                GIT_TAG v2.13.0
                GIT_SHALLOW TRUE)
            FetchContent_MakeAvailable(nanobind)
        endif()

        nanobind_add_module(
            aetherscan_python
            NB_STATIC
            src/python/bindings.cpp)
        set_target_properties(aetherscan_python PROPERTIES
            OUTPUT_NAME aetherscan_native)
        target_link_libraries(aetherscan_python PRIVATE
            AetherScan::SfM
            AetherScan::MVS
            AetherScan::Splat)
        target_compile_features(aetherscan_python PRIVATE cxx_std_20)
        if(MSVC)
            target_compile_options(aetherscan_python PRIVATE
                /W4 /permissive- /EHsc)
        else()
            target_compile_options(aetherscan_python PRIVATE
                -Wall -Wextra -Wpedantic)
        endif()
        if(AETHERSCAN_BUILD_TESTS)
            add_test(
                NAME aetherscan.python.import
                COMMAND "${Python_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/python_smoke.py"
                    "$<TARGET_FILE_DIR:aetherscan_python>")
        endif()
    endif()

    if(AETHERSCAN_BUILD_BENCHMARKS)
    add_executable(aetherscan_ba_benchmark src/tools/ba_benchmark.cpp)
    target_link_libraries(aetherscan_ba_benchmark PRIVATE AetherScan::BA)
    if(TARGET aetherscan_features)
        add_executable(aetherscan_features_match src/tools/features_match.cpp)
        target_link_libraries(aetherscan_features_match PRIVATE AetherScan::Features)
        if(WIN32)
            add_custom_command(TARGET aetherscan_features_match POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_RUNTIME_DLLS:aetherscan_features_match>
                    $<TARGET_FILE_DIR:aetherscan_features_match>
                COMMAND_EXPAND_LISTS)
        endif()
        add_executable(aetherscan_pair_verify src/tools/pair_verify.cpp)
        add_executable(aetherscan_sfm_realign src/tools/sfm_realign.cpp)
        target_link_libraries(aetherscan_sfm_realign PRIVATE AetherScan::SfM)
        if(WIN32)
            add_custom_command(TARGET aetherscan_sfm_realign POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_RUNTIME_DLLS:aetherscan_sfm_realign>
                    $<TARGET_FILE_DIR:aetherscan_sfm_realign>
                COMMAND_EXPAND_LISTS)
        endif()
        target_link_libraries(aetherscan_pair_verify PRIVATE AetherScan::SfM)
        if(WIN32)
            add_custom_command(TARGET aetherscan_pair_verify POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_RUNTIME_DLLS:aetherscan_pair_verify>
                    $<TARGET_FILE_DIR:aetherscan_pair_verify>
                COMMAND_EXPAND_LISTS)
        endif()
        add_executable(
            aetherscan_triangulation_benchmark
            src/tools/triangulation_benchmark.cpp)
        target_link_libraries(
            aetherscan_triangulation_benchmark PRIVATE AetherScan::SfM)
        if(WIN32)
            add_custom_command(
                TARGET aetherscan_triangulation_benchmark POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_RUNTIME_DLLS:aetherscan_triangulation_benchmark>
                    $<TARGET_FILE_DIR:aetherscan_triangulation_benchmark>
                COMMAND_EXPAND_LISTS)
        endif()
        add_library(aetherscan_cxxopts INTERFACE)
        add_library(cxxopts::cxxopts ALIAS aetherscan_cxxopts)
        target_include_directories(aetherscan_cxxopts INTERFACE
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cxxopts/include)

        add_executable(aetherscan src/tools/reconstruct.cpp)
        target_link_libraries(aetherscan PRIVATE AetherScan::MVS cxxopts::cxxopts)
        if(TARGET aetherscan_splat)
            add_executable(aetherscan_mvs_compare src/tools/mvs_compare.cpp)
            target_link_libraries(aetherscan_mvs_compare PRIVATE AetherScan::Splat AetherScan::MVS)
            target_link_libraries(aetherscan PRIVATE AetherScan::Splat)
            target_compile_definitions(aetherscan PRIVATE AETHERSCAN_HAS_SPLAT=1)
        endif()
        if(TARGET aetherscan_texture)
            target_link_libraries(aetherscan PRIVATE AetherScan::Texture)
            target_compile_definitions(aetherscan PRIVATE AETHERSCAN_HAS_TEXTURE=1)
        endif()
        if(TARGET aether::mesh)
            target_link_libraries(aetherscan PRIVATE aether::mesh)
            target_compile_definitions(
                aetherscan PRIVATE AETHERSCAN_HAS_AETHER_MESH=1)
        endif()
        if(WIN32)
            add_custom_command(TARGET aetherscan POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_RUNTIME_DLLS:aetherscan>
                    $<TARGET_FILE_DIR:aetherscan>
                COMMAND_EXPAND_LISTS)
        endif()

        if(AETHERSCAN_BUILD_EDITOR)
            find_package(Vulkan 1.2 REQUIRED)
            find_package(glfw3 CONFIG REQUIRED)
            include(FetchContent)
            get_filename_component(
                _aetherscan_vcpkg_root "${glfw3_DIR}/../../../.." ABSOLUTE)
            # DockSpace / DockBuilder live on the docking branch. Prefer a local
            # vcpkg docking tree when present so configure stays offline.
            file(GLOB _aetherscan_imgui_local
                "${_aetherscan_vcpkg_root}/buildtrees/imgui/src/*docking*.clean")
            if(_aetherscan_imgui_local)
                list(GET _aetherscan_imgui_local 0 _aetherscan_imgui_source)
                FetchContent_Declare(aetherscan_imgui
                    SOURCE_DIR "${_aetherscan_imgui_source}")
            else()
                FetchContent_Declare(
                    aetherscan_imgui
                    GIT_REPOSITORY https://github.com/ocornut/imgui.git
                    GIT_TAG v1.91.9-docking
                    GIT_SHALLOW TRUE)
            endif()
            FetchContent_MakeAvailable(aetherscan_imgui)
            add_library(aetherscan_imgui STATIC
                ${aetherscan_imgui_SOURCE_DIR}/imgui.cpp
                ${aetherscan_imgui_SOURCE_DIR}/imgui_draw.cpp
                ${aetherscan_imgui_SOURCE_DIR}/imgui_tables.cpp
                ${aetherscan_imgui_SOURCE_DIR}/imgui_widgets.cpp
                ${aetherscan_imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
                ${aetherscan_imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp)
            target_include_directories(aetherscan_imgui PUBLIC
                ${aetherscan_imgui_SOURCE_DIR}
                ${aetherscan_imgui_SOURCE_DIR}/backends)
            target_link_libraries(aetherscan_imgui PUBLIC
                Vulkan::Vulkan glfw)
            target_compile_features(aetherscan_imgui PUBLIC cxx_std_20)

            set(AETHERSCAN_EDITOR_DIR
                "${CMAKE_CURRENT_SOURCE_DIR}/../apps/editor")
            add_executable(aetherscan_editor
                ${AETHERSCAN_EDITOR_DIR}/console_view.cpp
                ${AETHERSCAN_EDITOR_DIR}/icons.cpp
                ${AETHERSCAN_EDITOR_DIR}/image_qa_view.cpp
                ${AETHERSCAN_EDITOR_DIR}/main.cpp
                ${AETHERSCAN_EDITOR_DIR}/pipeline.cpp
                ${AETHERSCAN_EDITOR_DIR}/sparse_view.cpp
                ${AETHERSCAN_EDITOR_DIR}/theme.cpp
                ${AETHERSCAN_EDITOR_DIR}/viewport_gizmo.cpp
                ${AETHERSCAN_EDITOR_DIR}/vulkan_backend.cpp)
            target_include_directories(aetherscan_editor PRIVATE
                ${AETHERSCAN_EDITOR_DIR})
            target_link_libraries(aetherscan_editor PRIVATE
                aetherscan_imgui AetherScan::SfM AetherScan::Splat
                Vulkan::Vulkan glfw)
            if(WIN32)
                # Native folder picker (IFileOpenDialog) and Explorer reveal.
                target_link_libraries(aetherscan_editor PRIVATE ole32 shell32)
            endif()
            target_compile_features(aetherscan_editor PRIVATE cxx_std_20)
            if(MSVC)
                target_compile_options(aetherscan_editor PRIVATE /utf-8)
            endif()
            target_compile_definitions(aetherscan_editor PRIVATE
                AETHERSCAN_CLI_PATH="$<TARGET_FILE:aetherscan>")
            add_dependencies(aetherscan_editor aetherscan)
            if(WIN32)
                add_custom_command(TARGET aetherscan_editor POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        $<TARGET_RUNTIME_DLLS:aetherscan_editor>
                        $<TARGET_FILE_DIR:aetherscan_editor>
                    COMMAND_EXPAND_LISTS)
            endif()
        endif()
    endif()
endif()

if(AETHERSCAN_BUILD_TESTS AND BUILD_TESTING)
    add_executable(aetherscan_ba_test tests/ba_optimizer_test.cpp)
    target_link_libraries(aetherscan_ba_test PRIVATE AetherScan::BA)
    add_test(NAME aetherscan.ba.optimizer COMMAND aetherscan_ba_test)
    add_executable(aetherscan_thread_pool_test tests/thread_pool_test.cpp)
    target_link_libraries(aetherscan_thread_pool_test PRIVATE AetherScan::BA)
    add_test(NAME aetherscan.parallel.thread_pool COMMAND aetherscan_thread_pool_test)
    add_executable(aetherscan_logging_test tests/logging_test.cpp)
    target_link_libraries(aetherscan_logging_test PRIVATE AetherScan::BA)
    add_test(NAME aetherscan.core.logging COMMAND aetherscan_logging_test)
    if(TARGET aetherscan_splat)
        add_executable(aetherscan_splat_test tests/splat_test.cpp)
        target_link_libraries(aetherscan_splat_test PRIVATE AetherScan::Splat)
        add_test(NAME aetherscan.splat.rasterizer COMMAND aetherscan_splat_test)
    endif()
    if(TARGET aetherscan_features)
        add_executable(aetherscan_features_test tests/features_test.cpp)
        target_link_libraries(aetherscan_features_test PRIVATE AetherScan::Features)
        if(WIN32)
            add_custom_command(TARGET aetherscan_features_test POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_RUNTIME_DLLS:aetherscan_features_test>
                    $<TARGET_FILE_DIR:aetherscan_features_test>
                COMMAND_EXPAND_LISTS)
        endif()
        add_test(NAME aetherscan.features COMMAND aetherscan_features_test)
        add_executable(aetherscan_two_view_test tests/two_view_test.cpp)
        target_link_libraries(aetherscan_two_view_test PRIVATE AetherScan::SfM)
        add_test(NAME aetherscan.sfm.two_view COMMAND aetherscan_two_view_test)
        add_executable(aetherscan_mapping_test tests/mapping_test.cpp)
        target_link_libraries(aetherscan_mapping_test PRIVATE AetherScan::SfM)
        add_test(NAME aetherscan.sfm.mapping COMMAND aetherscan_mapping_test)
        add_executable(aetherscan_vocabulary_test tests/vocabulary_test.cpp)
        target_link_libraries(aetherscan_vocabulary_test PRIVATE AetherScan::SfM)
        add_test(NAME aetherscan.sfm.vocabulary COMMAND aetherscan_vocabulary_test)
        add_executable(aetherscan_checkpoint_test tests/checkpoint_test.cpp)
        target_link_libraries(aetherscan_checkpoint_test PRIVATE AetherScan::SfM)
        add_test(NAME aetherscan.sfm.checkpoint COMMAND aetherscan_checkpoint_test)
        add_executable(aetherscan_hierarchical_test tests/hierarchical_test.cpp)
        add_executable(aetherscan_submap_recovery_test tests/submap_recovery_test.cpp)
        target_link_libraries(aetherscan_submap_recovery_test PRIVATE AetherScan::SfM)
        add_test(NAME aetherscan.sfm.submap_recovery COMMAND aetherscan_submap_recovery_test)
        target_link_libraries(aetherscan_hierarchical_test PRIVATE AetherScan::SfM)
        add_test(NAME aetherscan.sfm.hierarchical COMMAND aetherscan_hierarchical_test)
        add_executable(aetherscan_export_mvs_test tests/export_mvs_test.cpp)
        target_link_libraries(aetherscan_export_mvs_test PRIVATE AetherScan::SfM)
        add_test(NAME aetherscan.sfm.export_mvs COMMAND aetherscan_export_mvs_test)
        add_executable(aetherscan_project_io_test tests/project_io_test.cpp)
        target_link_libraries(aetherscan_project_io_test PRIVATE AetherScan::SfM)
        add_test(NAME aetherscan.project.io COMMAND aetherscan_project_io_test)
        add_executable(aetherscan_mvs_test tests/mvs_test.cpp)
        target_link_libraries(aetherscan_mvs_test PRIVATE AetherScan::MVS)
        add_test(NAME aetherscan.mvs.pipeline COMMAND aetherscan_mvs_test)
    endif()
endif()
