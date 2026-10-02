find_package(OpenMP)
find_package(CGAL QUIET)

set(PHOTARA_BA_SOURCES
    src/ba/linearizer_cpu.cpp
    src/ba/optimizer_cpu.cpp)

if(PHOTARA_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        photara_resolve_cuda_architectures(
            _photara_cuda_architectures)
        set(CMAKE_CUDA_ARCHITECTURES
            "${_photara_cuda_architectures}")
        message(STATUS
            "Photara CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
        find_package(CUDAToolkit 12.0 REQUIRED)
        list(APPEND PHOTARA_BA_SOURCES
            src/ba/linearizer_cuda.cu
            src/ba/bearing_cuda.cu
            src/ba/optimizer_cuda.cu)
    else()
        message(WARNING "CUDA compiler not found; building CPU-only Photara")
        set(PHOTARA_ENABLE_CUDA OFF)
        set(PHOTARA_ENABLE_SPLAT OFF)
    endif()
endif()

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/core/version.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/core/version.hpp"
    @ONLY)

add_library(photara_ba ${PHOTARA_BA_SOURCES})
add_library(Photara::BA ALIAS photara_ba)

target_include_directories(photara_ba
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>)
target_compile_features(photara_ba PUBLIC cxx_std_20)

if(OpenMP_CXX_FOUND)
    target_link_libraries(photara_ba PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(photara_ba PUBLIC PHOTARA_HAS_OPENMP=1)
endif()

if(PHOTARA_ENABLE_CUDA)
    target_link_libraries(photara_ba PUBLIC CUDA::cudart)
    target_compile_definitions(photara_ba PUBLIC PHOTARA_HAS_CUDA=1)
    set_target_properties(photara_ba PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

if(MSVC)
    target_compile_options(photara_ba PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /EHsc>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4,-EHsc>)
    if(PHOTARA_ENABLE_NATIVE_ARCH)
        target_compile_options(photara_ba PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
    endif()
else()
    target_compile_options(photara_ba PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
    if(PHOTARA_ENABLE_NATIVE_ARCH)
        target_compile_options(photara_ba PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
    endif()
endif()

set_target_properties(photara_ba PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(PHOTARA_HAS_VULKAN_BA FALSE)
if(PHOTARA_ENABLE_VK AND TARGET photara_vk::photara_vk AND PHOTARA_VK_DXC)
    set(PHOTARA_BEARING_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/ba/vulkan")
    set(PHOTARA_BEARING_SHADERS
        bearing_linearize.hlsl
        bearing_sum_cost.hlsl
        bearing_diagonal_scale.hlsl
        bearing_points_inverse.hlsl
        bearing_eliminate.hlsl
        bearing_camera_blocks.hlsl
        bearing_schur.hlsl
        bearing_baseline_gauge.hlsl
        bearing_chol_pivot.hlsl
        bearing_chol_rank1.hlsl
        bearing_chol_forward.hlsl
        bearing_chol_backward.hlsl
        bearing_update_points.hlsl
        bearing_update_cameras.hlsl
        bearing_model_decrease.hlsl
        bearing_sum_model.hlsl)
    set(PHOTARA_BA_OPTIMIZER_SHADERS
        ba_linearize.hlsl
        ba_assemble_points.hlsl
        ba_assemble_cameras.hlsl
        ba_damp_cameras.hlsl
        ba_finalize_intrinsics.hlsl
        ba_invert_points.hlsl
        ba_reduced_rhs.hlsl
        ba_schur_rhs.hlsl
        ba_copy_intrinsic_rhs.hlsl
        ba_intrinsic_schur.hlsl
        ba_point_gather.hlsl
        ba_point_reduce.hlsl
        ba_camera_gather.hlsl
        ba_camera_reduce.hlsl
        ba_factor.hlsl
        ba_precondition.hlsl
        ba_dot.hlsl
        ba_intrinsic_reduce.hlsl
        ba_precondition_intrinsics.hlsl
        ba_update_pcg.hlsl
        ba_direction.hlsl
        ba_alpha.hlsl
        ba_beta.hlsl
        ba_recover_points.hlsl
        ba_update_poses.hlsl
        ba_update_points.hlsl
        ba_mask_intrinsics.hlsl
        ba_update_intrinsics.hlsl
        ba_prior_cost.hlsl
        ba_linearized_cost.hlsl)
    set(PHOTARA_BEARING_SHADER_PATHS)
    foreach(shader IN LISTS PHOTARA_BEARING_SHADERS PHOTARA_BA_OPTIMIZER_SHADERS)
        list(APPEND PHOTARA_BEARING_SHADER_PATHS
            "${PHOTARA_BEARING_SHADER_DIR}/${shader}")
    endforeach()
    target_sources(photara_ba PRIVATE
        src/ba/bearing_vulkan.cpp
        src/ba/optimizer_vulkan.cpp)
    include("${PHOTARA_VK_CMAKE_DIR}/PhotaraVkShaders.cmake")
    photara_vk_embed_hlsl(
        TARGET photara_ba
        SHADERS ${PHOTARA_BEARING_SHADER_PATHS}
        PROFILE cs_6_0
        ENV vulkan1.2
        INCLUDE_DIR "${PHOTARA_BEARING_SHADER_DIR}"
        OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/bearing_vulkan"
        DEPENDS
            "${PHOTARA_BEARING_SHADER_DIR}/bearing_common.hlsli"
            "${PHOTARA_BEARING_SHADER_DIR}/ba_common.hlsli"
            "${PHOTARA_BEARING_SHADER_DIR}/ba_atomic.hlsli"
            "${PHOTARA_BEARING_SHADER_DIR}/ba_trig.hlsli")
    target_link_libraries(photara_ba PRIVATE photara_vk::photara_vk)
    target_compile_definitions(photara_ba PUBLIC PHOTARA_HAS_VULKAN_BA=1)
    set(PHOTARA_HAS_VULKAN_BA TRUE)
    message(STATUS "Photara Vulkan SfM solvers: ON")
else()
    message(STATUS "Photara Vulkan SfM solvers: OFF")
endif()

if(PHOTARA_ENABLE_FEATURES)
    include(PhotaraFindDeps)
    photara_find_freeimage()
    find_package(JPEG REQUIRED)
    find_package(PNG REQUIRED)
    find_package(Eigen3 CONFIG REQUIRED)
    photara_find_poselib()
    # Some Ceres/glog vcpkg exports retain gflags::gflags in their link
    # interface without declaring the dependency themselves. Ubuntu's Ceres
    # instead expects the un-namespaced `gflags` imported target.
    if(WIN32)
        set(GFLAGS_USE_TARGET_NAMESPACE TRUE)
    endif()
    find_package(gflags CONFIG REQUIRED)
    # Ubuntu Ceres exports glog::glog on its link interface; resolve it first.
    find_package(glog CONFIG REQUIRED)
    if(MSVC AND gflags_DIR)
        get_filename_component(_gflags_share_dir "${gflags_DIR}" DIRECTORY)
        get_filename_component(_gflags_prefix "${_gflags_share_dir}" DIRECTORY)
        link_directories("${_gflags_prefix}/lib")
    endif()
    find_package(Ceres CONFIG REQUIRED)
    photara_find_hnswlib()
    find_package(Boost REQUIRED)

    add_library(photara_features
        src/io/image_freeimage.cpp
        src/io/video_frames.cpp
        src/features/extractor.cpp
        src/features/registry.cpp
        src/features/sift_cpu.cpp
        src/features/sift_features.cpp
        src/features/siftgpu_features.cpp
        src/features/superpoint_extractor.cpp
        src/features/aliked_extractor.cpp
        src/features/lightglue_matcher.cpp
        src/features/lightglue_pipeline.cpp)
    add_library(Photara::Features ALIAS photara_features)
    option(PHOTARA_ENABLE_VULKAN_FEATURES
        "Vulkan compute feature extraction/matching (HLSL/DXC)" ON)
    if(PHOTARA_ENABLE_VULKAN_FEATURES)
        find_package(Vulkan 1.1 QUIET)
        find_program(PHOTARA_FEATURES_DXC_EXECUTABLE
            NAMES dxc
            HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
        if(NOT Vulkan_FOUND OR NOT PHOTARA_FEATURES_DXC_EXECUTABLE)
            message(STATUS
                "PHOTARA_ENABLE_VULKAN_FEATURES=ON but Vulkan SDK 1.1+ or dxc "
                "was not found; the Vulkan feature backend is disabled")
            set(PHOTARA_ENABLE_VULKAN_FEATURES OFF)
        endif()
    endif()
    if(PHOTARA_ENABLE_VULKAN_FEATURES)
        set(PHOTARA_FEATURE_SHADER_DIR
            "${CMAKE_CURRENT_SOURCE_DIR}/src/features/vulkan/shaders")
        set(PHOTARA_FEATURE_SHADER_SOURCES
            sift_convert.hlsl
            sift_upsample.hlsl
            sift_downsample.hlsl
            sift_blur.hlsl
            sift_dog.hlsl
            sift_key.hlsl
            sift_hist_init.hlsl
            sift_hist_reduce.hlsl
            sift_list_gen.hlsl
            sift_orientation.hlsl
            sift_descriptor.hlsl
            sift_descriptor_norm.hlsl
            sift_match_tiles.hlsl
            sift_match_tiles_64.hlsl
            sift_match_tiles_160.hlsl
            sift_match_tiles_dp4a_64.hlsl
            sift_match_tiles_dp4a_128.hlsl
            sift_match_tiles_dp4a_160.hlsl
            sift_match_wave_dp4a.hlsl
            sift_match_finish.hlsl)
        # Every shader includes sift_common.hlsli, and the dp4a variant includes
        # sift_match_tiles.hlsl; track them all so a shared-file edit rebuilds
        # the binaries that include it. Integer-dot-product variants need SM 6.4
        # and emit SPIR-V 1.5, which the host only selects when the device has it.
        file(GLOB PHOTARA_FEATURE_SHADER_DEPENDS
            "${PHOTARA_FEATURE_SHADER_DIR}/*.hlsl"
            "${PHOTARA_FEATURE_SHADER_DIR}/*.hlsli")
        set(PHOTARA_FEATURE_SHADER_CS60)
        set(PHOTARA_FEATURE_SHADER_DP4A)
        foreach(shader IN LISTS PHOTARA_FEATURE_SHADER_SOURCES)
            if(shader MATCHES "dp4a")
                list(APPEND PHOTARA_FEATURE_SHADER_DP4A
                    "${PHOTARA_FEATURE_SHADER_DIR}/${shader}")
            else()
                list(APPEND PHOTARA_FEATURE_SHADER_CS60
                    "${PHOTARA_FEATURE_SHADER_DIR}/${shader}")
            endif()
        endforeach()
        if(NOT TARGET photara_vk::photara_vk)
            add_subdirectory(
                "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/photara_vk"
                "${CMAKE_CURRENT_BINARY_DIR}/photara_vk")
        endif()
        if(NOT PHOTARA_VK_DXC)
            set(PHOTARA_VK_DXC "${PHOTARA_FEATURES_DXC_EXECUTABLE}")
        endif()
        include("${PHOTARA_VK_CMAKE_DIR}/PhotaraVkShaders.cmake")
        target_sources(photara_features PRIVATE
            src/features/vulkan/vk_context.cpp
            src/features/vulkan/sift_vk.cpp
            src/features/vulkan/matcher_vk.cpp)
        photara_vk_embed_hlsl(
            TARGET photara_features
            SHADERS ${PHOTARA_FEATURE_SHADER_CS60}
            PROFILE cs_6_0
            ENV vulkan1.1
            INCLUDE_DIR "${PHOTARA_FEATURE_SHADER_DIR}"
            OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/features_vulkan"
            DEPENDS ${PHOTARA_FEATURE_SHADER_DEPENDS})
        photara_vk_embed_hlsl(
            TARGET photara_features
            SHADERS ${PHOTARA_FEATURE_SHADER_DP4A}
            PROFILE cs_6_4
            ENV vulkan1.2
            INCLUDE_DIR "${PHOTARA_FEATURE_SHADER_DIR}"
            OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/features_vulkan"
            DEPENDS ${PHOTARA_FEATURE_SHADER_DEPENDS})
        target_include_directories(photara_features PRIVATE
            "${CMAKE_CURRENT_BINARY_DIR}/features_vulkan")
        target_link_libraries(photara_features PRIVATE Vulkan::Vulkan photara_vk::photara_vk)
        target_compile_definitions(photara_features PRIVATE
            PHOTARA_HAS_VULKAN_FEATURES=1)
        message(STATUS "Photara Vulkan feature backend: ON "
            "(dxc=${PHOTARA_FEATURES_DXC_EXECUTABLE})")
        add_executable(photara_vulkan_features_test tests/vulkan_features_test.cpp)
        target_link_libraries(photara_vulkan_features_test PRIVATE Photara::Features)
        target_compile_features(photara_vulkan_features_test PRIVATE cxx_std_20)
        if(WIN32)
            add_custom_command(TARGET photara_vulkan_features_test POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_RUNTIME_DLLS:photara_vulkan_features_test>
                    $<TARGET_FILE_DIR:photara_vulkan_features_test>
                COMMAND_EXPAND_LISTS)
        endif()
        add_test(NAME photara.features.vulkan COMMAND photara_vulkan_features_test)
        # Manual parity/performance harness (not a ctest entry): compares the
        # Vulkan SIFT translation and matcher against the SiftGPU CUDA path.
        add_executable(photara_vulkan_features_parity
            tests/vulkan_features_parity.cpp)
        target_link_libraries(photara_vulkan_features_parity
            PRIVATE Photara::Features)
        target_compile_features(photara_vulkan_features_parity PRIVATE cxx_std_20)
        if(WIN32)
            add_custom_command(TARGET photara_vulkan_features_parity POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_RUNTIME_DLLS:photara_vulkan_features_parity>
                    $<TARGET_FILE_DIR:photara_vulkan_features_parity>
                COMMAND_EXPAND_LISTS)
        endif()
    endif()
    if(PHOTARA_ENABLE_CUDA)
        add_library(photara_match_cuda STATIC
            src/features/matcher_cuda.cu
            src/features/sift_cuda.cu)
        target_include_directories(photara_match_cuda PRIVATE include)
        target_compile_features(photara_match_cuda PUBLIC cxx_std_20)
        target_link_libraries(photara_match_cuda PUBLIC CUDA::cudart)
        set_target_properties(photara_match_cuda PROPERTIES
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON)
        target_link_libraries(photara_features PRIVATE photara_match_cuda)
        target_compile_definitions(photara_features PRIVATE PHOTARA_MATCH_HAS_CUDA=1)
    endif()
    target_include_directories(photara_features PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    target_compile_features(photara_features PUBLIC cxx_std_20)
    target_link_libraries(photara_features PUBLIC
        freeimage::FreeImage
        JPEG::JPEG
        PNG::PNG
        Eigen3::Eigen
        hnswlib::hnswlib)
    target_compile_definitions(photara_features PUBLIC PHOTARA_HAS_FREEIMAGE=1)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(photara_features PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(photara_features PUBLIC PHOTARA_HAS_OPENMP=1)
    endif()
    if(MSVC AND PHOTARA_ENABLE_NATIVE_ARCH)
        target_compile_options(photara_features PRIVATE /arch:AVX2)
    elseif(PHOTARA_ENABLE_NATIVE_ARCH)
        target_compile_options(photara_features PRIVATE -march=native)
    endif()
    if(PHOTARA_HAS_ONNXRUNTIME AND TARGET onnxruntime::onnxruntime)
        target_link_libraries(photara_features PRIVATE onnxruntime::onnxruntime)
        target_compile_definitions(photara_features PRIVATE PHOTARA_HAS_ONNXRUNTIME=1)
        if(WIN32)
            set(_ort_lib_dir "${PHOTARA_ONNXRUNTIME_ROOT}/lib")
            foreach(_ort_dll IN ITEMS
                    onnxruntime.dll
                    onnxruntime_providers_shared.dll
                    onnxruntime_providers_cuda.dll)
                if(EXISTS "${_ort_lib_dir}/${_ort_dll}")
                    add_custom_command(TARGET photara_features POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                            "${_ort_lib_dir}/${_ort_dll}"
                            "$<TARGET_FILE_DIR:photara_features>/${_ort_dll}")
                endif()
            endforeach()
        endif()
    elseif(PHOTARA_ENABLE_ONNX)
        message(WARNING
            "PHOTARA_ENABLE_ONNX=ON but onnxruntime::onnxruntime is unavailable; "
            "LightGlue will not be compiled in")
    endif()
    if(MSVC)
        target_compile_options(photara_features PRIVATE /W4 /permissive- /EHsc)
    endif()

    add_library(photara_sfm
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
        src/sfm/export_colmap.cpp
        src/sfm/export_nerfstudio.cpp
        src/sfm/asfm.cpp
        src/sfm/appearance.cpp
        src/sfm/preview.cpp
        src/sfm/align_live.cpp
        src/project/archive.cpp
        src/project/document.cpp)

    # Derive cache invalidation from the implementation itself. This removes
    # the need to manually bump ad-hoc "vN"/dated ABI strings whenever an SfM,
    # feature, or BA algorithm changes.
    file(GLOB_RECURSE PHOTARA_FRONTEND_CACHE_ID_FILES CONFIGURE_DEPENDS
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
        "${CMAKE_CURRENT_SOURCE_DIR}/src/features/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/features/*.hpp"
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
    file(GLOB_RECURSE PHOTARA_RECONSTRUCTION_CACHE_ID_FILES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/include/ba/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/sfm/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/ba/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/ba/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/sfm/*.cpp")
    list(SORT PHOTARA_FRONTEND_CACHE_ID_FILES)
    list(SORT PHOTARA_RECONSTRUCTION_CACHE_ID_FILES)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        ${PHOTARA_FRONTEND_CACHE_ID_FILES} ${PHOTARA_RECONSTRUCTION_CACHE_ID_FILES})
    set(_photara_cache_build_options
        "cuda=${PHOTARA_ENABLE_CUDA};onnx=${PHOTARA_HAS_ONNXRUNTIME}\n")
    function(photara_source_fingerprint output files_variable)
        set(_material "${_photara_cache_build_options}")
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
    photara_source_fingerprint(
        PHOTARA_FRONTEND_CACHE_BUILD_ID PHOTARA_FRONTEND_CACHE_ID_FILES)
    photara_source_fingerprint(
        PHOTARA_RECONSTRUCTION_CACHE_BUILD_ID
        PHOTARA_RECONSTRUCTION_CACHE_ID_FILES)

    add_library(Photara::SfM ALIAS photara_sfm)
    target_include_directories(photara_sfm PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    target_compile_features(photara_sfm PUBLIC cxx_std_20)
    target_link_libraries(photara_sfm PUBLIC
        Photara::Features
        Photara::BA
        Eigen3::Eigen
        PoseLib::PoseLib
        Ceres::ceres)
    target_compile_definitions(photara_sfm PRIVATE
        PHOTARA_HAS_POSELIB=1
        PHOTARA_FRONTEND_CACHE_BUILD_ID="${PHOTARA_FRONTEND_CACHE_BUILD_ID}"
        PHOTARA_RECONSTRUCTION_CACHE_BUILD_ID="${PHOTARA_RECONSTRUCTION_CACHE_BUILD_ID}")
    if(OpenMP_CXX_FOUND)
        # Link explicitly instead of relying on transitive BA usage
        # requirements: triangulation and mapping own OpenMP regions.
        target_link_libraries(photara_sfm PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(photara_sfm PUBLIC PHOTARA_HAS_OPENMP=1)
    endif()
    if(MSVC)
        target_compile_options(photara_sfm PRIVATE /W4 /permissive- /EHsc)
        if(PHOTARA_ENABLE_NATIVE_ARCH)
            target_compile_options(photara_sfm PRIVATE /arch:AVX2)
        endif()
    else()
        target_compile_options(photara_sfm PRIVATE -Wall -Wextra -Wpedantic)
        if(PHOTARA_ENABLE_NATIVE_ARCH)
            target_compile_options(photara_sfm PRIVATE -march=native)
        endif()
    endif()

    add_library(photara_mvs
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
        src/io/mesh.cpp  # PLY/OBJ stay in mvs/export; GLB interchange lives in io/
        src/mvs/densify.cpp)
    if(PHOTARA_ENABLE_CUDA)
        # Keep CUDA compilation isolated from image-codec targets whose
        # imported MSVC-only options (for example /EHsc) are invalid for nvcc.
        add_library(photara_mvs_cuda STATIC
            src/mvs/patchmatch_cuda.cu)
        target_compile_features(photara_mvs_cuda PUBLIC cxx_std_20)
        target_link_libraries(photara_mvs_cuda PUBLIC CUDA::cudart)
        if(MSVC)
            target_compile_options(photara_mvs_cuda PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4,-EHsc>)
        endif()
        set_target_properties(photara_mvs_cuda PROPERTIES
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            POSITION_INDEPENDENT_CODE ON)
        target_link_libraries(photara_mvs PRIVATE photara_mvs_cuda)
        target_compile_definitions(photara_mvs PUBLIC
            PHOTARA_MVS_HAS_CUDA=1)
    endif()
    add_library(Photara::MVS ALIAS photara_mvs)
    target_include_directories(photara_mvs PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    target_compile_features(photara_mvs PUBLIC cxx_std_20)
    target_link_libraries(photara_mvs PUBLIC
        Photara::SfM
        Eigen3::Eigen
        Boost::headers)
    if(CGAL_FOUND)
        target_link_libraries(photara_mvs PUBLIC CGAL::CGAL)
        target_compile_definitions(photara_mvs PUBLIC PHOTARA_HAS_CGAL=1)
        message(STATUS "Enabling CGAL global Delaunay MVS meshing")
    else()
        message(WARNING
            "CGAL not found: default/high global MVS meshing is unavailable; "
            "splat TSDF meshing remains supported, but PAM and global "
            "Delaunay meshing require a CGAL-enabled build")
    endif()

    if(PHOTARA_ENABLE_SPLAT AND PHOTARA_ENABLE_CUDA)
        include(PhotaraFindDeps)
        photara_find_webpzstd()
        find_package(ZLIB REQUIRED)
        if(NOT TARGET zstd::libzstd AND NOT TARGET zstd::libzstd_shared AND NOT TARGET zstd::libzstd_static)
            message(FATAL_ERROR "zstd target missing after photara_find_webpzstd()")
        endif()
        if(NOT TARGET WebP::webp)
            message(FATAL_ERROR "WebP::webp target missing after photara_find_webpzstd()")
        endif()
        if(PHOTARA_ENABLE_VULKAN_COMPUTE)
            set(TINYTENSOR_ENABLE_VULKAN ON CACHE BOOL
                "Build the TinyTensor Vulkan compute backend" FORCE)
        endif()
        add_subdirectory(
            third_party/tinytensor
            ${CMAKE_CURRENT_BINARY_DIR}/third_party/tinytensor)
        add_subdirectory(
            "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/splat_drender"
            "${CMAKE_CURRENT_BINARY_DIR}/splat_drender")

        add_library(photara_splat_cuda STATIC
            src/splat/rasterizer.cu
            src/splat/pam_cuda.cu
            src/splat/sh_adam_quant.cu
            src/splat/cuda_ops.cu
            src/splat/optimizer.cpp
            src/splat/bilateral_grid.cu
            src/splat/ppisp.cu
            src/splat/cuda_vulkan_preview.cu
            src/splat/fused_ssim.cu
            src/splat/visualize.cu)
        target_include_directories(photara_splat_cuda
            PUBLIC
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:include>
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/src/splat)
        target_compile_features(photara_splat_cuda PUBLIC cxx_std_20)
        target_link_libraries(photara_splat_cuda PUBLIC
            tinytensor::tinytensor
            splat_drender::splat_drender
            CUDA::cudart)
        target_compile_definitions(photara_splat_cuda PUBLIC PHOTARA_HAS_SPLAT=1)
        target_compile_options(photara_splat_cuda PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr -use_fast_math>)
        # Match CPU nearest-neighbor distance ordering and field arithmetic.
        set_source_files_properties(src/splat/pam_cuda.cu PROPERTIES
            COMPILE_OPTIONS "--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false")
        set_target_properties(photara_splat_cuda PROPERTIES
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
            CUDA_SEPARABLE_COMPILATION OFF
            POSITION_INDEPENDENT_CODE ON)

        add_library(photara_splat
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
        add_library(Photara::Splat ALIAS photara_splat)
        target_include_directories(photara_splat
            PUBLIC
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:include>
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/src/splat)
        target_compile_features(photara_splat PUBLIC cxx_std_20)
        target_link_libraries(photara_splat PUBLIC
            Photara::MVS
            photara_splat_cuda
            WebP::webp
            ZLIB::ZLIB
            zstd::libzstd)
        target_compile_definitions(photara_splat PUBLIC PHOTARA_HAS_SPLAT=1)
        if(TARGET splat_drender::vulkan)
            target_sources(photara_splat_cuda PRIVATE
                src/splat/optimizer_vulkan.cpp
                src/splat/colour_correction_vulkan.cpp
                src/splat/rasterizer_vulkan.cpp)
            target_link_libraries(photara_splat_cuda PUBLIC
                splat_drender::vulkan)
            target_compile_definitions(photara_splat_cuda PUBLIC
                TINYTENSOR_HAS_VULKAN=1)
            target_sources(photara_splat PRIVATE src/splat/photometric_loss.cpp)
            target_link_libraries(photara_splat PUBLIC splat_drender::vulkan)
        endif()
        if(OpenMP_CXX_FOUND)
            target_link_libraries(photara_splat PUBLIC OpenMP::OpenMP_CXX)
            target_compile_definitions(photara_splat PUBLIC PHOTARA_HAS_OPENMP=1)
        endif()
    endif()
    if(OpenMP_CXX_FOUND)
        target_link_libraries(photara_mvs PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(photara_mvs PUBLIC PHOTARA_HAS_OPENMP=1)
    endif()
    if(MSVC)
        target_compile_options(photara_mvs PRIVATE /W4 /permissive- /EHsc)
        if(PHOTARA_ENABLE_NATIVE_ARCH)
            target_compile_options(photara_mvs PRIVATE /arch:AVX2)
        endif()
    else()
        target_compile_options(photara_mvs PRIVATE -Wall -Wextra -Wpedantic)
        if(PHOTARA_ENABLE_NATIVE_ARCH)
            target_compile_options(photara_mvs PRIVATE -march=native)
        endif()
    endif()

    if(PHOTARA_HAS_TEXTURE AND TARGET photara::render)
        add_library(photara_texture
            src/texture/bake.cpp
            src/texture/delight.cpp
            src/texture/export.cpp
            src/texture/mask.cpp
            src/texture/mesh_preview.cpp)
        add_library(Photara::Texture ALIAS photara_texture)
        target_include_directories(photara_texture PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>)
        target_compile_features(photara_texture PUBLIC cxx_std_20)
        target_link_libraries(photara_texture PUBLIC
            Photara::MVS
            photara::render)
        target_compile_definitions(photara_texture PUBLIC PHOTARA_HAS_TEXTURE=1)
        if(TARGET photara::mesh)
            target_link_libraries(photara_texture PRIVATE photara::mesh)
            target_compile_definitions(
                photara_texture PRIVATE PHOTARA_HAS_MESH_TOOLS=1)
        endif()
        file(TO_CMAKE_PATH "${PHOTARA_INTRINSIC_MODELS_DIR}"
            PHOTARA_INTRINSIC_MODELS_DIR)
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/include/texture/paths.hpp.in"
            "${CMAKE_CURRENT_BINARY_DIR}/generated/texture/paths.hpp"
            @ONLY)
        target_include_directories(photara_texture PRIVATE
            "${CMAKE_CURRENT_BINARY_DIR}/generated")
        if(PHOTARA_HAS_ONNXRUNTIME AND TARGET onnxruntime::onnxruntime)
            target_link_libraries(photara_texture PRIVATE onnxruntime::onnxruntime)
            target_compile_definitions(photara_texture
                PRIVATE PHOTARA_HAS_ONNXRUNTIME=1)
            message(STATUS
                "Enabling texture baking + C++ ONNX Intrinsic delighter")
        else()
            message(STATUS
                "Enabling texture baking; delighter disabled until "
                "PHOTARA_ENABLE_ONNX=ON (stage_*.onnx + ORT)")
        endif()
        if(MSVC)
            target_compile_options(photara_texture PRIVATE /W4 /permissive- /EHsc)
            if(PHOTARA_ENABLE_NATIVE_ARCH)
                target_compile_options(photara_texture PRIVATE /arch:AVX2)
            endif()
        else()
            target_compile_options(photara_texture PRIVATE -Wall -Wextra -Wpedantic)
            if(PHOTARA_ENABLE_NATIVE_ARCH)
                target_compile_options(photara_texture PRIVATE -march=native)
            endif()
        endif()
        endif()
    endif()

    if(PHOTARA_BUILD_PYTHON)
        if(NOT TARGET photara_splat)
            message(FATAL_ERROR
                "PHOTARA_BUILD_PYTHON requires CUDA splat support")
        endif()
        find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
            RESULT_VARIABLE _photara_nanobind_result
            OUTPUT_VARIABLE nanobind_ROOT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(_photara_nanobind_result EQUAL 0)
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
            photara_python
            NB_STATIC
            src/python/bindings.cpp)
        set_target_properties(photara_python PROPERTIES
            OUTPUT_NAME photara)
        target_link_libraries(photara_python PRIVATE
            Photara::SfM
            Photara::MVS
            Photara::Splat)
        target_compile_features(photara_python PRIVATE cxx_std_20)
        if(MSVC)
            target_compile_options(photara_python PRIVATE
                /W4 /permissive- /EHsc)
        else()
            target_compile_options(photara_python PRIVATE
                -Wall -Wextra -Wpedantic)
        endif()
        if(PHOTARA_BUILD_TESTS)
            add_test(
                NAME photara.python.import
                COMMAND "${Python_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/python_smoke.py"
                    "$<TARGET_FILE_DIR:photara_python>")
        endif()
    endif()

if(TARGET photara_features)
    add_library(photara_cxxopts INTERFACE)
    add_library(cxxopts::cxxopts ALIAS photara_cxxopts)
    target_include_directories(photara_cxxopts INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cxxopts/include)

    add_executable(photara
        src/tools/reconstruct.cpp
        src/tools/sam_model_cache.cpp
        src/tools/sam_masks.cpp)
    target_link_libraries(photara PRIVATE Photara::MVS cxxopts::cxxopts)
    if(PHOTARA_HAS_SAM)
        target_link_libraries(photara PRIVATE photara_sam3)
        target_compile_definitions(photara PRIVATE PHOTARA_HAS_SAM=1)
    endif()
    if(TARGET photara_splat)
        target_link_libraries(photara PRIVATE Photara::Splat)
        target_compile_definitions(photara PRIVATE PHOTARA_HAS_SPLAT=1)
    endif()
    if(TARGET photara_texture)
        target_link_libraries(photara PRIVATE Photara::Texture)
        target_compile_definitions(photara PRIVATE PHOTARA_HAS_TEXTURE=1)
    endif()
    if(TARGET photara::mesh)
        target_link_libraries(photara PRIVATE photara::mesh)
        target_compile_definitions(
            photara PRIVATE PHOTARA_HAS_MESH_TOOLS=1)
    endif()
    if(WIN32)
        add_custom_command(TARGET photara POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:photara>
                $<TARGET_FILE_DIR:photara>
            COMMAND_EXPAND_LISTS)
    endif()

    if(PHOTARA_BUILD_STUDIO)
            find_package(Vulkan 1.2 REQUIRED)
            find_package(glfw3 CONFIG QUIET)
            if(NOT TARGET glfw)
                find_package(PkgConfig REQUIRED)
                pkg_check_modules(PHOTARA_GLFW IMPORTED_TARGET glfw3)
                if(NOT TARGET PkgConfig::PHOTARA_GLFW)
                    message(FATAL_ERROR
                        "glfw3 was not found. Install the glfw3 development "
                        "package (for example libglfw3-dev) or point CMake at "
                        "a glfw3Config.cmake via CMAKE_PREFIX_PATH.")
                endif()
                add_library(glfw ALIAS PkgConfig::PHOTARA_GLFW)
            endif()
            include(FetchContent)
            set(_photara_imgui_local)
            if(glfw3_DIR)
                get_filename_component(
                    _photara_vcpkg_root "${glfw3_DIR}/../../../.." ABSOLUTE)
                # DockSpace / DockBuilder live on the docking branch. Prefer a
                # local vcpkg docking tree when present so configure stays offline.
                file(GLOB _photara_imgui_local
                    "${_photara_vcpkg_root}/buildtrees/imgui/src/*docking*.clean")
            endif()
            if(_photara_imgui_local)
                list(GET _photara_imgui_local 0 _photara_imgui_source)
                FetchContent_Declare(photara_imgui
                    SOURCE_DIR "${_photara_imgui_source}")
            else()
                FetchContent_Declare(
                    photara_imgui
                    GIT_REPOSITORY https://github.com/ocornut/imgui.git
                    GIT_TAG v1.91.9-docking
                    GIT_SHALLOW TRUE)
            endif()
            FetchContent_MakeAvailable(photara_imgui)
            add_library(photara_imgui STATIC
                ${photara_imgui_SOURCE_DIR}/imgui.cpp
                ${photara_imgui_SOURCE_DIR}/imgui_draw.cpp
                ${photara_imgui_SOURCE_DIR}/imgui_tables.cpp
                ${photara_imgui_SOURCE_DIR}/imgui_widgets.cpp
                ${photara_imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
                ${photara_imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp)
            target_include_directories(photara_imgui PUBLIC
                ${photara_imgui_SOURCE_DIR}
                ${photara_imgui_SOURCE_DIR}/backends)
            target_link_libraries(photara_imgui PUBLIC
                Vulkan::Vulkan glfw)
            target_compile_features(photara_imgui PUBLIC cxx_std_20)

            set(PHOTARA_STUDIO_DIR
                "${CMAKE_CURRENT_SOURCE_DIR}/../apps/editor")
            add_subdirectory(
                "${CMAKE_CURRENT_SOURCE_DIR}/../apps/editor/splat_render"
                "${CMAKE_CURRENT_BINARY_DIR}/splat_render")
            add_executable(photara_studio
                ${PHOTARA_STUDIO_DIR}/app_commands.cpp
                ${PHOTARA_STUDIO_DIR}/cache_maintenance.cpp
                ${PHOTARA_STUDIO_DIR}/console_view.cpp
                ${PHOTARA_STUDIO_DIR}/file_dialogs.cpp
                ${PHOTARA_STUDIO_DIR}/i18n.cpp
                ${PHOTARA_STUDIO_DIR}/icons.cpp
                ${PHOTARA_STUDIO_DIR}/image_qa_view.cpp
                ${PHOTARA_STUDIO_DIR}/main.cpp
                ${PHOTARA_STUDIO_DIR}/pipeline.cpp
                ${PHOTARA_STUDIO_DIR}/sam_model.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/tools/sam_model_cache.cpp
                ${PHOTARA_STUDIO_DIR}/sparse_view.cpp
                ${PHOTARA_STUDIO_DIR}/splat_edit.cpp
                ${PHOTARA_STUDIO_DIR}/theme.cpp
                ${PHOTARA_STUDIO_DIR}/ui_chrome.cpp
                ${PHOTARA_STUDIO_DIR}/ui_export.cpp
                ${PHOTARA_STUDIO_DIR}/ui_inspector.cpp
                ${PHOTARA_STUDIO_DIR}/ui_preferences.cpp
                ${PHOTARA_STUDIO_DIR}/ui_scene.cpp
                ${PHOTARA_STUDIO_DIR}/ui_viewport.cpp
                ${PHOTARA_STUDIO_DIR}/viewport_gizmo.cpp
                ${PHOTARA_STUDIO_DIR}/vulkan_backend.cpp)
            if(WIN32)
                get_filename_component(_photara_studio_icon
                    "${CMAKE_CURRENT_SOURCE_DIR}/../apps/icons/icon.ico"
                    ABSOLUTE)
                file(TO_CMAKE_PATH
                    "${_photara_studio_icon}" PHOTARA_STUDIO_ICON)
                configure_file(
                    "${PHOTARA_STUDIO_DIR}/photara_studio.rc.in"
                    "${CMAKE_CURRENT_BINARY_DIR}/photara_studio.rc"
                    @ONLY)
                target_sources(photara_studio PRIVATE
                    "${CMAKE_CURRENT_BINARY_DIR}/photara_studio.rc")
                set_source_files_properties(
                    "${CMAKE_CURRENT_BINARY_DIR}/photara_studio.rc"
                    PROPERTIES OBJECT_DEPENDS "${_photara_studio_icon}")
            endif()
            target_include_directories(photara_studio PRIVATE
                ${PHOTARA_STUDIO_DIR})
            target_link_libraries(photara_studio PRIVATE
                photara_imgui Photara::SfM Photara::Splat
                photara::splat_render photara_vk::photara_vk
                Vulkan::Vulkan glfw)
            if(PHOTARA_HAS_TEXTURE AND TARGET Photara::Texture)
                target_link_libraries(photara_studio PRIVATE Photara::Texture)
                target_compile_definitions(photara_studio PRIVATE
                    PHOTARA_HAS_TEXTURE=1)
            endif()
            if(WIN32)
                # Native folder picker (IFileOpenDialog) and Explorer reveal.
                target_link_libraries(photara_studio PRIVATE ole32 shell32)
            endif()
            target_compile_features(photara_studio PRIVATE cxx_std_20)
            if(MSVC)
                target_compile_options(photara_studio PRIVATE /utf-8)
            endif()
            if(PHOTARA_HAS_SAM)
                target_compile_definitions(photara_studio PRIVATE PHOTARA_HAS_SAM=1)
            endif()
            if(PHOTARA_ENABLE_CUDA)
                target_compile_definitions(photara_studio PRIVATE
                    PHOTARA_STUDIO_BA_CUDA=1)
            endif()
            if(TARGET splat_drender::vulkan)
                target_compile_definitions(photara_studio PRIVATE
                    PHOTARA_STUDIO_VULKAN_TRAINING=1)
            endif()
            if(PHOTARA_HAS_SAM AND GGML_CUDA)
                target_compile_definitions(photara_studio PRIVATE
                    PHOTARA_STUDIO_SAM_CUDA=1)
            endif()
            if(PHOTARA_HAS_SAM AND GGML_VULKAN)
                target_compile_definitions(photara_studio PRIVATE
                    PHOTARA_STUDIO_SAM_VULKAN=1)
            endif()
            target_compile_definitions(photara_studio PRIVATE
                PHOTARA_CLI_FILENAME="$<TARGET_FILE_NAME:photara>")
            add_dependencies(photara_studio photara)
            add_custom_command(TARGET photara_studio POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${CMAKE_CURRENT_SOURCE_DIR}/../apps/icons/icon.png"
                    "$<TARGET_FILE_DIR:photara_studio>/icon.png")
            if(WIN32)
                add_custom_command(TARGET photara_studio POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        $<TARGET_RUNTIME_DLLS:photara_studio>
                        $<TARGET_FILE_DIR:photara_studio>
                    COMMAND_EXPAND_LISTS)
            endif()
    endif()
endif()

if(PHOTARA_BUILD_BENCHMARKS)
    add_executable(photara_ba_benchmark src/tools/ba_benchmark.cpp)
    target_link_libraries(photara_ba_benchmark PRIVATE Photara::BA)
    if(PHOTARA_HAS_VULKAN_BA AND PHOTARA_ENABLE_CUDA)
        add_executable(
            photara_bearing_backend_benchmark
            src/tools/bearing_backend_benchmark.cpp)
        target_link_libraries(
            photara_bearing_backend_benchmark PRIVATE Photara::BA)
    endif()
    if(TARGET photara_features)
        add_executable(photara_features_match src/tools/features_match.cpp)
        target_link_libraries(photara_features_match PRIVATE Photara::Features)
        if(PHOTARA_ENABLE_CUDA)
            add_executable(photara_cuda_sift_benchmark
                src/tools/cuda_sift_benchmark.cpp)
            target_link_libraries(photara_cuda_sift_benchmark PRIVATE
                Photara::Features CUDA::cudart)
            add_executable(photara_cpu_sift_parity tests/cpu_sift_parity.cpp)
            target_link_libraries(photara_cpu_sift_parity PRIVATE Photara::Features)
            target_compile_features(photara_cpu_sift_parity PRIVATE cxx_std_20)
            if(WIN32)
                add_custom_command(TARGET photara_cpu_sift_parity POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        $<TARGET_RUNTIME_DLLS:photara_cpu_sift_parity>
                        $<TARGET_FILE_DIR:photara_cpu_sift_parity>
                    COMMAND_EXPAND_LISTS)
            endif()
        endif()
        if(WIN32)
            add_custom_command(TARGET photara_features_match POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_RUNTIME_DLLS:photara_features_match>
                    $<TARGET_FILE_DIR:photara_features_match>
                COMMAND_EXPAND_LISTS)
        endif()
        add_executable(photara_pair_verify src/tools/pair_verify.cpp)
        add_executable(photara_sfm_realign src/tools/sfm_realign.cpp)
        add_executable(photara_sfm_match_benchmark src/tools/sfm_match_benchmark.cpp)
        add_executable(photara_sfm_positioning_benchmark src/tools/sfm_positioning_benchmark.cpp)
        add_executable(photara_sfm_feature_compare
            src/tools/sfm_feature_compare.cpp)
        target_link_libraries(photara_sfm_feature_compare PRIVATE Photara::SfM)
        if(TARGET photara_splat)
            add_executable(photara_sfm_reference_eval
                src/tools/sfm_reference_eval.cpp)
            target_link_libraries(photara_sfm_reference_eval PRIVATE
                Photara::SfM Photara::Splat)
        endif()
        target_link_libraries(photara_sfm_positioning_benchmark PRIVATE Photara::SfM)
        if(WIN32)
            add_custom_command(TARGET photara_sfm_positioning_benchmark POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_RUNTIME_DLLS:photara_sfm_positioning_benchmark>
                    $<TARGET_FILE_DIR:photara_sfm_positioning_benchmark>
                COMMAND_EXPAND_LISTS)
        endif()
        target_link_libraries(photara_sfm_match_benchmark PRIVATE Photara::SfM)
        if(WIN32)
            add_custom_command(TARGET photara_sfm_match_benchmark POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_RUNTIME_DLLS:photara_sfm_match_benchmark>
                    $<TARGET_FILE_DIR:photara_sfm_match_benchmark>
                COMMAND_EXPAND_LISTS)
        endif()
        target_link_libraries(photara_sfm_realign PRIVATE Photara::SfM)
        if(WIN32)
            add_custom_command(TARGET photara_sfm_realign POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_RUNTIME_DLLS:photara_sfm_realign>
                    $<TARGET_FILE_DIR:photara_sfm_realign>
                COMMAND_EXPAND_LISTS)
        endif()
        target_link_libraries(photara_pair_verify PRIVATE Photara::SfM)
        if(WIN32)
            add_custom_command(TARGET photara_pair_verify POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_RUNTIME_DLLS:photara_pair_verify>
                    $<TARGET_FILE_DIR:photara_pair_verify>
                COMMAND_EXPAND_LISTS)
        endif()
        add_executable(
            photara_triangulation_benchmark
            src/tools/triangulation_benchmark.cpp)
        target_link_libraries(
            photara_triangulation_benchmark PRIVATE Photara::SfM)
        if(WIN32)
            add_custom_command(
                TARGET photara_triangulation_benchmark POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_RUNTIME_DLLS:photara_triangulation_benchmark>
                    $<TARGET_FILE_DIR:photara_triangulation_benchmark>
                COMMAND_EXPAND_LISTS)
        endif()
        if(TARGET photara_splat)
            add_executable(photara_mvs_compare src/tools/mvs_compare.cpp)
            target_link_libraries(photara_mvs_compare PRIVATE
                Photara::Splat Photara::MVS)
        endif()
    endif()
    photara_exclude_from_default(
        photara_ba_benchmark
        photara_bearing_backend_benchmark
        photara_features_match
        photara_cuda_sift_benchmark
        photara_cpu_sift_parity
        photara_pair_verify
        photara_sfm_realign
        photara_sfm_match_benchmark
        photara_sfm_positioning_benchmark
        photara_sfm_feature_compare
        photara_sfm_reference_eval
        photara_triangulation_benchmark
        photara_mvs_compare)
endif()

if(PHOTARA_BUILD_TESTS AND BUILD_TESTING)
    add_custom_target(photara_tests)
    if(PHOTARA_ENABLE_CUDA AND TARGET Ceres::ceres)
        add_executable(photara_bearing_cuda_test tests/bearing_cuda_test.cpp)
        target_link_libraries(photara_bearing_cuda_test PRIVATE Photara::BA Ceres::ceres)
        if(WIN32)
            add_custom_command(TARGET photara_bearing_cuda_test POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_RUNTIME_DLLS:photara_bearing_cuda_test>
                    $<TARGET_FILE_DIR:photara_bearing_cuda_test> COMMAND_EXPAND_LISTS)
        endif()
        add_test(NAME photara.bearing_cuda COMMAND photara_bearing_cuda_test)
    endif()
    if(PHOTARA_HAS_VULKAN_BA AND TARGET Ceres::ceres)
        add_executable(photara_bearing_vulkan_test tests/bearing_vulkan_test.cpp)
        target_link_libraries(photara_bearing_vulkan_test PRIVATE Photara::BA Ceres::ceres)
        if(WIN32)
            add_custom_command(TARGET photara_bearing_vulkan_test POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_RUNTIME_DLLS:photara_bearing_vulkan_test>
                    $<TARGET_FILE_DIR:photara_bearing_vulkan_test> COMMAND_EXPAND_LISTS)
        endif()
        add_test(NAME photara.bearing.vulkan COMMAND photara_bearing_vulkan_test)
    endif()
    add_executable(photara_ba_test tests/ba_optimizer_test.cpp)
    target_link_libraries(photara_ba_test PRIVATE Photara::BA)
    add_test(NAME photara.ba.optimizer COMMAND photara_ba_test)
    add_executable(photara_thread_pool_test tests/thread_pool_test.cpp)
    target_link_libraries(photara_thread_pool_test PRIVATE Photara::BA)
    add_test(NAME photara.parallel.thread_pool COMMAND photara_thread_pool_test)
    add_executable(photara_logging_test tests/logging_test.cpp)
    target_link_libraries(photara_logging_test PRIVATE Photara::BA)
    add_test(NAME photara.core.logging COMMAND photara_logging_test)
    if(TARGET photara_splat)
        add_executable(photara_splat_test tests/splat_test.cpp)
        add_executable(photara_splat_quality_eval src/tools/splat_quality_eval.cpp)
        target_link_libraries(photara_splat_quality_eval PRIVATE Photara::Splat)
        target_link_libraries(photara_splat_test PRIVATE Photara::Splat)
        add_test(NAME photara.splat.rasterizer COMMAND photara_splat_test)
        if(TARGET splat_drender::vulkan)
            # Vulkan backend of splat_drender against the CUDA backend it mirrors.
            add_executable(photara_splat_vulkan_forward_test
                "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/splat_drender/tests/vulkan_forward_compare.cpp")
            target_link_libraries(photara_splat_vulkan_forward_test PRIVATE Photara::Splat splat_drender::vulkan)
            if(WIN32)
                add_custom_command(TARGET photara_splat_vulkan_forward_test POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        $<TARGET_RUNTIME_DLLS:photara_splat_vulkan_forward_test>
                        $<TARGET_FILE_DIR:photara_splat_vulkan_forward_test>
                    COMMAND_EXPAND_LISTS)
            endif()
            add_test(NAME photara.splat.vulkan_forward COMMAND photara_splat_vulkan_forward_test)
            # Manual verification tool (not a ctest entry): renders a trained
            # model through both forward rasterizers and reports the per-channel
            # deltas plus side-by-side images.
            add_executable(photara_splat_vulkan_parity src/tools/splat_vulkan_parity.cpp)
            target_link_libraries(photara_splat_vulkan_parity PRIVATE Photara::Splat splat_drender::vulkan)
            if(WIN32)
                add_custom_command(TARGET photara_splat_vulkan_parity POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        $<TARGET_RUNTIME_DLLS:photara_splat_vulkan_parity>
                        $<TARGET_FILE_DIR:photara_splat_vulkan_parity>
                    COMMAND_EXPAND_LISTS)
            endif()
        endif()
    endif()
    if(TARGET tinytensor AND TINYTENSOR_HAS_VULKAN_BACKEND)
        add_executable(photara_tinytensor_vulkan_test tests/tinytensor_vulkan_test.cpp)
        target_link_libraries(photara_tinytensor_vulkan_test PRIVATE tinytensor::tinytensor)
        target_include_directories(photara_tinytensor_vulkan_test PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tinytensor)
        add_test(NAME photara.tinytensor.vulkan COMMAND photara_tinytensor_vulkan_test)
        # Manual verification tool (not a ctest entry): per-op cost of the
        # tinytensor Vulkan backend.
        add_executable(photara_tinytensor_vulkan_bench src/tools/tinytensor_vulkan_bench.cpp)
        target_link_libraries(photara_tinytensor_vulkan_bench PRIVATE tinytensor::tinytensor)
        target_include_directories(photara_tinytensor_vulkan_bench PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tinytensor)
        # The radix selection against a host sort of the same masked rows: the
        # IGS refinement replaced a sort with it, so this is what pins the two
        # to the same answer.
        add_executable(photara_select_check src/tools/select_check.cpp)
        target_link_libraries(photara_select_check PRIVATE tinytensor::tinytensor)
        target_include_directories(photara_select_check PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tinytensor)
        add_test(NAME photara.tinytensor.select COMMAND photara_select_check)
    endif()
    if(TARGET photara_features)
        add_executable(photara_features_test tests/features_test.cpp)
        target_link_libraries(photara_features_test PRIVATE Photara::Features)
        if(WIN32)
            add_custom_command(TARGET photara_features_test POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_RUNTIME_DLLS:photara_features_test>
                    $<TARGET_FILE_DIR:photara_features_test>
                COMMAND_EXPAND_LISTS)
        endif()
        add_test(NAME photara.features COMMAND photara_features_test)
        add_executable(photara_video_frames_test tests/video_frames_test.cpp)
        target_link_libraries(photara_video_frames_test PRIVATE Photara::Features)
        if(WIN32)
            add_custom_command(TARGET photara_video_frames_test POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_RUNTIME_DLLS:photara_video_frames_test>
                    $<TARGET_FILE_DIR:photara_video_frames_test>
                COMMAND_EXPAND_LISTS)
        endif()
        add_test(NAME photara.io.video_frames COMMAND photara_video_frames_test)
        add_executable(photara_two_view_test tests/two_view_test.cpp)
        target_link_libraries(photara_two_view_test PRIVATE Photara::SfM)
        add_test(NAME photara.sfm.two_view COMMAND photara_two_view_test)
        add_executable(photara_equirect_test tests/equirect_test.cpp)
        target_link_libraries(photara_equirect_test PRIVATE Photara::SfM)
        add_test(NAME photara.sfm.equirect COMMAND photara_equirect_test)
        # Manual verification tool (not a ctest entry): aligns a folder of
        # equirectangular panoramas and exports COLMAP text for external checking.
        add_executable(photara_equirect_dataset_check
            tests/equirect_dataset_check.cpp)
        target_link_libraries(photara_equirect_dataset_check PRIVATE Photara::SfM)
        add_executable(photara_mapping_test tests/mapping_test.cpp)
        add_executable(photara_sfm_refine_probe ../experiments/sfm_refine_probe.cpp)
        target_link_libraries(photara_sfm_refine_probe PRIVATE Photara::SfM)
        target_link_libraries(photara_mapping_test PRIVATE Photara::SfM)
        add_test(NAME photara.sfm.mapping COMMAND photara_mapping_test)
        add_executable(photara_vocabulary_test tests/vocabulary_test.cpp)
        target_link_libraries(photara_vocabulary_test PRIVATE Photara::SfM)
        add_test(NAME photara.sfm.vocabulary COMMAND photara_vocabulary_test)
        add_executable(photara_checkpoint_test tests/checkpoint_test.cpp)
        target_link_libraries(photara_checkpoint_test PRIVATE Photara::SfM)
        add_test(NAME photara.sfm.checkpoint COMMAND photara_checkpoint_test)
        add_executable(photara_hierarchical_test tests/hierarchical_test.cpp)
        add_executable(photara_submap_recovery_test tests/submap_recovery_test.cpp)
        target_link_libraries(photara_submap_recovery_test PRIVATE Photara::SfM)
        add_test(NAME photara.sfm.submap_recovery COMMAND photara_submap_recovery_test)
        target_link_libraries(photara_hierarchical_test PRIVATE Photara::SfM)
        add_test(NAME photara.sfm.hierarchical COMMAND photara_hierarchical_test)
        add_executable(photara_export_mvs_test tests/export_mvs_test.cpp)
        target_link_libraries(photara_export_mvs_test PRIVATE Photara::SfM)
        add_test(NAME photara.sfm.export_mvs COMMAND photara_export_mvs_test)
        add_executable(photara_project_io_test tests/project_io_test.cpp)
        target_link_libraries(photara_project_io_test PRIVATE Photara::SfM)
        add_test(NAME photara.project.io COMMAND photara_project_io_test)
        add_executable(photara_mvs_test tests/mvs_test.cpp)
        target_link_libraries(photara_mvs_test PRIVATE Photara::MVS)
        add_test(NAME photara.mvs.pipeline COMMAND photara_mvs_test)
    endif()
    set(_photara_test_targets
        photara_bearing_cuda_test
        photara_bearing_vulkan_test
        photara_ba_test
        photara_thread_pool_test
        photara_logging_test
        photara_splat_test
        photara_splat_quality_eval
        photara_splat_vulkan_forward_test
        photara_splat_vulkan_parity
        photara_tinytensor_vulkan_test
        photara_features_test
        photara_video_frames_test
        photara_two_view_test
        photara_equirect_test
        photara_equirect_dataset_check
        photara_mapping_test
        photara_sfm_refine_probe
        photara_vocabulary_test
        photara_checkpoint_test
        photara_hierarchical_test
        photara_submap_recovery_test
        photara_export_mvs_test
        photara_project_io_test
        photara_mvs_test)
    photara_exclude_from_default(${_photara_test_targets})
    foreach(_photara_test_target IN LISTS _photara_test_targets)
        if(TARGET "${_photara_test_target}")
            add_dependencies(photara_tests "${_photara_test_target}")
        endif()
    endforeach()
endif()
