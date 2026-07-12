find_package(OpenMP)

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
    find_package(Ceres CONFIG REQUIRED)
    find_package(hnswlib CONFIG REQUIRED)

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

    set(AETHERSCAN_ONNXRUNTIME_ROOT "" CACHE PATH "Optional ONNX Runtime SDK root")
    find_path(AETHERSCAN_ONNXRUNTIME_INCLUDE onnxruntime_cxx_api.h
        HINTS "${AETHERSCAN_ONNXRUNTIME_ROOT}/include")
    find_library(AETHERSCAN_ONNXRUNTIME_LIBRARY onnxruntime
        HINTS "${AETHERSCAN_ONNXRUNTIME_ROOT}/lib")

    add_library(aetherscan_features
        src/io/image_freeimage.cpp
        src/features/extractor.cpp
        src/features/registry.cpp
        src/features/sift_features.cpp
        src/features/siftgpu_features.cpp
        src/features/superpoint_extractor.cpp
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
    if(AETHERSCAN_ONNXRUNTIME_INCLUDE AND AETHERSCAN_ONNXRUNTIME_LIBRARY)
        target_include_directories(aetherscan_features PRIVATE ${AETHERSCAN_ONNXRUNTIME_INCLUDE})
        target_link_libraries(aetherscan_features PRIVATE ${AETHERSCAN_ONNXRUNTIME_LIBRARY})
        target_compile_definitions(aetherscan_features PRIVATE AETHERSCAN_HAS_ONNXRUNTIME=1)
        if(WIN32 AND EXISTS "${AETHERSCAN_ONNXRUNTIME_ROOT}/lib/onnxruntime.dll")
            add_custom_command(TARGET aetherscan_features POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${AETHERSCAN_ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
                    "$<TARGET_FILE_DIR:aetherscan_features>/onnxruntime.dll")
            if(EXISTS "${AETHERSCAN_ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll")
                add_custom_command(TARGET aetherscan_features POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${AETHERSCAN_ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll"
                        "$<TARGET_FILE_DIR:aetherscan_features>/onnxruntime_providers_shared.dll")
            endif()
        endif()
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
        src/sfm/frontend.cpp
        src/sfm/reconstruct.cpp
        src/sfm/export_mvs.cpp)
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
    target_compile_definitions(aetherscan_sfm PRIVATE AETHERSCAN_HAS_POSELIB=1)
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
        target_link_libraries(aetherscan_pair_verify PRIVATE AetherScan::SfM)
        if(WIN32)
            add_custom_command(TARGET aetherscan_pair_verify POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_RUNTIME_DLLS:aetherscan_pair_verify>
                    $<TARGET_FILE_DIR:aetherscan_pair_verify>
                COMMAND_EXPAND_LISTS)
        endif()
        add_executable(aetherscan src/tools/reconstruct.cpp)
        target_link_libraries(aetherscan PRIVATE AetherScan::SfM)
        if(WIN32)
            add_custom_command(TARGET aetherscan POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_RUNTIME_DLLS:aetherscan>
                    $<TARGET_FILE_DIR:aetherscan>
                COMMAND_EXPAND_LISTS)
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
        target_link_libraries(aetherscan_hierarchical_test PRIVATE AetherScan::SfM)
        add_test(NAME aetherscan.sfm.hierarchical COMMAND aetherscan_hierarchical_test)
        add_executable(aetherscan_export_mvs_test tests/export_mvs_test.cpp)
        target_link_libraries(aetherscan_export_mvs_test PRIVATE AetherScan::SfM)
        add_test(NAME aetherscan.sfm.export_mvs COMMAND aetherscan_export_mvs_test)
    endif()
endif()
