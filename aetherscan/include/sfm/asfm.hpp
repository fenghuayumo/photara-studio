#pragma once

#include "sfm/scene.hpp"

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace aetherscan::sfm {

inline constexpr std::uint32_t k_asfm_version = 3;
inline constexpr std::uint32_t k_asfm_min_reader = 1;
// Fisheye cameras were introduced in writer v2. Colour is additive in v3, so
// older readers that already understand fisheye can still open new files.
inline constexpr std::uint32_t k_asfm_fisheye_min_reader = 2;
// Equirectangular (360) cameras join fisheye and distortion pinholes in v3.
// Readers older than this must not silently reinterpret a spherical chart as a
// pinhole projection, so the floor matches the writer version for panoramas.
inline constexpr std::uint32_t k_asfm_equirectangular_min_reader = 3;

struct AsfmOptions {
    // If set, image paths are stored relative to this directory when they do
    // not escape it. Load resolves remaining relative paths against this base.
    std::filesystem::path path_base;
};

// Native SfM scene file (.asfm). Product subset: cameras, images, poses,
// keypoints, triangulated tracks. No descriptors, pairs, or resection state.
void save_asfm(
    const Scene& scene,
    const std::filesystem::path& path,
    const AsfmOptions& options = {});
Scene load_asfm(
    const std::filesystem::path& path,
    const AsfmOptions& options = {});

// ASCII XYZRGB PLY of triangulated tracks. Uses stored track colours when
// present; otherwise writes a neutral grey.
void save_sparse_ply(
    const Scene& scene, const std::filesystem::path& path);

std::vector<std::uint8_t> encode_asfm(
    const Scene& scene,
    const AsfmOptions& options = {});
Scene decode_asfm(
    std::span<const std::uint8_t> bytes,
    const AsfmOptions& options = {});

}  // namespace aetherscan::sfm
