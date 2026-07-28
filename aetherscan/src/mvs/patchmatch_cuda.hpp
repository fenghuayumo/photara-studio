#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace aetherscan::mvs::cuda_patchmatch {

inline constexpr unsigned k_max_sources = 16;

struct HostImage {
    std::uint32_t width{};
    std::uint32_t height{};
    float fx{};
    float fy{};
    float cx{};
    float cy{};
    const float* gray{};
    const std::uint8_t* mask{};
    const float* depth{};
};

struct HostSource {
    HostImage image;
    // Reference-camera to source-camera transform:
    // X_source = rotation * X_reference + translation.
    float rotation[9]{};
    float translation[3]{};
};

struct Request {
    int device{-1};
    HostImage reference;
    float* depth{};
    // Packed xyzxyz... camera-space unit normals.
    float* normal_xyz{};
    float* confidence{};
    const HostSource* sources{};
    unsigned source_count{};
    float depth_min{};
    float depth_max{};
    unsigned estimation_iters{};
    unsigned random_iters{};
    unsigned min_patch_views{};
    float geometric_weight{};
    unsigned random_seed{};
    bool use_geometric{};
    bool initialize_invalid{};

    bool use_roi{};
    // Reference-camera to world rotation (row-major) and camera center.
    float reference_to_world[9]{};
    float reference_center[3]{};
    // ROI axes are column vectors in world space (row-major matrix storage).
    float roi_center[3]{};
    float roi_axes[9]{};
    float roi_half_extent[3]{};
};

// Returns false only when CUDA/device support is unavailable. Runtime failures
// in an explicitly selected CUDA path are reported through `error`.
bool available(int device, std::string& device_name, std::string& error);
bool run(const Request& request, std::string& error);

}  // namespace aetherscan::mvs::cuda_patchmatch
