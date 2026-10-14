#pragma once

#include "mvs/types.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <vector>

namespace photara::splat {
struct GaussianModel;

namespace detail {
// Serializes checkpoint writes with the trainer; released by the OS on exit.
class TrainingCheckpointLease {
public:
    explicit TrainingCheckpointLease(const std::filesystem::path& path);
    ~TrainingCheckpointLease();
    TrainingCheckpointLease(const TrainingCheckpointLease&) = delete;
    TrainingCheckpointLease& operator=(const TrainingCheckpointLease&) = delete;
private:
    std::intptr_t handle_{-1};
};
}

struct TrainingCheckpointInfo {
    unsigned iteration{};
    unsigned total_iterations{};
    std::uint64_t gaussian_count{};
    std::uint64_t fingerprint{};
    mvs::OrientedBoundingBox focus_bounds;
    std::filesystem::path model_path;
};

std::vector<std::uint8_t> training_checkpoint_info_bytes(const TrainingCheckpointInfo& info);
// Read the referenced PLY or the model embedded in a saved project.
GaussianModel checkpoint_model(const std::filesystem::path& path);
namespace detail {
// Caller holds the checkpoint lease. Model publication precedes the manifest.
void save_model_checkpoint(const std::filesystem::path& path,
    const TrainingCheckpointInfo& info, const GaussianModel& model);
}

bool training_checkpoint_in_use(const std::filesystem::path& path);
// Accepts only PLY manifests and current embedded model progress.
std::optional<TrainingCheckpointInfo> inspect_training_checkpoint(
    const std::filesystem::path& path);
std::vector<std::uint8_t> training_checkpoint_bytes(const std::filesystem::path& path);
void restore_training_checkpoint(
    std::span<const std::uint8_t> bytes, const std::filesystem::path& path);
void remove_training_checkpoint(const std::filesystem::path& path);

} // namespace photara::splat
