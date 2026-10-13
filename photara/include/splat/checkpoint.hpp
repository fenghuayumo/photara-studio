#pragma once

#include "mvs/types.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <vector>

namespace photara::splat {

namespace detail {
// Serializes native project writes with the trainer; released by the OS on exit.
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
};

// Training state lives inside the native project; no persistent sidecar files.
std::filesystem::path training_checkpoint_path(const std::filesystem::path& project);
// Merge old visible/hidden recovery files into the project before cleaning them.
void migrate_training_checkpoints(const std::filesystem::path& project);
bool training_checkpoint_in_use(const std::filesystem::path& path);
// Tries the previous embedded generation if the current generation is damaged.
std::optional<TrainingCheckpointInfo> inspect_training_checkpoint(
    const std::filesystem::path& path);
std::vector<std::uint8_t> training_checkpoint_bytes(const std::filesystem::path& path);
void restore_training_checkpoint(
    std::span<const std::uint8_t> bytes, const std::filesystem::path& path);
void remove_training_checkpoint(const std::filesystem::path& path);

} // namespace photara::splat
