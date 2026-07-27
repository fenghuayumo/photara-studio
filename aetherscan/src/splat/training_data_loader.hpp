#pragma once

#include "mvs/types.hpp"
#include "splat/options.hpp"
#include "splat/types.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace aetherscan::splat::training_data {

// Owns the CPU-side packed RGBA8 view cache and performs the narrow
// host-to-device upload needed by an iteration. Keeping this separate from
// Trainer avoids coupling image I/O and prefetch scheduling to optimization.
class TrainingDataLoader {
public:
    TrainingDataLoader(
        const std::vector<mvs::MvsView>& source,
        const TrainingOptions& options, float resolution_scale = 1.F);
    ~TrainingDataLoader();

    TrainingDataLoader(TrainingDataLoader&&) noexcept;
    TrainingDataLoader& operator=(TrainingDataLoader&&) noexcept;

    TrainingDataLoader(const TrainingDataLoader&) = delete;
    TrainingDataLoader& operator=(const TrainingDataLoader&) = delete;

    [[nodiscard]] TrainingView get(std::size_t index);
    [[nodiscard]] bool has_mask(std::size_t index);
    void prefetch(std::size_t index);
    void set_resolution_scale(float scale);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] Camera training_camera(
    const mvs::MvsView& view, const TrainingOptions& options,
    float resolution_scale = 1.F);

[[nodiscard]] float progressive_resolution_scale(
    unsigned iteration, const TrainingOptions& options);

[[nodiscard]] std::vector<std::vector<std::size_t>>
compute_multi_view_neighbours(
    const std::vector<Camera>& cameras,
    const std::vector<std::size_t>& active_indices,
    const TrainingOptions& options);

}  // namespace aetherscan::splat::training_data
