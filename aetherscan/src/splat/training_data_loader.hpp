#pragma once

#include "mvs/types.hpp"
#include "splat/options.hpp"
#include "splat/types.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace aetherscan::splat::training_data {

struct CacheStats {
    std::size_t requests{};
    std::size_t device_hits{};
    std::size_t uploaded_bytes{};
    std::size_t device_resident_bytes{};
    std::size_t device_budget_bytes{};
    std::size_t dataset_packed_bytes{};
    std::size_t host_budget_bytes{};
};

// Owns bounded host and CUDA packed RGBA8 caches. Resident views are expanded
// on CUDA without uploading the same image again. Keeping this separate from
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
    // Image caches are reconstructable; Gaussian/optimizer state is not. Drop
    // packed CUDA views (and trim the CUDA memory pool) when a refinement is
    // about to need more headroom than currently remains.
    void ensure_device_headroom(std::size_t bytes);
    [[nodiscard]] CacheStats stats() const;

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
