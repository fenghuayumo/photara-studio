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
    double get_wall_ms{};
    // Requests served without a host decode, background host loads issued, and
    // the measured host-load / iteration cadence that drives the adaptive
    // prefetch lookahead.
    std::size_t host_hits{};
    std::size_t prefetch_issued{};
    double host_load_mean_ms{};
    double step_mean_ms{};
    std::size_t prefetch_depth{};
};

// Owns bounded host and CUDA packed RGBA8 caches. Resident views are expanded
// on CUDA without uploading the same image again. Host decoding may run on
// background threads (see TrainingOptions::training_prefetch_views); every CUDA
// call stays on the calling thread.
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
    // Current prefetch lookahead in views. Callers offer at least this many
    // upcoming views; the value grows while host loads take longer than one
    // iteration (see TrainingOptions::training_prefetch_adaptive).
    [[nodiscard]] std::size_t prefetch_depth() const;
    void set_resolution_scale(float scale);
    // Publish the trainer's current epoch order and cursor. One shuffled epoch
    // visits every training view exactly once, which makes least-recently-used
    // eviction degenerate (it evicts the view that is needed soonest). With the
    // order known, the image caches evict the entry whose next use is farthest
    // away instead. Passing nullptr restores plain LRU.
    void set_epoch_plan(const std::vector<std::size_t>* order, std::size_t cursor);
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
