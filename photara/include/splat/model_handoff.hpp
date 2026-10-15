#pragma once

#include "splat/types.hpp"

#include <chrono>
#include <filesystem>
#include <memory>

namespace photara::splat {

// Immutable final-model snapshot shared with the editor. Only the small ready
// manifest lives on disk; Gaussian arrays use pagefile/POSIX shared memory.
// Keep this owner alive until the reader has opened its own mapping.
class ModelHandoff {
public:
    ModelHandoff(const GaussianModel& model, const std::filesystem::path& ready);
    ~ModelHandoff();
    ModelHandoff(const ModelHandoff&) = delete;
    ModelHandoff& operator=(const ModelHandoff&) = delete;

    // The same CPU snapshot can be persisted without another GPU download.
    [[nodiscard]] const GaussianModel& model() const;
    bool wait_for_reader(std::chrono::milliseconds timeout) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Opens the mapping before acknowledging the publisher, validates the payload,
// and returns an owned CPU model. Safe after the publisher exits once opened.
GaussianModel receive_model_handoff(const std::filesystem::path& ready);

}  // namespace photara::splat
