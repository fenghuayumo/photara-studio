#pragma once

#include "features/features.hpp"

#include <cstdint>
#include <memory>
#include <span>

namespace photara::features {

class SiftCpuEngine {
public:
    explicit SiftCpuEngine(SiftOptions options);
    ~SiftCpuEngine();
    SiftCpuEngine(const SiftCpuEngine&) = delete;
    SiftCpuEngine& operator=(const SiftCpuEngine&) = delete;

    [[nodiscard]] FeatureSet extract(
        std::span<const std::uint8_t> pixels,
        std::uint32_t width,
        std::uint32_t height,
        std::size_t row_stride);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace photara::features
