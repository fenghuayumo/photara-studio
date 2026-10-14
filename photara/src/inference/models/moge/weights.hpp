#pragma once
#include "inference/models/moge.hpp"
#include "ggml-backend.h"
#include <memory>
#include <string>

namespace photara::inference::detail {
class MogeWeights {
public:
    // The backend must outlive the weights. Limit is a raw tensor-byte bound,
    // not a global GPU budget; 0 disables it. Allocation may require more.
    MogeWeights(const std::filesystem::path& path, ggml_backend_t backend,
        std::size_t maximum_tensor_bytes = 0);
    ~MogeWeights();
    MogeWeights(const MogeWeights&) = delete;
    MogeWeights& operator=(const MogeWeights&) = delete;
    [[nodiscard]] const MogeModelInfo& info() const noexcept;
    [[nodiscard]] ggml_tensor* tensor(const std::string& name) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace photara::inference::detail
