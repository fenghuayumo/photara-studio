#pragma once

// Internal model-adapter API. Public headers never expose ggml handles.
#include "inference/runtime.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"

namespace photara::inference::detail {
class GgmlRuntime {
public:
    explicit GgmlRuntime(const RuntimeOptions& options = {});
    ~GgmlRuntime();
    GgmlRuntime(const GgmlRuntime&) = delete;
    GgmlRuntime& operator=(const GgmlRuntime&) = delete;
    [[nodiscard]] const DeviceInfo& device() const noexcept { return device_; }
    [[nodiscard]] ggml_backend_t backend() const noexcept { return backend_; }
    // Allocate before uploading graph inputs. Throws for unsupported operators;
    // no hidden per-operator CPU fallback or scheduler is used.
    void allocate(ggml_cgraph* graph);
    void compute(ggml_cgraph* graph);
    [[nodiscard]] std::size_t compute_buffer_bytes() const;
private:
    DeviceInfo device_;
    ggml_backend_t backend_ = nullptr;
    ggml_gallocr_t allocator_ = nullptr;
};
} // namespace photara::inference::detail
