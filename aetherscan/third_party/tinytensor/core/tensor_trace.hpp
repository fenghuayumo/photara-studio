#pragma once

// TinyTensor Trace - Minimal tracing support for standalone library
// Original tracing functionality is removed as it depends on external systems

namespace tinytensor {

// Stub tracing - can be implemented later if needed
inline void trace_tensor_op(const char* op) {
    // No-op for standalone library
    // Could be extended with actual tracing
}

} // namespace tinytensor
