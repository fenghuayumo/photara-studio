#pragma once

#include <cstdint>

namespace tinytensor {

// ============================================================================
// Data Types - Similar to PyTorch/NumPy dtypes
// ============================================================================

enum class DataType : uint8_t {
    Bool = 0,
    UInt8 = 1,
    Int32 = 2,
    Int64 = 3,
    Float16 = 4,
    Float32 = 5,
    // Can be extended as needed
};

enum class Device : uint8_t {
    CPU = 0,
    CUDA = 1,
};

// ============================================================================
// Data Type Utilities
// ============================================================================

inline constexpr size_t dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::Bool:   return 1;
        case DataType::UInt8:  return 1;
        case DataType::Int32:  return 4;
        case DataType::Int64:  return 8;
        case DataType::Float16: return 2;
        case DataType::Float32: return 4;
        default: return 4;
    }
}

inline constexpr const char* dtype_name(DataType dtype) {
    switch (dtype) {
        case DataType::Bool:   return "bool";
        case DataType::UInt8:  return "uint8";
        case DataType::Int32:  return "int32";
        case DataType::Int64:  return "int64";
        case DataType::Float16: return "float16";
        case DataType::Float32: return "float32";
        default: return "unknown";
    }
}

inline constexpr bool is_bool_like(DataType dtype) {
    return dtype == DataType::Bool || dtype == DataType::UInt8;
}

inline constexpr const char* device_name(Device device) {
    switch (device) {
        case Device::CPU:  return "cpu";
        case Device::CUDA: return "cuda";
        default: return "unknown";
    }
}

} // namespace tinytensor
