# TinyTensor

A lightweight tensor library extracted from a larger project, adapted for use as a standalone third-party library.

## Overview

TinyTensor provides GPU-accelerated tensor operations with CUDA support, similar in spirit to PyTorch's tensor operations but designed for embedded graphics applications.

## Features

- **Tensor Operations**: Comprehensive tensor manipulation (broadcasting, masking, matrix ops, etc.)
- **CUDA Acceleration**: GPU kernels for performance-critical operations
- **Memory Management**: Built-in allocators (slab, arena, pinned memory)
- **Lazy Execution**: Lazy IR and executor for operation optimization
- **Advanced Operations**: Random ops, neural network ops, fused pointwise ops

## Namespace Migration

This library was originally part of a larger codebase using the `lfs::core` namespace. All code has been migrated to use the `tinytensor` namespace:

- `lfs::core` → `tinytensor`
- `lfs::core::tensor_ops` → `tinytensor::tensor_ops`
- `lfs::core::internal` → `tinytensor::internal`

## Dependencies

- CUDA Runtime API
- Thrust (CUDA library for parallel algorithms)
- C++20 standard library

## Integration

The `core/` directory contains minimal implementations of dependencies that were originally from the parent project:

- `logger.hpp`: Simple logging macros
- `export.hpp`: Export macros for library boundaries
- `tensor_fwd.hpp`: Forward declarations
- `cuda_debug.hpp`: CUDA error checking utilities
- `path_utils.hpp`: Path manipulation utilities
- `tensor_trace.hpp`: Tracing hooks (stub, can be extended)
- `pinned_memory_allocator.hpp`: Pinned memory allocator

These can be replaced with your project's own implementations if desired.

## File Structure

```
tinytensor/
├── core/              # Minimal dependency implementations
├── internal/          # Internal headers and CUDA kernels
├── *.cpp              # CPU implementation files
├── *.cu               # CUDA kernel files
├── CMakeLists.txt     # CMake build configuration
└── README.md          # This file
```

## Usage

```cpp
#include "tinytensor/internal/tensor_impl.hpp"

using namespace tinytensor;

// Create a tensor
Tensor t = Tensor::zeros({3, 4, 5}, DataType::Float32);

// Operations
Tensor result = t.add(1.0f);
Tensor matmul_result = a.matmul(b);
```

## Building

Use CMake only:

```cmake
add_subdirectory(external/tinytensor)
target_link_libraries(your_target PRIVATE tinytensor)
```

## License

This code was extracted from another project. Please refer to the original project's license for usage terms.
