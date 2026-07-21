#pragma once

// TinyTensor Export Macros - Standalone library doesn't need DLL export
// These are kept for compatibility with the original code

#define TINYTENSOR_API
#define TINYTENSOR_EXPORT

// For backward compatibility with original code using LFS_* macros
#define LFS_CORE_API TINYTENSOR_API
#define LFS_CORE_EXPORT TINYTENSOR_EXPORT
