#pragma once

// TinyTensor Export Macros - Standalone library doesn't need DLL export
// These are kept for compatibility with the original code

#define TINYTENSOR_API
#define TINYTENSOR_EXPORT

// Export qualifiers used throughout the TinyTensor headers (YISON_ prefix).
#define YISON_CORE_API TINYTENSOR_API
#define YISON_CORE_EXPORT TINYTENSOR_EXPORT
