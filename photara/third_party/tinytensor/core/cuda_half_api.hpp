#pragma once
#if !defined(TINYTENSOR_NO_CUDA)
#include <cuda_fp16.h>
#else
using __half = _Float16;
inline __half __float2half(float value) {
    return static_cast<__half>(value);
}
inline float __half2float(__half value) {
    return static_cast<float>(value);
}
#endif
