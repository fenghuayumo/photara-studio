#pragma once
#if !defined(TINYTENSOR_NO_CUDA)
#include <cuda_runtime.h>
#else
// Compile-time boundary for legacy CUDA branches. No allocation or operation
// may silently succeed when the CUDA backend is absent.
#include <cstddef>
#define __host__
#define __device__
#define __forceinline__ inline

using cudaStream_t = void*;
using cudaEvent_t = void*;
using cudaMemPool_t = void*;
using cudaError_t = int;
inline constexpr int cudaSuccess = 0;
inline constexpr int cudaErrorInvalidDevice = 101;
inline constexpr int cudaErrorInvalidValue = 1;
inline constexpr int cudaErrorNotReady = 600;
inline constexpr unsigned cudaEventDefault = 0, cudaEventDisableTiming = 2, cudaHostAllocDefault = 0;
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice };
enum cudaMemPoolAttr { cudaMemPoolAttrReleaseThreshold, cudaMemPoolAttrReservedMemCurrent, cudaMemPoolAttrUsedMemCurrent, cudaMemPoolAttrReservedMemHigh };
struct cudaDeviceProp { int multiProcessorCount{}, maxThreadsPerMultiProcessor{}, maxThreadsPerBlock{}, major{}, minor{}; };
struct cudaPointerAttributes { int type{}, device{}; void* devicePointer{}; void* hostPointer{}; };
inline const char* cudaGetErrorName(cudaError_t) { return "CUDA backend is disabled"; }
inline const char* cudaGetErrorString(cudaError_t) { return "This build has no CUDA backend; select Vulkan"; }
inline cudaError_t cudaGetDeviceCount(int* count) { *count = 0; return cudaSuccess; }
inline cudaError_t cudaGetLastError() { return cudaErrorInvalidDevice; }
#define TINYTENSOR_DISABLED_CUDA(name) template<class... Args> inline cudaError_t name(Args...) { return cudaErrorInvalidDevice; }
TINYTENSOR_DISABLED_CUDA(cudaDeviceGetDefaultMemPool)
TINYTENSOR_DISABLED_CUDA(cudaDeviceSynchronize)
TINYTENSOR_DISABLED_CUDA(cudaEventCreate)
TINYTENSOR_DISABLED_CUDA(cudaEventCreateWithFlags)
TINYTENSOR_DISABLED_CUDA(cudaEventDestroy)
TINYTENSOR_DISABLED_CUDA(cudaEventElapsedTime)
TINYTENSOR_DISABLED_CUDA(cudaEventQuery)
TINYTENSOR_DISABLED_CUDA(cudaEventRecord)
TINYTENSOR_DISABLED_CUDA(cudaEventSynchronize)
TINYTENSOR_DISABLED_CUDA(cudaFree)
TINYTENSOR_DISABLED_CUDA(cudaFreeAsync)
TINYTENSOR_DISABLED_CUDA(cudaFreeHost)
TINYTENSOR_DISABLED_CUDA(cudaGetDevice)
TINYTENSOR_DISABLED_CUDA(cudaGetDeviceProperties)
TINYTENSOR_DISABLED_CUDA(cudaHostAlloc)
TINYTENSOR_DISABLED_CUDA(cudaMalloc)
TINYTENSOR_DISABLED_CUDA(cudaMallocAsync)
TINYTENSOR_DISABLED_CUDA(cudaMemGetInfo)
TINYTENSOR_DISABLED_CUDA(cudaMemPoolGetAttribute)
TINYTENSOR_DISABLED_CUDA(cudaMemPoolSetAttribute)
TINYTENSOR_DISABLED_CUDA(cudaMemPoolTrimTo)
TINYTENSOR_DISABLED_CUDA(cudaMemcpy)
TINYTENSOR_DISABLED_CUDA(cudaMemcpyAsync)
TINYTENSOR_DISABLED_CUDA(cudaMemset)
TINYTENSOR_DISABLED_CUDA(cudaMemsetAsync)
TINYTENSOR_DISABLED_CUDA(cudaPointerGetAttributes)
TINYTENSOR_DISABLED_CUDA(cudaStreamSynchronize)
TINYTENSOR_DISABLED_CUDA(cudaStreamWaitEvent)
inline constexpr unsigned cudaStreamNonBlocking = 1;
TINYTENSOR_DISABLED_CUDA(cudaStreamCreateWithFlags)
TINYTENSOR_DISABLED_CUDA(cudaStreamDestroy)
#undef TINYTENSOR_DISABLED_CUDA
#endif
