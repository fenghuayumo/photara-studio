#include "splat/cuda_vulkan_preview.hpp"

#include <cuda_runtime.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace photara::splat {
namespace {
void check(const cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(
            std::string("CUDA/Vulkan preview ") + operation + ": " +
            cudaGetErrorString(result));
}

__global__ void planar_rgb_to_surface(
    cudaSurfaceObject_t destination, const float* source,
    std::uint32_t source_width, std::uint32_t source_height,
    std::uint32_t destination_width, std::uint32_t destination_height) {
    const std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= destination_width || y >= destination_height) return;
    if (source_width == 0 || source_height == 0) return;
    // Stretch the CUDA render to fill the shared Vulkan image. The editor
    // then maps that image onto the whole viewport, so letterboxing here
    // would show up as empty bars in the UI.
    const std::uint32_t sx = min(
        static_cast<std::uint32_t>(
            (static_cast<float>(x) + 0.5F) *
            static_cast<float>(source_width) /
            static_cast<float>(destination_width)),
        source_width - 1);
    const std::uint32_t sy = min(
        static_cast<std::uint32_t>(
            (static_cast<float>(y) + 0.5F) *
            static_cast<float>(source_height) /
            static_cast<float>(destination_height)),
        source_height - 1);
    const std::size_t pixels =
        static_cast<std::size_t>(source_width) * source_height;
    const std::size_t index =
        static_cast<std::size_t>(sy) * source_width + sx;
    const auto channel = [&](const std::size_t c) {
        return static_cast<unsigned char>(lrintf(
            fminf(fmaxf(source[c * pixels + index], 0.F), 1.F) * 255.F));
    };
    const uchar4 value = make_uchar4(channel(0), channel(1), channel(2), 255);
    surf2Dwrite(value, destination, x * sizeof(uchar4), y);
}
}  // namespace

struct CudaVulkanPreview::Impl {
    cudaExternalMemory_t memory{};
    cudaExternalSemaphore_t semaphore{};
    cudaMipmappedArray_t mipmapped{};
    cudaArray_t level{};
    cudaSurfaceObject_t surface{};
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint64_t frames{};

    ~Impl() {
        if (surface) cudaDestroySurfaceObject(surface);
        if (mipmapped) cudaFreeMipmappedArray(mipmapped);
        if (semaphore) cudaDestroyExternalSemaphore(semaphore);
        if (memory) cudaDestroyExternalMemory(memory);
    }
};

CudaVulkanPreview::CudaVulkanPreview(
    const CudaVulkanPreviewOptions& options)
    : impl_(std::make_unique<Impl>()) {
#if !defined(_WIN32)
    (void)options;
    throw std::runtime_error(
        "CUDA/Vulkan preview currently requires Win32 external handles");
#else
    if (!options.memory_handle || !options.semaphore_handle ||
        !options.allocation_size || !options.width || !options.height)
        throw std::invalid_argument("Incomplete CUDA/Vulkan preview options");
    impl_->width = options.width;
    impl_->height = options.height;
    if (options.device_luid != 0) {
        int count{};
        check(cudaGetDeviceCount(&count), "enumerate devices");
        int selected = -1;
        for (int index = 0; index < count; ++index) {
            cudaDeviceProp properties{};
            check(cudaGetDeviceProperties(&properties, index),
                  "query device LUID");
            std::uint64_t luid{};
            static_assert(sizeof(luid) == sizeof(properties.luid));
            std::memcpy(&luid, properties.luid, sizeof(luid));
            if (luid == options.device_luid &&
                properties.luidDeviceNodeMask ==
                    options.device_node_mask) {
                selected = index;
                break;
            }
        }
        if (selected < 0)
            throw std::runtime_error(
                "No CUDA device matches the Vulkan device LUID");
        check(cudaSetDevice(selected), "select Vulkan-matched device");
    }

    cudaExternalMemoryHandleDesc memory_desc{};
    memory_desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    memory_desc.handle.win32.handle = reinterpret_cast<void*>(
        static_cast<std::uintptr_t>(options.memory_handle));
    memory_desc.size = options.allocation_size;
    memory_desc.flags = cudaExternalMemoryDedicated;
    check(cudaImportExternalMemory(&impl_->memory, &memory_desc),
          "import memory");

    cudaExternalMemoryMipmappedArrayDesc array_desc{};
    array_desc.offset = 0;
    array_desc.extent = make_cudaExtent(options.width, options.height, 0);
    array_desc.formatDesc = cudaCreateChannelDesc<uchar4>();
    array_desc.numLevels = 1;
    array_desc.flags = cudaArraySurfaceLoadStore;
    check(cudaExternalMemoryGetMappedMipmappedArray(
              &impl_->mipmapped, impl_->memory, &array_desc),
          "map image");
    check(cudaGetMipmappedArrayLevel(
              &impl_->level, impl_->mipmapped, 0),
          "get image level");
    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = impl_->level;
    check(cudaCreateSurfaceObject(&impl_->surface, &resource),
          "create surface");

    cudaExternalSemaphoreHandleDesc semaphore_desc{};
    semaphore_desc.type =
        cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
    semaphore_desc.handle.win32.handle = reinterpret_cast<void*>(
        static_cast<std::uintptr_t>(options.semaphore_handle));
    check(cudaImportExternalSemaphore(
              &impl_->semaphore, &semaphore_desc),
          "import timeline semaphore");

    CloseHandle(reinterpret_cast<HANDLE>(
        static_cast<std::uintptr_t>(options.memory_handle)));
    CloseHandle(reinterpret_cast<HANDLE>(
        static_cast<std::uintptr_t>(options.semaphore_handle)));
#endif
}

CudaVulkanPreview::~CudaVulkanPreview() = default;
CudaVulkanPreview::CudaVulkanPreview(CudaVulkanPreview&&) noexcept = default;
CudaVulkanPreview& CudaVulkanPreview::operator=(
    CudaVulkanPreview&&) noexcept = default;

void CudaVulkanPreview::submit(
    const tinytensor::Tensor& planar_rgb,
    const std::uint32_t source_width,
    const std::uint32_t source_height) {
    if (!impl_ || planar_rgb.device() != tinytensor::Device::CUDA ||
        planar_rgb.dtype() != tinytensor::DataType::Float32 ||
        !planar_rgb.is_contiguous())
        throw std::invalid_argument(
            "CUDA/Vulkan preview requires contiguous CUDA float RGB");
    const std::uint64_t frame = ++impl_->frames;
    if (frame > 1) {
        cudaExternalSemaphoreWaitParams wait{};
        wait.params.fence.value = 2 * (frame - 1);
        check(cudaWaitExternalSemaphoresAsync(
                  &impl_->semaphore, &wait, 1, nullptr),
              "wait for Vulkan");
    }
    const dim3 block(16, 16);
    const dim3 grid(
        (impl_->width + block.x - 1) / block.x,
        (impl_->height + block.y - 1) / block.y);
    planar_rgb_to_surface<<<grid, block>>>(
        impl_->surface, planar_rgb.ptr<float>(), source_width,
        source_height, impl_->width, impl_->height);
    check(cudaGetLastError(), "launch conversion kernel");
    cudaExternalSemaphoreSignalParams signal{};
    signal.params.fence.value = 2 * frame - 1;
    check(cudaSignalExternalSemaphoresAsync(
              &impl_->semaphore, &signal, 1, nullptr),
          "signal Vulkan");
}

std::uint64_t CudaVulkanPreview::frame_count() const noexcept {
    return impl_ ? impl_->frames : 0;
}
}  // namespace photara::splat
