#include "ba/linearizer.hpp"

#include "reprojection_detail.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace aetherscan::ba {
namespace {

void check_cuda(const cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return;
    }
    std::ostringstream message;
    message << operation << " failed: " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    ~DeviceBuffer() { reset(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          size_(std::exchange(other.size_, 0)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    void resize(const std::size_t count) {
        if (count == size_) {
            return;
        }
        reset();
        if (count > 0) {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)), "cudaMalloc");
            size_ = count;
        }
    }

    void upload(const T* source, const std::size_t count) {
        resize(count);
        if (count > 0) {
            check_cuda(
                cudaMemcpy(data_, source, count * sizeof(T), cudaMemcpyHostToDevice),
                "cudaMemcpy host to device");
        }
    }

    void download(T* destination, const std::size_t count) const {
        if (count > size_) {
            throw std::out_of_range("Requested CUDA download exceeds device buffer");
        }
        if (count > 0) {
            check_cuda(
                cudaMemcpy(destination, data_, count * sizeof(T), cudaMemcpyDeviceToHost),
                "cudaMemcpy device to host");
        }
    }

    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }

private:
    void reset() noexcept {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
        data_ = nullptr;
        size_ = 0;
    }

    T* data_{nullptr};
    std::size_t size_{0};
};

__global__ void linearize_kernel(
    const Pose* poses,
    const PinholeIntrinsics* intrinsics,
    const Point3* points,
    const Index* camera_indices,
    const Index* point_indices,
    const double* observed_x,
    const double* observed_y,
    const double* observation_weights,
    const std::size_t observation_count,
    const LinearizerOptions options,
    LinearizedObservation* output) {
    for (std::size_t observation =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         observation < observation_count;
         observation += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const Index camera_index = camera_indices[observation];
        const Index point_index = point_indices[observation];
        detail::linearize_observation(
            poses[camera_index],
            intrinsics[camera_index],
            points[point_index],
            observed_x[observation],
            observed_y[observation],
            observation_weights[observation],
            options,
            output[observation]);
    }
}

}  // namespace

class CudaLinearizer::Impl {
public:
    explicit Impl(const LinearizerOptions options_value)
        : options(options_value) {
        check_cuda(cudaEventCreate(&started), "cudaEventCreate");
        try {
            check_cuda(cudaEventCreate(&stopped), "cudaEventCreate");
        } catch (...) {
            cudaEventDestroy(started);
            throw;
        }
    }

    ~Impl() {
        cudaEventDestroy(stopped);
        cudaEventDestroy(started);
    }

    LinearizerOptions options;
    DeviceBuffer<Pose> poses;
    DeviceBuffer<PinholeIntrinsics> intrinsics;
    DeviceBuffer<Point3> points;
    DeviceBuffer<Index> camera_indices;
    DeviceBuffer<Index> point_indices;
    DeviceBuffer<double> observed_x;
    DeviceBuffer<double> observed_y;
    DeviceBuffer<double> observation_weights;
    DeviceBuffer<LinearizedObservation> output;
    cudaEvent_t started{};
    cudaEvent_t stopped{};
};

CudaLinearizer::CudaLinearizer(const LinearizerOptions options)
    : impl_(std::make_unique<Impl>(options)) {}

CudaLinearizer::~CudaLinearizer() = default;
CudaLinearizer::CudaLinearizer(CudaLinearizer&&) noexcept = default;
CudaLinearizer& CudaLinearizer::operator=(CudaLinearizer&&) noexcept = default;

bool CudaLinearizer::is_available() noexcept {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

std::string CudaLinearizer::device_name() {
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    return properties.name;
}

void CudaLinearizer::upload(const Problem& problem) {
    problem.validate();
    impl_->poses.upload(problem.poses.data(), problem.poses.size());
    impl_->intrinsics.upload(problem.intrinsics.data(), problem.intrinsics.size());
    impl_->points.upload(problem.points.data(), problem.points.size());
    impl_->camera_indices.upload(
        problem.observations.camera.data(), problem.observations.size());
    impl_->point_indices.upload(
        problem.observations.point.data(), problem.observations.size());
    impl_->observed_x.upload(problem.observations.x.data(), problem.observations.size());
    impl_->observed_y.upload(problem.observations.y.data(), problem.observations.size());
    impl_->observation_weights.upload(
        problem.observations.weight.data(), problem.observations.size());
    impl_->output.resize(problem.observations.size());
}

EvaluationStats CudaLinearizer::evaluate() {
    const auto count = impl_->camera_indices.size();
    if (count == 0) {
        throw std::logic_error("Upload a BA problem before CUDA evaluation");
    }

    constexpr unsigned threads_per_block = 256;
    const auto required_blocks =
        (count + threads_per_block - 1) / threads_per_block;
    const auto blocks = static_cast<unsigned>(std::min<std::size_t>(required_blocks, 65535));

    check_cuda(cudaEventRecord(impl_->started), "cudaEventRecord");
    linearize_kernel<<<blocks, threads_per_block>>>(
        impl_->poses.data(),
        impl_->intrinsics.data(),
        impl_->points.data(),
        impl_->camera_indices.data(),
        impl_->point_indices.data(),
        impl_->observed_x.data(),
        impl_->observed_y.data(),
        impl_->observation_weights.data(),
        count,
        impl_->options,
        impl_->output.data());
    check_cuda(cudaGetLastError(), "linearize_kernel launch");
    check_cuda(cudaEventRecord(impl_->stopped), "cudaEventRecord");
    check_cuda(cudaEventSynchronize(impl_->stopped), "cudaEventSynchronize");

    float milliseconds = 0.0F;
    check_cuda(
        cudaEventElapsedTime(&milliseconds, impl_->started, impl_->stopped),
        "cudaEventElapsedTime");
    return EvaluationStats{static_cast<double>(milliseconds), count};
}

void CudaLinearizer::download(LinearizationOutput& output) const {
    output.resize(impl_->output.size());
    impl_->output.download(output.observations.data(), output.observations.size());
}

std::size_t CudaLinearizer::observation_count() const noexcept {
    return impl_->output.size();
}

}  // namespace aetherscan::ba
