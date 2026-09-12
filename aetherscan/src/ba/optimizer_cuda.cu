#include "ba/optimizer.hpp"

#include "ba/linearizer.hpp"
#include "core/logging.hpp"
#include "reprojection_detail.cuh"

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <cstdlib>
#include <vector>

namespace aetherscan::ba {
namespace {

constexpr unsigned threads = 256;
constexpr unsigned warp_size = 32;
constexpr unsigned warps_per_block = threads / warp_size;
constexpr std::size_t cross_n = 18;

void check(const cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    std::ostringstream message;
    message << operation << " failed: " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
}

unsigned blocks(const std::size_t count) {
    return static_cast<unsigned>(std::min<std::size_t>((count + threads - 1) / threads, 65535));
}

unsigned warp_blocks(const std::size_t count) {
    constexpr std::size_t max_grid_x = 2147483647;
    return static_cast<unsigned>(std::min<std::size_t>(
        (count + warps_per_block - 1) / warps_per_block, max_grid_x));
}

template <class T>
class Buffer {
public:
    Buffer() = default;
    ~Buffer() { if (data_) cudaFree(data_); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void resize(const std::size_t size) {
        if (size == size_) return;
        if (data_) check(cudaFree(data_), "cudaFree");
        data_ = nullptr; size_ = 0;
        if (size) {
            check(cudaMalloc(reinterpret_cast<void**>(&data_), size * sizeof(T)), "cudaMalloc");
            size_ = size;
        }
    }
    void upload(const T* source, const std::size_t size) {
        resize(size);
        if (size) check(cudaMemcpy(data_, source, size * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }
    void download(T* destination, const std::size_t size) const {
        if (size > size_) throw std::out_of_range("CUDA buffer download exceeds allocation");
        if (size) check(cudaMemcpy(destination, data_, size * sizeof(T), cudaMemcpyDeviceToHost), "download");
    }
    void zero() { if (size_) check(cudaMemset(data_, 0, size_ * sizeof(T)), "cudaMemset"); }
    void swap(Buffer& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
    }
    T* data() { return data_; }
    const T* data() const { return data_; }
    std::size_t size() const { return size_; }
private:
    T* data_{};
    std::size_t size_{};
};

__device__ void mul3(const double* matrix, const double* vector, double* output) {
#pragma unroll
    for (int row = 0; row < 3; ++row)
        output[row] = matrix[row * 3] * vector[0] + matrix[row * 3 + 1] * vector[1] +
                      matrix[row * 3 + 2] * vector[2];
}

__global__ void linearize_kernel(
    const Pose* poses, const PinholeIntrinsics* intrinsics, const Index* pose_group,
    const Point3* points,
    const Index* cameras, const Index* point_ids, const double* x, const double* y,
    const double* weights, const std::size_t count, const LinearizerOptions options,
    LinearizedObservation* output) {
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        detail::linearize_observation(poses[cameras[i]], intrinsics[pose_group[cameras[i]]],
                                      points[point_ids[i]],
                                      x[i], y[i], weights[i], options, output[i]);
}

__device__ double warp_sum(double value) {
#pragma unroll
    for (unsigned offset = warp_size / 2; offset > 0; offset /= 2)
        value += __shfl_down_sync(0xFFFFFFFFu, value, offset);
    return value;
}

__global__ void assemble_points_and_cross_kernel(
    const LinearizedObservation* values,
    const std::size_t* point_offsets,
    const std::size_t* point_observations,
    const std::size_t point_count,
    const std::size_t intrinsic_dof,
    double* point_h,
    double* point_b,
    double* cross,
    double* point_intr) {
    const unsigned lane = threadIdx.x % warp_size;
    const unsigned warp = threadIdx.x / warp_size;
    const std::size_t point =
        static_cast<std::size_t>(blockIdx.x) * warps_per_block + warp;
    if (point >= point_count) return;

    double local_h[9]{};
    double local_b[3]{};
    for (std::size_t cursor = point_offsets[point] + lane;
         cursor < point_offsets[point + 1]; cursor += warp_size) {
        const std::size_t observation = point_observations[cursor];
        const auto& value = values[observation];
        double* w = cross + observation * cross_n;
        if (!value.valid) {
#pragma unroll
            for (int i = 0; i < static_cast<int>(cross_n); ++i) w[i] = 0.0;
            if (intrinsic_dof > 0) {
                double* destination = point_intr + observation * intrinsic_dof * 3;
                for (std::size_t i = 0; i < intrinsic_dof * 3; ++i)
                    destination[i] = 0.0;
            }
            continue;
        }
#pragma unroll
        for (int row = 0; row < 6; ++row) {
            const double jc0 = value.pose_jacobian[row];
            const double jc1 = value.pose_jacobian[6 + row];
#pragma unroll
            for (int column = 0; column < 3; ++column)
                w[row * 3 + column] = jc0 * value.point_jacobian[column] +
                                      jc1 * value.point_jacobian[3 + column];
        }
#pragma unroll
            for (int row = 0; row < 3; ++row) {
                const double jp0 = value.point_jacobian[row];
                const double jp1 = value.point_jacobian[3 + row];
                if (intrinsic_dof > 0) {
                    double* destination = point_intr + observation * intrinsic_dof * 3;
#pragma unroll 1
                    for (std::size_t param = 0; param < intrinsic_dof; ++param)
                        destination[row * intrinsic_dof + param] =
                            jp0 * value.intrinsic_jacobian[param] +
                            jp1 * value.intrinsic_jacobian[
                                k_max_intrinsic_params + param];
                }
                local_b[row] -=
                    jp0 * value.residual[0] + jp1 * value.residual[1];
#pragma unroll
            for (int column = 0; column < 3; ++column)
                local_h[row * 3 + column] +=
                    jp0 * value.point_jacobian[column] +
                    jp1 * value.point_jacobian[3 + column];
        }
    }

#pragma unroll
    for (int i = 0; i < 9; ++i) {
        const double total = warp_sum(local_h[i]);
        if (lane == 0) point_h[point * 9 + i] = total;
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const double total = warp_sum(local_b[i]);
        if (lane == 0) point_b[point * 3 + i] = total;
    }
}

__global__ void assemble_cameras_and_intrinsics_kernel(
    const LinearizedObservation* values,
    const std::size_t* camera_offsets,
    const std::size_t* camera_observations,
    const std::size_t camera_count,
    const Index* pose_group,
    const std::size_t intrinsic_dof,
    double* camera_h,
    double* camera_b,
    double* pose_intr,
    double* intrinsic_h,
    double* intrinsic_b) {
    const unsigned lane = threadIdx.x % warp_size;
    const unsigned warp = threadIdx.x / warp_size;
    const std::size_t camera =
        static_cast<std::size_t>(blockIdx.x) * warps_per_block + warp;
    if (camera >= camera_count) return;

    double local_h[36]{};
    double local_b[6]{};
    // Per-camera H_ci (6 x k), and per-camera contribution to the group's
    // H_ii (k x k) and b_i (k). Reduced across the warp like the pose block.
    double local_ci[6 * k_max_intrinsic_params]{};
    double local_ii[k_max_intrinsic_params * k_max_intrinsic_params]{};
    double local_bi[k_max_intrinsic_params]{};
    for (std::size_t cursor = camera_offsets[camera] + lane;
         cursor < camera_offsets[camera + 1]; cursor += warp_size) {
        const std::size_t observation = camera_observations[cursor];
        const auto& value = values[observation];
        if (!value.valid) continue;
#pragma unroll
        for (int row = 0; row < 6; ++row) {
            const double jc0 = value.pose_jacobian[row];
            const double jc1 = value.pose_jacobian[6 + row];
            local_b[row] -=
                jc0 * value.residual[0] + jc1 * value.residual[1];
#pragma unroll
            for (int column = 0; column < 6; ++column)
                local_h[row * 6 + column] +=
                    jc0 * value.pose_jacobian[column] +
                    jc1 * value.pose_jacobian[6 + column];
        }
        if (intrinsic_dof > 0) {
            for (std::size_t param = 0; param < intrinsic_dof; ++param) {
                const double ji0 = value.intrinsic_jacobian[param];
                const double ji1 =
                    value.intrinsic_jacobian[k_max_intrinsic_params + param];
                local_bi[param] -=
                    ji0 * value.residual[0] + ji1 * value.residual[1];
#pragma unroll 1
                for (int row = 0; row < 6; ++row)
                    local_ci[row * k_max_intrinsic_params + param] +=
                        value.pose_jacobian[row] * ji0 +
                        value.pose_jacobian[6 + row] * ji1;
#pragma unroll 1
                for (std::size_t other = 0; other < intrinsic_dof; ++other)
                    local_ii[param * k_max_intrinsic_params + other] +=
                        ji0 * value.intrinsic_jacobian[other] +
                        ji1 * value.intrinsic_jacobian[
                            k_max_intrinsic_params + other];
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 36; ++i) {
        const double total = warp_sum(local_h[i]);
        if (lane == 0) camera_h[camera * 36 + i] = total;
    }
#pragma unroll
    for (int i = 0; i < 6; ++i) {
        const double total = warp_sum(local_b[i]);
        if (lane == 0) camera_b[camera * 6 + i] = total;
    }
    if (intrinsic_dof == 0) return;
    for (int row = 0; row < 6; ++row) {
        for (std::size_t param = 0; param < intrinsic_dof; ++param) {
            const double total = warp_sum(
                local_ci[row * k_max_intrinsic_params + param]);
            if (lane == 0)
                pose_intr[camera * 6 * intrinsic_dof + row * intrinsic_dof + param] =
                    total;
        }
    }
    const std::size_t group = pose_group[camera];
    for (std::size_t param = 0; param < intrinsic_dof; ++param) {
        const double total = warp_sum(local_bi[param]);
        if (lane == 0)
            atomicAdd(intrinsic_b + group * intrinsic_dof + param, total);
        for (std::size_t other = 0; other < intrinsic_dof; ++other) {
            const double h_total = warp_sum(
                local_ii[param * k_max_intrinsic_params + other]);
            if (lane == 0)
                atomicAdd(intrinsic_h + group * intrinsic_dof * intrinsic_dof +
                          param * intrinsic_dof + other, h_total);
        }
    }
}

__global__ void damp_cameras(double* hessian, const std::size_t count, const double damping) {
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::size_t>(blockDim.x) * gridDim.x)
#pragma unroll
        for (int d = 0; d < 6; ++d) {
            double& diagonal = hessian[i * 36 + d * 6 + d];
            diagonal += damping * (diagonal + 1.0);
        }
}

struct IntrinsicSettings {
    double focal_prior_weight;
    double min_focal_ratio;
    double max_focal_ratio;
    bool optimize_focal;
    bool optimize_aspect_ratio;
    bool optimize_principal_point;
    bool optimize_distortion;
};

// Applies LM damping to H_ii, then the soft focal prior, then overrides
// constant groups with identity rows (CPU assemble_system + apply_focal_prior
// order, so the reduced intrinsic system stays semantically identical).
__global__ void finalize_intrinsics_kernel(
    double* intrinsic_h, double* intrinsic_b,
    const PinholeIntrinsics* intrinsics,
    const PinholeIntrinsics* initial,
    const unsigned char* constant,
    const std::size_t group_count, const std::size_t intrinsic_dof,
    const double damping, const double observation_count,
    const IntrinsicSettings settings) {
    for (std::size_t group = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         group < group_count; group += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        double* hessian = intrinsic_h + group * intrinsic_dof * intrinsic_dof;
        double* rhs = intrinsic_b + group * intrinsic_dof;
        for (std::size_t diagonal = 0; diagonal < intrinsic_dof; ++diagonal)
            hessian[diagonal * intrinsic_dof + diagonal] +=
                damping * (hessian[diagonal * intrinsic_dof + diagonal] + 1.0);
        if (constant[group] != 0) {
            for (std::size_t row = 0; row < intrinsic_dof * intrinsic_dof; ++row)
                hessian[row] = 0.0;
            for (std::size_t diagonal = 0; diagonal < intrinsic_dof; ++diagonal)
                hessian[diagonal * intrinsic_dof + diagonal] = 1.0;
            for (std::size_t param = 0; param < intrinsic_dof; ++param)
                rhs[param] = 0.0;
            continue;
        }
        if (!settings.optimize_focal ||
            settings.focal_prior_weight <= 0.0)
            continue;
        const PinholeIntrinsics& current = intrinsics[group];
        if (settings.optimize_aspect_ratio) {
            const double fx0 = fmax(initial[group].fx, 1.0);
            const double fy0 = fmax(initial[group].fy, 1.0);
            const double weight_fx =
                0.5 * settings.focal_prior_weight * observation_count / (fx0 * fx0);
            const double weight_fy =
                0.5 * settings.focal_prior_weight * observation_count / (fy0 * fy0);
            hessian[0] += weight_fx;
            hessian[intrinsic_dof + 1] += weight_fy;
            rhs[0] -= weight_fx * (current.fx - fx0);
            rhs[1] -= weight_fy * (current.fy - fy0);
        } else {
            const double f0 = fmax(
                0.5 * (initial[group].fx + initial[group].fy), 1.0);
            const double f = 0.5 * (current.fx + current.fy);
            const double weight =
                settings.focal_prior_weight * observation_count / (f0 * f0);
            hessian[0] += weight;
            rhs[0] -= weight * (f - f0);
        }
    }
}

__global__ void invert_points(
    double* hessian, double* rhs, const std::size_t count,
    const double damping, const bool fix_first, const bool optimize_points) {
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        double* m = hessian + i * 9;
        if (!optimize_points || (fix_first && i == 0)) {
            for (int k = 0; k < 9; ++k) m[k] = 0.0;
            for (int k = 0; k < 3; ++k) rhs[i * 3 + k] = 0.0;
            continue;
        }
#pragma unroll
        for (int d = 0; d < 3; ++d) m[d * 3 + d] += damping * (m[d * 3 + d] + 1.0);
        const double a=m[0], b=m[1], c=m[2], d=m[4], e=m[5], f=m[8];
        const double c00=d*f-e*e, c01=c*e-b*f, c02=b*e-c*d;
        const double c11=a*f-c*c, c12=b*c-a*e, c22=a*d-b*b;
        const double determinant=a*c00+b*c01+c*c02;
        if (!(determinant > 1e-30) || !isfinite(determinant)) {
            for (int k = 0; k < 9; ++k) m[k] = 0.0;
            continue;
        }
        const double s=1.0/determinant;
        m[0]=c00*s; m[1]=c01*s; m[2]=c02*s; m[3]=m[1]; m[4]=c11*s;
        m[5]=c12*s; m[6]=m[2]; m[7]=m[5]; m[8]=c22*s;
    }
}

__global__ void reduced_point_rhs_kernel(
    const double* inverse, const double* rhs, const std::size_t count, double* reduced) {
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        mul3(inverse + i * 9, rhs + i * 3, reduced + i * 3);
}

__global__ void schur_rhs_kernel(
    const std::size_t* offsets, const std::size_t* observations, const Index* points,
    const double* camera_rhs, const double* reduced_points, const double* cross,
    const std::size_t camera_count, const bool fix_first,
    const bool optimize_rotations, const bool optimize_translations,
    double* output) {
    for (std::size_t camera = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         camera < camera_count; camera += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        double result[6];
        for (int row=0; row<6; ++row) result[row]=camera_rhs[camera*6+row];
        if (fix_first && camera == 0) for (double& value : result) value=0.0;
        else for (std::size_t cursor=offsets[camera]; cursor<offsets[camera+1]; ++cursor) {
            const std::size_t obs=observations[cursor];
            const double* w=cross+obs*18;
            const double* p=reduced_points+static_cast<std::size_t>(points[obs])*3;
            for (int row=0; row<6; ++row)
                result[row]-=w[row*3]*p[0]+w[row*3+1]*p[1]+w[row*3+2]*p[2];
        }
        for (int row=0; row<6; ++row) {
            const bool active = row < 3 ? optimize_rotations : optimize_translations;
            output[camera*6+row]=active ? result[row] : 0.0;
        }
        }
}

__global__ void copy_intrinsic_rhs_kernel(
    const double* intrinsic_rhs, const std::size_t intrinsic_values,
    double* output) {
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < intrinsic_values; index += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        output[index] = intrinsic_rhs[index];
}

// rhs_i -= sum_obs W_ip H_pp^-1 b_p, mirroring build_joint_rhs().
__global__ void intrinsic_schur_rhs_kernel(
    const std::size_t* point_offsets, const std::size_t* point_observations,
    const Index* cameras, const Index* pose_group,
    const double* reduced_points, const double* point_intr,
    const std::size_t point_count, const std::size_t camera_values,
    const std::size_t intrinsic_dof, double* output) {
    for (std::size_t point = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         point < point_count; point += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const double* reduced = reduced_points + point * 3;
        for (std::size_t cursor = point_offsets[point];
             cursor < point_offsets[point + 1]; ++cursor) {
            const std::size_t observation = point_observations[cursor];
            const std::size_t group = pose_group[cameras[observation]];
            const double* w = point_intr + observation * intrinsic_dof * 3;
            for (std::size_t param = 0; param < intrinsic_dof; ++param)
                atomicAdd(
                    output + camera_values + group * intrinsic_dof + param,
                    -(w[param] * reduced[0] +
                      w[intrinsic_dof + param] * reduced[1] +
                      w[2 * intrinsic_dof + param] * reduced[2]));
        }
    }
}

// The Schur product is memory-bound. Traversing CSR blocks with one thread
// per point or camera serializes scattered 144-byte cross blocks and reaches
// ~1% of device bandwidth; gathering per observation keeps reads unit-stride
// and reduces into small per-block scratch buffers with hardware atomics.
__global__ void point_gather_kernel(
    const Index* cameras, const Index* point_ids, const Index* pose_group,
    const double* cross, const double* point_intr, const double* input,
    const std::size_t intrinsic_dof, const std::size_t camera_values,
    const std::size_t observation_count, const bool fix_first_camera,
    double* point_scratch) {
    for (std::size_t obs=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         obs<observation_count; obs+=static_cast<std::size_t>(blockDim.x)*gridDim.x) {
        const std::size_t camera=cameras[obs];
        double sum[3]{};
        if (!(fix_first_camera && camera==0)) {
            const double* w=cross+obs*18; const double* x=input+camera*6;
            for (int axis=0; axis<3; ++axis)
                for (int row=0; row<6; ++row)
                    sum[axis]+=w[row*3+axis]*x[row];
        }
        if (intrinsic_dof>0) {
            const std::size_t group=pose_group[camera];
            const double* xi=input+camera_values+group*intrinsic_dof;
            const double* pi=point_intr+obs*intrinsic_dof*3;
            for (int axis=0; axis<3; ++axis)
                for (std::size_t param=0; param<intrinsic_dof; ++param)
                    sum[axis]+=pi[axis*intrinsic_dof+param]*xi[param];
        }
        double* destination=point_scratch+
            static_cast<std::size_t>(point_ids[obs])*3;
        for (int axis=0; axis<3; ++axis) atomicAdd(destination+axis,sum[axis]);
    }
}

__global__ void point_reduce_kernel(
    const double* inverse, const double* point_scratch,
    const std::size_t point_count, double* temporary) {
    for (std::size_t point=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         point<point_count; point+=static_cast<std::size_t>(blockDim.x)*gridDim.x)
        mul3(inverse+point*9,point_scratch+point*3,temporary+point*3);
}

struct CameraChunk {
    std::size_t begin, end;
    Index camera;
};

// Reduce a camera's CSR chunk before touching global memory. Pose and
// intrinsic products share the same landmark load, including shared groups.
__global__ void camera_chunk_gather_kernel(
    const CameraChunk* chunks, const std::size_t chunk_count,
    const std::size_t* camera_observations, const Index* point_ids,
    const Index* pose_group, const double* cross, const double* point_intr,
    const double* point_temporary, const std::size_t intrinsic_dof,
    double* camera_scratch, double* intrinsic_scratch) {
    __shared__ double partial[warps_per_block];
    for (std::size_t chunk_id=blockIdx.x; chunk_id<chunk_count; chunk_id+=gridDim.x) {
        const CameraChunk chunk=chunks[chunk_id];
        const std::size_t index=chunk.begin+threadIdx.x;
        double sums[6+k_max_intrinsic_params]{};
        if (index<chunk.end) {
            const std::size_t obs=camera_observations[index];
            const double* p=point_temporary+static_cast<std::size_t>(point_ids[obs])*3;
            const double* w=cross+obs*18;
            for (int row=0;row<6;++row)
                sums[row]=w[row*3]*p[0]+w[row*3+1]*p[1]+w[row*3+2]*p[2];
            const double* pi=point_intr+obs*intrinsic_dof*3;
            for (std::size_t row=0;row<intrinsic_dof;++row)
                sums[6+row]=pi[row]*p[0]+pi[intrinsic_dof+row]*p[1]+
                    pi[2*intrinsic_dof+row]*p[2];
        }
        const unsigned lane=threadIdx.x%warp_size;
        const unsigned warp=threadIdx.x/warp_size;
        for (std::size_t row=0;row<6+intrinsic_dof;++row) {
            double value=sums[row];
            for (unsigned offset=warp_size/2;offset;offset/=2)
                value+=__shfl_down_sync(0xffffffff,value,offset);
            if (lane==0) partial[warp]=value;
            __syncthreads();
            if (warp==0) {
                value=lane<warps_per_block ? partial[lane] : 0.0;
                for (unsigned offset=warp_size/2;offset;offset/=2)
                    value+=__shfl_down_sync(0xffffffff,value,offset);
                if (lane==0) {
                    double* destination=row<6
                        ? camera_scratch+static_cast<std::size_t>(chunk.camera)*6+row
                        : intrinsic_scratch+static_cast<std::size_t>(pose_group[chunk.camera])*intrinsic_dof+row-6;
                    atomicAdd(destination,value);
                }
            }
            __syncthreads();
        }
    }
}

__global__ void camera_gather_kernel(
    const Index* cameras, const Index* point_ids, const double* cross,
    const double* point_temporary, const std::size_t observation_count,
    double* camera_scratch) {
    for (std::size_t obs=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         obs<observation_count; obs+=static_cast<std::size_t>(blockDim.x)*gridDim.x) {
        const double* w=cross+obs*18;
        const double* p=point_temporary+static_cast<std::size_t>(point_ids[obs])*3;
        double* destination=camera_scratch+static_cast<std::size_t>(cameras[obs])*6;
        for (int row=0; row<6; ++row)
            atomicAdd(destination+row,
                w[row*3]*p[0]+w[row*3+1]*p[1]+w[row*3+2]*p[2]);
    }
}

__global__ void camera_reduce_kernel(
    const Index* pose_group, const double* hessian, const double* pose_intr,
    const double* input, const double* camera_scratch,
    const std::size_t camera_count, const std::size_t intrinsic_dof,
    const std::size_t camera_values, const bool fix_first,
    const bool optimize_rotations, const bool optimize_translations,
    double* output) {
    for (std::size_t camera=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         camera<camera_count; camera+=static_cast<std::size_t>(blockDim.x)*gridDim.x) {
        double* out=output+camera*6;
        const double* x=input+camera*6;
        if (fix_first && camera==0) {
            for (int row=0; row<6; ++row) out[row]=x[row];
            continue;
        }
        double result[6]{}; const double* h=hessian+camera*36;
        const double* scratch=camera_scratch+camera*6;
        for (int row=0; row<6; ++row) {
            for (int column=0; column<6; ++column)
                result[row]+=h[row*6+column]*x[column];
            result[row]-=scratch[row];
        }
        if (intrinsic_dof>0) {
            const std::size_t group=pose_group[camera];
            const double* xi=input+camera_values+group*intrinsic_dof;
            const double* ci=pose_intr+camera*6*intrinsic_dof;
            for (int row=0; row<6; ++row)
                for (std::size_t param=0; param<intrinsic_dof; ++param)
                    result[row]+=ci[row*intrinsic_dof+param]*xi[param];
        }
        for (int row=0; row<6; ++row) {
            const bool active = row < 3 ? optimize_rotations : optimize_translations;
            out[row]=active ? result[row] : x[row];
        }
    }
}

// The damped block is unchanged throughout PCG; factor it once per LM step.
__global__ void factor_preconditioner_kernel(
    const double* hessian, const std::size_t count, const std::size_t dim,
    const std::size_t stride, double* factors) {
    for (std::size_t block=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         block<count;block+=static_cast<std::size_t>(blockDim.x)*gridDim.x) {
        const double* matrix=hessian+block*dim*dim;
        double lower[k_max_intrinsic_params*k_max_intrinsic_params]{};
        bool valid=true;
        for (std::size_t row=0;row<dim && valid;++row)
            for (std::size_t column=0;column<=row;++column) {
                double value=matrix[row*dim+column];
                for (std::size_t k=0;k<column;++k)
                    value-=lower[row*stride+k]*lower[column*stride+k];
                if (row==column) {
                    if (!(value>1e-24)||!isfinite(value)) { valid=false;break; }
                    lower[row*stride+column]=sqrt(value);
                } else lower[row*stride+column]=value/lower[column*stride+column];
            }
        if (!valid) lower[0]=0.0;
        for (std::size_t i=0;i<stride*stride;++i)
            factors[block*stride*stride+i]=lower[i];
    }
}

__global__ void precondition_kernel(
    const double* hessian, const double* factors, const double* residual, const std::size_t camera_count,
    const bool fix_first, const bool optimize_rotations,
    const bool optimize_translations, double* output) {
    for (std::size_t camera=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         camera<camera_count; camera+=static_cast<std::size_t>(blockDim.x)*gridDim.x) {
        double* solution=output+camera*6;
        if (fix_first && camera==0) { for (int i=0;i<6;++i) solution[i]=0.0; continue; }
        const double* matrix=hessian+camera*36;
        const double* lower=factors+camera*36;
        const bool valid=lower[0]>0.0;
        if (!valid) {
            for (int i=0;i<6;++i)
                solution[i]=residual[camera*6+i]/fmax(matrix[i*6+i],1e-12);
        } else {
            double temporary[6]{};
            for (int row=0;row<6;++row) { double value=residual[camera*6+row];
                for (int k=0;k<row;++k) value-=lower[row*6+k]*temporary[k];
                temporary[row]=value/lower[row*6+row]; }
            for (int row=5;row>=0;--row) { double value=temporary[row];
                for (int k=row+1;k<6;++k) value-=lower[k*6+row]*solution[k];
                solution[row]=value/lower[row*6+row]; }
        }
        for (int row=0;row<6;++row) {
            const bool active = row < 3 ? optimize_rotations : optimize_translations;
            if (!active) solution[row]=0.0;
        }
    }
}

__global__ void dot_kernel(const double* a, const double* b, const std::size_t count, double* result) {
    __shared__ double shared[threads];
    double sum=0.0;
    for (std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         i<count; i+=static_cast<std::size_t>(blockDim.x)*gridDim.x) sum+=a[i]*b[i];
    shared[threadIdx.x]=sum; __syncthreads();
    for (unsigned stride=blockDim.x/2; stride; stride/=2) {
        if (threadIdx.x<stride) shared[threadIdx.x]+=shared[threadIdx.x+stride];
        __syncthreads();
    }
    if (threadIdx.x==0) atomicAdd(result, shared[0]);
}

// y_i += H_ii x_i (small dense blocks; one block per thread).
__global__ void intrinsic_block_product_kernel(
    const double* intrinsic_h, const double* input,
    const std::size_t group_count, const std::size_t intrinsic_dof,
    const std::size_t camera_values, double* output) {
    for (std::size_t group=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         group<group_count; group+=static_cast<std::size_t>(blockDim.x)*gridDim.x) {
        const double* h=intrinsic_h+group*intrinsic_dof*intrinsic_dof;
        const double* x=input+camera_values+group*intrinsic_dof;
        double* y=output+camera_values+group*intrinsic_dof;
        for (std::size_t row=0; row<intrinsic_dof; ++row) {
            double value=0.0;
            for (std::size_t column=0; column<intrinsic_dof; ++column)
                value+=h[row*intrinsic_dof+column]*x[column];
            y[row]+=value;
        }
    }
}

// y_i = H_ii x_i + sum_cameras H_ic x_c - scratch (the accumulated Schur term).
__global__ void intrinsic_reduce_kernel(
    const Index* pose_group, const double* intrinsic_hessian, const double* pose_intr,
    const double* input, const double* intrinsic_scratch,
    const std::size_t camera_count, const std::size_t group_count,
    const std::size_t intrinsic_dof, const std::size_t camera_values,
    double* output) {
    for (std::size_t camera=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         camera<camera_count; camera+=static_cast<std::size_t>(blockDim.x)*gridDim.x) {
        const std::size_t group=pose_group[camera];
        double* destination=output+camera_values+group*intrinsic_dof;
        const double* x=input+camera*6;
        const double* ci=pose_intr+camera*6*intrinsic_dof;
        for (std::size_t param=0; param<intrinsic_dof; ++param) {
            double value=0.0;
            for (int row=0; row<6; ++row)
                value+=ci[row*intrinsic_dof+param]*x[row];
            atomicAdd(destination+param,value);
        }
    }
    for (std::size_t group=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         group<group_count; group+=static_cast<std::size_t>(blockDim.x)*gridDim.x) {
        const double* h=intrinsic_hessian+group*intrinsic_dof*intrinsic_dof;
        const double* x=input+camera_values+group*intrinsic_dof;
        const double* scratch=intrinsic_scratch+group*intrinsic_dof;
        double* y=output+camera_values+group*intrinsic_dof;
        for (std::size_t row=0; row<intrinsic_dof; ++row) {
            double value=0.0;
            for (std::size_t column=0; column<intrinsic_dof; ++column)
                value+=h[row*intrinsic_dof+column]*x[column];
            atomicAdd(y+row,value-scratch[row]);
        }
    }
}

__global__ void zero_values_kernel(double* values, const std::size_t count) {
    for (std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         i<count; i+=static_cast<std::size_t>(blockDim.x)*gridDim.x)
        values[i]=0.0;
}

// Block-Jacobi solve for the tiny per-group intrinsic blocks.
__global__ void precondition_intrinsics_kernel(
    const double* intrinsic_h, const double* factors, const double* residual, const std::size_t group_count,
    const std::size_t intrinsic_dof, const std::size_t camera_values,
    double* output) {
    for (std::size_t group=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         group<group_count; group+=static_cast<std::size_t>(blockDim.x)*gridDim.x) {
        const double* matrix=intrinsic_h+group*intrinsic_dof*intrinsic_dof;
        const double* rhs=residual+camera_values+group*intrinsic_dof;
        double* solution=output+camera_values+group*intrinsic_dof;
        const double* lower=factors+group*k_max_intrinsic_params*k_max_intrinsic_params;
        const bool valid=lower[0]>0.0;
        if (!valid) {
            for (std::size_t i=0; i<intrinsic_dof; ++i)
                solution[i]=rhs[i]/fmax(matrix[i*intrinsic_dof+i],1e-12);
        } else {
            double temporary[k_max_intrinsic_params]{};
            for (std::size_t row=0; row<intrinsic_dof; ++row) {
                double value=rhs[row];
                for (std::size_t k=0; k<row; ++k)
                    value-=lower[row*k_max_intrinsic_params+k]*temporary[k];
                temporary[row]=value/lower[row*k_max_intrinsic_params+row];
            }
            for (std::size_t row=intrinsic_dof; row-- > 0;) {
                double value=temporary[row];
                for (std::size_t k=row+1; k<intrinsic_dof; ++k)
                    value-=lower[k*k_max_intrinsic_params+row]*solution[k];
                solution[row]=value/lower[row*k_max_intrinsic_params+row];
            }
        }
    }
}

__global__ void update_pcg_kernel(
    double* solution, double* residual, const double* direction, const double* product,
    const std::size_t count, const double* alpha) {
    for (std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         i<count; i+=static_cast<std::size_t>(blockDim.x)*gridDim.x) {
        solution[i]+=*alpha*direction[i]; residual[i]-=*alpha*product[i];
    }
}

__global__ void direction_kernel(
    double* direction, const double* z, const std::size_t count, const double* beta) {
    for (std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         i<count; i+=static_cast<std::size_t>(blockDim.x)*gridDim.x) direction[i]=z[i]+*beta*direction[i];
}

__global__ void alpha_kernel(double* values) {
    if (threadIdx.x==0 && blockIdx.x==0)
        values[4]=(values[1]>1e-30 && isfinite(values[1])) ? values[0]/values[1] : 0.0;
}

__global__ void beta_kernel(double* values) {
    if (threadIdx.x==0 && blockIdx.x==0) {
        values[4]=(fabs(values[0])>1e-30 && isfinite(values[3])) ? values[3]/values[0] : 0.0;
        values[0]=values[3];
    }
}

__global__ void recover_points_kernel(
    const std::size_t* offsets, const std::size_t* observations, const Index* cameras,
    const Index* pose_group, const double* cross, const double* point_intr,
    const double* inverse, const double* rhs, const double* camera_step,
    const std::size_t intrinsic_dof, const std::size_t camera_values,
    const std::size_t point_count, double* point_step) {
    for (std::size_t point=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         point<point_count; point+=static_cast<std::size_t>(blockDim.x)*gridDim.x) {
        double value[3]={rhs[point*3],rhs[point*3+1],rhs[point*3+2]};
        for (std::size_t cursor=offsets[point]; cursor<offsets[point+1]; ++cursor) {
            const std::size_t obs=observations[cursor], camera=cameras[obs];
            const double* w=cross+obs*18; const double* step=camera_step+camera*6;
            for (int column=0; column<3; ++column)
                for (int row=0; row<6; ++row) value[column]-=w[row*3+column]*step[row];
            if (intrinsic_dof>0) {
                const std::size_t group=pose_group[camera];
                const double* si=camera_step+camera_values+group*intrinsic_dof;
                const double* pi=point_intr+obs*intrinsic_dof*3;
                for (int axis=0; axis<3; ++axis)
                    for (std::size_t param=0; param<intrinsic_dof; ++param)
                        value[axis]-=pi[axis*intrinsic_dof+param]*si[param];
            }
        }
        mul3(inverse+point*9, value, point_step+point*3);
    }
}

__global__ void update_poses_kernel(
    Pose* poses, const double* step, const std::size_t count,
    const bool fix_first, const bool optimize_rotations,
    const bool optimize_translations) {
    for (std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         i<count; i+=static_cast<std::size_t>(blockDim.x)*gridDim.x) {
        if (fix_first && i==0) continue;
        Pose& p=poses[i]; const double* s=step+i*6;
        if (optimize_rotations) {
            const double angle=sqrt(s[0]*s[0]+s[1]*s[1]+s[2]*s[2]);
            double dw=1.0, dx=0.5*s[0], dy=0.5*s[1], dz=0.5*s[2];
            if (angle>1e-12) { dw=cos(0.5*angle); const double k=sin(0.5*angle)/angle;
                dx=k*s[0]; dy=k*s[1]; dz=k*s[2]; }
            const double qw=dw*p.qw-dx*p.qx-dy*p.qy-dz*p.qz;
            const double qx=dw*p.qx+dx*p.qw+dy*p.qz-dz*p.qy;
            const double qy=dw*p.qy-dx*p.qz+dy*p.qw+dz*p.qx;
            const double qz=dw*p.qz+dx*p.qy-dy*p.qx+dz*p.qw;
            const double inv=rsqrt(qw*qw+qx*qx+qy*qy+qz*qz);
            p.qw=qw*inv; p.qx=qx*inv; p.qy=qy*inv; p.qz=qz*inv;
        }
        if (optimize_translations) {
            p.cx+=s[3]; p.cy+=s[4]; p.cz+=s[5];
        }
    }
}

__global__ void update_points_kernel(Point3* points, const double* step, const std::size_t count) {
    for (std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         i<count; i+=static_cast<std::size_t>(blockDim.x)*gridDim.x) {
        points[i].x+=step[i*3]; points[i].y+=step[i*3+1]; points[i].z+=step[i*3+2];
    }
}

// Constant intrinsic groups contribute observations but are not estimated:
// their cross blocks are zeroed so the joint solve keeps x_i identically zero
// (mirrors the CPU zeroing of pose/point intrinsic cross blocks).
__global__ void mask_constant_intrinsics_kernel(
    const Index* cameras, const Index* pose_group, const unsigned char* constant,
    const std::size_t camera_count, const std::size_t observation_count,
    const std::size_t intrinsic_dof, double* pose_intr, double* point_intr) {
    for (std::size_t camera=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         camera<camera_count; camera+=static_cast<std::size_t>(blockDim.x)*gridDim.x) {
        if (constant[pose_group[camera]]==0) continue;
        double* destination=pose_intr+camera*6*intrinsic_dof;
        for (std::size_t i=0; i<6*intrinsic_dof; ++i) destination[i]=0.0;
    }
    for (std::size_t obs=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         obs<observation_count; obs+=static_cast<std::size_t>(blockDim.x)*gridDim.x) {
        if (constant[pose_group[cameras[obs]]]==0) continue;
        double* destination=point_intr+obs*intrinsic_dof*3;
        for (std::size_t i=0; i<intrinsic_dof*3; ++i) destination[i]=0.0;
    }
}

__global__ void update_intrinsics_kernel(
    PinholeIntrinsics* intrinsics, const PinholeIntrinsics* initial,
    const unsigned char* constant, const double* step,
    const std::size_t group_count, const std::size_t intrinsic_dof,
    const std::size_t camera_values, const IntrinsicSettings settings) {
    for (std::size_t group=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         group<group_count; group+=static_cast<std::size_t>(blockDim.x)*gridDim.x) {
        if (constant[group]!=0) continue;
        PinholeIntrinsics& value=intrinsics[group];
        const double* s=step+camera_values+group*intrinsic_dof;
        std::size_t index=0;
        if (settings.optimize_focal) {
            value.fx+=s[index++];
            if (settings.optimize_aspect_ratio) value.fy+=s[index++];
            else value.fy+=s[index-1];
            double minimum_fx=1.0, maximum_fx=1e308, minimum_fy=1.0, maximum_fy=1e308;
            const double fx0=fmax(initial[group].fx,1.0);
            const double fy0=fmax(initial[group].fy,1.0);
            minimum_fx=fmax(1.0,fx0*settings.min_focal_ratio);
            maximum_fx=fx0*settings.max_focal_ratio;
            minimum_fy=fmax(1.0,fy0*settings.min_focal_ratio);
            maximum_fy=fy0*settings.max_focal_ratio;
            value.fx=fmin(fmax(value.fx,minimum_fx),maximum_fx);
            value.fy=fmin(fmax(value.fy,minimum_fy),maximum_fy);
        }
        if (settings.optimize_principal_point) {
            value.cx+=s[index++]; value.cy+=s[index++];
        }
        if (settings.optimize_distortion) {
            value.k1+=s[index++]; value.k2+=s[index++];
            value.p1+=s[index++]; value.p2+=s[index++];
        }
    }
}

__global__ void prior_cost_kernel(
    const PinholeIntrinsics* intrinsics, const PinholeIntrinsics* initial,
    const unsigned char* constant, const std::size_t group_count,
    const double prior_weight, const double observation_count,
    const bool optimize_focal, const bool optimize_aspect_ratio,
    double* total) {
    if (!optimize_focal || prior_weight<=0.0) return;
    double local=0.0;
    for (std::size_t group=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         group<group_count; group+=static_cast<std::size_t>(blockDim.x)*gridDim.x) {
        if (constant[group]!=0) continue;
        if (optimize_aspect_ratio) {
            const double fx0=fmax(initial[group].fx,1.0);
            const double fy0=fmax(initial[group].fy,1.0);
            const double rx=(intrinsics[group].fx-fx0)/fx0;
            const double ry=(intrinsics[group].fy-fy0)/fy0;
            local+=0.25*prior_weight*observation_count*(rx*rx+ry*ry);
        } else {
            const double f0=fmax(0.5*(initial[group].fx+initial[group].fy),1.0);
            const double f=0.5*(intrinsics[group].fx+intrinsics[group].fy);
            const double r=(f-f0)/f0;
            local+=0.5*prior_weight*observation_count*r*r;
        }
    }
    atomicAdd(total,local);
}

__global__ void linearized_cost_kernel(
    const LinearizedObservation* values,
    const double* weights,
    const std::size_t count,
    const double huber,
    double* total) {
    double local=0.0;
    for (std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         i<count; i+=static_cast<std::size_t>(blockDim.x)*gridDim.x) {
        if (weights[i] == 0.0) continue;
        const LinearizedObservation value = values[i];
        if (!value.valid || !(value.robust_weight > 0.0)) {
            local += weights[i] * 1e12;
            continue;
        }
        const double weighted_squared =
            value.residual[0] * value.residual[0] +
            value.residual[1] * value.residual[1];
        const double norm = sqrt(weighted_squared / value.robust_weight);
        if (!isfinite(norm)) {
            local += weights[i] * 1e12;
            continue;
        }
        local += weights[i] *
                 ((huber > 0 && norm > huber)
                      ? huber * (norm - 0.5 * huber)
                      : 0.5 * norm * norm);
    }
    atomicAdd(total,local);
}

}  // namespace

class CudaOptimizer::Impl {
public:
    explicit Impl(OptimizerOptions value) : options(value) {}
    OptimizerOptions options;
    std::size_t camera_count{}, point_count{}, observation_count{};
    std::size_t group_count{}, intrinsic_dof{};
    std::size_t camera_values{}, intrinsic_values{}, total_values{};
    Buffer<Pose> poses, pose_backup;
    Buffer<PinholeIntrinsics> intrinsics, intrinsic_backup, initial_intrinsics;
    Buffer<unsigned char> intrinsic_constant;
    Buffer<Point3> points, point_backup;
    Buffer<Index> cameras, point_ids;
    Buffer<Index> pose_group;
    Buffer<double> ox, oy, weights;
    Buffer<std::size_t> point_offsets, point_observations, camera_offsets, camera_observations;
    Buffer<CameraChunk> camera_chunks;
    Buffer<LinearizedObservation> linearized, candidate_linearized;
    Buffer<double> camera_h, camera_b, point_inverse, point_b, cross, reduced_point;
    Buffer<double> camera_factor, intrinsic_factor;
    Buffer<double> intrinsic_h, intrinsic_b, pose_intr, point_intr;
    Buffer<double> point_scratch, camera_scratch, intrinsic_scratch;
    Buffer<double> rhs, solution, residual, z, direction, product, point_temporary, point_step;
    Buffer<double> scalar, pcg_scalars;
    cudaGraph_t pcg_graph{nullptr};
    cudaGraphExec_t pcg_graph_exec{nullptr};
    std::size_t graph_iterations{};

    ~Impl() {
        if (pcg_graph_exec) cudaGraphExecDestroy(pcg_graph_exec);
        if (pcg_graph) cudaGraphDestroy(pcg_graph);
    }

    [[nodiscard]] IntrinsicSettings intrinsic_settings() const {
        return IntrinsicSettings{options.focal_prior_weight, options.min_focal_ratio,
            options.max_focal_ratio, options.optimize_focal,
            options.optimize_aspect_ratio, options.optimize_principal_point,
            options.optimize_distortion};
    }
    [[nodiscard]] LinearizerOptions linearizer_settings() const {
        return LinearizerOptions{options.huber_delta, options.minimum_depth,
            options.optimize_focal, options.optimize_aspect_ratio,
            options.optimize_principal_point, options.optimize_distortion};
    }

    double dot(const Buffer<double>& a, const Buffer<double>& b, const std::size_t count) {
        scalar.zero(); dot_kernel<<<blocks(count),threads>>>(a.data(),b.data(),count,scalar.data());
        check(cudaGetLastError(),"dot_kernel"); double result{}; scalar.download(&result,1); return result;
    }
    void dot_to(const Buffer<double>& a, const Buffer<double>& b, const std::size_t count, const std::size_t index) {
        check(cudaMemset(pcg_scalars.data()+index,0,sizeof(double)),"PCG scalar reset");
        dot_kernel<<<blocks(count),threads>>>(a.data(),b.data(),count,pcg_scalars.data()+index);
        check(cudaGetLastError(),"PCG dot_kernel");
    }
    void dot_to(const Buffer<double>& a, const Buffer<double>& b, const std::size_t count,
                const std::size_t index, cudaStream_t stream) {
        check(cudaMemsetAsync(pcg_scalars.data()+index,0,sizeof(double),stream),"PCG scalar reset");
        dot_kernel<<<blocks(count),threads,0,stream>>>(a.data(),b.data(),count,pcg_scalars.data()+index);
        check(cudaGetLastError(),"PCG dot_kernel capture");
    }
    double cost(const Buffer<LinearizedObservation>& values) {
        scalar.zero();
        linearized_cost_kernel<<<blocks(observation_count),threads>>>(
            values.data(),weights.data(),observation_count,
            options.huber_delta,scalar.data());
        check(cudaGetLastError(),"linearized_cost_kernel");
        double result{};
        if (intrinsic_dof>0) {
            prior_cost_kernel<<<1,1>>>(intrinsics.data(),initial_intrinsics.data(),
                intrinsic_constant.data(),group_count,options.focal_prior_weight,
                static_cast<double>(std::max<std::size_t>(observation_count,1)),
                options.optimize_focal,options.optimize_aspect_ratio,scalar.data());
            check(cudaGetLastError(),"prior_cost_kernel");
        }
        scalar.download(&result,1);
        return result;
    }
    void multiply() {
        check(cudaMemset(point_scratch.data(),0,point_count*3*sizeof(double)),
            "point scratch reset");
        check(cudaMemset(camera_scratch.data(),0,camera_count*6*sizeof(double)),
            "camera scratch reset");
        point_gather_kernel<<<blocks(observation_count),threads>>>(
            cameras.data(),point_ids.data(),pose_group.data(),cross.data(),
            point_intr.data(),direction.data(),intrinsic_dof,camera_values,
            observation_count,options.fix_first_pose,point_scratch.data());
        point_reduce_kernel<<<blocks(point_count),threads>>>(
            point_inverse.data(),point_scratch.data(),point_count,
            point_temporary.data());
        if (intrinsic_dof>0) {
            check(cudaMemset(intrinsic_scratch.data(),0,
                intrinsic_values*sizeof(double)),"intrinsic scratch reset");
            check(cudaMemset(product.data()+camera_values,0,
                intrinsic_values*sizeof(double)),"intrinsic product reset");
        }
        if (intrinsic_dof>0)
        camera_chunk_gather_kernel<<<static_cast<unsigned>(std::min<std::size_t>(camera_chunks.size(),65535)),threads>>>(
            camera_chunks.data(),camera_chunks.size(),camera_observations.data(),
            point_ids.data(),pose_group.data(),cross.data(),point_intr.data(),
            point_temporary.data(),intrinsic_dof,camera_scratch.data(),intrinsic_scratch.data());
        else
        camera_gather_kernel<<<blocks(observation_count),threads>>>(
            cameras.data(),point_ids.data(),cross.data(),point_temporary.data(),
            observation_count,camera_scratch.data());
        camera_reduce_kernel<<<blocks(camera_count),threads>>>(
            pose_group.data(),camera_h.data(),pose_intr.data(),direction.data(),
            camera_scratch.data(),camera_count,intrinsic_dof,camera_values,
            options.fix_first_pose,options.optimize_rotations,
            options.optimize_translations,product.data());
        if (intrinsic_dof>0)
            intrinsic_reduce_kernel<<<blocks(std::max(camera_count,group_count)),threads>>>(
                pose_group.data(),intrinsic_h.data(),pose_intr.data(),direction.data(),
                intrinsic_scratch.data(),camera_count,group_count,intrinsic_dof,
                camera_values,product.data());
        check(cudaGetLastError(),"Schur multiply");
    }
    void record_pcg_iteration(cudaStream_t stream) {
        check(cudaMemsetAsync(point_scratch.data(),0,
            point_count*3*sizeof(double),stream),"point scratch reset");
        check(cudaMemsetAsync(camera_scratch.data(),0,
            camera_count*6*sizeof(double),stream),"camera scratch reset");
        point_gather_kernel<<<blocks(observation_count),threads,0,stream>>>(
            cameras.data(),point_ids.data(),pose_group.data(),cross.data(),
            point_intr.data(),direction.data(),intrinsic_dof,camera_values,
            observation_count,options.fix_first_pose,point_scratch.data());
        check(cudaGetLastError(),"point gather capture");
        point_reduce_kernel<<<blocks(point_count),threads,0,stream>>>(
            point_inverse.data(),point_scratch.data(),point_count,
            point_temporary.data());
        check(cudaGetLastError(),"point reduce capture");
        if (intrinsic_dof>0) {
            zero_values_kernel<<<blocks(intrinsic_values),threads,0,stream>>>(
                intrinsic_scratch.data(),intrinsic_values);
            check(cudaGetLastError(),"intrinsic scratch zero capture");
            zero_values_kernel<<<blocks(intrinsic_values),threads,0,stream>>>(
                product.data()+camera_values,intrinsic_values);
            check(cudaGetLastError(),"intrinsic product zero capture");
        }
        if (intrinsic_dof>0)
        camera_chunk_gather_kernel<<<static_cast<unsigned>(std::min<std::size_t>(camera_chunks.size(),65535)),threads,0,stream>>>(
            camera_chunks.data(),camera_chunks.size(),camera_observations.data(),
            point_ids.data(),pose_group.data(),cross.data(),point_intr.data(),
            point_temporary.data(),intrinsic_dof,camera_scratch.data(),intrinsic_scratch.data());
        else
        camera_gather_kernel<<<blocks(observation_count),threads,0,stream>>>(
            cameras.data(),point_ids.data(),cross.data(),point_temporary.data(),
            observation_count,camera_scratch.data());
        check(cudaGetLastError(),"camera chunk gather capture");
        camera_reduce_kernel<<<blocks(camera_count),threads,0,stream>>>(
            pose_group.data(),camera_h.data(),pose_intr.data(),direction.data(),
            camera_scratch.data(),camera_count,intrinsic_dof,camera_values,
            options.fix_first_pose,options.optimize_rotations,
            options.optimize_translations,product.data());
        check(cudaGetLastError(),"camera reduce capture");
        if (intrinsic_dof>0) {
            intrinsic_reduce_kernel<<<blocks(std::max(camera_count,group_count)),threads,0,stream>>>(
                pose_group.data(),intrinsic_h.data(),pose_intr.data(),direction.data(),
                intrinsic_scratch.data(),camera_count,group_count,intrinsic_dof,
                camera_values,product.data());
            check(cudaGetLastError(),"intrinsic reduce capture");
        }
        dot_to(direction,product,total_values,1,stream);
        alpha_kernel<<<1,1,0,stream>>>(pcg_scalars.data());
        update_pcg_kernel<<<blocks(total_values),threads,0,stream>>>(
            solution.data(),residual.data(),direction.data(),product.data(),
            total_values,pcg_scalars.data()+4);
        precondition_kernel<<<blocks(camera_count),threads,0,stream>>>(
            camera_h.data(),camera_factor.data(),residual.data(),camera_count,options.fix_first_pose,
            options.optimize_rotations,options.optimize_translations,z.data());
        if (intrinsic_dof>0)
            precondition_intrinsics_kernel<<<blocks(group_count),threads,0,stream>>>(
                intrinsic_h.data(),intrinsic_factor.data(),residual.data(),group_count,intrinsic_dof,
                camera_values,z.data());
        dot_to(residual,z,total_values,3,stream);
        beta_kernel<<<1,1,0,stream>>>(pcg_scalars.data());
        direction_kernel<<<blocks(total_values),threads,0,stream>>>(
            direction.data(),z.data(),total_values,pcg_scalars.data()+4);
    }

    // One CUDA-graph replay per ten PCG iterations. Each synchronized
    // iteration would otherwise pay dozens of WDDM kernel submissions, which
    // dominates the actual Schur-complement arithmetic on Windows.
    std::size_t solve_pcg_graph(const double target) {
        const std::size_t budget=options.maximum_pcg_iterations;
        if (budget==0) return 0;
        if (pcg_graph_exec==nullptr) {
            constexpr std::size_t k_chunk=10;
            graph_iterations=std::min(k_chunk,budget);
            cudaStream_t capture{};
            const cudaError_t stream_status =
                cudaStreamCreateWithFlags(&capture,cudaStreamNonBlocking);
            if (stream_status!=cudaSuccess) {
                throw std::runtime_error(std::string("PCG graph stream creation failed: ")+
                    cudaGetErrorString(stream_status));
            }
            check(cudaStreamBeginCapture(capture,cudaStreamCaptureModeThreadLocal),"PCG capture begin");
            try {
                for (std::size_t i=0;i<graph_iterations;++i)
                    record_pcg_iteration(capture);
                check(cudaMemsetAsync(pcg_scalars.data()+2,0,sizeof(double),capture),
                    "PCG residual reset");
                dot_kernel<<<blocks(total_values),threads,0,capture>>>(
                    residual.data(),residual.data(),total_values,pcg_scalars.data()+2);
                check(cudaGetLastError(),"PCG residual dot capture");
                check(cudaStreamEndCapture(capture,&pcg_graph),"PCG capture end");
                check(cudaGraphInstantiateWithFlags(
                          &pcg_graph_exec,pcg_graph,0),
                    "PCG graph instantiate");
            } catch (...) {
                cudaStreamDestroy(capture);
                if (pcg_graph) { cudaGraphDestroy(pcg_graph); pcg_graph=nullptr; }
                throw;
            }
            check(cudaStreamDestroy(capture),"PCG graph stream destroy");
        }
        std::size_t executed=0;
        while (executed+graph_iterations<=budget) {
            check(cudaGraphLaunch(pcg_graph_exec,0),"PCG graph launch");
            double residual_norm{};
            check(cudaMemcpy(&residual_norm,pcg_scalars.data()+2,sizeof(double),
                cudaMemcpyDeviceToHost),"PCG graph residual");
            executed+=graph_iterations;
            if (!std::isfinite(residual_norm)||residual_norm<=target) return executed;
        }
        for (;executed<budget;++executed) record_pcg_iteration(0);
        return executed;
    }
};

CudaOptimizer::CudaOptimizer(OptimizerOptions options) : impl_(std::make_unique<Impl>(options)) {}
CudaOptimizer::~CudaOptimizer()=default;
CudaOptimizer::CudaOptimizer(CudaOptimizer&&) noexcept=default;
CudaOptimizer& CudaOptimizer::operator=(CudaOptimizer&&) noexcept=default;
bool CudaOptimizer::is_available() noexcept { int count=0; return cudaGetDeviceCount(&count)==cudaSuccess && count>0; }
std::string CudaOptimizer::device_name() { int device=0; check(cudaGetDevice(&device),"cudaGetDevice");
    cudaDeviceProp p{}; check(cudaGetDeviceProperties(&p,device),"cudaGetDeviceProperties"); return p.name; }

void CudaOptimizer::upload(const Problem& problem) {
    problem.validate(); auto& d=*impl_; d.camera_count=problem.poses.size(); d.point_count=problem.points.size();
    // Captured kernels retain pointers and topology from the previous upload.
    if (d.pcg_graph_exec) {
        check(cudaGraphExecDestroy(d.pcg_graph_exec),"PCG graph reset");
        d.pcg_graph_exec=nullptr;
    }
    if (d.pcg_graph) {
        check(cudaGraphDestroy(d.pcg_graph),"PCG graph reset");
        d.pcg_graph=nullptr;
    }
    d.observation_count=problem.observations.size();
    d.intrinsic_dof=intrinsic_dof(d.linearizer_settings());
    d.group_count=problem.intrinsics.size();
    d.camera_values=d.camera_count*6;
    d.intrinsic_values=d.group_count*d.intrinsic_dof;
    d.total_values=d.camera_values+d.intrinsic_values;
    d.poses.upload(problem.poses.data(),d.camera_count); d.pose_backup.resize(d.camera_count);
    std::vector<Index> pose_group(d.camera_count);
    for (std::size_t pose=0; pose<d.camera_count; ++pose)
        pose_group[pose]=problem.intrinsic_index(pose);
    d.pose_group.upload(pose_group.data(),d.camera_count);
    std::vector<PinholeIntrinsics> group_intrinsics=problem.intrinsics;
    // Shared-focal optimization keeps fx and fy moving together (CPU parity).
    if (d.options.optimize_focal && !d.options.optimize_aspect_ratio) {
        for (PinholeIntrinsics& intrinsics : group_intrinsics) {
            const double focal=0.5*(intrinsics.fx+intrinsics.fy);
            intrinsics.fx=focal; intrinsics.fy=focal;
        }
    }
    d.intrinsics.upload(group_intrinsics.data(),d.group_count);
    d.intrinsic_backup.resize(d.group_count);
    const std::vector<PinholeIntrinsics> group_initial=
        problem.initial_intrinsics.empty() ? group_intrinsics : problem.initial_intrinsics;
    d.initial_intrinsics.upload(group_initial.data(),d.group_count);
    const std::vector<unsigned char> zero_constants(d.group_count,0);
    d.intrinsic_constant.upload(
        problem.intrinsic_constant.empty() ? zero_constants.data()
                                           : problem.intrinsic_constant.data(),
        d.group_count);
    d.points.upload(problem.points.data(),d.point_count); d.point_backup.resize(d.point_count);
    d.cameras.upload(problem.observations.camera.data(),d.observation_count);
    d.point_ids.upload(problem.observations.point.data(),d.observation_count);
    d.ox.upload(problem.observations.x.data(),d.observation_count); d.oy.upload(problem.observations.y.data(),d.observation_count);
    d.weights.upload(problem.observations.weight.data(),d.observation_count);
    std::vector<std::size_t> po(d.point_count+1),co(d.camera_count+1),pi(d.observation_count),ci(d.observation_count);
    for (std::size_t i=0;i<d.observation_count;++i) { ++po[problem.observations.point[i]+1]; ++co[problem.observations.camera[i]+1]; }
    std::partial_sum(po.begin(),po.end(),po.begin()); std::partial_sum(co.begin(),co.end(),co.begin());
    auto pc=po,cc=co; for (std::size_t i=0;i<d.observation_count;++i) {
        pi[pc[problem.observations.point[i]]++]=i; ci[cc[problem.observations.camera[i]]++]=i; }
    d.point_offsets.upload(po.data(),po.size()); d.point_observations.upload(pi.data(),pi.size());
    d.camera_offsets.upload(co.data(),co.size()); d.camera_observations.upload(ci.data(),ci.size());
    std::vector<CameraChunk> chunks;
    for (std::size_t camera=0;camera<d.camera_count;++camera)
        for (std::size_t begin=co[camera];begin<co[camera+1];begin+=threads)
            chunks.push_back({begin,std::min(begin+threads,co[camera+1]),static_cast<Index>(camera)});
    d.camera_chunks.upload(chunks.data(),chunks.size());
    d.linearized.resize(d.observation_count); d.candidate_linearized.resize(d.observation_count);
    d.camera_h.resize(d.camera_count*36); d.camera_b.resize(d.camera_count*6);
    d.camera_factor.resize(d.camera_count*36);
    d.intrinsic_factor.resize(d.group_count*k_max_intrinsic_params*k_max_intrinsic_params);
    d.point_inverse.resize(d.point_count*9); d.point_b.resize(d.point_count*3); d.cross.resize(d.observation_count*18);
    if (d.intrinsic_dof>0) {
        d.intrinsic_h.resize(d.group_count*d.intrinsic_dof*d.intrinsic_dof);
        d.intrinsic_b.resize(d.group_count*d.intrinsic_dof);
        d.pose_intr.resize(d.camera_count*6*d.intrinsic_dof);
        d.point_intr.resize(d.observation_count*3*d.intrinsic_dof);
        d.intrinsic_scratch.resize(d.intrinsic_values);
    }
    d.point_scratch.resize(d.point_count*3);
    d.camera_scratch.resize(d.camera_count*6);
    d.reduced_point.resize(d.point_count*3);
    d.rhs.resize(d.total_values); d.solution.resize(d.total_values); d.residual.resize(d.total_values);
    d.z.resize(d.total_values); d.direction.resize(d.total_values);
    d.product.resize(d.total_values); d.point_temporary.resize(d.point_count*3); d.point_step.resize(d.point_count*3);
    d.scalar.resize(1); d.pcg_scalars.resize(8);
}

OptimizerSummary CudaOptimizer::optimize() {
    core::StageScope stage("ba.cuda");
    auto& d=*impl_; if (!d.observation_count) throw std::logic_error("Upload a BA problem before optimization");
    const auto started=std::chrono::steady_clock::now(); OptimizerSummary summary;
    linearize_kernel<<<blocks(d.observation_count),threads>>>(
        d.poses.data(),d.intrinsics.data(),d.pose_group.data(),d.points.data(),
        d.cameras.data(),d.point_ids.data(),d.ox.data(),d.oy.data(),
        d.weights.data(),d.observation_count,
        d.linearizer_settings(),
        d.linearized.data());
    summary.initial_cost=d.cost(d.linearized); summary.final_cost=summary.initial_cost;
    double damping=d.options.initial_damping;
    const std::size_t camera_values=d.camera_count*6;
    const std::size_t intrinsic_values=d.intrinsic_values;
    const std::size_t total_values=d.total_values;
    for (std::size_t iteration=0;iteration<d.options.maximum_iterations;++iteration) {
        assemble_points_and_cross_kernel<<<warp_blocks(d.point_count),threads>>>(
            d.linearized.data(),d.point_offsets.data(),d.point_observations.data(),
            d.point_count,d.intrinsic_dof,d.point_inverse.data(),d.point_b.data(),
            d.cross.data(),d.point_intr.data());
        if (d.intrinsic_dof>0) {
            check(cudaMemset(d.intrinsic_h.data(),0,
                d.intrinsic_h.size()*sizeof(double)),"intrinsic Hessian reset");
            check(cudaMemset(d.intrinsic_b.data(),0,
                d.intrinsic_b.size()*sizeof(double)),"intrinsic rhs reset");
        }
        assemble_cameras_and_intrinsics_kernel<<<warp_blocks(d.camera_count),threads>>>(
            d.linearized.data(),d.camera_offsets.data(),d.camera_observations.data(),
            d.camera_count,d.pose_group.data(),d.intrinsic_dof,d.camera_h.data(),
            d.camera_b.data(),d.pose_intr.data(),d.intrinsic_h.data(),d.intrinsic_b.data());
        damp_cameras<<<blocks(d.camera_count),threads>>>(d.camera_h.data(),d.camera_count,damping);
        if (d.intrinsic_dof>0) {
            mask_constant_intrinsics_kernel<<<blocks(d.camera_count),threads>>>(
                d.cameras.data(),d.pose_group.data(),d.intrinsic_constant.data(),
                d.camera_count,d.observation_count,d.intrinsic_dof,
                d.pose_intr.data(),d.point_intr.data());
            finalize_intrinsics_kernel<<<blocks(d.group_count),threads>>>(
                d.intrinsic_h.data(),d.intrinsic_b.data(),d.intrinsics.data(),
                d.initial_intrinsics.data(),d.intrinsic_constant.data(),
                d.group_count,d.intrinsic_dof,damping,
                static_cast<double>(std::max<std::size_t>(d.observation_count,1)),
                d.intrinsic_settings());
        }
        invert_points<<<blocks(d.point_count),threads>>>(
            d.point_inverse.data(),d.point_b.data(),d.point_count,damping,
            d.options.fix_first_point,d.options.optimize_points);
        reduced_point_rhs_kernel<<<blocks(d.point_count),threads>>>(d.point_inverse.data(),d.point_b.data(),d.point_count,d.reduced_point.data());
        schur_rhs_kernel<<<blocks(d.camera_count),threads>>>(d.camera_offsets.data(),d.camera_observations.data(),d.point_ids.data(),
            d.camera_b.data(),d.reduced_point.data(),d.cross.data(),d.camera_count,
            d.options.fix_first_pose,d.options.optimize_rotations,
            d.options.optimize_translations,d.rhs.data());
        if (d.intrinsic_dof>0) {
            copy_intrinsic_rhs_kernel<<<blocks(intrinsic_values),threads>>>(
                d.intrinsic_b.data(),intrinsic_values,d.rhs.data()+camera_values);
            intrinsic_schur_rhs_kernel<<<blocks(d.point_count),threads>>>(
                d.point_offsets.data(),d.point_observations.data(),d.cameras.data(),
                d.pose_group.data(),d.reduced_point.data(),d.point_intr.data(),
                d.point_count,camera_values,d.intrinsic_dof,d.rhs.data());
        }
        check(cudaGetLastError(),"system assembly");
        factor_preconditioner_kernel<<<blocks(d.camera_count),threads>>>(
            d.camera_h.data(),d.camera_count,6,6,d.camera_factor.data());
        if (d.intrinsic_dof>0)
            factor_preconditioner_kernel<<<blocks(d.group_count),threads>>>(
                d.intrinsic_h.data(),d.group_count,d.intrinsic_dof,
                k_max_intrinsic_params,d.intrinsic_factor.data());
        std::size_t pcg=0;
        {
          d.solution.zero(); check(cudaMemcpy(d.residual.data(),d.rhs.data(),total_values*sizeof(double),cudaMemcpyDeviceToDevice),"PCG init");
          precondition_kernel<<<blocks(d.camera_count),threads>>>(d.camera_h.data(),d.camera_factor.data(),d.residual.data(),d.camera_count,
              d.options.fix_first_pose,d.options.optimize_rotations,d.options.optimize_translations,d.z.data());
          if (d.intrinsic_dof>0)
              precondition_intrinsics_kernel<<<blocks(d.group_count),threads>>>(
                  d.intrinsic_h.data(),d.intrinsic_factor.data(),d.residual.data(),d.group_count,d.intrinsic_dof,
                  camera_values,d.z.data());
          check(cudaMemcpy(d.direction.data(),d.z.data(),camera_values*sizeof(double),cudaMemcpyDeviceToDevice),"PCG direction");
          if (d.intrinsic_dof>0)
              check(cudaMemcpy(d.direction.data()+camera_values,d.z.data()+camera_values,
                  intrinsic_values*sizeof(double),cudaMemcpyDeviceToDevice),"PCG intrinsic direction");
          d.dot_to(d.residual,d.z,total_values,0); d.dot_to(d.rhs,d.rhs,total_values,1);
          double rhs_norm{}; check(cudaMemcpy(&rhs_norm,d.pcg_scalars.data()+1,sizeof(double),cudaMemcpyDeviceToHost),"PCG rhs norm");
          const double target=d.options.pcg_tolerance*d.options.pcg_tolerance*std::max(rhs_norm,1e-30);
          bool graph_solved=false;
          try {
              pcg=d.solve_pcg_graph(target);
              graph_solved=true;
          } catch (const std::exception& error) {
              // The failed capture can leave a sticky launch error; clear it
              // before the fallback path reports a phantom failure.
              (void)cudaGetLastError();
              core::Logger::instance().warning(
                  "CUDA PCG graph capture unavailable; falling back to the "
                  "synchronized iteration path: ",error.what());
              d.solution.zero();
              check(cudaMemcpy(d.residual.data(),d.rhs.data(),total_values*sizeof(double),cudaMemcpyDeviceToDevice),"PCG init");
              precondition_kernel<<<blocks(d.camera_count),threads>>>(d.camera_h.data(),d.camera_factor.data(),d.residual.data(),d.camera_count,
                  d.options.fix_first_pose,d.options.optimize_rotations,d.options.optimize_translations,d.z.data());
              if (d.intrinsic_dof>0)
                  precondition_intrinsics_kernel<<<blocks(d.group_count),threads>>>(
                      d.intrinsic_h.data(),d.intrinsic_factor.data(),d.residual.data(),d.group_count,d.intrinsic_dof,
                      camera_values,d.z.data());
              check(cudaMemcpy(d.direction.data(),d.z.data(),total_values*sizeof(double),cudaMemcpyDeviceToDevice),"PCG direction");
              d.dot_to(d.residual,d.z,total_values,0);
          }
          if (!graph_solved)
          for (;pcg<d.options.maximum_pcg_iterations;++pcg) {
            d.multiply(); d.dot_to(d.direction,d.product,total_values,1); alpha_kernel<<<1,1>>>(d.pcg_scalars.data());
            update_pcg_kernel<<<blocks(total_values),threads>>>(d.solution.data(),d.residual.data(),d.direction.data(),d.product.data(),total_values,d.pcg_scalars.data()+4);
            if ((pcg+1)%10==0 || pcg+1==d.options.maximum_pcg_iterations) {
                d.dot_to(d.residual,d.residual,total_values,2); double residual_norm{};
                check(cudaMemcpy(&residual_norm,d.pcg_scalars.data()+2,sizeof(double),cudaMemcpyDeviceToHost),"PCG residual norm");
                if (!std::isfinite(residual_norm) || residual_norm<=target) { ++pcg; break; }
            }
            precondition_kernel<<<blocks(d.camera_count),threads>>>(d.camera_h.data(),d.camera_factor.data(),d.residual.data(),d.camera_count,
                d.options.fix_first_pose,d.options.optimize_rotations,d.options.optimize_translations,d.z.data());
            if (d.intrinsic_dof>0)
                precondition_intrinsics_kernel<<<blocks(d.group_count),threads>>>(
                    d.intrinsic_h.data(),d.intrinsic_factor.data(),d.residual.data(),d.group_count,d.intrinsic_dof,
                    camera_values,d.z.data());
            d.dot_to(d.residual,d.z,total_values,3); beta_kernel<<<1,1>>>(d.pcg_scalars.data());
            direction_kernel<<<blocks(total_values),threads>>>(d.direction.data(),d.z.data(),total_values,d.pcg_scalars.data()+4);
          }
        }
        recover_points_kernel<<<blocks(d.point_count),threads>>>(d.point_offsets.data(),d.point_observations.data(),d.cameras.data(),
            d.pose_group.data(),d.cross.data(),d.point_intr.data(),d.point_inverse.data(),
            d.point_b.data(),d.solution.data(),d.intrinsic_dof,camera_values,
            d.point_count,d.point_step.data());
        double norm_sq=d.dot(d.solution,d.solution,total_values)+
            d.dot(d.point_step,d.point_step,d.point_count*3);
        const double norm=std::sqrt(norm_sq);
        check(cudaMemcpy(d.pose_backup.data(),d.poses.data(),d.camera_count*sizeof(Pose),cudaMemcpyDeviceToDevice),"pose backup");
        check(cudaMemcpy(d.point_backup.data(),d.points.data(),d.point_count*sizeof(Point3),cudaMemcpyDeviceToDevice),"point backup");
        check(cudaMemcpy(d.intrinsic_backup.data(),d.intrinsics.data(),
            d.group_count*sizeof(PinholeIntrinsics),cudaMemcpyDeviceToDevice),"intrinsic backup");
        update_poses_kernel<<<blocks(d.camera_count),threads>>>(d.poses.data(),d.solution.data(),d.camera_count,
            d.options.fix_first_pose,d.options.optimize_rotations,d.options.optimize_translations);
        update_points_kernel<<<blocks(d.point_count),threads>>>(d.points.data(),d.point_step.data(),d.point_count);
        if (d.intrinsic_dof>0)
            update_intrinsics_kernel<<<blocks(d.group_count),threads>>>(
                d.intrinsics.data(),d.initial_intrinsics.data(),d.intrinsic_constant.data(),
                d.solution.data(),d.group_count,d.intrinsic_dof,camera_values,
                d.intrinsic_settings());
        linearize_kernel<<<blocks(d.observation_count),threads>>>(
            d.poses.data(),d.intrinsics.data(),d.pose_group.data(),d.points.data(),
            d.cameras.data(),d.point_ids.data(),d.ox.data(),d.oy.data(),
            d.weights.data(),d.observation_count,
            d.linearizer_settings(),
            d.candidate_linearized.data());
        const double candidate=d.cost(d.candidate_linearized);
        const bool accepted=std::isfinite(candidate)&&candidate<summary.final_cost;
        summary.iterations.push_back({iteration,accepted?candidate:summary.final_cost,damping,norm,pcg,accepted});
        core::Logger::instance().debug(
            "CUDA BA iteration=",iteration," cost=",summary.iterations.back().cost,
            " damping=",damping," step_norm=",norm,
            " pcg_iterations=",pcg," accepted=",accepted);
        if (accepted) { const double previous=summary.final_cost; summary.final_cost=candidate;
            d.linearized.swap(d.candidate_linearized); ++summary.successful_steps;
            damping=std::max(d.options.minimum_damping,damping/3.0);
            if (norm<=d.options.step_tolerance || previous-candidate<=d.options.function_tolerance*std::max(1.0,previous)) {
                summary.termination=TerminationReason::converged; break; }
        } else { check(cudaMemcpy(d.poses.data(),d.pose_backup.data(),d.camera_count*sizeof(Pose),cudaMemcpyDeviceToDevice),"pose rollback");
            check(cudaMemcpy(d.points.data(),d.point_backup.data(),d.point_count*sizeof(Point3),cudaMemcpyDeviceToDevice),"point rollback");
            check(cudaMemcpy(d.intrinsics.data(),d.intrinsic_backup.data(),
                d.group_count*sizeof(PinholeIntrinsics),cudaMemcpyDeviceToDevice),"intrinsic rollback");
            ++summary.unsuccessful_steps; damping=std::min(d.options.maximum_damping,damping*10.0);
            if (damping>=d.options.maximum_damping) {
                summary.termination=summary.successful_steps>0
                    ? TerminationReason::maximum_iterations
                    : TerminationReason::numerical_failure;
                break;
            } }
    }
    check(cudaDeviceSynchronize(),"CUDA optimizer synchronize");
    summary.total_time_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    stage.finish(summary.brief_report());
    return summary;
}

void CudaOptimizer::download(Problem& problem) const {
    auto& d=*impl_; if (problem.poses.size()!=d.camera_count || problem.points.size()!=d.point_count)
        throw std::invalid_argument("Download target topology differs from uploaded BA problem");
    d.poses.download(problem.poses.data(),d.camera_count); d.points.download(problem.points.data(),d.point_count);
    if (problem.intrinsics.size()!=d.group_count)
        throw std::invalid_argument("Download target intrinsics differ from uploaded BA problem");
    d.intrinsics.download(problem.intrinsics.data(),d.group_count);
}

OptimizerSummary optimize_cuda(Problem& problem,const OptimizerOptions& options) {
    CudaOptimizer optimizer(options); optimizer.upload(problem); auto summary=optimizer.optimize(); optimizer.download(problem); return summary;
}

}  // namespace aetherscan::ba
