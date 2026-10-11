#include "pam_cuda.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cfloat>
#include <stdexcept>
#include <string>

namespace photara::splat::detail {
namespace {

constexpr unsigned block_size = 128;
constexpr std::size_t query_chunk = 65536;

__device__ bool greater(float a, int ai, float b, int bi) {
    return a > b || (a == b && ai > bi);
}

__device__ void sift_down(float* distances, int* indices, int count) {
    int parent = 0;
    while (2 * parent + 1 < count) {
        int child = 2 * parent + 1;
        if (child + 1 < count && greater(distances[child + 1], indices[child + 1],
                                        distances[child], indices[child])) ++child;
        if (!greater(distances[child], indices[child],
                     distances[parent], indices[parent])) break;
        float d = distances[parent]; distances[parent] = distances[child]; distances[child] = d;
        int i = indices[parent]; indices[parent] = indices[child]; indices[child] = i;
        parent = child;
    }
}

__device__ float3 field_gradient(
    const int* nodes, const float* values, const float* bounds, int root,
    float3 query, int neighbors) {
    float distances[pam_gpu_max_neighbors];
    int indices[pam_gpu_max_neighbors];
    int count = 0;
    // A balanced tree with int node indices has depth at most 32. Deferred
    // far branches are tested AFTER the near subtree, just like CPU search.
    int pending[64];
    float planes[64];
    int size = 1;
    pending[0] = root; planes[0] = -1.F;
    while (size) {
        --size;
        const int node = pending[size];
        if (node < 0 || (count == neighbors && planes[size] > distances[0])) continue;
        if (count == neighbors) {
            const float* box = bounds + 6 * node;
            const float bx = fmaxf(fmaxf(box[0] - query.x, query.x - box[3]), 0.F);
            const float by = fmaxf(fmaxf(box[1] - query.y, query.y - box[4]), 0.F);
            const float bz = fmaxf(fmaxf(box[2] - query.z, query.z - box[5]), 0.F);
            // Conservative tolerance prevents rounding at box boundaries from
            // dropping a neighbor visited by the CPU reference.
            const float lower_bound = bx * bx + (by * by + bz * bz);
            if (lower_bound > distances[0] + 16.F * FLT_EPSILON * (1.F + distances[0]))
                continue;
        }
        const int* entry = nodes + 4 * node;
        const int point = entry[0];
        const float* g = values + 19 * point;
        const float dx = query.x - g[0], dy = query.y - g[1], dz = query.z - g[2];
        // Eigen's three-component reduction is x + (y + z). Preserve
        // its ordering so equidistant neighbors select the same heap entries.
        const float distance = dx * dx + (dy * dy + dz * dz);
        if (count < neighbors) {
            int position = count++;
            distances[position] = distance; indices[position] = point;
            while (position > 0) {
                const int parent = (position - 1) / 2;
                if (!greater(distances[position], indices[position],
                             distances[parent], indices[parent])) break;
                float d = distances[parent]; distances[parent] = distances[position]; distances[position] = d;
                int i = indices[parent]; indices[parent] = indices[position]; indices[position] = i;
                position = parent;
            }
        } else if (distance < distances[0]) {
            distances[0] = distance; indices[0] = point;
            sift_down(distances, indices, count);
        }
        const float delta = entry[3] == 0 ? dx : entry[3] == 1 ? dy : dz;
        const int near_child = delta < 0.F ? entry[1] : entry[2];
        const int far_child = delta < 0.F ? entry[2] : entry[1];
        if (far_child >= 0) { pending[size] = far_child; planes[size++] = delta * delta; }
        if (near_child >= 0) { pending[size] = near_child; planes[size++] = -1.F; }
    }
    int ordered[pam_gpu_max_neighbors];
    const int found = count;
    while (count) {
        ordered[count - 1] = indices[0];
        --count;
        if (count) {
            distances[0] = distances[count]; indices[0] = indices[count];
            sift_down(distances, indices, count);
        }
    }
    float3 result = make_float3(0.F, 0.F, 0.F);
    for (int i = 0; i < found; ++i) {
        const float* g = values + 19 * ordered[i];
        const float dx = query.x - g[0], dy = query.y - g[1], dz = query.z - g[2];
        if (g[3] * dx + (g[4] * dy + g[5] * dz) < 0.F) continue;
        const float x = g[9] * dx + (g[10] * dy + g[11] * dz);
        const float y = g[12] * dx + (g[13] * dy + g[14] * dz);
        const float z = g[15] * dx + (g[16] * dy + g[17] * dz);
        const float exponent = -0.5F * (x * x * g[6] + (y * y * g[7] + z * z * g[8]));
        const float gaussian = fminf(fmaxf(g[18] * expf(exponent), 0.F), 1.F - 1e-7F);
        const float weight = gaussian / (1.F - gaussian + 1e-8F);
        const float vx = x * g[6], vy = y * g[7], vz = z * g[8];
        result.x += weight * (g[9] * vx + (g[12] * vy + g[15] * vz));
        result.y += weight * (g[10] * vx + (g[13] * vy + g[16] * vz));
        result.z += weight * (g[11] * vx + (g[14] * vy + g[17] * vz));
    }
    return result;
}

__global__ void gradient_kernel(
    const int* nodes, const float* values, const float* bounds, int root, const float* points,
    float* gradients, std::size_t count, int neighbors) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float3 g = field_gradient(nodes, values, bounds, root,
        make_float3(points[3*i], points[3*i+1], points[3*i+2]), neighbors);
    gradients[3*i] = g.x; gradients[3*i+1] = g.y; gradients[3*i+2] = g.z;
}

__global__ void refine_kernel(
    const int* nodes, const float* values, const float* bounds, int root, float* points,
    const float* occupancy, std::size_t count, int neighbors,
    float iso_value, float minimum_norm_squared, float step) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float3 g = field_gradient(nodes, values, bounds, root,
        make_float3(points[3*i], points[3*i+1], points[3*i+2]), neighbors);
    const float norm = g.x*g.x + (g.y*g.y + g.z*g.z);
    if (!(norm > minimum_norm_squared)) return;
    const float alpha = fminf(fmaxf((occupancy[i] - iso_value) / norm, -1.F), 1.F);
    points[3*i] += step * alpha * g.x;
    points[3*i+1] += step * alpha * g.y;
    points[3*i+2] += step * alpha * g.z;
}

void check(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("PAM CUDA field: ") + cudaGetErrorString(error));
}

int validate(const PamGpuField& field, const tinytensor::Tensor& points, unsigned neighbors) {
    if (!points.is_valid() || points.device() != tinytensor::Device::CUDA ||
        !points.is_contiguous() || points.dtype() != tinytensor::DataType::Float32 ||
        points.shape().rank() != 2 || points.shape()[1] != 3 ||
        field.root < 0 || !field.nodes.is_valid() || !field.values.is_valid() || !field.bounds.is_valid() ||
        neighbors == 0 || neighbors > pam_gpu_max_neighbors)
        throw std::invalid_argument("Invalid PAM CUDA field query");
    return static_cast<int>(std::min<std::size_t>(neighbors, field.values.shape()[0]));
}

}  // namespace

tinytensor::Tensor pam_field_gradients(
    const PamGpuField& field, const tinytensor::Tensor& points, unsigned neighbors) {
    const int k = validate(field, points, neighbors);
    const std::size_t count = points.shape()[0];
    auto gradients = tinytensor::Tensor::empty({count, std::size_t{3}}, tinytensor::Device::CUDA);
    for (std::size_t begin = 0; begin < count; begin += query_chunk) {
        const std::size_t size = std::min(query_chunk, count - begin);
        gradient_kernel<<<(size + block_size - 1) / block_size, block_size>>>(
            field.nodes.ptr<int>(), field.values.ptr<float>(), field.bounds.ptr<float>(), field.root,
            points.ptr<float>() + 3*begin, gradients.ptr<float>() + 3*begin, size, k);
        check(cudaGetLastError());
    }
    return gradients;
}

void pam_refine_points(
    const PamGpuField& field, tinytensor::Tensor& points,
    const tinytensor::Tensor& occupancy, unsigned neighbors,
    float iso_value, float minimum_gradient_norm_squared, float step) {
    const int k = validate(field, points, neighbors);
    const std::size_t count = points.shape()[0];
    if (!occupancy.is_valid() || occupancy.device() != tinytensor::Device::CUDA ||
        !occupancy.is_contiguous() || occupancy.dtype() != tinytensor::DataType::Float32 ||
        occupancy.shape().rank() != 1 || occupancy.shape()[0] != count)
        throw std::invalid_argument("Invalid PAM CUDA occupancy");
    for (std::size_t begin = 0; begin < count; begin += query_chunk) {
        const std::size_t size = std::min(query_chunk, count - begin);
        refine_kernel<<<(size + block_size - 1) / block_size, block_size>>>(
            field.nodes.ptr<int>(), field.values.ptr<float>(), field.bounds.ptr<float>(), field.root,
            points.ptr<float>() + 3*begin, occupancy.ptr<float>() + begin, size, k,
            iso_value, minimum_gradient_norm_squared, step);
        check(cudaGetLastError());
    }
}

}  // namespace photara::splat::detail
