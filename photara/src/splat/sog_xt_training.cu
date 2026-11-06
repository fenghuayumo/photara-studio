#include "formats_sog_xt.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

namespace photara::splat::detail {
namespace {
__device__ float quantize(float value, float low, float high, unsigned levels) {
    if (high <= low)
        return low;
    return low + (high - low) *
                     roundf(fminf(1.F, fmaxf(0.F, (value - low) / (high - low))) * (levels - 1)) /
                     (levels - 1);
}
__device__ float field_value(
    unsigned d,
    unsigned i,
    const float* means,
    const float* scales,
    const float* q,
    const float* opacity,
    const float* sh,
    unsigned stride) {
    if (d < 3)
        return copysignf(log1pf(fabsf(means[i * 3 + d])), means[i * 3 + d]);
    if (d < 6)
        return scales[i * 3 + d - 3];
    if (d < 10)
        return q[i * 4 + d - 6];
    if (d == 10)
        return opacity[i];
    return sh[i * stride + d - 11];
}
__device__ float* field_ptr(
    unsigned d,
    unsigned i,
    float* means,
    float* scales,
    float* q,
    float* opacity,
    float* sh,
    unsigned stride) {
    if (d < 3)
        return means + i * 3 + d;
    if (d < 6)
        return scales + i * 3 + d - 3;
    if (d < 10)
        return q + i * 4 + d - 6;
    if (d == 10)
        return opacity + i;
    return sh + i * stride + d - 11;
}
__global__ void fields_forward(
    unsigned n,
    unsigned stride,
    const float* low,
    const float* high,
    const float* m,
    const float* s,
    const float* q,
    const float* a,
    const float* sh,
    float* om,
    float* os,
    float* oq,
    float* oa,
    float* osh) {
    unsigned at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at >= n * 14)
        return;
    unsigned i = at / 14, d = at % 14;
    float v = field_value(d, i, m, s, q, a, sh, stride);
    v = quantize(v, low[d], high[d], d < 3 ? 65536 : d >= 6 && d < 10 ? 100 : 256);
    if (d < 3)
        v = copysignf(expm1f(fabsf(v)), v);
    *field_ptr(d, i, om, os, oq, oa, osh, stride) = v;
}
__global__ void center_sum(
    unsigned n,
    unsigned dims,
    unsigned stride,
    const int* labels,
    const int* counts,
    const float* sh,
    float* centers) {
    unsigned at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at >= n * dims)
        return;
    unsigned i = at / dims, d = at % dims;
    atomicAdd(
        centers + std::size_t(labels[i]) * dims + d, sh[i * stride + 3 + d] / counts[labels[i]]);
}
__global__ void center_quant(
    unsigned size,
    unsigned dims,
    const float* low,
    const float* high,
    float* centers) {
    unsigned at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at < size)
        centers[at] = quantize(centers[at], low[at % dims], high[at % dims], 100);
}
__global__ void center_expand(
    unsigned n,
    unsigned dims,
    unsigned stride,
    const int* labels,
    const float* centers,
    float* sh) {
    unsigned at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at < n * dims)
        sh[(at / dims) * stride + 3 + at % dims] =
            centers[std::size_t(labels[at / dims]) * dims + at % dims];
}
__global__ void center_grad_sum(
    unsigned n,
    unsigned dims,
    unsigned stride,
    const int* labels,
    const float* gradient,
    float* sums) {
    unsigned at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at < n * dims)
        atomicAdd(
            sums + std::size_t(labels[at / dims]) * dims + at % dims,
            gradient[(at / dims) * stride + 3 + at % dims]);
}
__global__ void center_grad_expand(
    unsigned n,
    unsigned dims,
    unsigned stride,
    const int* labels,
    const int* counts,
    const float* sums,
    const float* raw,
    const float* low,
    const float* high,
    float* gradient) {
    unsigned at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at < n * dims) {
        unsigned d = at % dims;
        auto c = std::size_t(labels[at / dims]) * dims + d;
        gradient[(at / dims) * stride + 3 + d] =
            raw[c] >= low[d] && raw[c] <= high[d] && high[d] > low[d]
                ? sums[c] / counts[labels[at / dims]]
                : 0;
    }
}
__global__ void tv_gradient(
    unsigned n,
    unsigned stride,
    const int* neighbors,
    const float* low,
    const float* high,
    const float* m,
    const float* s,
    const float* q,
    const float* a,
    const float* sh,
    float* gm,
    float* gs,
    float* gq,
    float* ga,
    float* gsh,
    float weight,
    float qweight) {
    unsigned at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at >= n * 14)
        return;
    unsigned i = at / 14, d = at % 14;
    float range = high[d] - low[d];
    if (range <= 0)
        return;
    const float v = field_value(d, i, m, s, q, a, sh, stride);
    const float w = (d >= 6 && d < 10 ? qweight : weight) / (float(n) * 2 * 14 * range);
    for (unsigned axis = 0; axis < 2; ++axis) {
        int j = neighbors[i * 2 + axis];
        if (j < 0)
            continue;
        float delta = v - field_value(d, unsigned(j), m, s, q, a, sh, stride);
        float g = delta > 0 ? w : delta < 0 ? -w : 0;
        // TV is defined on encoded proxies. Its position derivative below
        // combines with the signed-log codec Jacobian in fields_backward.
        if (d < 3) {
            atomicAdd(gm + i * 3 + d, g / (1.F + fabsf(m[i * 3 + d])));
            atomicAdd(gm + j * 3 + d, -g / (1.F + fabsf(m[j * 3 + d])));
        } else {
            atomicAdd(field_ptr(d, i, gm, gs, gq, ga, gsh, stride), g);
            atomicAdd(field_ptr(d, unsigned(j), gm, gs, gq, ga, gsh, stride), -g);
        }
    }
}
__global__ void fields_backward(
    unsigned n,
    unsigned stride,
    const float* low,
    const float* high,
    const float* m,
    const float* s,
    const float* q,
    const float* a,
    const float* sh,
    const float* decoded_m,
    float* gm,
    float* gs,
    float* gq,
    float* ga,
    float* gsh) {
    unsigned at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at >= n * 14)
        return;
    unsigned i = at / 14, d = at % 14;
    float v = field_value(d, i, m, s, q, a, sh, stride);
    auto* g = field_ptr(d, i, gm, gs, gq, ga, gsh, stride);
    if (v < low[d] || v > high[d] || high[d] <= low[d])
        *g = 0;
    else if (d < 3)
        *g *= (1.F + fabsf(decoded_m[i * 3 + d])) / (1.F + fabsf(m[i * 3 + d]));
}
void check() {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}
} // namespace

GaussianModel sog_xt_training_forward(SogXtTrainingCodec& codec) {
    using tinytensor::Tensor;
    auto& p = codec.latent;
    if (p.means.device() != tinytensor::Device::CUDA)
        throw std::invalid_argument("SOG-XT adaptation requires CUDA tensors");
    GaussianModel out = p;
    out.means = Tensor::empty_like(p.means);
    out.log_scales = Tensor::empty_like(p.log_scales);
    out.quaternions = Tensor::empty_like(p.quaternions);
    out.opacity_logits = Tensor::empty_like(p.opacity_logits);
    out.sh = Tensor::empty_like(p.sh);
    unsigned n = unsigned(p.size()), stride = unsigned(p.sh.shape()[1] * 3),
             dims = codec.dimensions;
    fields_forward<<<(n * 14 + 255) / 256, 256>>>(
        n, stride, codec.low.ptr<float>(), codec.high.ptr<float>(), p.means.ptr<float>(),
        p.log_scales.ptr<float>(), p.quaternions.ptr<float>(), p.opacity_logits.ptr<float>(),
        p.sh.ptr<float>(), out.means.ptr<float>(), out.log_scales.ptr<float>(),
        out.quaternions.ptr<float>(), out.opacity_logits.ptr<float>(), out.sh.ptr<float>());
    codec.centers = Tensor::zeros(
        {std::size_t(codec.center_side) * codec.center_side, std::size_t(dims)}, p.means.device());
    center_sum<<<(n * dims + 255) / 256, 256>>>(
        n, dims, stride, codec.labels.ptr<int>(), codec.counts.ptr<int>(), p.sh.ptr<float>(),
        codec.centers.ptr<float>());
    codec.raw_centers = codec.centers.clone();
    center_quant<<<(codec.centers.numel() + 255) / 256, 256>>>(
        unsigned(codec.centers.numel()), dims, codec.sh_low.ptr<float>(),
        codec.sh_high.ptr<float>(), codec.centers.ptr<float>());
    center_expand<<<(n * dims + 255) / 256, 256>>>(
        n, dims, stride, codec.labels.ptr<int>(), codec.centers.ptr<float>(), out.sh.ptr<float>());
    check();
    return out;
}

void sog_xt_training_backward(
    SogXtTrainingCodec& codec,
    const GaussianModel& decoded,
    ModelGradients& g,
    float tv,
    float qtv) {
    auto& p = codec.latent;
    unsigned n = unsigned(p.size()), stride = unsigned(p.sh.shape()[1] * 3),
             dims = codec.dimensions;
    fields_backward<<<(n * 14 + 255) / 256, 256>>>(
        n, stride, codec.low.ptr<float>(), codec.high.ptr<float>(), p.means.ptr<float>(),
        p.log_scales.ptr<float>(), p.quaternions.ptr<float>(), p.opacity_logits.ptr<float>(),
        p.sh.ptr<float>(), decoded.means.ptr<float>(), g.means.ptr<float>(),
        g.log_scales.ptr<float>(), g.quaternions.ptr<float>(), g.opacity_logits.ptr<float>(),
        g.sh.ptr<float>());
    // TV acts on latent proxy images with STE rounding; add after the codec
    // Jacobian so its own proxy-space Jacobian is applied exactly once.
    if (tv > 0 || qtv > 0)
        tv_gradient<<<(n * 14 + 255) / 256, 256>>>(
            n, stride, codec.neighbors.ptr<int>(), codec.low.ptr<float>(), codec.high.ptr<float>(),
            p.means.ptr<float>(), p.log_scales.ptr<float>(), p.quaternions.ptr<float>(),
            p.opacity_logits.ptr<float>(), p.sh.ptr<float>(), g.means.ptr<float>(),
            g.log_scales.ptr<float>(), g.quaternions.ptr<float>(), g.opacity_logits.ptr<float>(),
            g.sh.ptr<float>(), tv, qtv);
    auto sums = tinytensor::Tensor::zeros_like(codec.centers);
    center_grad_sum<<<(n * dims + 255) / 256, 256>>>(
        n, dims, stride, codec.labels.ptr<int>(), g.sh.ptr<float>(), sums.ptr<float>());
    center_grad_expand<<<(n * dims + 255) / 256, 256>>>(
        n, dims, stride, codec.labels.ptr<int>(), codec.counts.ptr<int>(), sums.ptr<float>(),
        codec.raw_centers.ptr<float>(), codec.sh_low.ptr<float>(), codec.sh_high.ptr<float>(),
        g.sh.ptr<float>());
    check();
}
} // namespace photara::splat::detail
