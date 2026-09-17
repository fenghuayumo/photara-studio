// Minimal row-major 3x3 matrix used across all device kernels.
//
// Formulas translated from column-major literature use at(col, row) so the
// indexing reads exactly like the GLM-style [col][row] expressions they
// come from; native code uses [row][col].
#pragma once

#include <cmath>
#include <cuda_runtime.h>

#if defined(__CUDACC__)
#define SD_D2 __device__
#else
#define SD_D2 inline
#endif

namespace splat_drender::mat {

struct Mat3 {
    float m[3][3]{};

    SD_D2 float* operator[](int r) { return m[r]; }
    SD_D2 const float* operator[](int r) const { return m[r]; }

    // glm-compatible accessor: at(column, row).
    SD_D2 float at(int c, int r) const { return m[r][c]; }

    static SD_D2 Mat3 diag(float x, float y, float z) {
        Mat3 o;
        o.m[0][0] = x; o.m[1][1] = y; o.m[2][2] = z;
        return o;
    }
};

SD_D2 inline Mat3 operator*(const Mat3& a, const Mat3& b) {
    Mat3 o;
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j)
            o.m[i][j] = a.m[i][0] * b.m[0][j] + a.m[i][1] * b.m[1][j] +
                        a.m[i][2] * b.m[2][j];
    return o;
}

SD_D2 inline Mat3 operator+(const Mat3& a, const Mat3& b) {
    Mat3 o;
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j) o.m[i][j] = a.m[i][j] + b.m[i][j];
    return o;
}

SD_D2 inline float3 operator*(const Mat3& a, float3 v) {
    return make_float3(
        a.m[0][0] * v.x + a.m[0][1] * v.y + a.m[0][2] * v.z,
        a.m[1][0] * v.x + a.m[1][1] * v.y + a.m[1][2] * v.z,
        a.m[2][0] * v.x + a.m[2][1] * v.y + a.m[2][2] * v.z);
}

SD_D2 inline Mat3 transpose(const Mat3& a) {
    Mat3 o;
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j)
            o.m[i][j] = a.m[j][i];
    return o;
}

SD_D2 inline Mat3 outer(float3 u, float3 v) {
    Mat3 o;
    o.m[0][0] = u.x * v.x; o.m[0][1] = u.x * v.y; o.m[0][2] = u.x * v.z;
    o.m[1][0] = u.y * v.x; o.m[1][1] = u.y * v.y; o.m[1][2] = u.y * v.z;
    o.m[2][0] = u.z * v.x; o.m[2][1] = u.z * v.y; o.m[2][2] = u.z * v.z;
    return o;
}

SD_D2 inline float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// World-to-camera transform stored as 16 contiguous floats (row-major
// 3x4); matches the layout produced by aetherscan's Camera.
SD_D2 inline float3 xform_point(const float3 p, const float* v) {
    return make_float3(
        v[0] * p.x + v[4] * p.y + v[8] * p.z + v[12],
        v[1] * p.x + v[5] * p.y + v[9] * p.z + v[13],
        v[2] * p.x + v[6] * p.y + v[10] * p.z + v[14]);
}

// Transpose (world) direction transform of the same matrix.
SD_D2 inline float3 xform_dir_transpose(float3 d, const float* v) {
    return make_float3(
        v[0] * d.x + v[1] * d.y + v[2] * d.z,
        v[4] * d.x + v[5] * d.y + v[6] * d.z,
        v[8] * d.x + v[9] * d.y + v[10] * d.z);
}

// Rotation part of world_to_camera, transposed (camera-from-world axes in
// world columns / world-from-camera rotation in rows).
SD_D2 inline Mat3 world_rotation_transposed(const float* v) {
    Mat3 w;
    w.m[0][0] = v[0]; w.m[0][1] = v[1]; w.m[0][2] = v[2];
    w.m[1][0] = v[4]; w.m[1][1] = v[5]; w.m[1][2] = v[6];
    w.m[2][0] = v[8]; w.m[2][1] = v[9]; w.m[2][2] = v[10];
    return w;
}

// Scalar-first quaternion to rotation matrix (rows).
SD_D2 inline Mat3 quat_rotation(float4 q) {
    const float r = q.x, x = q.y, y = q.z, z = q.w;
    Mat3 o;
    o.m[0][0] = 1.f - 2.f * (y * y + z * z);
    o.m[0][1] = 2.f * (x * y + r * z);
    o.m[0][2] = 2.f * (x * z - r * y);
    o.m[1][0] = 2.f * (x * y - r * z);
    o.m[1][1] = 1.f - 2.f * (x * x + z * z);
    o.m[1][2] = 2.f * (y * z + r * x);
    o.m[2][0] = 2.f * (x * z + r * y);
    o.m[2][1] = 2.f * (y * z - r * x);
    o.m[2][2] = 1.f - 2.f * (x * x + y * y);
    return o;
}

// Jacobian of the gradient of ||v|| w.r.t. v, applied to dv.
SD_D2 inline float3 dnormalize(float3 v, float3 dv) {
    const float inv = 1.f / sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    const float3 n = make_float3(v.x * inv, v.y * inv, v.z * inv);
    return make_float3(
        ((1.f - n.x * n.x) * dv.x - n.y * n.x * dv.y - n.z * n.x * dv.z) * inv,
        (-n.x * n.y * dv.x + (1.f - n.y * n.y) * dv.y - n.z * n.y * dv.z) * inv,
        (-n.x * n.z * dv.x - n.y * n.z * dv.y + (1.f - n.z * n.z) * dv.z) * inv);
}

#if defined(__CUDACC__)
// Device-only helpers (warp intrinsics / block sync): warp-level sums used
// before atomicAdd and the block-wide max for contributor counting.
SD_D2 inline void warp_sum(float& v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
}

SD_D2 inline void warp_sum(float2& v) {
    warp_sum(v.x);
    warp_sum(v.y);
}

SD_D2 inline void warp_sum(float3& v) {
    warp_sum(v.x);
    warp_sum(v.y);
    warp_sum(v.z);
}

SD_D2 inline void warp_sum(float4& v) {
    warp_sum(v.x);
    warp_sum(v.y);
    warp_sum(v.z);
    warp_sum(v.w);
}

template <int N>
SD_D2 inline void warp_sum(float (&v)[N]) {
#pragma unroll
    for (int i = 0; i < N; ++i) warp_sum(v[i]);
}

SD_D2 inline unsigned block_max(unsigned v, unsigned* warp_scratch) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v = max(v, __shfl_xor_sync(0xffffffffu, v, offset));
    const int rank = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const int lane = rank & 31;
    const int warp = rank >> 5;
    const int num_warps = int(blockDim.x * blockDim.y * blockDim.z + 31) >> 5;
    if (lane == 0) warp_scratch[warp] = v;
    __syncthreads();
    unsigned result = warp_scratch[0];
    for (int w = 1; w < num_warps; ++w) result = max(result, warp_scratch[w]);
    __syncthreads();  // scratch may be reused by the next point batch
    return result;
}

// Float sibling of block_max, for block-uniform convergence tests. Must be
// called by every thread in the block, like block_max itself.
SD_D2 inline float block_max(float v, float* warp_scratch) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    const int rank = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const int lane = rank & 31;
    const int warp = rank >> 5;
    const int num_warps = int(blockDim.x * blockDim.y * blockDim.z + 31) >> 5;
    if (lane == 0) warp_scratch[warp] = v;
    __syncthreads();
    float result = warp_scratch[0];
    for (int w = 1; w < num_warps; ++w) result = fmaxf(result, warp_scratch[w]);
    __syncthreads();  // scratch may be reused by the next point batch
    return result;
}
#endif  // __CUDACC__

}  // namespace splat_drender::mat

// Vector arithmetic for CUDA built-in types, at global scope so ADL finds
// them for float2/float3/float4 arguments anywhere.
SD_D2 inline float2 operator+(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}
SD_D2 inline float2 operator-(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}
SD_D2 inline float2 operator*(float2 a, float s) {
    return make_float2(a.x * s, a.y * s);
}
SD_D2 inline float2 operator/(float2 a, float s) {
    return make_float2(a.x / s, a.y / s);
}
SD_D2 inline float2& operator+=(float2& a, float2 b) {
    a.x += b.x; a.y += b.y; return a;
}
SD_D2 inline float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
SD_D2 inline float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
SD_D2 inline float3 operator*(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
SD_D2 inline float3 operator*(float s, float3 a) { return a * s; }
SD_D2 inline float3 operator/(float3 a, float s) {
    return make_float3(a.x / s, a.y / s, a.z / s);
}
SD_D2 inline float3& operator+=(float3& a, float3 b) {
    a.x += b.x; a.y += b.y; a.z += b.z; return a;
}
SD_D2 inline float4 operator+(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
SD_D2 inline float4 operator*(float4 a, float s) {
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
SD_D2 inline float4 operator/(float4 a, float s) {
    return make_float4(a.x / s, a.y / s, a.z / s, a.w / s);
}
