#include "common.hlsli"

struct PushConstants
{
    uint op;
    uint count;
    uint seed;
    uint dst_offset;
    uint src_offset;
    uint dtype;
    uint elem_size;
    uint rows;
    uint cols;
    uint p0_bits;
    uint p1_bits;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer dst;
[[vk::binding(1, 0)]] RWByteAddressBuffer src;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

static const uint kOpUniform = 0;
static const uint kOpNormal = 1;
static const uint kOpBernoulli = 2;
static const uint kOpRandint = 3;
static const uint kOpArange = 4;
static const uint kOpEye = 5;
static const uint kOpDiag = 6;

uint pcg_hash(uint value)
{
    uint state = value * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rand01(uint idx)
{
    const uint hashed = pcg_hash(pc.seed ^ pcg_hash(idx + 1u));
    return float(hashed >> 8u) * (1.0f / 16777216.0f);
}

[numthreads(TT_GROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint index = dtid.x;
    if (index >= pc.count)
    {
        return;
    }
    const uint addr = pc.dst_offset + index * pc.elem_size;
    const float p0 = asfloat(pc.p0_bits);
    const float p1 = asfloat(pc.p1_bits);

    if (pc.op == kOpUniform)
    {
        const float u = rand01(index);
        store_from_float(dst, addr, pc.dtype, p0 + (p1 - p0) * u);
        return;
    }
    if (pc.op == kOpNormal)
    {
        const float u1 = max(rand01(index * 2u), 1e-7f);
        const float u2 = rand01(index * 2u + 1u);
        const float z = sqrt(-2.0f * log(u1)) * cos(6.28318530718f * u2);
        store_from_float(dst, addr, pc.dtype, p0 + p1 * z);
        return;
    }
    if (pc.op == kOpBernoulli)
    {
        store_from_float(dst, addr, pc.dtype, rand01(index) < p0 ? 1.0f : 0.0f);
        return;
    }
    if (pc.op == kOpRandint)
    {
        const int low = int(p0);
        const int high = int(p1);
        const int span = max(high - low, 1);
        const int value = low + int(rand01(index) * float(span));
        store_from_float(dst, addr, pc.dtype, float(min(value, high - 1)));
        return;
    }
    if (pc.op == kOpArange)
    {
        store_from_float(dst, addr, pc.dtype, p0 + float(index) * p1);
        return;
    }
    if (pc.op == kOpEye)
    {
        const uint row = index / max(pc.cols, 1u);
        const uint col = index % max(pc.cols, 1u);
        store_from_float(dst, addr, pc.dtype, row == col ? 1.0f : 0.0f);
        return;
    }
    const uint n = pc.rows;
    const uint row = index / max(n, 1u);
    const uint col = index % max(n, 1u);
    float value = 0.0f;
    if (row == col && row < n)
    {
        value = load_as_float(src, pc.src_offset + row * 4u, kDtypeFloat32);
    }
    store_from_float(dst, addr, pc.dtype, value);
}
