#include "common.hlsli"

// Full reduction in workgroup-sized passes. Each pass emits one partial per
// group; the host repeats it until a single value remains. Sum/mean/max/min
// stay Float32. Any/all collapse to 0/1 and the last pass writes one byte.

struct PushConstants
{
    uint count;
    uint op;
    uint src_offset;
    uint dst_offset;
    uint original_count;
    uint final_pass;
    uint src_elem;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer src;
[[vk::binding(1, 0)]] RWByteAddressBuffer dst;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

static const uint kGroupSize = 256u;
static const uint kItemsPerThread = 4u;
static const uint kSum = 0u;
static const uint kMean = 1u;
static const uint kMax = 2u;
static const uint kMin = 3u;
static const uint kAny = 5u;
static const uint kAll = 6u;

groupshared float partial[kGroupSize];

float identity_value()
{
    if (pc.op == kMax) return -3.402823466e+38f;
    if (pc.op == kMin) return 3.402823466e+38f;
    if (pc.op == kAll) return 1.0f;
    return 0.0f;
}

float combine(float left, float right)
{
    if (pc.op == kMax) return max(left, right);
    if (pc.op == kMin) return min(left, right);
    if (pc.op == kAny) return (left != 0.0f || right != 0.0f) ? 1.0f : 0.0f;
    if (pc.op == kAll) return (left != 0.0f && right != 0.0f) ? 1.0f : 0.0f;
    return left + right;
}

float load_item(uint index)
{
    const uint addr = pc.src_offset + index * pc.src_elem;
    if (pc.src_elem == 1u)
    {
        return load_u8(src, addr) != 0u ? 1.0f : 0.0f;
    }
    const float value = asfloat(src.Load(addr));
    if (pc.op == kAny || pc.op == kAll)
    {
        return value != 0.0f ? 1.0f : 0.0f;
    }
    return value;
}

[numthreads(kGroupSize, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread : SV_GroupThreadID)
{
    const uint lane = group_thread.x;
    const uint first = group_id.x * (kGroupSize * kItemsPerThread) + lane;
    float value = identity_value();
    [unroll]
    for (uint item = 0u; item < kItemsPerThread; ++item)
    {
        const uint index = first + item * kGroupSize;
        if (index < pc.count)
        {
            value = combine(value, load_item(index));
        }
    }
    partial[lane] = value;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint stride = kGroupSize / 2u; stride > 0u; stride >>= 1u)
    {
        if (lane < stride)
        {
            partial[lane] = combine(partial[lane], partial[lane + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (lane == 0u)
    {
        float result = partial[0];
        if (pc.final_pass != 0u && pc.op == kMean && pc.original_count != 0u)
        {
            result /= float(pc.original_count);
        }
        if (pc.final_pass != 0u && (pc.op == kAny || pc.op == kAll))
        {
            store_u8(dst, pc.dst_offset, result != 0.0f ? 1u : 0u);
            return;
        }
        dst.Store(pc.dst_offset + group_id.x * 4u, asuint(result));
    }
}
