#include "common.hlsli"

// Full argmax/argmin. Each group keeps the winning value and its original
// index. Later passes combine those pairs. Ties keep the smaller index, which
// matches a serial scan that only replaces on a strict comparison.

struct PushConstants
{
    uint count;
    uint argmax;
    uint src_offset;
    uint dst_offset;
    uint src_is_pair;
    uint final_pass;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer src;
[[vk::binding(1, 0)]] RWByteAddressBuffer dst;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

static const uint kGroupSize = 256u;
static const uint kItemsPerThread = 4u;
static const uint kEmpty = 0xFFFFFFFFu;

groupshared uint shared_value[kGroupSize];
groupshared uint shared_index[kGroupSize];

struct Pair
{
    float value;
    uint index;
};

Pair load_pair(uint index)
{
    Pair item;
    if (pc.src_is_pair != 0u)
    {
        const uint2 packed = src.Load2(pc.src_offset + index * 8u);
        item.value = asfloat(packed.x);
        item.index = packed.y;
        return item;
    }
    item.value = asfloat(src.Load(pc.src_offset + index * 4u));
    item.index = index;
    return item;
}

bool wins(Pair candidate, Pair current)
{
    if (current.index == kEmpty) return candidate.index != kEmpty;
    if (candidate.index == kEmpty) return false;
    if (pc.argmax != 0u)
    {
        if (candidate.value > current.value) return true;
        if (candidate.value < current.value) return false;
    }
    else
    {
        if (candidate.value < current.value) return true;
        if (candidate.value > current.value) return false;
    }
    return candidate.value == current.value && candidate.index < current.index;
}

[numthreads(kGroupSize, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread : SV_GroupThreadID)
{
    const uint lane = group_thread.x;
    const uint origin = group_id.x * (kGroupSize * kItemsPerThread);
    Pair best;
    best.value = 0.0f;
    best.index = kEmpty;
    [unroll]
    for (uint item = 0u; item < kItemsPerThread; ++item)
    {
        const uint index = origin + lane + item * kGroupSize;
        if (index < pc.count)
        {
            const Pair candidate = load_pair(index);
            if (wins(candidate, best)) best = candidate;
        }
    }
    shared_value[lane] = asuint(best.value);
    shared_index[lane] = best.index;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint stride = kGroupSize / 2u; stride > 0u; stride >>= 1u)
    {
        if (lane < stride)
        {
            Pair other;
            other.value = asfloat(shared_value[lane + stride]);
            other.index = shared_index[lane + stride];
            if (wins(other, best)) best = other;
            shared_value[lane] = asuint(best.value);
            shared_index[lane] = best.index;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (lane == 0u)
    {
        if (pc.final_pass != 0u)
        {
            dst.Store2(pc.dst_offset, uint2(best.index, 0u));
            return;
        }
        dst.Store2(pc.dst_offset + group_id.x * 8u, uint2(asuint(best.value), best.index));
    }
}
