// Fast contiguous Float32 full reduction.  The generic reduce shader maps one
// output to one thread, which is appropriate for short reduced dimensions but
// makes a scalar sum/mean serial.  This pass emits one partial per workgroup;
// the host records the same pass recursively until one value remains.

struct PushConstants
{
    uint count;
    uint op;
    uint src_offset;
    uint dst_offset;
    uint original_count;
    uint final_pass;
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

groupshared float partial[kGroupSize];

float identity_value()
{
    if (pc.op == kMax) return -3.402823466e+38f;
    if (pc.op == kMin) return 3.402823466e+38f;
    return 0.0f;
}

float combine(float left, float right)
{
    if (pc.op == kMax) return max(left, right);
    if (pc.op == kMin) return min(left, right);
    return left + right;
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
            value = combine(value, asfloat(src.Load(pc.src_offset + index * 4u)));
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
        dst.Store(pc.dst_offset + group_id.x * 4u, asuint(result));
    }
}
