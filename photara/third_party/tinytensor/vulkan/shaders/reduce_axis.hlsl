#include "common.hlsli"

// One reduced axis. mode 0 is one thread per output (strided inner dimension).
// mode 1 is one workgroup per contiguous row, so a long row is not a serial loop.

struct PushConstants
{
    uint mode;
    uint outer;
    uint reduce;
    uint inner;
    uint op;
    uint src_offset;
    uint dst_offset;
    uint elem_in;
    uint elem_out;
    uint dtype_in;
    uint dtype_out;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer src;
[[vk::binding(1, 0)]] RWByteAddressBuffer dst;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

static const uint kSum = 0u;
static const uint kMean = 1u;
static const uint kMax = 2u;
static const uint kMin = 3u;
static const uint kProd = 4u;
static const uint kAny = 5u;
static const uint kAll = 6u;
static const uint kArgmax = 9u;
static const uint kArgmin = 10u;
static const uint kEmpty = 0xFFFFFFFFu;

groupshared uint shared_value[TT_GROUP_SIZE];
groupshared uint shared_index[TT_GROUP_SIZE];

float load_at(uint elem)
{
    return load_as_float(src, pc.src_offset + elem * pc.elem_in, pc.dtype_in);
}

void store_at(uint elem, float value, uint index)
{
    const uint addr = pc.dst_offset + elem * pc.elem_out;
    if (pc.op == kArgmax || pc.op == kArgmin)
    {
        dst.Store2(addr, uint2(index, 0u));
        return;
    }
    store_from_float(dst, addr, pc.dtype_out, value);
}

float identity_value()
{
    if (pc.op == kMax || pc.op == kArgmax) return -3.402823466e+38f;
    if (pc.op == kMin || pc.op == kArgmin) return 3.402823466e+38f;
    if (pc.op == kProd || pc.op == kAll) return 1.0f;
    return 0.0f;
}

bool keeps_index()
{
    return pc.op == kArgmax || pc.op == kArgmin;
}

void consider(inout float acc, inout uint best, uint r, float value)
{
    if (pc.op == kSum || pc.op == kMean) acc += value;
    else if (pc.op == kProd) acc *= value;
    else if (pc.op == kAny) acc = (acc != 0.0f || value != 0.0f) ? 1.0f : 0.0f;
    else if (pc.op == kAll) acc = (acc != 0.0f && value != 0.0f) ? 1.0f : 0.0f;
    else if (pc.op == kMax || pc.op == kArgmax)
    {
        if (r == 0u || value > acc)
        {
            acc = value;
            best = r;
        }
    }
    else if (pc.op == kMin || pc.op == kArgmin)
    {
        if (r == 0u || value < acc)
        {
            acc = value;
            best = r;
        }
    }
}

[numthreads(TT_GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread : SV_GroupThreadID, uint3 dtid : SV_DispatchThreadID)
{
    if (pc.mode == 0u)
    {
        const uint outputs = pc.outer * max(pc.inner, 1u);
        const uint index = dtid.x;
        if (index >= outputs || pc.reduce == 0u) return;
        const uint inner = max(pc.inner, 1u);
        const uint outer_index = index / inner;
        const uint inner_index = index % inner;
        float acc = identity_value();
        uint best = 0u;
        for (uint r = 0u; r < pc.reduce; ++r)
        {
            const uint elem = (outer_index * pc.reduce + r) * inner + inner_index;
            consider(acc, best, r, load_at(elem));
        }
        if (pc.op == kMean && pc.reduce != 0u) acc /= float(pc.reduce);
        store_at(index, acc, best);
        return;
    }

    const uint row = group_id.x;
    if (row >= pc.outer || pc.reduce == 0u) return;
    const uint lane = group_thread.x;
    float acc = identity_value();
    uint best = kEmpty;
    for (uint r = lane; r < pc.reduce; r += TT_GROUP_SIZE)
    {
        const float value = load_at(row * pc.reduce + r);
        if (keeps_index())
        {
            const bool seed = best == kEmpty;
            const bool better = pc.op == kArgmax
                                    ? value > acc || (value == acc && r < best)
                                    : value < acc || (value == acc && r < best);
            if (seed || better)
            {
                acc = value;
                best = r;
            }
        }
        else
        {
            consider(acc, best, r == lane ? 0u : 1u, value);
        }
    }
    if (!keeps_index() && best == kEmpty && lane >= pc.reduce)
    {
        acc = identity_value();
    }
    shared_value[lane] = asuint(acc);
    shared_index[lane] = keeps_index() ? best : 0u;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint stride = TT_GROUP_SIZE / 2u; stride > 0u; stride >>= 1u)
    {
        if (lane < stride)
        {
            const float other_value = asfloat(shared_value[lane + stride]);
            const uint other_index = shared_index[lane + stride];
            if (keeps_index())
            {
                const bool seed = best == kEmpty && other_index != kEmpty;
                const bool better = pc.op == kArgmax
                                        ? other_value > acc || (other_value == acc && other_index < best)
                                        : other_value < acc || (other_value == acc && other_index < best);
                if (seed || (other_index != kEmpty && better))
                {
                    acc = other_value;
                    best = other_index;
                }
            }
            else
            {
                consider(acc, best, 1u, other_value);
            }
            shared_value[lane] = asuint(acc);
            shared_index[lane] = best;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (lane == 0u)
    {
        if (pc.op == kMean && pc.reduce != 0u) acc /= float(pc.reduce);
        if (pc.op == kArgmax || pc.op == kArgmin) best = best == kEmpty ? 0u : best;
        store_at(row, acc, best);
    }
}
