#include "common.hlsli"

struct PushConstants
{
    uint out_count;
    uint rank;
    uint reduce_count;
    uint op;
    uint dtype_in;
    uint dtype_out;
    uint src_offset;
    uint dst_offset;
    uint elem_in;
    uint elem_out;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer src;
[[vk::binding(1, 0)]] RWByteAddressBuffer dst;
[[vk::binding(2, 0)]] RWByteAddressBuffer meta;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

static const uint kSum = 0;
static const uint kMean = 1;
static const uint kMax = 2;
static const uint kMin = 3;
static const uint kProd = 4;
static const uint kAny = 5;
static const uint kAll = 6;
static const uint kArgmax = 9;
static const uint kArgmin = 10;

uint meta_u(uint index)
{
    return meta.Load(index * 4u);
}

[numthreads(TT_GROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint out_index = dtid.x;
    if (out_index >= pc.out_count || pc.rank > 8u)
    {
        return;
    }

    uint out_coord[8];
    {
        uint remaining = out_index;
        [unroll]
        for (int axis = 7; axis >= 0; --axis)
        {
            if (uint(axis) < pc.rank)
            {
                const uint dim = max(meta_u(16u + uint(axis)), 1u);
                out_coord[axis] = remaining % dim;
                remaining /= dim;
            }
            else
            {
                out_coord[axis] = 0u;
            }
        }
    }

    float acc = 0.0f;
    if (pc.op == kProd || pc.op == kAll) acc = 1.0f;
    if (pc.op == kMax || pc.op == kArgmax) acc = -3.4e38f;
    if (pc.op == kMin || pc.op == kArgmin) acc = 3.4e38f;
    uint best_index = 0u;

    for (uint r = 0u; r < max(pc.reduce_count, 1u); ++r)
    {
        uint red_remaining = r;
        uint in_coord[8];
        [unroll]
        for (int axis = 7; axis >= 0; --axis)
        {
            in_coord[axis] = 0u;
            if (uint(axis) >= pc.rank)
            {
                continue;
            }
            if (meta_u(8u + uint(axis)) != 0u)
            {
                const uint dim = max(meta_u(uint(axis)), 1u);
                in_coord[axis] = red_remaining % dim;
                red_remaining /= dim;
            }
            else
            {
                in_coord[axis] = out_coord[axis];
            }
        }

        uint linear_index = 0u;
        uint stride = 1u;
        [unroll]
        for (int axis2 = 7; axis2 >= 0; --axis2)
        {
            if (uint(axis2) < pc.rank)
            {
                linear_index += in_coord[axis2] * stride;
                stride *= max(meta_u(uint(axis2)), 1u);
            }
        }
        // The loop above accumulates stride from axis 7 down, which is wrong
        // for ranks below 8 because higher axes still multiply when skipped.
        // Recompute row-major from axis 0 using the coordinates already stored.
        linear_index = 0u;
        [unroll]
        for (int axis3 = 0; axis3 < 8; ++axis3)
        {
            if (uint(axis3) < pc.rank)
            {
                uint axis_stride = 1u;
                [unroll]
                for (int inner = axis3 + 1; inner < 8; ++inner)
                {
                    if (uint(inner) < pc.rank)
                    {
                        axis_stride *= max(meta_u(uint(inner)), 1u);
                    }
                }
                linear_index += in_coord[axis3] * axis_stride;
            }
        }

        const float value = load_as_float(
            src, pc.src_offset + linear_index * pc.elem_in, pc.dtype_in);
        if (pc.op == kSum || pc.op == kMean) acc += value;
        else if (pc.op == kProd) acc *= value;
        else if (pc.op == kAny) acc = (acc != 0.0f || value != 0.0f) ? 1.0f : 0.0f;
        else if (pc.op == kAll) acc = (acc != 0.0f && value != 0.0f) ? 1.0f : 0.0f;
        else if ((pc.op == kMax || pc.op == kArgmax) && (r == 0u || value > acc))
        {
            acc = value;
            best_index = r;
        }
        else if ((pc.op == kMin || pc.op == kArgmin) && (r == 0u || value < acc))
        {
            acc = value;
            best_index = r;
        }
    }

    if (pc.op == kMean && pc.reduce_count > 0u)
    {
        acc /= float(pc.reduce_count);
    }
    const uint dst_addr = pc.dst_offset + out_index * pc.elem_out;
    if (pc.op == kArgmax || pc.op == kArgmin)
    {
        store_i64_from_i32(dst, dst_addr, int(best_index));
        return;
    }
    store_from_float(dst, dst_addr, pc.dtype_out, acc);
}
