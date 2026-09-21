#include "common.hlsli"

struct PushConstants
{
    uint op;
    uint count;
    uint dtype_a;
    uint dtype_b;
    uint dtype_out;
    uint off_a;
    uint off_b;
    uint off_out;
    uint elem_size_a;
    uint elem_size_b;
    uint elem_size_out;
    uint has_b;
    uint scalar_bits;
    uint _pad;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer buffer_a;
[[vk::binding(1, 0)]] RWByteAddressBuffer buffer_b;
[[vk::binding(2, 0)]] RWByteAddressBuffer buffer_out;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

static const uint kOpFill = 0;
static const uint kOpConvert = 1;
static const uint kOpNot = 2;
static const uint kOpAnd = 3;
static const uint kOpOr = 4;
static const uint kOpXor = 5;
static const uint kOpEq = 6;
static const uint kOpNe = 7;
static const uint kOpLt = 8;
static const uint kOpLe = 9;
static const uint kOpGt = 10;
static const uint kOpGe = 11;

bool to_bool(float value)
{
    return value != 0.0f;
}

[numthreads(TT_GROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint index = dtid.x;
    if (index >= pc.count)
    {
        return;
    }

    const uint out_off = pc.off_out + index * pc.elem_size_out;
    if (pc.op == kOpFill)
    {
        store_from_float(buffer_out, out_off, pc.dtype_out, asfloat(pc.scalar_bits));
        return;
    }

    const uint a_off = pc.off_a + index * pc.elem_size_a;
    const float a = load_as_float(buffer_a, a_off, pc.dtype_a);
    float b = asfloat(pc.scalar_bits);
    if (pc.has_b != 0)
    {
        const uint b_off = pc.off_b + index * pc.elem_size_b;
        b = load_as_float(buffer_b, b_off, pc.dtype_b);
    }

    if (pc.op == kOpConvert)
    {
        if (pc.dtype_a == kDtypeInt64 && pc.dtype_out == kDtypeInt32)
        {
            store_i32(buffer_out, out_off, i64_as_i32(load_u64(buffer_a, a_off)));
            return;
        }
        if (pc.dtype_a == kDtypeInt32 && pc.dtype_out == kDtypeInt64)
        {
            store_i64_from_i32(buffer_out, out_off, load_i32(buffer_a, a_off));
            return;
        }
        store_from_float(buffer_out, out_off, pc.dtype_out, a);
        return;
    }

    if (pc.op == kOpNot)
    {
        store_u8(buffer_out, out_off, to_bool(a) ? 0u : 1u);
        return;
    }
    if (pc.op == kOpAnd)
    {
        store_u8(buffer_out, out_off, (to_bool(a) && to_bool(b)) ? 1u : 0u);
        return;
    }
    if (pc.op == kOpOr)
    {
        store_u8(buffer_out, out_off, (to_bool(a) || to_bool(b)) ? 1u : 0u);
        return;
    }
    if (pc.op == kOpXor)
    {
        store_u8(buffer_out, out_off, (to_bool(a) != to_bool(b)) ? 1u : 0u);
        return;
    }

    bool cmp = false;
    if (pc.op == kOpEq)
    {
        cmp = a == b;
    }
    else if (pc.op == kOpNe)
    {
        cmp = a != b;
    }
    else if (pc.op == kOpLt)
    {
        cmp = a < b;
    }
    else if (pc.op == kOpLe)
    {
        cmp = a <= b;
    }
    else if (pc.op == kOpGt)
    {
        cmp = a > b;
    }
    else if (pc.op == kOpGe)
    {
        cmp = a >= b;
    }
    store_u8(buffer_out, out_off, cmp ? 1u : 0u);
}
