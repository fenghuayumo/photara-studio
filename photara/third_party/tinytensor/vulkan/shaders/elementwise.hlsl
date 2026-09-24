#include "common.hlsli"

struct PushConstants
{
    uint op;
    uint count;
    uint dtype_a;
    uint dtype_b;
    uint dtype_c;
    uint dtype_out;
    uint off_a;
    uint off_b;
    uint off_c;
    uint off_out;
    uint elem_size_a;
    uint elem_size_b;
    uint elem_size_c;
    uint elem_size_out;
    uint has_b;
    uint has_c;
    uint scalar_bits;
    uint scalar2_bits;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer buffer_a;
[[vk::binding(1, 0)]] RWByteAddressBuffer buffer_b;
[[vk::binding(2, 0)]] RWByteAddressBuffer buffer_out;
[[vk::binding(3, 0)]] RWByteAddressBuffer buffer_c;
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
static const uint kOpNeg = 12;
static const uint kOpAbs = 13;
static const uint kOpSign = 14;
static const uint kOpReciprocal = 15;
static const uint kOpExp = 16;
static const uint kOpExp2 = 17;
static const uint kOpLog = 18;
static const uint kOpLog2 = 19;
static const uint kOpLog10 = 20;
static const uint kOpLog1p = 21;
static const uint kOpSqrt = 22;
static const uint kOpRsqrt = 23;
static const uint kOpSquare = 24;
static const uint kOpSin = 25;
static const uint kOpCos = 26;
static const uint kOpTan = 27;
static const uint kOpAsin = 28;
static const uint kOpAcos = 29;
static const uint kOpAtan = 30;
static const uint kOpSinh = 31;
static const uint kOpCosh = 32;
static const uint kOpTanh = 33;
static const uint kOpSigmoid = 34;
static const uint kOpRelu = 35;
static const uint kOpGelu = 36;
static const uint kOpSwish = 37;
static const uint kOpFloor = 38;
static const uint kOpCeil = 39;
static const uint kOpRound = 40;
static const uint kOpTrunc = 41;
static const uint kOpIsNan = 42;
static const uint kOpIsInf = 43;
static const uint kOpIsFinite = 44;
static const uint kOpAdd = 45;
static const uint kOpSub = 46;
static const uint kOpMul = 47;
static const uint kOpDiv = 48;
static const uint kOpPow = 49;
static const uint kOpMod = 50;
static const uint kOpMaximum = 51;
static const uint kOpMinimum = 52;
static const uint kOpClamp = 53;
static const uint kOpWhere = 54;
static const uint kOpMaskFill = 55;

bool to_bool(float value)
{
    return value != 0.0f;
}

bool is_compare(uint op)
{
    return op >= kOpEq && op <= kOpGe;
}

float apply_unary(uint op, float a, float lo, float hi)
{
    if (op == kOpNeg) return -a;
    if (op == kOpAbs) return abs(a);
    if (op == kOpSign) return float((a > 0.0f) - (a < 0.0f));
    if (op == kOpReciprocal) return 1.0f / (a + 1e-8f);
    if (op == kOpExp) return exp(a);
    if (op == kOpExp2) return exp2(a);
    if (op == kOpLog) return log(max(a, 1e-10f));
    if (op == kOpLog2) return log2(max(a, 1e-10f));
    if (op == kOpLog10) return log10(max(a, 1e-10f));
    if (op == kOpLog1p) return log(1.0f + a);
    if (op == kOpSqrt) return sqrt(max(a, 0.0f));
    if (op == kOpRsqrt) return rsqrt(max(a, 1e-10f));
    if (op == kOpSquare) return a * a;
    if (op == kOpSin) return sin(a);
    if (op == kOpCos) return cos(a);
    if (op == kOpTan) return tan(a);
    if (op == kOpAsin) return asin(clamp(a, -1.0f, 1.0f));
    if (op == kOpAcos) return acos(clamp(a, -1.0f, 1.0f));
    if (op == kOpAtan) return atan(a);
    if (op == kOpSinh) return sinh(a);
    if (op == kOpCosh) return cosh(a);
    if (op == kOpTanh) return tanh(a);
    if (op == kOpSigmoid) return 1.0f / (1.0f + exp(-a));
    if (op == kOpRelu) return max(a, 0.0f);
    if (op == kOpGelu)
    {
        const float inner = 0.7978845608f * (a + 0.044715f * a * a * a);
        return 0.5f * a * (1.0f + tanh(inner));
    }
    if (op == kOpSwish) return a / (1.0f + exp(-a));
    if (op == kOpFloor) return floor(a);
    if (op == kOpCeil) return ceil(a);
    if (op == kOpRound) return round(a);
    if (op == kOpTrunc) return trunc(a);
    if (op == kOpClamp) return clamp(a, lo, hi);
    return a;
}

float apply_binary(uint op, float a, float b)
{
    if (op == kOpAdd) return a + b;
    if (op == kOpSub) return a - b;
    if (op == kOpMul) return a * b;
    if (op == kOpDiv) return a / b;
    if (op == kOpPow)
    {
        if (b == 2.0f) return a * a;
        return pow(a, b);
    }
    if (op == kOpMod) return fmod(a, b);
    if (op == kOpMaximum) return max(a, b);
    if (op == kOpMinimum) return min(a, b);
    return a;
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
    const float scalar = asfloat(pc.scalar_bits);
    const float scalar2 = asfloat(pc.scalar2_bits);
    if (pc.op == kOpFill)
    {
        store_from_float(buffer_out, out_off, pc.dtype_out, scalar);
        return;
    }

    const uint a_off = pc.off_a + index * pc.elem_size_a;
    if (pc.op == kOpWhere)
    {
        const float cond = load_as_float(buffer_a, a_off, pc.dtype_a);
        if (to_bool(cond))
        {
            const uint src_off = pc.off_b + index * pc.elem_size_b;
            copy_elem(buffer_out, out_off, buffer_b, src_off, pc.elem_size_out);
        }
        else
        {
            const uint src_off = pc.off_c + index * pc.elem_size_c;
            copy_elem(buffer_out, out_off, buffer_c, src_off, pc.elem_size_out);
        }
        return;
    }
    if (pc.op == kOpMaskFill)
    {
        const uint b_off = pc.off_b + index * pc.elem_size_b;
        if (load_as_float(buffer_b, b_off, pc.dtype_b) != 0.0f)
        {
            store_from_float(buffer_out, out_off, pc.dtype_out, scalar);
        }
        else
        {
            copy_elem(buffer_out, out_off, buffer_a, a_off, pc.elem_size_out);
        }
        return;
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
        if (pc.dtype_a == kDtypeFloat32 && pc.dtype_out == kDtypeInt64)
        {
            store_i64_from_i32(buffer_out, out_off, int(load_f32(buffer_a, a_off)));
            return;
        }
        store_from_float(buffer_out, out_off, pc.dtype_out, load_as_float(buffer_a, a_off, pc.dtype_a));
        return;
    }

    const float a = load_as_float(buffer_a, a_off, pc.dtype_a);
    float b = scalar;
    if (pc.has_b != 0)
    {
        b = load_as_float(buffer_b, pc.off_b + index * pc.elem_size_b, pc.dtype_b);
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
    if (pc.op == kOpIsNan)
    {
        store_u8(buffer_out, out_off, isnan(a) ? 1u : 0u);
        return;
    }
    if (pc.op == kOpIsInf)
    {
        store_u8(buffer_out, out_off, isinf(a) ? 1u : 0u);
        return;
    }
    if (pc.op == kOpIsFinite)
    {
        store_u8(buffer_out, out_off, (!isnan(a) && !isinf(a)) ? 1u : 0u);
        return;
    }

    if (is_compare(pc.op) && pc.dtype_a == kDtypeInt64 && pc.has_b != 0 && pc.dtype_b == kDtypeInt64)
    {
        const int order = cmp_i64(
            load_u64(buffer_a, a_off),
            load_u64(buffer_b, pc.off_b + index * pc.elem_size_b));
        bool cmp = false;
        if (pc.op == kOpEq) cmp = order == 0;
        else if (pc.op == kOpNe) cmp = order != 0;
        else if (pc.op == kOpLt) cmp = order < 0;
        else if (pc.op == kOpLe) cmp = order <= 0;
        else if (pc.op == kOpGt) cmp = order > 0;
        else cmp = order >= 0;
        store_u8(buffer_out, out_off, cmp ? 1u : 0u);
        return;
    }

    if (is_compare(pc.op))
    {
        bool cmp = false;
        if (pc.op == kOpEq) cmp = a == b;
        else if (pc.op == kOpNe) cmp = a != b;
        else if (pc.op == kOpLt) cmp = a < b;
        else if (pc.op == kOpLe) cmp = a <= b;
        else if (pc.op == kOpGt) cmp = a > b;
        else cmp = a >= b;
        store_u8(buffer_out, out_off, cmp ? 1u : 0u);
        return;
    }

    const bool arithmetic_binary = pc.op >= kOpAdd && pc.op <= kOpMinimum;
    const float value = arithmetic_binary
        ? apply_binary(pc.op, a, b)
        : apply_unary(pc.op, a, scalar, scalar2);
    store_from_float(buffer_out, out_off, pc.dtype_out, value);
}
