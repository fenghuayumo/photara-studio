#ifndef TINYTENSOR_VULKAN_COMMON_HLSLI
#define TINYTENSOR_VULKAN_COMMON_HLSLI

#define TT_GROUP_SIZE 256

uint aligned4(uint byte_offset)
{
    return byte_offset & ~3u;
}

uint byte_shift(uint byte_offset)
{
    return (byte_offset & 3u) * 8u;
}

uint load_u8(RWByteAddressBuffer buf, uint byte_offset)
{
    const uint word = buf.Load(aligned4(byte_offset));
    return (word >> byte_shift(byte_offset)) & 0xFFu;
}

void store_u8(RWByteAddressBuffer buf, uint byte_offset, uint value)
{
    const uint shift = byte_shift(byte_offset);
    const uint mask = 0xFFu << shift;
    const uint aligned = aligned4(byte_offset);
    buf.InterlockedAnd(aligned, ~mask);
    buf.InterlockedOr(aligned, (value & 0xFFu) << shift);
}

uint load_u16(RWByteAddressBuffer buf, uint byte_offset)
{
    const uint word = buf.Load(aligned4(byte_offset));
    return (word >> byte_shift(byte_offset)) & 0xFFFFu;
}

void store_u16(RWByteAddressBuffer buf, uint byte_offset, uint value)
{
    const uint shift = byte_shift(byte_offset);
    const uint mask = 0xFFFFu << shift;
    const uint aligned = aligned4(byte_offset);
    buf.InterlockedAnd(aligned, ~mask);
    buf.InterlockedOr(aligned, (value & 0xFFFFu) << shift);
}

uint load_u32(RWByteAddressBuffer buf, uint byte_offset)
{
    return buf.Load(byte_offset);
}

void store_u32(RWByteAddressBuffer buf, uint byte_offset, uint value)
{
    buf.Store(byte_offset, value);
}

float load_f32(RWByteAddressBuffer buf, uint byte_offset)
{
    return asfloat(buf.Load(byte_offset));
}

void store_f32(RWByteAddressBuffer buf, uint byte_offset, float value)
{
    buf.Store(byte_offset, asuint(value));
}

int load_i32(RWByteAddressBuffer buf, uint byte_offset)
{
    return asint(buf.Load(byte_offset));
}

void store_i32(RWByteAddressBuffer buf, uint byte_offset, int value)
{
    buf.Store(byte_offset, asuint(value));
}

uint2 load_u64(RWByteAddressBuffer buf, uint byte_offset)
{
    return buf.Load2(byte_offset);
}

void store_u64(RWByteAddressBuffer buf, uint byte_offset, uint2 value)
{
    buf.Store2(byte_offset, value);
}

void store_i64_from_i32(RWByteAddressBuffer buf, uint byte_offset, int value)
{
    const uint lo = asuint(value);
    const uint hi = value < 0 ? 0xFFFFFFFFu : 0u;
    buf.Store2(byte_offset, uint2(lo, hi));
}

int i64_as_i32(uint2 words)
{
    return asint(words.x);
}

// DataType enum matches tinytensor::DataType.
static const uint kDtypeBool = 0;
static const uint kDtypeUInt8 = 1;
static const uint kDtypeInt32 = 2;
static const uint kDtypeInt64 = 3;
static const uint kDtypeFloat16 = 4;
static const uint kDtypeFloat32 = 5;

uint dtype_size(uint dtype)
{
    if (dtype == kDtypeBool || dtype == kDtypeUInt8)
    {
        return 1;
    }
    if (dtype == kDtypeFloat16)
    {
        return 2;
    }
    if (dtype == kDtypeInt64)
    {
        return 8;
    }
    return 4;
}

float unpack_f16(uint bits)
{
    const uint sign = (bits >> 15) & 1u;
    const uint exponent = (bits >> 10) & 31u;
    const uint fraction = bits & 1023u;
    if (exponent == 0u)
    {
        if (fraction == 0u)
        {
            return sign != 0u ? -0.0f : 0.0f;
        }
        const float value = float(fraction) * exp2(-24.0f);
        return sign != 0u ? -value : value;
    }
    if (exponent == 31u)
    {
        if (fraction == 0u)
        {
            return asfloat(sign != 0u ? 0xFF800000u : 0x7F800000u);
        }
        return asfloat(0x7FC00000u);
    }
    const float value = (1.0f + float(fraction) * (1.0f / 1024.0f)) * exp2(float(int(exponent) - 15));
    return sign != 0u ? -value : value;
}

uint pack_f16(float value)
{
    const uint bits = asuint(value);
    const uint sign = (bits >> 16) & 0x8000u;
    const uint raw_exp = (bits >> 23) & 0xFFu;
    uint mantissa = bits & 0x7FFFFFu;
    if (raw_exp == 0xFFu)
    {
        return sign | (mantissa == 0u ? 0x7C00u : 0x7E00u);
    }
    int exponent = int(raw_exp) - 127 + 15;
    if (exponent <= 0)
    {
        if (exponent < -10)
        {
            return sign;
        }
        mantissa |= 0x800000u;
        const uint shift = uint(1 - exponent);
        return sign | (mantissa >> (shift + 13));
    }
    if (exponent >= 31)
    {
        return sign | 0x7C00u;
    }
    uint rounded = mantissa + 0x1000u;
    if ((rounded & 0x800000u) != 0u)
    {
        rounded = 0u;
        exponent += 1;
        if (exponent >= 31)
        {
            return sign | 0x7C00u;
        }
    }
    return sign | (uint(exponent) << 10) | (rounded >> 13);
}

float load_f16(RWByteAddressBuffer buf, uint byte_offset)
{
    return unpack_f16(load_u16(buf, byte_offset));
}

void store_f16(RWByteAddressBuffer buf, uint byte_offset, float value)
{
    store_u16(buf, byte_offset, pack_f16(value));
}

int cmp_i64(uint2 lhs, uint2 rhs)
{
    const int hi_lhs = asint(lhs.y);
    const int hi_rhs = asint(rhs.y);
    if (hi_lhs != hi_rhs)
    {
        return hi_lhs < hi_rhs ? -1 : 1;
    }
    if (lhs.x != rhs.x)
    {
        return lhs.x < rhs.x ? -1 : 1;
    }
    return 0;
}

float load_as_float(RWByteAddressBuffer buf, uint byte_offset, uint dtype)
{
    if (dtype == kDtypeBool || dtype == kDtypeUInt8)
    {
        return load_u8(buf, byte_offset) != 0u ? 1.0f : 0.0f;
    }
    if (dtype == kDtypeInt32)
    {
        return float(load_i32(buf, byte_offset));
    }
    if (dtype == kDtypeInt64)
    {
        const uint2 words = load_u64(buf, byte_offset);
        const float hi = float(asint(words.y)) * 4294967296.0f;
        return hi + float(words.x);
    }
    if (dtype == kDtypeFloat16)
    {
        return load_f16(buf, byte_offset);
    }
    return load_f32(buf, byte_offset);
}

void store_from_float(RWByteAddressBuffer buf, uint byte_offset, uint dtype, float value)
{
    if (dtype == kDtypeBool)
    {
        store_u8(buf, byte_offset, value != 0.0f ? 1u : 0u);
        return;
    }
    if (dtype == kDtypeUInt8)
    {
        const int clamped = clamp(int(round(value)), 0, 255);
        store_u8(buf, byte_offset, uint(clamped));
        return;
    }
    if (dtype == kDtypeInt32)
    {
        store_i32(buf, byte_offset, int(value));
        return;
    }
    if (dtype == kDtypeInt64)
    {
        store_i64_from_i32(buf, byte_offset, int(value));
        return;
    }
    if (dtype == kDtypeFloat16)
    {
        store_f16(buf, byte_offset, value);
        return;
    }
    store_f32(buf, byte_offset, value);
}

void copy_elem(RWByteAddressBuffer dst, uint dst_off, RWByteAddressBuffer src, uint src_off, uint elem_size)
{
    if (elem_size == 1)
    {
        store_u8(dst, dst_off, load_u8(src, src_off));
        return;
    }
    if (elem_size == 2)
    {
        store_u16(dst, dst_off, load_u16(src, src_off));
        return;
    }
    if (elem_size == 8)
    {
        dst.Store2(dst_off, src.Load2(src_off));
        return;
    }
    dst.Store(dst_off, src.Load(src_off));
}

#endif
