#ifndef PHOTARA_SIFT_COMMON_HLSLI
#define PHOTARA_SIFT_COMMON_HLSLI

// Byte-buffer helpers shared by the Vulkan SIFT/matcher passes. Everything is
// addressed through RWByteAddressBuffer with explicit offsets so the HLSL and
// host layouts cannot drift, mirroring the raw-pointer CUDA kernels this
// pipeline translates.

float load_f32(RWByteAddressBuffer buf, uint byte_offset)
{
    return asfloat(buf.Load(byte_offset));
}

void store_f32(RWByteAddressBuffer buf, uint byte_offset, float value)
{
    buf.Store(byte_offset, asuint(value));
}

float2 load_f2(RWByteAddressBuffer buf, uint byte_offset)
{
    const uint2 words = buf.Load2(byte_offset);
    return float2(asfloat(words.x), asfloat(words.y));
}

void store_f2(RWByteAddressBuffer buf, uint byte_offset, float2 value)
{
    buf.Store2(byte_offset, uint2(asuint(value.x), asuint(value.y)));
}

float4 load_f4(RWByteAddressBuffer buf, uint byte_offset)
{
    const uint4 words = buf.Load4(byte_offset);
    return float4(asfloat(words.x), asfloat(words.y), asfloat(words.z), asfloat(words.w));
}

void store_f4(RWByteAddressBuffer buf, uint byte_offset, float4 value)
{
    buf.Store4(byte_offset, uint4(asuint(value.x), asuint(value.y),
                                  asuint(value.z), asuint(value.w)));
}

int4 load_i4(RWByteAddressBuffer buf, uint byte_offset)
{
    return asint(buf.Load4(byte_offset));
}

void store_i4(RWByteAddressBuffer buf, uint byte_offset, int4 value)
{
    buf.Store4(byte_offset, asuint(value));
}

// tex1Dfetch semantics of the CUDA kernels: no address clamping, out-of-range
// reads return zero.
float fetch_f32(RWByteAddressBuffer buf, uint index, uint count)
{
    return index < count ? load_f32(buf, index * 4u) : 0.0f;
}

int clampi(int value, int low, int high)
{
    return value < low ? low : (value > high ? high : value);
}

#endif
