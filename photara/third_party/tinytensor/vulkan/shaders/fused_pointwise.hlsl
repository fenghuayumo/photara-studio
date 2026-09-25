// Float32 unary/scalar chain used by TinyTensor's existing lazy pointwise IR.
// Keeping the recipe in a small storage buffer avoids exceeding Vulkan's
// portable 128-byte push-constant limit for the 16-op recipe.

struct PushConstants
{
    uint count;
    uint src_offset;
    uint dst_offset;
    uint num_ops;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer src;
[[vk::binding(1, 0)]] RWByteAddressBuffer dst;
[[vk::binding(2, 0)]] RWByteAddressBuffer recipe;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

float apply_op(float value, uint kind, float scalar)
{
    switch (kind)
    {
    case 0u: return value + scalar;
    case 1u: return value - scalar;
    case 2u: return value * scalar;
    case 3u: return value / scalar;
    case 10u: return abs(value);
    case 11u: return -value;
    case 12u: return exp(value);
    case 13u: return log(max(value, 1e-10f));
    case 14u: return sqrt(max(value, 0.0f));
    case 15u: return 1.0f / (1.0f + exp(-value));
    case 16u: return max(value, 0.0f);
    case 17u: return value * value;
    case 18u: return tanh(value);
    case 19u: return rsqrt(max(value, 1e-10f));
    case 20u: return float((value > 0.0f) - (value < 0.0f));
    case 21u: return 1.0f / (value + 1e-8f);
    case 22u: return floor(value);
    case 23u: return ceil(value);
    case 24u: return round(value);
    default: return value;
    }
}

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint index = dtid.x;
    if (index >= pc.count) return;
    float value = asfloat(src.Load(pc.src_offset + index * 4u));
    [loop]
    for (uint op = 0u; op < pc.num_ops; ++op)
    {
        const uint address = op * 8u;
        value = apply_op(value, recipe.Load(address), asfloat(recipe.Load(address + 4u)));
    }
    dst.Store(pc.dst_offset + index * 4u, asuint(value));
}
