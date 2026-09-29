#include "sift_common.hlsli"

// NormalizeDescriptor: L2 normalize, clamp at 0.2, renormalize.
struct PushConstants
{
    uint count;
    uint pad0;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer descriptors;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[numthreads(64, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint index = dtid.x;
    if (index >= pc.count) return;

    float values[128];
    float norm1 = 0.0f;
    [unroll]
    for (uint i = 0u; i < 32u; ++i)
    {
        const float4 temp = load_f4(descriptors, (index * 128u + i * 4u) * 4u);
        values[i * 4u + 0u] = temp.x;
        values[i * 4u + 1u] = temp.y;
        values[i * 4u + 2u] = temp.z;
        values[i * 4u + 3u] = temp.w;
        norm1 += dot(temp, temp);
    }
    norm1 = rsqrt(norm1);

    float norm2 = 0.0f;
    [unroll]
    for (uint i = 0u; i < 128u; ++i)
    {
        values[i] = min(0.2f, values[i] * norm1);
        norm2 += values[i] * values[i];
    }
    norm2 = rsqrt(norm2);

    [unroll]
    for (uint i = 0u; i < 32u; ++i)
    {
        store_f4(descriptors, (index * 128u + i * 4u) * 4u,
                 float4(values[i * 4u + 0u], values[i * 4u + 1u],
                        values[i * 4u + 2u], values[i * 4u + 3u]) * norm2);
    }
}
