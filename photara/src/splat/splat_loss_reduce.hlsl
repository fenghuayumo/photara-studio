#include "splat_math.hlsli"

// The fused SSIM shader stores [gradient, loss_map] in one buffer. This pass
// reduces only the valid 11x11-window interior of the loss map. Pass 0 writes
// one partial per workgroup; pass 1 reduces those partials to reduction[0].
StructuredBuffer<float> packed_loss : register(t0);
RWStructuredBuffer<float> reduction : register(u1);

groupshared float local_sum[256];

float push_float(uint value) { return asfloat(value); }

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint local_id : SV_GroupIndex) {
    uint width = pc.u0;
    uint height = pc.u1;
    uint pass = pc.u2;
    uint pixels = width * height;
    float sum = 0.0f;

    if (pass == 0u) {
        uint index = group_id.x * 256u + local_id;
        uint count = 3u * pixels;
        if (index < count) {
            uint pixel = index % pixels;
            uint x = pixel % width;
            uint y = pixel / width;
            if (x >= 5u && x < width - 5u && y >= 5u && y < height - 5u)
                sum = packed_loss[count + index];
        }
    } else {
        uint partial_count = pc.u3;
        for (uint index = local_id; index < partial_count; index += 256u)
            sum += reduction[index];
    }

    local_sum[local_id] = sum;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (local_id < stride) local_sum[local_id] += local_sum[local_id + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (local_id == 0u) {
        float value = local_sum[0];
        if (pass != 0u) value *= push_float(pc.u4);
        reduction[pass == 0u ? group_id.x : 0u] = value;
    }
}
