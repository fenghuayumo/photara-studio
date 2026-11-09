#include "splat_ssim_common.hlsli"

groupshared float3 derivative_tile[26 * 26];
groupshared float3 horizontal[26 * 16];

[numthreads(16, 16, 1)]
void main(uint3 group : SV_GroupID, uint3 tid : SV_GroupThreadID, uint lane : SV_GroupIndex) {
    uint width = pc.u0, height = pc.u1, pixels = width * height, channel = group.z;
    for (uint i = lane; i < 26u * 26u; i += 256u) {
        int x = int(group.x * 16u + i % 26u) - 5;
        int y = int(group.y * 16u + i / 26u) - 5;
        float3 v = 0;
        // The loss is normalized on the valid 11x11 crop only.
        if (x >= 5 && y >= 5 && x < int(width) - 5 && y < int(height) - 5) {
            uint j = channel * pixels + uint(y) * width + uint(x);
            v = float3(derivative0[j], derivative1[j], derivative2[j]);
        }
        derivative_tile[i] = v;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint i = lane; i < 26u * 16u; i += 256u) {
        uint row = i / 16u, col = i % 16u;
        float3 sum = 0;
        [unroll] for (uint d = 0; d < 11u; ++d)
            sum += gaussian[d] * derivative_tile[row * 26u + col + d];
        horizontal[i] = sum;
    }
    GroupMemoryBarrierWithGroupSync();
    uint x = group.x * 16u + tid.x, y = group.y * 16u + tid.y;
    if (x >= width || y >= height) return;
    float3 sum = 0;
    [unroll] for (uint d = 0; d < 11u; ++d)
        sum += gaussian[d] * horizontal[(tid.y + d) * 16u + tid.x];
    bool masked = pc.u3 != 0u;
    float pixel1 = prediction_at(channel, x, y, width, height, masked);
    float pixel2 = target_at(channel, x, y, width, height, masked);
    float sign_value = pixel1 > pixel2 ? 1.0f : pixel1 < pixel2 ? -1.0f : 0.0f;
    float l1_chain = x >= 5u && x < width - 5u && y >= 5u && y < height - 5u
        ? push_float(pc.u4) : 0.0f;
    float valid = masked ? mask_image[y * width + x] : 1.0f;
    output[channel * pixels + y * width + x] = (sum.x + 2.0f * pixel1 * sum.y +
        pixel2 * sum.z + (1.0f - push_float(pc.u2)) * sign_value * l1_chain) * valid;
}
