#include "splat_ssim_common.hlsli"

// A 16x16 output tile shares the 5-pixel halo and horizontal moments instead
// of writing five full images between the two convolution directions.
groupshared float2 image_tile[26 * 26];
groupshared float moments[26 * 16 * 5];

[numthreads(16, 16, 1)]
void main(uint3 group : SV_GroupID, uint3 tid : SV_GroupThreadID, uint lane : SV_GroupIndex) {
    uint width = pc.u0, height = pc.u1, channel = group.z;
    bool masked = pc.u3 != 0u;
    for (uint i = lane; i < 26u * 26u; i += 256u) {
        int x = int(group.x * 16u + i % 26u) - 5;
        int y = int(group.y * 16u + i / 26u) - 5;
        image_tile[i] = float2(prediction_at(channel, x, y, width, height, masked),
                               target_at(channel, x, y, width, height, masked));
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint i = lane; i < 26u * 16u; i += 256u) {
        uint row = i / 16u, col = i % 16u;
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0;
        [unroll] for (uint d = 0; d < 11u; ++d) {
            float2 v = image_tile[row * 26u + col + d];
            float q = gaussian[d];
            s0 += q * v.x; s1 += q * v.x * v.x;
            s2 += q * v.y; s3 += q * v.y * v.y; s4 += q * v.x * v.y;
        }
        moments[5u * i] = s0; moments[5u * i + 1u] = s1;
        moments[5u * i + 2u] = s2; moments[5u * i + 3u] = s3;
        moments[5u * i + 4u] = s4;
    }
    GroupMemoryBarrierWithGroupSync();
    uint x = group.x * 16u + tid.x, y = group.y * 16u + tid.y;
    if (x >= width || y >= height) return;
    float mu1 = 0, moment1 = 0, mu2 = 0, moment2 = 0, cross = 0;
    [unroll] for (uint d = 0; d < 11u; ++d) {
        uint i = 5u * ((tid.y + d) * 16u + tid.x);
        float q = gaussian[d];
        mu1 += q * moments[i]; moment1 += q * moments[i + 1u];
        mu2 += q * moments[i + 2u]; moment2 += q * moments[i + 3u];
        cross += q * moments[i + 4u];
    }
    float variance1 = moment1 - mu1 * mu1, variance2 = moment2 - mu2 * mu2;
    float covariance = cross - mu1 * mu2;
    float a = mu1 * mu1 + mu2 * mu2 + 0.0001f, b = variance1 + variance2 + 0.0009f;
    float c = 2.0f * mu1 * mu2 + 0.0001f, d = 2.0f * covariance + 0.0009f;
    float weight = push_float(pc.u2), normalization = push_float(pc.u4);
    uint pixels = width * height, index = channel * pixels + y * width + x;
    derivative0[index] = -weight * normalization *
        ((2.0f * mu2 * d) / (a * b) - (2.0f * mu2 * c) / (a * b) -
         (2.0f * mu1 * c * d) / (a * a * b) + (2.0f * mu1 * c * d) / (a * b * b));
    derivative1[index] = -weight * normalization * (-c * d / (a * b * b));
    derivative2[index] = -weight * normalization * (2.0f * c / (a * b));
    if (pc.u6 != 0u) {
        float2 center = image_tile[(tid.y + 5u) * 26u + tid.x + 5u];
        output[3u * pixels + index] = weight * (1.0f - c * d / (a * b)) +
            (1.0f - weight) * abs(center.x - center.y);
    }
}
