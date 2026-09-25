#include "splat_math.hlsli"

[[vk::binding(0, 0)]] StructuredBuffer<uint> ranges;
[[vk::binding(1, 0)]] StructuredBuffer<uint> instances;
[[vk::binding(2, 0)]] StructuredBuffer<float4> gauss_f;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> out_f;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> out_u;
[[vk::binding(5, 0)]] StructuredBuffer<uint> bucket_offset;
[[vk::binding(6, 0)]] RWStructuredBuffer<float4> snap;

groupshared uint sid[256];
groupshared float2 sxy[256];
groupshared float4 sconic[256];
groupshared float3 srgb[256];
groupshared float4 splane[256];
groupshared float3 snormal[256];
groupshared uint alive_count;
groupshared uint red[256];

[numthreads(16, 16, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread : SV_GroupThreadID, uint rank : SV_GroupIndex) {
    uint width = pc.u0;
    uint height = pc.u1;
    uint tiles_x = pc.u2;
    bool geometry = (pc.u3 & 4u) != 0;
    bool snapshots = (pc.u3 & 8u) != 0;
    uint mode = pc.u4;
    int wrap_width = int(pc.u5);
    float fx = asfloat(pc.u6);
    float fy = asfloat(pc.u7);
    float cx = asfloat(pc.u8);
    float cy = asfloat(pc.u9);
    float k1 = asfloat(pc.u10);
    float k2 = asfloat(pc.u11);
    float k3 = asfloat(pc.u12);
    float k4 = asfloat(pc.u13);
    float bg0 = asfloat(pc.u14);
    float bg1 = asfloat(pc.u15);
    float bg2 = asfloat(pc.u16);
    uint gaussian_count = pc.u17;
    uint pixel_count = pc.u18;
    uint bucket_limit = pc.u19;

    uint tile_id = group_id.y * tiles_x + group_id.x;
    uint pix_x = group_id.x * 16 + group_thread.x;
    uint pix_y = group_id.y * 16 + group_thread.y;
    bool inside = pix_x < width && pix_y < height;
    uint pix = width * pix_y + pix_x;
    float2 pixf = float2(float(pix_x), float(pix_y));
    uint range_begin = ranges[tile_id * 2];
    uint range_end = ranges[tile_id * 2 + 1];
    uint todo = range_end - range_begin;
    uint rounds = (todo + 255u) >> 8;
    uint bucket_base = tile_id == 0 ? 0u : bucket_offset[tile_id - 1];
    uint bucket_count = (todo + 31u) >> 5;
    for (uint bucket = rank; bucket < bucket_count; bucket += 256u) {
        out_u[gaussian_count + pixel_count + (height + 15u) / 16u * tiles_x + bucket_base + bucket] = tile_id;
    }

    float transmittance = 1.0f;
    float color0 = 0.0f;
    float color1 = 0.0f;
    float color2 = 0.0f;
    float normal0 = 0.0f;
    float normal1 = 0.0f;
    float normal2 = 0.0f;
    float depth_seed = 0.0f;
    uint contributor = 0;
    uint last_contributor = 0;
    bool done = !inside;

    [loop] for (uint round = 0; round < rounds; ++round) {
        if (rank == 0) alive_count = 0;
        GroupMemoryBarrierWithGroupSync();
        if (!done) InterlockedAdd(alive_count, 1u);
        GroupMemoryBarrierWithGroupSync();
        if (alive_count == 0) break;

        uint progress = round * 256u + rank;
        if (range_begin + progress < range_end) {
            uint gaussian = instances[range_begin + progress];
            sid[rank] = gaussian;
            sxy[rank] = gauss_f[gaussian * 8].xy;
            sconic[rank] = gauss_f[gaussian * 8 + 1];
            srgb[rank] = gauss_f[gaussian * 8 + 2].xyz;
            if (geometry) {
                splane[rank] = gauss_f[gaussian * 8 + 3];
                snormal[rank] = gauss_f[gaussian * 8 + 4].xyz;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        uint remaining = todo - round * 256u;
        uint batch = min(256u, remaining);
        uint sub_count = (batch + 31u) >> 5;
        [loop] for (uint sub = 0; sub < sub_count; ++sub) {
            if (snapshots && !done) {
                uint slot = ((bucket_base + round * 8u + sub) << 8) + rank;
                snap[slot] = float4(color0, color1, color2, transmittance);
                if (geometry) snap[bucket_limit * 256u + slot] = float4(normal0, normal1, normal2, 0.0f);
            }
            uint j_end = min(batch, (sub + 1u) << 5);
            [loop] for (uint j = sub << 5; !done && j < j_end; ++j) {
                ++contributor;
                float2 delta = float2(wrap_dx(sxy[j].x - pixf.x, wrap_width, mode), sxy[j].y - pixf.y);
                float4 conic = sconic[j];
                float power = gaussian_power(conic, delta.x, delta.y);
                if (power > 0.0f) continue;
                float alpha = min(kAlphaClip, conic.w * exp(power));
                if (alpha < kAlphaFloor) continue;
                float test_t = transmittance * (1.0f - alpha);
                if (test_t < kTransmittanceFloor) {
                    done = true;
                    continue;
                }
                float weight = alpha * transmittance;
                InterlockedOr(out_u[sid[j]], 1u);
                color0 += srgb[j].x * weight;
                color1 += srgb[j].y * weight;
                color2 += srgb[j].z * weight;
                if (geometry) {
                    float depth = splane[j].x * delta.x + splane[j].y * delta.y + splane[j].z;
                    normal0 += snormal[j].x * weight;
                    normal1 += snormal[j].y * weight;
                    normal2 += snormal[j].z * weight;
                    depth_seed = transmittance > 0.5f ? depth : depth_seed;
                }
                transmittance = test_t;
                last_contributor = contributor;
            }
        }
    }

    red[rank] = last_contributor;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint stride = 128u; stride > 0u; stride >>= 1) {
        if (rank < stride) red[rank] = max(red[rank], red[rank + stride]);
        GroupMemoryBarrierWithGroupSync();
    }
    uint block_max = red[0];
    if (rank == 0) out_u[gaussian_count + pixel_count + tile_id] = block_max;

    float median = 0.0f;
    bool in_range = transmittance <= kDepthMinTransmittance;
    float depth_min = max(depth_seed - kDepthSeedWindow, 0.0f);
    float depth_max = max(depth_seed + kDepthSeedWindow, 0.0f);
    float probe[9];
    [unroll] for (uint probe_reset = 0; probe_reset < 9; ++probe_reset) probe[probe_reset] = 1.0f;

    if (geometry) {
        [loop] for (uint refine = 0; refine < uint(kDepthRefinements); ++refine) {
            bool first = refine == 0;
            uint probe_begin = first ? 0u : 1u;
            uint probe_end = first ? 9u : 8u;
            [unroll] for (uint probe_fill = probe_begin; probe_fill < probe_end; ++probe_fill) probe[probe_fill] = 1.0f;
            float interval = (depth_max - depth_min) * (1.0f / float(kDepthSplit));
            bool local_done = !in_range;
            uint todo_depth = block_max;
            uint contributor_depth = 0;
            uint depth_rounds = (block_max + 255u) >> 8;
            [loop] for (uint depth_round = 0; depth_round < depth_rounds; ++depth_round) {
                if (rank == 0) alive_count = 0;
                GroupMemoryBarrierWithGroupSync();
                if (!local_done) InterlockedAdd(alive_count, 1u);
                GroupMemoryBarrierWithGroupSync();
                if (alive_count == 0) break;

                uint depth_progress = depth_round * 256u + rank;
                if (depth_progress < block_max) {
                    uint gaussian_depth = instances[range_begin + depth_progress];
                    sxy[rank] = gauss_f[gaussian_depth * 8].xy;
                    sconic[rank] = gauss_f[gaussian_depth * 8 + 1];
                    splane[rank] = gauss_f[gaussian_depth * 8 + 3];
                }
                GroupMemoryBarrierWithGroupSync();
                uint depth_remaining = todo_depth - depth_round * 256u;
                uint depth_batch = min(256u, depth_remaining);
                [loop] for (uint dj = 0; !local_done && dj < depth_batch; ++dj) {
                    ++contributor_depth;
                    local_done = contributor_depth >= last_contributor;
                    float2 delta_d = float2(wrap_dx(sxy[dj].x - pixf.x, wrap_width, mode), sxy[dj].y - pixf.y);
                    float4 conic_d = sconic[dj];
                    float power_d = gaussian_power(conic_d, delta_d.x, delta_d.y);
                    if (power_d > 0.0f) continue;
                    float alpha_d = min(kAlphaClip, conic_d.w * exp(power_d));
                    if (alpha_d < kAlphaFloor) continue;
                    float peak = splane[dj].x * delta_d.x + splane[dj].y * delta_d.y + splane[dj].z;
                    float rsigma = splane[dj].w;
                    bool ball = rsigma > 0.0f;
                    [unroll] for (uint sample = probe_begin; sample < probe_end; ++sample) {
                        float ts = depth_min + interval * float(sample);
                        float delta_t = (ts - peak) * rsigma;
                        float response = ball ? exp(-0.5f * delta_t * delta_t) : 0.0f;
                        float complement = 1.0f - alpha_d * response;
                        float root = rsqrt(complement);
                        probe[sample] *= (ts > peak ? (1.0f - alpha_d) : complement) * root;
                    }
                }
            }
            if (first) in_range = probe[0] >= 0.5f && probe[kDepthSplit] <= 0.5f && in_range;
            uint start_id = 0;
            [unroll] for (uint bracket = 1; bracket < uint(kDepthSplit); ++bracket) {
                if (probe[bracket] >= 0.5f) start_id = bracket;
            }
            float next_max = depth_min + float(start_id + 1u) * interval;
            float next_min = depth_min + float(start_id) * interval;
            probe[0] = probe[start_id];
            probe[kDepthSplit] = probe[start_id + 1u];
            depth_max = next_max;
            depth_min = next_min;
        }
        float weight_max = saturate((probe[0] - 0.5f) / (probe[0] - probe[kDepthSplit]));
        median = in_range ? weight_max * depth_max + (1.0f - weight_max) * depth_min : 0.0f;
    }

    if (inside) {
        out_u[gaussian_count + pix] = last_contributor;
        out_f[8u * pixel_count + pix] = color0;
        out_f[8u * pixel_count + pixel_count + pix] = color1;
        out_f[8u * pixel_count + 2u * pixel_count + pix] = color2;
        out_f[pix] = color0 + transmittance * bg0;
        out_f[pixel_count + pix] = color1 + transmittance * bg1;
        out_f[2u * pixel_count + pix] = color2 + transmittance * bg2;
        out_f[3u * pixel_count + pix] = 1.0f - transmittance;
        if (geometry) {
            float ray_z = pixel_ray_z(pixf.x, pixf.y, mode, fx, fy, cx, cy, k1, k2, k3, k4);
            out_f[7u * pixel_count + pix] = median * ray_z;
            float norm_weight = 1.0f - transmittance;
            out_f[4u * pixel_count + pix] = last_contributor != 0 ? normal0 / norm_weight : 0.0f;
            out_f[5u * pixel_count + pix] = last_contributor != 0 ? normal1 / norm_weight : 0.0f;
            out_f[6u * pixel_count + pix] = last_contributor != 0 ? normal2 / norm_weight : 0.0f;
        }
    }
}
