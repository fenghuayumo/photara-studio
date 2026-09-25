#include "splat_math.hlsli"

[[vk::binding(0, 0)]] StructuredBuffer<uint> ranges;
[[vk::binding(1, 0)]] StructuredBuffer<uint> instances;
[[vk::binding(2, 0)]] StructuredBuffer<float4> gauss_f;
[[vk::binding(3, 0)]] StructuredBuffer<float> out_f;
[[vk::binding(4, 0)]] StructuredBuffer<uint> out_u;
[[vk::binding(5, 0)]] StructuredBuffer<uint> bucket_offset;
[[vk::binding(6, 0)]] StructuredBuffer<float4> snap;
[[vk::binding(7, 0)]] StructuredBuffer<float> loss_color;
[[vk::binding(8, 0)]] StructuredBuffer<float> loss_alpha;
// IEEE-754 bits are used so this works on Vulkan 1.2 devices without the
// optional shaderBufferFloat32AtomicAdd feature.
[[vk::binding(9, 0)]] RWStructuredBuffer<uint> grad_values;

void atomic_add_f32(uint index, float value) {
    if (value == 0.0f || isnan(value) || isinf(value)) return;
    uint expected = grad_values[index];
    [loop] for (;;) {
        uint desired = asuint(asfloat(expected) + value);
        uint observed;
        InterlockedCompareExchange(grad_values[index], expected, desired, observed);
        if (observed == expected) return;
        expected = observed;
    }
}

groupshared float4 wave_state[256];
groupshared float4 wave_loss[256];
groupshared float4 wave_final[256];
groupshared uint wave_last[256];

void commit_gradient(
    uint gaussian, uint gaussian_count, float3 acc_mean,
    float4 acc_conic, float3 acc_color) {
    uint mean_base = gaussian * 3u;
    uint conic_base = gaussian_count * 3u + gaussian * 4u;
    uint color_base = gaussian_count * 7u + gaussian * 3u;
    atomic_add_f32(mean_base, acc_mean.x);
    atomic_add_f32(mean_base + 1u, acc_mean.y);
    atomic_add_f32(mean_base + 2u, acc_mean.z);
    atomic_add_f32(conic_base, acc_conic.x);
    atomic_add_f32(conic_base + 1u, acc_conic.y);
    atomic_add_f32(conic_base + 2u, acc_conic.z);
    atomic_add_f32(conic_base + 3u, acc_conic.w);
    atomic_add_f32(color_base, acc_color.x);
    atomic_add_f32(color_base + 1u, acc_color.y);
    atomic_add_f32(color_base + 2u, acc_color.z);
}

// NVIDIA and other wave32 devices can run the same diagonal wavefront as the
// CUDA FasterGS kernel: pixel state enters lane 0 and advances one Gaussian
// per lane. This removes the portable path's replay of preceding bucket lanes.
void wave32_main(uint3 group_id, uint group_thread) {
    uint lane = group_thread & 31u;
    uint warp = group_thread >> 5u;
    uint bucket_idx = group_id.x * 8u + warp;
    uint bucket_limit = pc.u8;
    bool bucket_active = bucket_idx < bucket_limit;

    uint width = pc.u0;
    uint height = pc.u1;
    uint tiles_x = pc.u2;
    uint mode = pc.u3;
    int wrap_width = int(pc.u4);
    uint gaussian_count = pc.u5;
    uint pixel_count = pc.u6;
    uint tile_count = pc.u7;
    float3 background = float3(asfloat(pc.u9), asfloat(pc.u10), asfloat(pc.u11));

    uint tile_id = bucket_active
        ? out_u[gaussian_count + pixel_count + tile_count + bucket_idx]
        : 0u;
    bucket_active = bucket_active && tile_id < tile_count;
    uint range_begin = bucket_active ? ranges[tile_id * 2u] : 0u;
    uint range_end = bucket_active ? ranges[tile_id * 2u + 1u] : 0u;
    uint tile_first = bucket_active && tile_id != 0u ? bucket_offset[tile_id - 1u] : 0u;
    bucket_active = bucket_active && bucket_idx >= tile_first;
    uint tile_bucket = bucket_active ? bucket_idx - tile_first : 0u;
    uint tile_max = bucket_active ? out_u[gaussian_count + pixel_count + tile_id] : 0u;
    bucket_active = bucket_active && tile_bucket * 32u < tile_max;
    uint tile_n = range_end - range_begin;
    uint pos = tile_bucket * 32u + lane;
    bool valid_instance = bucket_active && pos < tile_n;

    uint gaussian = valid_instance ? instances[range_begin + pos] : 0u;
    float2 mean = valid_instance ? gauss_f[gaussian * 8u].xy : 0.0f;
    float4 conic = valid_instance ? gauss_f[gaussian * 8u + 1u] : 0.0f;
    float3 color = valid_instance ? gauss_f[gaussian * 8u + 2u].xyz : 0.0f;
    uint4 bounds = valid_instance ? asuint(gauss_f[gaussian * 8u + 5u]) : 0u;
    uint pix_min_x = (tile_id % tiles_x) * 16u;
    uint pix_min_y = (tile_id / tiles_x) * 16u;

    float3 acc_color = 0.0f;
    float3 acc_mean = 0.0f;
    float4 acc_conic = 0.0f;
    float3 color_after = 0.0f;
    float3 pixel_grad = 0.0f;
    float transmittance = 0.0f;
    float final_t = 0.0f;
    float d_final_t = 0.0f;
    uint last_contributor = 0u;

    [loop] for (uint i = 0u; i < 287u; ++i) {
        if ((i & 31u) == 0u) {
            uint local = i + lane;
            float4 initial_state = float4(0.0f, 0.0f, 0.0f, 1.0f);
            float4 initial_loss = 0.0f;
            float4 initial_final = 0.0f;
            uint initial_last = 0u;
            if (bucket_active && local < 256u) {
                uint px = pix_min_x + (local & 15u);
                uint py = pix_min_y + (local >> 4u);
                if (px < width && py < height) {
                    uint pix = py * width + px;
                    float4 snapshot = snap[bucket_idx * 256u + local];
                    float3 total = float3(
                        out_f[8u * pixel_count + pix],
                        out_f[9u * pixel_count + pix],
                        out_f[10u * pixel_count + pix]);
                    float3 loss = float3(
                        loss_color[pix], loss_color[pixel_count + pix],
                        loss_color[2u * pixel_count + pix]);
                    float alpha_final = out_f[3u * pixel_count + pix];
                    float final_loss = dot(background, loss) - loss_alpha[pix];
                    initial_state = float4(total - snapshot.xyz, snapshot.w);
                    initial_loss = float4(loss, final_loss);
                    initial_final = float4(1.0f - alpha_final, 0.0f, 0.0f, 0.0f);
                    initial_last = out_u[gaussian_count + pix];
                }
            }
            uint shared_index = warp * 32u + lane;
            wave_state[shared_index] = initial_state;
            wave_loss[shared_index] = initial_loss;
            wave_final[shared_index] = initial_final;
            wave_last[shared_index] = initial_last;
            GroupMemoryBarrierWithGroupSync();
        }

        if (i > 0u) {
            uint source_lane = lane == 0u ? 0u : lane - 1u;
            color_after.x = WaveReadLaneAt(color_after.x, source_lane);
            color_after.y = WaveReadLaneAt(color_after.y, source_lane);
            color_after.z = WaveReadLaneAt(color_after.z, source_lane);
            pixel_grad.x = WaveReadLaneAt(pixel_grad.x, source_lane);
            pixel_grad.y = WaveReadLaneAt(pixel_grad.y, source_lane);
            pixel_grad.z = WaveReadLaneAt(pixel_grad.z, source_lane);
            transmittance = WaveReadLaneAt(transmittance, source_lane);
            final_t = WaveReadLaneAt(final_t, source_lane);
            d_final_t = WaveReadLaneAt(d_final_t, source_lane);
            last_contributor = WaveReadLaneAt(last_contributor, source_lane);
        }

        int idx = int(i) - int(lane);
        if (lane == 0u && idx >= 0 && idx < 256) {
            uint shared_index = warp * 32u + (uint(idx) & 31u);
            float4 initial_state = wave_state[shared_index];
            float4 initial_loss = wave_loss[shared_index];
            color_after = initial_state.xyz;
            transmittance = initial_state.w;
            pixel_grad = initial_loss.xyz;
            d_final_t = initial_loss.w;
            final_t = wave_final[shared_index].x;
            last_contributor = wave_last[shared_index];
        }
        if (idx < 0 || idx >= 256 || !valid_instance || pos >= last_contributor) continue;

        uint px = pix_min_x + (uint(idx) & 15u);
        uint py = pix_min_y + (uint(idx) >> 4u);
        if (px >= width || py >= height) continue;
        if (wrap_width == 0 &&
            (px < bounds.x || px >= bounds.y || py < bounds.z || py >= bounds.w)) continue;
        float2 delta = float2(
            wrap_dx(mean.x - float(px), wrap_width, mode), mean.y - float(py));
        float power = gaussian_power(conic, delta.x, delta.y);
        if (power > 0.0f) continue;
        float G = exp(power);
        float alpha = min(kAlphaClip, conic.w * G);
        if (alpha < kAlphaFloor) continue;

        float blend_weight = alpha * transmittance;
        float inv_1ma = 1.0f / (1.0f - alpha);
        color_after -= blend_weight * color;
        float d_opacity = transmittance * dot(color, pixel_grad) -
                          dot(color_after, pixel_grad) * inv_1ma -
                          final_t * inv_1ma * d_final_t;
        if (mode == kModeFisheye && conic.w * G >= kAlphaClip) d_opacity = 0.0f;
        acc_color += blend_weight * pixel_grad;
        float dG = conic.w * d_opacity;
        float gdx = G * delta.x;
        float gdy = G * delta.y;
        float d_del_x = dG * (-gdx * conic.x - gdy * conic.y);
        float d_del_y = dG * (-gdy * conic.z - gdx * conic.y);
        acc_mean += float3(d_del_x, d_del_y, abs(d_del_x) + abs(d_del_y));
        acc_conic += float4(
            -0.5f * gdx * delta.x * dG,
            -0.5f * gdx * delta.y * dG,
            -0.5f * gdy * delta.y * dG,
            G * d_opacity);
        transmittance *= 1.0f - alpha;
    }
    if (valid_instance) commit_gradient(gaussian, gaussian_count, acc_mean, acc_conic, acc_color);
}

// Eight 32-thread logical buckets per workgroup, matching the CUDA launch.
// This first portable implementation deliberately avoids subgroup operations:
// every lane reconstructs at most the 31 preceding splats from the bucket
// snapshot. A later subgroup-specialized path can replace that local replay
// without changing the saved-state or public gradient ABI.
void portable_main(uint3 group_id, uint group_thread) {
    uint lane = group_thread & 31u;
    uint bucket_idx = group_id.x * 8u + (group_thread >> 5u);
    uint bucket_limit = pc.u8;
    if (bucket_idx >= bucket_limit) return;

    uint width = pc.u0;
    uint height = pc.u1;
    uint tiles_x = pc.u2;
    uint mode = pc.u3;
    int wrap_width = int(pc.u4);
    uint gaussian_count = pc.u5;
    uint pixel_count = pc.u6;
    uint tile_count = pc.u7;
    float3 background = float3(asfloat(pc.u9), asfloat(pc.u10), asfloat(pc.u11));

    uint tile_id = out_u[gaussian_count + pixel_count + tile_count + bucket_idx];
    if (tile_id >= tile_count) return;
    uint range_begin = ranges[tile_id * 2u];
    uint range_end = ranges[tile_id * 2u + 1u];
    uint tile_first = tile_id == 0u ? 0u : bucket_offset[tile_id - 1u];
    if (bucket_idx < tile_first) return;
    uint tile_bucket = bucket_idx - tile_first;
    uint tile_max = out_u[gaussian_count + pixel_count + tile_id];
    if (tile_bucket * 32u >= tile_max) return;

    uint tile_n = range_end - range_begin;
    uint pos = tile_bucket * 32u + lane;
    if (pos >= tile_n) return;
    uint gaussian = instances[range_begin + pos];
    float2 mean = gauss_f[gaussian * 8u].xy;
    float4 conic = gauss_f[gaussian * 8u + 1u];
    float3 color = gauss_f[gaussian * 8u + 2u].xyz;
    uint4 bounds = asuint(gauss_f[gaussian * 8u + 5u]);

    float3 acc_color = 0.0f;
    float3 acc_mean = 0.0f;
    float4 acc_conic = 0.0f;
    uint pix_min_x = (tile_id % tiles_x) * 16u;
    uint pix_min_y = (tile_id / tiles_x) * 16u;

    [loop] for (uint local = 0u; local < 256u; ++local) {
        uint px = pix_min_x + (local & 15u);
        uint py = pix_min_y + (local >> 4u);
        if (px >= width || py >= height) continue;
        uint pix = py * width + px;
        uint last_contributor = out_u[gaussian_count + pix];
        if (pos >= last_contributor) continue;
        if (wrap_width == 0 &&
            (px < bounds.x || px >= bounds.y || py < bounds.z || py >= bounds.w)) continue;

        float4 state = snap[bucket_idx * 256u + local];
        float3 color_after = float3(
            out_f[8u * pixel_count + pix],
            out_f[9u * pixel_count + pix],
            out_f[10u * pixel_count + pix]) - state.xyz;
        float transmittance = state.w;
        float G = 0.0f;
        float alpha = 0.0f;
        float2 delta = 0.0f;
        bool accepted = false;

        // Replay this bucket through this lane. color_after starts as all
        // contributions after the snapshot and is reduced to contributions
        // strictly after this lane, exactly as in the CUDA reverse wavefront.
        [loop] for (uint k = 0u; k <= lane; ++k) {
            uint qpos = tile_bucket * 32u + k;
            if (qpos >= tile_n || qpos >= last_contributor) break;
            uint q = instances[range_begin + qpos];
            float2 qmean = gauss_f[q * 8u].xy;
            float4 qconic = gauss_f[q * 8u + 1u];
            float2 qdelta = float2(
                wrap_dx(qmean.x - float(px), wrap_width, mode),
                qmean.y - float(py));
            float power = gaussian_power(qconic, qdelta.x, qdelta.y);
            if (power > 0.0f) continue;
            float qG = exp(power);
            float qalpha = min(kAlphaClip, qconic.w * qG);
            if (qalpha < kAlphaFloor) continue;
            float weight = qalpha * transmittance;
            color_after -= weight * gauss_f[q * 8u + 2u].xyz;
            if (k == lane) {
                G = qG;
                alpha = qalpha;
                delta = qdelta;
                accepted = true;
                break;
            }
            transmittance *= 1.0f - qalpha;
        }
        if (!accepted) continue;

        float3 pixel_grad = float3(
            loss_color[pix], loss_color[pixel_count + pix],
            loss_color[2u * pixel_count + pix]);
        float final_t = 1.0f - out_f[3u * pixel_count + pix];
        float inv_1ma = 1.0f / (1.0f - alpha);
        float d_final_t = dot(background, pixel_grad) - loss_alpha[pix];
        float d_opacity = transmittance * dot(color, pixel_grad) -
                          dot(color_after, pixel_grad) * inv_1ma -
                          final_t * inv_1ma * d_final_t;
        if (mode == kModeFisheye && conic.w * G >= kAlphaClip) d_opacity = 0.0f;

        float blend_weight = alpha * transmittance;
        acc_color += blend_weight * pixel_grad;
        float dG = conic.w * d_opacity;
        float gdx = G * delta.x;
        float gdy = G * delta.y;
        float d_del_x = dG * (-gdx * conic.x - gdy * conic.y);
        float d_del_y = dG * (-gdy * conic.z - gdx * conic.y);
        acc_mean += float3(d_del_x, d_del_y, abs(d_del_x) + abs(d_del_y));
        acc_conic += float4(
            -0.5f * gdx * delta.x * dG,
            -0.5f * gdx * delta.y * dG,
            -0.5f * gdy * delta.y * dG,
            G * d_opacity);
    }

    commit_gradient(gaussian, gaussian_count, acc_mean, acc_conic, acc_color);
}

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint group_thread : SV_GroupIndex) {
    if (WaveGetLaneCount() == 32u) wave32_main(group_id, group_thread);
    else portable_main(group_id, group_thread);
}
