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
[[vk::binding(9, 0)]] StructuredBuffer<float> loss_normal;
[[vk::binding(10, 0)]] StructuredBuffer<float2> median_state;
[[vk::binding(11, 0)]] RWStructuredBuffer<uint> grad_values;

groupshared float2 gs_mean[256];
groupshared float4 gs_conic[256];
groupshared float3 gs_color[256];
groupshared float4 gs_bounds[256];

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

void commit_gradient(uint gaussian, uint count, float3 acc_mean,
                     float4 acc_conic, float3 acc_color, float acc_refine) {
    uint mean_base = gaussian * 3u;
    uint conic_base = count * 3u + gaussian * 4u;
    uint color_base = count * 7u + gaussian * 3u;
    uint refine_base = count * 17u + gaussian;
    atomic_add_f32(mean_base, acc_mean.x);
    atomic_add_f32(mean_base + 1u, acc_mean.y);
    atomic_add_f32(mean_base + 2u, acc_mean.z);
    [unroll] for (uint i = 0u; i < 4u; ++i) atomic_add_f32(conic_base + i, acc_conic[i]);
    [unroll] for (uint c = 0u; c < 3u; ++c) atomic_add_f32(color_base + c, acc_color[c]);
    atomic_add_f32(refine_base, acc_refine);
}

// No-geometry training path. A 32-lane subgroup owns one snapshot bucket. The
// bucket entries are staged once in group memory and subgroup prefix operations
// replace each lane's O(lane) replay of its preceding Gaussians.
[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint group_thread : SV_GroupIndex) {
    uint lane = group_thread & 31u;
    uint bucket_idx = group_id.x * 8u + (group_thread >> 5u);
    uint bucket_limit = pc.u8;
    uint width = pc.u0, height = pc.u1, tiles_x = pc.u2, mode = pc.u3;
    int wrap_width = int(pc.u4);
    uint count = pc.u5, pixel_count = pc.u6, tile_count = pc.u7;
    float3 background = float3(asfloat(pc.u9), asfloat(pc.u10), asfloat(pc.u11));

    uint tile_id = 0;
    uint range_begin = 0, range_end = 0;
    uint tile_bucket = 0, tile_n = 0;
    uint pos = 0;
    bool bucket_active = bucket_idx < bucket_limit;
    if (bucket_active) {
        tile_id = out_u[count + pixel_count + tile_count + bucket_idx];
        bucket_active = tile_id < tile_count;
    }
    if (bucket_active) {
        range_begin = ranges[tile_id * 2u];
        range_end = ranges[tile_id * 2u + 1u];
        uint tile_first = tile_id == 0u ? 0u : bucket_offset[tile_id - 1u];
        bucket_active = bucket_idx >= tile_first;
        if (bucket_active) {
            tile_bucket = bucket_idx - tile_first;
            uint tile_max = out_u[count + pixel_count + tile_id];
            bucket_active = tile_bucket * 32u < tile_max;
        }
    }
    if (bucket_active) {
        tile_n = range_end - range_begin;
        pos = tile_bucket * 32u + lane;
    }

    gs_mean[group_thread] = 0.0f;
    gs_conic[group_thread] = 0.0f;
    gs_color[group_thread] = 0.0f;
    gs_bounds[group_thread] = 0.0f;
    uint gaussian = 0;
    if (bucket_active && pos < tile_n) {
        gaussian = instances[range_begin + pos];
        gs_mean[group_thread] = gauss_f[gaussian * 5u].xy;
        gs_conic[group_thread] = gauss_f[gaussian * 5u + 1u];
        gs_color[group_thread] = gauss_f[gaussian * 5u + 2u].xyz;
        gs_bounds[group_thread] = gauss_f[gaussian * 5u + 3u];
    }
    GroupMemoryBarrierWithGroupSync();
    if (!bucket_active) return;

    const bool entry_valid = pos < tile_n;
    const uint slot = group_thread;
    const float2 mean = gs_mean[slot];
    const float4 conic = gs_conic[slot];
    const float3 color = gs_color[slot];
    const uint4 bounds = asuint(gs_bounds[slot]);
    const uint pix_min_x = (tile_id % tiles_x) * 16u;
    const uint pix_min_y = (tile_id / tiles_x) * 16u;
    const uint qbase = tile_bucket * 32u;

    float3 acc_color = 0.0f, acc_mean = 0.0f;
    float4 acc_conic = 0.0f;
    float acc_refine = 0.0f;

    // The tile's 256 pixels are staged 32 at a time, one per lane, instead of
    // one pixel per iteration with lane 0 loading and the subgroup broadcasting.
    // The per-pixel tail of the loop is memory-latency bound: a lane 0 load
    // followed by six broadcasts pays a full round trip for every pixel, while
    // 32 parallel lane loads pay one (and coalesce: the pixels of a lane group
    // are adjacent, so the reads of out_u/out_f/loss_color are contiguous).
    [loop] for (uint base = 0u; base < 256u; base += 32u) {
        const uint staged_local = base + lane;
        const uint staged_px = pix_min_x + (staged_local & 15u);
        const uint staged_py = pix_min_y + (staged_local >> 4u);
        const uint staged_pixel = staged_py * width + staged_px;
        const bool staged_in_image = staged_px < width && staged_py < height;
        uint stage_last = 0;
        float4 stage_state = 0.0f;
        float3 stage_final = 0.0f;
        float stage_alpha = 0.0f;
        float3 stage_grad = 0.0f;
        float stage_loss_alpha = 0.0f;
        if (staged_in_image) {
            stage_last = out_u[count + staged_pixel];
            stage_state = snap[bucket_idx * 256u + staged_local];
            stage_final = float3(out_f[8u * pixel_count + staged_pixel],
                out_f[9u * pixel_count + staged_pixel],
                out_f[10u * pixel_count + staged_pixel]);
            stage_alpha = out_f[3u * pixel_count + staged_pixel];
            stage_grad = float3(loss_color[staged_pixel],
                                loss_color[pixel_count + staged_pixel],
                                loss_color[2u * pixel_count + staged_pixel]);
            stage_loss_alpha =
                pc.u13 != 0u ? loss_alpha[staged_pixel] : 0.0f;
        }

        [loop] for (uint j = 0u; j < 32u; ++j) {
        const uint local = base + j;
        const uint px = pix_min_x + (local & 15u);
        const uint py = pix_min_y + (local >> 4u);
        const uint last = WaveReadLaneAt(stage_last, j);
        const float4 state = WaveReadLaneAt(stage_state, j);
        const float3 final_color = WaveReadLaneAt(stage_final, j);
        const float alpha_final = WaveReadLaneAt(stage_alpha, j);
        const float3 pixel_grad = WaveReadLaneAt(stage_grad, j);
        const float loss_alpha_value = WaveReadLaneAt(stage_loss_alpha, j);

        const bool pixel_active = entry_valid && px < width && py < height &&
            pos < last &&
            (wrap_width != 0 ||
             (px >= bounds.x && px < bounds.y && py >= bounds.z && py < bounds.w));
        float2 qdelta = float2(
            wrap_dx(gs_mean[slot].x - float(px), wrap_width, mode),
            gs_mean[slot].y - float(py));
        float power = gaussian_power(gs_conic[slot], qdelta.x, qdelta.y);
        float qG = power <= 0.0f ? exp(power) : 0.0f;
        float qalpha = pixel_active && qG > 0.0f
            ? min(kAlphaClip, gs_conic[slot].w * qG)
            : 0.0f;
        qalpha = qalpha >= kAlphaFloor ? qalpha : 0.0f;

        const float transmittance =
            state.w * WavePrefixProduct(1.0f - qalpha);
        const float blend_weight = qalpha * transmittance;
        // The prefix color only ever reaches the gradient projected onto the
        // pixel gradient, so the channel-wise prefix sum folds into one scalar
        // prefix over the projected contributions.
        const float color_after_dot =
            dot(final_color - state.xyz - blend_weight * color, pixel_grad) -
            WavePrefixSum(blend_weight * dot(gs_color[slot], pixel_grad));
        if (!pixel_active || qalpha == 0.0f) continue;

        const float final_t = 1.0f - alpha_final;
        const float d_final_t_render =
            dot(background, pixel_grad) - loss_alpha_value;
        const float inv_1ma = 1.0f / (1.0f - qalpha);
        float d_opacity_render =
            transmittance * dot(color, pixel_grad) - color_after_dot * inv_1ma;
        float d_opacity = d_opacity_render;
        d_opacity_render -= final_t * inv_1ma * d_final_t_render;
        d_opacity -= final_t * inv_1ma * d_final_t_render;
        if (mode == kModeFisheye && conic.w * qG >= kAlphaClip) {
            d_opacity = 0.0f;
            d_opacity_render = 0.0f;
        }

        acc_color += blend_weight * pixel_grad;
        const float dG = conic.w * d_opacity;
        const float dG_render = conic.w * d_opacity_render;
        const float gdx = qG * qdelta.x, gdy = qG * qdelta.y;
        const float d_del_x = dG * (-gdx * conic.x - gdy * conic.y);
        const float d_del_y = dG * (-gdy * conic.z - gdx * conic.y);
        acc_mean += float3(d_del_x, d_del_y, abs(d_del_x) + abs(d_del_y));
        const float refine_x =
            dG_render * (-gdx * conic.x - gdy * conic.y) * float(width);
        const float refine_y =
            dG_render * (-gdy * conic.z - gdx * conic.y) * float(height);
        acc_refine += length(float2(refine_x, refine_y)) /
                      max(alpha_final, 1.0e-5f);
        acc_conic += float4(-0.5f * gdx * qdelta.x * dG,
                            -0.5f * gdx * qdelta.y * dG,
                            -0.5f * gdy * qdelta.y * dG, qG * d_opacity);
        }
    }

    if (entry_valid)
        commit_gradient(gaussian, count, acc_mean, acc_conic, acc_color,
                        acc_refine);
}
